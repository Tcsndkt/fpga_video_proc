# -*- coding: utf-8 -*-
"""移动摄像头消抖 + 运动物体检测（video_proc3）

功能
----
1) 消抖：Shi-Tomasi 角点 + 金字塔 LK 光流 + RANSAC 相似变换，
   轨迹滑动窗口平滑后反向补偿（残差 R = S ∘ C⁻¹，余弦软阈值区分手抖与运镜）；
2) 检测：DIS 稠密光流残差减去"分块中值背景运动场"（消掉视差），
   再用透视加权阈值判定运动：
       thr(x,y) = max(min_mag, rel_sens × 该处背景速度, noise_k × 噪声底)
   背景速度正比于当地的"像素/米"，所以判据对画面高度 y 不变：
   近处门槛自动抬高、远处自动降低，正好抵消近大远小；
3) 去重（宁缺毋滥）：合并同一物体的碎块（框间空隙 + IoMin 包含度）、
   显示前去掉嵌套在大框里的小框、只显示"真的移动过"的轨迹；
4) 分辨率自适应（输出 = 输入分辨率）、显示按屏幕缩放、帧率上限 30fps。

运行
----
    py video_proc3.py                       # 摄像头 0
    py video_proc3.py --video in.mp4        # 视频文件（播完自动循环；--no-loop 关闭）
    py video_proc3.py -o out.mp4            # 保存对比视频（尺寸随输入分辨率）
    py video_proc3.py --out-panel right -o out.mp4   # 只存"消抖+检测"那一半
    py video_proc3.py --max-width 1280      # 可选：给处理与输出封顶（默认 0 = 不限制）
    py video_proc3.py --selftest            # 自检（无需摄像头）

Controls 面板
------------
    min_area_1e4 / max_area_pct  物体面积占画面比例的下限 / 上限（0 = 上限不限）
    sens_x100    绝对下限：相对背景多走多少像素才算动 ×100（60 = 0.6 像素）
    rel_x100     透视加权：要跑出当地背景速度的百分之多少 ×100（60 = 60%）
    jitter_pct   消抖的抖动/运镜分界，占画面宽度 %（10 = 10%）
    roi_top_pct / roi_bot_pct    只在画面高度的这条横带里检测
    channel      视频输入通道

按键：q/Q/ESC 退出（或关闭窗口）、d 掩码视图、b 开关框选、s 存图
"""

import argparse
import os
import tempfile
import time
from collections import deque

import cv2
import numpy as np


FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=7)


LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

GAP = 4
FPS_CAP = 30.0

TB_MIN_AREA = "min_area_1e4"
TB_MAX_AREA = "max_area_pct"
TB_SENS = "sens_x100"
TB_REL = "rel_x100"
TB_JITTER = "jitter_pct"
TB_ROI_TOP = "roi_top_pct"
TB_ROI_BOT = "roi_bot_pct"
TB_CHANNEL = "channel"

TB_MIN_AREA_MAX = 300
TB_MAX_AREA_MAX = 100
TB_SENS_MAX = 300
TB_REL_MAX = 300
TB_JITTER_MAX = 50
TB_ROI_MAX = 100

MAIN_WIN = "Stabilize + Motion (video_proc3)"
MASK_WIN = "foreground mask"


def clamp(v, lo, hi):
    """把 v 夹在 [lo, hi]（滑条值、驱动报的帧率都要靠它兜住）"""
    return lo if v < lo else (hi if v > hi else v)


def sim(dx, dy, da):
    """构造"内容位移"相似变换矩阵，与 estimateAffinePartial2D 的输出同参数化"""
    c, s = np.cos(da), np.sin(da)
    return np.array([[c, -s, dx],
                     [s, c, dy]], np.float32)


def compose(A, B):
    """两个 2x3 仿射矩阵复合 A ∘ B（先作用 B 再作用 A），用于残差 S ∘ C⁻¹"""
    A3 = np.vstack([np.asarray(A, np.float64), [0.0, 0.0, 1.0]])
    B3 = np.vstack([np.asarray(B, np.float64), [0.0, 0.0, 1.0]])
    return (A3 @ B3)[:2]


def soft_gate(dist, low, high):
    """余弦软阈值：dist<=low 全补偿、dist>=high 不补偿、中间平滑过渡"""
    if dist <= low:
        return 1.0
    if dist >= high:
        return 0.0
    return 0.5 * (1.0 + np.cos(np.pi * (dist - low) / (high - low)))


def box_iou(a, b):
    """两个 [x,y,w,h] 的 IoU，跟踪时判断"前后帧是不是同一个目标" """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def box_overlap_min(a, b):
    """交集面积 ÷ 较小框面积（IoMin，即"包含度"）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    small = min(aw * ah, bw * bh)
    return (iw * ih) / small if small > 0 else 0.0


def _gap(a, b):
    """两个框之间的空隙（欧氏距离，重叠为 0）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0.0, max(ax, bx) - min(ax + aw, bx + bw))
    dy = max(0.0, max(ay, by) - min(ay + ah, by + bh))
    return float(np.hypot(dx, dy))


def box_overlap_min(a, b):
    """交集面积 ÷ 较小框面积（IoMin，即"包含度"）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    small = min(aw * ah, bw * bh)
    return (iw * ih) / small if small > 0 else 0.0


def merge_rects(rects, gap_ratio=0.5, contain_ratio=0.5, max_iter=6):
    """把"属于同一个物体"的框并成一个（本版主要的去重手段）。"""
    rects = [[float(v) for v in r] for r in rects]
    if len(rects) < 2 or (gap_ratio <= 0.0 and contain_ratio <= 0.0):
        return [[int(round(v)) for v in r] for r in rects]
    for _ in range(max_iter):
        n = len(rects)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        merged = False
        for i in range(n):
            for j in range(i + 1, n):
                hit = (contain_ratio > 0.0
                       and box_overlap_min(rects[i], rects[j]) >= contain_ratio)
                if not hit and gap_ratio > 0.0:
                    ref = gap_ratio * min(min(rects[i][2], rects[i][3]),
                                          min(rects[j][2], rects[j][3]))
                    hit = _gap(rects[i], rects[j]) <= ref
                if hit:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
                        merged = True
        if not merged:
            break
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(rects[i])
        rects = []
        for g in groups.values():
            x0 = min(r[0] for r in g)
            y0 = min(r[1] for r in g)
            x1 = max(r[0] + r[2] for r in g)
            y1 = max(r[1] + r[3] for r in g)
            rects.append([x0, y0, x1 - x0, y1 - y0])
    return [[int(round(v)) for v in r] for r in rects]


def suppress_nested(tracks, contain_ratio=0.5):
    """显示前的去重：把"套在大框里的小框"丢掉。"""
    if contain_ratio <= 0.0 or len(tracks) < 2:
        return list(tracks)
    kept = []
    for tr in sorted(tracks, key=lambda t: -float(t["box"][2] * t["box"][3])):
        if not any(box_overlap_min(tr["box"], k["box"]) >= contain_ratio for k in kept):
            kept.append(tr)
    return kept


def plan_work_width(W, H, ratio=0.5, wmin=160, wmax=480, area_max=150000):
    """按画面尺寸算出处理宽度：宽度比例与面积预算取小者。"""
    w = clamp(float(ratio) * W, float(wmin), float(wmax))
    w = min(w, W * (float(area_max) / float(max(1, W * H))) ** 0.5)
    return int(clamp(round(w), 64, W))


def plan_source_fps(fps_src, fps_cap=FPS_CAP, max_skip=8):
    """把源帧率压到上限内，返回 (每帧先丢几帧, 有效帧率)。丢帧不改播放速度。"""
    f = float(fps_src) if fps_src else 0.0
    if not (1e-3 < f < 240.0):
        f = float(fps_cap)
    if f <= float(fps_cap) * 1.05:
        return 0, f
    skip = int(clamp(int(round(f / float(fps_cap))) - 1, 0, int(max_skip)))
    return skip, f / (skip + 1)


def limit_frame(frame, max_width, buf=None):
    """可选地把画面限制到 max_width 宽（max_width=0 表示不限制，零拷贝返回）"""
    if not max_width or frame.shape[1] <= max_width:
        return frame, buf
    tw = int(max_width)
    th = max(2, int(round(frame.shape[0] * tw / float(frame.shape[1]))))
    if buf is None or buf.shape[:2] != (th, tw):
        buf = np.empty((th, tw, 3), frame.dtype)
    cv2.resize(frame, (tw, th), dst=buf, interpolation=cv2.INTER_AREA)
    return buf, buf


def screen_size():
    """取屏幕分辨率（把窗口缩到屏幕内用）；取不到返回 None，程序照常跑"""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
        root.destroy()
        return size
    except Exception:
        return None


def plan_view_scale(canvas_w, canvas_h, screen, view_width=0):
    """显示缩放比例（只影响屏幕上的那一份，不影响保存的视频）"""
    if view_width and int(view_width) > 0:
        return min(1.0, float(view_width) / max(1, canvas_w))
    if not screen:
        return min(1.0, 1600.0 / max(1, canvas_w))
    return min(1.0, screen[0] * 0.95 / max(1, canvas_w),
               screen[1] * 0.90 / max(1, canvas_h))


def fit_window(win, w, h, screen):
    """把窗口缩到屏幕内（只缩小、不放大）；拿不到屏幕信息就什么都不做"""
    if not screen:
        return
    k = min(1.0, screen[0] * 0.95 / max(1, w), screen[1] * 0.90 / max(1, h))
    cv2.resizeWindow(win, max(160, int(w * k)), max(120, int(h * k)))


def put_text(img, text, org, scale, color, tscale=1.0):
    """带黑色描边的文字（亮背景/暗背景都读得清），返回文字宽度"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), max(2, int(round(3 * tscale))), cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                max(1, int(round(tscale))), cv2.LINE_AA)
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]


def build_sources(args):
    """命令行参数 -> 视频通道列表（每个通道 = 一路摄像头 或 一个视频文件）"""
    if args.video:
        return [("file", v) for v in args.video]
    return [("camera", i) for i in range(max(1, int(args.cameras)))]


class FrameSource:
    """视频输入通道：打开 / 读帧(含丢帧限速) / 回到开头 / 释放"""

    def __init__(self, sources, width=640, height=480, fps_cap=FPS_CAP):
        self.sources = list(sources)
        self.width, self.height = int(width), int(height)
        self.fps_cap = float(fps_cap)
        self.index = -1
        self.cap = None
        self.kind = "camera"
        self.skip = 0
        self.fps_src = float(fps_cap)
        self.eff_fps = float(fps_cap)
        self.dt = 1.0 / float(fps_cap)

    def count(self):
        return max(1, len(self.sources))

    def label(self, index=None):
        i = self.index if index is None else int(index)
        if 0 <= i < len(self.sources):
            kind, val = self.sources[i]
            return ("camera %d" % val) if kind == "camera" else os.path.basename(str(val))
        return "?"

    @property
    def is_camera(self):
        return self.kind == "camera"

    def open(self, index):
        """打开第 index 个通道；失败返回 False 且保持原通道不变"""
        index = int(clamp(int(index), 0, len(self.sources) - 1))
        kind, val = self.sources[index]
        if kind == "camera":
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(val), backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps_cap)
        else:
            cap = cv2.VideoCapture(str(val))
        if not cap.isOpened():
            cap.release()
            print("无法打开视频通道 %d: %s" % (index, self.label(index)))
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.release()
        self.cap = cap
        self.index = index
        self.kind = kind
        f = cap.get(cv2.CAP_PROP_FPS)
        self.fps_src = float(f) if f else 0.0
        self.skip, self.eff_fps = plan_source_fps(self.fps_src, self.fps_cap)
        self.dt = 1.0 / self.eff_fps
        return True

    def read(self):
        """读一帧；源帧率超上限时先 grab() 丢帧（grab 只挪指针不解码，很便宜）"""
        if self.cap is None:
            return False, None
        for _ in range(self.skip):
            if not self.cap.grab():
                return False, None
        return self.cap.read()

    def rewind(self):
        """回到文件开头（循环播放用）"""
        if self.cap is None or self.kind != "file":
            return False
        if self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if pos != pos or pos < 1.0:
                return True
        return self.open(self.index)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.index = -1


class VideoStabilizer:
    """把"手持抖动"从画面运动里剥离出去。"""

    def __init__(self, width, height, proc_width=320, window=30,
                 jitter_ratio=0.10, band=0.03, crop_ratio=0.10):
        self.W, self.H = int(width), int(height)
        self.scale = min(1.0, float(proc_width) / self.W)
        self.inv_scale = 1.0 / self.scale
        self.pw = max(64, int(round(self.W * self.scale)))
        self.ph = max(48, int(round(self.H * self.scale)))

        self.min_pts = 30
        self.redetect_every = 15
        self.max_err = 8.0
        self.ransac_thresh = 2.0

        self.prev_gray = None
        self.prev_pts = None
        self.n_frame = 0

        self.window = max(2, int(window))
        self.traj = np.zeros((self.window, 3), np.float32)
        self.traj_i = 0
        self.traj_n = 0
        self.curr = np.zeros(3, np.float32)
        self.smooth = np.zeros(3, np.float32)

        self.jitter_ratio = float(jitter_ratio)
        self.band = float(band)
        self.gate_low, self.gate_high = 0.0, 1.0
        self.set_jitter_ratio(jitter_ratio, band)

        self.crop = int(crop_ratio * min(self.W, self.H))
        self.warp_buf = np.empty((self.H, self.W, 3), np.uint8)

    def set_jitter_ratio(self, jitter_ratio, band=None):
        """运行时改"抖动/运镜"分界（占画面宽度的比例，O(1)，可每帧调用）"""
        self.jitter_ratio = float(jitter_ratio)
        if band is not None:
            self.band = float(band)
        low = (self.jitter_ratio - self.band) * self.W
        high = (self.jitter_ratio + self.band) * self.W
        self.gate_low = max(1.0, low)
        self.gate_high = max(self.gate_low + 1.0, high)

    def _crop(self, frame):
        """裁掉四周 crop 像素（切片是视图，零拷贝）"""
        m = self.crop
        return frame[m:self.H - m, m:self.W - m]

    def process(self, frame):
        """处理一帧，返回 (消抖并裁剪后的图, info)。"""
        self.n_frame += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.scale < 1.0:
            small = cv2.resize(gray, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        else:
            small = gray
        cv2.GaussianBlur(small, (5, 5), 0, dst=small)

        info = {"ok": False, "dx": 0.0, "dy": 0.0, "da": 0.0, "inliers": 0}

        if self.prev_gray is None:
            self.prev_gray = small
            self.prev_pts = cv2.goodFeaturesToTrack(small, mask=None, **FEATURE_PARAMS)
            return self._crop(frame), info


        if self.prev_pts is None or len(self.prev_pts) < 4:
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None,
                                                    **FEATURE_PARAMS)

        M = None
        n_in = 0
        if self.prev_pts is not None and len(self.prev_pts) >= 4:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, small, self.prev_pts, None, **LK_PARAMS)
            if next_pts is not None:
                ok = status.ravel() == 1
                new_pts, old_pts = next_pts[ok], self.prev_pts[ok]
                keep = err[ok] < self.max_err
                new_pts, old_pts = new_pts[keep], old_pts[keep]
                if len(new_pts) >= 4:
                    M, inl = cv2.estimateAffinePartial2D(
                        old_pts, new_pts, method=cv2.RANSAC,
                        ransacReprojThreshold=self.ransac_thresh,
                        maxIters=2000, confidence=0.995)
                    if inl is not None:
                        n_in = int(inl.ravel().astype(bool).sum())
                if len(new_pts) >= self.min_pts and self.n_frame % self.redetect_every != 0:
                    self.prev_pts = new_pts.reshape(-1, 1, 2).astype(np.float32)
                else:
                    self.prev_pts = cv2.goodFeaturesToTrack(small, mask=None,
                                                            **FEATURE_PARAMS)
        if self.prev_pts is None or len(self.prev_pts) < 4:
            self.prev_pts = cv2.goodFeaturesToTrack(small, mask=None, **FEATURE_PARAMS)
        self.prev_gray = small

        out = frame
        if M is not None:
            dx, dy = float(M[0, 2]) * self.inv_scale, float(M[1, 2]) * self.inv_scale
            da = float(np.arctan2(M[1, 0], M[0, 0]))


            self.curr += (dx, dy, da)
            self.traj[self.traj_i] = self.curr
            self.traj_i = (self.traj_i + 1) % self.window
            self.traj_n = min(self.window, self.traj_n + 1)
            self.smooth = self.traj[:self.traj_n].mean(axis=0)


            R = compose(sim(self.smooth[0], self.smooth[1], self.smooth[2]),
                        cv2.invertAffineTransform(sim(*self.curr)))
            rdx, rdy = float(R[0, 2]), float(R[1, 2])
            rda = float(np.arctan2(R[1, 0], R[0, 0]))

            g = soft_gate(float(np.hypot(rdx, rdy)), self.gate_low, self.gate_high)
            if g > 0.0:
                out = cv2.warpAffine(frame, sim(g * rdx, g * rdy, g * rda), (self.W, self.H),
                                     dst=self.warp_buf, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            info.update(ok=True, dx=dx, dy=dy, da=da)
        info["inliers"] = n_in
        return self._crop(out), info


class MotionDetector:
    """在消抖后的画面上找"相对背景自己会动"的东西。"""

    def __init__(self, width, height, detect_width=480, min_area_ratio=0.0005,
                 max_area_ratio=0.0, warmup=2, min_mag=0.6, rel_sens=0.6, noise_k=3.0,
                 box_fill_min=0.12, bg_block_ratio=0.06, bg_stride=4,
                 bg_smooth_ratio=0.04, merge_gap=0.5, contain_ratio=0.5,
                 roi_top=0.0, roi_bottom=1.0):
        self.W, self.H = int(width), int(height)
        self.scale = min(1.0, float(detect_width) / self.W)
        self.inv_scale = 1.0 / self.scale
        self.dw = max(64, int(round(self.W * self.scale)))
        self.dh = max(48, int(round(self.H * self.scale)))


        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)


        self.min_area = 40
        self.min_box = 0.0
        self.max_box = float("inf")
        self.set_area_limits(min_area_ratio, max_area_ratio)


        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        self.warmup = int(warmup)
        self.min_mag = float(min_mag)
        self.rel_sens = float(rel_sens)
        self.noise_k = float(noise_k)
        self.box_fill_min = float(box_fill_min)
        self.merge_gap = float(clamp(merge_gap, 0.0, 3.0))
        self.contain_ratio = float(clamp(contain_ratio, 0.0, 1.0))


        self.bg_block_ratio = float(bg_block_ratio)
        self.bg_stride = max(1, int(bg_stride))
        self.bg_block = int(round(self.bg_block_ratio * self.dw)) if self.bg_block_ratio > 0 else 0
        self.bg_smooth = max(1.0, float(bg_smooth_ratio) * self.dw)

        self.roi_top, self.roi_bottom = 0.0, 1.0
        self.set_roi(roi_top, roi_bottom)

        self.n_frame = 0
        self.prev_gray = None
        self.mask = np.zeros((self.dh, self.dw), np.uint8)
        self.thresh = 0.0


        self._dx = np.empty((self.dh, self.dw), np.float32)
        self._dy = np.empty((self.dh, self.dw), np.float32)
        self._bgx = np.empty((self.dh, self.dw), np.float32)
        self._bgy = np.empty((self.dh, self.dw), np.float32)
        self._bgmag = np.empty((self.dh, self.dw), np.float32)
        self._thr_map = np.empty((self.dh, self.dw), np.float32)
        self._mag = np.empty((self.dh, self.dw), np.float32)
        kk = max(2, self.bg_block // self.bg_stride) if self.bg_block > 0 else 0
        sh = ((self.dh // self.bg_stride) // kk) * kk * self.bg_stride if kk else 1
        sw = ((self.dw // self.bg_stride) // kk) * kk * self.bg_stride if kk else 1
        self._bgsx = np.empty((max(1, sh), max(1, sw)), np.float32)
        self._bgsy = np.empty((max(1, sh), max(1, sw)), np.float32)


    def set_area_limits(self, min_ratio, max_ratio):
        """面积占画面比例的上下限（上限 < 下限时自动交换，避免区间为空）"""
        area = float(self.dw * self.dh)
        lo, hi = float(min_ratio), float(max_ratio)
        if hi > 0.0 and hi < lo:
            lo, hi = hi, lo
        self.min_area_ratio, self.max_area_ratio = lo, hi
        self.min_area = max(20, int(lo * area))
        self.min_box = lo * area
        self.max_box = (hi * area) if hi > 0.0 else float("inf")

    def set_min_mag(self, v):
        """设置绝对下限（像素）：比这更小的残差一律不算运动"""
        self.min_mag = float(clamp(v, 0.05, 20.0))

    def set_rel_sens(self, v):
        """设置相对灵敏度：物体至少要跑出当地背景速度的这个比例才算运动。"""
        self.rel_sens = float(clamp(v, 0.0, 5.0))

    def set_roi(self, top_ratio, bottom_ratio):
        """只在这条水平带里检测（走路拍摄时上下两条带子是假框主产区）"""
        top = float(clamp(top_ratio, 0.0, 0.9))
        bot = float(clamp(bottom_ratio, 0.1, 1.0))
        if bot - top < 0.05:
            bot = min(1.0, top + 0.05)
        self.roi_top, self.roi_bottom = top, bot


    def background_flow(self, fx, fy):
        """估计"背景运动场" (bx, by)：分块中值 + 按覆盖区域放大。"""
        h, w = fx.shape
        k = max(2, self.bg_block // self.bg_stride) if self.bg_block > 0 else 0
        nby = (h // self.bg_stride) // k if k else 0
        nbx = (w // self.bg_stride) // k if k else 0

        if nby < 2 or nbx < 2:
            self._bgx[:] = float(np.median(fx[::2, ::2]))
            self._bgy[:] = float(np.median(fy[::2, ::2]))
            return self._bgx, self._bgy

        hs = nby * k * self.bg_stride
        ws = nbx * k * self.bg_stride

        def expand(f, small, big):
            a = f[:hs:self.bg_stride, :ws:self.bg_stride]
            a = a.reshape(nby, k, nbx, k).transpose(0, 2, 1, 3).reshape(nby, nbx, k * k)
            coarse = np.median(a, axis=2).astype(np.float32)
            cv2.resize(coarse, (ws, hs), dst=small, interpolation=cv2.INTER_LINEAR)
            big[:hs, :ws] = small
            if hs < h:
                big[hs:, :] = big[hs - 1:hs, :]
            if ws < w:
                big[:, ws:] = big[:, ws - 1:ws]
            return big

        expand(fx, self._bgsx, self._bgx)
        expand(fy, self._bgsy, self._bgy)
        return self._bgx, self._bgy


    def perspective_threshold(self, bgx, bgy, noise):
        """【本版核心】按位置给运动定门槛：thr = max(min_mag, rel_sens·S, noise_k·噪声底)。"""
        cv2.magnitude(bgx, bgy, self._bgmag)

        cv2.GaussianBlur(self._bgmag, (0, 0), self.bg_smooth, dst=self._bgmag)
        np.multiply(self._bgmag, self.rel_sens, out=self._thr_map)
        floor = max(self.min_mag, self.noise_k * noise)
        np.maximum(self._thr_map, floor, out=self._thr_map)
        return self._thr_map


    def apply(self, bgr):
        """输入消抖后的画面，返回前景掩码（0/255，检测尺度）"""
        self.n_frame += 1
        if self.scale < 1.0:
            small = cv2.resize(bgr, (self.dw, self.dh), interpolation=cv2.INTER_AREA)
        else:
            small = bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cv2.GaussianBlur(gray, (5, 5), 0, dst=gray)

        if self.prev_gray is None:
            self.prev_gray = gray
            return self.mask

        flow = self.dis.calc(self.prev_gray, gray, None)
        self.prev_gray = gray
        fx, fy = flow[..., 0], flow[..., 1]

        bx, by = self.background_flow(fx, fy)
        cv2.subtract(fx, bx, dst=self._dx)
        cv2.subtract(fy, by, dst=self._dy)
        res = cv2.magnitude(self._dx, self._dy, self._mag)

        noise = float(np.median(np.abs(res[::2, ::2])))
        thr_map = self.perspective_threshold(bx, by, noise)
        self.thresh = float(np.median(thr_map))

        mask = cv2.compare(res, thr_map, cv2.CMP_GT)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        if self.roi_top > 0.0 or self.roi_bottom < 1.0:
            mask[:int(self.roi_top * self.dh), :] = 0
            mask[int(self.roi_bottom * self.dh):, :] = 0
        self.mask = mask
        return mask


    def boxes(self):
        """返回工作分辨率下的候选框 [[x,y,w,h], ...]"""
        if self.n_frame <= self.warmup or self.mask is None:
            return []
        cnts, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for c in cnts:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue
            a = float(w * h)
            if a < self.min_box or a > self.max_box:
                continue
            if not (0.15 <= w / float(h) <= 8.0):
                continue
            rects.append([int(x), int(y), int(w), int(h)])
        if len(rects) > 1 and (self.merge_gap > 0.0 or self.contain_ratio > 0.0):
            rects = merge_rects(rects, self.merge_gap, self.contain_ratio)
            rects = [r for r in rects
                     if self.min_box <= float(max(1, r[2]) * max(1, r[3])) <= self.max_box]
        inv = self.inv_scale
        return [[int(r[0] * inv), int(r[1] * inv), int(r[2] * inv), int(r[3] * inv)]
                for r in rects]

    def judge(self, boxes):
        """判定候选框是否真是目标：框内运动像素的填充率（真目标是一整块，噪声是零散点）。"""
        flags = []
        for (x, y, w, h) in boxes:
            x1, y1 = max(0, int(x * self.scale)), max(0, int(y * self.scale))
            x2, y2 = min(self.dw, int((x + w) * self.scale)), min(self.dh, int((y + h) * self.scale))
            fill = 0.0
            if x2 > x1 and y2 > y1:
                sub = self.mask[y1:y2, x1:x2]
                fill = float(np.count_nonzero(sub)) / float(sub.size)
            flags.append(fill >= self.box_fill_min)
        return flags


class BoxTracker:
    """IoU + 速度外推关联的跟踪器：稳定 ID、EMA 平滑框、速度方向、尾迹。"""

    def __init__(self, iou_thresh=0.2, min_hits=3, max_missing=15,
                 smooth=0.5, trail_len=40):
        self.iou_thresh = iou_thresh
        self.min_hits = min_hits
        self.max_missing = max_missing
        self.smooth = smooth
        self.trail_len = trail_len
        self.tracks = []
        self.next_id = 1

    def _new_track(self, box, moving):
        return {"id": self.next_id, "box": np.array(box, np.float32),
                "hits": 1, "missed": 0, "vel": np.zeros(2, np.float32),
                "trail": deque([(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)],
                               maxlen=self.trail_len),
                "flags": deque([bool(moving)], maxlen=10)}

    def _predict(self, tr, dt):
        """按速度外推 dt 秒后的框（常速模型），用于关联"""
        if dt <= 1e-4:
            return tr["box"]
        v = tr["vel"]
        return np.array([tr["box"][0] + v[0] * dt, tr["box"][1] + v[1] * dt,
                         tr["box"][2], tr["box"][3]], np.float32)

    def _update(self, tr, box, moving, dt):
        old = np.array([tr["box"][0] + tr["box"][2] / 2.0,
                        tr["box"][1] + tr["box"][3] / 2.0], np.float32)
        tr["box"] = tr["box"] + self.smooth * (np.array(box, np.float32) - tr["box"])
        new = np.array([tr["box"][0] + tr["box"][2] / 2.0,
                        tr["box"][1] + tr["box"][3] / 2.0], np.float32)
        if dt > 1e-4:
            v = (new - old) / dt
            tr["vel"] = tr["vel"] + 0.4 * (v - tr["vel"])
        tr["trail"].append((float(new[0]), float(new[1])))
        tr["flags"].append(bool(moving))
        tr["hits"] += 1
        tr["missed"] = 0

    def update(self, boxes, moving, dt):
        """每帧调用：记丢失 -> 速度外推+IoU 贪心匹配 -> 新建轨迹 -> 清理 -> 输出确认过的"""
        for tr in self.tracks:
            tr["missed"] += 1
        used_d = set()
        if len(boxes):
            pairs = []
            for ti, tr in enumerate(self.tracks):
                pred = self._predict(tr, dt)
                for di, d in enumerate(boxes):
                    v = box_iou(pred, d)
                    if v >= self.iou_thresh:
                        pairs.append((v, ti, di))
            pairs.sort(reverse=True)
            used_t = set()
            for _, ti, di in pairs:
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                self._update(self.tracks[ti], boxes[di], moving[di], dt)
        for di, d in enumerate(boxes):
            if di not in used_d:
                self.tracks.append(self._new_track(d, moving[di]))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missing]
        return [t for t in self.tracks if t["hits"] >= self.min_hits]

    @staticmethod
    def is_moving(tr, min_votes=2, min_speed=25.0):
        """两种判据任一成立就算"运动目标"：多帧投票 或 消抖后仍在持续移动"""
        if sum(tr["flags"]) >= min_votes:
            return True
        return float(np.hypot(tr["vel"][0], tr["vel"][1])) > min_speed and tr["hits"] >= 5

    @staticmethod
    def is_really_moving(tr, min_disp_px=6.0, disp_ratio=0.35):
        """显示前的质量门限：这条轨迹**真的动过**吗？"""
        trail = list(tr["trail"])
        if len(trail) < 2:
            return False
        k = min(20, len(trail))
        net = float(np.hypot(trail[-1][0] - trail[-k][0], trail[-1][1] - trail[-k][1]))
        size = float(np.sqrt(max(1.0, tr["box"][2] * tr["box"][3])))
        return net >= max(float(min_disp_px), float(disp_ratio) * size)


def draw_tracks(img, tracks, x_off=0, y_off=0, arrow_min_speed=12.0, scale=1.0):
    """把跟踪结果画到画布上（scale 让线宽/字号跟着分辨率走）"""
    th = max(1, int(round(2 * scale)))
    th_txt = max(1, int(round(1 * scale)))
    for tr in tracks:
        x, y, w, h = [int(v) for v in tr["box"]]
        x += x_off
        y += y_off
        moving = BoxTracker.is_moving(tr)
        color = (0, 0, 255) if moving else (0, 215, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, th)
        if len(tr["trail"]) > 1:
            pts = np.array(tr["trail"], np.int32).reshape(-1, 1, 2)
            pts[:, :, 0] += x_off
            pts[:, :, 1] += y_off
            cv2.polylines(img, [pts], False, color, th_txt, cv2.LINE_AA)
        spd = float(np.hypot(tr["vel"][0], tr["vel"][1]))
        if spd > arrow_min_speed:
            ax, ay = int(x + w / 2), int(y + h / 2)
            cv2.arrowedLine(img, (ax, ay),
                            (int(ax + tr["vel"][0] / spd * 30 * scale),
                             int(ay + tr["vel"][1] / spd * 30 * scale)),
                            color, th, tipLength=0.3)
        label = "ID%d %s %.0f" % (tr["id"], "MOVING" if moving else "check", spd)
        ty = y - int(9 * scale) if (y - y_off) > 20 * scale else y + h + int(19 * scale)
        put_text(img, label, (x, ty), 0.5 * scale, color, scale)


class TrackbarPanel:
    """运行时调参面板（滑条）。主循环每帧主动读一次，回调留空避免 GUI 线程竞态。"""

    WIN = "Controls"

    def __init__(self, n_channels, min_area_ratio=0.0005, max_area_ratio=0.0,
                 min_mag=0.6, rel_sens=0.6, jitter_ratio=0.10,
                 roi_top=0.0, roi_bottom=1.0, channel=0):
        self.ch_max = max(1, max(1, int(n_channels)) - 1)
        self._defaults = {
            TB_MIN_AREA: int(round(min_area_ratio * 10000)),
            TB_MAX_AREA: int(round(max_area_ratio * 100)),
            TB_SENS: int(round(clamp(min_mag, 0.05, TB_SENS_MAX / 100.0) * 100)),
            TB_REL: int(round(clamp(rel_sens, 0.0, TB_REL_MAX / 100.0) * 100)),
            TB_JITTER: int(round(clamp(jitter_ratio, 0.01, 0.5) * 100)),
            TB_ROI_TOP: int(round(clamp(roi_top, 0.0, 0.9) * 100)),
            TB_ROI_BOT: int(round(clamp(roi_bottom, 0.1, 1.0) * 100)),
            TB_CHANNEL: int(clamp(channel, 0, self.ch_max)),
        }
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, 470, 300)
        for name, mx in ((TB_MIN_AREA, TB_MIN_AREA_MAX), (TB_MAX_AREA, TB_MAX_AREA_MAX),
                         (TB_SENS, TB_SENS_MAX), (TB_REL, TB_REL_MAX),
                         (TB_JITTER, TB_JITTER_MAX), (TB_ROI_TOP, TB_ROI_MAX),
                         (TB_ROI_BOT, TB_ROI_MAX), (TB_CHANNEL, self.ch_max)):
            cv2.createTrackbar(name, self.WIN, self._defaults[name], mx, lambda _v: None)

    def get(self, name):
        """读滑条值；窗口被关掉时 OpenCV 返回 -1，这时退回默认值"""
        try:
            v = int(cv2.getTrackbarPos(name, self.WIN))
        except cv2.error:
            v = -1
        return self._defaults.get(name, 0) if v < 0 else v

    def set(self, name, value):
        try:
            cv2.setTrackbarPos(name, self.WIN, int(value))
        except cv2.error:
            pass

    def apply(self, det, stab=None):
        """把面板值下发到检测器/消抖器（都是 O(1) 赋值，每帧调用无开销）"""
        det.set_area_limits(self.get(TB_MIN_AREA) / 10000.0, self.get(TB_MAX_AREA) / 100.0)
        det.set_min_mag(self.get(TB_SENS) / 100.0)
        det.set_rel_sens(self.get(TB_REL) / 100.0)
        det.set_roi(self.get(TB_ROI_TOP) / 100.0, self.get(TB_ROI_BOT) / 100.0)
        if stab is not None:
            stab.set_jitter_ratio(self.get(TB_JITTER) / 100.0)

    def status(self, src_label):
        """一行状态文字（英文：putText 画不了中文）"""
        mx = self.get(TB_MAX_AREA)
        return "ch%d %s | minA %.2f%% maxA %s | abs %.2fpx rel %.2f | jit %d%% roi %d-%d%%" % (
            self.get(TB_CHANNEL), src_label, self.get(TB_MIN_AREA) / 100.0,
            ("%d%%" % mx) if mx > 0 else "off", self.get(TB_SENS) / 100.0,
            self.get(TB_REL) / 100.0, self.get(TB_JITTER),
            self.get(TB_ROI_TOP), self.get(TB_ROI_BOT))


class Pipeline:
    """消抖 -> 检测 -> 跟踪 -> 画布。"""

    def __init__(self, W, H, args):
        self.W, self.H = int(W), int(H)
        stab_w = args.stab_width or plan_work_width(self.W, self.H, ratio=0.5,
                                                    wmin=160, wmax=480, area_max=150000)
        self.stab = VideoStabilizer(self.W, self.H, proc_width=stab_w, window=args.window,
                                    jitter_ratio=args.jitter_ratio, band=args.band,
                                    crop_ratio=args.crop_ratio)
        self.m = self.stab.crop
        self.cw, self.ch = self.W - 2 * self.m, self.H - 2 * self.m

        det_w = args.det_width or plan_work_width(self.cw, self.ch, ratio=0.75,
                                                  wmin=240, wmax=640, area_max=350000)
        self.det_w = det_w
        self.det = MotionDetector(self.cw, self.ch, detect_width=det_w,
                                  min_area_ratio=args.min_area_ratio,
                                  max_area_ratio=args.max_area_ratio,
                                  min_mag=args.min_mag, rel_sens=args.rel_sens,
                                  noise_k=args.noise_k, warmup=args.warmup,
                                  bg_block_ratio=args.bg_block_ratio,
                                  merge_gap=args.merge_gap,
                                  contain_ratio=args.contain_ratio,
                                  roi_top=args.roi_top / 100.0,
                                  roi_bottom=args.roi_bottom / 100.0)
        self.tracker = BoxTracker()
        self._saved = None

        span = self.cw * 2 + GAP
        if args.layout == "v" or (args.layout == "auto" and span > 5.5 * self.ch):
            self.layout = "v"
            self.canvas = np.zeros((self.ch * 2 + GAP, self.cw, 3), np.uint8)
            self.x_off, self.y_off = 0, self.ch + GAP
        else:
            self.layout = "h"
            self.canvas = np.zeros((self.ch, span, 3), np.uint8)
            self.x_off, self.y_off = self.cw + GAP, 0

    def save_view(self, canvas, mode="both"):
        """返回要写进视频的那一份：both = 整张画布；right = 只取消抖/检测那一侧。"""
        if mode != "right":
            return canvas
        if self._saved is None:
            self._saved = np.empty((self.ch, self.cw, 3), np.uint8)
        np.copyto(self._saved,
                  canvas[:, self.x_off:] if self.layout == "h" else canvas[self.y_off:, :])
        return self._saved

    def step(self, frame, dt, show_obj=True, mask_overlay=False):
        """跑一帧，返回 (canvas, tracks, mask, detector)。"""
        stab_crop, _info = self.stab.process(frame)
        orig_crop = frame[self.m:self.H - self.m, self.m:self.W - self.m]
        mask = None
        tracks = []
        if show_obj:
            mask = self.det.apply(stab_crop)
            boxes = self.det.boxes()
            tracks = self.tracker.update(boxes, self.det.judge(boxes) if boxes else [], dt)

        self.canvas[:, :self.cw] = orig_crop
        right = self.canvas[:, self.x_off:] if self.layout == "h" else self.canvas[self.y_off:, :]
        right[:] = stab_crop
        if mask_overlay and mask is not None:
            big = cv2.resize(mask, (self.cw, self.ch), interpolation=cv2.INTER_NEAREST)
            sel = big > 0
            right[sel] = (0.55 * right[sel] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
        return self.canvas, tracks, mask, self.det


def run(args):
    cv2.setUseOptimized(True)

    sources = build_sources(args)
    src = FrameSource(sources, width=args.width, height=args.height, fps_cap=args.fps_cap)
    start_ch = args.channel if args.channel is not None else (0 if args.video else args.camera)
    if not src.open(start_ch):
        for i in range(src.count()):
            if src.open(i):
                print("已自动改用通道 %d: %s" % (i, src.label()))
                break
        else:
            print("没有可用的视频通道，退出。")
            return 1

    panel = TrackbarPanel(src.count(), min_area_ratio=args.min_area_ratio,
                          max_area_ratio=args.max_area_ratio, min_mag=args.min_mag,
                          rel_sens=args.rel_sens, jitter_ratio=args.jitter_ratio,
                          roi_top=args.roi_top, roi_bottom=args.roi_bottom,
                          channel=src.index)
    cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    screen = screen_size()
    mask_win_size = None
    print("""
============== Controls 面板（拖动即时生效）==============
  min_area_1e4/max_area_pct 面积占画面比例的下限/上限（0 = 上限不限）
  sens_x100  绝对下限：多走多少像素才算动 ×100（60 = 0.6 像素）
  rel_x100   透视加权：跑出当地背景速度的百分之多少 ×100（60 = 60%）
             近处门槛自动抬高、远处自动降低，抵消"近大远小"
  jitter_pct 消抖的抖动/运镜分界（占画面宽 %）   roi_*_pct 只看这条横带
  channel    视频输入通道
按键: q/Q/ESC 退出（或关闭窗口）| d 掩码视图 | b 开关检测 | s 存图
==========================================================
""")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    writer_seg = 0
    writer_shape = None
    pipe = None
    read_buf = None
    n_loop = 0
    show_obj = not args.no_object
    show_debug = args.debug
    t_prev = time.perf_counter()
    fps_hist = deque(maxlen=30)
    fps = 0.0
    frame = None
    t_frame = 0
    tscale = 1.0
    view_scale = 1.0

    while True:
        t_frame += 1
        t_now = time.perf_counter()
        wall_dt = max(1e-3, t_now - t_prev)
        t_prev = t_now
        fps_hist.append(1.0 / wall_dt)
        fps = sum(fps_hist) / len(fps_hist)


        ch_sel = panel.get(TB_CHANNEL)
        if ch_sel != src.index and 0 <= ch_sel < src.count():
            if not src.open(ch_sel):
                panel.set(TB_CHANNEL, src.index)
                continue
            ok, frame = src.read()
            if not ok:
                print("通道 %d 读帧失败，保持原画面" % ch_sel)
                continue
            pipe, read_buf = None, None
            print("已切换到通道 %d: %s  分辨率 %s  有效 %.1ffps"
                  % (ch_sel, src.label(), frame.shape[1::-1], src.eff_fps))
        else:
            ok, frame = src.read()
            if not ok and args.loop and not src.is_camera and src.rewind():
                ok, frame = src.read()
                if ok:
                    n_loop += 1
                    pipe, read_buf = None, None
                    print("视频循环第 %d 次：%s 从头重新播放" % (n_loop, src.label()))
            if not ok:
                print("视频结束 / 读帧失败")
                break

        frame, read_buf = limit_frame(frame, args.max_width, read_buf)
        H, W = frame.shape[:2]


        if pipe is None or pipe.W != W or pipe.H != H:
            pipe = Pipeline(W, H, args)
            tscale = float(clamp(pipe.cw / 640.0, 0.6, 3.0))
            view_scale = plan_view_scale(pipe.canvas.shape[1], pipe.canvas.shape[0],
                                         screen, args.view_width)
            vw = max(160, int(pipe.canvas.shape[1] * view_scale))
            vh = max(120, int(pipe.canvas.shape[0] * view_scale))
            cv2.resizeWindow(MAIN_WIN, vw, vh)
            print("流水线已按 %dx%d 重建：输出 %dx%d（与原片同分辨率，消抖裁边 %d px）"
                  "｜显示 %dx%d（%.0f%%）｜检测处理宽度 %d"
                  % (W, H, pipe.canvas.shape[1], pipe.canvas.shape[0], pipe.m,
                     vw, vh, 100.0 * view_scale, pipe.det.dw))

        panel.apply(pipe.det, pipe.stab)
        dt = wall_dt if src.is_camera else src.dt
        canvas, tracks, mask, det = pipe.step(frame, dt, show_obj, args.mask_overlay)


        shown = tracks
        if args.motion_gate:
            shown = [t for t in shown if BoxTracker.is_really_moving(t)]
        shown = suppress_nested(shown, args.contain_ratio)
        if args.max_show and len(shown) > args.max_show:
            shown = sorted(shown, key=lambda t: (float(np.hypot(t["vel"][0], t["vel"][1])),
                                                 float(t["box"][2] * t["box"][3]), t["hits"]),
                           reverse=True)[:args.max_show]


        saved = pipe.save_view(canvas, args.out_panel)
        if args.output and (writer is None or writer_shape != saved.shape):
            if writer is not None:
                writer.release()
                writer_seg += 1
                print("画面尺寸变化，输出改为新分段")
            root, ext = os.path.splitext(args.output)
            path = args.output if writer_seg == 0 else "%s_part%d%s" % (root, writer_seg + 1, ext)
            writer = cv2.VideoWriter(path, fourcc, clamp(src.eff_fps, 5.0, args.fps_cap),
                                     (saved.shape[1], saved.shape[0]))
            if writer.isOpened():
                writer_shape = saved.shape
                print("正在保存 %dx%d -> %s" % (saved.shape[1], saved.shape[0], path))
            else:
                print("警告：无法创建输出视频", path)
                writer, args.output = None, None


        draw_tracks(canvas, shown, x_off=pipe.x_off, y_off=pipe.y_off, scale=tscale)
        pad = int(10 * tscale)
        put_text(canvas, "Original", (pad, int(30 * tscale)), 0.9 * tscale, (0, 0, 255), tscale)
        put_text(canvas, "Stabilized", (pipe.x_off + pad, pipe.y_off + int(30 * tscale)),
                 0.9 * tscale, (0, 255, 0), tscale)
        hint = "q / ESC = quit   (or close this window)"
        hw = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * tscale, 1)[0][0]
        put_text(canvas, hint, (max(pad, canvas.shape[1] - hw - pad), int(30 * tscale)),
                 0.5 * tscale, (0, 255, 255), tscale)
        status = "FPS %.1f/%.0f  %dx%d  thr %.2fpx  tracks %s/%s  %s  %s" % (
            fps, args.fps_cap, pipe.cw, pipe.ch, det.thresh, len(shown), len(tracks),
            "OBJ-ON" if show_obj else "OBJ-OFF", panel.status(src.label()))
        put_text(canvas, status, (pad, int(62 * tscale)), 0.6 * tscale, (255, 255, 0), tscale)

        if writer is not None:
            writer.write(saved)
        view = canvas
        if view_scale < 0.995:
            view = cv2.resize(canvas, None, fx=view_scale, fy=view_scale,
                              interpolation=cv2.INTER_AREA)
        cv2.imshow(MAIN_WIN, view)
        if show_debug and mask is not None:
            mask_img = cv2.resize(mask, (pipe.cw, pipe.ch), interpolation=cv2.INTER_NEAREST)
            if mask_win_size != mask_img.shape:
                cv2.namedWindow(MASK_WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                fit_window(MASK_WIN, mask_img.shape[1], mask_img.shape[0], screen)
                mask_win_size = mask_img.shape
            cv2.imshow(MASK_WIN, mask_img)


        target_dt = (1.0 / args.fps_cap) if src.is_camera else src.dt
        delay_ms = int(round(max(0.0, target_dt - (time.perf_counter() - t_now)) * 1000.0)) + 1
        key = cv2.waitKey(max(1, delay_ms)) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                try:
                    cv2.destroyWindow(MASK_WIN)
                except cv2.error:
                    pass
                mask_win_size = None
        elif key == ord('b'):
            show_obj = not show_obj
        elif key == ord('s'):
            fn = "snapshot_%d.png" % int(time.time())
            cv2.imwrite(fn, canvas)
            print("已保存", fn)


        if t_frame > 2:
            try:
                visible = cv2.getWindowProperty(MAIN_WIN, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                visible = 0
            if visible < 1:
                print("视频窗口已关闭，退出。")
                break

    src.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return 0


def _synth_sequence(width=640, height=480, n=150, seed=0, ow=60, oh=60,
                    amp=6.0, rot_deg=0.6):
    """合成"手持抖动 + 画面内运动物体"序列，返回 (frames, boxes)"""
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), 60, np.uint8)
    for _ in range(800):
        cv2.circle(base, (int(rng.integers(0, width)), int(rng.integers(0, height))),
                   int(rng.integers(2, 6)), (200, 200, 200), -1)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    cv2.putText(base, "STATIC BACKGROUND", (30, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    obj_tex = cv2.GaussianBlur(
        cv2.resize(rng.integers(40, 230, size=(oh // 4, ow // 4, 3), dtype=np.uint8),
                   (ow, oh), interpolation=cv2.INTER_NEAREST), (3, 3), 0)

    cx, cy = width / 2.0, height / 2.0
    frames, boxes = [], []
    for t in range(n):
        ox = int(width * 0.5 + 0.26 * width * np.sin(2 * np.pi * t / 70.0)) - ow // 2
        oy = int(height * 0.5 + 0.16 * height * np.sin(2 * np.pi * t / 45.0)) - oh // 2
        img = base.copy()
        img[oy:oy + oh, ox:ox + ow] = obj_tex
        cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (255, 255, 255), 2)
        jx = amp * np.sin(2 * np.pi * t / 9.0) + 0.6 * amp * np.sin(2 * np.pi * t / 4.0 + 1.0)
        jy = 0.8 * amp * np.sin(2 * np.pi * t / 11.0 + 0.6)
        ja = np.deg2rad(rot_deg * np.sin(2 * np.pi * t / 13.0))
        c, s = np.cos(ja), np.sin(ja)
        M = np.float32([[c, -s, cx * (1 - c) + cy * s + jx],
                        [s, c, cy * (1 - c) - cx * s + jy]])
        frames.append(cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE))
        boxes.append((ox, oy, ow, oh))
    return frames, boxes


def _check_perspective():
    """【核心新功能的确定性检查】同样"相对速度"的物体，在任何画面高度判定必须一致。"""
    det = MotionDetector(320, 240, detect_width=320, min_mag=0.6, rel_sens=0.6)
    h, w = det.dh, det.dw
    bgx = np.zeros((h, w), np.float32)
    bgy = np.zeros((h, w), np.float32)
    bgx[:h // 2, :] = 0.5
    bgx[h // 2:, :] = 10.0
    thr = det.perspective_threshold(bgx, bgy, 0.0)

    far_r, near_r = 10, h - 10
    thr_far, thr_near = float(thr[far_r, w // 2]), float(thr[near_r, w // 2])
    got_far = 1.5 * 0.5 > thr_far
    got_near = 1.5 * 10.0 > thr_near
    junk_far = 0.2 * 0.5 > thr_far
    junk_near = 0.2 * 10.0 > thr_near
    fixed_far = 1.5 * 0.5 > 1.0
    fixed_near = 0.2 * 10.0 > 1.0
    print("       相对判据：远处门槛 %.2fpx 近处门槛 %.2fpx（相差 %.0f 倍，正好抵消透视）"
          % (thr_far, thr_near, thr_near / max(1e-6, thr_far)))
    print("       1.5 倍背景速度的目标 -> 远处 %s / 近处 %s（应都检出）"
          % ("检出" if got_far else "漏检", "检出" if got_near else "漏检"))
    print("       0.2 倍背景速度的杂波 -> 远处 %s / 近处 %s（应都拒掉）"
          % ("误检" if junk_far else "拒掉", "误检" if junk_near else "拒掉"))
    print("       同一场景用固定 1.0px 阈值：远处真目标 %s、近处杂波 %s（这就是旧版偏差）"
          % ("漏检" if not fixed_far else "检出", "误检" if fixed_near else "拒掉"))
    return got_far and got_near and (not junk_far) and (not junk_near)


def _check_background_flow():
    """确定性检查：背景运动场能扣掉"随 y 变化的视差"，但不吃掉比方块小的独立运动目标。"""
    h, w = 240, 320
    yy = np.arange(h, dtype=np.float32)[:, None]
    base = np.repeat(0.2 + 8.0 * (yy / float(h)) ** 2, w, axis=1)
    det = MotionDetector(w, h, detect_width=w, bg_block_ratio=0.06)

    small = base.copy()
    small[110:118, 110:118] += 4.0
    bx, _ = det.background_flow(small, np.zeros_like(small))
    bg_err = float(np.abs(small - bx)[20:80, 20:80].mean())
    res_small = float(np.abs(small - bx)[111:117, 111:117].mean())

    big = base.copy()
    big[100:130, 100:130] += 4.0
    bx2, _ = det.background_flow(big, np.zeros_like(big))
    res_big = float(np.abs(big - bx2)[108:122, 108:122].mean())
    print("       背景残差 %.3f 像素（应<0.3）；小目标处残差 %.2f 像素（应>3）"
          % (bg_err, res_small))
    print("       大目标(比方块大)内部残差 %.2f 像素 —— 被背景场吃掉一部分，属已知上限"
          % res_big)
    return bg_err < 0.3 and res_small > 3.0


def _check_boxes_filters():
    """确定性检查：成框过滤 —— 孤立目标必须出框、面积上下限生效、写反要自动交换。"""
    det = MotionDetector(640, 480, detect_width=480, min_area_ratio=0.0005)
    det.n_frame = 100

    det.mask = np.zeros((det.dh, det.dw), np.uint8)
    det.mask[100:140, 120:160] = 255
    n_one = len(det.boxes())

    bw, bh = int(det.dw * 0.8), int(det.dh * 0.75)
    det.mask = np.zeros((det.dh, det.dw), np.uint8)
    det.mask[10:10 + bh, 10:10 + bw] = 255
    det.set_area_limits(0.0005, 0.0)
    n_off = len(det.boxes())
    det.set_area_limits(0.0005, 0.25)
    n_max = len(det.boxes())
    det.set_area_limits(0.90, 0.0)
    n_min = len(det.boxes())
    det.set_area_limits(0.90, 0.0005)
    n_swap = len(det.boxes())
    print("       孤立 40x40 -> %d 个框；占画面 60%% 的大块：上限关->%d 上限25%%->%d "
          "下限90%%->%d 写反->%d" % (n_one, n_off, n_max, n_min, n_swap))
    return n_one == 1 and n_off == 1 and n_max == 0 and n_min == 0 and n_swap == 1


def _check_nested_boxes():
    """确定性检查（宁缺毋滥）：大框里套小框只留大框；分开的两个框都要留。"""
    a = merge_rects([[100, 100, 120, 100], [140, 130, 30, 30]], 0.5, 0.5)
    far = merge_rects([[0, 0, 40, 40], [200, 0, 40, 40]], 0.5, 0.5)

    def mk(box, tid):
        return {"id": tid, "box": np.array(box, np.float32), "hits": 5, "missed": 0,
                "vel": np.zeros(2, np.float32), "trail": deque(), "flags": deque()}
    kept = suppress_nested([mk([100, 100, 100, 100], 1), mk([140, 140, 20, 20], 2)], 0.5)
    kept2 = suppress_nested([mk([0, 0, 40, 40], 1), mk([200, 0, 40, 40], 2)], 0.5)
    print("       合并：大框+框内小框 -> %d 个（应 1）；分开的两个 -> %d 个（应 2）"
          % (len(a), len(far)))
    print("       显示去重：嵌套轨迹 -> 保留 %d 条（应 1）；分开的 -> %d 条（应 2）"
          % (len(kept), len(kept2)))
    return len(a) == 1 and len(far) == 2 and len(kept) == 1 and len(kept2) == 2


def _check_motion_gate():
    """确定性检查：钉在原地的框必须被挡掉，真的移动过的才放行"""
    dt = 1.0 / 30.0
    trk = BoxTracker()
    st = trk._new_track([100, 100, 40, 40], True)
    mv = trk._new_track([300, 100, 40, 40], True)
    for k in range(19):
        trk._update(st, [100, 100, 40, 40], True, dt)
        trk._update(mv, [300 + 2 * (k + 1), 100, 40, 40], True, dt)
    a, b = BoxTracker.is_really_moving(st), BoxTracker.is_really_moving(mv)
    print("       原地不动 -> %s；每帧走 2 像素 -> %s（期望挡掉/放行）"
          % ("放行" if a else "挡掉", "放行" if b else "挡掉"))
    return (not a) and b


def _check_output_resolution():
    """确定性检查：输出尺寸必须跟着输入分辨率走（曾经被 --max-width 默认值压成同一个尺寸）"""
    args = build_parser().parse_args([])
    got = {}
    for (w, h) in ((640, 480), (1280, 720), (1920, 1080), (720, 1280)):
        limited, _ = limit_frame(np.zeros((h, w, 3), np.uint8), args.max_width, None)
        pipe = Pipeline(limited.shape[1], limited.shape[0], args)
        got[(w, h)] = (pipe.canvas.shape[1], pipe.canvas.shape[0])
        print("       %4dx%-4d -> 输出 %4dx%-4d（内部检测宽度 %d，与输出无关）"
              % (w, h, pipe.canvas.shape[1], pipe.canvas.shape[0], pipe.det.dw))
    return len(set(got.values())) == len(got) and got[(1920, 1080)][0] > got[(640, 480)][0]


def _write_test_video(path, n=20, w=320, h=240, fps=30.0, step=8):
    """写一段"白点匀速移动"的测试视频。本机没有编码器时返回 False（跳过该检查）。"""
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        return False
    for i in range(n):
        img = np.full((h, w, 3), 30, np.uint8)
        cv2.circle(img, (10 + i * step, h // 2), 6, (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    return True


def _blob_x(bgr):
    """取画面里那个白点所在的列（按列统计亮像素数，取最多的那列）"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return int(np.argmax((g > 200).sum(axis=0)))


def _check_fps_cap(fps_cap=FPS_CAP):
    """确定性检查：60fps 源被降到 30fps，且播放速度不变（丢帧而不是慢放）"""
    path = os.path.join(tempfile.gettempdir(), "video_proc3_fps_%d.mp4" % os.getpid())
    if not _write_test_video(path, 60, 320, 240, 60.0, 4):
        print("       跳过：本机无法创建测试视频")
        return None
    src = FrameSource([("file", path)], 640, 480, fps_cap)
    if not src.open(0) or src.fps_src <= fps_cap * 1.05:
        print("       跳过：编码器写出的帧率是 %.1f（未超上限）" % src.fps_src)
        src.release()
        os.remove(path)
        return None
    xs = []
    while len(xs) < 8:
        ok, f = src.read()
        if not ok:
            break
        xs.append(_blob_x(f))
    step = float(np.median(np.diff(xs))) if len(xs) >= 2 else 0.0
    src.release()
    os.remove(path)
    print("       源 60fps -> 有效 30fps；相邻帧白点位移 %.1fpx（应≈8）" % step)
    return 6.5 <= step <= 9.5


def _check_loop():
    """确定性检查：文件读完能回到开头继续读"""
    path = os.path.join(tempfile.gettempdir(), "video_proc3_loop_%d.mp4" % os.getpid())
    n = 20
    if not _write_test_video(path, n, 320, 240, 30.0, 8):
        print("       跳过：本机无法创建测试视频")
        return None
    src = FrameSource([("file", path)], 640, 480, FPS_CAP)
    if not src.open(0):
        os.remove(path)
        return None
    xs = []
    while True:
        ok, f = src.read()
        if not ok:
            break
        xs.append(_blob_x(f))
    did = src.rewind()
    ok2, f2 = src.read()
    x0 = _blob_x(f2) if ok2 else -1
    src.release()
    os.remove(path)
    print("       读满 %d 帧后 rewind=%s，重读首帧 x=%d" % (len(xs), did, x0))
    return bool(did) and ok2 and len(xs) == n and bool(xs) and x0 == xs[0]


def _check_end_to_end():
    """端到端：合成"抖动 + 运动物体"，跑完整链路（消抖 -> 检测 -> 跟踪），量命中率。"""
    W, H, N = 640, 480, 150
    frames, boxes = _synth_sequence(W, H, N)
    stab = VideoStabilizer(W, H, proc_width=plan_work_width(W, H, 0.5), window=30)
    m = stab.crop
    cw, ch = W - 2 * m, H - 2 * m
    det = MotionDetector(cw, ch, detect_width=plan_work_width(cw, ch, 0.75, 240, 640, 350000),
                         min_area_ratio=0.0005, max_area_ratio=0.25, warmup=25)
    trk = BoxTracker()
    dt = 1.0 / 25.0
    nstat = nhit = 0
    cerr = []
    for i, f in enumerate(frames):
        stab_crop, _ = stab.process(f)
        det.apply(stab_crop)
        bx = det.boxes()
        tracks = trk.update(bx, det.judge(bx) if bx else [], dt)
        if i <= det.warmup:
            continue
        nstat += 1
        tb = boxes[i]
        truth = (tb[0] - m, tb[1] - m, tb[2], tb[3])
        best, bb = 0.0, None
        for t in tracks:
            v = box_iou(t["box"], truth)
            if v > best:
                best, bb = v, t["box"]
        if best >= 0.2:
            nhit += 1
            cerr.append(float(np.hypot(bb[0] + bb[2] / 2 - (truth[0] + truth[2] / 2),
                                       bb[1] + bb[3] / 2 - (truth[1] + truth[3] / 2))))
    hit = nhit / max(1, nstat)
    print("       统计 %d 帧：命中率(IoU>=0.2) %.1f%%  平均中心误差 %s"
          % (nstat, 100 * hit, ("%.1f px" % float(np.mean(cerr))) if cerr else "n/a"))
    return hit >= 0.5


def run_selftest():
    """无摄像头自检：地基 -> 核心新功能 -> 去重 -> 端到端，逐项量化。"""
    print("=" * 62)
    print("video_proc3 自检（消抖 + 透视加权运动检测）  OpenCV %s" % cv2.__version__)
    print("=" * 62)

    print("\n[0] 分辨率自适应规划")
    for (w, h) in ((640, 480), (1280, 720), (1920, 1080), (720, 1280), (1080, 1920)):
        print("       %4dx%-4d -> 检测处理宽度 %4d" % (w, h, plan_work_width(w, h, 0.75, 240, 640, 350000)))
    for f in (24.0, 30.0, 60.0, 120.0):
        sk, eff = plan_source_fps(f, FPS_CAP)
        print("       帧率 %6.1ffps -> 丢 %d 帧/次，有效 %5.1ffps" % (f, sk, eff))
    results = {"自适应规划": True}

    checks = (
        ("输出尺寸跟随输入分辨率", _check_output_resolution),
        ("背景运动场（扣视差、留独立运动）", _check_background_flow),
        ("透视加权阈值（核心：各高度判定一致）", _check_perspective),
        ("大框套小框去重（合并 + 显示去重）", _check_nested_boxes),
        ("成框过滤（孤立目标出框 / 面积上下限）", _check_boxes_filters),
        ("运动门限（钉在原地的假框不能显示）", _check_motion_gate),
        ("帧率上限 30", lambda: _check_fps_cap(FPS_CAP)),
        ("视频循环", _check_loop),
        ("端到端：抖动 + 运动物体（消抖->检测->跟踪）", _check_end_to_end),
    )
    for i, (title, fn) in enumerate(checks, 1):
        print("\n[%d] %s" % (i, title))
        r = fn()
        results[title.split("（")[0].split("：")[0]] = True if r is None else r

    print("\n====================== 自检结论 ======================")
    all_ok = all(results.values())
    for k, v in results.items():
        print("       %-12s %s" % (k[:12], "PASS" if v else "FAIL"))
    print("       %s" % ("全部通过" if all_ok else "存在失败项"))
    print("=====================================================")
    return 0 if all_ok else 1


def build_parser():
    ap = argparse.ArgumentParser(description="移动摄像头消抖 + 运动物体检测（video_proc3）")

    ap.add_argument("--video", nargs="+", default=None, help="输入视频文件（可多个，各算一个通道）")
    ap.add_argument("--camera", type=int, default=0, help="启动时的摄像头编号")
    ap.add_argument("--channel", type=int, default=None, help="启动时的通道序号")
    ap.add_argument("--cameras", type=int, default=3, help="暴露几个摄像头通道")
    ap.add_argument("--width", type=int, default=640, help="摄像头采集宽度（期望值）")
    ap.add_argument("--height", type=int, default=480, help="摄像头采集高度（期望值）")
    ap.add_argument("--fps-cap", dest="fps_cap", type=float, default=FPS_CAP, help="帧率上限")
    ap.add_argument("--no-loop", dest="loop", action="store_false", default=True,
                    help="视频播完不再循环（默认循环）")

    ap.add_argument("--max-width", dest="max_width", type=int, default=0,
                    help="处理与输出的宽度上限（默认 0 = 不限制，输出与输入同分辨率）")
    ap.add_argument("--view-width", dest="view_width", type=int, default=0,
                    help="显示窗口宽度上限（默认 0 = 按屏幕自动；只影响屏幕，不改保存的视频）")
    ap.add_argument("--out-panel", dest="out_panel", choices=("both", "right"), default="both",
                    help="保存哪种画面：both = 对比画布；right = 只存消抖+检测那一半")
    ap.add_argument("--layout", choices=("auto", "h", "v"), default="auto", help="画布布局")

    ap.add_argument("--det-width", dest="det_width", type=int, default=0,
                    help="检测处理宽度（默认 0 = 按画面尺寸自动规划，夹在 240~640，"
                         "并受 35 万像素的面积预算约束）。调小提速、调大更细")
    ap.add_argument("--stab-width", dest="stab_width", type=int, default=None,
                    help="消抖光流处理宽度（默认按画面宽度自动算）")
    ap.add_argument("--window", type=int, default=30, help="消抖轨迹平滑窗口(帧)")
    ap.add_argument("--jitter-ratio", dest="jitter_ratio", type=float, default=0.10,
                    help="抖动/运镜分界，占画面宽度比例（默认 0.10）")
    ap.add_argument("--band", type=float, default=0.03, help="软阈值过渡带宽度")
    ap.add_argument("--crop-ratio", dest="crop_ratio", type=float, default=0.10,
                    help="消抖输出裁剪比例（默认 0.10）")

    ap.add_argument("--min-mag", dest="min_mag", type=float, default=0.6,
                    help="残差绝对下限（像素，默认 0.6）：比它小的残差一律不算运动")
    ap.add_argument("--rel-sens", dest="rel_sens", type=float, default=0.6,
                    help="【透视加权】物体要跑出当地背景速度的比例（默认 0.6 = 60%%。"
                         "近处地面一帧走十几像素、远处只走零点几像素，"
                         "这个比例是透视不变的判据；调大更干净、调小更灵敏）")
    ap.add_argument("--noise-k", dest="noise_k", type=float, default=3.0,
                    help="噪声底倍数（默认 3：门槛 = max(min_mag, rel×背景速度, 3×残差中位)）")
    ap.add_argument("--bg-block-ratio", dest="bg_block_ratio", type=float, default=0.06,
                    help="背景运动场的方块边长占检测宽度比例（默认 0.06；0 = 退化成减全场中值）")
    ap.add_argument("--min-area-ratio", dest="min_area_ratio", type=float, default=0.0005,
                    help="物体面积占画面比例下限（默认 0.0005 = 0.05%%）")
    ap.add_argument("--max-area-ratio", dest="max_area_ratio", type=float, default=0.0,
                    help="物体面积占画面比例上限（默认 0 = 不限制）")
    ap.add_argument("--merge-gap", dest="merge_gap", type=float, default=0.5,
                    help="合并同一物体碎块的空隙比例（默认 0.5；0 = 不合并）")
    ap.add_argument("--contain-ratio", dest="contain_ratio", type=float, default=0.5,
                    help="【去嵌套框】包含度 IoMin 阈值（默认 0.5：小框被大框吃掉一半以上"
                         "就判为重复，只留大框；0 = 关闭）")
    ap.add_argument("--roi-top", dest="roi_top", type=float, default=0.0,
                    help="只在画面高度的该百分数以下检测（默认 0 = 不限）")
    ap.add_argument("--roi-bottom", dest="roi_bottom", type=float, default=100.0,
                    help="只在画面高度的该百分数以上检测（默认 100 = 不限）")
    ap.add_argument("--warmup", type=int, default=2, help="前几帧不出框")
    ap.add_argument("--no-motion-gate", dest="motion_gate", action="store_false", default=True,
                    help="关闭[必须真的移动过]的显示门限（默认开启）")
    ap.add_argument("--max-show", dest="max_show", type=int, default=12,
                    help="最多同时显示几个目标（默认 12；0 = 不限制）")
    ap.add_argument("--no-object", dest="no_object", action="store_true", help="只看画面不检测")
    ap.add_argument("--mask-overlay", dest="mask_overlay", action="store_true",
                    help="把前景掩码以半透明红色叠在输出画面上（默认关闭；"
                         "掩码来自低分辨率的最近邻放大，且帧差掩码天生带拖影，"
                         "只在调参时打开。只看掩码可以按 d 开掩码窗口）")
    ap.add_argument("--debug", action="store_true", help="启动时打开掩码视图")

    ap.add_argument("-o", "--output", default=None, help="保存对比视频")
    ap.add_argument("--selftest", action="store_true", help="跑合成数据自检")
    return ap


if __name__ == "__main__":
    _args = build_parser().parse_args()
    if _args.selftest:
        raise SystemExit(run_selftest())
    raise SystemExit(run(_args))
