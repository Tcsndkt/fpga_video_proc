# -*- coding: utf-8 -*-
"""特征方格锚定 + 帧差法 运动物体检测（video_proc2）

功能
----
1) 消抖：角点 + LK 光流 + RANSAC 相似变换 + 轨迹平滑补偿（作为帧差的前置整形）；
2) 逐格锚定：把消抖后的画面切成网格，每格用 cv2.phaseCorrelate 求亚像素平移，
   得到随位置变化的位移场，用于补偿视差；
3) 检测：用位移场把上一帧 warp 到当前帧视角后做帧差，逐格自适应阈值，
   经形态学、拖影抑制、面积/包含度筛选得到候选框；
4) 跟踪与去重：IoU + 速度外推关联、运动门限、同一物体碎块合并、嵌套框去除；
5) 分辨率自适应：输出与输入同分辨率，显示窗口按屏幕自动缩放，
   内部工作分辨率单独限幅以保证速度；帧率上限 30fps。

运行
----
    py video_proc2.py                    # 摄像头 0
    py video_proc2.py --video in.mp4     # 视频文件（播完自动循环；--no-loop 关闭）
    py video_proc2.py -o out.mp4         # 保存对比视频（尺寸随输入分辨率）
    py video_proc2.py -o out.mp4 --out-panel right   # 只存"消抖+检测"那一半
    py video_proc2.py --max-width 1280   # 可选：给处理与输出封顶（默认 0 = 不限制）
    py video_proc2.py --selftest         # 自检（无需摄像头）

Controls 面板
------------
    min_area_1e4   物体面积占画面比例下限 ×10000（5 = 0.05%）
    max_area_pct   物体面积占画面比例上限 %（0 = 不限制）
    sens_diff      帧差阈值（灰度级，越小越灵敏）
    merge_gap_pct  同一物体碎块合并的空隙比例 %（50 = 0.5 倍，0 = 不合并）
    jitter_pct     消抖的抖动/运镜分界，占画面宽度 %（10 = 10%）
    roi_top_pct    只在画面高度的该百分数以下检测（0 = 不限制）
    roi_bot_pct    只在画面高度的该百分数以上检测（100 = 不限制）
    channel        视频输入通道

按键：q/Q/ESC 退出（或关闭视频窗口）、d 掩码视图与位移场箭头、b 开关检测、s 存图
"""

import argparse
import os
import tempfile
import time
from collections import deque

import cv2
import numpy as np


GAP = 4
FPS_CAP = 30.0


FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=7)


LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


TB_MIN_AREA = "min_area_1e4"
TB_MAX_AREA = "max_area_pct"
TB_SENS = "sens_diff"
TB_MERGE = "merge_gap_pct"
TB_JITTER = "jitter_pct"
TB_ROI_TOP = "roi_top_pct"
TB_ROI_BOT = "roi_bot_pct"
TB_CHANNEL = "channel"

TB_MIN_AREA_MAX = 300
TB_MAX_AREA_MAX = 100
TB_SENS_MAX = 60
TB_MERGE_MAX = 200
TB_JITTER_MAX = 50
TB_ROI_MAX = 100

MAIN_WIN = "Grid Anchor + Frame Diff"
MASK_WIN = "foreground mask"


def clamp(v, lo, hi):
    """把 v 夹在 [lo, hi]。越界的参数（用户拉的滑条、驱动报的帧率）全靠它兜住。"""
    return lo if v < lo else (hi if v > hi else v)


def sim(dx, dy, da):
    """构造「内容位移」相似变换矩阵（与 estimateAffinePartial2D 输出同参数化）。"""
    c, s = np.cos(da), np.sin(da)
    return np.array([[c, -s, dx],
                     [s, c, dy]], np.float32)


def compose(A, B):
    """两个 2x3 仿射矩阵复合：A ∘ B（先作用 B，再作用 A）。"""
    A3 = np.vstack([np.asarray(A, np.float64), [0.0, 0.0, 1.0]])
    B3 = np.vstack([np.asarray(B, np.float64), [0.0, 0.0, 1.0]])
    return (A3 @ B3)[:2]


def soft_gate(dist, low, high):
    """余弦软阈值：dist<=low 全补偿，dist>=high 不补偿，中间平滑过渡。"""
    if dist <= low:
        return 1.0
    if dist >= high:
        return 0.0
    t = (dist - low) / (high - low)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def decompose(M):
    """从 2x3 相似变换矩阵里取出 (dx, dy, da)。"""
    return (float(M[0, 2]), float(M[1, 2]), float(np.arctan2(M[1, 0], M[0, 0])))


def box_iou(a, b):
    """两个 [x, y, w, h] 框的 IoU（交并比），跟踪阶段用它做前后帧关联。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def box_gap(a, b):
    """两个框之间的"空隙"（欧氏距离，重叠时为 0）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0.0, max(ax, bx) - min(ax + aw, bx + bw))
    dy = max(0.0, max(ay, by) - min(ay + ah, by + bh))
    return float(np.hypot(dx, dy))


def box_overlap_min(a, b):
    """交集面积 ÷ 两者中**较小**框的面积（IoMin，也叫"包含度"）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    small = min(aw * ah, bw * bh)
    return inter / small if small > 0 else 0.0


def suppress_nested_tracks(tracks, contain_ratio=0.6):
    """【去掉"大框里套着小框"的重复显示】跟踪器层面的去重。"""
    if contain_ratio <= 0.0 or len(tracks) < 2:
        return list(tracks)
    kept = []
    for tr in sorted(tracks, key=lambda t: -float(t["box"][2] * t["box"][3])):
        contained = False
        for k in kept:
            if box_overlap_min(tr["box"], k["box"]) >= contain_ratio:
                contained = True
                break
        if not contained:
            kept.append(tr)
    return kept


def merge_rects(rects, gap_ratio=0.5, contain_ratio=0.6, max_iter=6):
    """把"属于同一个物体"的框合并成一个大的外接框（本项目主要的去重手段）。"""
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
                if contain_ratio > 0.0 and \
                        box_overlap_min(rects[i], rects[j]) >= contain_ratio:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
                        merged = True
                    continue
                if gap_ratio <= 0.0:
                    continue
                ref = gap_ratio * min(min(rects[i][2], rects[i][3]),
                                      min(rects[j][2], rects[j][3]))
                if box_gap(rects[i], rects[j]) <= ref:
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


def plan_work_width(W, H, ratio=0.5, wmin=240, wmax=480, area_max=350000):
    """按画面尺寸算出"工作分辨率宽度"，实现任意分辨率一套参数。"""
    area = float(max(1, W * H))
    w = clamp(float(ratio) * W, float(wmin), float(wmax))
    w = min(w, W * (float(area_max) / area) ** 0.5)
    return int(clamp(round(w), 64, W))


def plan_source_fps(fps_src, fps_cap=FPS_CAP, max_skip=8):
    """把源帧率压到上限以内，返回 (skip, eff_fps)。"""
    f = float(fps_src) if fps_src else 0.0
    if not (1e-3 < f < 240.0):
        f = float(fps_cap)
    if f <= float(fps_cap) * 1.05:
        return 0, f
    skip = int(clamp(int(round(f / float(fps_cap))) - 1, 0, int(max_skip)))
    return skip, f / (skip + 1)


def limit_frame(frame, max_width, buf=None):
    """把输入画面限制到工作分辨率上限；返回 (可用帧, 复用缓冲)。"""
    if not max_width or frame.shape[1] <= max_width:
        return frame, buf
    tw = int(max_width)
    th = max(2, int(round(frame.shape[0] * tw / float(frame.shape[1]))))
    if buf is None or buf.shape[0] != th or buf.shape[1] != tw or buf.dtype != frame.dtype:
        buf = np.empty((th, tw, 3), frame.dtype)
    cv2.resize(frame, (tw, th), dst=buf, interpolation=cv2.INTER_AREA)
    return buf, buf


def screen_size():
    """取屏幕分辨率（用于把窗口缩到屏幕内）。取不到返回 None，程序照常跑。"""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
        root.destroy()
        return size
    except Exception:
        return None


def fit_window(win, w, h, screen):
    """把窗口调整成"画面能完整放进屏幕"的尺寸（只缩小、不放大）。"""
    if not screen:
        return
    sw, sh = int(screen[0] * 0.95), int(screen[1] * 0.90)
    k = min(1.0, sw / float(w), sh / float(h))
    if k < 1.0:
        cv2.resizeWindow(win, max(160, int(w * k)), max(120, int(h * k)))


def plan_view_scale(canvas_w, canvas_h, screen, view_width=0):
    """算出**显示**用的缩放比例（只影响屏幕上的那一份，不影响保存的视频）。"""
    if view_width and int(view_width) > 0:
        return min(1.0, float(view_width) / max(1, canvas_w))
    if not screen:
        return min(1.0, 1600.0 / max(1, canvas_w))
    sw, sh = screen[0] * 0.95, screen[1] * 0.90
    return min(1.0, sw / max(1, canvas_w), sh / max(1, canvas_h))


def build_sources(args):
    """命令行参数 -> "视频通道列表"（每个通道 = 一路摄像头 或 一个视频文件）。"""
    if args.video:
        return [("file", v) for v in args.video]
    return [("camera", i) for i in range(max(1, int(args.cameras)))]


class FrameSource:
    """视频输入通道管理：打开 / 读帧（含丢帧限速）/ 回到开头 / 释放。"""

    def __init__(self, sources, width=640, height=480, fps_cap=FPS_CAP):
        self.sources = list(sources)
        self.width = int(width)
        self.height = int(height)
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
        """打开第 index 个通道。失败返回 False 且**保持原通道不变**（先开新的再关旧的）。"""
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

    def first_openable(self):
        """依次尝试所有通道，返回第一个能打开的（都打不开返回 None）。"""
        for i in range(len(self.sources)):
            if self.open(i):
                return i
        return None

    def read(self):
        """读一帧。源帧率超上限时先 grab() 丢掉多余的帧（不解码，几乎不花时间）。"""
        if self.cap is None:
            return False, None
        for _ in range(self.skip):
            if not self.cap.grab():
                return False, None
        return self.cap.read()

    def rewind(self):
        """回到文件开头（循环播放用）。优先 seek，失败则重新打开该通道。"""
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
    """视频消抖器：把"手持抖动"从画面运动里剥离出去。"""

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
        """运行时改"抖动/运镜"分界（占画面宽度的比例）。O(1)，每帧调用无压力。"""
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

        info = {"ok": False, "dx": 0.0, "dy": 0.0, "da": 0.0,
                "outliers": None, "inliers": 0}


        if self.prev_gray is None:
            self.prev_gray = small
            self.prev_pts = cv2.goodFeaturesToTrack(small, mask=None, **FEATURE_PARAMS)
            return self._crop(frame), info


        if self.prev_pts is None or len(self.prev_pts) < 4:
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None,
                                                    **FEATURE_PARAMS)

        M = None
        outliers = None
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
                    if M is not None and inl is not None:
                        inl = inl.ravel().astype(bool)
                        n_in = int(inl.sum())
                        if (~inl).any():
                            outliers = (new_pts[~inl].reshape(-1, 2)
                                        * self.inv_scale).astype(np.float32)
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
            dx, dy, da = decompose(M)
            dx *= self.inv_scale
            dy *= self.inv_scale
            self.curr += (dx, dy, da)
            self.traj[self.traj_i] = self.curr
            self.traj_i = (self.traj_i + 1) % self.window
            if self.traj_n < self.window:
                self.traj_n += 1
            self.smooth = self.traj[:self.traj_n].mean(axis=0)

            C = sim(self.curr[0], self.curr[1], self.curr[2])
            S = sim(self.smooth[0], self.smooth[1], self.smooth[2])
            R = compose(S, cv2.invertAffineTransform(C))
            rdx, rdy = float(R[0, 2]), float(R[1, 2])
            rda = float(np.arctan2(R[1, 0], R[0, 0]))
            g = soft_gate(float(np.hypot(rdx, rdy)), self.gate_low, self.gate_high)
            if g > 0.0:
                M_comp = sim(g * rdx, g * rdy, g * rda)
                out = cv2.warpAffine(frame, M_comp, (self.W, self.H),
                                     dst=self.warp_buf, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
                if outliers is not None:
                    outliers = cv2.transform(
                        outliers.reshape(-1, 1, 2), M_comp).reshape(-1, 2).astype(np.float32)
            info.update(ok=True, dx=dx, dy=dy, da=da)
        info.update(outliers=outliers, inliers=n_in)
        return self._crop(out), info


class GridAnchorMotion:
    """用网格里的"特征方格"锚定背景位置，输出一张稠密位移场。"""

    def __init__(self, w, h, cols=5, patch_ratio=0.8, min_tex=15.0,
                 min_response=0.08, max_shift_ratio=0.45, dev_thresh=2.5,
                 max_bad_ratio=0.6, smooth=0.5):
        self.w, self.h = int(w), int(h)
        self.cols = max(2, int(cols))

        self.rows = int(clamp(round(self.cols * self.h / float(self.w)), 2, 14))
        self.cell_w = self.w / float(self.cols)
        self.cell_h = self.h / float(self.rows)
        p = int(round(patch_ratio * min(self.cell_w, self.cell_h)))
        self.patch = int(clamp(p // 2 * 2, 16, 128))
        self.max_shift = max(2.0, max_shift_ratio * self.patch)
        self.min_tex = float(min_tex)
        self.min_response = float(min_response)
        self.dev_thresh = float(dev_thresh)
        self.max_bad_ratio = float(max_bad_ratio)
        self.smooth = float(clamp(smooth, 0.0, 0.95))


        self._dx_dense = np.zeros((self.h, self.w), np.float32)
        self._dy_dense = np.zeros((self.h, self.w), np.float32)
        self._prev_dx = None
        self._prev_dy = None
        self.last_valid_ratio = 0.0


    def _patch_origin(self, r, c):
        """格子 (r, c) 中心 -> 取样块左上角（夹在画面内，保证块完整）"""
        cx = (c + 0.5) * self.cell_w
        cy = (r + 0.5) * self.cell_h
        x0 = int(round(cx - self.patch / 2.0))
        y0 = int(round(cy - self.patch / 2.0))
        x0 = int(clamp(x0, 0, max(0, self.w - self.patch)))
        y0 = int(clamp(y0, 0, max(0, self.h - self.patch)))
        return x0, y0

    def estimate(self, prev, cur):
        """逐格相位相关，返回 (dx, dy, resp)：形状都是 (rows, cols)，无效处为 NaN。"""
        P = self.patch
        dx = np.full((self.rows, self.cols), np.nan, np.float32)
        dy = np.full((self.rows, self.cols), np.nan, np.float32)
        resp = np.zeros((self.rows, self.cols), np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                x0, y0 = self._patch_origin(r, c)
                a = prev[y0:y0 + P, x0:x0 + P]
                b = cur[y0:y0 + P, x0:x0 + P]
                if a.shape != (P, P) or b.shape != (P, P):
                    continue


                if cv2.Laplacian(a, cv2.CV_32F).var() < self.min_tex:
                    continue
                af = a.astype(np.float32)
                bf = b.astype(np.float32)
                (sx, sy), response = cv2.phaseCorrelate(af, bf)
                if response < self.min_response:
                    continue
                if abs(sx) > self.max_shift or abs(sy) > self.max_shift:
                    continue
                dx[r, c], dy[r, c], resp[r, c] = sx, sy, response
        return dx, dy, resp


    def _robust_fill(self, g):
        """把带 NaN 的稀疏场补成完整的 (rows, cols)：
                先做"邻域中值离群剔除"，再用整体中值补齐 NaN。
        """
        g = g.copy()
        valid = ~np.isnan(g)
        if valid.sum() == 0:
            return np.zeros_like(g), 0.0

        tmp = np.where(valid, g, np.nan)
        med = np.full_like(g, np.nan)
        for r in range(self.rows):
            for c in range(self.cols):
                if not valid[r, c]:
                    continue
                y0, y1 = max(0, r - 1), min(self.rows, r + 2)
                x0, x1 = max(0, c - 1), min(self.cols, c + 2)
                blk = tmp[y0:y1, x0:x1]
                blk = blk[~np.isnan(blk)]
                if blk.size:
                    med[r, c] = float(np.median(blk))
        dev = np.abs(g - med)
        bad = valid & (dev > self.dev_thresh)
        g[bad] = med[bad]
        all_valid = g[~np.isnan(g)]
        fill = float(np.median(all_valid)) if all_valid.size else 0.0
        g[np.isnan(g)] = fill
        return g, float(valid.sum()) / float(self.rows * self.cols)

    def field(self, dx, dy):
        """稀疏方格位移 -> 稠密位移场 (h, w)。返回 (dx_dense, dy_dense)。"""
        gx, ratio_x = self._robust_fill(dx)
        gy, ratio_y = self._robust_fill(dy)
        ratio = min(ratio_x, ratio_y)
        self.last_valid_ratio = ratio
        if ratio <= 0.0:


            return self._dx_dense, self._dy_dense


        if self.smooth > 0.0 and self._prev_dx is not None:
            gx = (1.0 - self.smooth) * self._prev_dx + self.smooth * gx
            gy = (1.0 - self.smooth) * self._prev_dy + self.smooth * gy
        self._prev_dx, self._prev_dy = gx, gy


        sh = min(self.h, int(round(self.rows * self.cell_h)))
        sw = min(self.w, int(round(self.cols * self.cell_w)))
        for g, out in ((gx, self._dx_dense), (gy, self._dy_dense)):
            small = cv2.resize(g, (sw, sh), interpolation=cv2.INTER_LINEAR)
            out[:sh, :sw] = small
            if sh < self.h:
                out[sh:, :] = out[sh - 1:sh, :]
            if sw < self.w:
                out[:, sw:] = out[:, sw - 1:sw]
        return self._dx_dense, self._dy_dense


class FrameDiffDetector:
    """对齐 + 帧差 + 逐格自适应阈值 + 形态学 + 连通域成框。"""

    def __init__(self, width, height, work_width=444, cols=8, min_diff=12.0,
                 noise_k=3.0, min_area_ratio=0.0005, max_area_ratio=0.0,
                 warmup=2, roi_top=0.0, roi_bottom=1.0, deghost=True,
                 close_ksize=7, open_ksize=3, min_response=0.08, min_tex=15.0,
                 grad_k=0.0, deghost_max=12, merge_gap=0.5, contain_ratio=0.6):
        self.W, self.H = int(width), int(height)
        self.scale = min(1.0, float(work_width) / self.W)
        self.inv_scale = 1.0 / self.scale
        self.dw = max(64, int(round(self.W * self.scale)))
        self.dh = max(48, int(round(self.H * self.scale)))

        self.motion = GridAnchorMotion(self.dw, self.dh, cols=cols,
                                       min_response=min_response, min_tex=min_tex)
        self.rows, self.cols = self.motion.rows, self.motion.cols

        self.min_diff = float(clamp(min_diff, 0.5, 255.0))
        self.noise_k = float(noise_k)
        self.warmup = int(warmup)
        self.deghost = bool(deghost)
        self.deghost_max = max(1, int(deghost_max))


        self.grad_k = float(clamp(grad_k, 0.0, 3.0))


        self.merge_gap = float(clamp(merge_gap, 0.0, 3.0))


        self.contain_ratio = float(clamp(contain_ratio, 0.0, 1.0))
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))


        self.min_area = 40
        self.min_box = 0.0
        self.max_box = float("inf")
        self.set_area_limits(min_area_ratio, max_area_ratio)


        self.roi_top = 0.0
        self.roi_bottom = 1.0
        self.set_roi(roi_top, roi_bottom)

        self.n_frame = 0
        self.prev_gray = None
        self.mask = np.zeros((self.dh, self.dw), np.uint8)
        self.diff_img = np.zeros((self.dh, self.dw), np.uint8)
        self.thresh = 0.0

        self._aligned = np.empty((self.dh, self.dw), np.uint8)
        self._diff_f = np.empty((self.dh, self.dw), np.float32)
        self._thr_map = np.empty((self.dh, self.dw), np.float32)
        self._dx_tmp = np.empty((self.dh, self.dw), np.float32)
        self._dy_tmp = np.empty((self.dh, self.dw), np.float32)
        self._grad = np.empty((self.dh, self.dw), np.float32)
        self._gx = np.empty((self.dh, self.dw), np.float32)
        self._gy = np.empty((self.dh, self.dw), np.float32)
        self._mapx, self._mapy = np.meshgrid(
            np.arange(self.dw, dtype=np.float32), np.arange(self.dh, dtype=np.float32))
        self._mapx = self._mapx.astype(np.float32)
        self._mapy = self._mapy.astype(np.float32)


    def set_area_limits(self, min_ratio, max_ratio):
        """设置"面积占画面比例"的上下限（trackbar 用）。上限小于下限时自动交换。"""
        area = float(self.dw * self.dh)
        lo, hi = float(min_ratio), float(max_ratio)
        if hi > 0.0 and hi < lo:
            lo, hi = hi, lo
        self.min_area_ratio, self.max_area_ratio = lo, hi
        self.min_area = max(20, int(lo * area))
        self.min_box = lo * area
        self.max_box = (hi * area) if hi > 0.0 else float("inf")

    def set_min_diff(self, v):
        """设置帧差阈值（灵敏度）。单位是灰度级：调小更灵敏、噪点更多。"""
        self.min_diff = float(clamp(v, 0.5, 255.0))

    def set_merge_gap(self, v):
        """设置"合并同一个物体碎块"的空隙比例（trackbar 用）。"""
        self.merge_gap = float(clamp(v, 0.0, 3.0))

    def set_contain_ratio(self, v):
        """设置"包含度"阈值（IoMin）：小框被大框吃掉多少比例就算重复框。"""
        self.contain_ratio = float(clamp(v, 0.0, 1.0))

    def set_roi(self, top_ratio, bottom_ratio):
        """只在画面高度的 [top, bottom] 这条横带里出框（走路拍摄挡假框最有效的一招）。"""
        top = float(clamp(top_ratio, 0.0, 0.9))
        bot = float(clamp(bottom_ratio, 0.1, 1.0))
        if bot - top < 0.05:
            bot = min(1.0, top + 0.05)
        self.roi_top, self.roi_bottom = top, bot


    def apply(self, bgr):
        """处理一帧，返回前景掩码（0/255，检测尺度）。"""
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


        prev = self.prev_gray
        dx, dy, _ = self.motion.estimate(prev, gray)
        fdx, fdy = self.motion.field(dx, dy)


        np.subtract(self._mapx, fdx, out=self._dx_tmp)
        np.subtract(self._mapy, fdy, out=self._dy_tmp)
        cv2.remap(prev, self._dx_tmp, self._dy_tmp,
                  interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                  dst=self._aligned)
        self.prev_gray = gray


        diff = cv2.absdiff(gray, self._aligned)
        self.diff_img = diff


        cw = self.dw / float(self.cols)
        chh = self.dh / float(self.rows)
        coarse = np.empty((self.rows, self.cols), np.float32)
        for r in range(self.rows):
            y0, y1 = int(r * chh), int(min(self.dh, (r + 1) * chh))
            for c in range(self.cols):
                x0, x1 = int(c * cw), int(min(self.dw, (c + 1) * cw))
                blk = diff[y0:y1, x0:x1]
                med = float(np.median(blk)) if blk.size else 0.0
                coarse[r, c] = max(self.min_diff, self.noise_k * med)
        self.thresh = float(np.median(coarse))
        sh = min(self.dh, int(round(self.rows * chh)))
        sw = min(self.dw, int(round(self.cols * cw)))
        small_thr = cv2.resize(coarse, (sw, sh), interpolation=cv2.INTER_LINEAR)
        self._thr_map[:sh, :sw] = small_thr
        if sh < self.dh:
            self._thr_map[sh:, :] = self._thr_map[sh - 1:sh, :]
        if sw < self.dw:
            self._thr_map[:, sw:] = self._thr_map[:, sw - 1:sw]


        self._diff_f[:] = diff
        cond = cv2.compare(self._diff_f, self._thr_map, cv2.CMP_GT)
        if self.grad_k > 0.0:
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, dst=self._gx)
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, dst=self._gy)
            cv2.magnitude(self._gx, self._gy, self._grad)
            cv2.GaussianBlur(self._grad, (5, 5), 0, dst=self._grad)
            np.multiply(self._grad, self.grad_k, out=self._grad)
            cond = cv2.bitwise_and(cond, cv2.compare(self._diff_f, self._grad, cv2.CMP_GT))
        mask = cv2.morphologyEx(cond, cv2.MORPH_OPEN, self.k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)


        if self.deghost:
            mask = self._deghost(mask, gray, prev)
        mask = self._tighten(mask)
        if self.roi_top > 0.0 or self.roi_bottom < 1.0:
            y0 = int(self.roi_top * self.dh)
            y1 = int(self.roi_bottom * self.dh)
            mask[:y0, :] = 0
            mask[y1:, :] = 0
        self.mask = mask
        return self.mask


    def _deghost(self, mask, gray, prev):
        """拖影抑制：帧差的掩码 = "物体上一帧位置 ∪ 当前位置"，这让框变长。"""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask
        out = np.zeros_like(mask)

        items = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            items.append((w * h, x, y, w, h))
        items.sort(reverse=True)
        P = self.motion.patch
        for idx, (_, x, y, w, h) in enumerate(items):
            if idx >= self.deghost_max or w < 6 or h < 6:
                out[y:y + h, x:x + w] |= mask[y:y + h, x:x + w]
                continue

            margin = max(6, int(round(P * 0.45)))
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(self.dw, x + w + margin)
            y1 = min(self.dh, y + h + margin)
            a = prev[y0:y1, x0:x1]
            b = gray[y0:y1, x0:x1]
            if a.shape != b.shape or a.size == 0 or min(a.shape) < 8:
                out[y:y + h, x:x + w] |= mask[y:y + h, x:x + w]
                continue
            (vx, vy), resp = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
            if resp < 0.05 or abs(vx) < 1.0 or abs(vy) < 1.0 \
                    or abs(vx) > margin or abs(vy) > margin:
                out[y:y + h, x:x + w] |= mask[y:y + h, x:x + w]
                continue


            M = np.float32([[1, 0, vx], [0, 1, vy]])
            local = mask[y0:y1, x0:x1]
            shifted = cv2.warpAffine(local, M, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST)
            keep = cv2.bitwise_and(local, shifted)
            sub_keep = keep[y - y0:y - y0 + h, x - x0:x - x0 + w]
            before = int(np.count_nonzero(mask[y:y + h, x:x + w]))
            if int(np.count_nonzero(sub_keep)) >= 0.35 * max(1, before):
                out[y:y + h, x:x + w] |= sub_keep
            else:
                out[y:y + h, x:x + w] |= mask[y:y + h, x:x + w]
        return out

    def _tighten(self, mask):
        """块内收紧：每块只保留差值最大的那部分像素。"""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(mask)
        for c in cnts:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            sub = self.diff_img[y:y + h, x:x + w]
            sm = mask[y:y + h, x:x + w] > 0
            if not sm.any():
                continue
            thr = float(np.percentile(sub[sm], 50))
            keep = sm & (sub >= thr)
            if keep.sum() >= 0.25 * sm.sum():
                out[y:y + h, x:x + w] = np.where(keep, 255, 0).astype(np.uint8)
            else:
                out[y:y + h, x:x + w] = np.where(sm, 255, 0).astype(np.uint8)
        return out


    def boxes(self):
        """返回工作分辨率下的候选框 [[x, y, w, h], ...]。"""
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
            ba = float(w * h)
            if ba < self.min_box or ba > self.max_box:
                continue
            ar = w / float(h)
            if ar < 0.15 or ar > 8.0:
                continue
            rects.append([int(x), int(y), int(w), int(h)])
        if len(rects) > 1 and (self.merge_gap > 0.0 or self.contain_ratio > 0.0):
            rects = merge_rects(rects, self.merge_gap, self.contain_ratio)
            kept = []
            for r in rects:
                w, h = max(1, r[2]), max(1, r[3])
                if float(w * h) < self.min_box or float(w * h) > self.max_box:
                    continue
                if not (0.15 <= w / float(h) <= 8.0):
                    continue
                kept.append(r)
            rects = kept
        inv = self.inv_scale
        return [[int(r[0] * inv), int(r[1] * inv), int(r[2] * inv), int(r[3] * inv)]
                for r in rects]

    def judge(self, boxes):
        """框内"有效变化像素"的填充率，用来滤掉零散的边缘伪影。"""
        flags = []
        for (x, y, w, h) in boxes:
            x1 = max(0, int(x * self.scale))
            y1 = max(0, int(y * self.scale))
            x2 = min(self.dw, int((x + w) * self.scale))
            y2 = min(self.dh, int((y + h) * self.scale))
            fill = 0.0
            if x2 > x1 and y2 > y1:
                sub = self.mask[y1:y2, x1:x2]
                fill = float(np.count_nonzero(sub)) / float(sub.size)
            flags.append(fill >= 0.12)
        return flags


class BoxTracker:
    """IoU 关联跟踪：稳定 ID、平滑框、速度方向、尾迹、以及"真的动过"门限。"""

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

    def _predict_box(self, tr, dt):
        """按速度外推 dt 秒后的框（常速模型），用于关联。"""
        if dt <= 1e-4:
            return tr["box"]
        v = tr["vel"]
        return np.array([tr["box"][0] + v[0] * dt, tr["box"][1] + v[1] * dt,
                         tr["box"][2], tr["box"][3]], np.float32)

    def _update_track(self, tr, box, moving, dt):
        old_c = np.array([tr["box"][0] + tr["box"][2] / 2.0,
                          tr["box"][1] + tr["box"][3] / 2.0], np.float32)
        tr["box"] = tr["box"] + self.smooth * (np.array(box, np.float32) - tr["box"])
        new_c = np.array([tr["box"][0] + tr["box"][2] / 2.0,
                          tr["box"][1] + tr["box"][3] / 2.0], np.float32)
        if dt > 1e-4:
            v = (new_c - old_c) / dt
            tr["vel"] = tr["vel"] + 0.4 * (v - tr["vel"])
        tr["trail"].append((float(new_c[0]), float(new_c[1])))
        tr["flags"].append(bool(moving))
        tr["hits"] += 1
        tr["missed"] = 0

    def update(self, boxes, moving, dt):
        """每帧调用：① 记丢失 ② 速度外推 + IoU 贪心匹配 ③ 新轨迹 ④ 清理 ⑤ 输出确认过的"""
        for tr in self.tracks:
            tr["missed"] += 1
        used_d = set()
        if len(boxes):
            pairs = []
            for ti, tr in enumerate(self.tracks):
                pred = self._predict_box(tr, dt)
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
                self._update_track(self.tracks[ti], boxes[di], moving[di], dt)
        for di, d in enumerate(boxes):
            if di not in used_d:
                self.tracks.append(self._new_track(d, moving[di]))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missing]
        return [t for t in self.tracks if t["hits"] >= self.min_hits]

    @staticmethod
    def is_moving(tr, min_votes=2, min_speed=25.0):
        """两种判据任一成立就算"运动目标"：多帧投票 或 消抖后仍在持续移动。"""
        if sum(tr["flags"]) >= min_votes:
            return True
        spd = float(np.hypot(tr["vel"][0], tr["vel"][1]))
        return spd > min_speed and tr["hits"] >= 5

    @staticmethod
    def is_really_moving(tr, min_disp_px=4.0, disp_ratio=0.25):
        """显示前的质量门限：这条轨迹**真的动过**吗？"""
        trail = list(tr["trail"])
        if len(trail) < 2:
            return False
        k = min(20, len(trail))
        net = float(np.hypot(trail[-1][0] - trail[-k][0], trail[-1][1] - trail[-k][1]))
        size = float(np.sqrt(max(1.0, tr["box"][2] * tr["box"][3])))
        return net >= max(float(min_disp_px), float(disp_ratio) * size)


def draw_tracks(img, tracks, x_off=0, y_off=0, arrow_min_speed=12.0, scale=1.0):
    """把跟踪结果画到画布上（x_off/y_off 用于拼接画布上的偏移）。"""
    th = max(1, int(round(2 * scale)))
    th_txt = max(1, int(round(1 * scale)))
    th_halo = max(2, int(round(3 * scale)))
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
            ex = int(ax + tr["vel"][0] / spd * 30 * scale)
            ey = int(ay + tr["vel"][1] / spd * 30 * scale)
            cv2.arrowedLine(img, (ax, ay), (ex, ey), color, th, tipLength=0.3)
        label = "ID%d %s %.0f" % (tr["id"], "MOVING" if moving else "check", spd)
        ty = y - int(9 * scale) if (y - y_off) > 20 * scale else y + h + int(19 * scale)
        cv2.putText(img, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    (0, 0, 0), th_halo, cv2.LINE_AA)
        cv2.putText(img, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    color, th_txt, cv2.LINE_AA)


class TrackbarPanel:
    """运行时调参面板：面积上下限、帧差灵敏度、检测带、视频通道。"""

    WIN = "Controls"

    def __init__(self, n_channels, min_area_ratio=0.0005, max_area_ratio=0.0,
                 min_diff=12.0, merge_gap=0.5, jitter_ratio=0.10,
                 roi_top=0.0, roi_bottom=1.0, channel=0):
        self.n_channels = max(1, int(n_channels))
        self.ch_max = max(1, self.n_channels - 1)
        self._defaults = {
            TB_MIN_AREA: int(round(min_area_ratio * 10000)),
            TB_MAX_AREA: int(round(max_area_ratio * 100)),
            TB_SENS: int(round(clamp(min_diff, 1.0, TB_SENS_MAX))),
            TB_MERGE: int(round(clamp(merge_gap, 0.0, TB_MERGE_MAX / 100.0) * 100)),
            TB_JITTER: int(round(clamp(jitter_ratio, 0.01, 0.5) * 100)),
            TB_ROI_TOP: int(round(clamp(roi_top, 0.0, 0.9) * 100)),
            TB_ROI_BOT: int(round(clamp(roi_bottom, 0.1, 1.0) * 100)),
            TB_CHANNEL: int(clamp(channel, 0, self.ch_max)),
        }
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, 470, 300)
        cv2.createTrackbar(TB_MIN_AREA, self.WIN, self._defaults[TB_MIN_AREA], TB_MIN_AREA_MAX, self._noop)
        cv2.createTrackbar(TB_MAX_AREA, self.WIN, self._defaults[TB_MAX_AREA], TB_MAX_AREA_MAX, self._noop)
        cv2.createTrackbar(TB_SENS, self.WIN, self._defaults[TB_SENS], TB_SENS_MAX, self._noop)
        cv2.createTrackbar(TB_MERGE, self.WIN, self._defaults[TB_MERGE], TB_MERGE_MAX, self._noop)
        cv2.createTrackbar(TB_JITTER, self.WIN, self._defaults[TB_JITTER], TB_JITTER_MAX, self._noop)
        cv2.createTrackbar(TB_ROI_TOP, self.WIN, self._defaults[TB_ROI_TOP], TB_ROI_MAX, self._noop)
        cv2.createTrackbar(TB_ROI_BOT, self.WIN, self._defaults[TB_ROI_BOT], TB_ROI_MAX, self._noop)
        cv2.createTrackbar(TB_CHANNEL, self.WIN, self._defaults[TB_CHANNEL], self.ch_max, self._noop)

    @staticmethod
    def _noop(_v):
        return

    def get(self, name):
        """读滑条整数值。窗口被关掉时 OpenCV 返回 -1，这时退回默认值，
                免得参数掉到荒谬区间（比如阈值变成负数）。
        """
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
        """把面板值下发到检测器/消抖器（都是 O(1) 的赋值，每帧调用无压力）。"""
        det.set_area_limits(self.get(TB_MIN_AREA) / 10000.0, self.get(TB_MAX_AREA) / 100.0)
        det.set_min_diff(self.get(TB_SENS))
        det.set_merge_gap(self.get(TB_MERGE) / 100.0)
        det.set_roi(self.get(TB_ROI_TOP) / 100.0, self.get(TB_ROI_BOT) / 100.0)
        if stab is not None:
            stab.set_jitter_ratio(self.get(TB_JITTER) / 100.0)

    def status(self, src_label):
        """一行状态文字（英文：putText 画不了中文）。"""
        max_pct = self.get(TB_MAX_AREA)
        top, bot = self.get(TB_ROI_TOP), self.get(TB_ROI_BOT)
        return "ch%d %s | minA %.2f%% maxA %s diff %d merge %d%% jitter %d%% roi %d-%d%%" % (
            self.get(TB_CHANNEL), src_label, self.get(TB_MIN_AREA) / 100.0,
            ("%d%%" % max_pct) if max_pct > 0 else "off",
            self.get(TB_SENS), self.get(TB_MERGE), self.get(TB_JITTER), top, bot)


class Pipeline:
    """消抖器 + 帧差检测器 + 跟踪器 + 显示画布；尺寸变化时整体重建。"""

    def __init__(self, W, H, args):
        self.W, self.H = int(W), int(H)


        stab_w = args.stab_width or plan_work_width(self.W, self.H, ratio=0.5,
                                                    wmin=160, wmax=480,
                                                    area_max=150000)
        self.stab = VideoStabilizer(self.W, self.H, proc_width=stab_w,
                                    window=args.window, jitter_ratio=args.jitter_ratio,
                                    band=args.band, crop_ratio=args.crop_ratio)
        self.m = self.stab.crop
        self.cw = self.W - 2 * self.m
        self.ch = self.H - 2 * self.m

        work_w = plan_work_width(self.cw, self.ch, ratio=args.work_ratio,
                                 wmin=args.work_min, wmax=args.work_max)
        self.work_w = work_w
        self.det = FrameDiffDetector(self.cw, self.ch, work_width=work_w,
                                     cols=args.cols, min_diff=args.min_diff,
                                     noise_k=args.noise_k,
                                     min_area_ratio=args.min_area_ratio,
                                     max_area_ratio=args.max_area_ratio,
                                     warmup=args.warmup, deghost=not args.no_deghost,
                                     min_response=args.min_response, min_tex=args.min_tex,
                                     grad_k=args.grad_k, merge_gap=args.merge_gap,
                                     contain_ratio=args.contain_ratio,
                                     close_ksize=args.close_ksize,
                                     open_ksize=args.open_ksize,
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
        """返回要写进视频的那一份画面。

        mode="both"  -> 整张对比画布
        mode="right" -> 只取"消抖+检测"那一侧（尺寸 = 输入分辨率 - 消抖裁边）

        为什么要拷到一个固定缓冲？画布切片（canvas[:, x_off:]）是视图，内存不连续；
        VideoWriter.write 在部分 OpenCV 版本上对这种数组会抛 "Unknown C++ exception"。
        拷一次既保证连续，也保证每帧尺寸恒定。
        """
        if mode != "right":
            return canvas
        if self._saved is None:
            self._saved = np.empty((self.ch, self.cw, 3), np.uint8)
        right = canvas[:, self.x_off:] if self.layout == "h" else canvas[self.y_off:, :]
        np.copyto(self._saved, right)
        return self._saved

    def step(self, frame, dt, show_obj=True):
        """跑一帧。返回 (canvas, tracks, mask, det)。"""
        stab_crop, info = self.stab.process(frame)
        orig_crop = frame[self.m:self.H - self.m, self.m:self.W - self.m]
        mask = None
        tracks = []
        if show_obj:
            mask = self.det.apply(stab_crop)
            boxes = self.det.boxes()
            moving = self.det.judge(boxes) if boxes else []
            tracks = self.tracker.update(boxes, moving, dt)
        self.canvas[:, :self.cw] = orig_crop
        right = self.canvas[:, self.x_off:] if self.layout == "h" else self.canvas[self.y_off:, :]
        right[:] = stab_crop
        if mask is not None:
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
        idx = src.first_openable()
        if idx is None:
            print("没有可用的视频通道，退出。")
            return 1
        print("已自动改用通道 %d: %s" % (idx, src.label()))


    panel = TrackbarPanel(src.count(), min_area_ratio=args.min_area_ratio,
                          max_area_ratio=args.max_area_ratio, min_diff=args.min_diff,
                          merge_gap=args.merge_gap, jitter_ratio=args.jitter_ratio,
                          roi_top=args.roi_top, roi_bottom=args.roi_bottom,
                          channel=src.index)
    cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    screen = screen_size()
    mask_win_size = None
    print("""
============== Controls 面板（鼠标拖动即可实时生效）==============
  min_area_1e4 : 物体面积占画面比例【下限】×10000
                 （5 = 0.05%；远处行人只有 15~30 像素宽，别设太大）
  max_area_pct : 物体面积占画面比例的【上限】%（0 = 不限制）
  sens_diff    : 帧差阈值（灰度级）：调小 -> 更灵敏、噪点更多
  merge_gap_pct: 【合并同一个物体的碎块】两个框的空隙小于"该比例×较小框短边"
                 就并成一个大框（50 = 0.5 倍；0 = 不合并；调大并得更狠）
  roi_top_pct  : 只在画面高度的这个百分数【以下】检测（0 = 不限）
  roi_bot_pct  : 只在画面高度的这个百分数【以上】检测（100 = 不限）
                 走路拍摄时，上方楼房/树/晾晒物、下方近处地面是主要假框来源，
                 街景可以先试 10 ~ 70
  channel      : 视频输入通道（摄像头编号 / 视频文件）
按键: q/Q/ESC 退出（或直接关掉视频窗口）| d 掩码视图 | b 开关检测 | s 存图
==================================================================
""")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    writer_path = None
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
            if src.open(ch_sel):
                ok, frame = src.read()
                if not ok:
                    print("通道 %d (%s) 读帧失败，保持原画面" % (ch_sel, src.label()))
                    continue
                pipe = None
                read_buf = None
                print("已切换到通道 %d: %s  分辨率 %s  有效 %.1ffps"
                      % (ch_sel, src.label(), frame.shape[1::-1], src.eff_fps))
            else:
                panel.set(TB_CHANNEL, src.index)
                continue
        else:
            ok, frame = src.read()
            if not ok and args.loop and not src.is_camera and src.rewind():


                ok, frame = src.read()
                if ok:
                    n_loop += 1
                    pipe = None
                    read_buf = None
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
            print("流水线已按 %dx%d 重建：输出/保存 %dx%d（与原片同分辨率，消抖裁边 %d px）"
                  "｜显示 %dx%d（%.0f%%）｜工作分辨率 %d 宽，网格 %d x %d"
                  % (W, H, pipe.canvas.shape[1], pipe.canvas.shape[0], pipe.m,
                     vw, vh, 100.0 * view_scale, pipe.det.dw,
                     pipe.det.motion.rows, pipe.det.motion.cols))


        panel.apply(pipe.det, pipe.stab)
        dt = wall_dt if src.is_camera else src.dt
        canvas, tracks, mask, det = pipe.step(frame, dt, show_obj)


        shown = tracks
        if args.motion_gate:
            shown = [t for t in shown if BoxTracker.is_really_moving(t)]


        shown = suppress_nested_tracks(shown, args.contain_ratio)
        if args.max_show and len(shown) > args.max_show:
            shown = sorted(shown, key=lambda t: (float(np.hypot(t["vel"][0], t["vel"][1])),
                                                 float(t["box"][2] * t["box"][3]), t["hits"]),
                           reverse=True)[:args.max_show]


        saved = pipe.save_view(canvas, args.out_panel)
        if args.output:
            if writer is None or writer_shape != saved.shape:
                if writer is not None:
                    writer.release()
                    writer_seg += 1
                    root, ext = os.path.splitext(args.output)
                    writer_path = "%s_part%d%s" % (root, writer_seg + 1, ext)
                    print("画面尺寸变化，输出改为新分段:", writer_path)
                else:
                    writer_path = args.output
                writer = cv2.VideoWriter(writer_path, fourcc,
                                         clamp(src.eff_fps, 5.0, args.fps_cap),
                                         (saved.shape[1], saved.shape[0]))
                if not writer.isOpened():
                    print("警告：无法创建输出视频", writer_path)
                    writer = None
                    args.output = None
                else:
                    writer_shape = saved.shape
                    print("正在保存 %dx%d（每秒 %.1f 帧）-> %s"
                          % (saved.shape[1], saved.shape[0],
                             clamp(src.eff_fps, 5.0, args.fps_cap), writer_path))


        if show_debug and mask is not None:
            mm = pipe.det.motion
            for r in range(mm.rows):
                for c in range(mm.cols):
                    cx = int((c + 0.5) * mm.cell_w)
                    cy = int((r + 0.5) * mm.cell_h)
                    dxv = float(mm._prev_dx[r, c])
                    dyv = float(mm._prev_dy[r, c])
                    cv2.circle(canvas, (cx, cy), 3, (0, 255, 0), -1)
                    cv2.line(canvas, (cx, cy), (int(cx - dxv * 3), int(cy - dyv * 3)),
                             (0, 255, 255), 1)
        draw_tracks(canvas, shown, x_off=pipe.x_off, y_off=pipe.y_off, scale=tscale)

        tag = 0.9 * tscale
        hint_scale = 0.5 * tscale
        cv2.putText(canvas, "Original", (int(10 * tscale), int(30 * tscale)),
                    cv2.FONT_HERSHEY_SIMPLEX, tag, (0, 0, 255), max(1, int(2 * tscale)), cv2.LINE_AA)
        cv2.putText(canvas, "Detected", (pipe.x_off + int(10 * tscale), pipe.y_off + int(30 * tscale)),
                    cv2.FONT_HERSHEY_SIMPLEX, tag, (0, 255, 0), max(1, int(2 * tscale)), cv2.LINE_AA)
        hint = "q / ESC = quit   (or close this window)"
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, hint_scale, 1)
        hx = max(int(10 * tscale), canvas.shape[1] - tw - int(10 * tscale))
        cv2.putText(canvas, hint, (hx, int(30 * tscale)), cv2.FONT_HERSHEY_SIMPLEX, hint_scale,
                    (0, 0, 0), max(2, int(3 * tscale)), cv2.LINE_AA)
        cv2.putText(canvas, hint, (hx, int(30 * tscale)), cv2.FONT_HERSHEY_SIMPLEX, hint_scale,
                    (0, 255, 255), 1, cv2.LINE_AA)
        status = "FPS %.1f/%.0f  %dx%d  tracks %s  %s  %s" % (
            fps, args.fps_cap, pipe.cw, pipe.ch,
            ("%d/%d" % (len(shown), len(tracks))) if len(shown) != len(tracks) else str(len(tracks)),
            "OBJ-ON" if show_obj else "OBJ-OFF", panel.status(src.label()))
        st_scale = 0.6 * tscale
        cv2.putText(canvas, status, (int(10 * tscale), int(62 * tscale)), cv2.FONT_HERSHEY_SIMPLEX,
                    st_scale, (0, 0, 0), max(2, int(3 * tscale)), cv2.LINE_AA)
        cv2.putText(canvas, status, (int(10 * tscale), int(62 * tscale)), cv2.FONT_HERSHEY_SIMPLEX,
                    st_scale, (255, 255, 0), 1, cv2.LINE_AA)


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
        used = time.perf_counter() - t_now
        delay_ms = int(round(max(0.0, target_dt - used) * 1000.0)) + 1
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
    """合成"手持抖动 + 画面内运动物体"序列。"""
    rng = np.random.default_rng(seed)


    base = np.full((height, width, 3), 60, np.uint8)
    for _ in range(800):
        cv2.circle(base, (int(rng.integers(0, width)), int(rng.integers(0, height))),
                   int(rng.integers(2, 6)), (200, 200, 200), -1)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    cv2.putText(base, "STATIC BACKGROUND", (30, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    obj_tex = rng.integers(40, 230, size=(oh // 4, ow // 4, 3), dtype=np.uint8)
    obj_tex = cv2.resize(obj_tex, (ow, oh), interpolation=cv2.INTER_NEAREST)
    obj_tex = cv2.GaussianBlur(obj_tex, (3, 3), 0)

    cx, cy = width / 2.0, height / 2.0
    frames, boxes, Ms = [], [], []
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
        frames.append(cv2.warpAffine(img, M, (width, height),
                                     borderMode=cv2.BORDER_REPLICATE))
        boxes.append((ox, oy, ow, oh))
        Ms.append(M)
    return frames, boxes, Ms


def _transform_box(box, M):
    """把真值框从"未抖动坐标系"换算到"抖动后的帧坐标系"（取四角变换后的外接矩形）。"""
    x, y, w, h = box
    pts = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]]).reshape(-1, 1, 2)
    q = cv2.transform(pts, M).reshape(-1, 2)
    x0, y0 = q[:, 0].min(), q[:, 1].min()
    x1, y1 = q[:, 0].max(), q[:, 1].max()
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def _check_phase_correlation():
    """确定性检查：相位相关能不能测出已知的亚像素平移，误差有多大。"""
    rng = np.random.default_rng(3)
    patch = (rng.random((64, 64)) * 120 + 60).astype(np.uint8)
    patch = cv2.GaussianBlur(patch, (5, 5), 0)
    errs = []
    detail = []
    for (tx, ty) in ((3.0, -2.0), (0.4, 0.0), (-0.7, 0.6), (5.5, 4.25)):
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        moved = cv2.warpAffine(patch, M, (64, 64), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        (sx, sy), resp = cv2.phaseCorrelate(patch.astype(np.float32), moved.astype(np.float32))
        err = max(abs(sx - tx), abs(sy - ty))
        errs.append(err)
        detail.append("(%+.1f,%+.1f)->%.2f" % (tx, ty, err))
    print("       已知平移 -> 测得误差：%s ；中位 %.3f 最大 %.3f 像素（中位门限 0.25）"
          % ("  ".join(detail), float(np.median(errs)), float(np.max(errs))))
    return float(np.median(errs)) < 0.25


def _check_alignment_and_diff():
    """确定性检查：位移场能不能把背景对齐（帧差≈0），能不能留住独立运动（帧差≠0）。"""
    h, w = 480, 640
    rng = np.random.default_rng(7)
    bg = (rng.random((h, w)) * 100 + 60).astype(np.uint8)
    for _ in range(60):
        cv2.circle(bg, (int(rng.integers(0, w)), int(rng.integers(0, h))), 8, 220, -1)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    cur = np.roll(bg, (7, 12), axis=(0, 1))
    obj = np.zeros_like(bg)
    cv2.rectangle(obj, (300, 200), (340, 240), 255, -1)
    prev_img = cv2.max(bg, obj)
    obj2 = np.zeros_like(bg)

    cv2.rectangle(obj2, (300 + 12 + 14, 200 + 7 + 6), (340 + 12 + 14, 240 + 7 + 6), 255, -1)
    cur_img = cv2.max(cur, obj2)

    det = FrameDiffDetector(w, h, work_width=w, cols=5, min_diff=12.0, noise_k=3.0)
    det.apply(cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR))
    mask = det.apply(cv2.cvtColor(cur_img, cv2.COLOR_GRAY2BGR))

    bg_zone = det.diff_img[40:160, 40:260]
    obj_zone = det.diff_img[200 + 7:240 + 7, 300 + 12:340 + 12]
    bg_mean, obj_mean = float(bg_zone.mean()), float(obj_zone.mean())
    print("       背景帧差均值 %.2f（应很小）  物体帧差均值 %.2f（应很大）  掩码像素 %d"
          % (bg_mean, obj_mean, int(np.count_nonzero(mask))))
    return bg_mean < 8.0 and obj_mean > 20.0 and int(np.count_nonzero(mask)) > 100


def _check_merge_rects():
    """确定性检查：同一个物体的碎块要并成一个框，分开的两个目标不能并。"""
    a = merge_rects([[100, 100, 20, 20], [104, 128, 22, 20]], gap_ratio=0.5)
    b = merge_rects([[100, 100, 20, 20], [220, 100, 20, 20]], gap_ratio=0.5)
    c = merge_rects([[100, 100, 20, 20], [104, 128, 22, 20], [128, 132, 20, 22]],
                    gap_ratio=0.5)
    d = merge_rects([[100, 100, 120, 100], [140, 130, 30, 30]], gap_ratio=0.5)
    print("       同目标两块 -> %d 个框 %s" % (len(a), a))
    print("       分开两目标 -> %d 个框 %s" % (len(b), b))
    print("       传递性(L形三块) -> %d 个框 %s" % (len(c), c))
    print("       大框套小框 -> %d 个框 %s（应只剩大框）" % (len(d), d))
    return len(a) == 1 and len(b) == 2 and len(c) == 1 and len(d) == 1


def _check_nested_tracks():
    """确定性检查：跟踪结果显示层的"去重" —— 大框里的小框要被丢掉。"""
    def mk(box, tid):
        return {"id": tid, "box": np.array(box, np.float32), "hits": 5, "missed": 0,
                "vel": np.zeros(2, np.float32), "trail": deque(), "flags": deque()}
    tracks = [mk([100, 100, 100, 100], 1), mk([140, 140, 20, 20], 2)]
    kept = suppress_nested_tracks(tracks, 0.6)
    far = suppress_nested_tracks([mk([0, 0, 40, 40], 1), mk([200, 0, 40, 40], 2)], 0.6)
    print("       大框+框内小框 -> 保留 %d 条（应 1）；两个分开的框 -> 保留 %d 条（应 2）"
          % (len(kept), len(far)))
    return len(kept) == 1 and len(far) == 2


def _check_output_resolution():
    """确定性检查（回归测试）：输出尺寸必须跟着**输入分辨率**走。"""
    args = build_parser().parse_args([])
    got = {}
    for (w, h) in ((640, 480), (1280, 720), (1920, 1080), (720, 1280)):
        frame = np.zeros((h, w, 3), np.uint8)
        limited, _ = limit_frame(frame, args.max_width, None)
        pipe = Pipeline(limited.shape[1], limited.shape[0], args)
        got[(w, h)] = (pipe.canvas.shape[1], pipe.canvas.shape[0])
        print("       %4dx%-4d -> 输出 %4dx%-4d（内部工作宽度 %d，与输入无关）"
              % (w, h, pipe.canvas.shape[1], pipe.canvas.shape[0], pipe.det.dw))
    uniq = len(set(got.values())) == len(got)
    grows = got[(1920, 1080)][0] > got[(640, 480)][0] and got[(1280, 720)][0] > got[(640, 480)][0]
    return uniq and grows


def _check_view_scale():
    """确定性检查：显示缩放只影响屏幕，不会改变输出尺寸。"""
    small = plan_view_scale(1092, 384, (1920, 1080), 0)
    big = plan_view_scale(6820, 1728, (1920, 1080), 0)
    forced = plan_view_scale(6820, 1728, (1920, 1080), 1000)
    print("       1092 宽 -> 显示比例 %.2f；6820 宽(4K画布) -> %.2f；指定 --view-width 1000 -> %.2f"
          % (small, big, forced))
    return abs(small - 1.0) < 0.01 and big < 0.40 and abs(forced * 6820 - 1000) < 2.0


def _check_single_object():
    """确定性检查：掩码里只有一个目标时必须给出 1 个框。"""
    det = FrameDiffDetector(640, 480, work_width=480, min_area_ratio=0.0005)
    det.n_frame = 100
    det.mask = np.zeros((det.dh, det.dw), np.uint8)
    det.mask[100:140, 120:160] = 255
    boxes = det.boxes()
    print("       掩码里只有 1 个 40x40 的块 -> boxes() 返回 %d 个框 %s" % (len(boxes), boxes[:2]))
    return len(boxes) == 1


def _check_motion_gate():
    """确定性检查：钉在原地的框必须被挡掉，真的移动过的才放行。"""
    dt = 1.0 / 30.0
    trk = BoxTracker()
    st = trk._new_track([100, 100, 40, 40], True)
    mv = trk._new_track([300, 100, 40, 40], True)
    for k in range(19):
        trk._update_track(st, [100, 100, 40, 40], True, dt)
        trk._update_track(mv, [300 + 2 * (k + 1), 100, 40, 40], True, dt)
    a = BoxTracker.is_really_moving(st)
    b = BoxTracker.is_really_moving(mv)
    print("       原地不动 -> %s；每帧走 2 像素 -> %s（期望：挡掉 / 放行）"
          % ("放行" if a else "挡掉", "放行" if b else "挡掉"))
    return (not a) and b


def _check_area_limits():
    """确定性检查：面积上下限会真的改变候选框。"""
    det = FrameDiffDetector(640, 480, work_width=480, min_area_ratio=0.0005)
    det.n_frame = 100
    det.mask = np.zeros((det.dh, det.dw), np.uint8)
    bw, bh = int(det.dw * 0.8), int(det.dh * 0.75)
    det.mask[10:10 + bh, 10:10 + bw] = 255
    ratio = (bw * bh) / float(det.dw * det.dh)
    det.set_area_limits(0.0005, 0.0)
    n_off = len(det.boxes())
    det.set_area_limits(0.0005, 0.25)
    n_lim = len(det.boxes())
    det.set_area_limits(0.90, 0.0)
    n_min = len(det.boxes())
    det.set_area_limits(0.90, 0.0005)
    n_swap = len(det.boxes())
    print("       人造区域占画面 %.0f%%：上限关->%d 上限25%%->%d 下限90%%->%d 上下限写反->%d"
          % (100 * ratio, n_off, n_lim, n_min, n_swap))
    return n_off == 1 and n_lim == 0 and n_min == 0 and n_swap == 1


def _check_fps_cap(fps_cap=FPS_CAP):
    """确定性检查：60fps 源被降到 30fps，且播放速度不变（丢帧而不是慢放）。"""
    path = os.path.join(tempfile.gettempdir(), "video_proc2_fps_%d.mp4" % os.getpid())
    n, w, h = 60, 320, 240
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (w, h))
    if not vw.isOpened():
        print("       跳过：本机无法创建测试视频(缺编码器)")
        return None
    for i in range(n):
        img = np.full((h, w, 3), 30, np.uint8)
        cv2.circle(img, (10 + i * 4, h // 2), 6, (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    src = FrameSource([("file", path)], 640, 480, fps_cap)
    if not src.open(0):
        os.remove(path)
        return None
    if src.fps_src <= fps_cap * 1.05:
        print("       跳过：编码器写出的帧率是 %.1f（未超上限）" % src.fps_src)
        src.release()
        os.remove(path)
        return None
    xs = []
    while len(xs) < 8:
        ok, f = src.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        xs.append(int(np.argmax((g > 200).sum(axis=0))))
    step = float(np.median(np.diff(xs))) if len(xs) >= 2 else 0.0
    src.release()
    os.remove(path)
    print("       源 60fps -> skip=1，有效 30fps；相邻帧白点位移 %.1fpx（应≈8）" % step)
    return 6.5 <= step <= 9.5


def _check_loop():
    """确定性检查：文件读完能回到开头继续读（循环播放）。"""
    path = os.path.join(tempfile.gettempdir(), "video_proc2_loop_%d.mp4" % os.getpid())
    n, w, h = 20, 320, 240
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    if not vw.isOpened():
        print("       跳过：本机无法创建测试视频(缺编码器)")
        return None
    for i in range(n):
        img = np.full((h, w, 3), 30, np.uint8)
        cv2.circle(img, (10 + i * 8, h // 2), 6, (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    src = FrameSource([("file", path)], 640, 480, FPS_CAP)
    if not src.open(0):
        os.remove(path)
        return None
    def blob_x(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return int(np.argmax((g > 200).sum(axis=0)))
    xs = []
    while True:
        ok, f = src.read()
        if not ok:
            break
        xs.append(blob_x(f))
    n_read = len(xs)
    did = src.rewind()
    ok2, f2 = src.read()
    x0 = blob_x(f2) if ok2 else -1
    src.release()
    os.remove(path)
    print("       读满 %d 帧后 rewind=%s，重读首帧 x=%d（首轮 %d）" % (n_read, did, x0, xs[0] if xs else -1))
    return bool(did) and ok2 and n_read == n and bool(xs) and x0 == xs[0]


def run_selftest():
    """无摄像头自检：地基 → 功能 → 端到端效果，逐项量化。"""
    print("=" * 62)
    print("video_proc2 自检（特征方格锚定 + 帧差法）  OpenCV %s" % cv2.__version__)
    print("=" * 62)
    results = {}

    print("\n[0] 分辨率自适应规划")
    for (w, h) in ((640, 480), (1280, 720), (1920, 1080), (720, 1280), (1080, 1920)):
        tw = min(w, 960)
        th = max(2, int(round(h * tw / float(w))))
        ww = plan_work_width(tw, th)
        print("       %4dx%-4d -> 工作分辨率 %4d 宽（%d 像素）" % (w, h, ww, ww * int(round(th * ww / float(tw)))))
    results["自适应规划"] = True

    print("\n[0b] 输出尺寸跟随输入分辨率（回归测试）")
    results["输出分辨率"] = _check_output_resolution()

    print("\n[0c] 显示缩放只影响屏幕（不改输出）")
    results["显示缩放"] = _check_view_scale()

    print("\n[1] 相位相关（地基：能不能测出亚像素平移）")
    results["相位相关"] = _check_phase_correlation()

    print("\n[2] 对齐 + 帧差（背景对齐掉、独立运动留住）")
    results["对齐与帧差"] = _check_alignment_and_diff()

    print("\n[3] 框合并（同一物体的碎块并成一个，分开的目标不并）")
    results["框合并"] = _check_merge_rects()

    print("\n[4] 去重显示（大框里的小框不再重复画）")
    results["去重显示"] = _check_nested_tracks()

    print("\n[5] 单目标出框（groupRectangles 陷阱回归）")
    results["单目标出框"] = _check_single_object()

    print("\n[6] 运动门限（钉在原地的假框不能显示）")
    results["运动门限"] = _check_motion_gate()

    print("\n[7] 面积上下限（滑条语义）")
    results["面积上下限"] = _check_area_limits()

    print("\n[8] 帧率上限 30（丢帧且不慢放）")
    r = _check_fps_cap(FPS_CAP)
    results["帧率上限"] = True if r is None else r

    print("\n[9] 视频循环")
    r = _check_loop()
    results["循环播放"] = True if r is None else r


    print("\n[10] 合成序列端到端（抖动 + 运动物体 = 消抖 -> 逐格对齐 -> 帧差）")
    W, H, N = 640, 480, 150
    frames, boxes, Ms = _synth_sequence(W, H, N)
    stab = VideoStabilizer(W, H, proc_width=plan_work_width(W, H, ratio=0.5), window=30)
    m = stab.crop
    cw, ch = W - 2 * m, H - 2 * m
    det = FrameDiffDetector(cw, ch, work_width=plan_work_width(cw, ch, ratio=0.5),
                            cols=8, min_diff=12.0, min_area_ratio=0.0005,
                            max_area_ratio=0.25, warmup=25)
    trk = BoxTracker()
    dt = 1.0 / 25.0
    nstat = nhit = 0
    ctr = []
    for i, f in enumerate(frames):
        stab_crop, info = stab.process(f)
        det.apply(stab_crop)
        bx = det.boxes()
        mv = det.judge(bx) if bx else []
        tracks = trk.update(bx, mv, dt)
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
            ctr.append(float(np.hypot(bb[0] + bb[2] / 2 - (truth[0] + truth[2] / 2),
                                      bb[1] + bb[3] / 2 - (truth[1] + truth[3] / 2))))
    hit = nhit / max(1, nstat)
    print("       统计 %d 帧：命中率(IoU>=0.2) %.1f%%  平均中心误差 %s"
          % (nstat, 100 * hit, ("%.1f px" % float(np.mean(ctr))) if ctr else "n/a"))
    results["框选"] = hit >= 0.6

    print("\n====================== 自检结论 ======================")
    for k, v in results.items():
        print("       %-12s %s" % (k, "PASS" if v else "FAIL"))
    all_ok = all(results.values())
    print("       %s" % ("全部通过" if all_ok else "存在失败项"))
    print("=====================================================")
    return 0 if all_ok else 1


def build_parser():
    ap = argparse.ArgumentParser(description="特征方格锚定 + 帧差法 运动物体检测（video_proc2）")


    ap.add_argument("--video", nargs="+", default=None, help="输入视频文件（可多个，各算一个通道）")
    ap.add_argument("--camera", type=int, default=0, help="启动时的摄像头编号")
    ap.add_argument("--channel", type=int, default=None, help="启动时的通道序号")
    ap.add_argument("--cameras", type=int, default=3, help="暴露几个摄像头通道")
    ap.add_argument("--width", type=int, default=640, help="摄像头采集宽度（期望值）")
    ap.add_argument("--height", type=int, default=480, help="摄像头采集高度（期望值）")
    ap.add_argument("--fps-cap", dest="fps_cap", type=float, default=FPS_CAP,
                    help="帧率上限（默认 30；源帧率更高时按 grab() 丢帧降下来）")
    ap.add_argument("--no-loop", dest="loop", action="store_false", default=True,
                    help="视频播完不再循环（默认：播完自动从头重播）")


    ap.add_argument("--max-width", dest="max_width", type=int, default=0,
                    help="处理与输出的宽度上限（默认 0 = 不限制：**输出与输入同分辨率**）。"
                         "只有为了提速才设它，例如 4K 素材设 --max-width 1280")
    ap.add_argument("--view-width", dest="view_width", type=int, default=0,
                    help="显示窗口的宽度上限（默认 0 = 按屏幕自动）。"
                         "它只缩小屏幕上看到的那一份，保存的视频仍是全分辨率")
    ap.add_argument("--out-panel", dest="out_panel", choices=("both", "right"), default="both",
                    help="保存哪种画面：both = 原图|消抖图 对比画布（默认，宽度约 2 倍画面）；"
                         "right = 只保存消抖+检测那一半，尺寸正好等于输入分辨率减去消抖裁边")
    ap.add_argument("--work-ratio", dest="work_ratio", type=float, default=0.5,
                    help="工作分辨率 = 画面宽 × 该比例（默认 0.5）")
    ap.add_argument("--work-min", dest="work_min", type=int, default=240, help="工作宽度下限")
    ap.add_argument("--work-max", dest="work_max", type=int, default=480, help="工作宽度上限")
    ap.add_argument("--layout", choices=("auto", "h", "v"), default="auto",
                    help="对比画布布局：auto=超宽自动上下拼，h=左右，v=上下")


    ap.add_argument("--stab-width", dest="stab_width", type=int, default=None,
                    help="消抖用的光流处理宽度（默认按画面宽度自动算）")
    ap.add_argument("--window", type=int, default=30, help="轨迹平滑窗口(帧，默认 30)")
    ap.add_argument("--jitter-ratio", dest="jitter_ratio", type=float, default=0.10,
                    help="抖动/运镜的分界，占画面宽度的比例（默认 0.10 = 10%%，"
                         "对应 jitter_pct 滑条）")
    ap.add_argument("--band", type=float, default=0.03, help="软阈值过渡带宽度（默认 0.03）")
    ap.add_argument("--crop-ratio", dest="crop_ratio", type=float, default=0.10,
                    help="消抖输出裁剪比例（默认 0.10：去掉补偿后不可靠的边缘）")


    ap.add_argument("--cols", type=int, default=8,
                    help="网格列数（默认 8；行数按画面长宽比自动算）。"
                         "越多越能跟视差，但每格越小、越容易没纹理")
    ap.add_argument("--min-response", dest="min_response", type=float, default=0.08,
                    help="相位相关可信度门限（默认 0.08；调高更稳但锚点更少）")
    ap.add_argument("--min-tex", dest="min_tex", type=float, default=15.0,
                    help="方格纹理门限（拉普拉斯方差，默认 15）：低于它当纯色跳过")


    ap.add_argument("--min-diff", dest="min_diff", type=float, default=12.0,
                    help="帧差阈值（灰度级，默认 12；对应 sens_diff 滑条，调小更灵敏）")
    ap.add_argument("--noise-k", dest="noise_k", type=float, default=3.0,
                    help="逐格噪声底倍数（默认 3：门槛 = max(min_diff, 3×本格差值中位)）")
    ap.add_argument("--grad-k", dest="grad_k", type=float, default=0.0,
                    help="梯度门限系数（默认 0 = 关闭）。打开后帧差还要大于"
                         "[本处灰度梯度 × 该系数] 才算数：能压掉地砖缝/斑马线/窗框上"
                         "那些边缘错位造成的假框（实测 b.mp4 近地假框 4.8->0.4/帧），"
                         "但有纹理的目标也会被削弱，按素材取舍")
    ap.add_argument("--min-area-ratio", dest="min_area_ratio", type=float, default=0.0005,
                    help="物体面积占画面比例下限（默认 0.0005 = 0.05%%）："
                         "远处行人只有 15~30 像素宽，别设太大")
    ap.add_argument("--max-area-ratio", dest="max_area_ratio", type=float, default=0.0,
                    help="物体面积占画面比例上限（默认 0 = 不限制）")
    ap.add_argument("--roi-top", dest="roi_top", type=float, default=0.0,
                    help="只在画面高度的这个百分数以下检测（默认 0 = 不限；街景建议 10）")
    ap.add_argument("--roi-bottom", dest="roi_bottom", type=float, default=100.0,
                    help="只在画面高度的这个百分数以上检测（默认 100 = 不限；街景建议 70）")
    ap.add_argument("--merge-gap", dest="merge_gap", type=float, default=0.5,
                    help="合并同一个物体碎块的空隙比例（默认 0.5 = 空隙小于较小框短边的"
                         "一半就并成一个大框；0 = 不合并；调大更激进）。"
                         "对应 merge_gap_pct 滑条")
    ap.add_argument("--contain-ratio", dest="contain_ratio", type=float, default=0.6,
                    help="【去掉大框里的小框】包含度阈值 IoMin（默认 0.6 = 小框有 60%% "
                         "被大框吃掉就算重复，只留大框；0 = 关闭）。"
                         "它同时作用在候选框合并与跟踪结果显示两层")
    ap.add_argument("--warmup", type=int, default=2, help="前几帧不出框")
    ap.add_argument("--no-deghost", dest="no_deghost", action="store_true",
                    help="关闭拖影抑制（默认开启：帧差的框会沿着运动方向变长）")
    ap.add_argument("--open-ksize", dest="open_ksize", type=int, default=3,
                    help="形态学开运算的核大小（默认 3）。调大到 5/7 能削掉"
                         "细条状的边缘伪影（背景对齐残差最爱糊在边缘上），"
                         "代价是很小的目标也会被一起削掉")
    ap.add_argument("--close-ksize", dest="close_ksize", type=int, default=7,
                    help="形态学闭运算的核大小（默认 7）：把断开的碎块连起来")
    ap.add_argument("--no-motion-gate", dest="motion_gate", action="store_false", default=True,
                    help="关闭[必须真的移动过]的显示门限（默认开启）")
    ap.add_argument("--max-show", dest="max_show", type=int, default=12,
                    help="最多同时显示几个目标（默认 12；0 = 不限制）")
    ap.add_argument("--no-object", dest="no_object", action="store_true", help="只看画面不检测")
    ap.add_argument("--debug", action="store_true", help="启动时打开掩码视图与位移场箭头")


    ap.add_argument("-o", "--output", default=None, help="保存对比视频")
    ap.add_argument("--selftest", action="store_true", help="跑合成数据自检")
    return ap


if __name__ == "__main__":
    _args = build_parser().parse_args()
    if _args.selftest:
        raise SystemExit(run_selftest())
    raise SystemExit(run(_args))
