# -*- coding: utf-8 -*-
"""移动摄像头视频消抖 + 运动物体检测框选（video_proc1）

功能
----
1) 消抖：Shi-Tomasi 角点 + 金字塔 LK 光流 + RANSAC 相似变换，
   轨迹滑动窗口平滑后反向补偿；
2) 检测：消抖后画面的 DIS 稠密光流残差减去"背景运动场"，
   经形态学、连通域、面积占比上下限筛选得到候选框，再由框内填充率确认；
3) 跟踪：IoU + 速度外推关联、EMA 平滑、ID 维持、轨迹尾迹；
4) 分辨率自适应：输出与输入同分辨率（消抖会裁掉四周约 10%），
   显示窗口按屏幕自动缩放，内部处理分辨率单独限幅以保证速度；
5) 运行时调参（Controls 面板）、视频输入通道切换、帧率上限 30fps。

运行
----
    py video_proc1.py                    # 摄像头 0
    py video_proc1.py --video in.mp4     # 视频文件（播完自动循环；--no-loop 关闭）
    py video_proc1.py -o out.mp4         # 保存对比视频（尺寸随输入分辨率）
    py video_proc1.py -o out.mp4 --out-panel right   # 只存"消抖+检测"那一半
    py video_proc1.py --max-width 1280   # 可选：给处理与输出封顶（默认 0 = 不限制）
    py video_proc1.py --selftest         # 自检（无需摄像头）

Controls 面板
------------
    min_area_1e4  物体面积占画面比例下限 ×10000（15 = 0.15%）
    max_area_pct  物体面积占画面比例上限 %（0 = 不限制）
    sens_x100     残差灵敏度（相对背景的帧间位移阈值）×100（100 = 1.0 像素）
    jitter_pct    抖动/运镜分界，占画面宽度 %（10 = 10%）
    roi_top_pct   只在画面高度的该百分数以下检测（0 = 不限制）
    roi_bot_pct   只在画面高度的该百分数以上检测（100 = 不限制）
    channel       视频输入通道

按键：q/Q/ESC 退出（或关闭视频窗口）、d 掩码视图、b 开关框选、s 存图、a 自动灵敏度
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
TB_JITTER = "jitter_pct"
TB_SENS = "sens_x100"
TB_ROI_TOP = "roi_top_pct"
TB_ROI_BOT = "roi_bot_pct"
TB_CHANNEL = "channel"

TB_MIN_AREA_MAX = 300
TB_MAX_AREA_MAX = 100
TB_JITTER_MAX = 50
TB_SENS_MAX = 300
TB_ROI_MAX = 100


MAIN_WIN = "Stabilize + Motion Detection"
MASK_WIN = "foreground mask"


def clamp(v, lo, hi):
    """规范上下限
        把 v 夹到 [lo, hi]。写成函数是为了让调用处的意图一眼可见。
    """
    return lo if v < lo else (hi if v > hi else v)


def sim(dx, dy, da):
    """输入旋转角度，平移量，输出位移矩阵
        构造「内容位移」相似变换矩阵（与 estimateAffinePartial2D 的输出同参数化）。
    """
    c, s = np.cos(da), np.sin(da)
    return np.array([[c, -s, dx],
                     [s, c, dy]], np.float32)


def compose(A, B):
    """计算两个2*3矩阵的乘积
        两个 2x3 仿射矩阵复合：A ∘ B（先作用 B，再作用 A）。
    """
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


def box_iou(a, b):
    """两个 [x, y, w, h] 框的 IoU（交并比），取值 0~1。"""
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


def quick_motion(g1, g2, n_feat=300):
    """独立估计两帧之间的相似变换 (dx, dy, da)，仅用于自检/评估。"""
    p = cv2.goodFeaturesToTrack(g1, n_feat, 0.01, 8)
    if p is None or len(p) < 4:
        return None
    q, st, er = cv2.calcOpticalFlowPyrLK(g1, g2, p, None, **LK_PARAMS)

    ok = (st.ravel() == 1) & (er.ravel() < 8.0)
    if int(ok.sum()) < 4:
        return None
    M, _ = cv2.estimateAffinePartial2D(p[ok], q[ok], method=cv2.RANSAC,
                                       ransacReprojThreshold=2.0)
    if M is None:
        return None
    return (float(M[0, 2]), float(M[1, 2]), float(np.arctan2(M[1, 0], M[0, 0])))


def decompose(M):
    """从 2x3 相似变换矩阵中取出 (dx, dy, da)。"""
    return (float(M[0, 2]), float(M[1, 2]), float(np.arctan2(M[1, 0], M[0, 0])))


def plan_widths(W, H, proc_ratio=0.5, proc_min=160, proc_max=480,
                det_ratio=0.75, det_min=240, det_max=640,
                proc_area_max=150000, det_area_max=350000):
    """按画面尺寸算出 (消抖处理宽度, 检测处理宽度)，实现"任意分辨率一套参数"。"""


    area = float(max(1, W * H))


    proc_w = clamp(float(proc_ratio) * W, float(proc_min), float(proc_max))

    proc_w = min(proc_w, W * (float(proc_area_max) / area) ** 0.5)
    proc_w = int(round(proc_w))


    det_w = clamp(float(det_ratio) * W, float(det_min), float(det_max))
    det_w = min(det_w, W * (float(det_area_max) / area) ** 0.5)
    det_w = int(round(det_w))

    det_w = max(det_w, proc_w)

    proc_w = int(clamp(proc_w, 32, W))
    det_w = int(clamp(det_w, 32, W))
    return proc_w, det_w


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
    """把输入画面限制到工作分辨率上限，返回 (可用帧, 复用缓冲)。"""
    if not max_width or frame.shape[1] <= max_width:
        return frame, buf
    tw = int(max_width)
    th = max(2, int(round(frame.shape[0] * tw / float(frame.shape[1]))))
    if buf is None or buf.shape[0] != th or buf.shape[1] != tw or buf.dtype != frame.dtype:
        buf = np.empty((th, tw, 3), frame.dtype)

    cv2.resize(frame, (tw, th), dst=buf, interpolation=cv2.INTER_AREA)
    return buf, buf


def build_sources(args):
    """把命令行参数翻译成"通道列表"。"""
    if args.video:
        return [("file", v) for v in args.video]
    n = max(1, int(args.cameras))
    return [("camera", i) for i in range(n)]


class FrameSource:
    """视频输入「通道」管理：一个通道 = 一路摄像头 或 一个视频文件。"""

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
        """通道总数（给 trackbar 量程用）"""
        return max(1, len(self.sources))

    def label(self, index=None):
        """通道的可读名字，用于控制台打印和画面状态栏"""
        i = self.index if index is None else int(index)
        if 0 <= i < len(self.sources):
            kind, val = self.sources[i]
            if kind == "camera":
                return "camera %d" % val
            return os.path.basename(str(val)) or str(val)
        return "?"

    @property
    def is_camera(self):
        """是不是摄像头。计时方式不同：摄像头用挂钟时间，文件用视频时间戳。"""
        return self.kind == "camera"


    def open(self, index):
        """打开第 index 个通道。成功返回 True，失败返回 False 且**保持原通道不变**。"""
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
        """依次尝试所有通道，返回第一个能打开的通道号（都打不开返回 None）。"""
        for i in range(len(self.sources)):
            if self.open(i):
                return i
        return None


    def read(self):
        """读下一帧，返回 (ok, frame)。"""
        if self.cap is None:
            return False, None
        for _ in range(self.skip):
            if not self.cap.grab():
                return False, None
        ok, frame = self.cap.read()
        return ok, frame

    def rewind(self):
        """回到视频开头，用于"文件播完自动重复播放"。返回 True 表示可以继续读。"""
        if self.cap is None or self.kind != "file":
            return False
        if self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

            if pos != pos or pos < 1.0:
                return True

        return self.open(self.index)

    def release(self):
        """释放当前采集器（可以重复调用，cv2 的 release 是幂等的）"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.index = -1


class VideoStabilizer:
    """视频消抖器：把"手持抖动"从画面运动里剥离出去。"""

    def __init__(self, width, height, proc_width=320, window=30,
                 jitter_ratio=0.10, band=0.03, crop_ratio=0.10,
                 reuse_buffer=True):


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
        self.gate_low = 0.0
        self.gate_high = 1.0
        self.set_jitter_ratio(jitter_ratio, band)


        self.crop = int(crop_ratio * min(self.W, self.H))


        self.reuse_buffer = bool(reuse_buffer)
        self.warp_buf = np.empty((self.H, self.W, 3), np.uint8) if self.reuse_buffer else None


    def set_jitter_ratio(self, jitter_ratio, band=None):
        """运行时设置"抖动/运镜"的分界（trackbar 用）。"""
        self.jitter_ratio = float(jitter_ratio)
        if band is not None:
            self.band = float(band)
        low = (self.jitter_ratio - self.band) * self.W
        high = (self.jitter_ratio + self.band) * self.W


        self.gate_low = max(1.0, low)
        self.gate_high = max(self.gate_low + 1.0, high)


    def _crop(self, frame):
        """裁掉四周 crop 像素，返回中心区域。"""
        m = self.crop
        return frame[m:self.H - m, m:self.W - m]


    def process(self, frame):
        """处理一帧。返回 (消抖并裁剪后的图, info)"""
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
                new_pts = next_pts[ok]
                old_pts = self.prev_pts[ok]
                good_err = err[ok]

                keep = good_err < self.max_err
                new_pts = new_pts[keep]
                old_pts = old_pts[keep]

                if len(new_pts) >= 4:


                    M, inl = cv2.estimateAffinePartial2D(
                        old_pts, new_pts, method=cv2.RANSAC,
                        ransacReprojThreshold=self.ransac_thresh,
                        maxIters=2000, confidence=0.995)

                    if M is not None and inl is not None:
                        inl = inl.ravel().astype(bool)
                        n_in = int(inl.sum())
                        bad = ~inl
                        if bad.any():


                            outliers = (new_pts[bad].reshape(-1, 2)
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
                                     dst=self.warp_buf,
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)


                if outliers is not None:
                    outliers = cv2.transform(
                        outliers.reshape(-1, 1, 2), M_comp).reshape(-1, 2).astype(np.float32)

            info.update(ok=True, dx=dx, dy=dy, da=da)

        info.update(outliers=outliers, inliers=n_in)
        return self._crop(out), info


class MotionDetector:
    """基于稠密光流残差的运动物体检测。"""

    def __init__(self, width, height, detect_width=480, min_area_ratio=0.0005,
                 max_area_ratio=0.0, warmup=2, min_mag=1.0, noise_k=3.0,
                 box_fill_min=0.15, tighten_q=50, tighten_keep=0.25,
                 bg_block_ratio=0.06, bg_stride=4,
                 roi_top=0.0, roi_bottom=1.0):
        self.W, self.H = int(width), int(height)


        self.scale = min(1.0, float(detect_width) / self.W)
        self.inv_scale = 1.0 / self.scale
        self.dw = max(32, int(round(self.W * self.scale)))
        self.dh = max(24, int(round(self.H * self.scale)))


        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)


        self.min_area = 40
        self.min_box = 0.0
        self.max_box = float("inf")
        self.set_area_limits(min_area_ratio, max_area_ratio)


        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        self.warmup = int(warmup)
        self.min_mag = float(min_mag)
        self.noise_k = float(noise_k)
        self.box_fill_min = float(box_fill_min)
        self.tighten_q = float(tighten_q)
        self.tighten_keep = float(tighten_keep)


        self.bg_block_ratio = float(bg_block_ratio)
        self.bg_stride = max(1, int(bg_stride))
        self.bg_block = int(round(self.bg_block_ratio * self.dw)) if self.bg_block_ratio > 0 else 0


        self.roi_top = 0.0
        self.roi_bottom = 1.0
        self.set_roi(roi_top, roi_bottom)

        self.n_frame = 0
        self.prev_gray = None
        self.mask = np.zeros((self.dh, self.dw), np.uint8)
        self.mag = np.zeros((self.dh, self.dw), np.float32)
        self.thresh = 0.0

        self._mag_buf = np.empty((self.dh, self.dw), np.float32)
        self._dx_buf = np.empty((self.dh, self.dw), np.float32)
        self._dy_buf = np.empty((self.dh, self.dw), np.float32)
        self._bgx = np.empty((self.dh, self.dw), np.float32)
        self._bgy = np.empty((self.dh, self.dw), np.float32)

        kk = max(2, self.bg_block // self.bg_stride) if self.bg_block > 0 else 0
        sh = ((self.dh // self.bg_stride) // kk) * kk * self.bg_stride if kk else 1
        sw = ((self.dw // self.bg_stride) // kk) * kk * self.bg_stride if kk else 1
        self._bgsx = np.empty((max(1, sh), max(1, sw)), np.float32)
        self._bgsy = np.empty((max(1, sh), max(1, sw)), np.float32)


    def set_roi(self, top_ratio, bottom_ratio):
        """设置"只在这条水平带里检测"，参数是占画面高度的比例（0~1）。"""
        top = float(clamp(top_ratio, 0.0, 0.9))
        bot = float(clamp(bottom_ratio, 0.1, 1.0))
        if bot - top < 0.05:
            bot = min(1.0, top + 0.05)
        self.roi_top, self.roi_bottom = top, bot


    def set_area_limits(self, min_ratio, max_ratio):
        """运行时设置"物体体积占画面比例"的下限与上限（给 trackbar 用）。"""
        area = float(self.dw * self.dh)
        lo, hi = float(min_ratio), float(max_ratio)
        if hi > 0.0 and hi < lo:
            lo, hi = hi, lo
        self.min_area_ratio, self.max_area_ratio = lo, hi

        self.min_area = max(40, int(lo * area))

        self.min_box = lo * area
        self.max_box = (hi * area) if hi > 0.0 else float("inf")


    def set_min_mag(self, min_mag):
        """运行时设置"残差灵敏度"下限（给 trackbar 用）。"""

        self.min_mag = float(clamp(min_mag, 0.05, 20.0))


    def background_flow(self, fx, fy):
        """估计"背景运动场"，返回 (bx, by)，形状与 fx/fy 相同。"""
        h, w = fx.shape

        k = max(2, self.bg_block // self.bg_stride) if self.bg_block > 0 else 0
        nby = (h // self.bg_stride) // k if k else 0
        nbx = (w // self.bg_stride) // k if k else 0
        if nby < 2 or nbx < 2:


            bx = np.full_like(fx, float(np.median(fx[::2, ::2])))
            by = np.full_like(fy, float(np.median(fy[::2, ::2])))
            return bx, by

        hs = nby * k * self.bg_stride
        ws = nbx * k * self.bg_stride

        def coarse(f):
            """把整幅流场降成 (nby, nbx) 的稀疏背景场，每个格点是该块的中值。"""
            a = f[:hs:self.bg_stride, :ws:self.bg_stride]
            a = a.reshape(nby, k, nbx, k).transpose(0, 2, 1, 3).reshape(nby, nbx, k * k)

            return np.median(a, axis=2).astype(np.float32)

        def expand(f, small, big):
            """稀疏场 -> 全尺寸背景场：按覆盖区域放大 + 右下角边缘复制补齐。"""
            cv2.resize(coarse(f), (ws, hs), dst=small, interpolation=cv2.INTER_LINEAR)
            big[:hs, :ws] = small
            if hs < h:
                big[hs:, :] = big[hs - 1:hs, :]
            if ws < w:
                big[:, ws:] = big[:, ws - 1:ws]
            return big

        expand(fx, self._bgsx, self._bgx)
        expand(fy, self._bgsy, self._bgy)
        return self._bgx, self._bgy


    def apply(self, bgr):
        """输入消抖后的画面，返回前景掩码（0/255）。"""
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

        fx = flow[..., 0]
        fy = flow[..., 1]


        bx, by = self.background_flow(fx, fy)


        cv2.subtract(fx, bx, dst=self._dx_buf)
        cv2.subtract(fy, by, dst=self._dy_buf)


        res = cv2.magnitude(self._dx_buf, self._dy_buf, self._mag_buf)
        self.mag = res


        noise = float(np.median(np.abs(res[::2, ::2])))
        thr = max(self.min_mag, self.noise_k * noise)
        self.thresh = thr


        mask = cv2.compare(res, thr, cv2.CMP_GT)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        self.mask = self._tighten(mask)

        if self.roi_top > 0.0 or self.roi_bottom < 1.0:
            y0 = int(self.roi_top * self.dh)
            y1 = int(self.roi_bottom * self.dh)
            self.mask[:y0, :] = 0
            self.mask[y1:, :] = 0
        return self.mask

    def _tighten(self, mask):
        """块内二次裁剪：每个连通块只保留残差最高的那部分像素（分位数裁剪）。"""
        if self.tighten_q <= 0:
            return mask

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(mask)
        for c in cnts:
            if cv2.contourArea(c) < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)

            sub = self.mag[y:y + h, x:x + w]
            sm = mask[y:y + h, x:x + w] > 0
            if not sm.any():
                continue

            thr = float(np.percentile(sub[sm], self.tighten_q))

            keep = sm & (sub >= thr)
            if keep.sum() >= self.tighten_keep * sm.sum():
                out[y:y + h, x:x + w] = np.where(keep, 255, 0).astype(np.uint8)
            else:
                out[y:y + h, x:x + w] = np.where(sm, 255, 0).astype(np.uint8)
        return out

    def boxes(self):
        """返回工作分辨率（裁剪后坐标系）的候选框 [[x, y, w, h], ...]"""
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

            box_area = float(w * h)
            if box_area < self.min_box or box_area > self.max_box:
                continue
            ar = w / float(h)
            if ar < 0.15 or ar > 8.0:
                continue
            rects.append([int(x), int(y), int(w), int(h)])
        if len(rects) > 1:


            rects, _ = cv2.groupRectangles(rects, 0, 0.3)
            rects = [[int(r[0]), int(r[1]), int(r[2]), int(r[3])] for r in rects]

        inv = self.inv_scale
        return [[int(r[0] * inv), int(r[1] * inv), int(r[2] * inv), int(r[3] * inv)]
                for r in rects]

    def judge(self, boxes, outliers=None, min_pts=1):
        """判断每个候选框是否真的是一个「运动目标」。"""
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

            hard = False
            if outliers is not None and len(outliers):
                m = 0.2 * max(w, h)

                inside = ((outliers[:, 0] >= x - m) & (outliers[:, 0] <= x + w + m) &
                          (outliers[:, 1] >= y - m) & (outliers[:, 1] <= y + h + m))
                hard = int(inside.sum()) >= min_pts

            flags.append(fill >= self.box_fill_min or hard)
        return flags


class BoxTracker:
    """简单的 IoU 关联跟踪：稳定 ID、平滑框、速度方向、轨迹尾迹。"""

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
        """新建一条轨迹（用检测到的框初始化，hits 从 1 开始）"""
        return {"id": self.next_id, "box": np.array(box, np.float32),
                "hits": 1, "missed": 0, "vel": np.zeros(2, np.float32),


                "trail": deque([(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)],
                               maxlen=self.trail_len),
                "flags": deque([bool(moving)], maxlen=10)}

    def _update_track(self, tr, box, moving, dt):
        """用新检测到的框更新已有轨迹"""

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

    def _predict_box(self, tr, dt):
        """按当前速度把框外推 dt 秒，用于关联（常速模型）。"""
        if dt <= 1e-4:
            return tr["box"]
        v = tr["vel"]
        return np.array([tr["box"][0] + v[0] * dt, tr["box"][1] + v[1] * dt,
                         tr["box"][2], tr["box"][3]], np.float32)

    @staticmethod
    def is_really_moving(tr, min_disp_px=6.0, disp_ratio=0.35):
        """这条轨迹到底"动过"没有 —— 显示前的质量门限。"""
        trail = list(tr["trail"])
        if len(trail) < 2:
            return False
        k = min(20, len(trail))
        net = float(np.hypot(trail[-1][0] - trail[-k][0], trail[-1][1] - trail[-k][1]))
        size = float(np.sqrt(max(1.0, tr["box"][2] * tr["box"][3])))
        return net >= max(float(min_disp_px), float(disp_ratio) * size)

    def update(self, boxes, moving, dt):
        """每帧调用一次：把本帧检测到的框关联到已有轨迹上。"""

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
        """两种判据任一成立就算「运动目标」："""
        if sum(tr["flags"]) >= min_votes:
            return True
        spd = float(np.hypot(tr["vel"][0], tr["vel"][1]))

        return spd > min_speed and tr["hits"] >= 5


def draw_tracks(img, tracks, x_off=0, y_off=0, arrow_min_speed=12.0, scale=1.0):
    """把跟踪结果画到图像上（x_off/y_off 用于拼接画布上的偏移）。"""
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


            ax = int(x + w / 2)
            ay = int(y + h / 2)
            ex = int(ax + tr["vel"][0] / spd * 30 * scale)
            ey = int(ay + tr["vel"][1] / spd * 30 * scale)
            cv2.arrowedLine(img, (ax, ay), (ex, ey), color, th, tipLength=0.3)


        label = "ID%d %s %.0f" % (tr["id"], "MOVING" if moving else "check", spd)


        ty = y - int(9 * scale) if (y - y_off) > 20 * scale else y + h + int(19 * scale)


        cv2.putText(img, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, (0, 0, 0), th_halo, cv2.LINE_AA)
        cv2.putText(img, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, color, th_txt, cv2.LINE_AA)


class TrackbarPanel:
    """把"运行时调参"集中到一个 Controls 窗口里的 5 个 trackbar。"""

    WIN = "Controls"

    def __init__(self, n_channels, min_area_ratio=0.0015, max_area_ratio=0.0,
                 jitter_ratio=0.10, min_mag=1.0, roi_top=0.0, roi_bottom=1.0,
                 channel=0):
        self.n_channels = max(1, int(n_channels))


        self.ch_max = max(1, self.n_channels - 1)


        self._defaults = {
            TB_MIN_AREA: int(round(min_area_ratio * 10000)),
            TB_MAX_AREA: int(round(max_area_ratio * 100)),
            TB_JITTER: int(round(jitter_ratio * 100)),
            TB_SENS: int(round(clamp(min_mag, 0.05, TB_SENS_MAX / 100.0) * 100)),
            TB_ROI_TOP: int(round(clamp(roi_top, 0.0, 0.9) * 100)),
            TB_ROI_BOT: int(round(clamp(roi_bottom, 0.1, 1.0) * 100)),
            TB_CHANNEL: int(clamp(channel, 0, self.ch_max)),
        }

        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, 460, 240)


        cv2.createTrackbar(TB_MIN_AREA, self.WIN,
                           self._defaults[TB_MIN_AREA], TB_MIN_AREA_MAX, self._noop)
        cv2.createTrackbar(TB_MAX_AREA, self.WIN,
                           self._defaults[TB_MAX_AREA], TB_MAX_AREA_MAX, self._noop)
        cv2.createTrackbar(TB_JITTER, self.WIN,
                           self._defaults[TB_JITTER], TB_JITTER_MAX, self._noop)
        cv2.createTrackbar(TB_SENS, self.WIN,
                           self._defaults[TB_SENS], TB_SENS_MAX, self._noop)
        cv2.createTrackbar(TB_ROI_TOP, self.WIN,
                           self._defaults[TB_ROI_TOP], TB_ROI_MAX, self._noop)
        cv2.createTrackbar(TB_ROI_BOT, self.WIN,
                           self._defaults[TB_ROI_BOT], TB_ROI_MAX, self._noop)
        cv2.createTrackbar(TB_CHANNEL, self.WIN,
                           self._defaults[TB_CHANNEL], self.ch_max, self._noop)

    @staticmethod
    def _noop(_value):
        """空回调：取值只由主循环轮询读取，这里什么都不做（见类文档说明）。"""
        return


    def get(self, name):
        """读取某个滑条的当前整数值。"""
        try:
            v = int(cv2.getTrackbarPos(name, self.WIN))
        except cv2.error:
            v = -1
        return self._defaults.get(name, 0) if v < 0 else v

    def set(self, name, value):
        """设置滑条位置。只在"切换通道失败、需要把滑条拨回去"时用。"""
        try:
            cv2.setTrackbarPos(name, self.WIN, int(value))
        except cv2.error:
            pass


    def apply(self, stab, det):
        """把面板上的值换算成真实参数，写进消抖器和检测器。"""
        det.set_area_limits(self.get(TB_MIN_AREA) / 10000.0,
                            self.get(TB_MAX_AREA) / 100.0)
        det.set_min_mag(self.get(TB_SENS) / 100.0)
        det.set_roi(self.get(TB_ROI_TOP) / 100.0,
                    self.get(TB_ROI_BOT) / 100.0)
        stab.set_jitter_ratio(self.get(TB_JITTER) / 100.0)

    def status(self, src_label):
        """拼一行状态文字，贴在画面上（英文，因为 putText 画不了中文）。"""
        max_pct = self.get(TB_MAX_AREA)
        top, bot = self.get(TB_ROI_TOP), self.get(TB_ROI_BOT)
        return "ch%d %s | minA %.2f%% maxA %s sens %.2f jitter %d%%%s" % (
            self.get(TB_CHANNEL), src_label,
            self.get(TB_MIN_AREA) / 100.0,
            ("%d%%" % max_pct) if max_pct > 0 else "off",
            self.get(TB_SENS) / 100.0,
            self.get(TB_JITTER),
            (" roi %d-%d%%" % (top, bot)) if (top > 0 or bot < 100) else "")


class Pipeline:
    """一条"尺寸相关"的处理流水线：消抖器 + 检测器 + 跟踪器 + 显示画布。"""

    def __init__(self, W, H, args):
        self.W, self.H = int(W), int(H)


        proc_w, det_w = plan_widths(self.W, self.H,
                                    proc_ratio=args.proc_ratio, proc_min=args.proc_min,
                                    proc_max=args.proc_max, det_ratio=args.det_ratio,
                                    det_min=args.det_min, det_max=args.det_max)
        self.proc_w, self.det_w = proc_w, det_w


        self.stab = VideoStabilizer(self.W, self.H, proc_width=proc_w,
                                    window=args.window, jitter_ratio=args.jitter_ratio,
                                    band=args.band, crop_ratio=args.crop_ratio)
        self.m = self.stab.crop


        self.cw = self.W - 2 * self.m
        self.ch = self.H - 2 * self.m


        self.det = MotionDetector(self.cw, self.ch, detect_width=det_w,
                                  min_area_ratio=args.min_area_ratio,
                                  max_area_ratio=args.max_area_ratio,
                                  min_mag=args.min_mag,
                                  bg_block_ratio=args.bg_block_ratio,
                                  roi_top=args.roi_top / 100.0,
                                  roi_bottom=args.roi_bottom / 100.0,
                                  warmup=args.warmup)
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
        """跑完一帧：消抖 -> 检测 -> 判定 -> 跟踪 -> 拼接。"""

        stab_crop, info = self.stab.process(frame)

        orig_crop = frame[self.m:self.H - self.m, self.m:self.W - self.m]


        tracks = []
        mask = None
        if show_obj:
            mask = self.det.apply(stab_crop)
            boxes = self.det.boxes()
            out_pts = info["outliers"]
            if out_pts is not None and len(out_pts):


                out_pts = out_pts - self.m

            moving = self.det.judge(boxes, out_pts) if boxes else []
            tracks = self.tracker.update(boxes, moving, dt)


        if self.layout == "h":
            self.canvas[:, :self.cw] = orig_crop
            self.canvas[:, self.x_off:] = stab_crop
        else:
            self.canvas[:self.ch, :] = orig_crop
            self.canvas[self.y_off:, :] = stab_crop
        return self.canvas, tracks, mask, info


def screen_size():
    """取屏幕分辨率，用于把显示窗口缩放到屏幕以内。取不到就返回 None。"""
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
    """把窗口调整成"画面能完整放进屏幕"的尺寸（保持宽高比，不放大）。"""
    if not screen:
        return
    sw, sh = int(screen[0] * 0.95), int(screen[1] * 0.90)
    k = min(1.0, sw / float(w), sh / float(h))
    if k < 1.0:
        cv2.resizeWindow(win, max(160, int(w * k)), max(120, int(h * k)))


def plan_view_scale(canvas_w, canvas_h, screen, view_width=0):
    """算出**显示**用的缩放比例（只影响屏幕上的那一份，不改保存的视频）。"""
    if view_width and int(view_width) > 0:
        return min(1.0, float(view_width) / max(1, canvas_w))
    if not screen:
        return min(1.0, 1600.0 / max(1, canvas_w))
    sw, sh = screen[0] * 0.95, screen[1] * 0.90
    return min(1.0, sw / max(1, canvas_w), sh / max(1, canvas_h))


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


    panel = TrackbarPanel(src.count(),
                          min_area_ratio=args.min_area_ratio,
                          max_area_ratio=args.max_area_ratio,
                          jitter_ratio=args.jitter_ratio,
                          min_mag=args.min_mag,
                          roi_top=args.roi_top / 100.0,
                          roi_bottom=args.roi_bottom / 100.0,
                          channel=src.index)


    cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    screen = screen_size()
    mask_win_size = None
    print("""
============== Controls 面板（鼠标拖动即可实时生效）==============
  min_area_1e4 : 要识别物体的体积占画面比例的【下限】 ×10000
                 （5 = 0.05%；调大 -> 少误检，但会漏掉远处小目标。
                   远处行人只有 15~30 像素宽 ≈ 0.03%~0.08%，调大到 15 就全没了）
  max_area_pct : 要识别物体的体积占画面比例的【上限】，百分数
                 （25 = 25%；0 = 不限制；调小 -> 能滤掉糊满画面的假框）
  sens_x100    : 残差灵敏度：一帧内相对背景多走多少像素才算运动
                 （100 = 1.00 像素；【框不出来就往下拉，乱框就往上抬】）
                 实测 b.mp4：100 -> 16% 的帧有目标，70 -> 44%，50 -> 78%
  roi_top_pct  : 只在画面高度的这个百分数【以下】检测（0 = 不限）
  roi_bot_pct  : 只在画面高度的这个百分数【以上】检测（100 = 不限）
                 走路拍摄时上方楼房/树/晾晒物、下方近处地面是主要假框来源，
                 街景可以先试 10 ~ 70
  jitter_pct   : 抖动/运镜的分界，占画面【宽度】的百分数
                 （10 = 10%；调大 -> 更稳但更迟钝，调小 -> 更跟手但更晃）
  channel      : 视频输入通道（摄像头编号 / 视频文件），切换会自动重开设备
视频文件播放到结尾会自动从头重新播放（循环）；不想要循环就在命令行加 --no-loop。
按键: q/ESC 退出 | d 调试视图 | b 开关框选 | s 保存当前帧
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


    auto_sens = args.auto_sens
    auto_val = args.min_mag
    auto_last_set = int(round(auto_val * 100))
    auto_hist = deque(maxlen=max(5, int(args.auto_window)))
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
                print("已切换到通道 %d: %s  分辨率 %s  源帧率 %.1ffps -> 有效 %.1ffps"
                      % (ch_sel, src.label(), frame.shape[1::-1], src.fps_src, src.eff_fps))
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
                  "｜显示 %dx%d（%.0f%%）｜消抖处理宽度 %d，检测处理宽度 %d，%s布局"
                  % (W, H, pipe.canvas.shape[1], pipe.canvas.shape[0], pipe.m,
                     vw, vh, 100.0 * view_scale, pipe.proc_w, pipe.det_w,
                     "左右拼接" if pipe.layout == "h" else "上下拼接"))


        panel.apply(pipe.stab, pipe.det)


        dt = wall_dt if src.is_camera else src.dt
        canvas, tracks, mask, info = pipe.step(frame, dt, show_obj)


        shown = tracks
        if args.motion_gate:
            shown = [t for t in shown if BoxTracker.is_really_moving(t)]
        if args.max_show and len(shown) > args.max_show:
            shown = sorted(shown,
                           key=lambda t: (float(np.hypot(t["vel"][0], t["vel"][1])),
                                          float(t["box"][2] * t["box"][3]),
                                          t["hits"]),
                           reverse=True)[:args.max_show]


        if auto_sens and show_obj:
            cur_slider = panel.get(TB_SENS)
            if cur_slider != auto_last_set:
                auto_sens = False
                print("检测到手动调整灵敏度 -> 自动灵敏度已关闭（按 a 重新打开）")
            else:
                auto_hist.append(len(shown))
                if len(auto_hist) >= auto_hist.maxlen:
                    avg = sum(auto_hist) / float(len(auto_hist))
                    step = 0.0
                    if avg > 3.0 * args.auto_high:
                        step = 0.15
                    elif avg > args.auto_high:
                        step = 0.05
                    elif avg < args.auto_low:
                        step = -0.05
                    if step:
                        auto_val = clamp(auto_val + step, 0.4, 2.5)
                        auto_last_set = int(round(auto_val * 100))
                        panel.set(TB_SENS, auto_last_set)
                        auto_hist.clear()


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

                wr_fps = clamp(src.eff_fps, 5.0, args.fps_cap)
                writer = cv2.VideoWriter(writer_path, fourcc, wr_fps,
                                         (saved.shape[1], saved.shape[0]))
                if not writer.isOpened():
                    print("警告：无法创建输出视频", writer_path)
                    writer = None
                    args.output = None
                else:
                    writer_shape = saved.shape
                    print("正在保存 %dx%d（每秒 %.1f 帧）-> %s"
                          % (saved.shape[1], saved.shape[0], wr_fps, writer_path))


        if show_debug and info["outliers"] is not None:
            for px, py in info["outliers"][:200]:
                cv2.circle(canvas,
                           (int(px - pipe.m) + pipe.x_off, int(py - pipe.m) + pipe.y_off),
                           2, (0, 255, 0), -1)

        draw_tracks(canvas, shown, x_off=pipe.x_off, y_off=pipe.y_off, scale=tscale)


        ttl = 0.9 * tscale
        hint_scale = 0.5 * tscale
        st_scale = 0.6 * tscale
        th2 = max(1, int(round(2 * tscale)))
        th3 = max(2, int(round(3 * tscale)))
        cv2.putText(canvas, "Original", (int(10 * tscale), int(30 * tscale)),
                    cv2.FONT_HERSHEY_SIMPLEX, ttl, (0, 0, 255), th2, cv2.LINE_AA)
        cv2.putText(canvas, "Stabilized",
                    (pipe.x_off + int(10 * tscale), pipe.y_off + int(30 * tscale)),
                    cv2.FONT_HERSHEY_SIMPLEX, ttl, (0, 255, 0), th2, cv2.LINE_AA)

        hint = "q / ESC = quit   (or close this window)"
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, hint_scale, 1)
        hx = max(int(10 * tscale), canvas.shape[1] - tw - int(10 * tscale))
        cv2.putText(canvas, hint, (hx, int(30 * tscale)),
                    cv2.FONT_HERSHEY_SIMPLEX, hint_scale, (0, 0, 0), th3, cv2.LINE_AA)
        cv2.putText(canvas, hint, (hx, int(30 * tscale)),
                    cv2.FONT_HERSHEY_SIMPLEX, hint_scale, (0, 255, 255), 1, cv2.LINE_AA)
        status = "FPS %.1f/%.0f  %dx%d  tracks %s  %s  %s%s" % (
            fps, args.fps_cap, pipe.cw, pipe.ch,
            ("%d/%d" % (len(shown), len(tracks))) if len(shown) != len(tracks) else str(len(tracks)),
            "OBJ-ON" if show_obj else "OBJ-OFF", panel.status(src.label()),
            "  AUTO-SENS" if auto_sens else "")

        cv2.putText(canvas, status, (int(10 * tscale), int(62 * tscale)),
                    cv2.FONT_HERSHEY_SIMPLEX, st_scale, (0, 0, 0), th3, cv2.LINE_AA)
        cv2.putText(canvas, status, (int(10 * tscale), int(62 * tscale)),
                    cv2.FONT_HERSHEY_SIMPLEX, st_scale, (255, 255, 0), 1, cv2.LINE_AA)


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
        elif key == ord('a'):
            auto_sens = not auto_sens
            auto_val = panel.get(TB_SENS) / 100.0
            auto_last_set = panel.get(TB_SENS)
            auto_hist.clear()
            print("自动灵敏度:", "开" if auto_sens else "关（当前值就固定在滑条上）")
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


def _synth_sequence(width=640, height=480, n=150, seed=0, ow=70, oh=70):
    """生成「手持抖动 + 画面内运动物体」的合成序列（用于自检，不需要摄像头）。"""
    rng = np.random.default_rng(seed)


    base = np.full((height, width, 3), 55, np.uint8)
    for _ in range(700):
        cv2.circle(base, (int(rng.integers(0, width)), int(rng.integers(0, height))),
                   int(rng.integers(2, 6)), (205, 205, 205), -1)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    cv2.putText(base, "STATIC BACKGROUND", (30, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cx, cy = width / 2.0, height / 2.0
    frames, boxes = [], []
    for t in range(n):
        ox = int(width * 0.5 + 0.26 * width * np.sin(2 * np.pi * t / 70.0)) - ow // 2
        oy = int(height * 0.5 + 0.16 * height * np.sin(2 * np.pi * t / 45.0)) - oh // 2
        img = base.copy()
        cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (30, 120, 255), -1)
        cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (255, 255, 255), 2)


        jx = 5.0 * np.sin(2 * np.pi * t / 9.0) + 3.0 * np.sin(2 * np.pi * t / 4.0 + 1.0)
        jy = 4.0 * np.sin(2 * np.pi * t / 11.0 + 0.6) + 2.5 * np.sin(2 * np.pi * t / 5.0)
        ja = np.deg2rad(0.8 * np.sin(2 * np.pi * t / 13.0)
                        + 0.4 * np.sin(2 * np.pi * t / 6.5 + 0.3))


        c, s = np.cos(ja), np.sin(ja)
        M = np.float32([[c, -s, cx * (1 - c) + cy * s + jx],
                        [s, c, cy * (1 - c) - cx * s + jy]])


        frames.append(cv2.warpAffine(img, M, (width, height),
                                     borderMode=cv2.BORDER_REPLICATE))
        boxes.append((ox, oy, ow, oh))
    return frames, boxes


def _check_output_resolution():
    """确定性检查（回归测试）：输出（画布/保存视频）尺寸必须跟着输入分辨率走。"""
    args = build_parser().parse_args([])
    got = {}
    for (w, h) in ((640, 480), (1280, 720), (1920, 1080), (720, 1280)):
        frame = np.zeros((h, w, 3), np.uint8)
        limited, _ = limit_frame(frame, args.max_width, None)
        pipe = Pipeline(limited.shape[1], limited.shape[0], args)
        got[(w, h)] = (pipe.canvas.shape[1], pipe.canvas.shape[0])
        print("       %4dx%-4d -> 输出 %4dx%-4d（内部处理宽度 消抖%d/检测%d，与输出无关）"
              % (w, h, pipe.canvas.shape[1], pipe.canvas.shape[0], pipe.proc_w, pipe.det_w))
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
    """确定性检查：**孤立的一个目标必须出框**。"""
    det = MotionDetector(640, 480, detect_width=480, min_area_ratio=0.0015)
    det.n_frame = 100
    det.mask = np.zeros((det.dh, det.dw), np.uint8)

    det.mask[100:140, 120:160] = 255
    boxes = det.boxes()
    print("       掩码里只有 1 个 40x40 的块 -> boxes() 返回 %d 个框 %s"
          % (len(boxes), boxes[:2]))

    ok_pos = bool(boxes) and abs(boxes[0][0] - 120 / det.scale) <= 2
    return len(boxes) == 1 and ok_pos


def _check_background_flow():
    """确定性检查：背景运动场能扣掉"视差"，但不吃掉独立运动的小物体。"""
    h, w = 360, 480
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    bgx = 2.0 + 20.0 * (yy / float(h))
    bgy = 0.5 * (yy / float(h))
    det = MotionDetector(w, h, detect_width=w, bg_block_ratio=0.06, bg_stride=4)

    def residual_with_object(osz):
        """在背景场里放一个 osz x osz 的独立运动块，返回 (背景残差均值, 块内残差均值)"""
        fx, fy = bgx.copy(), bgy.copy()
        ox, oy = 300, 200
        fx[oy:oy + osz, ox:ox + osz] += 4.0
        bx, by = det.background_flow(fx, fy)
        res = np.hypot(fx - bx, fy - by)
        inner = res[oy + 2:oy + osz - 2, ox + 2:ox + osz - 2]
        outside = res.copy()
        outside[oy - 12:oy + osz + 12, ox - 12:ox + osz + 12] = np.nan
        return float(np.nanmean(outside)), float(inner.mean())

    bg_small, obj_small = residual_with_object(14)
    bg_big, obj_big = residual_with_object(90)
    print("       小目标(14x14): 背景残差 %.3f 像素（应≈0）  块内残差 %.3f 像素（应≈4）"
          % (bg_small, obj_small))
    print("       大目标(90x90): 背景残差 %.3f 像素            块内残差 %.3f 像素"
          "（比块大 -> 内部被当背景，只剩轮廓；外框仍能罩住物体）" % (bg_big, obj_big))

    return bg_small < 0.5 and obj_small > 3.0


def _check_motion_gate():
    """确定性检查：钉在原地不动的框必须被挡掉，真的移动过的才放行。"""
    dt = 1.0 / 30.0
    trk = BoxTracker()
    static = trk._new_track([100, 100, 40, 40], True)
    moving = trk._new_track([300, 100, 40, 40], True)
    for k in range(19):
        trk._update_track(static, [100, 100, 40, 40], True, dt)
        trk._update_track(moving, [300 + 2 * (k + 1), 100, 40, 40], True, dt)
    ok_static = BoxTracker.is_really_moving(static)
    ok_moving = BoxTracker.is_really_moving(moving)
    print("       原地不动的框 -> %s；每帧走 2 像素的框 -> %s（期望：挡掉 / 放行）"
          % ("放行" if ok_static else "挡掉", "放行" if ok_moving else "挡掉"))
    return (not ok_static) and ok_moving


def _check_area_limits():
    """确定性检查：面积的上下限滑条真的会改变候选框。"""
    det = MotionDetector(640, 480, detect_width=480,
                         min_area_ratio=0.0015, max_area_ratio=0.0)

    det.n_frame = 100
    det.mask = np.zeros((det.dh, det.dw), np.uint8)
    bw, bh = int(det.dw * 0.8), int(det.dh * 0.75)
    det.mask[10:10 + bh, 10:10 + bw] = 255
    ratio = (bw * bh) / float(det.dw * det.dh)

    det.set_area_limits(0.0015, 0.0)
    n_off = len(det.boxes())
    det.set_area_limits(0.0015, 0.25)
    n_lim = len(det.boxes())
    det.set_area_limits(0.90, 0.0)
    n_min = len(det.boxes())


    det.set_area_limits(0.90, 0.001)
    n_swap = len(det.boxes())

    print("       人造区域占画面 %.0f%%" % (100 * ratio))
    print("       上限关闭 -> %d 个框 | 上限25%% -> %d 个框 | 下限90%% -> %d 个框"
          " | 上下限写反 -> %d 个框" % (n_off, n_lim, n_min, n_swap))
    return (n_off == 1) and (n_lim == 0) and (n_min == 0) and (n_swap == 1)


def _check_fps_cap(fps_cap=FPS_CAP):
    """确定性检查：60fps 的视频源会被降到 30fps 且播放速度不变。"""
    path = os.path.join(tempfile.gettempdir(), "video_proc1_fpscap_%d.mp4" % os.getpid())
    n, w, h = 60, 320, 240
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (w, h))
    if not vw.isOpened():
        print("       跳过：本机 OpenCV 无法创建测试视频(缺编码器)")
        return None
    for i in range(n):
        img = np.full((h, w, 3), 30, np.uint8)
        cv2.circle(img, (10 + i * 4, h // 2), 6, (255, 255, 255), -1)
        vw.write(img)
    vw.release()

    src = FrameSource([("file", path)], 640, 480, fps_cap)
    ok = src.open(0)
    if not ok:
        print("       跳过：测试视频打不开")
        return None
    if src.fps_src <= fps_cap * 1.05:
        print("       跳过：编码器写出的帧率是 %.1f（未超过上限）" % src.fps_src)
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
    n_read = len(xs)
    src.release()
    os.remove(path)

    step = float(np.median(np.diff(xs))) if n_read >= 2 else 0.0
    dt = src.dt
    print("       源帧率 %.1ffps -> skip=%d，有效 %.1ffps (dt=%.4fs)"
          % (src.fps_src, src.skip, src.eff_fps, dt))
    print("       读 %d 帧(原视频 %d 帧)，相邻帧白点位移中位数 %.1fpx（应为 ~8px）"
          % (n_read, n, step))
    ok_step = 6.5 <= step <= 9.5
    ok_dt = dt <= 1.0 / fps_cap * 1.1
    ok_skip = src.skip == 1
    return ok_skip and ok_step and ok_dt


def _check_loop():
    """确定性检查：文件读到结尾后能回到开头继续读（重复播放）。"""
    path = os.path.join(tempfile.gettempdir(), "video_proc1_loop_%d.mp4" % os.getpid())
    n, w, h = 20, 320, 240
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    if not vw.isOpened():
        print("       跳过：本机 OpenCV 无法创建测试视频(缺编码器)")
        return None
    for i in range(n):
        img = np.full((h, w, 3), 30, np.uint8)
        cv2.circle(img, (10 + i * 8, h // 2), 6, (255, 255, 255), -1)
        vw.write(img)
    vw.release()

    def blob_x(bgr):
        """白点在某一列最亮：按列统计亮像素数，取最大的那列 = 白点圆心 x"""
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return int(np.argmax((g > 200).sum(axis=0)))

    src = FrameSource([("file", path)], 640, 480, FPS_CAP)
    if not src.open(0):
        print("       跳过：测试视频打不开")
        os.remove(path)
        return None
    xs = []
    while True:
        ok, f = src.read()
        if not ok:
            break
        xs.append(blob_x(f))
    n_read = len(xs)
    did_rewind = src.rewind()
    ok2, f2 = src.read()
    x0 = blob_x(f2) if ok2 else -1
    src.release()
    os.remove(path)

    print("       读满 %d 帧(源 %d 帧)后 rewind=%s，重读首帧白点 x=%d（首轮首帧 x=%d）"
          % (n_read, n, did_rewind, x0, xs[0] if xs else -1))
    return bool(did_rewind) and ok2 and n_read == n and bool(xs) and x0 == xs[0]


def run_selftest():
    """无摄像头自检：新功能 + 算法效果一起量化。"""
    print("=" * 62)
    print("video_proc1 自检（无需摄像头）  OpenCV %s" % cv2.__version__)
    print("=" * 62)
    results = {}


    print("\n[0] 分辨率 / 帧率自适应规划")
    print("       画面尺寸        工作分辨率   消抖宽度  检测宽度")
    for (w, h) in ((640, 480), (854, 480), (1280, 720), (1920, 1080),
                   (3840, 2160), (1080, 1920)):


        tw = w
        th = h
        pw, dw_ = plan_widths(tw, th)
        print("       %4dx%-4d  ->  %4dx%-4d   %4d      %4d"
              % (w, h, tw, th, pw, dw_))
    for f in (24.0, 30.0, 60.0, 120.0):
        sk, eff = plan_source_fps(f, FPS_CAP)
        print("       帧率 %6.1ffps -> skip=%d，有效 %5.1ffps" % (f, sk, eff))
    results["自适应规划"] = True


    print("\n[0b] 输出尺寸跟随输入分辨率（回归测试）")
    results["输出分辨率"] = _check_output_resolution()

    print("\n[0c] 显示缩放只影响屏幕（不改输出）")
    results["显示缩放"] = _check_view_scale()


    print("\n[1] 单目标出框（原版 groupRectangles 会丢弃孤立目标的回归测试）")
    results["单目标出框"] = _check_single_object()


    print("\n[2] 背景运动场（扣掉视差，但保留独立运动）")
    results["背景运动场"] = _check_background_flow()


    print("\n[3] 运动门限（消抖残留噪声是钉在原地的，不能当目标显示）")
    results["运动门限"] = _check_motion_gate()


    print("\n[4] 面积占比上下限（trackbar 的语义检查）")
    results["面积上下限"] = _check_area_limits()


    print("\n[5] 帧率上限 %.0ffps（丢帧后播放速度是否保持）" % FPS_CAP)
    r2 = _check_fps_cap(FPS_CAP)
    results["帧率上限"] = True if r2 is None else r2


    print("\n[6] 视频循环（读到结尾后回到开头继续读）")
    r3 = _check_loop()
    results["循环播放"] = True if r3 is None else r3


    W, H, N = 640, 480, 150
    print("\n[7] 生成合成测试序列（手持抖动 + 运动物体）...")
    frames, true_boxes = _synth_sequence(W, H, N)


    proc_w, det_w = plan_widths(W, H)
    stab = VideoStabilizer(W, H, proc_width=proc_w, window=30)
    m = stab.crop
    cw, ch = W - 2 * m, H - 2 * m
    det = MotionDetector(cw, ch, detect_width=det_w, min_area_ratio=0.0008,
                         max_area_ratio=0.25, warmup=30)
    tracker = BoxTracker()

    raw_motions, stab_motions = [], []
    stab_prev = None
    raw_prev = None
    n_stat, n_hit, n_miss = 0, 0, 0
    n_flow_vote = 0
    ctr_err = []
    dt = 1.0 / 25.0

    for i, f in enumerate(frames):
        s, info = stab.process(f)


        g_raw = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g_stab = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        if raw_prev is not None:
            a = quick_motion(raw_prev, g_raw)
            b = quick_motion(stab_prev, g_stab)
            if a:
                raw_motions.append(a)
            if b:
                stab_motions.append(b)
        raw_prev, stab_prev = g_raw, g_stab


        det.apply(s)
        boxes = det.boxes()
        out_pts = info["outliers"]
        if out_pts is not None and len(out_pts):
            out_pts = out_pts - m
        moving = det.judge(boxes, out_pts) if boxes else []
        tracks = tracker.update(boxes, moving, dt)

        if i <= det.warmup:
            continue
        n_stat += 1
        if any(moving):
            n_flow_vote += 1
        tx, ty, tw_, th_ = true_boxes[i]


        truth = (tx - m, ty - m, tw_, th_)
        best, best_box = 0.0, None
        for tr in tracks:
            v = box_iou(tr["box"], truth)
            if v > best:
                best, best_box = v, tr["box"]
        if best >= 0.2:
            n_hit += 1
            ctr_err.append(float(np.hypot(best_box[0] + best_box[2] / 2 - (truth[0] + tw_ / 2),
                                          best_box[1] + best_box[3] / 2 - (truth[1] + th_ / 2))))
        else:
            n_miss += 1

    def rms(vals):
        """均方根：先平方、再求平均、最后开方（数值稳定性好，对大值更敏感）"""
        return float(np.sqrt(np.mean(np.asarray(vals) ** 2))) if len(vals) else 0.0

    raw_t = rms([np.hypot(a[0], a[1]) for a in raw_motions])
    stab_t = rms([np.hypot(a[0], a[1]) for a in stab_motions])
    raw_r = rms([a[2] for a in raw_motions])
    stab_r = rms([a[2] for a in stab_motions])

    red_t = 100 * (1 - stab_t / raw_t) if raw_t > 1e-6 else 0.0
    red_r = 100 * (1 - stab_r / raw_r) if raw_r > 1e-6 else 0.0
    hit_rate = n_hit / max(1, n_stat)

    print("       [消抖] 帧间残余运动（越小越稳）")
    print("              原始画面 RMS: 平移 %6.2f px   旋转 %5.3f 度"
          % (raw_t, np.degrees(raw_r)))
    print("              消抖画面 RMS: 平移 %6.2f px   旋转 %5.3f 度"
          % (stab_t, np.degrees(stab_r)))
    print("              抖动抑制: 平移 %5.1f%%   旋转 %5.1f%%" % (red_t, red_r))
    print("       [检测] 统计帧数 %d" % n_stat)
    print("              命中率(与真值框 IoU>=0.2): %.1f%%" % (100 * hit_rate))
    print("              漏检率: %.1f%%   平均框中心误差: %s"
          % (100 * n_miss / max(1, n_stat),
             ("%.1f px" % float(np.mean(ctr_err))) if ctr_err else "n/a"))
    print("              运动判定为真的帧占比: %.1f%%"
          % (100.0 * n_flow_vote / max(1, n_stat)))


    ok1 = (stab_t < 0.4 * raw_t) and (stab_r < 0.5 * raw_r)
    ok2 = hit_rate >= 0.6
    results["消抖"] = ok1
    results["框选"] = ok2

    print("\n====================== 自检结论 ======================")
    for k, v in results.items():
        print("       %-12s %s" % (k, "PASS" if v else "FAIL"))
    all_ok = all(results.values())
    print("       %s" % ("全部通过" if all_ok else "存在失败项"))
    print("=====================================================")
    return 0 if all_ok else 1


def build_parser():
    """构造命令行参数解析器。"""
    ap = argparse.ArgumentParser(description="移动摄像头消抖 + 运动物体检测框选（video_proc1）")


    ap.add_argument("--video", nargs="+", default=None,
                    help="输入视频文件，可以给多个（每个文件一个通道，运行时用 channel 切换）")
    ap.add_argument("--camera", type=int, default=0,
                    help="启动时选中的摄像头编号（也就是通道号，默认 0）")
    ap.add_argument("--channel", type=int, default=None,
                    help="启动时选中的视频通道序号（默认：有 --video 时为 0，否则等于 --camera）")
    ap.add_argument("--cameras", type=int, default=3,
                    help="暴露多少个摄像头通道（默认 3，即 0/1/2 都能在面板里切）")
    ap.add_argument("--width", type=int, default=640, help="摄像头采集宽度（期望值）")
    ap.add_argument("--height", type=int, default=480, help="摄像头采集高度（期望值）")
    ap.add_argument("--fps-cap", dest="fps_cap", type=float, default=FPS_CAP,
                    help="帧率上限（默认 30；源帧率更高时按 grab() 丢帧降下来）")


    ap.add_argument("--max-width", dest="max_width", type=int, default=0,
                    help="处理与输出的宽度上限（默认 0 = 不限制：输出与输入同分辨率）。"
                         "只有为了提速才设它，例如 4K 素材设 --max-width 1280")
    ap.add_argument("--view-width", dest="view_width", type=int, default=0,
                    help="显示窗口的宽度上限（默认 0 = 按屏幕自动）。"
                         "它只缩小屏幕上看到的那一份，保存的视频仍是全分辨率")
    ap.add_argument("--out-panel", dest="out_panel", choices=("both", "right"), default="both",
                    help="保存哪种画面：both = 原图|消抖图 对比画布（默认，宽度约 2 倍画面）；"
                         "right = 只保存消抖+检测那一半，尺寸正好等于输入分辨率减去消抖裁边")
    ap.add_argument("--proc-ratio", dest="proc_ratio", type=float, default=0.5,
                    help="消抖光流宽度 = 画面宽度 × 该比例（默认 0.5）")
    ap.add_argument("--proc-min", dest="proc_min", type=int, default=160,
                    help="消抖光流宽度下限（默认 160）")
    ap.add_argument("--proc-max", dest="proc_max", type=int, default=480,
                    help="消抖光流宽度上限（默认 480）")
    ap.add_argument("--det-ratio", dest="det_ratio", type=float, default=0.75,
                    help="检测光流宽度 = 画面宽度 × 该比例（默认 0.75）")
    ap.add_argument("--det-min", dest="det_min", type=int, default=240,
                    help="检测光流宽度下限（默认 240）")
    ap.add_argument("--det-max", dest="det_max", type=int, default=640,
                    help="检测光流宽度上限（默认 640）")
    ap.add_argument("--layout", choices=("auto", "h", "v"), default="auto",
                    help="对比画布布局：auto=超宽画幅自动上下拼，h=左右，v=上下")


    ap.add_argument("--window", type=int, default=30, help="轨迹平滑窗口(帧)")
    ap.add_argument("--jitter-ratio", dest="jitter_ratio", type=float, default=0.10,
                    help="抖动判定界限，占画面宽度的比例（默认 0.10 = 10%%，对应 jitter_pct 滑条）")
    ap.add_argument("--band", type=float, default=0.03,
                    help="软阈值过渡带宽度（默认 0.03，即 ±3%%）")
    ap.add_argument("--crop-ratio", dest="crop_ratio", type=float, default=0.10,
                    help="输出裁剪比例（默认 0.10）")


    ap.add_argument("--min-area-ratio", dest="min_area_ratio", type=float, default=0.0005,
                    help="物体面积占画面比例下限（默认 0.0005 = 0.05%%，对应 min_area_1e4 滑条）。"
                         "远处行人只有 15~30 像素宽 ≈ 画面的 0.03%%~0.08%%，"
                         "这个值调大到 0.15%% 就会把他们全部滤掉")
    ap.add_argument("--max-area-ratio", dest="max_area_ratio", type=float, default=0.0,
                    help="物体面积占画面比例上限（默认 0 = 不限制，对应 max_area_pct 滑条）")
    ap.add_argument("--min-mag", dest="min_mag", type=float, default=1.0,
                    help="残差灵敏度：一帧内相对背景多走多少像素才算运动"
                         "（默认 1.0 = 干净优先；越小越灵敏、假框也越多，"
                         "对应 sens_x100 滑条）")
    ap.add_argument("--roi-top", dest="roi_top", type=float, default=0.0,
                    help="只在画面高度的这个百分数以下检测（默认 0 = 不限制）。"
                         "走路拍摄时上方楼房/树/晾晒物随风摆动是主要假框来源，"
                         "街景建议 10；对应 roi_top_pct 滑条")
    ap.add_argument("--roi-bottom", dest="roi_bottom", type=float, default=100.0,
                    help="只在画面高度的这个百分数以上检测（默认 100 = 不限制）。"
                         "下方近处地面视差最大、假框多，街景建议 70；对应 roi_bot_pct 滑条")
    ap.add_argument("--no-motion-gate", dest="motion_gate", action="store_false",
                    default=True,
                    help="关闭[必须真的移动过]这一显示门限（默认开启）。"
                         "关掉后连静止的残留噪声框也会显示出来，一般只在排查漏检时用")
    ap.add_argument("--auto-sens", dest="auto_sens", action="store_true", default=False,
                    help="打开自动灵敏度（默认关）。自动档盯住屏幕上的目标个数"
                         "自动微调 sens_x100；注意安静画面里它会把灵敏度压低、"
                         "从而带来更多假框，所以默认不用。按 a 键可随时开关")
    ap.add_argument("--auto-low", dest="auto_low", type=float, default=0.3,
                    help="自动灵敏度的目标下界（个/帧，默认 0.3；低于它就降低灵敏度）")
    ap.add_argument("--auto-high", dest="auto_high", type=float, default=3.0,
                    help="自动灵敏度的目标上界（个/帧，默认 3.0；高于它就抬高灵敏度）")
    ap.add_argument("--auto-window", dest="auto_window", type=int, default=20,
                    help="自动灵敏度的评估窗口(帧，默认 20)：每积累这么多帧才动一小步")
    ap.add_argument("--max-show", dest="max_show", type=int, default=12,
                    help="画面上最多同时显示几个目标（默认 12；0 = 不限制）。"
                         "镜头贴着玻璃/墙面走时会瞬间冒出几十个候选，"
                         "不限量的话屏幕会被框糊满；跟踪器内部仍维护全部目标")
    ap.add_argument("--bg-block-ratio", dest="bg_block_ratio", type=float, default=0.06,
                    help="背景运动场（视差补偿）的方块边长占画面宽度的比例"
                         "（默认 0.06；传 0 = 关闭，退回只减全场中值的旧行为）")
    ap.add_argument("--warmup", type=int, default=2, help="前几帧不出框(等光流稳定)")
    ap.add_argument("--no-loop", dest="loop", action="store_false", default=True,
                    help="视频文件播完不再重复播放（默认：播完自动从头循环）")
    ap.add_argument("--no-object", dest="no_object", action="store_true",
                    help="关闭运动物体检测，只看消抖效果")


    ap.add_argument("--debug", action="store_true", help="启动时打开调试视图")
    ap.add_argument("-o", "--output", default=None, help="保存对比视频到文件")
    ap.add_argument("--selftest", action="store_true", help="跑合成数据自检(含新功能检查)")
    return ap


if __name__ == "__main__":


    _args = build_parser().parse_args()
    if _args.selftest:


        raise SystemExit(run_selftest())
    raise SystemExit(run(_args))
