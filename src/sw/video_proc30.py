"""
目标跟随（丢失后颜色比例特征重捕获 + 点击选物）video_proc30
阶段 3/4（源自 attempt10）：注释瘦身 + NCC 搜索降 1/2 采样（_match 1.00ms->0.16ms）
"""

import argparse
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

FPS_CAP = 60.0
ELEM = cv2.getStructuringElement

def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return 0.0 if inter <= 0 else inter / (aw * ah + bw * bh - inter)

# 一、视频输入

class FrameSource:
    """
    单一视频输入：打开 / 读帧(按上限丢帧) / 回到开头 / 释放。
    """

    def __init__(self, source, is_camera=False, width=640, height=480, fps_cap=FPS_CAP):
        self.source = source
        self.is_camera = bool(is_camera)
        self.width, self.height = int(width), int(height)
        self.fps_cap = float(fps_cap)
        self.cap = None
        self.skip = 0
        self.fps_src = self.eff_fps = float(fps_cap)
        self.dt = 1.0 / float(fps_cap)
        self.label = ("camera %s" % source) if is_camera else os.path.basename(str(source))

    def open(self):
        if self.is_camera:
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(self.source), backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps_cap)
        else:
            cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            cap.release()
            print("无法打开: %s" % self.label)
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.release()
        self.cap = cap
        f = cap.get(cv2.CAP_PROP_FPS)
        self.fps_src = float(f) if f else 0.0
        if 1e-3 < self.fps_src < 240.0 and self.fps_src > self.fps_cap * 1.05:
            self.skip = int(clamp(int(round(self.fps_src / self.fps_cap)) - 1, 0, 8))
            self.eff_fps = self.fps_src / (self.skip + 1)
        else:
            self.skip = 0
            self.eff_fps = self.fps_src if self.fps_src > 1e-3 else self.fps_cap
        self.dt = 1.0 / self.eff_fps
        return True

    def read(self):
        if self.cap is None:
            return False, None
        for _ in range(self.skip):          # grab() 只挪指针不解码，用来丢帧限速
            if not self.cap.grab():
                return False, None
        return self.cap.read()

    def rewind(self):
        if self.cap is None or self.is_camera:
            return False
        if self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if pos != pos or pos < 1.0:
                return True
        return self.open()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

class FollowController:
    """速度式控制 + 速度前馈。角速度按 lp_tau 低通、限幅、限加速度，再积分成角度。
    """

    def __init__(self, hfov=60.0, vfov=36.0, kp=2.5, kd=0.15, kff=1.0, deadband=0.03,
                 max_rate=60.0, max_acc=400.0, lp_tau=0.12, lead=0.12, ang_limit=120.0,
                 ff_limit=0.5):
        self.hfov, self.vfov = float(hfov), float(vfov)
        self.kp, self.kd, self.kff = float(kp), float(kd), float(kff)
        self.deadband = float(deadband)
        self.max_rate, self.max_acc = float(max_rate), float(max_acc)
        self.lp_tau, self.lead = float(lp_tau), float(lead)
        self.ang_limit = float(ang_limit)
        self.ff_limit = float(ff_limit)
        self.pan = self.tilt = 0.0
        self.rate = np.zeros(2)
        self.cam_rate = np.zeros(2)
        self.info = {"ex": 0.0, "ey": 0.0, "wx": 0.0, "wy": 0.0, "pan": 0.0,
                     "tilt": 0.0, "dead": True, "hold": False}

    def reset(self):
        self.rate[:] = 0.0

    def update(self, e, e_vel, dt, hold=False, cam_rate=None):
        ex, ey = float(e[0]), float(e[1])
        vx, vy = float(e_vel[0]), float(e_vel[1])
        ex_l = ex + vx * self.lead
        ey_l = ey + vy * self.lead
        if abs(ex) < self.deadband:          # 死区：目标在中央小幅游走时不动
            ex_l = 0.0
        if abs(ey) < self.deadband:
            ey_l = 0.0
        wx = (self.kp * ex_l + self.kd * vx) * self.hfov
        wy = -(self.kp * ey_l + self.kd * vy) * self.vfov
        if cam_rate is not None:
            cr = np.asarray(cam_rate, np.float64)
            self.cam_rate = self.cam_rate + 0.5 * (cr - self.cam_rate)
        if self.kff > 0.0 and cam_rate is not None:
            # 目标"在世界里"的角速度估计 = 实测云台角速度 + 目标在画面里的角速度
            ox = self.cam_rate[0] + vx * self.hfov
            oy = self.cam_rate[1] - vy * self.vfov
            # **可信度闸门**：估计值超出"我们可能跟上"的范围时，说明它不是目标的真实运动 （多半是我们测不到的相机/平台运动，或坏的速度估计）。此时前馈会把云台 一路推到底 —— 实测视频在环里出现过
            lim = self.ff_limit * self.max_rate
            if abs(ox) > lim or abs(oy) > lim:
                ox = oy = 0.0
            wx += self.kff * ox
            wy += self.kff * oy
        if hold:
            wx = wy = 0.0
        wx = clamp(wx, -self.max_rate, self.max_rate)
        wy = clamp(wy, -self.max_rate, self.max_rate)
        a = dt / (self.lp_tau + dt) if self.lp_tau > 0 else 1.0
        dw = self.max_acc * dt
        for i, w in ((0, wx), (1, wy)):
            want = self.rate[i] + a * (w - self.rate[i])
            self.rate[i] = clamp(want, self.rate[i] - dw, self.rate[i] + dw)
            self.rate[i] = clamp(self.rate[i], -self.max_rate, self.max_rate)
        self.pan = clamp(self.pan + self.rate[0] * dt, -self.ang_limit, self.ang_limit)
        self.tilt = clamp(self.tilt + self.rate[1] * dt, -self.ang_limit, self.ang_limit)
        self.info = {"ex": ex, "ey": ey, "wx": self.rate[0], "wy": self.rate[1],
                     "pan": self.pan, "tilt": self.tilt,
                     "dead": abs(ex) < self.deadband and abs(ey) < self.deadband,
                     "hold": bool(hold)}
        return self.pan, self.tilt, self.info

    def bearing_of(self, e):
        """
        由"当前角度 + 目标在画面里的偏移"算出目标的世界方位 (pan, tilt)。
        """
        return (self.pan + float(e[0]) * self.hfov,
                self.tilt - float(e[1]) * self.vfov)

    def slew_to(self, pan_t, tilt_t, dt, rate=40.0, tol=0.5):
        """限速平滑转到目标角度（回正用；不是瞬间跳变，避免舵机猛甩）"""
        for i, (cur, tgt) in enumerate(((self.pan, pan_t), (self.tilt, tilt_t))):
            d = tgt - cur
            if abs(d) < tol:
                self.rate[i] = 0.0
                continue
            want = math.copysign(min(rate, abs(d) / max(1e-3, dt)), d)
            self.rate[i] = clamp(want, self.rate[i] - self.max_acc * dt,
                                 self.rate[i] + self.max_acc * dt)
            self.rate[i] = clamp(self.rate[i], -self.max_rate, self.max_rate)
        self.pan = clamp(self.pan + self.rate[0] * dt, -self.ang_limit, self.ang_limit)
        self.tilt = clamp(self.tilt + self.rate[1] * dt, -self.ang_limit, self.ang_limit)
        self.info = {"ex": 0.0, "ey": 0.0, "wx": self.rate[0], "wy": self.rate[1],
                     "pan": self.pan, "tilt": self.tilt, "dead": False, "hold": False}
        return self.pan, self.tilt, self.info

# 四、云台接口：虚拟（仿真）/ 串口

class CameraLink:
    def command(self, pan_deg, tilt_deg, dt):
        raise NotImplementedError

    def state(self):
        return {"pan": 0.0, "tilt": 0.0, "ok": True}

    def close(self):
        pass

class SimCamera(CameraLink):
    """虚拟云台：模拟真实执行器的四件事 —— 延迟 / 速率限幅 / 齿轮回差 / 死区-量化-抖动。

    不建这些，仿真里调出来的增益一上真机就振荡。默认参数取 MG90S 量级。
    """

    def __init__(self, hfov=60.0, vfov=36.0, frame_w=640, frame_h=480, latency=0.10,
                 max_rate=120.0, max_acc=600.0, deadband_deg=0.5, backlash_deg=1.0,
                 quant_deg=0.18, noise_deg=0.05):
        self.hfov, self.vfov = float(hfov), float(vfov)
        self.W, self.H = int(frame_w), int(frame_h)
        self.ppd_x, self.ppd_y = self.W / self.hfov, self.H / self.vfov
        self.latency, self.max_rate, self.max_acc = float(latency), float(max_rate), float(max_acc)
        self.deadband_deg, self.backlash = float(deadband_deg), float(backlash_deg)
        self.quant, self.noise = float(quant_deg), float(noise_deg)
        self.cmd = np.zeros(2)
        self.ang = np.zeros(2)
        self.rate = np.zeros(2)
        self._hist = deque()
        self._t = 0.0
        self._last_cmd = np.zeros(2)
        self.rng = np.random.default_rng(12345)

    def command(self, pan_deg, tilt_deg, dt):
        self.cmd[:] = (pan_deg, tilt_deg)
        self._t += dt
        self._hist.append((self._t, float(pan_deg), float(tilt_deg)))
        while len(self._hist) > 1 and self._hist[1][0] <= self._t - self.latency:
            self._hist.popleft()
        if self._hist[0][0] > self._t - self.latency:
            return self.ang.copy()          # 延迟期内：执行器还没收到有效指令
        target = np.array(self._hist[0][1:])
        for i in range(2):                  # 舵机死区：忽略很小的角度变化
            if abs(target[i] - self._last_cmd[i]) < self.deadband_deg:
                target[i] = self._last_cmd[i]
            else:
                self._last_cmd[i] = target[i]
        if self.quant > 0:
            target = np.round(target / self.quant) * self.quant
        if self.noise > 0:
            target = target + self.rng.normal(0.0, self.noise, 2)
        want = (target - self.ang) / max(1e-3, dt)
        dw = np.clip(want - self.rate, -self.max_acc * dt, self.max_acc * dt)
        self.rate = np.clip(self.rate + dw, -self.max_rate, self.max_rate)
        new_ang = self.ang + self.rate * dt
        for i in range(2):                  # 回差：反向要先走过间隙
            d = new_ang[i] - self.ang[i]
            if d > 0:
                self.ang[i] = max(self.ang[i], new_ang[i] - self.backlash * 0.5)
            elif d < 0:
                self.ang[i] = min(self.ang[i], new_ang[i] + self.backlash * 0.5)
        return self.ang.copy()

    def state(self):
        return {"pan": float(self.ang[0]), "tilt": float(self.ang[1]), "ok": True}

    def offset_px(self):
        """视窗偏移（世界坐标像素）：pan 正=向右转 -> 视窗右移；tilt 正=向上看 -> 视窗上移"""
        return (self.ang[0] * self.ppd_x, -self.ang[1] * self.ppd_y)

    def valid_rect(self):
        """**真实像素**的矩形 (x0, y0, x1, y1)；矩形外是因为云台平移露出来的黑边。
            render() 里 dst(x,y) = src(x-ox, y-oy)，所以输出有效的条件是
            0 <= x-ox < W 且 0 <= y-oy < H。黑边必须从检测里排除：黑区没有内容，但它的
        """
        ox, oy = self.offset_px()
        dx, dy = int(round(ox)), int(round(oy))
        return (max(0, dx), max(0, dy), min(self.W, self.W + dx), min(self.H, self.H + dy))

    def render(self, frame):
        """
        非拼接图模式：直接按当前角度平移画面，露出来的地方填黑。
        """
        ox, oy = self.offset_px()
        M = np.float32([[1, 0, -ox], [0, 1, -oy]])
        return cv2.warpAffine(frame, M, (self.W, self.H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

class ServoSerial(CameraLink):
    """真实云台（串口驱动板）：发绝对角度，脉宽换算放在单片机上（换舵机只改单片机）"""

    def __init__(self, port, baud=115200, timeout=0.02):
        import serial                        # pyserial 只在真硬件模式下需要
        self.ser = serial.Serial(port, int(baud), timeout=timeout)
        self.cache = {"pan": 0.0, "tilt": 0.0, "ok": True}

    def command(self, pan_deg, tilt_deg, dt):
        self.ser.write(("#P%+.2f T%+.2f\n" % (pan_deg, tilt_deg)).encode("ascii"))
        line = self.ser.readline().decode("ascii", "ignore").strip()
        if line.startswith("P"):
            try:
                parts = line.replace("T", " ").split()
                self.cache = {"pan": float(parts[0][1:]), "tilt": float(parts[1]), "ok": True}
            except Exception:
                pass

    def state(self):
        return self.cache

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

def make_camera(args, W, H):
    if args.real == "serial":
        return ServoSerial(args.port, args.baud)
    return SimCamera(hfov=args.hfov, vfov=args.vfov, frame_w=W, frame_h=H,
                     latency=args.latency, max_rate=args.sim_rate)

# 六、云台角度存储（JSON）

class GimbalState:
    """云台角度/标定/行程的持久化。没有位置回读的舵机上电时不知道自己指在哪，
    这个文件是软件侧唯一的记忆；**它不能替代编码器**（碰撞/丢步的绝对漂移发现不了）。"""

    def __init__(self, path=None, ang_limit=120.0, px_per_deg=12.0):
        self.path = path
        self.data = {"pan": 0.0, "tilt": 0.0, "ang_limit": float(ang_limit),
                     "px_per_deg": float(px_per_deg), "travel_pan": 0.0,
                     "travel_tilt": 0.0, "frames": 0, "loaded": False}
        self._last = np.zeros(2)
        self._dirty = 0
        if path:
            self.load()

    def load(self):
        try:
            import json
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                for k in self.data:
                    if k in d:
                        self.data[k] = d[k]
                self.data["loaded"] = True
                self._last = np.array([float(self.data["pan"]), float(self.data["tilt"])])
        except Exception:
            pass                       # 文件不存在/损坏都当全新开始
        return self.data["loaded"]

    def save(self, force=False):
        if not self.path:
            return False
        try:
            import json
            self._dirty = 0
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
            return True
        except Exception as e:
            if force:
                print("云台状态保存失败:", e)
            return False

    def update(self, pan, tilt):
        p, t = float(pan), float(tilt)
        self.data["travel_pan"] += abs(p - self._last[0])
        self.data["travel_tilt"] += abs(t - self._last[1])
        self._last = np.array([p, t])
        self.data["pan"], self.data["tilt"] = p, t
        self.data["frames"] = int(self.data["frames"]) + 1
        self._dirty += 1
        if self._dirty >= 300:
            self.save()

    def margin(self, pan, tilt):
        lim = float(self.data["ang_limit"])
        return lim - float(pan), lim + float(pan), lim - float(tilt), lim + float(tilt)

def put_text(img, text, org, scale, color, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)

def draw_overlay(img, box, score, state, err, dead, paused, search_rect=None, src="ncc"):
    H, W = img.shape[:2]
    cx, cy = W // 2, H // 2
    db = int(0.03 * W)
    cv2.line(img, (cx - 24, cy), (cx + 24, cy), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 24), (cx, cy + 24), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (cx - db, cy - db), (cx + db, cy + db), (0, 200, 0), 1)
    if box is not None:
        x, y, w, h = [int(round(v)) for v in box]
        color = (0, 0, 255) if src == "recipe" else (0, 200, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        bx, by = int(x + w / 2), int(y + h / 2)
        cv2.circle(img, (bx, by), 3, color, -1)
        cv2.arrowedLine(img, (bx, by), (cx, cy), (0, 200, 0), 1, tipLength=0.12)
    if search_rect is not None:
        sx, sy, sw, sh = [int(round(v)) for v in search_rect]
        cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
    put_text(img, "%s %.2f  err %+.3f %+.3f  %s%s" %
             (state, score, err[0], err[1], "DEAD" if dead else "    ",
              "  PAUSED" if paused else ""), (8, H - 10), 0.55, (0, 255, 255))

# 八、合成测试场（自检 / 场景矩阵用，完全确定性）

class SynthStage:
    """
    大画布 + 随机纹理背景 + 一个高纹理目标，可按视窗取景。
    """

    def __init__(self, W=640, H=480, scale=3, seed=0, tw=80, th=120):
        rng = np.random.default_rng(seed)
        self.W, self.H = int(W), int(H)
        self.sw, self.sh = self.W * scale, self.H * scale
        scene = (rng.random((self.sh, self.sw)) * 120 + 60).astype(np.uint8)
        for _ in range(60):
            x, y = int(rng.integers(0, self.sw - 60)), int(rng.integers(0, self.sh - 60))
            cv2.rectangle(scene, (x, y), (x + int(rng.integers(20, 60)),
                                          y + int(rng.integers(20, 60))),
                          int(rng.integers(40, 220)), -1)
        self.scene = cv2.GaussianBlur(scene, (5, 5), 0)
        self.tgt = (rng.random((th, tw)) * 150 + 50).astype(np.uint8)
        for _ in range(6):
            cv2.circle(self.tgt, (int(rng.integers(4, tw - 4)), int(rng.integers(4, th - 4))),
                       int(rng.integers(3, 8)), int(rng.integers(30, 255)), -1)
        self.tgt = cv2.GaussianBlur(self.tgt, (3, 3), 0)
        self.tw, self.th = tw, th
        self.cx, self.cy = self.sw / 2.0, self.sh / 2.0
        self.rel = np.zeros(2, np.float32)
        self.vel = np.zeros(2, np.float32)
        self.hide_until = 0.0
        self.t = 0.0
        self.truth_box = None

    def step(self, dt):
        self.t += dt
        self.rel += self.vel * dt

    def _box_on_screen(self, rel, off_px=None):
        off = np.zeros(2, np.float32) if off_px is None else np.asarray(off_px, np.float32)
        r = np.asarray(rel, np.float32) - off
        return np.array([self.W / 2.0 + r[0] - self.tw / 2.0,
                         self.H / 2.0 + r[1] - self.th / 2.0, self.tw, self.th], np.float32)

    def render(self, off_px, tint=None):
        """render：off_px 为视窗偏移；tint 可给目标上色（颜色干扰物测试用）"""
        x0 = int(clamp(int(round(self.cx + off_px[0] - self.W / 2.0)), 0, self.sw - self.W))
        y0 = int(clamp(int(round(self.cy + off_px[1] - self.H / 2.0)), 0, self.sh - self.H))
        frame = self.scene[y0:y0 + self.H, x0:x0 + self.W].copy()

        def paste(rel_pos, patch):
            px = int(round(self.cx + rel_pos[0] - x0 - patch.shape[1] / 2.0))
            py = int(round(self.cy + rel_pos[1] - y0 - patch.shape[0] / 2.0))
            xs, ys = max(0, px), max(0, py)
            xe, ye = min(self.W, px + patch.shape[1]), min(self.H, py + patch.shape[0])
            if xe > xs and ye > ys:
                frame[ys:ye, xs:xe] = patch[ys - py:ye - py, xs - px:xe - px]

        if self.t >= self.hide_until:
            paste(self.rel, self.tgt)
            self.truth_box = self._box_on_screen(self.rel, off_px)
        else:
            self.truth_box = None
        out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if tint is not None:                 # 直接给目标区域上色（造颜色干扰物/换色）
            tb = self._box_on_screen(self.rel, off_px)
            x, y, w, h = [int(round(v)) for v in tb]
            xs, ys = max(0, x), max(0, y)
            xe, ye = min(self.W, x + w), min(self.H, y + h)
            if xe > xs and ye > ys:
                sub = out[ys:ye, xs:xe].astype(np.float32)
                col = np.array(tint, np.float32)
                out[ys:ye, xs:xe] = np.clip(0.35 * sub + 0.65 * col, 0, 255).astype(np.uint8)
        return out

    def truth(self):
        return self.truth_box

# 九、主循环（GUI）：静态等待模式

FPS_CAP = 30.0
GAP = 4
MAIN_WIN = "Follow static-wait (attempt9)"

class TargetTracker:
    """
    局部跟踪（NCC+PSR） + HSV(HS) 前景/背景模型 + 分层区域重检测。
    """

    STATE_TRACK = "track"
    STATE_LOST = "lost"

    HS_H, HS_S = 16, 8                     # 直方图分箱：H 16 × S 8 = 128

    def __init__(self, search_ratio=0.3, score_keep=0.50, psr_keep=2.0,
                 score_hi=0.70, psr_hi=3.0, lost_after=12, bwd_score=0.45,
                 bwd_psr=1.5, llr_min=0.02, hist_lr=0.10, hist_floor=0.01,
                 fill_min=0.35, size_ema=0.15, min_box=10.0, mask_step=0.06,
                 mask_step_max=0.12,
                 edge_min=0.05, cent_pull=0.15, ref_lo=0.45, ref_hi=2.2, ref_q=0.75,
                 ref_lr=0.02, edge_margin=0.04, search_pad_min=56,
                 llr_floor_frac=0.35, size_dev_cap=0.25,
                 ref_grow_lr=0.35, ref_grow_ok=0.80, step_grow=0.18, ema_grow=0.55,
                 gm_comp=True, gm_scale=0.25, gm_min_resp=0.10, gm_max=120.0,
                 gm_rot_min=0.35, gm_rot_max=12.0, gm_orb_n=400, gm_orb_min=10,
                 search_margin=1.0, ncc_ds=0.5,
                 size_every=2, bwd_every=3, refine_every=2, model_every=2,
                 score_psr_free=0.85, score_soft=0.68, soft_pull=0.35, soft_jump=0.9,
                 jump_cap=0.18, jump_penalty=0.8, slow_gain=0.55, search_grow=0.9):
        self.search_ratio = float(search_ratio)
        self.score_keep, self.psr_keep = float(score_keep), float(psr_keep)
        self.score_hi, self.psr_hi = float(score_hi), float(psr_hi)
        self.lost_after = max(1, int(lost_after))
        self.bwd_score, self.bwd_psr = float(bwd_score), float(bwd_psr)
        self.llr_min = float(llr_min)
        self.hist_lr, self.hist_floor = float(hist_lr), float(hist_floor)
        self.edge_margin = float(edge_margin)
        self.fill_min = float(fill_min)      # 颜色比例下限（掩膜填充率）
        self.size_ema = float(size_ema)      # 尺寸跟随速度（小=稳，大=跟得快）
        self.min_box = float(min_box)        # 框最小边长（防"缩没了"）
        self.mask_step = float(mask_step)    # 掩膜路径的单帧尺寸变化上限（6%）
        self.mask_step_max = float(mask_step_max)   # 硬上限（12%，任何情况下不许超）
        self.edge_min = float(edge_min)      # 框内最低边缘密度（空框/黑边挡在这里）
        self.cent_pull = float(cent_pull)    # 框心向掩膜质心牵引的比例（0=不牵引）
        self.ref_lo, self.ref_hi = float(ref_lo), float(ref_hi)   # 允许的面积比区间
        self.ref_q = float(ref_q)            # 掩膜阈值锚：框内 LLR 的这个分位
        self._grow_ok = False                # 置信帧才允许突破绝对放大上限
        self.ref_lr = float(ref_lr)          # 基准的自更新速率（只在置信帧）
        self.search_pad_min = max(8, int(search_pad_min))
        self.llr_floor_frac = float(llr_floor_frac)
        self.size_dev_cap = float(size_dev_cap)
        # **基准"跟涨"速率**（关键修复，见 _fit_size）：ref_lr=0.02 是"防棘轮"的慢刹车， 目标真走近（面积每帧 +6%）时 ref_area 永远追不上 ->
        self.ref_grow_lr = float(ref_grow_lr)
        self.ref_grow_ok = float(ref_grow_ok)   # 触发快速跟涨的"掩膜/基准面积比"阈值
        self.step_grow = float(step_grow)       # 确认真增长时的单帧尺寸步长上限
        self.ema_grow = float(ema_grow)         # 确认真增长时的尺寸 EMA（更快跟涨）
        self.gm_comp = bool(gm_comp)
        self.gm_scale = float(gm_scale)         # 相位相关降采样倍率（0.25 = 4x 提速）
        self.ncc_ds = float(ncc_ds)             # video_proc30：NCC 搜索窗降采样倍率（0.5 = 4x 提速）
        self.gm_min_resp = float(gm_min_resp)   # 相位相关响应下限（低于则视为不可信）
        self.gm_max = float(gm_max)             # 单帧全局位移限幅（防纯旋转被放大）
        self.gm_rot_min = float(gm_rot_min)     # 低于此角度的旋转不做去旋转（省一次 warp）
        self.gm_rot_max = float(gm_rot_max)     # 单帧旋转限幅
        self.gm_orb_n = int(gm_orb_n)           # ORB 特征数
        self.gm_orb_min = int(gm_orb_min)       # 用于估计的最小内点数
        self._gm_prev, self._gm_win = None, None
        self._gm_kp = (None, None)
        self._orb = self._bf = None
        self._rot_ctx = None
        self.search_margin = float(search_margin)
        self.size_every = max(1, int(size_every))
        self.bwd_every = max(1, int(bwd_every))
        self.refine_every = max(1, int(refine_every))
        self.model_every = max(1, int(model_every))
        self.score_psr_free = float(score_psr_free)
        self.score_soft = float(score_soft)   # 次级豁免的最低分（配 d_peak 判据）
        self.soft_pull = float(soft_pull)     # 峰值允许偏离预测中心的比例
        self.soft_jump = float(soft_jump)     # 峰值帧间允许跳动的比例（稳定性判据）
        self.jump_cap = float(jump_cap)
        self.jump_penalty = float(jump_penalty)
        self.slow_gain = float(slow_gain)
        self.search_grow = float(search_grow)
        self._last_peak = None

        self.box = None
        self.template = None        # 工作模板（float32，已去均值）
        self.vel = np.zeros(2, np.float32)
        self.score = self.psr = 0.0
        self.state = self.STATE_LOST
        self.missing = 0
        self.lost_since = None
        self.lost_box = None        # 丢失瞬间的目标框（重捕获以它的中心为圆心）
        self.fg = self.bg = None    # HS 直方图（前景/背景，用于 LLR 图）
        self.color_sep = 0.0        # 颜色模型的信息量（fg/bg 直方图距离）
        self.sep_min = 0.15         # 弱门槛：只判"HS 直方图有没有差别"。
                                    # 注意它在真实场景里几乎恒为 0.22~0.29（连地砖都 0.287）， 所以**不能**拿它当"目标够不够彩色"的判据 —— 真正的把关是下面那几道几何闸门 + 颜色比例。
        self.prev_gray = self.prev_box = None
        self.valid = None                   # 真实像素矩形 (x0,y0,x1,y1)；外圈是黑边，不算
        self.invalid_llr = -20.0             # 黑边处强制的 LLR（"绝对不是目标"）
        # ---- 颜色比例特征存储（目标自身统计量基准）---- 存的是目标的**绝对**基准：面积、长宽比、掩膜填充率、边缘密度。 为什么必须存绝对基准：闸门若只跟**当前框**比（面积比 0.3~2.2
        self.ref_area = None
        self.ref_aspect = None
        self.ref_fill = None
        self.ref_edge = None
        self.ref_llr_q = None                # 基准框内 LLR 的低分位 -> 掩膜阈值锚
        self.color_recipe = None             # [(hs_bin, weight), ...] 按权重降序
        self.recipe_k = 3                    # 只存占比最大的 3 种颜色
        self.recipe_min_sim = 0.55           # 配方相似度门槛（直方图交集，0~1）
        self.recipe_confirm_hi = 0.8         # 相似度≥此值只需 1 帧确认，否则 2 帧
        self.recipe_leak_max = 0.25          # 框外环带彩色占比上限（挡"从大物体切子窗"）
        self.recipe_ratio_tol = 0.45         # 配方**占比**允许的相对偏差（主色/配比顺序）
        self.init_area = None                # 用户选定框的面积（绝对放大上限的锚）
        self.grow_hi = 3.0                   # 不置信时最多放大到初始面积的几倍
        self.anchor_box = None               # 最后一次**置信**框 = 丢失点附近搜索的锚点
        self.src = "ncc"                     # 本帧定位来源："ncc"=帧间NCC(绿框) / "recipe"=颜色构成配方(红框)
        self._cand_hits = 0
        self._cand_box = None
        self.n_recover = 0
        self._lut = None                     # LLR 查找表缓存（只有模型更新后才失效）
        self._lut_dirty = True
        self._bwd_tick = 0
        self._tick = 0
        self._ext_cache = None
        self._view_ang = None       # 本帧 view 对应的云台角 (pan,tilt,deg)
        self._wt_prev = None        # 上一帧 view 的云台角（补偿基准）
        self._view_wh = None        # 本帧 view 的 (W,H)
        self._ppd = None            # (px/deg_x, px/deg_y)，由 set_view_angles 传入
        self._last_dt = 1 / 30.0
        self._warp_dxy = (0.0, 0.0)  # 本帧 warp 造成的假位移（速度估计要扣掉）
        self._warp_dpan = 0.0
        self._gpan_px = (0.0, 0.0)   # 本帧云台自身运动造成的画面位移（进搜索半径）

    @staticmethod
    def _prep(patch):
        p = patch.astype(np.float32)
        return p - float(p.mean())

    @staticmethod
    def _grab(gray, box, pad=False):
        """
        按框取一块。pad=True 时框外**补黑**（不复制边缘；画面外本来就是"没有"）。
        """
        x, y, w, h = [int(round(v)) for v in box]
        if w < 4 or h < 4:
            return None
        H, W = gray.shape[:2]
        if not pad:
            if x < 0 or y < 0 or x + w > W or y + h > H:
                return None
            return gray[y:y + h, x:x + w]
        cx0, cy0 = max(0, x), max(0, y)
        cx1, cy1 = min(W, x + w), min(H, y + h)
        if cx1 <= cx0 or cy1 <= cy0:
            return np.zeros((h, w) + gray.shape[2:], gray.dtype)
        return cv2.copyMakeBorder(gray[cy0:cy1, cx0:cx1], cy0 - y, y + h - cy1,
                                  cx0 - x, x + w - cx1, cv2.BORDER_CONSTANT, value=0)

    @staticmethod
    def _subpixel(res, loc):
        """抛物线拟合求亚像素峰（整数像素对"保持居中"不够）"""
        x, y = loc
        if x <= 0 or y <= 0 or x >= res.shape[1] - 1 or y >= res.shape[0] - 1:
            return float(x), float(y)

        def peak(v0, v1, v2):
            d = v0 - 2.0 * v1 + v2
            return 0.0 if abs(d) < 1e-9 else float(np.clip(0.5 * (v0 - v2) / d, -1.0, 1.0))
        return (x + peak(res[y, x - 1], res[y, x], res[y, x + 1]),
                y + peak(res[y - 1, x], res[y, x], res[y + 1, x]))

    @staticmethod
    def _psr(res, loc, r=4, valid=None):
        """
        峰值旁瓣比：峰值比旁瓣高出几个标准差。valid 排除补边区（那些响应不是证据）
        """
        h, w = res.shape
        x0, x1 = max(0, loc[0] - r), min(w, loc[0] + r + 1)
        y0, y1 = max(0, loc[1] - r), min(h, loc[1] + r + 1)
        m = np.ones(res.shape, bool)
        m[y0:y1, x0:x1] = False
        if valid is not None:
            mv = m & valid
            if mv.sum() >= 8:
                m = mv
        s = res[m]
        if s.size < 8:
            return float("inf")            # 样本不足 = PSR 无法估计，不是"峰不突出"
        sd = float(s.std())
        return float((res[loc[1], loc[0]] - float(s.mean())) / (sd + 1e-6))

    def _match(self, gray, tmpl, center, size, search_ratio, full=False, rot=0.0):
        """
        在 center 附近搜 tmpl，返回 ((x,y,w,h), 分数, PSR)。
        """
        H, W = gray.shape[:2]
        tw, th = tmpl.shape[1], tmpl.shape[0]
        ds = float(getattr(self, "ncc_ds", 1.0))
        if full or abs(rot) >= self.gm_rot_min or ds >= 0.999 or min(tw, th) < 8:
            ds = 1.0
        if ds < 0.999:
            return self._match_ds(gray, tmpl, center, size, search_ratio, ds)
        valid = None
        if full:
            win, wx0, wy0 = gray, 0, 0
        else:
            mx = int(round(max(tw, size[0]) * search_ratio)) + 2
            my = int(round(max(th, size[1]) * search_ratio)) + 2
            # **绝对下限（关键修复）**：search_ratio 是"占框尺寸的比例"，小框（94×84） ≤28px，于是"画面在动、框不动"然后丢）。给搜索半径加一个与目标尺寸无关的
            mx = max(mx, self.search_pad_min)
            my = max(my, self.search_pad_min)
            # **云台速度前馈必须进搜索半径**：云台一帧转 Δpan 度，目标在画面里就瞬移
            if self._gpan_px:
                mx = max(mx, int(round(abs(self._gpan_px[0]))) + 6)
                my = max(my, int(round(abs(self._gpan_px[1]))) + 6)
            ww, wh = 2 * mx + tw, 2 * my + th
            wx0 = int(round(center[0] - tw / 2.0 - mx))
            wy0 = int(round(center[1] - th / 2.0 - my))
            cx0, cy0 = max(0, wx0), max(0, wy0)
            cx1, cy1 = min(W, wx0 + ww), min(H, wy0 + wh)
            if cx1 - cx0 < tw + 2 or cy1 - cy0 < th + 2:
                return None                 # 连模板都放不下
            win = cv2.copyMakeBorder(gray[cy0:cy1, cx0:cx1], cy0 - wy0,
                                     wy0 + wh - cy1, cx0 - wx0, wx0 + ww - cx1,
                                     cv2.BORDER_CONSTANT, value=0)
            if abs(rot) >= self.gm_rot_min and win.size:
                cwin = (win.shape[1] / 2.0, win.shape[0] / 2.0)
                M = cv2.getRotationMatrix2D(cwin, rot, 1.0)
                win = cv2.warpAffine(win, M, (win.shape[1], win.shape[0]),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                self._rot_ctx = (rot, cwin, wx0, wy0)
            else:
                self._rot_ctx = None
        if win.shape[0] < th + 2 or win.shape[1] < tw + 2:
            return None
        res = cv2.matchTemplate(win.astype(np.float32), tmpl, cv2.TM_CCOEFF_NORMED)
        if full:
            x0, y0 = 0, 0
        else:
            x0, y0 = wx0, wy0
            oxs = np.arange(res.shape[1]) + wx0
            oys = np.arange(res.shape[0]) + wy0
            valid = (((oxs >= 0) & (oxs <= W - tw))[None, :]
                     & ((oys >= 0) & (oys <= H - th))[:, None])
            if self.valid is not None:
                # 黑边里的位置不许参与竞争：那里没有真实像素（黑区方差为 0， 归一化相关会给虚假高分，实测能把框吸到黑边/画面角落上）
                vx0, vy0, vx1, vy1 = self.valid
                valid &= (((oxs >= vx0) & (oxs + tw <= vx1))[None, :]
                          & ((oys >= vy0) & (oys + th <= vy1))[:, None])
            res = np.where(valid, res, -1.0).astype(np.float32)
        _, score, _, loc = cv2.minMaxLoc(res)
        px, py = self._subpixel(res, loc)
        ax, ay = x0 + px, y0 + py
        rc = getattr(self, "_rot_ctx", None)
        if rc is not None:
            rot_used, (ccx, ccy), wx0_, wy0_ = rc
            lx, ly = ax - wx0_, ay - wy0_
            th_ = math.radians(rot_used)
            rx = ccx + (lx - ccx) * math.cos(th_) - (ly - ccy) * math.sin(th_)
            ry = ccy + (lx - ccx) * math.sin(th_) + (ly - ccy) * math.cos(th_)
            ax, ay = wx0_ + rx, wy0_ + ry
        return ((ax, ay, tw, th), float(score),
                self._psr(res, loc, valid=valid))

    def _match_ds(self, gray, tmpl, center, size, search_ratio, ds):
        """降采样版 _match（video_proc30）。把搜索窗与模板各按 ds 缩放，在**小图**上做
        TM_CCOEFF_NORMED，峰值/PSR/亚像素都在小图网格里算，最后把峰值坐标 /ds 映射回原图。
        搜索窗、valid 掩膜、PSR 邻居半径都按小图尺度同步缩放，保证语义一致。
        """
        H, W = gray.shape[:2]
        tw, th = tmpl.shape[1], tmpl.shape[0]
        tws, ths = max(2, int(round(tw * ds))), max(2, int(round(th * ds)))
        mx = max(int(round(max(tw, size[0]) * search_ratio)) + 2, self.search_pad_min)
        my = max(int(round(max(th, size[1]) * search_ratio)) + 2, self.search_pad_min)
        if self._gpan_px:
            mx = max(mx, int(round(abs(self._gpan_px[0]))) + 6)
            my = max(my, int(round(abs(self._gpan_px[1]))) + 6)
        mxs, mys = max(int(round(mx * ds)), 4), max(int(round(my * ds)), 4)
        wx0 = int(round(center[0] - tw / 2.0 - mx))
        wy0 = int(round(center[1] - th / 2.0 - my))
        ww, wh = 2 * mx + tw, 2 * my + th
        cx0, cy0 = max(0, wx0), max(0, wy0)
        cx1, cy1 = min(W, wx0 + ww), min(H, wy0 + wh)
        if cx1 - cx0 < tw + 2 or cy1 - cy0 < th + 2:
            return None
        win = cv2.copyMakeBorder(gray[cy0:cy1, cx0:cx1], cy0 - wy0, wy0 + wh - cy1,
                                 cx0 - wx0, wx0 + ww - cx1, cv2.BORDER_CONSTANT, value=0)
        gw = cv2.resize(win, (max(2, int(round(win.shape[1] * ds))),
                              max(2, int(round(win.shape[0] * ds)))),
                        interpolation=cv2.INTER_AREA).astype(np.float32)
        gt = cv2.resize(tmpl.astype(np.float32), (tws, ths), interpolation=cv2.INTER_AREA)
        if gw.shape[0] < ths + 2 or gw.shape[1] < tws + 2:
            return None
        res = cv2.matchTemplate(gw, gt, cv2.TM_CCOEFF_NORMED)
        # 小图 → 原图的列/行映射：小图第 k 列采样点在原图窗口的 x = (k+0.5)/ds - 0.5（像素中心约定）。
        kx = (np.arange(res.shape[1]) + 0.5) / ds - 0.5 + wx0
        ky = (np.arange(res.shape[0]) + 0.5) / ds - 0.5 + wy0
        valid = (((kx >= 0) & (kx <= W - tw))[None, :]
                 & ((ky >= 0) & (ky <= H - th))[:, None])
        if self.valid is not None:
            vx0, vy0, vx1, vy1 = self.valid
            valid &= (((kx >= vx0) & (kx + tw <= vx1))[None, :]
                      & ((ky >= vy0) & (ky + th <= vy1))[:, None])
        res = np.where(valid, res, -1.0).astype(np.float32)
        _, score, _, loc = cv2.minMaxLoc(res)
        # 不能再用 loc + px（会重复计一次 loc —— 这正是之前偏了 (mx,my) 的原因）。
        px, py = self._subpixel(res, loc)
        ax = (px + 0.5) / ds - 0.5 + wx0
        ay = (py + 0.5) / ds - 0.5 + wy0
        return ((ax, ay, tw, th), float(score),
                self._psr(res, loc, r=max(1, int(round(4 * ds))), valid=valid))

    def _hs_idx(self, hsv):
        """把 HSV 降到 128 个 (H,S) 格子。**不用 V**：V 是亮度，光照一变就废。
        """
        return ((hsv[..., 0].astype(np.int32) >> 4) * self.HS_S
                + (hsv[..., 1].astype(np.int32) >> 5))

    def _hs_hist(self, hsv):
        """S 加权的 HS 直方图。
        """
        idx = self._hs_idx(hsv).ravel()
        w = (0.25 + 0.75 * (hsv[..., 1].astype(np.float64) / 255.0)).ravel()
        h = np.bincount(idx, weights=w, minlength=self.HS_H * self.HS_S)
        return h / (h.sum() + 1e-9)

    def _llr_lut(self):
        """LLR 查找表（128 格）。缓存：只有前景/背景模型更新后才重算 ——
        原来每调用一次就做一遍 log，而一帧里 llr_map 会被调用十几次。"""
        if self._lut is None or self._lut_dirty:
            fg = np.maximum(self.fg, self.hist_floor)
            bg = np.maximum(self.bg, self.hist_floor)
            self._lut = np.log(fg / bg).astype(np.float32)
            self._lut_dirty = False
        return self._lut

    def llr_map(self, hsv):
        """逐像素"更像目标还是更像背景"的对数似然比图"""
        m = self._llr_lut()[self._hs_idx(hsv)]
        if self.valid is not None:
            x0, y0, x1, y1 = self.valid
            m[:y0, :] = self.invalid_llr
            m[y1:, :] = self.invalid_llr
            m[:, :x0] = self.invalid_llr
            m[:, x1:] = self.invalid_llr
        return m

    def set_valid_rect(self, rect):
        """告诉跟踪器哪块是**真实像素**（云台平移露出的黑边不算检测范围）。

        视频在环时每帧调用；真机没有黑边，保持 None 即可。
        """
        self.valid = None if rect is None else tuple(int(round(v)) for v in rect)

    def _valid_ok(self, rect):
        """矩形是否在真实像素范围内（和有效区有交集才算）"""
        if self.valid is None:
            return True
        x0, y0, x1, y1 = self.valid
        return not (rect[0] + rect[2] <= x0 or rect[0] >= x1
                    or rect[1] + rect[3] <= y0 or rect[1] >= y1)

    def _clip_roi(self, x0, y0, x1, y1):
        """把 ROI 夹到真实像素范围内"""
        if self.valid is None:
            return x0, y0, x1, y1
        vx0, vy0, vx1, vy1 = self.valid
        return max(x0, vx0), max(y0, vy0), min(x1, vx1), min(y1, vy1)

    def _ring_box(self, box, grow=0.6):
        x, y, w, h = box
        return (x - w * grow, y - h * grow, w * (1 + 2 * grow), h * (1 + 2 * grow))

    def _update_model(self, bgr, box):
        """
        只在置信帧、框完整在**真实像素**内时更新模型。
        """
        H, W = bgr.shape[:2]
        x, y, w, h = box
        if x < 2 or y < 2 or x + w > W - 2 or y + h > H - 2:
            return
        if self.valid is not None:
            vx0, vy0, vx1, vy1 = self.valid
            if x < vx0 + 1 or y < vy0 + 1 or x + w > vx1 - 1 or y + h > vy1 - 1:
                return
        rb = self._grab(bgr, box, pad=True)
        ring = self._grab(bgr, self._ring_box(box), pad=True)
        if rb is None or ring is None:
            return
        hf = self._hs_hist(cv2.cvtColor(rb, cv2.COLOR_BGR2HSV))
        hb = self._hs_hist(cv2.cvtColor(ring, cv2.COLOR_BGR2HSV))
        lr = self.hist_lr
        if self.fg is None:
            self.fg, self.bg = hf.copy(), hb.copy()
        else:
            self.fg = (1 - lr) * self.fg + lr * hf
            self.bg = (1 - lr) * self.bg + lr * hb
        self._lut_dirty = True               # 模型变了 -> 作废 LLR 缓存
        # 颜色模型的"信息量" = 前景/背景直方图的距离（0..1）。 纯灰或目标与周围同色的场景里两者几乎相同，这时**颜色判据必须自动失效**， 退回到只用灰度 NCC —— 否则 LLR 恒为
        self.color_sep = 0.5 * float(np.abs(self.fg - self.bg).sum())

    @property
    def color_ok(self):
        """颜色模型是否携带信息（不携带时所有颜色判据自动旁路）"""
        return self.color_sep >= self.sep_min

    @property
    def _tmpl_textured(self):
        """当前工作模板是否**有纹理**（去均值后的标准差）。
        """
        if self.template is None:
            return False
        return float(self.template.std()) >= 6.0

    def border_edge(self, bgr, box, band=2):
        """框的**边框**压在物体轮廓上的程度（归一化 Sobel 幅值在框边 1~2px 上的均值）。
        """
        x, y, w, h = [int(round(v)) for v in box]
        H, W = bgr.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return 0.0
        sub = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(sub, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sub, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ehi = float(np.percentile(mag, 95)) or 1.0
        mag = np.clip(mag / ehi, 0.0, 1.0)
        b = int(max(1, min(band, (x1 - x0) // 4, (y1 - y0) // 4)))
        ring = np.concatenate([mag[:b, :].ravel(), mag[-b:, :].ravel(),
                               mag[:, :b].ravel(), mag[:, -b:].ravel()])
        return float(ring.mean()) if ring.size else 0.0

    def _box_llr_q(self, bgr, box, q):
        """框内 LLR 的 q 分位（掩膜阈值锚："目标自己有多像目标"）"""
        if self.fg is None:
            return None
        x, y, w, h = [int(round(v)) for v in box]
        x0, y0, x1, y1 = self._clip_roi(x, y, x + w, y + h)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return None
        sub = bgr[y0:y1, x0:x1]
        llr = self.llr_map(cv2.cvtColor(sub, cv2.COLOR_BGR2HSV))
        return float(np.percentile(llr, q * 100.0))

    def measure_extent(self, bgr, box, margin=0.35):
        """
        【颜色比例 + 边界（掩膜连通域）+ 物体边缘】量目标实际占多大。
        """
        H, W = bgr.shape[:2]
        x, y, w, h = [int(round(v)) for v in box]
        gx, gy = int(w * margin), int(h * margin)
        # ROI 的**上限由存储的目标基准尺寸决定**，不由当前框决定：否则框一涨 ROI 就跟着涨， 掩膜在更大的范围里再连到更多背景，正反馈棘轮（d.mp4 紫伞行人：ROI 跟着框涨，
        if self.ref_area is not None:
            # 真增长（_grow_streak 高）时把 ROI 上限也一起放开，否则 ROI 卡在旧基准上， 掩膜根本看不到已经长大的目标 -> 测量值偏小 -> 框永远追不上（实测：
            roi_k = 1.6
            if getattr(self, "_grow_streak", 0) >= 2:
                roi_k = 1.6 + 0.35 * min(4, self._grow_streak)
            rw = int(round(math.sqrt(self.ref_area) * roi_k)) + 2
            gx, gy = min(gx, max(6, rw - w)), min(gy, max(6, rw - h))
        x0, y0 = max(0, x - gx), max(0, y - gy)
        x1, y1 = min(W, x + w + gx), min(H, y + h + gy)
        x0, y0, x1, y1 = self._clip_roi(x0, y0, x1, y1)      # 黑边不进检测范围
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        sub = bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        llr = self.llr_map(hsv)
        lo, hi = float(llr.min()), float(llr.max())
        color_use = (hi - lo) >= 1e-3
        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        gg = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gh = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(gg, gh)
        ehi = float(np.percentile(edge, 95)) or 1.0
        edge = np.clip(edge / ehi, 0.0, 1.0)
        if color_use:
            if self.ref_llr_q is not None:
                thr = self.ref_llr_q
                mask = ((llr >= thr).astype(np.uint8)) * 255
            else:
                ev = (llr - lo) / (hi - lo)
                u8 = (ev * 255.0).astype(np.uint8)
                _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            ev = edge
            u8 = (ev * 255.0).astype(np.uint8)
            _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = max(3, int(round(0.05 * min(x1 - x0, y1 - y0))) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        # 闭运算（大核）把轮廓连成闭合边界，再填洞 -> 得到**实心**的物体区域。 这一步是"边界/边缘检测"能用来量尺寸的关键：轮廓 + 填充 = 物体范围。
        ck = max(3, int(round(0.08 * min(x1 - x0, y1 - y0))) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))
        ff = mask.copy()
        ffm = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
        cv2.floodFill(ff, ffm, (0, 0), 255)                 # 外部背景 -> 255
        mask = cv2.bitwise_or(mask, cv2.bitwise_not(ff))    # 内部空洞补上
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return None
        # 取**覆盖当前框中心**的那个连通域（不是"最大"的）：ROI 里还可能有别的同类色物体， 最大连通域可能整个跑到旁边去（实测掩膜会连到湿路面/背景上）。
        lx, ly = int(clamp(x + w / 2.0 - x0, 0, mask.shape[1] - 1)), \
            int(clamp(y + h / 2.0 - y0, 0, mask.shape[0] - 1))
        lid = int(_lab[ly, lx])
        if lid == 0:                     # 框心不在掩膜里 -> 退而求其次：与框重叠最大的连通域
            best_ov, lid = 0, 0
            bx0, by0 = max(0, x - x0), max(0, y - y0)
            bx1, by1 = min(mask.shape[1], x + w - x0), min(mask.shape[0], y + h - y0)
            for j in range(1, n):
                sx, sy, sw, sh, _a = stats[j]
                ox = max(0, min(bx1, sx + sw) - max(bx0, sx))
                oy = max(0, min(by1, sy + sh) - max(by0, sy))
                if ox * oy > best_ov:
                    best_ov, lid = ox * oy, j
            if lid == 0:
                return None
        i = lid
        mx, my, mw, mh, area = (int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]),
                                int(stats[i, 3]), float(stats[i, 4]))
        if area < 12 or mw < 4 or mh < 4:
            return None
        fill = area / float(mw * mh)
        cx_m, cy_m = float(cent[i][0]) + x0, float(cent[i][1]) + y0
        ix0, iy0 = max(0, x - x0), max(0, y - y0)
        ix1, iy1 = min(mask.shape[1], x + w - x0), min(mask.shape[0], y + h - y0)
        fit_in_box = (float((mask[iy0:iy1, ix0:ix1] > 0).sum()) / max(1.0, w * h)
                      if ix1 > ix0 and iy1 > iy0 else 0.0)
        eb = edge[iy0:iy1, ix0:ix1] if ix1 > ix0 and iy1 > iy0 else None
        edge_den = float((eb > 0.35).mean()) if eb is not None and eb.size else 0.0
        return ((float(mx + x0), float(my + y0), float(mw), float(mh)), float(fill),
                float(area), (cx_m, cy_m), float(fit_in_box), float(edge_den))

    def _fit_size(self, box, extent):
        """
        用掩膜的几何范围 + 颜色比例来修框大小与框心（返回新框或 None=保持不动）。
        """
        mbox, fill, marea, centroid, fit_in_box, edge_den = extent
        bx, by, bw, bh = box
        if fill < self.fill_min or marea < 12:
            return None
        # ---- 和**存储的目标基准**比（不是和当前框比，否则会棘轮）----
        if self.ref_area is not None:
            if not (self.ref_lo * self.ref_area <= marea <= self.ref_hi * self.ref_area):
                return None
            ar = (mbox[2] / max(1.0, mbox[3]))
            if self.ref_aspect is not None and not (0.5 * self.ref_aspect <= ar
                                                    <= 2.0 * self.ref_aspect):
                return None
            if self.ref_fill is not None and fill < 0.5 * self.ref_fill:
                return None
        if edge_den < self.edge_min and self._last_border_edge < self.edge_min:
            return None
        if not self._valid_ok(mbox):
            return None
        ratio = marea / max(1.0, bw * bh)
        if ratio < 0.3 or ratio > 1.6:
            return None
        ar_init = mbox[2] / max(1.0, mbox[3])
        if not (0.75 * self.init_aspect <= ar_init <= 1.35 * self.init_aspect):
            return None
        # 同理，**掩膜外接框的填充率**相对初始框不能塌：横向细带 fill 会很低。
        if self.ref_fill is not None and fill < 0.45 * self.ref_fill:
            return None
        if abs(mbox[2] - bw) > self.mask_step_max * bw or \
                abs(mbox[3] - bh) > self.mask_step_max * bh:
            return None
        # 颜色比例：框内被判成目标的像素占比太低时，只有"掩膜明显比框大"（说明是 框缩太小了、目标其实还在）才允许继续调整，否则一律冻结 —— 这一条挡住的是 "掩膜连到旁边同类色物体、把框往那边拽"的漂移。
        if fit_in_box < self.fill_min * 0.7 and marea < 1.5 * bw * bh:
            return None
        ar_box = bw / max(1.0, bh)
        ar_mask = mbox[2] / max(1.0, mbox[3])
        if ar_mask / ar_box < 0.5 or ar_mask / ar_box > 2.0:
            return None
        if math.hypot(centroid[0] - (bx + bw / 2.0), centroid[1] - (by + bh / 2.0)) \
                > 0.6 * max(bw, bh):
            return None
        st = self.mask_step
        ema = self.size_ema
        if getattr(self, "_grow_streak", 0) >= 2:
            # 持续真增长：步长放宽 + EMA 加快。二者缺一不可—— 只放宽步长，EMA=0.15 仍把每帧实际变化压到 2.7%，追不上 6%/帧； 只加快 EMA，步长又卡在 6%。实测这样改后
            st = max(st, self.step_grow)
            ema = max(ema, self.ema_grow)
        tw = (1 - ema) * bw + ema * mbox[2]
        th = (1 - ema) * bh + ema * mbox[3]
        tw = float(clamp(tw, bw * (1 - st), bw * (1 + st)))
        th = float(clamp(th, bh * (1 - st), bh * (1 + st)))
        tw = float(clamp(tw, self.min_box, 0.98 * 1e4))
        th = float(clamp(th, self.min_box, 0.98 * 1e4))
        # **绝对放大上限**：相对**用户选定的初始框**，不置信时最多放大 grow_hi 倍。 挡住"掩膜系统性偏大把框越推越大"（d.mp4 湿路面紫倒影实测推到 5.5 倍，模板跟着 烂掉 ->
        if self.init_area and not self._grow_ok:
            cap = self.grow_hi * self.init_area
            if marea > cap or tw * th > cap * 1.3:
                return None
        # **累计尺寸偏离上限（"防慢爬"关键一条）**：上面所有闸门都是**单帧**的相对闸门， 偏离不许超过 size_dev_cap（默认 25%）。真走近时 _grow_streak 会松开这条，
        if self.init_area and getattr(self, "_grow_streak", 0) < 2:
            dev = (tw * th) / self.init_area
            if dev > 1.0 + self.size_dev_cap or dev < 1.0 - self.size_dev_cap:
                return None
        return np.array([mbox[0], mbox[1], tw, th], np.float32)

    def llr_of(self, hsv, box):
        if self.fg is None:
            return 0.0
        x, y, w, h = [int(round(v)) for v in box]
        H, W = hsv.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return 0.0
        sub = hsv[y0:y1, x0:x1]
        return float(np.mean(self.llr_map(sub)))

    def _box_llr_mean(self, bgr, box):
        """框内平均 LLR（目标似然）。用于**改尺寸前的颜色核对**。
        """
        if self.fg is None:
            return 0.0
        H, W = bgr.shape[:2]
        x0 = int(clamp(box[0], 0, W - 2))
        y0 = int(clamp(box[1], 0, H - 2))
        x1 = int(clamp(box[0] + box[2], x0 + 2, W))
        y1 = int(clamp(box[1] + box[3], y0 + 2, H))
        sub = bgr[y0:y1, x0:x1]
        if sub.size == 0 or sub.shape[0] < 3 or sub.shape[1] < 3:
            return 0.0
        return float(np.mean(self.llr_map(cv2.cvtColor(sub, cv2.COLOR_BGR2HSV))))

    def build_recipe(self, bgr, box):
        """
        统计框内**占比最大的前 K 种颜色**及其相对占比，存成"颜色配方"。
        """
        if self.fg is None:
            return
        H, W = bgr.shape[:2]
        x0 = int(clamp(box[0], 0, W - 2))
        y0 = int(clamp(box[1], 0, H - 2))
        x1 = int(clamp(box[0] + box[2], x0 + 2, W))
        y1 = int(clamp(box[1] + box[3], y0 + 2, H))
        sub = bgr[y0:y1, x0:x1]
        if sub.size == 0 or sub.shape[0] < 3 or sub.shape[1] < 3:
            return
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1]
        hb = (hsv[..., 0].astype(np.int32) >> 4).ravel()   # 只留 H 桶（0..HS_H-1）
        m = (sat.ravel() >= 40)              # 只统计有颜色的像素（S≥40）
        if int(m.sum()) < 16:
            self.color_recipe = None         # 无彩色像素 -> 无配方
            return
        hist = np.bincount(hb[m], minlength=self.HS_H).astype(np.float64)
        total = float(hist.sum()) + 1e-9
        merged = {}                              # 代表桶 -> 权重
        for c in np.argsort(hist)[::-1]:
            if hist[c] <= 0:
                continue
            w = float(hist[c]) / total
            if w < 0.05:
                continue                         # 零星色（<5%）丢弃，避免噪声进配方
            for rep in list(merged):
                if min((c - rep) % self.HS_H, (rep - c) % self.HS_H) <= 1:
                    merged[rep] += w             # 与已有代表色相邻 -> 并入
                    break
            else:
                merged[c] = w                    # 新的独立颜色
        rec = sorted(merged.items(), key=lambda kv: -kv[1])[:self.recipe_k]
        rec = [(int(b), float(w)) for b, w in rec if w >= 0.08]
        self.color_recipe = rec if rec else None

    def recipe_sim(self, hsv, box):
        """候选框的**颜色构成**与存储配方的相似度（0..1）。
            做法：算候选框内（只取有颜色像素）的颜色直方图，取前 K 种颜色，
            用直方图**交集**相似度 Σ min(a_i, b_i) 比对配方 —— 对"紫多黄少"这种
        """
        if self.color_recipe is None:
            return None, 0
        H, W = hsv.shape[:2]
        x0 = int(clamp(box[0], 0, W - 2))
        y0 = int(clamp(box[1], 0, H - 2))
        x1 = int(clamp(box[0] + box[2], x0 + 2, W))
        y1 = int(clamp(box[1] + box[3], y0 + 2, H))
        sub = hsv[y0:y1, x0:x1]
        if sub.size == 0 or sub.shape[0] < 3 or sub.shape[1] < 3:
            return None, 0
        sat = sub[..., 1]
        hb = (sub[..., 0].astype(np.int32) >> 4).ravel()   # H 桶
        m = (sat.ravel() >= 40)
        n = int(m.sum())
        if n < 16:
            return None, n
        hist = np.bincount(hb[m], minlength=self.HS_H).astype(np.float64)
        total = float(hist.sum()) + 1e-9
        hp = hist / total
        acc = np.zeros(len(self.color_recipe))
        used = np.zeros(self.HS_H, bool)
        for c in range(self.HS_H):
            if hp[c] <= 0:
                continue
            d = [min((c - b) % self.HS_H, (b - c) % self.HS_H)
                 for b, _ in self.color_recipe]
            j = int(np.argmin(d))
            if d[j] <= 1:                       # 落在某个配方色 ±1 桶内
                acc[j] += float(hp[c])
                used[c] = True
        hit = float(np.sum([min(w, a) for (_, w), a in
                            zip(self.color_recipe, acc)]))
        dev = float(np.sum([abs(a - w) for (_, w), a in
                            zip(self.color_recipe, acc)]))
        dev += float(hp[~used].sum())
        ratio_fit = max(0.0, 1.0 - 0.6 * dev)
        return float(hit * ratio_fit), n

    def init(self, gray, bgr, box):
        tmpl = self._grab(gray, box)
        if tmpl is None:
            return False
        self.box = np.array(box, np.float32)
        self.init_area = float(box[2] * box[3])   # 用户选定框的面积 = 绝对放大上限的锚
        # **用户选定框的"几何锚"（不可漂移）**：这是 _fit_size 的绝对长宽比闸门基准。 为什么必须单独记：原来只用 ref_aspect，而 ref_aspect 每帧都从**当前框**学
        self.init_w, self.init_h = float(box[2]), float(box[3])
        self.init_aspect = float(box[2]) / max(1.0, float(box[3]))
        self.llr_init = None
        self.template = self._prep(tmpl)
        self.vel[:] = 0.0
        self.score, self.psr = 1.0, 99.0
        self.missing, self._cand_hits, self._cand_box = 0, 0, None
        self.state, self.lost_since = self.STATE_TRACK, None
        self.fg = self.bg = None
        self._update_model(bgr, self.box)
        self.llr_init = self._box_llr_mean(bgr, self.box)
        self.build_recipe(bgr, self.box)     # 存目标的"颜色配方"（占比最大的前 K 色）
        self.prev_gray, self.prev_box = gray.copy(), self.box.copy()
        self._last_border_edge = 1.0   # 首帧无历史，默认放行（真值框由用户保证）
        self._grow_streak = 0
        self._last_peak = None          # 上一帧被接受的峰值（时间稳定性判据）
        self._wt_prev = None
        self._view_wh = (float(gray.shape[1]), float(gray.shape[0]))
        return True

    def set_view_angles(self, pan=None, tilt=None, wh=None, ppd=None):
        """告知跟踪器：**本帧 view 画面**对应的云台绝对角（度）与画面尺寸。
            必须每帧调（主循环在 cam.render 之后）。有了它，update() 才能把上一帧
            框从"旧视角坐标系"换算到"本帧视角坐标系"—— 见 _warp_compensate。
        """
        if pan is None:
            self._view_ang = None
            self._wt_prev = None
            return
        if wh is not None:
            self._view_wh = (float(wh[0]), float(wh[1]))
        if ppd is not None:
            self._ppd = (float(ppd[0]), float(ppd[1]))
        self._view_ang = (float(pan), float(tilt if tilt is not None else 0.0))

    def _warp_compensate(self, ang_now):
        """
        把 self.box 从**上一帧 view 坐标系**换算到**本帧 view 坐标系**。
        """
        if ang_now is None or self._wt_prev is None or self.box is None:
            return
        if self._view_wh is None:
            return
        Wf, Hf = self._view_wh
        if self._ppd is not None:
            ppd_x, ppd_y = self._ppd
        else:
            ppd_x, ppd_y = Wf / 60.0, Hf / 36.0     # 兜底：常见 60°/36° 镜头
        dpan = float(ang_now[0]) - self._wt_prev[0]
        dtilt = float(ang_now[1]) - self._wt_prev[1]
        ddx = -dpan * ppd_x
        ddy = dtilt * ppd_y
        self.box[0] += ddx
        self.box[1] += ddy
        self._warp_dxy = (ddx, ddy)
        self._warp_dpan = dpan

    def _backward_ok(self, gray, box):
        """互一致性：把"当前帧候选框里的外观"当模板，回上一帧的原位置找一次。

        掺了背景的模板在这一步会露馅（上一帧那个位置还没有那块背景）。
        """
        if self.prev_gray is None or self.prev_box is None:
            return True
        tmpl = self._grab(gray, box, pad=True)
        if tmpl is None:
            return False
        pb = self.prev_box
        r = self._match(self.prev_gray, self._prep(tmpl),
                        (pb[0] + pb[2] / 2.0, pb[1] + pb[3] / 2.0), (pb[2], pb[3]), 0.4)
        return r is not None and r[1] >= self.bwd_score and r[2] >= self.bwd_psr

    def _global_shift(self, gray):
        """估计相对上一帧的**全局画面运动**（旋转角 + 平移），用于搜索中心预测与去旋转。
        """
        if not self.gm_comp or self.prev_gray is None:
            return 0.0, 0.0, 0.0
        g = gray
        if self.gm_scale < 0.999:
            g = cv2.resize(g, None, fx=self.gm_scale, fy=self.gm_scale,
                           interpolation=cv2.INTER_AREA)
        p = getattr(self, "_gm_prev", None)
        dx = dy = rot = 0.0
        if p is not None and p.shape == g.shape:
            if self._gm_win is None or self._gm_win.shape != g.shape:
                self._gm_win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
            try:
                (px, py), resp = cv2.phaseCorrelate(p.astype(np.float32),
                                                    g.astype(np.float32), self._gm_win)
                if np.isfinite(px) and np.isfinite(py) and resp >= self.gm_min_resp:
                    dx, dy = -px / self.gm_scale, -py / self.gm_scale
                rot = self._estimate_rotation(p, g)
            except cv2.error:
                pass
        self._gm_prev = g
        lim = self.gm_max
        return (float(np.clip(dx, -lim, lim)), float(np.clip(dy, -lim, lim)),
                float(np.clip(rot, -self.gm_rot_max, self.gm_rot_max)))

    def _estimate_rotation(self, prev_small, cur_small):
        """用 ORB 稀疏特征估全局旋转角（度）。降采样图上做，实测 ~2ms、精度 ~0.15°。"""
        if self._orb is None:
            self._orb = cv2.ORB_create(nfeatures=self.gm_orb_n)
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        kp0, des0 = getattr(self, "_gm_kp", (None, None))
        kp1, des1 = self._orb.detectAndCompute(cur_small, None)
        self._gm_kp = (kp1, des1)
        if des0 is None or des1 is None or len(kp0) < 8 or len(kp1) < 8:
            return 0.0
        try:
            ms = self._bf.match(des0, des1)
        except cv2.error:
            return 0.0
        if len(ms) < self.gm_orb_min:
            return 0.0
        p0 = np.float32([kp0[m.queryIdx].pt for m in ms]).reshape(-1, 1, 2)
        p1 = np.float32([kp1[m.trainIdx].pt for m in ms]).reshape(-1, 1, 2)
        M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0)
        if M is None or inl is None or int(inl.sum()) < self.gm_orb_min:
            return 0.0
        # cur 转过的角度"（实测验证： 真值 -2° 时 arctan2 给 +2.09°）。
        return float(-np.degrees(np.arctan2(M[1, 0], M[0, 0])))

    def update(self, gray, bgr, dt):
        """跟踪一帧，返回 (ok, box, score, state)"""
        if self.template is None or self.box is None:
            return False, None, 0.0, self.STATE_LOST
        if self.state == self.STATE_LOST:
            self.missing += 1
            return False, self.box.copy(), self.score, self.state
        self.src = "ncc"

        # ⓪ **云台自身运动的"前馈"补偿（修"画面动、框不动然后丢"）** 但**搜索窗必须按这个量前馈**，否则：一次 5° 的云台动作就让目标在画面里
        self._last_dt = float(dt)
        gpan_dx = gpdy = 0.0
        if self._view_ang is not None:
            if self._wt_prev is None:
                self._wt_prev = self._view_ang      # 首帧：只登记基准，不给前馈
            else:
                dpan = self._view_ang[0] - self._wt_prev[0]
                dtilt = self._view_ang[1] - self._wt_prev[1]
                if self._ppd is not None:
                    gpan_dx = -dpan * self._ppd[0]
                    gpdy = dtilt * self._ppd[1]
                    self._gpan_px = (gpan_dx, gpdy)
                self._wt_prev = self._view_ang

        # ① 单尺度定位（最便宜，先解决"在哪"） **关键：搜索窗中心要补偿"全局画面运动"**（云台转动/手持平移）。
        gdx, gdy, grot = self._global_shift(gray)
        gimbal_active = (gpan_dx != 0.0 or gpdy != 0.0)
        if gimbal_active:
            gdx = gdy = 0.0
            vpx = vpy = 0.0
        else:
            vpx, vpy = self.vel[0] * dt, self.vel[1] * dt
        cx = self.box[0] + self.box[2] / 2.0 + vpx + gdx + gpan_dx
        cy = self.box[1] + self.box[3] / 2.0 + vpy + gdy + gpdy
        search_ratio = self.search_ratio
        if self.missing > 0:
            search_ratio = self.search_ratio * (1.0 + self.search_grow * self.missing)
        r = self._match(gray, self.template, (cx, cy), (self.box[2], self.box[3]),
                        search_ratio, rot=grot)
        best = None if r is None else (r[1], r[2], r[0][0], r[0][1],
                                       self.box[2], self.box[3])
        self._tick = getattr(self, "_tick", 0) + 1
        do_size = (self._tick % self.size_every == 0)
        if do_size or getattr(self, "_ext_cache", None) is None:
            self._ext_cache = self.measure_extent(bgr, self.box) if self.fg is not None else None
        extent = self._ext_cache
        if best is None and self._tick % self.refine_every == 0:
            rr = self._match(gray, self.template, (cx, cy),
                             (self.box[2], self.box[3]), min(0.8, self.search_ratio * 2))
            if rr is not None:
                best = (rr[1], rr[2], rr[0][0], rr[0][1], self.box[2], self.box[3])

        if best is None:
            self.missing += 1
            if self.missing >= self.lost_after and self.state != self.STATE_LOST:
                self._mark_lost(getattr(self, "_now", 0.0))
            return False, self.box.copy(), self.score, self.state

        score, psr, px, py, tw, th = best
        # 只有"可接受"的匹配才允许更新位置/速度/尺度。分数太低说明这一帧根本没匹配上 但豁免必须**只对有纹理的模板**生效：平坦区/黑边上的归一化相关是退化的
        strong = score >= self.score_psr_free and self._tmpl_textured
        d_peak = math.hypot((px + tw / 2.0) - cx, (py + th / 2.0) - cy)
        pp = getattr(self, "_last_peak", None)
        peak_stable = (pp is None) or (math.hypot((px + tw / 2.0) - pp[0],
                                      (py + th / 2.0) - pp[1]) <= self.soft_jump * max(tw, th))
        # False，把本来稳定的匹配也拒掉 -> missing 一路累到 lost（实测 d.mp4 f141-145 峰值稳定在 639 却被判不稳）。
        self._last_peak = (px + tw / 2.0, py + th / 2.0)
        soft = (score >= self.score_soft and self._tmpl_textured
                and (d_peak <= self.soft_pull * max(tw, th) or peak_stable))
        keep_p = (score >= self.score_keep and psr >= self.psr_keep)
        slow = (score >= self.score_keep and peak_stable and self._tmpl_textured)
        accepted = strong or soft or keep_p or slow
        slow_move = slow and not (strong or soft or keep_p)   # 仅slow通道时降速跟随
        # **跳变闸门（修"框不跟手"的直接一刀）**：实测 d.mp4 f76 一帧内框从原位置 帧间真实目标的表观位移不会超过一个物理上限（经验：≤0.35·框尺寸 + 速度项），
        pred_c = np.array([cx, cy], np.float32)          # 搜索中心 = 上一帧框心 + 预测
        peak_c = np.array([px + tw / 2.0, py + th / 2.0], np.float32)
        jump = float(np.hypot(*(peak_c - pred_c)))
        jump_cap = self.jump_cap * max(tw, th)           # 允许的单帧最大跳变
        if jump > jump_cap and not strong:
            u = (peak_c - pred_c) / max(1e-6, jump)
            peak_c = pred_c + u * jump_cap
            px, py = peak_c[0] - tw / 2.0, peak_c[1] - th / 2.0
            score = score * self.jump_penalty
            self._last_peak = (peak_c[0], peak_c[1])
        if not accepted:
            self.missing += 1
            if self.missing >= self.lost_after and self.state != self.STATE_LOST:
                self._mark_lost(getattr(self, "_now", 0.0))
            return True, self.box.copy(), score, self.state
        new_c = np.array([px + tw / 2.0, py + th / 2.0], np.float32)
        old_c = self.box[:2] + self.box[3 - 1:] / 2.0
        if slow_move:
            new_c = old_c + self.slow_gain * (new_c - old_c)
        elif dt > 1e-4:
            dv = (new_c - old_c) / dt
            self.vel = self.vel + 0.5 * (dv - self.vel)
            # 速度闸门：坏的速度估计会同时污染搜索窗预测和速度前馈（实测出现过 +2.87 帧/秒 ≈2000px/s 的假速度）。真目标超过 0.8 帧/秒就没法用 NCC 跟了， 所以超过就截断。
            vlim = np.array([0.8 * gray.shape[1], 0.8 * gray.shape[0]], np.float32)
            self.vel = np.clip(self.vel, -vlim, vlim)

        old_h = float(self.box[3])
        self._last_border_edge = self.border_edge(bgr, self.box)
        fitted = self._fit_size(self.box, extent) if extent is not None else None
        if fitted is not None:
            # **边界贴合判据**：新框的边框边缘密度不能明显低于旧框（0.75 倍以内）， 否则说明新框涨进了平坦背景、或缩进了目标内部 —— 一律不采纳（冻结尺寸）。
            # 这条是独立判据，不依赖掩膜的面积结论，所以能真正断掉棘轮。
            be_cur = self._last_border_edge
            be_new = self.border_edge(bgr, fitted)
            # **颜色分布特征比对（修"目标莫名其妙丢失"的真正一刀）**： 判据必须是**绝对**的，不能是"新框 vs 当前框"：渐进 1px 漂移时新旧框内
            llr_new = self._box_llr_mean(bgr, fitted)
            if self.llr_init is not None and self.llr_init > 0.05:
                color_ok_size = llr_new >= self.llr_floor_frac * self.llr_init
            else:
                color_ok_size = True
            mcx, mcy = extent[3]
            pull_ok = (math.hypot(mcx - (px + tw / 2.0), mcy - (py + th / 2.0))
                       <= 0.5 * max(tw, th))
            pull = self.cent_pull if pull_ok else 0.0
            small = (abs(fitted[2] - self.box[2]) < 0.02 * self.box[2]
                     and abs(fitted[3] - self.box[3]) < 0.02 * self.box[3])
            if small or (be_new >= 0.75 * be_cur and color_ok_size):
                ncx = (1 - pull) * (px + tw / 2.0) + pull * mcx
                ncy = (1 - pull) * (py + th / 2.0) + pull * mcy
                self.box = np.array([ncx - fitted[2] / 2.0, ncy - fitted[3] / 2.0,
                                     fitted[2], fitted[3]], np.float32)
            else:
                self.box = np.array([px, py, self.box[2], old_h], np.float32)
        else:
            self.box = np.array([px, py, self.box[2], old_h], np.float32)
        Wf, Hf = float(gray.shape[1]), float(gray.shape[0])
        if self.valid is not None:
            vx0, vy0, vx1, vy1 = self.valid
        else:
            vx0, vy0, vx1, vy1 = 0.0, 0.0, Wf, Hf
        self.box[0] = float(clamp(self.box[0], vx0, max(vx0, vx1 - self.box[2])))
        self.box[1] = float(clamp(self.box[1], vy0, max(vy0, vy1 - self.box[3])))
        self.box[2] = float(min(self.box[2], vx1 - vx0))
        self.box[3] = float(min(self.box[3], vy1 - vy0))
        bw_, bh_ = float(self.box[2]), float(self.box[3])
        if bw_ > 1.0 and bh_ > 1.0:
            ar_cur = bw_ / bh_
            ar_lo = 0.75 * self.init_aspect
            ar_hi = 1.35 * self.init_aspect
            if ar_cur > ar_hi or ar_cur < ar_lo:
                ar_tgt = clamp(ar_cur, ar_lo, ar_hi)
                area_ = bw_ * bh_
                nbw = math.sqrt(area_ * ar_tgt)
                nbh = area_ / max(1e-6, nbw)
                ccx, ccy = self.box[0] + bw_ / 2.0, self.box[1] + bh_ / 2.0
                self.box[2], self.box[3] = float(nbw), float(nbh)
                self.box[0] = float(ccx - nbw / 2.0)
                self.box[1] = float(ccy - nbh / 2.0)
        self.score, self.psr = score, psr

        patch = self._grab(bgr, self.box, pad=True)
        llr_val = 0.0
        if patch is not None:
            llr_val = self.llr_of(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV),
                                  (0, 0, patch.shape[1], patch.shape[0]))
        near_edge = (self.box[0] < self.edge_margin * Wf or self.box[1] < self.edge_margin * Hf
                     or self.box[0] + self.box[2] > (1 - self.edge_margin) * Wf
                     or self.box[1] + self.box[3] > (1 - self.edge_margin) * Hf)
        strong2 = score >= self.score_psr_free and self._tmpl_textured
        # **confident 拆成两个用途**（修"目标走近变大后丢失"）：
        conf_score = strong2 or (score >= self.score_hi and psr >= self.psr_hi)
        conf_model = conf_score and not near_edge
        conf_size = conf_score
        confident = conf_model          # 兼容原有引用
        # missing 不减、12 帧后仍判 lost（实测 d.mp4 f135-145 就是这种情况）。
        keep = strong2 or soft or slow or (score >= self.score_keep and psr >= self.psr_keep)
        llr_ok = (not self.color_ok) or llr_val >= self.llr_min
        # **模板/模型更新的颜色守门（绝对基准）**：llr_min=0.02 太低，实测框滑到湿路面 **最终执行环节**。所以模板更新必须额外过一道**绝对**闸门：框内 LLR 不能低于
        if self.llr_init is not None and self.llr_init > 0.05:
            tmpl_color_ok = (llr_val >= self.llr_floor_frac * self.llr_init)
        else:
            tmpl_color_ok = True

        if keep:
            # 模板更新条件从 confident(分数>=0.70) 放宽到 keep(>=0.50)： 防污染由 _backward_ok（互一致性：当前外观必须能匹配回上一帧的位置）负责，
            self._bwd_tick = (self._bwd_tick + 1) % self.bwd_every
            bwd_ok = (self._bwd_tick != 0) or self._backward_ok(gray, self.box)
            if keep and llr_ok and tmpl_color_ok and patch is not None and bwd_ok:
                new = self._prep(self._grab(gray, self.box, pad=True))
                old_t = self.template
                if old_t.shape != new.shape:      # 尺寸变了先缩放旧模板再混合（保住历史外观）
                    old_t = cv2.resize(old_t, (new.shape[1], new.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
                self.template = 0.85 * old_t + 0.15 * new
            if llr_ok and tmpl_color_ok:
                # 但**锚点每帧都要更新**（它是丢失后搜索的圆心，不能滞后）。
                self.anchor_box = self.box.copy()
                if self._tick % self.model_every == 0:
                    self._update_model(bgr, self.box)
                    self.build_recipe(bgr, self.box)    # 配方同步刷新
                    # 颜色比例特征基准：只在**可接受且颜色可信**的帧以很慢的 EMA 更新 （真走近/走远跟得上，一次坏测量拉不动）。条件不能用 confident(0.7)： d.mp4 里分数长期在
                    self.ref_llr_q = self._box_llr_q(bgr, self.box, self.ref_q)
                    self._grow_ok = bool(conf_size)
                    ex = extent if extent is not None else None
                    if ex is not None:
                        # 基准由**被验证过的框**驱动（不是由掩膜测量驱动）：否则测量自身的 偏差会把基准一起拖走，棘轮只是变慢。框只有在通过边界贴合判据后才变， 所以用框的面积/长宽比当基准是稳的；fill/edge
                        a = self.box[2] * self.box[3]
                        asp = self.box[2] / max(1.0, self.box[3])
                        fl, ed = ex[1], ex[5]
                        marea = ex[2]
                        if self.ref_area is None:
                            self.ref_area, self.ref_aspect = a, asp
                            self.ref_fill, self.ref_edge = fl, ed
                        else:
                            if marea > self.ref_grow_ok * self.ref_area:
                                self._grow_streak = getattr(self, "_grow_streak", 0) + 1
                            else:
                                self._grow_streak = 0
                            if self._grow_streak >= 3:
                                k = self.ref_grow_lr        # 证据充分 -> 快速跟涨
                            else:
                                k = self.ref_lr             # 否则慢刹车（防棘轮）
                            self.ref_area = (1 - k) * self.ref_area + k * a
                            ka = 0.02
                            na = (1 - ka) * self.ref_aspect + ka * asp
                            self.ref_aspect = float(clamp(na,
                                                0.7 * self.init_aspect,
                                                1.4 * self.init_aspect))
                            self.ref_fill = (1 - k) * self.ref_fill + k * fl
                            self.ref_edge = (1 - k) * self.ref_edge + k * ed
            self.missing = max(0, self.missing - 1)
            self.state = self.STATE_TRACK
            self.lost_since = None
        else:
            self.missing += 1
            if self.missing >= self.lost_after and self.state != self.STATE_LOST:
                self._mark_lost(getattr(self, "_now", 0.0))
        self.prev_gray, self.prev_box = gray, self.box.copy()
        return True, self.box.copy(), score, self.state

    def search_region(self, bgr, gray, center, margin, now):
        """
        丢失后重捕获：在 center 周围找目标。要求连续 2 帧命中同一位置才确认。
        """
        if self.template is None or self.box is None:
            return None
        H, W = gray.shape[:2]
        bw, bh = int(round(self.box[2])), int(round(self.box[3]))
        rw = min(W, max(bw + 4, int(bw * (1 + 2 * margin))))
        rh = min(H, max(bh + 4, int(bh * (1 + 2 * margin))))
        x0 = int(clamp(center[0] - rw / 2.0, 0, W - rw))
        y0 = int(clamp(center[1] - rh / 2.0, 0, H - rh))

        tgt_ar = bw / float(bh)
        tgt_area = float(bw * bh)
        use_color = (self.fg is not None and self.color_ok
                     and self.color_recipe is not None)
        self.src = "recipe" if use_color else "ncc"
        best = None
        if use_color:
            sub_bgr = bgr[y0:y0 + rh, x0:x0 + rw]
            if sub_bgr.shape[0] < bh + 2 or sub_bgr.shape[1] < bw + 2:
                return None
            sub_hsv = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2HSV)
            # 点亮，湿路面/反光不会混入。±1 容差很关键——目标色常正好压在 H 档边界上 （如紫 H≈128 会同时落在桶 7/8），只认单桶会把目标切成两半。
            hb = (sub_hsv[..., 0].astype(np.int32) >> 4)
            sm = sub_hsv[..., 1]
            mm = np.zeros(hb.shape, np.uint8)
            for b, _w in self.color_recipe:
                for dh in (-1, 0, 1):
                    mm |= (hb == ((b + dh) % self.HS_H))
            mm &= (sm >= 40).astype(np.uint8)
            mm = cv2.dilate(mm, np.ones((5, 5), np.uint8), iterations=1)
            ncc, lab, stats, cent = cv2.connectedComponentsWithStats(mm, 8)
            cands = []                              # (sim, n, angle, box)
            area_cap = 0.5 * rw * rh                # 占搜索区一半以上 = 背景，不是物体
            for i in range(1, ncc):
                lx, ly, lw, lh, area = stats[i]
                if area < max(0.06 * tgt_area, 24):
                    continue                        # 太小的碎片 -> 噪声
                if area > area_cap:
                    continue                        # 大得离谱 -> 背景连成一片，丢弃
                ar = lw / float(max(1, lh))
                if not (tgt_ar / 2.6 <= ar <= tgt_ar * 2.6):
                    continue                        # 形状离目标太远
                if not (tgt_area / 16.0 <= lw * lh <= tgt_area * 16.0):
                    continue                        # 尺度离目标太远（±4 倍边长）
                sim, ncol = self.recipe_sim(sub_hsv, (lx, ly, lw, lh))
                if sim is None or ncol < 0.1 * lw * lh:
                    continue                        # 彩色像素太少 -> 不可信
                ang = 0.0 if (cent[i][0] is None) else \
                    np.hypot(cent[i][0] - 0.5 * rw, cent[i][1] - 0.5 * rh)
                cands.append((float(sim), int(ncol), float(ang),
                              (float(lx), float(ly), float(lw), float(lh))))
            if cands:
                cands.sort(key=lambda t: (-t[0], t[2]))   # 先相似度，再离中心近
                sim, ncol, ang, (lx, ly, lw, lh) = cands[0]
                if sim >= self.recipe_min_sim:
                    pad = max(2, int(0.10 * min(lw, lh)))
                    qx0 = max(0, int(lx) - pad)
                    qy0 = max(0, int(ly) - pad)
                    qx1 = min(rw, int(lx + lw) + pad)
                    qy1 = min(rh, int(ly + lh) + pad)
                    ring = sub_hsv[qy0:qy1, qx0:qx1]
                    rs = ring[..., 1] >= 40
                    border = np.zeros(rs.shape, bool)
                    border[:pad, :] = True
                    border[-pad:, :] = True
                    border[:, :pad] = True
                    border[:, -pad:] = True
                    leak = float(rs[border].mean()) if border.any() else 1.0
                    if leak <= self.recipe_leak_max:
                        bx, by = x0 + lx, y0 + ly
                        best = (sim, sim, float(bx), float(by),
                                float(lw), float(lh), 1)
        if best is None and not use_color:
            sub = gray[y0:y0 + rh, x0:x0 + rw]
            if sub.shape[0] < bh + 2 or sub.shape[1] < bw + 2:
                return None
            res = cv2.matchTemplate(sub.astype(np.float32),
                                    self._prep(self.template), cv2.TM_CCOEFF_NORMED)
            _, mx, _, mloc = cv2.minMaxLoc(res)
            if mx >= self.score_keep:
                bx = x0 + mloc[0]
                by = y0 + mloc[1]
                best = (mx, mx, float(bx), float(by), float(bw), float(bh), 2)
        if best is None:
            self._cand_hits = 0
            return None

        box = np.array([best[2], best[3], best[4], best[5]], np.float32)
        # ④ 连续命中确认：候选位置必须基本不动（防止把两个东西连起来当一次命中）
        if self._cand_box is not None and \
                np.hypot(box[0] - self._cand_box[0], box[1] - self._cand_box[1]) \
                > 0.5 * max(box[2], box[3]):
            self._cand_hits = 0
        self._cand_box = box
        self._cand_hits += 1
        need = 1 if best[0] >= self.recipe_confirm_hi else 2
        if self._cand_hits < need:
            return None
        self.box = box
        patch = self._grab(gray, box, pad=True)
        if patch is None:
            self._cand_hits = 0
            return None
        self.template = self._prep(patch)
        self.vel[:] = 0.0
        self.score = float(best[1])
        self.psr = 0.0
        self.missing, self._cand_hits, self._cand_box = 0, 0, None
        self.state, self.lost_since, self.lost_box = self.STATE_TRACK, None, None
        self.prev_gray, self.prev_box = gray.copy(), box.copy()
        self._update_model(bgr, box)
        self.n_recover += 1
        return box.copy(), self.score, self.psr

    def _mark_lost(self, now):
        """进入丢失状态，并记住丢失时的目标框（重捕获以它为中心）。"""
        self.lost_since = float(now)
        self.lost_box = None if self.box is None else np.array(self.box, np.float32)
        self.state = self.STATE_LOST

    def region_margin(self, lost_sec):
        """丢失后搜索域：**固定策略**——云台不动，以丢失时目标框中心为中心，
            长宽各扩大 2 倍（margin=1.0，即搜索窗面积 = 目标框的 4 倍）。
            不再按丢失时长分层升级：4 倍面积一次搜索仅数毫秒，逐级放大只会让目标
        """
        return self.search_margin, 1

# 三、控制器：像素误差 -> 角速度 -> 角度指令

def segment_at(bgr, px, py, seed_frac=0.06, win_frac=0.28, iters=4):
    """【点击选物】点一下 (px,py)，自动找出该点所在物体的**边界**并返回外接框。
    返回 (x, y, w, h) 或 None。
    """
    H, W = bgr.shape[:2]
    px, py = int(round(px)), int(round(py))
    if not (0 <= px < W and 0 <= py < H):
        return None
    rw = max(64, int(round(min(H, W) * win_frac)))
    x0 = int(clamp(px - rw // 2, 0, max(0, W - rw)))
    y0 = int(clamp(py - rw // 2, 0, max(0, H - rw)))
    x1, y1 = min(W, x0 + rw), min(H, y0 + rw)
    sub = bgr[y0:y1, x0:x1]
    if sub.shape[0] < 24 or sub.shape[1] < 24:
        return None
    lx, ly = px - x0, py - y0                    # 点击点在子窗里的坐标

    box = _grow_by_color(sub, lx, ly, seed_frac)
    if box is not None:
        box = _snap_to_edges(sub, box)
        gx, gy, gw, gh = box
        area = gw * gh
        # 尺度先验：太小（碎片）或"铺满整个搜索窗且毫无边界"（同色背景成片）都拒绝。 注意：物体本来就可能比搜索窗大（近景伞），所以"占满窗口"只在**边缘吸附后 仍无一条边落在强边缘上**时才判失败 ——
        if 0.004 * rw * rw <= area <= 0.985 * rw * rw and gw >= 10 and gh >= 10:
            return (float(gx + x0), float(gy + y0), float(gw), float(gh))

    best = _grabcut_box(sub, lx, ly, iters)
    if best is not None:
        return (float(best[0] + x0), float(best[1] + y0), float(best[2]), float(best[3]))

    seed = (float(clamp(px - rw * 0.12, 0, W - 1)), float(clamp(py - rw * 0.12, 0, H - 1)),
            float(max(12, rw * 0.24)), float(max(12, rw * 0.24)))
    tr = TargetTracker()
    if not tr.init(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr, seed):
        return None
    ex = tr.measure_extent(bgr, tr.box)
    if ex is None:
        return None
    bx = ex[0]
    if not (bx[0] <= px <= bx[0] + bx[2] and bx[1] <= py <= bx[1] + bx[3]):
        return None
    return bx

def _grow_by_color(sub, lx, ly, seed_frac=0.06, d_h=18, d_s=45, d_v=110):
    """
    局部颜色区域生长：返回含点击点的连通域外接框 (gx,gy,gw,gh) 或 None。
    """
    h, w = sub.shape[:2]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV).astype(np.int32)
    s = max(2, int(round(min(h, w) * seed_frac)))
    sy0, sy1 = max(0, ly - s), min(h, ly + s + 1)
    sx0, sx1 = max(0, lx - s), min(w, lx + s + 1)
    seed = hsv[sy0:sy1, sx0:sx1].reshape(-1, 3)
    if seed.size == 0:
        return None
    med = np.median(seed, axis=0)
    # **用 MAD（中位绝对偏差）代替 std 估散布**：std 会被种子块里的少数背景/高光像素 拉大，阈值随之被撑爆。d.mp4 紫伞实测：种子块 25×25 里混进几粒背景，std_V=21.4
    mad = np.median(np.abs(seed - med), axis=0) * 1.4826

    def _blob(v_cap, s_cap):
        """
        给定 S/V 阈值上限，跑一遍"颜色掩膜 -> 形态学 -> 取含点击点的连通域"，返回
        """
        th = max(d_h, min(50, 3.0 * float(mad[0]) + 10))
        ts = max(d_s, min(s_cap, 3.0 * float(mad[1]) + 25))
        tv = min(v_cap, 3.0 * float(mad[2]) + 45)
        dh = np.abs(hsv[..., 0] - int(med[0]))
        dh = np.minimum(dh, 180 - dh)               # H 环向距离
        ds = np.abs(hsv[..., 1] - int(med[1]))
        dv = np.abs(hsv[..., 2] - int(med[2]))
        # S 低（灰/白）的像素 H 无意义 -> 只按 S/V 判；否则 H/S 为主，V 仅做宽松上界
        if med[1] < 32:
            mk = ((ds <= ts) & (dv <= tv)).astype(np.uint8) * 255
        else:
            mk = ((dh <= th) & (ds <= ts) & (dv <= tv)).astype(np.uint8) * 255
        k = max(3, int(round(0.02 * min(h, w))) | 1)
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, el)
        ck = max(3, int(round(0.05 * min(h, w))) | 1)
        mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))
        ff = mk.copy()
        ffm = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, ffm, (0, 0), 255)
        mk = cv2.bitwise_or(mk, cv2.bitwise_not(ff))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mk, 8)
        if n <= 1:
            return None
        lid = int(lab[ly, lx])
        if lid == 0:                                # 点击点不在掩膜里 -> 取最近质心的连通域
            d = [np.hypot(cent[j][0] - lx, cent[j][1] - ly) if j else 1e9 for j in range(n)]
            lid = int(np.argmin(d))
            if lid == 0 or d[lid] > 0.35 * min(h, w):
                return None
        gx, gy, gw, gh = (int(stats[lid, 0]), int(stats[lid, 1]),
                          int(stats[lid, 2]), int(stats[lid, 3]))
        area = float(stats[lid, 4])
        return (gx, gy, gw, gh, area / float(max(1, gw * gh)), lab == lid)

    r = _blob(160, 90)
    # **整窗泄漏闸门 -> 收紧 V 重试**：区域生长把画面里一大片（含树叶/路面）连成一块时， 外接框会几乎铺满整个搜索窗（实测：点伞阴面时种子紫灰 ≈ 潮湿路面灰，漏到 183×202， 填充
    if r is not None and (r[2] >= 0.92 * w or r[3] >= 0.92 * h):
        r2 = _blob(48, 60)
        if r2 is not None and r2[2] < 0.92 * w and r2[3] < 0.92 * h:
            r = r2
        else:
            return None
    if r is None:
        return None
    gx, gy, gw, gh, fill, sub_lab = r
    if gh > 1.15 * gw and fill < 0.72:
        sub_lab = sub_lab.astype(np.uint8)
        rows = sub_lab[gy:gy + gh, gx:gx + gw].sum(axis=1).astype(np.float32)
        main_w = float(np.median(rows[rows >= 0.5 * rows.max()])) or 1.0
        keep = rows >= 0.55 * main_w          # 逐行：够宽才算"主体"
        cyr = int(clamp(ly - gy, 0, gh - 1))
        top = cyr
        miss = 0
        while top > 0:
            if keep[top - 1]:
                top -= 1; miss = 0
            else:
                miss += 1
                if miss >= 2:
                    break
                top -= 1
        bot = cyr
        miss = 0
        while bot < gh - 1:
            if keep[bot + 1]:
                bot += 1; miss = 0
            else:
                miss += 1
                if miss >= 2:
                    break
                bot += 1
        if bot - top + 1 >= 8:
            gy, gh = gy + top, bot - top + 1
            band = sub_lab[gy:gy + gh, gx:gx + gw].sum(axis=0)
            cols = np.where(band > 0)[0]
            if cols.size:
                gx, gw = gx + int(cols[0]), int(cols[-1] - cols[0] + 1)
    return (gx, gy, gw, gh)

def _snap_to_edges(sub, box, snap=12):
    """
    把外接框的每条边向附近最强 Sobel 边缘吸附，让框贴合物体真实轮廓。
    """
    gx, gy, gw, gh = box
    h, w = sub.shape[:2]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ex = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    ey = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    def snap_axis(profile, cur, lo, hi):
        """profile: 1D 边缘强度；在 cur±snap 内找最强的位置（带轻微向 cur 的偏好）"""
        a = max(lo, cur - snap)
        b = min(hi, cur + snap)
        if b - a < 2:
            return cur
        seg = profile[a:b]
        if seg.max() <= 1e-6:
            return cur
        idx = int(np.argmax(seg))
        # 只接受"明显强于窗口内均值"的峰，否则保持不动（防止被噪声拽走）
        if seg[idx] < 1.6 * (seg.mean() + 1e-6):
            return cur
        return a + idx

    x_lo = max(0, gx - snap)
    left = snap_axis(ex[max(0, gy):min(h, gy + gh), :].mean(axis=0), gx, 0, w - 1)
    right = snap_axis(ex[max(0, gy):min(h, gy + gh), :].mean(axis=0), gx + gw - 1, 0, w - 1)
    coly = ey[:, max(0, gx):min(w, gx + gw)].mean(axis=1)
    top = snap_axis(coly, gy, 0, h - 1)
    bot = snap_axis(coly, gy + gh - 1, 0, h - 1)
    nx, ny = min(left, right), min(top, bot)
    nw, nh = abs(right - left) + 1, abs(bot - top) + 1
    if nw < 8 or nh < 8:
        return box
    return (int(nx), int(ny), int(nw), int(nh))

def _grabcut_box(sub, lx, ly, iters=4, seed_frac=0.06):
    """GrabCut 两段式（回退路径）：返回子窗坐标外接框或 None。
    子窗本来就 ≤~200px，再小反而丢细节。求得框后按 2x 映射回原坐标。
    """
    h0, w0 = sub.shape[:2]
    SC = 0.3 if min(h0, w0) >= 160 else (0.4 if min(h0, w0) >= 96 else 1.0)
    if SC != 1.0:
        sub = cv2.resize(sub, (max(24, int(round(w0 * SC))), max(24, int(round(h0 * SC)))),
                         interpolation=cv2.INTER_AREA)
    h, w = sub.shape[:2]
    lx, ly = int(round(lx * SC)), int(round(ly * SC))
    try:
        m = np.zeros((h, w), np.uint8)
        rect = (int(0.06 * w), int(0.06 * h), int(0.88 * w), int(0.88 * h))
        bgm = np.zeros((1, 65), np.float64)
        fgm = np.zeros((1, 65), np.float64)
        cv2.grabCut(sub, m, rect, bgm, fgm, 1, cv2.GC_INIT_WITH_RECT)
        m2 = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        s = max(3, int(round(min(h, w) * seed_frac)))
        cy0, cy1 = max(0, ly - s), min(h, ly + s)
        cx0, cx1 = max(0, lx - s), min(w, lx + s)
        full = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        full[m2 > 0] = cv2.GC_PR_FGD
        full[:, :3] = full[:, -3:] = full[:3, :] = full[-3:, :] = cv2.GC_BGD
        full[cy0:cy1, cx0:cx1] = cv2.GC_FGD
        # mask 初始化已有一次 RECT 的结果打底，再迭代 1 次即收敛：GrabCut 在大面积 同色区域上耗费随迭代次数非线性增长（4 次 ≈ 3 倍），实测 1 次效果与 4 次一致。
        cv2.grabCut(sub, full, None, bgm, fgm, 1, cv2.GC_INIT_WITH_MASK)
        fg = np.where((full == cv2.GC_FGD) | (full == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        k = max(3, int(round(0.02 * min(h, w))) | 1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1,) * 2))
        n, lab, stats, _c = cv2.connectedComponentsWithStats(fg, 8)
        if n <= 1:
            return None
        lid = int(lab[ly, lx]) if 0 <= ly < h and 0 <= lx < w else 0
        if lid == 0:
            return None
        sx, sy, sw, sh = (int(stats[lid, 0]), int(stats[lid, 1]),
                          int(stats[lid, 2]), int(stats[lid, 3]))
        frac = float(stats[lid, 4]) / float(h * w)
        if 0.02 <= frac <= 0.92 and sw >= 8 and sh >= 8:
            return (sx / SC, sy / SC, sw / SC, sh / SC)
    except cv2.error:
        return None
    return None

# 七、绘制

def run(args):
    src = FrameSource(args.video[0] if args.video else args.camera,
                      is_camera=not args.video, width=args.width, height=args.height,
                      fps_cap=args.fps_cap)
    if not src.open():
        return 1
    ok, frame = src.read()
    if not ok:
        print("读帧失败")
        return 1
    H, W = frame.shape[:2]

    cam = make_camera(args, W, H)
    rig = isinstance(cam, SimCamera) and not src.is_camera
    two_panel = rig
    tracker = TargetTracker(score_keep=args.score_keep, lost_after=args.lost_after,
                            llr_min=args.llr_min)
    # 视频在环时**默认关掉速度前馈**：前馈要算"目标在世界里的角速度 = 云台实测角速度 +
    kff = (0.0 if rig else 1.0) if args.kff is None else float(args.kff)
    ctrl = FollowController(hfov=args.hfov, vfov=args.vfov, kp=args.kp, kd=args.kd,
                            kff=kff, deadband=args.deadband, max_rate=args.max_rate,
                            lp_tau=args.lp_tau, ang_limit=args.ang_limit)
    if rig and args.kff is None:
        print("视频在环：速度前馈默认关闭（视频自身的相机运动测不到，前馈会追着它跑）；"
              "要强制打开用 --kff 1")

    gstate = None if args.no_state else GimbalState(args.gimbal_state, args.ang_limit,
                                                    W / ctrl.hfov)
    if gstate is not None and gstate.data.get("loaded"):
        ctrl.pan, ctrl.tilt = float(gstate.data["pan"]), float(gstate.data["tilt"])
        if isinstance(cam, SimCamera):
            cam.ang[:] = (ctrl.pan, ctrl.tilt)
        print("已恢复云台角度 pan=%+.1f° tilt=%+.1f°" % (ctrl.pan, ctrl.tilt))

    cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(MAIN_WIN, min(W * (2 if two_panel else 1), 1500), min(H, 900))

    ui = {"drag": None, "box": None, "paused": False, "debug": True,
          "drive": not args.no_cam, "recenter": not args.no_start_recenter,
          "pending_click": None}
    anchor_bearing = [0.0, 0.0]          # 最后一次确信看到目标的世界方位
    anchor_box = None

    def on_mouse(event, x, y, flags, param):
        off = (W + GAP) if two_panel else 0
        if two_panel and x < off:
            return                        # 左面板是"世界"，点了也会对错坐标
        x -= off
        if event == cv2.EVENT_LBUTTONDOWN:
            ui["drag"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and ui["drag"] is not None:
            x0, y0 = ui["drag"]
            ui["box"] = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
        elif event == cv2.EVENT_LBUTTONUP:
            if ui["box"] is not None and ui["box"][2] > 8 and ui["box"][3] > 8:
                ui["pending_init"] = ui["box"]
            elif ui["drag"] is not None and abs(x - ui["drag"][0]) <= 4 \
                    and abs(y - ui["drag"][1]) <= 4:
                ui["pending_click"] = (x, y)
            ui["drag"] = ui["box"] = None

    try:
        cv2.setMouseCallback(MAIN_WIN, on_mouse)
    except cv2.error:
        pass

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if args.init:
        tracker.init(gray, frame, [float(v) for v in args.init])
    else:
        print("提示：用鼠标在窗口里拖一个框选中目标（或下次用 --init x y w h）")

    print("""attempt9 目标跟随（静态等待模式 + 点击选物）
  鼠标**点一下**=自动识别该点所在物体并框选   鼠标**拖框**=手动框选
  r 重选+回正  c 回正  空格 暂停  p 重播  b 云台开关  d 叠加  s 存图  q 退出
  控制器 kp=%.2f kd=%.2f kff=%.2f 死区=%.1f%% 限速=%.0f°/s 低通=%.2fs
  云台 %s 软限位 ±%.0f°  HFOV=%.0f°(%.2f px/°)  丢失策略: 云台不动 + 颜色比例特征搜索(框心外扩2倍/面积4倍, 每帧)"""
          % (ctrl.kp, ctrl.kd, ctrl.kff, ctrl.deadband * 100, ctrl.max_rate, ctrl.lp_tau,
             type(cam).__name__, args.ang_limit, ctrl.hfov, W / ctrl.hfov))

    fps_hist = deque(maxlen=30)
    t_prev = time.perf_counter()
    t_frame = 0
    err = np.zeros(2)
    prev_cam = (0.0, 0.0)
    frame_i = 0
    win_gone = 0                          # 关窗确认计数（连续多少次读不到窗口）

    while True:
        t_frame += 1
        t_now = time.perf_counter()
        dt = max(1e-3, t_now - t_prev)
        t_prev = t_now
        fps_hist.append(1.0 / dt)
        tracker._now = t_now

        if ui.get("replay"):
            ui["replay"] = False
            if src.is_camera:
                print("摄像头输入无法重播")
            elif src.rewind():
                ctrl.reset()
                tracker.vel[:] = 0.0
                ui["relock"] = True
                ui["recenter"] = True
                print("已回到视频开头，云台平滑回正")

        if ui["paused"]:
            ok = True
        elif t_frame > 1:
            ok, frame = src.read()
            if not ok and args.loop and not src.is_camera and src.rewind():
                ok, frame = src.read()
            if not ok:
                break
            if frame.shape[1] != W or frame.shape[0] != H:
                W, H = frame.shape[1], frame.shape[0]
                cam.W, cam.H = W, H
                cam.ppd_x, cam.ppd_y = W / ctrl.hfov, H / ctrl.vfov
        else:
            ok = True
        if not ui["paused"]:
            frame_i += 1

        # 云台转动 -> 画面变化：以**当前帧**为取景源（清晰、实时），云台角度直接平移画面， 露出画面外的地方填黑。为什么不做拼接图取景：单帧拼接在手持/走动视频上有视差，
        if isinstance(cam, SimCamera):
            view = cam.render(frame)
            tracker.set_valid_rect(cam.valid_rect())
            # 搜索窗**前馈**，否则一次云台动作就把目标推出搜索半径（见 update 的 ⓪）。
            tracker.set_view_angles(cam.ang[0], cam.ang[1], (W, H),
                                    (cam.ppd_x, cam.ppd_y))
        else:
            view = frame
            tracker.set_valid_rect(None)
            tracker.set_view_angles(None)
        if view.shape[1] != W or view.shape[0] != H:
            view = cv2.resize(view, (W, H))
        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)

        # 每帧只挪一下就停 -> 按 r 等于没反应（踩过这个坑，现在有整机自检）
        recentering = bool(ui["recenter"]) and not ui["paused"]
        if recentering:
            if max(abs(ctrl.pan), abs(ctrl.tilt)) < args.recenter_tol:
                ui["recenter"] = False
                ctrl.pan = ctrl.tilt = 0.0
                ctrl.reset()
                recentering = False
                print("已回正（pan/tilt = 0°）")
            else:
                ctrl.slew_to(0.0, 0.0, dt, rate=args.recenter_rate, tol=args.recenter_tol)

        if ui.pop("relock", False) and tracker.template is not None:
            rc = (W / 2.0, H / 2.0) if tracker.box is None else \
                (tracker.box[0] + tracker.box[2] / 2.0, tracker.box[1] + tracker.box[3] / 2.0)
            if tracker.search_region(view, gray, rc, tracker.search_margin, t_now):
                print("重播后已用颜色模型重新锁定目标")
            else:
                print("重播后没找到目标：鼠标拖框重选")

        if ui.get("pending_init") is not None:
            b = ui.pop("pending_init")
            if tracker.init(gray, view, list(b)):
                ctrl.reset()
                print("已锁定目标框", [int(v) for v in b])
        if ui.get("pending_click") is not None:
            cxp, cyp = ui.pop("pending_click")
            t_seg = time.perf_counter()
            sb = segment_at(view, cxp, cyp)
            ms_seg = (time.perf_counter() - t_seg) * 1000.0
            if sb is not None and tracker.init(gray, view, list(sb)):
                ctrl.reset()
                print("点击选物 (%.0f,%.0f) -> 边界框 [%d,%d,%d,%d]（%.0f ms）"
                      % (cxp, cyp, sb[0], sb[1], sb[2], sb[3], ms_seg))
            else:
                print("点击选物失败（该点附近没有可分割的物体），请改用拖框")
        if ui.get("want_reset"):
            ui.pop("want_reset")
            tracker.template = None
            tracker.box = None
            print("已清除目标，请重新拖框")

        ok_tr, box, score, state = False, tracker.box, tracker.score, tracker.state
        search_rect = None
        if not ui["paused"] and not recentering and tracker.template is not None:
            try:
                ok_tr, box, score, state = tracker.update(gray, view, dt)
            except Exception as exc:
                print("跟踪器异常（已忽略本帧）：%s: %s" % (type(exc).__name__, exc))
            if state == TargetTracker.STATE_LOST:
                margin = tracker.search_margin
                b0 = tracker.lost_box if tracker.lost_box is not None else tracker.box
                if b0 is not None:
                    bw, bh = float(b0[2]), float(b0[3])
                    rw = min(W, max(bw + 4, bw * (1 + 2 * margin)))
                    rh = min(H, max(bh + 4, bh * (1 + 2 * margin)))
                    ccx = b0[0] + bw / 2.0
                    ccy = b0[1] + bh / 2.0
                    sx = int(clamp(ccx - rw / 2.0, 0, W - rw))
                    sy = int(clamp(ccy - rh / 2.0, 0, H - rh))
                    search_rect = (sx, sy, rw, rh)
                    try:
                        got = tracker.search_region(view, gray, (ccx, ccy), margin, t_now)
                    except Exception as exc:
                        got = None
                        print("区域搜索异常（已忽略）：%s: %s" % (type(exc).__name__, exc))
                    if got is not None:
                        box, score, state = got[0], got[1], TargetTracker.STATE_TRACK
            if state == TargetTracker.STATE_TRACK and tracker.anchor_box is not None:
                ab = tracker.anchor_box
                anchor_bearing[:] = ctrl.bearing_of(
                    ((ab[0] + ab[2] / 2.0 - W / 2.0) / W,
                     (ab[1] + ab[3] / 2.0 - H / 2.0) / H))
                anchor_box = ab.copy()

        hold = (tracker.template is None or state == TargetTracker.STATE_LOST
                or ui["paused"] or recentering)
        if box is not None:
            e = np.array([(box[0] + box[2] / 2.0 - W / 2.0) / W,
                          (box[1] + box[3] / 2.0 - H / 2.0) / H], np.float32)
            e_vel = np.array([tracker.vel[0] / W, tracker.vel[1] / H], np.float32)
        else:
            e = np.zeros(2, np.float32)
            e_vel = np.zeros(2, np.float32)
        err = e
        st_cam = cam.state()
        cam_rate = ((st_cam["pan"] - prev_cam[0]) / dt, (st_cam["tilt"] - prev_cam[1]) / dt)
        prev_cam = (st_cam["pan"], st_cam["tilt"])
        if ui["paused"] or recentering:
            info = dict(ctrl.info)
            info["hold"] = True
            pan, tilt = ctrl.pan, ctrl.tilt
        else:
            pan, tilt, info = ctrl.update(e, e_vel, dt, hold=hold, cam_rate=cam_rate)
        if ui["drive"]:
            cam.command(pan, tilt, dt)
        if gstate is not None:
            gstate.update(ctrl.pan, ctrl.tilt)

        if two_panel:
            canvas = np.zeros((H, W * 2 + GAP, 3), np.uint8)
            canvas[:, :W] = frame
            canvas[:, W + GAP:] = view
            ovl_x = W + GAP
        else:
            canvas = np.zeros((H, W, 3), np.uint8)
            canvas[:, :] = view
            ovl_x = 0
        if ui["debug"]:
            draw_overlay(canvas[:, ovl_x:], box, score, state, err, info["dead"],
                         ui["paused"], search_rect, tracker.src)
            if two_panel:
                cv2.putText(canvas, "world (source video)", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "camera view (simulated gimbal)", (ovl_x + 8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            lines = [
                "FPS %.1f  %s  score %.2f  %s" % (
                    sum(fps_hist) / len(fps_hist), state, score,
                    "*** PAUSED ***" if ui["paused"] else ""),
                "cmd ang %+6.1f %+6.1f   rate %+6.1f %+6.1f deg/s  hold=%s" % (
                    info["pan"], info["tilt"], info["wx"], info["wy"], hold),
                "cam ang %+6.1f %+6.1f   limit ±%.0f°   travel %.0f°/%.0f°   drive=%s" % (
                    st_cam["pan"], st_cam["tilt"], args.ang_limit,
                    gstate.data["travel_pan"] if gstate else 0.0,
                    gstate.data["travel_tilt"] if gstate else 0.0, ui["drive"]),
                "lost_t=%.1fs   region %.1f×框(面积%g×)   recover=%d" % (
                    (t_now - tracker.lost_since) if tracker.lost_since else 0.0,
                    tracker.search_margin, (1 + 2 * tracker.search_margin) ** 2,
                    tracker.n_recover),
                "keys: space=pause p=replay r=reselect c=recenter b=drive d=overlay q=quit",
            ]
            for i, s in enumerate(lines):
                put_text(canvas, s, (8, 46 + i * 20), 0.55, (255, 255, 0))
        else:
            put_text(canvas, "FPS %.1f  %s  err %+.3f %+.3f" % (
                sum(fps_hist) / len(fps_hist), state, err[0], err[1]),
                (8, 22), 0.55, (255, 255, 0))

        cv2.imshow(MAIN_WIN, canvas)
        target_dt = (1.0 / args.fps_cap) if src.is_camera else src.dt
        delay = int(round(max(0.0, target_dt - (time.perf_counter() - t_now)) * 1000)) + 1
        key = cv2.waitKey(max(1, delay)) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == ord(" "):
            ui["paused"] = not ui["paused"]
            print("暂停（画面与云台冻结）" if ui["paused"] else "继续")
        elif key == ord("p"):
            ui["replay"] = True
        elif key == ord("r"):
            ui["want_reset"] = True
            ui["recenter"] = True
        elif key == ord("c"):
            ui["recenter"] = True
            print("回正：平滑转回 0°")
        elif key == ord("b"):
            ui["drive"] = not ui["drive"]
        elif key == ord("d"):
            ui["debug"] = not ui["debug"]
        elif key == ord("s"):
            fn = "snapshot_%d.png" % int(time.time())
            cv2.imwrite(fn, canvas)
            print("已保存", fn)
        if t_frame > 2:
            # 关窗退出必须**连续多次**确认才认：忙帧（目标出框后每帧都在搜索，单帧 50ms+） 或用户拖动/缩放窗口时，这个属性会短暂读到 0，而这里以前是"一次读到 0 就 break" ——
            try:
                vis = cv2.getWindowProperty(MAIN_WIN, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                vis = 1                      # 读不到不等于窗口没了
            win_gone = win_gone + 1 if vis < 1 else 0
            if win_gone >= 30:
                break

    src.release()
    cam.close()
    if gstate is not None:
        gstate.save(force=True)
        print("云台状态已保存（pan=%+.1f° tilt=%+.1f° 行程 %.0f°/%.0f°）"
              % (gstate.data["pan"], gstate.data["tilt"],
                 gstate.data["travel_pan"], gstate.data["travel_tilt"]))
    cv2.destroyAllWindows()
    return 0

# 十、闭环试验台 + 场景矩阵（4 个场景）

# 十一、自检（8 项）

def _check_tracker():
    """静止定位 + 遮挡后判定丢失 + 区域搜索找回"""
    st = SynthStage(320, 240, seed=1, tw=40, th=60)
    bgr = st.render(np.zeros(2, np.float32))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tr = TargetTracker()
    truth = st._box_on_screen(st.rel)
    tr.init(gray, bgr, truth)
    ok, box, score, state = tr.update(gray, bgr, 1 / 30.0)
    err = float(np.hypot(box[0] - truth[0], box[1] - truth[1]))
    for i in range(20):                       # 遮挡 20 帧
        st.hide_until = 1.0
        g = cv2.cvtColor(st.render(np.zeros(2, np.float32)), cv2.COLOR_BGR2GRAY)
        b = st.render(np.zeros(2, np.float32))
        tr.update(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), b, 1 / 30.0)
    lost = tr.state == TargetTracker.STATE_LOST
    st.hide_until = 0.0
    st.rel = np.array([40.0, 20.0], np.float32)      # 遮挡期间目标移动了
    got, iou = None, 0.0
    for k in range(6):
        b = st.render(np.zeros(2, np.float32))
        g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        r = tr.search_region(b, g, (160.0, 120.0), 1.0, 1.0 + 0.1 * k)
        if r is not None:
            got = r[0]
            iou = box_iou(r[0], st._box_on_screen(st.rel))
            break
    print("       静止定位误差 %.2fpx；遮挡判定丢失=%s；区域搜索找回 IoU %.2f"
          % (err, lost, iou))
    return err < 1.5 and lost and got is not None and iou > 0.5

def _check_region_search():
    """分层区域搜索：位移 + 尺度变化 + 颜色干扰物"""
    st = SynthStage(640, 480, seed=5, tw=60, th=90)
    bgr = st.render(np.zeros(2, np.float32))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tr = TargetTracker()
    tr.init(gray, bgr, st._box_on_screen(st.rel))
    for i in range(5):                        # 先正常跟几帧，把模型建起来
        b = st.render(np.zeros(2, np.float32))
        tr.update(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), b, 1 / 30.0)
    st.rel = np.array([2.2 * 60.0, 0.6 * 90.0], np.float32)
    for i in range(5):
        k = 1.0 - 0.06 * i
        st.tgt = cv2.resize(st.tgt, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        b = st.render(np.zeros(2, np.float32))
        g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        r = tr.search_region(b, g, (320.0, 240.0), 3.0, 2.0 + 0.1 * i)
        if r is not None:
            break
    iou = box_iou(r[0], st._box_on_screen(st.rel)) if r is not None else 0.0
    print("       位移 2.2 框宽 + 缩小 30%%：区域搜索 %s IoU %.2f"
          % ("成功" if r is not None else "失败", iou))
    st2 = SynthStage(640, 480, seed=5, tw=60, th=90)
    bgr0 = st2.render(np.zeros(2, np.float32), tint=(0, 0, 220))
    tr2 = TargetTracker()
    tr2.init(cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY), bgr0, st2._box_on_screen(st2.rel))
    for i in range(5):
        b = st2.render(np.zeros(2, np.float32), tint=(0, 0, 220))
        tr2.update(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), b, 1 / 30.0)
    st2.rel = np.array([90.0, 40.0], np.float32)
    b = st2.render(np.zeros(2, np.float32), tint=(0, 0, 220))
    tb = st2._box_on_screen(st2.rel)
    x, y = int(round(tb[0] - 90.0)), int(round(tb[1]))       # 干扰物在真目标左边 90px
    w, h = int(round(tb[2])), int(round(tb[3]))
    x, y = max(0, x), max(0, y)
    b[y:y + h, x:x + w] = np.clip(0.35 * b[y:y + h, x:x + w].astype(np.float32)
                                  + 0.65 * np.array([220, 0, 0], np.float32),
                                  0, 255).astype(np.uint8)
    r2 = None
    for k in range(5):
        r2 = tr2.search_region(b, cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (320.0, 240.0),
                               3.0, 3.0 + 0.1 * k)
        if r2 is not None:
            break
    iou2 = box_iou(r2[0], tb) if r2 is not None else 0.0
    # 顺带验证 HSV 模型本身：目标块 LLR 必须 >0，纯背景块必须更低（否则颜色判据没用）
    tb2 = st2._box_on_screen(st2.rel)
    lv_fg = tr2.llr_of(cv2.cvtColor(b, cv2.COLOR_BGR2HSV), tb2)
    bx, by = int(max(0, tb2[0] - 200)), int(max(0, tb2[1] - 40))
    lv_bg = tr2.llr_of(cv2.cvtColor(b, cv2.COLOR_BGR2HSV), (bx, by, 40, 50))
    print("       颜色干扰物（同形不同色）：%s，IoU(真目标) %.2f；LLR 目标 %+.2f / 背景 %+.2f"
          % ("成功" if r2 is not None else "失败", iou2, lv_fg, lv_bg))
    return iou > 0.4 and iou2 > 0.4 and lv_fg > 0 and lv_bg < lv_fg

def _check_box_scale():
    """框选自适应（颜色掩膜 + 边界 + 颜色比例）：静止不漂、缩放跟得上、不吃背景。
    """
    def once(rate, frames=120):
        st = SynthStage(640, 480, seed=11, tw=120, th=160)
        base = np.clip(st.tgt.astype(np.float32) * 0.8 + 80.0, 0, 255).astype(np.uint8)
        b0 = st.render(np.zeros(2, np.float32), tint=(0, 0, 200))
        tr = TargetTracker()
        tr.init(cv2.cvtColor(b0, cv2.COLOR_BGR2GRAY), b0, st._box_on_screen(st.rel))
        errs, fills, state, t = [], [], "track", 0.0
        for i in range(frames):
            t = i / 30.0
            s = max(0.3, 1.0 - rate * t)
            st.tgt = cv2.resize(base, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            st.tw, st.th = st.tgt.shape[1], st.tgt.shape[0]
            b = st.render(np.zeros(2, np.float32), tint=(0, 0, 200))
            _ok, box, _score, state = tr.update(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), b,
                                                1 / 30.0)
            truth = st._box_on_screen(st.rel)
            ex = tr.measure_extent(b, tr.box)
            fills.append(ex[1] if ex else float("nan"))
            if i > 15:
                errs.append(abs(box[3] - truth[3]) / truth[3])
            if state != TargetTracker.STATE_TRACK:
                break
        e = 100 * float(np.mean(errs)) if errs else 999.0
        return e, float(np.nanmean(fills)) if fills else float("nan"), state, t

    e0, f0, s0, _ = once(0.0)
    e1, f1, s1, _ = once(0.03)
    e2, f2, s2, _ = once(0.10)
    e3, f3, s3, _ = once(-0.10)
    print("       静止 尺寸误差 %.1f%%（填充率 %.2f）| 缩小 3%%/s %.1f%% | 缩小 10%%/s %.1f%%"
          " | 放大 10%%/s %.1f%%" % (e0, f0, e1, e2, e3))
    print("       颜色比例(填充率)均值 %.2f/%.2f/%.2f/%.2f（掩膜外接框里目标像素占比，"
          "低于 %.2f 就冻结不动）" % (f0, f1, f2, f3, 0.35))
    return (s0 == s1 == s2 == s3 == "track" and e0 < 2.0 and e1 < 8.0 and e2 < 8.0
            and e3 < 8.0)

def _check_black_region():
    """
    黑边（云台平移露出来的部分）必须被排除在检测范围之外。
    """
    st = SynthStage(640, 480, seed=3, tw=60, th=90)
    base = st.render(np.zeros(2, np.float32), tint=(0, 0, 200))
    dx = 200                                  # 假装云台转了 200px：内容左移，右侧是黑边
    view = np.zeros_like(base)
    view[:, :640 - dx] = base[:, dx:]
    valid = (0, 0, 640 - dx, 480)
    tr = TargetTracker()
    tr.set_valid_rect(valid)
    tb = st._box_on_screen(st.rel)
    vis = (tb[0] - dx, tb[1], tb[2], tb[3])
    tr.init(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), view, vis)
    over = 0
    for i in range(20):
        _ok, b, _s, state = tr.update(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), view, 1 / 30.0)
        if b[0] + b[2] > valid[2] + 1e-6:
            over += 1
    iou = box_iou(b, vis)

    flat = np.zeros_like(base)
    flat[:, :valid[2]] = 90
    flat[100:200, 100:200] = 30
    tr2 = TargetTracker()
    tr2.set_valid_rect(valid)
    tr2.init(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), flat, (350, 150, 60, 90))
    fake = 0
    for i in range(30):
        _ok, b2, _s2, state2 = tr2.update(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), flat,
                                          1 / 30.0)
        if state2 == TargetTracker.STATE_TRACK and b2[0] + b2[2] > valid[2] + 1e-6:
            fake += 1
    print("       有效区 x<%.0f；目标在区内：框越界 %d 次、IoU %.2f（状态 %s）"
          % (valid[2], over, iou, state))
    print("       低纹理+黑边：越界假跟 %d 次，最终状态 %s（应为 lost）" % (fake, state2))
    return over == 0 and iou > 0.6 and fake == 0 and state2 == TargetTracker.STATE_LOST

def _check_video_d():
    """
    用真实视频 d.mp4（雨天街道，打紫伞朝镜头走来的人）验证两件事：
    """
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d.mp4")
    if not os.path.exists(path):
        print("       跳过：同目录没有 d.mp4")
        return True
    BOX = (510, 260, 180, 280)                 # 紫伞行人（f500 处）
    N = 149
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    frames = []
    for _ in range(N):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    tr = TargetTracker()
    tr.init(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), frames[0], BOX)
    scores, lost_a = [], None
    for i, f in enumerate(frames):
        _ok, b, score, state = tr.update(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), f, 1 / 30.0)
        scores.append(score)
        if state != TargetTracker.STATE_TRACK:
            lost_a = i
            break
    med = float(np.median(scores))
    print("       A 连续跟踪 %d 帧：%s，分数中位 %.2f" %
          (i + 1 if lost_a is None else lost_a, "未丢失" if lost_a is None else "丢失@f%d" % lost_a, med))
    tr2 = TargetTracker()
    tr2.init(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), frames[0], BOX)
    x, y, w, h = [int(v) for v in BOX]
    lost_b = rec = None
    for i, f in enumerate(frames):
        t = i / 30.0
        v = f.copy()
        if 40 <= i < 70:                       # 30 帧 = 1s 遮挡
            v[max(0, y - 25):y + h + 25, max(0, x - 25):x + w + 25] = 128
        _ok, b, score, state = tr2.update(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY), v, 1 / 30.0)
        if state == TargetTracker.STATE_LOST:
            if lost_b is None:
                lost_b = i
            b0 = tr2.lost_box if tr2.lost_box is not None else tr2.anchor_box
            if b0 is not None:
                margin, every = tr2.region_margin(t - (tr2.lost_since or t))
                if i % max(1, every) == 0:
                    got = tr2.search_region(v, cv2.cvtColor(v, cv2.COLOR_BGR2GRAY),
                                            (b0[0] + b0[2] / 2.0, b0[1] + b0[3] / 2.0),
                                            margin, t)
                    if got is not None:
                        rec = i
                        break
    print("       B 遮挡 1s：丢失@f%s，重捕获 %s" %
          (lost_b, ("f%d（遮挡结束 f70 后 %d 帧）" % (rec, rec - 70)) if rec else "失败"))
    ok = lost_a is None and med >= 0.80 and lost_b is not None and rec is not None and rec <= 75
    frames.clear()                             # 及时释放 149 张大帧，免得拖慢后续检查
    del tr, tr2
    return ok

def _check_no_sudden_loss():
    """
    【无突变突然丢失】回归：小目标/小搜索窗上 PSR 会被低估，不许因此丢掉好匹配。
    """
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d.mp4")
    if not os.path.exists(path):
        print("       跳过：同目录没有 d.mp4")
        return True
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    frames = []
    for _ in range(149):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 149:
        print("       跳过：d.mp4 不足 149 帧")
        return True
    ok_all, worst = True, 0
    boxes = [(560, 300, 60, 90), (540, 300, 80, 110)]
    Hf, Wf = frames[0].shape[:2]
    for box in boxes:
        tr = TargetTracker()
        tr.init(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), frames[0], box)
        lost, last_b, near_border = None, None, False
        for i, f in enumerate(frames):
            _ok, b, sc, state = tr.update(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), f, 1 / 30.0)
            last_b = b
            if state != TargetTracker.STATE_TRACK:
                lost = i
                break
        if lost is not None and last_b is not None:
            near_border = (last_b[0] <= 1.0 or last_b[1] <= 1.0
                           or last_b[0] + last_b[2] >= Wf - 1.0
                           or last_b[1] + last_b[3] >= Hf - 1.0)
        if lost is not None and not near_border:
            ok_all = False
            worst = max(worst, lost)
        print("       小框 %s：%s" % (str(box),
              "跟满 149 帧" if lost is None
              else ("如实丢失@f%d（目标走出边框）" % lost if near_border
                    else "无突变丢失@f%d" % lost)))
    return ok_all

def _check_click_segment():
    """
    点击选物：点物体 -> 框住该物体的边界（紧、稳）；点空地 -> 必须拒绝。
    """
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d.mp4")
    if not os.path.exists(path):
        print("       跳过：同目录没有 d.mp4")
        return True
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    ok, f = cap.read()
    cap.release()
    if not ok:
        print("       跳过：读帧失败")
        return True
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    pm = cv2.inRange(hsv, (125, 60, 50), (170, 255, 255))
    pm = cv2.morphologyEx(pm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(pm, 8)
    if n < 2:
        print("       跳过：f500 没抠出紫色区域")
        return True
    j = 1 + int(np.argmax(st[1:, 4]))
    gt = tuple(float(v) for v in st[j, :4])
    # ---- 伞上多个点，必须都紧、且彼此稳 ----
    pts = [(606, 372), (600, 340), (640, 360), (610, 400)]
    segment_at(f, 606, 372)                    # 预热：cv2 首次调用有一次性初始化开销
    boxes, ms = [], 0.0
    for (px, py) in pts:
        ts = []
        b = None
        for _r in range(3):
            t0 = time.perf_counter()
            b = segment_at(f, px, py)
            ts.append((time.perf_counter() - t0) * 1000.0)
        ms = max(ms, float(np.median(ts)))
        boxes.append(None if b is None else tuple(float(v) for v in b))
    ious = [box_iou(b, gt) if b else 0.0 for b in boxes]
    tight = sum(1 for v in ious if v >= 0.55) >= 3           # 至少 3/4 个点够紧
    pair = []
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            if boxes[a] and boxes[b]:
                pair.append(box_iou(boxes[a], boxes[b]))
    stable = (np.mean(pair) >= 0.5) if pair else False
    b2 = segment_at(f, 700, 900)                              # 空地
    ok_air = (b2 is None or box_iou(b2, gt) < 0.1)
    print("       伞上 4 点 IoU(紫伞GT) %s（需≥3 点≥0.55）" % [round(v, 2) for v in ious])
    print("       点间一致性 IoU 均值 %.2f（需≥0.5）；单次最长 %.0f ms（需<40）"
          % (np.mean(pair) if pair else 0.0, ms))
    print("       点空地 -> %s（必须拒绝或与目标无关）"
          % ("拒绝" if b2 is None else "框 [%d,%d,%d,%d]" % (b2[0], b2[1], b2[2], b2[3])))
    return tight and stable and ms < 40.0 and ok_air

def _check_target_exits():
    """目标**走出边框**：不许抛异常、必须如实判丢失、框必须留在画面内、丢失后搜索不许炸。
    """
    st = SynthStage(640, 480, seed=2, tw=60, th=90)
    b0 = st.render(np.zeros(2, np.float32), tint=(0, 0, 200))
    tr = TargetTracker()
    tr.init(cv2.cvtColor(b0, cv2.COLOR_BGR2GRAY), b0, st._box_on_screen(st.rel))
    err, state, box = None, "track", None
    for i in range(120):
        st.rel = np.array([60.0 + 9.0 * i, 5.0 * i], np.float32)    # 一路走出右下角
        b = st.render(np.zeros(2, np.float32), tint=(0, 0, 200))
        g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        try:
            _ok, box, _s, state = tr.update(g, b, 1 / 30.0)
            if state == TargetTracker.STATE_LOST and i % 20 == 0:
                m, _e = tr.region_margin(i / 30.0 - (tr.lost_since or i / 30.0))
                tr.search_region(b, g, (320.0, 240.0), m, i / 30.0)
        except Exception as exc:
            err = "%s: %s" % (type(exc).__name__, exc)
            break
    inside = (box is not None and box[0] >= -1.0 and box[1] >= -1.0
              and box[0] + box[2] <= 641.0 and box[1] + box[3] <= 481.0)
    tiers = [tr.region_margin(s)[0] for s in (0.1, 1.0, 3.0, 10.0)]
    tiers_ok = all(abs(v - tr.search_margin) < 1e-9 for v in tiers)
    print("       走出边框：%s；末状态 %s；框留在画面内=%s；固定搜索域 %s=%s"
          % ("无异常" if err is None else "异常 " + err, state, inside, tiers, tiers_ok))
    return (err is None and state == TargetTracker.STATE_LOST and inside and tiers_ok)

def _check_controller():
    c = FollowController(hfov=60.0, vfov=36.0, kp=2.0, kd=0.0, kff=0.0, deadband=0.05,
                         max_rate=30.0, max_acc=1e9, lp_tau=0.0, lead=0.0)
    c.update((0.02, 0.0), (0.0, 0.0), 1 / 30.0)
    dead = abs(c.pan) == 0.0
    c2 = FollowController(max_rate=20.0, kff=0.0, lp_tau=0.0, max_acc=1e6, deadband=0.0)
    c2.update((1.0, 0.0), (0.0, 0.0), 1 / 30.0)
    rate = abs(c2.rate[0])
    c3 = FollowController(max_rate=60.0, kff=0.0, lp_tau=0.0, max_acc=1e6, deadband=0.0,
                          lead=0.2)
    c3.update((0.1, 0.0), (1.0, 0.0), 1 / 30.0)
    c4 = FollowController(max_rate=60.0, kff=0.0, lp_tau=0.0, max_acc=1e6, deadband=0.0,
                          lead=0.2)
    c4.update((0.1, 0.0), (0.0, 0.0), 1 / 30.0)
    print("       死区=%s；限速后角速度 %.1f（应=20）；预测使角速度 %.1f > %.1f"
          % (dead, rate, abs(c3.rate[0]), abs(c4.rate[0])))
    return dead and abs(rate - 20.0) < 1e-6 and abs(c3.rate[0]) > abs(c4.rate[0])

def _check_sim_plant():
    cam = SimCamera(hfov=60.0, vfov=36.0, frame_w=640, frame_h=480, latency=0.0,
                    max_rate=100.0, max_acc=1e9, deadband_deg=0.0, backlash_deg=0.0,
                    quant_deg=0.0, noise_deg=0.0)
    for i in range(30):
        cam.command(300.0, 0.0, 1 / 30.0)
    rate = float(cam.state()["pan"])
    cam2 = SimCamera(latency=0.10, max_rate=1e6, max_acc=1e9, deadband_deg=0.0,
                     backlash_deg=0.0, quant_deg=0.0, noise_deg=0.0)
    for i in range(2):
        cam2.command(30.0, 0.0, 1 / 30.0)
    early = float(cam2.state()["pan"])
    for i in range(6):
        cam2.command(30.0, 0.0, 1 / 30.0)
    late = float(cam2.state()["pan"])
    cam3 = SimCamera(latency=0.0, max_rate=1e6, max_acc=1e9, deadband_deg=0.0,
                     backlash_deg=2.0, quant_deg=0.0, noise_deg=0.0)
    cam3.command(10.0, 0.0, 1 / 30.0)
    fwd = float(cam3.state()["pan"])
    held = fwd
    for drop in (0.5, 1.0, 1.5):
        cam3.command(10.0 - drop, 0.0, 1 / 30.0)
        held = float(cam3.state()["pan"])
    print("       限速 300°阶跃1s->%.0f°；延迟 0.067s 时 %.2f° / 0.27s 时 %.0f°；"
          "回差 1.5° 内不动 %.2f->%.2f" % (rate, early, late, fwd, held))
    return abs(rate - 100.0) < 12.0 and early < 1.0 and late > 20.0 and abs(held - fwd) < 0.6

def _check_gimbal_state_and_recenter(tmp_state):
    """角度存储往返 + 损坏文件降级 + 整机回正（预置角度 -> 按 r -> 回到 0）"""
    import tempfile
    if os.path.exists(tmp_state):
        os.remove(tmp_state)
    gs = GimbalState(tmp_state, ang_limit=90.0, px_per_deg=12.0)
    for i in range(10):
        gs.update(3.0 * i, -1.0 * i)
    gs.save(force=True)
    gs2 = GimbalState(tmp_state, ang_limit=90.0, px_per_deg=12.0)
    ok_state = (gs2.data.get("loaded") and abs(gs2.data["travel_pan"] - 27.0) < 1e-6)
    with open(tmp_state, "w", encoding="utf-8") as f:
        f.write("{ broken json ")
    broken_ok = GimbalState(tmp_state).data.get("loaded") is False
    print("       角度存储往返=%s（行程 %.0f°）损坏文件降级=%s"
          % (ok_state, gs2.data["travel_pan"], broken_ok))

    # 整机回正：为什么必须整机测 —— 单测 slew_to 是对的，但主循环里回正若被 同一帧的 ctrl.update(hold=True) 清零，就"每帧只挪一下"，按 r 等于没反应
    V = None
    for cand in ("b.mp4", os.path.join(os.path.dirname(os.path.abspath(__file__)), "b.mp4")):
        if os.path.isfile(cand):
            V = cand
            break
    if V is None:
        print("       整机回正：跳过（找不到 b.mp4）")
        return ok_state and broken_ok
    saved = {k: getattr(cv2, k) for k in
             ("namedWindow", "resizeWindow", "setMouseCallback", "imshow", "waitKey",
              "destroyAllWindows", "getWindowProperty")}
    angles = []
    orig = SimCamera.command
    SimCamera.command = lambda self, p, t, dt: (angles.append((float(p), float(t))),
                                                orig(self, p, t, dt))[1]
    cv2.namedWindow = cv2.resizeWindow = cv2.setMouseCallback = lambda *a, **k: None
    cv2.imshow = lambda *a, **k: None
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.getWindowProperty = lambda w, p: 1
    st = {"n": 0}

    def fw(d=0):
        st["n"] += 1
        return ord("q") if st["n"] >= 400 else (ord("r") if st["n"] == 10 else 255)
    cv2.waitKey = fw
    state = os.path.join(tempfile.gettempdir(), "a8_recenter.json")
    g0 = GimbalState(state)
    g0.data["pan"], g0.data["tilt"] = 30.0, -20.0
    g0.save(force=True)
    try:
        run(build_parser().parse_args(["--video", V, "--init", "450", "400", "80", "120",
                                       "--no-loop", "--no-start-recenter",
                                       "--gimbal-state", state]))
    finally:
        SimCamera.command = orig
        for k, v in saved.items():
            setattr(cv2, k, v)
    final = angles[-1] if angles else (99.0, 99.0)
    got_zero = abs(final[0]) <= 1.0 and abs(final[1]) <= 1.0
    print("       整机回正：预置 +30°；按 r 后最终 pan=%+.1f° tilt=%+.1f°（应≈0）"
          % (final[0], final[1]))
    for p in (state, tmp_state):
        try:
            os.remove(p)
        except OSError:
            pass
    return ok_state and broken_ok and got_zero

def run_selftest():
    """自检 8 项：跟踪/区域搜索/颜色模型/控制器/云台模型/闭环/角度存储与回正"""
    print("=" * 68)
    print("attempt9 自检（颜色比例特征重捕获 + HSV 模型）  OpenCV %s"
          % cv2.__version__)
    print("=" * 68)
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "a9_state.json")
    checks = (("跟踪器（定位/丢失/区域搜索找回）", _check_tracker),
              ("颜色比例重捕获（位移+尺度+颜色干扰物）", _check_region_search),
              ("框选自适应（颜色掩膜+边界+颜色比例）", _check_box_scale),
              ("黑边不进检测范围（有效像素矩形）", _check_black_region),
              ("真实视频 d.mp4：不误丢 + 丢失后重捕获", _check_video_d),
              ("无突变突然丢失（小目标 PSR 低估回归）", _check_no_sudden_loss),
              ("点击选物（点一下自动识别物体边界）", _check_click_segment),
              ("目标走出边框：不异常/如实丢失/搜索不炸", _check_target_exits),
              ("控制器（死区/限幅/预测）", _check_controller),
              ("虚拟云台（延迟/限速/回差）", _check_sim_plant),
              ("角度存储与整机回正", lambda: _check_gimbal_state_and_recenter(tmp)))
    res = {}
    for i, (title, fn) in enumerate(checks, 1):
        print("\n[%d] %s" % (i, title))
        try:
            res[title.split("（")[0]] = bool(fn())
        except Exception as e:
            print("       异常: %s: %s" % (type(e).__name__, e))
            res[title.split("（")[0]] = False
    print("\n====================== 自检结论 ======================")
    for k, v in res.items():
        print("       %-26s %s" % (k, "PASS" if v else "FAIL"))
    ok = all(res.values())
    print("       %s" % ("全部通过" if ok else "存在失败项"))
    print("=====================================================")
    return 0 if ok else 1

# 十二、命令行

def build_parser():
    ap = argparse.ArgumentParser(description="目标跟随（静态等待模式）attempt8")
    ap.add_argument("--video", nargs="+", default=None, help="视频文件（取第一个）")
    ap.add_argument("--camera", type=int, default=0, help="摄像头编号")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps-cap", dest="fps_cap", type=float, default=FPS_CAP)
    ap.add_argument("--no-loop", dest="loop", action="store_false", default=True)
    ap.add_argument("--init", nargs=4, type=float, default=None,
                    metavar=("X", "Y", "W", "H"), help="初始目标框（不给就鼠标拖框）")

    ap.add_argument("--hfov", type=float, default=60.0, help="水平视野角(度)，必须与镜头一致")
    ap.add_argument("--vfov", type=float, default=36.0)
    ap.add_argument("--kp", type=float, default=2.5)
    ap.add_argument("--kd", type=float, default=0.15)
    ap.add_argument("--kff", type=float, default=None,
                    help="速度前馈增益；默认自动（真机 1.0，视频在环 0）。"
                         "视频在环开它会一路朝一个方向走到底，除非你自己测出视频的运动")
    ap.add_argument("--deadband", type=float, default=0.03, help="死区（占画面比例）")
    ap.add_argument("--max-rate", dest="max_rate", type=float, default=60.0)
    ap.add_argument("--lp-tau", dest="lp_tau", type=float, default=0.12)
    ap.add_argument("--no-predict", dest="no_predict", action="store_true",
                    help="关掉延迟预测与速度前馈（对照组）")
    ap.add_argument("--ang-limit", dest="ang_limit", type=float, default=120.0)

    ap.add_argument("--score-keep", dest="score_keep", type=float, default=0.50,
                    help="可接受阈值（低于它才算没跟住）")
    ap.add_argument("--lost-after", dest="lost_after", type=int, default=12,
                    help="连续多少帧不可接受就判丢失（30fps 下 12 帧=0.4s）")
    ap.add_argument("--llr-min", dest="llr_min", type=float, default=0.02,
                    help="前景/背景对数似然比门槛（决定要不要写进模型）")

    ap.add_argument("--real", choices=("none", "serial"), default="none",
                    help="云台接口：none=虚拟云台(默认) / serial=串口")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no-cam", dest="no_cam", action="store_true", help="只跟踪不驱动云台")

    ap.add_argument("--latency", type=float, default=0.10, help="仿真云台延迟(s)")
    ap.add_argument("--sim-rate", dest="sim_rate", type=float, default=120.0)
    ap.add_argument("--sim-backlash", dest="sim_backlash", type=float, default=1.0)

    ap.add_argument("--gimbal-state", dest="gimbal_state", default="gimbal_state.json")
    ap.add_argument("--no-state", dest="no_state", action="store_true")
    ap.add_argument("--no-start-recenter", dest="no_start_recenter", action="store_true")
    ap.add_argument("--recenter-rate", dest="recenter_rate", type=float, default=40.0)
    ap.add_argument("--recenter-tol", dest="recenter_tol", type=float, default=0.5)

    ap.add_argument("--selftest", action="store_true", help="跑自检")
    return ap

def _install_crash_reporter():
    """
    顶层异常兜底：把完整 traceback + 排查用的现场状态打到 stderr。
    """
    import traceback as _tb
    import sys as _sys
    _real_hook = _sys.excepthook

    def _hook(etype, value, tb):
        try:
            frames = _tb.extract_tb(tb)
            print("\n" + "=" * 70, file=_sys.stderr)
            print("程序崩溃（已捕获现场，下面是完整堆栈）", file=_sys.stderr)
            print("=" * 70, file=_sys.stderr)
            _tb.print_exception(etype, value, tb)
            if frames:
                f = frames[-1]
                print(">>> 出错位置：%s 第 %d 行（%s）" % (f.filename, f.lineno, f.name),
                      file=_sys.stderr)
                print(">>> 出错行内容：%s" % (f.line or "").strip(), file=_sys.stderr)
            print(">>> 请把以上全部内容贴回给开发者，即可定位。", file=_sys.stderr)
        except Exception:
            pass
        _real_hook(etype, value, tb)

    _sys.excepthook = _hook

if __name__ == "__main__":
    _install_crash_reporter()
    _a = build_parser().parse_args()
    if _a.selftest:
        raise SystemExit(run_selftest())
    raise SystemExit(run(_a))
