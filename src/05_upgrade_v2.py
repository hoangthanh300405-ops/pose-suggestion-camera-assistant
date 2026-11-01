# -*- coding: utf-8 -*-
"""
05_upgrade_v2.py — Bản nâng cấp "một file chạy được" cho Pose Suggestion Camera Assistant.

Thay thế/bổ sung cho 02 (PoseRecommender), 03 (OverlayRenderer), 04 (pipeline).
Giữ nguyên định dạng pose_templates_v1.json và checkpoint scene_analyzer.pt.

NÂNG CẤP so với bản cũ
  [BUG-1] Template dùng tên khớp 'left_shoulder' (mock) vs 'l_shoulder' (MPII) -> trước đây
          chỉ còn 2 khớp chung nên demo luôn ra score 0.0. Giờ tự chuẩn hoá tên + cảnh báo
          template nghèo khớp (validate_templates).
  [BUG-2] Chuẩn hoá toạ độ trên hệ 0-1 của MediaPipe làm méo tỉ lệ (x/w, y/h) trong khi
          template MPII là pixel đẳng hướng. Giờ đổi sang pixel trước khi chuẩn hoá.
  [BUG-3] cv2.putText không vẽ được tiếng Việt / "✓" / "→" (ra dấu ?). Giờ vẽ bằng PIL.
  [NEW-1] Điểm giống = vị trí khớp + GÓC khớp (khuỷu, gối, tay-thân, nghiêng vai/thân).
  [NEW-2] delta_hint TÍNH TỪ SỰ KHÁC BIỆT thật (vd "Gập khuỷu tay phải thêm"), không còn là text cố định.
  [NEW-3] Ghost overlay: vẽ khung xương MẪU mờ, căn theo hông + kích thước thân người dùng.
  [NEW-4] One-Euro filter làm mượt landmark + hysteresis để gợi ý không nhấp nháy.
  [NEW-5] Tự đếm người (num_poses) và ước lượng độ sáng; SceneAnalyzer chạy mỗi N frame + bỏ phiếu.
  [NEW-6] LatencyMeter: p50/p95 theo từng stage (số liệu cho CV; nhớ đo lại trên điện thoại thật).

Cách chạy (Colab):
    !python 05_upgrade_v2.py --selftest
    !python 05_upgrade_v2.py --source anh.jpg  --out ket_qua.jpg
    !python 05_upgrade_v2.py --source clip.mp4 --out ket_qua.mp4
"""

import os
import sys
import json
import math
import time
import argparse
import logging
import unicodedata
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("PoseAssistantV2")

# ===========================================================================
# CONFIG
# ===========================================================================
BASE_DRIVE = "/content/drive/MyDrive/pose-assistant"
TEMPLATE_PATH = f"{BASE_DRIVE}/templates/pose_templates_v1.json"
SCENE_CKPT = f"{BASE_DRIVE}/checkpoints/scene_analyzer.pt"
POSE_MODEL_PATH = "pose_landmarker.task"
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

VIS_THRESHOLD = 0.5          # khớp dưới ngưỡng này bị loại khỏi so khớp
CLOSE_MATCH_THRESHOLD = 0.85  # CHƯA calibrate — cần nhãn tay (xem calibrate ở 02)
ANGLE_WEIGHT = 0.6           # trọng số điểm góc so với điểm vị trí
POS_DECAY = 2.0
ANGLE_TAU_DEG = 40.0
HINT_MIN_DEG = 20.0          # lệch góc nhỏ hơn mức này -> coi như đã đúng
MIN_TEMPLATE_JOINTS = 10     # template ít khớp hơn sẽ bị cảnh báo

# Tên khớp thống nhất (kiểu MPII: l_/r_) — MediaPipe landmark index tương ứng
MP_INDEX = {
    "nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
}
SKELETON = [
    ("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"),
    ("r_shoulder", "r_hip"), ("l_hip", "r_hip"), ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
]
ANGLE_DEFS = {  # góc tại khớp giữa
    "l_elbow": ("l_shoulder", "l_elbow", "l_wrist"),
    "r_elbow": ("r_shoulder", "r_elbow", "r_wrist"),
    "l_knee": ("l_hip", "l_knee", "l_ankle"),
    "r_knee": ("r_hip", "r_knee", "r_ankle"),
    "l_arm_raise": ("l_hip", "l_shoulder", "l_elbow"),
    "r_arm_raise": ("r_hip", "r_shoulder", "r_elbow"),
}


def canon_name(name: str) -> str:
    """left_shoulder -> l_shoulder, right_knee -> r_knee (BUG-1)."""
    if name.startswith("left_"):
        return "l_" + name[5:]
    if name.startswith("right_"):
        return "r_" + name[6:]
    return name


# ===========================================================================
# KEYPOINT UTILS
# ===========================================================================
def landmarks_to_px(lm_norm: Dict[str, Tuple[float, float, float]], w: int, h: int) -> dict:
    """{name:(x01,y01,vis)} -> {name:(x_px,y_px,vis)} + pelvis/thorax (BUG-2)."""
    kps = {canon_name(n): (x * w, y * h, v) for n, (x, y, v) in lm_norm.items()}
    return add_pelvis_thorax(kps)


def add_pelvis_thorax(kps: dict) -> dict:
    out = dict(kps)
    if "l_hip" in out and "r_hip" in out and "pelvis" not in out:
        a, b = out["l_hip"], out["r_hip"]
        out["pelvis"] = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[2], b[2]))
    if "l_shoulder" in out and "r_shoulder" in out and "thorax" not in out:
        a, b = out["l_shoulder"], out["r_shoulder"]
        out["thorax"] = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[2], b[2]))
    return out


def normalize(kps: dict, vis_thr: float = VIS_THRESHOLD) -> Optional[Dict[str, Tuple[float, float]]]:
    """Gốc = hông, đơn vị = chiều dài thân (hông->ngực). Loại khớp visibility thấp."""
    good = {n: v for n, v in kps.items() if (v[2] if len(v) > 2 else 1.0) >= vis_thr}
    if "pelvis" not in good or "thorax" not in good:
        return None
    px, py = good["pelvis"][:2]
    scale = math.hypot(good["thorax"][0] - px, good["thorax"][1] - py)
    if scale < 1e-6:
        return None
    return {n: ((v[0] - px) / scale, (v[1] - py) / scale) for n, v in good.items()}


def mirror(norm: dict) -> dict:
    out = {}
    for n, (x, y) in norm.items():
        if n.startswith("l_"):
            n = "r_" + n[2:]
        elif n.startswith("r_"):
            n = "l_" + n[2:]
        out[n] = (-x, y)
    return out


def _angle(a, b, c) -> Optional[float]:
    v1, v2 = np.subtract(a, b), np.subtract(c, b)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1, 1))))


def features(norm: dict) -> Dict[str, float]:
    """Đặc trưng góc (độ). Chỉ trả về đặc trưng có đủ khớp."""
    f = {}
    for name, (a, b, c) in ANGLE_DEFS.items():
        if a in norm and b in norm and c in norm:
            ang = _angle(norm[a], norm[b], norm[c])
            if ang is not None:
                f[name] = ang
    if "l_shoulder" in norm and "r_shoulder" in norm:
        dx = norm["r_shoulder"][0] - norm["l_shoulder"][0]
        dy = norm["r_shoulder"][1] - norm["l_shoulder"][1]
        t = math.degrees(math.atan2(dy, dx))
        f["shoulder_tilt"] = t - 180 * np.sign(t) if abs(t) > 90 else t
    if "thorax" in norm and "pelvis" in norm:
        dx = norm["thorax"][0] - norm["pelvis"][0]
        dy = norm["thorax"][1] - norm["pelvis"][1]
        f["torso_lean"] = math.degrees(math.atan2(dx, -dy))
    return f


def pos_similarity(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    if len(common) < 4:
        return 0.0
    d = np.mean([math.hypot(a[j][0] - b[j][0], a[j][1] - b[j][1]) for j in common])
    return float(math.exp(-d * POS_DECAY))


def angle_similarity(fa: dict, fb: dict) -> Optional[float]:
    common = set(fa) & set(fb)
    if len(common) < 3:
        return None
    d = np.mean([abs(fa[k] - fb[k]) for k in common])
    return float(math.exp(-d / ANGLE_TAU_DEG))


def combined_score(live: dict, ref: dict, ref_feats: dict) -> Tuple[float, dict]:
    ps = pos_similarity(live, ref)
    fl = features(live)
    as_ = angle_similarity(fl, ref_feats)
    score = ps if as_ is None else (1 - ANGLE_WEIGHT) * ps + ANGLE_WEIGHT * as_
    return score, fl


# ===========================================================================
# HINT TỰ SINH TỪ SAI LỆCH THẬT (NEW-2)
# ===========================================================================
def compute_delta_hint(fl: dict, fr: dict, mirrored: bool) -> Optional[str]:
    diffs = {k: fl[k] - fr[k] for k in fl if k in fr}
    if not diffs:
        return None
    k = max(diffs, key=lambda x: abs(diffs[x]))
    d = diffs[k]
    if abs(d) < HINT_MIN_DEG:
        return None
    side = ""
    if k[:2] in ("l_", "r_"):
        s = k[0]
        if mirrored:  # so khớp trên bản lật -> bên thật của người dùng là bên đối diện
            s = "r" if s == "l" else "l"
        side = "trái" if s == "l" else "phải"
    if k.endswith("elbow"):
        return f"Gập khuỷu tay {side} thêm" if d > 0 else f"Duỗi tay {side} ra"
    if k.endswith("knee"):
        return f"Hơi gập gối {side}" if d > 0 else f"Duỗi thẳng chân {side}"
    if k.endswith("arm_raise"):
        return f"Hạ tay {side} gần thân hơn" if d > 0 else f"Nâng tay {side} ra xa thân hơn"
    if k == "shoulder_tilt":
        return "Chỉnh độ nghiêng vai theo khung xương mờ"
    return "Chỉnh độ nghiêng thân người theo khung xương mờ"


# ===========================================================================
# RECOMMENDER
# ===========================================================================
@dataclass
class SceneInfo:
    location: Optional[str] = None   # None = chưa biết -> bỏ qua điều kiện location
    num_people: int = 1
    lighting: str = "any"


@dataclass
class Recommendation:
    template_id: str
    pose_name: str
    caption: str
    delta_hint: Optional[str]
    similarity_score: float
    is_close_match: bool
    confidence: float
    scene_filter_level: str
    mirrored: bool
    ref_norm: dict


def validate_templates(templates: List[dict]) -> None:
    for t in templates:
        n = len(t["_ref_norm"] or {})
        if t["_ref_norm"] is None:
            logger.warning("Template %s thiếu pelvis/thorax -> luôn score 0.", t["template_id"])
        elif n < MIN_TEMPLATE_JOINTS:
            logger.warning("Template %s chỉ có %d khớp (<%d) — nhiều khả năng là template MOCK, "
                           "điểm sẽ kém tin cậy. Thay bằng template MPII thật.",
                           t["template_id"], n, MIN_TEMPLATE_JOINTS)
    ids = [t["template_id"] for t in templates]
    if any(str(i).startswith("demo_") for i in ids):
        logger.warning("Phát hiện template 'demo_*' — file đang là bản mock (Colab AI từng ghi đè). "
                       "Khôi phục pose_templates_v1.json thật trước khi benchmark.")


class PoseRecommenderV2:
    def __init__(self, template_path: str):
        with open(template_path, "r", encoding="utf-8") as f:
            self.templates = json.load(f)["templates"]
        for t in self.templates:
            raw = {canon_name(n): tuple(v) for n, v in t["keypoints_reference"].items()}
            raw = {n: (v[0], v[1], v[2] if len(v) > 2 else 1.0) for n, v in raw.items()}
            t["_ref_norm"] = normalize(add_pelvis_thorax(raw), vis_thr=0.0)
            t["_ref_feats"] = features(t["_ref_norm"]) if t["_ref_norm"] else {}
        validate_templates(self.templates)

    def _filter_by_scene(self, scene: SceneInfo) -> Tuple[List[dict], str]:
        def ok(t, lt, pp, lo):
            c = t.get("scene_condition", {})
            if lo and scene.location is not None and c.get("location") not in (scene.location, "any", None):
                return False
            if pp and c.get("num_people") not in (scene.num_people, "any", None):
                return False
            if lt and c.get("lighting") not in (scene.lighting, "any", None):
                return False
            return True
        for name, flags in [("exact", (1, 1, 1)), ("relaxed_lighting", (0, 1, 1)),
                            ("relaxed_lighting_people", (0, 0, 1)), ("relaxed_all", (0, 0, 0))]:
            m = [t for t in self.templates if ok(t, *flags)]
            if m:
                return m, name
        return [], "none"

    def recommend(self, kps_px: dict, scene: SceneInfo) -> List[Recommendation]:
        """kps_px: kết quả landmarks_to_px(). Trả về TẤT CẢ ứng viên đã xếp hạng."""
        live = normalize(kps_px)
        if live is None:
            return []
        vis = [v[2] for n, v in kps_px.items() if v[2] >= VIS_THRESHOLD]
        confidence = float(np.mean(vis)) if vis else 0.0
        candidates, level = self._filter_by_scene(scene)
        live_m = mirror(live)
        out = []
        for t in candidates:
            ref = t["_ref_norm"]
            if ref is None:
                out.append((0.0, t, False, {}))
                continue
            s1, f1 = combined_score(live, ref, t["_ref_feats"])
            s2, f2 = combined_score(live_m, ref, t["_ref_feats"])
            out.append((s1, t, False, f1) if s1 >= s2 else (s2, t, True, f2))
        out.sort(key=lambda x: x[0], reverse=True)
        recs = []
        for score, t, mir, fl in out:
            close = score >= CLOSE_MATCH_THRESHOLD
            hint = None if close else (compute_delta_hint(fl, t["_ref_feats"], mir) or t.get("delta_hint"))
            recs.append(Recommendation(
                t["template_id"], t["pose_name"], t["caption"], hint, round(score, 3),
                close, round(confidence, 3), level, mir, t["_ref_norm"] or {}))
        return recs


class StableSelector:
    """Hysteresis: chỉ đổi gợi ý khi ứng viên mới hơn rõ rệt trong `hold` frame liên tiếp."""
    def __init__(self, margin: float = 0.08, hold: int = 8):
        self.margin, self.hold = margin, hold
        self.current_id: Optional[str] = None
        self.count = 0

    def update(self, recs: List[Recommendation]) -> Optional[Recommendation]:
        if not recs:
            return None
        by_id = {r.template_id: r for r in recs}
        top = recs[0]
        if self.current_id not in by_id:
            self.current_id, self.count = top.template_id, 0
            return top
        cur = by_id[self.current_id]
        if top.template_id != cur.template_id and top.similarity_score > cur.similarity_score + self.margin:
            self.count += 1
            if self.count >= self.hold:
                self.current_id, self.count = top.template_id, 0
                return top
        else:
            self.count = 0
        return cur


# ===========================================================================
# SMOOTHING — One-Euro filter (NEW-4)
# ===========================================================================
class OneEuro:
    def __init__(self, min_cutoff=1.5, beta=10.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


class LandmarkSmoother:
    def __init__(self, **kw):
        self.kw, self.filters = kw, {}

    def reset(self):
        self.filters = {}

    def update(self, lm_norm: dict, t: float) -> dict:
        out = {}
        for n, (x, y, v) in lm_norm.items():
            f = self.filters.setdefault(n, OneEuro(**self.kw))
            sx, sy = f(np.array([x, y]), t)
            out[n] = (float(sx), float(sy), v)
        return out


# ===========================================================================
# RENDERING (BUG-3: Unicode qua PIL)
# ===========================================================================
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def draw_texts(img, items: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]]):
    """items: (text, (x,y), size, color_BGR). Trả về ảnh BGR."""
    import cv2
    font_path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
    try:
        from PIL import Image, ImageDraw, ImageFont
        if font_path is None:
            raise ImportError("no unicode font")
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for text, xy, size, bgr in items:
            d.text(xy, text, font=ImageFont.truetype(font_path, size), fill=(bgr[2], bgr[1], bgr[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        logger.warning("Không có font Unicode/PIL -> vẽ chữ không dấu (cài fonts-dejavu để có tiếng Việt).")
        for text, (x, y), size, bgr in items:
            cv2.putText(img, _strip_accents(text), (x, y + size), cv2.FONT_HERSHEY_SIMPLEX,
                        size / 32.0, bgr, 2, cv2.LINE_AA)
        return img


def draw_skeleton(img, kps_px: dict, color_hi=(0, 255, 0), color_lo=(0, 165, 255)):
    import cv2
    for a, b in SKELETON:
        if a in kps_px and b in kps_px:
            va, vb = kps_px[a][2], kps_px[b][2]
            c = color_hi if min(va, vb) >= VIS_THRESHOLD else color_lo
            cv2.line(img, (int(kps_px[a][0]), int(kps_px[a][1])),
                     (int(kps_px[b][0]), int(kps_px[b][1])), c, 3)
    for n, (x, y, v) in kps_px.items():
        if n in ("pelvis", "thorax"):
            continue
        cv2.circle(img, (int(x), int(y)), 5, color_hi if v >= VIS_THRESHOLD else color_lo, -1)
    return img


def draw_ghost(img, rec: Recommendation, kps_px: dict, alpha: float = 0.55):
    """NEW-3: khung xương MẪU mờ, căn theo hông + chiều dài thân của người dùng."""
    import cv2
    if "pelvis" not in kps_px or "thorax" not in kps_px or not rec.ref_norm:
        return img
    px, py = kps_px["pelvis"][:2]
    scale = math.hypot(kps_px["thorax"][0] - px, kps_px["thorax"][1] - py)
    ref = mirror(rec.ref_norm) if rec.mirrored else rec.ref_norm
    pts = {n: (int(px + x * scale), int(py + y * scale)) for n, (x, y) in ref.items()}
    layer = img.copy()
    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(layer, pts[a], pts[b], (255, 200, 0), 4, cv2.LINE_AA)
    for n, p in pts.items():
        if n not in ("pelvis", "thorax"):
            cv2.circle(layer, p, 7, (255, 200, 0), -1, cv2.LINE_AA)
    return cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)


def draw_caption(img, rec: Recommendation, scene: SceneInfo):
    import cv2
    h, w = img.shape[:2]
    band = 100
    layer = img.copy()
    cv2.rectangle(layer, (0, h - band), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(layer, 0.55, img, 0.45, 0)
    s = max(16, w // 40)
    items = [(("✓ " if rec.is_close_match else "") + rec.caption, (15, h - band + 8), s + 4, (255, 255, 255))]
    if rec.delta_hint and not rec.is_close_match:
        items.append((f"→ {rec.delta_hint}", (15, h - band + 12 + s + 8), s, (0, 255, 255)))
    badge = f"{scene.location or '?'} · {scene.num_people} người · {scene.lighting} · score {rec.similarity_score:.2f} · conf {rec.confidence:.2f}"
    items.append((badge, (15, 10), max(14, s - 4), (255, 255, 255)))
    return draw_texts(img, items)


# ===========================================================================
# ESTIMATOR / SCENE / LATENCY
# ===========================================================================
class PoseEstimatorMP:
    def __init__(self, model_path=POSE_MODEL_PATH, num_poses=4):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        if not os.path.exists(model_path):
            os.system(f"wget -q -O {model_path} {POSE_MODEL_URL}")
        self.mp = mp
        self.lm = mp_vision.PoseLandmarker.create_from_options(mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO, num_poses=num_poses))
        self._last_ts = -1

    def detect(self, frame_bgr, t_sec: float) -> List[dict]:
        import cv2
        ts = max(int(t_sec * 1000), self._last_ts + 1)  # VIDEO mode cần timestamp tăng dần
        self._last_ts = ts
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(mp_img, ts)
        people = []
        for lms in res.pose_landmarks:
            people.append({n: (lms[i].x, lms[i].y, lms[i].visibility) for n, i in MP_INDEX.items()})
        return people


def pick_main_person(people: List[dict]) -> dict:
    def area(p):
        xs, ys = [v[0] for v in p.values()], [v[1] for v in p.values()]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))
    return max(people, key=area)


class SceneAnalyzerRT:
    """Chạy MobileNetV3 mỗi `every` frame + bỏ phiếu đa số 5 lần gần nhất."""
    LABELS = ["indoor", "outdoor"]

    def __init__(self, ckpt=SCENE_CKPT, every=15):
        self.every, self.votes, self.model = every, deque(maxlen=5), None
        if not os.path.exists(ckpt):
            logger.warning("Không có checkpoint %s -> bỏ qua indoor/outdoor.", ckpt)
            return
        import torch
        import torch.nn as nn
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model, self.torch = m.eval(), torch

    def location(self, frame_bgr, frame_idx: int) -> Optional[str]:
        if self.model is None:
            return None
        if frame_idx % self.every == 0 or not self.votes:
            import cv2
            x = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255
            x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            t = self.torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)
            with self.torch.no_grad():
                self.votes.append(int(self.model(t).argmax(1)))
        return self.LABELS[max(set(self.votes), key=list(self.votes).count)]


def estimate_lighting(frame_bgr) -> str:
    import cv2
    v = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
    return "low" if v < 80 else ("bright" if v > 170 else "normal")


class LatencyMeter:
    def __init__(self):
        self.d = defaultdict(list)

    @contextmanager
    def measure(self, name):
        t = time.perf_counter()
        yield
        self.d[name].append((time.perf_counter() - t) * 1000)

    def report(self):
        print(f"\n{'Stage':<12}{'mean ms':>10}{'p50':>9}{'p95':>9}{'N':>7}")
        for k, v in self.d.items():
            print(f"{k:<12}{np.mean(v):>10.1f}{np.percentile(v, 50):>9.1f}{np.percentile(v, 95):>9.1f}{len(v):>7}")
        print("(Đo trên máy này — KHÔNG phải số liệu điện thoại. Đo lại trên thiết bị thật trước khi ghi vào CV.)")


# ===========================================================================
# PIPELINE
# ===========================================================================
class PoseAssistantV2:
    def __init__(self, template_path=TEMPLATE_PATH, scene_ckpt=SCENE_CKPT, use_scene=True):
        self.estimator = PoseEstimatorMP()
        self.scene_rt = SceneAnalyzerRT(scene_ckpt) if use_scene else SceneAnalyzerRT("__none__")
        self.recommender = PoseRecommenderV2(template_path)
        self.smoother, self.selector, self.meter = LandmarkSmoother(), StableSelector(), LatencyMeter()
        self.idx, self.lighting, self.missing = 0, "any", 0

    def process_frame(self, frame, t_sec: float):
        h, w = frame.shape[:2]
        with self.meter.measure("pose"):
            people = self.estimator.detect(frame, t_sec)
        if not people:
            self.missing += 1
            if self.missing > 10:
                self.smoother.reset()
            self.idx += 1
            return draw_texts(frame, [("Không thấy người trong khung hình", (15, 15), 22, (0, 0, 255))]), None
        self.missing = 0
        with self.meter.measure("scene"):
            loc = self.scene_rt.location(frame, self.idx)
            if self.idx % 15 == 0:
                self.lighting = estimate_lighting(frame)
        scene = SceneInfo(location=loc, num_people=len(people), lighting=self.lighting)
        with self.meter.measure("recommend"):
            smoothed = self.smoother.update(pick_main_person(people), t_sec)
            kps = landmarks_to_px(smoothed, w, h)
            rec = self.selector.update(self.recommender.recommend(kps, scene))
        with self.meter.measure("render"):
            out = draw_skeleton(frame.copy(), kps)
            if rec:
                out = draw_ghost(out, rec, kps)
                out = draw_caption(out, rec, scene)
        self.idx += 1
        return out, rec


def run_source(source: str, out_path: str, template_path: str, scene_ckpt: str, use_scene: bool):
    import cv2
    app = PoseAssistantV2(template_path, scene_ckpt, use_scene)
    if source.lower().endswith((".jpg", ".jpeg", ".png")):
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(source)
        out, rec = app.process_frame(frame, 0.0)
        cv2.imwrite(out_path, out)
        print(f"Đã lưu: {out_path} | gợi ý: {rec.pose_name if rec else 'không có'}")
    else:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer, i = None, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out, _ = app.process_frame(frame, i / fps)
            if writer is None:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
            writer.write(out)
            i += 1
        cap.release()
        if writer:
            writer.release()
        print(f"Đã xử lý {i} frame -> {out_path}")
    app.meter.report()


# ===========================================================================
# SELFTEST (chỉ cần numpy; chạy được không cần mediapipe/torch)
# ===========================================================================
def _skeleton(kind="stand", mirrored=False) -> dict:
    """Khung xương tổng hợp (đơn vị: chiều dài thân = 1)."""
    k = {"pelvis": (0, 0), "thorax": (0, -1), "l_shoulder": (-.5, -1), "r_shoulder": (.5, -1),
         "l_hip": (-.3, 0), "r_hip": (.3, 0), "l_knee": (-.3, .9), "r_knee": (.3, .9),
         "l_ankle": (-.3, 1.8), "r_ankle": (.3, 1.8),
         "l_elbow": (-.55, -.5), "l_wrist": (-.55, 0), "r_elbow": (.55, -.5), "r_wrist": (.55, 0)}
    if kind == "hip":  # tay phải chống hông
        k["r_elbow"], k["r_wrist"] = (1.0, -.5), (.4, -.05)
    if mirrored:
        k = {("r_" + n[2:] if n.startswith("l_") else "l_" + n[2:] if n.startswith("r_") else n): (-x, y)
             for n, (x, y) in k.items()}
    return k


def _live(kind, w=1280, h=720, scale=300, off=(640, 300), mirrored=False, noise=0.0, rng=None):
    k = _skeleton(kind, mirrored)
    lm = {}
    for n, (x, y) in k.items():
        px, py = off[0] + x * scale, off[1] + y * scale
        if noise:
            px, py = px + rng.normal(0, noise), py + rng.normal(0, noise)
        lm[n] = (px / w, py / h, 0.95)
    return lm, w, h


def selftest() -> int:
    import tempfile
    fails = []

    def check(name, cond):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    def make_templates(names_style):
        def conv(kd, s):
            return {(n.replace("l_", "left_", 1) if names_style == "mp" and n.startswith("l_") else
                     n.replace("r_", "right_", 1) if names_style == "mp" and n.startswith("r_") else n):
                    [x * s + 400, y * s + 300] for n, (x, y) in kd.items()}
        return {"templates": [
            {"template_id": "t_stand", "pose_name": "stand", "caption": "Đứng thẳng", "delta_hint": "x",
             "keypoints_reference": conv(_skeleton("stand"), 180), "scene_condition": {"location": "any"}},
            {"template_id": "t_hip", "pose_name": "hand_on_hip", "caption": "Chống hông", "delta_hint": "x",
             "keypoints_reference": conv(_skeleton("hip"), 180), "scene_condition": {"location": "any"}}]}

    for style in ("mpii", "mp"):  # 'mp' tái hiện đúng bug left_/right_ của file mock
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(make_templates(style), f, ensure_ascii=False)
        rec = PoseRecommenderV2(f.name)
        scene = SceneInfo("outdoor", 1)

        lm, w, h = _live("stand")
        r = rec.recommend(landmarks_to_px(lm, w, h), scene)
        check(f"[{style}] đứng thẳng -> template 'stand', score>0.9", r[0].pose_name == "stand" and r[0].similarity_score > 0.9)

        lm, w, h = _live("hip", w=1920, h=1080, scale=500, off=(900, 400))  # khác kích thước/tỉ lệ ảnh
        r = rec.recommend(landmarks_to_px(lm, w, h), scene)
        check(f"[{style}] chống hông (ảnh 16:9 khác cỡ) -> 'hand_on_hip'", r[0].pose_name == "hand_on_hip" and r[0].similarity_score > 0.9)

        lm, w, h = _live("hip", mirrored=True)
        r = rec.recommend(landmarks_to_px(lm, w, h), scene)
        check(f"[{style}] chống hông tay TRÁI -> khớp qua bản lật", r[0].pose_name == "hand_on_hip" and r[0].mirrored)

    # Hint tính từ sai lệch thật
    t = make_templates("mpii")
    t["templates"] = [t["templates"][1]]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(t, f, ensure_ascii=False)
    rec = PoseRecommenderV2(f.name)
    lm, w, h = _live("stand")
    r = rec.recommend(landmarks_to_px(lm, w, h), SceneInfo("outdoor", 1))[0]
    print("   hint:", r.delta_hint)
    check("hint nhắc đến khuỷu tay khi tay đang duỗi thẳng", r.delta_hint is not None and "tay" in r.delta_hint)

    # Smoothing giảm rung
    rng = np.random.default_rng(0)
    sm, raw_xs, sm_xs = LandmarkSmoother(), [], []
    for i in range(120):
        lm, w, h = _live("stand", noise=3.0, rng=rng)
        raw_xs.append(lm["nose"][0] if "nose" in lm else lm["l_shoulder"][0])
        sm_xs.append(sm.update(lm, i / 30)["l_shoulder"][0])
    check("One-Euro giảm độ rung (std) ≥ 50%", np.std(sm_xs[30:]) < 0.5 * np.std(raw_xs[30:]))

    # Hysteresis chống nhấp nháy
    def mk(a, b):
        return [Recommendation("A", "A", "", None, a, False, 1, "exact", False, {}),
                Recommendation("B", "B", "", None, b, False, 1, "exact", False, {})]
    sel, flips, cur = StableSelector(), 0, None
    for i in range(100):
        r = sel.update(sorted(mk(0.70 + 0.03 * (i % 2), 0.70 + 0.03 * ((i + 1) % 2)), key=lambda x: -x.similarity_score))
        flips += int(cur is not None and r.template_id != cur)
        cur = r.template_id
    check("hysteresis: điểm dao động nhỏ không làm đổi gợi ý", flips == 0)

    # Render (nếu có cv2)
    try:
        import cv2
        img = np.full((720, 1280, 3), 60, np.uint8)
        lm, w, h = _live("stand")
        kps = landmarks_to_px(lm, w, h)
        rr = PoseRecommenderV2(f.name).recommend(kps, SceneInfo("outdoor", 1))[0]
        out = draw_caption(draw_ghost(draw_skeleton(img.copy(), kps), rr, kps), rr, SceneInfo("outdoor", 1))
        check("render skeleton+ghost+caption chạy được", out.shape == img.shape and (out != img).any())
    except ImportError:
        print("[SKIP] không có cv2 — bỏ qua test render")

    print("\nTẤT CẢ TEST ĐỀU PASS" if not fails else f"\nFAIL: {fails}")
    return 1 if fails else 0


def _clean_argv(argv):
    """Colab/Jupyter chèn thêm '-f <kernel.json>' vào sys.argv -> loại bỏ để argparse không lỗi."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == "-f":
            skip = True
        else:
            out.append(a)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pose Suggestion Camera Assistant v2")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--source", help="đường dẫn ảnh/video hoặc chỉ số webcam")
    ap.add_argument("--out", default="/content/overlay_v2.mp4")
    ap.add_argument("--templates", default=TEMPLATE_PATH)
    ap.add_argument("--scene-ckpt", default=SCENE_CKPT)
    ap.add_argument("--no-scene", action="store_true")
    a = ap.parse_args(_clean_argv(sys.argv[1:] if argv is None else argv))
    if a.selftest:
        return selftest()
    if not a.source:
        ap.print_help()
        return 0
    run_source(a.source, a.out, a.templates, a.scene_ckpt, not a.no_scene)
    return 0


if __name__ == "__main__":
    rc = main()
    if rc and "ipykernel" not in sys.modules:
        sys.exit(rc)
