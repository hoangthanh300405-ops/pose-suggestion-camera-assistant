# -*- coding: utf-8 -*-
"""
02_three_models.py  (BẢN FINAL - ĐÃ SỬA LỖI ĐỌC DRIVE)

Gộp 3 model trong pipeline vào 1 file, CHẠY SAU khi đã chạy xong
01_data_download_and_processing.py:

  MODEL 1 — PoseEstimator: benchmark MediaPipe Pose Landmarker trên COCO Keypoints
  MODEL 2 — SceneAnalyzer: benchmark MobileNetV3 (fine-tune head) trên Places365
  MODEL 3 — PoseRecommender: rule engine gợi ý tư thế
"""

import os
import json
import random
import logging
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import Places365
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from pycocotools.coco import COCO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("PoseAssistant")

BASE_DRIVE = '/content/drive/MyDrive/pose-assistant'

# ===========================================================================
# MODEL 1/3 — PoseEstimator: benchmark MediaPipe trên COCO Keypoints
# ===========================================================================
COCO_IMG_DIR = f'{BASE_DRIVE}/coco_keypoints/val2017'
COCO_ANN_FILE = f'{BASE_DRIVE}/coco_keypoints/person_keypoints_val2017.json'
POSE_MODEL_PATH = 'pose_landmarker.task'

POSE_NUM_SAMPLES = 200
POSE_PCK_THRESHOLD = 0.5
POSE_RANDOM_SEED = 42

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KEYPOINT_INDEX = {name: i for i, name in enumerate(COCO_KEYPOINT_NAMES)}

COCO_TO_MEDIAPIPE_IDX = {
    "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
}

def build_pose_landmarker():
    base_options = mp_tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)

def select_single_person_annotations(coco: COCO):
    ann_ids = coco.getAnnIds(iscrowd=False)
    all_anns = [a for a in coco.loadAnns(ann_ids) if a.get('num_keypoints', 0) > 0]
    anns_by_image = defaultdict(list)
    for a in all_anns:
        anns_by_image[a['image_id']].append(a)
    return [anns[0] for anns in anns_by_image.values() if len(anns) == 1]

def evaluate_instance(pred_landmarks, gt_keypoints, torso_diameter, img_w, img_h, threshold=POSE_PCK_THRESHOLD):
    if torso_diameter < 1e-6:
        return {}
    results = {}
    for name, idx in KEYPOINT_INDEX.items():
        gx, gy, gv = gt_keypoints[idx * 3], gt_keypoints[idx * 3 + 1], gt_keypoints[idx * 3 + 2]
        if gv == 0:
            continue
        mp_idx = COCO_TO_MEDIAPIPE_IDX[name]
        pred = pred_landmarks[mp_idx]
        px, py = pred.x * img_w, pred.y * img_h
        dist = np.hypot(px - gx, py - gy)
        results[name] = int(dist < threshold * torso_diameter)
    return results

def compute_torso_diameter(gt_keypoints):
    ls_x, ls_y, ls_v = gt_keypoints[KEYPOINT_INDEX["left_shoulder"] * 3: KEYPOINT_INDEX["left_shoulder"] * 3 + 3]
    rh_x, rh_y, rh_v = gt_keypoints[KEYPOINT_INDEX["right_hip"] * 3: KEYPOINT_INDEX["right_hip"] * 3 + 3]
    if ls_v == 0 or rh_v == 0:
        return None
    return float(np.hypot(ls_x - rh_x, ls_y - rh_y))

def run_pose_estimator_benchmark():
    if not os.path.exists(COCO_ANN_FILE):
        print(f"ERROR: Không tìm thấy {COCO_ANN_FILE}. Đảm bảo đã mount Drive và chạy 01_data_download_and_processing.py.")
        return
    coco = COCO(COCO_ANN_FILE)
    landmarker = build_pose_landmarker()
    single_person_anns = select_single_person_annotations(coco)
    print(f"Tổng annotation hợp lệ 1-người/ảnh: {len(single_person_anns)}")
    random.seed(POSE_RANDOM_SEED)
    sampled_anns = random.sample(single_person_anns, min(POSE_NUM_SAMPLES, len(single_person_anns)))
    per_keypoint_correct = {name: 0 for name in COCO_KEYPOINT_NAMES}
    per_keypoint_total = {name: 0 for name in COCO_KEYPOINT_NAMES}
    skipped_no_detection = skipped_no_torso = skipped_read_error = 0
    for ann in sampled_anns:
        img_info = coco.loadImgs(ann['image_id'])[0]
        img_path = f"{COCO_IMG_DIR}/{img_info['file_name']}"
        image = cv2.imread(img_path)
        if image is None:
            skipped_read_error += 1
            continue
        img_h, img_w = image.shape[:2]
        torso_diameter = compute_torso_diameter(ann['keypoints'])
        if torso_diameter is None:
            skipped_no_torso += 1
            continue
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)
        if not result.pose_landmarks:
            skipped_no_detection += 1
            continue
        pred_landmarks = result.pose_landmarks[0]
        per_instance_results = evaluate_instance(pred_landmarks, ann['keypoints'], torso_diameter, img_w, img_h)
        for name, correct in per_instance_results.items():
            per_keypoint_total[name] += 1
            per_keypoint_correct[name] += correct
    total_skipped = skipped_no_detection + skipped_no_torso + skipped_read_error
    evaluated = len(sampled_anns) - total_skipped
    print(f"\nĐã benchmark trên {evaluated}/{len(sampled_anns)} ảnh")
    print(f"  Bỏ qua vì không phát hiện được người: {skipped_no_detection}")
    print(f"  Bỏ qua vì thiếu vai/hông để tính torso: {skipped_no_torso}")
    print(f"  Bỏ qua vì lỗi đọc file ảnh: {skipped_read_error}\n")
    print(f"{'Keypoint':<16} {'PCK':>8} {'N':>8}")
    print("-" * 34)
    total_correct, total_count = 0, 0
    for name in COCO_KEYPOINT_NAMES:
        n = per_keypoint_total[name]
        c = per_keypoint_correct[name]
        pck = c / n if n > 0 else float('nan')
        print(f"{name:<16} {pck:>8.3f} {n:>8}")
        total_correct += c
        total_count += n
    overall_pck = total_correct / total_count if total_count > 0 else float('nan')
    print("-" * 34)
    print(f"{'OVERALL':<16} {overall_pck:>8.3f} {total_count:>8}")
    return overall_pck

# ===========================================================================
# MODEL 2/3 — SceneAnalyzer: benchmark MobileNetV3 (fine-tune head) trên Places365
# ===========================================================================
PLACES_ROOT = f'{BASE_DRIVE}/places365/torchvision_val'
IO_MAPPING_FILE = 'IO_places365.txt'
SCENE_CALIB_FRACTION = 0.7
SCENE_NUM_EPOCHS = 3
SCENE_BATCH_SIZE = 64
SCENE_LR = 1e-3
SCENE_RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SCENE_CLASS_NAMES = ["indoor", "outdoor"]

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def download_io_mapping():
    if not os.path.exists(IO_MAPPING_FILE):
        os.system(f'wget -q -O {IO_MAPPING_FILE} https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt')

def load_io_labels(class_to_idx):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    category_names, io_labels = [], []
    with open(IO_MAPPING_FILE) as f:
        for line in f:
            items = line.rstrip().split()
            category_names.append(items[0])
            io_labels.append(int(items[-1]) - 1)
    io_labels = np.array(io_labels)
    return io_labels

class IndoorOutdoorWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, io_labels):
        self.base_dataset = base_dataset
        self.io_labels = io_labels

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                image, label_365 = self.base_dataset[idx]
                return image, int(self.io_labels[label_365])
            except OSError as e:
                if attempt == max_retries - 1:
                    # Nếu quá số lần thử lại vẫn lỗi Drive, trả về một ảnh ngẫu nhiên khác
                    alt_idx = random.randint(0, len(self.base_dataset) - 1)
                    image, label_365 = self.base_dataset[alt_idx]
                    return image, int(self.io_labels[label_365])
                time.sleep(1)

def build_scene_datasets():
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    base_dataset = Places365(root=PLACES_ROOT, split='val', small=True, download=False, transform=transform)
    io_labels = load_io_labels(class_to_idx=base_dataset.class_to_idx)
    full_dataset = IndoorOutdoorWrapper(base_dataset, io_labels)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    split_point = int(len(indices) * SCENE_CALIB_FRACTION)
    calib_dataset = Subset(full_dataset, indices[:split_point])
    test_dataset = Subset(full_dataset, indices[split_point:])
    return calib_dataset, test_dataset, io_labels

def report_class_balance(dataset, io_labels, split_name):
    base_dataset = dataset.dataset.base_dataset
    indices = dataset.indices
    counts = {0: 0, 1: 0}
    for idx in indices:
        label_365 = base_dataset.targets[idx]
        counts[int(io_labels[label_365])] += 1
    total = sum(counts.values())
    majority_class = max(counts, key=counts.get)
    baseline_acc = counts[majority_class] / total
    print(f"[{split_name}] Phân bố lớp: indoor={counts[0]} ({counts[0]/total:.1%}), outdoor={counts[1]} ({counts[1]/total:.1%})")
    print(f"[{split_name}] Baseline accuracy = {baseline_acc:.4f}\n")
    return baseline_acc

def build_scene_model():
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model.to(DEVICE)

def train_scene_head(model, calib_loader):
    optimizer = torch.optim.Adam(model.classifier[-1].parameters(), lr=SCENE_LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(SCENE_NUM_EPOCHS):
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in calib_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
        print(f"Epoch {epoch+1}/{SCENE_NUM_EPOCHS} — loss: {total_loss/total:.4f}, acc: {correct/total:.4f}")

def evaluate_scene_model(model, test_loader, baseline_acc):
    model.eval()
    correct, total = 0, 0
    confusion = np.zeros((2, 2), dtype=int)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1
    accuracy = correct / total
    print(f"\nAccuracy trên tập test: {accuracy:.4f} ({correct}/{total})")
    print(f"So với baseline: {baseline_acc:.4f}\n")
    print("Confusion matrix:")
    print(f"{'':>10}{'pred_indoor':>14}{'pred_outdoor':>14}")
    print(f"{'indoor':>10}{confusion[0,0]:>14}{confusion[0,1]:>14}")
    print(f"{'outdoor':>10}{confusion[1,0]:>14}{confusion[1,1]:>14}\n")
    return accuracy, confusion

def run_scene_analyzer_benchmark():
    if not os.path.exists(PLACES_ROOT):
        print(f"ERROR: Không tìm thấy {PLACES_ROOT}. Đảm bảo đã chạy 01_data_download_and_processing.py.")
        return
    set_all_seeds(SCENE_RANDOM_SEED)
    download_io_mapping()
    calib_dataset, test_dataset, io_labels = build_scene_datasets()
    report_class_balance(calib_dataset, io_labels, "Calibration")
    baseline_acc = report_class_balance(test_dataset, io_labels, "Test")
    # Đặt num_workers=0 để tránh tối đa lỗi nghẽn Google Drive FUSE
    calib_loader = DataLoader(calib_dataset, batch_size=SCENE_BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=SCENE_BATCH_SIZE, shuffle=False, num_workers=0)
    model = build_scene_model()
    train_scene_head(model, calib_loader)
    evaluate_scene_model(model, test_loader, baseline_acc)
    checkpoint_path = f'{BASE_DRIVE}/checkpoints/scene_analyzer.pt'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nĐã lưu model tại: {checkpoint_path}")

# ===========================================================================
# MODEL 3/3 — PoseRecommender: rule engine gợi ý tư thế
# ===========================================================================
MPII_TO_MEDIAPIPE = {
    "r_ankle": "right_ankle", "r_knee": "right_knee", "r_hip": "right_hip",
    "l_hip": "left_hip", "l_knee": "left_knee", "l_ankle": "left_ankle",
    "r_wrist": "right_wrist", "r_elbow": "right_elbow", "r_shoulder": "right_shoulder",
    "l_shoulder": "left_shoulder", "l_elbow": "left_elbow", "l_wrist": "left_wrist",
}
DEFAULT_VISIBILITY_THRESHOLD = 0.5
TEMPLATE_PATH = f'{BASE_DRIVE}/templates/pose_templates_v1.json'

def mediapipe_to_common_format(landmarks: Dict[str, Tuple[float, float, float]]) -> dict:
    kps = {}
    for mpii_name, mp_name in MPII_TO_MEDIAPIPE.items():
        if mp_name in landmarks:
            kps[mpii_name] = landmarks[mp_name]
    if "l_hip" in kps and "r_hip" in kps:
        lx, ly, lv = kps["l_hip"]
        rx, ry, rv = kps["r_hip"]
        kps["pelvis"] = ((lx + rx) / 2, (ly + ry) / 2, min(lv, rv))
    if "l_shoulder" in kps and "r_shoulder" in kps:
        lx, ly, lv = kps["l_shoulder"]
        rx, ry, rv = kps["r_shoulder"]
        kps["thorax"] = ((lx + rx) / 2, (ly + ry) / 2, min(lv, rv))
    if "thorax" in kps:
        kps["upper_neck"] = kps["thorax"]
    if "nose" in landmarks and "thorax" in kps:
        nx, ny, nv = landmarks["nose"]
        tx, ty, tv = kps["thorax"]
        kps["head_top"] = (nx, ny - abs(ty - ny) * 0.5, min(nv, tv))
    return kps

def filter_low_confidence(kps: dict, threshold: float = DEFAULT_VISIBILITY_THRESHOLD) -> dict:
    filtered = {}
    for name, val in kps.items():
        visibility = val[2] if len(val) > 2 else 1.0
        if visibility >= threshold:
            filtered[name] = val
    return filtered

def average_confidence(kps: dict) -> float:
    visibilities = [v[2] if len(v) > 2 else 1.0 for v in kps.values()]
    return float(np.mean(visibilities)) if visibilities else 0.0

def normalize_keypoints(kps: dict) -> Optional[Dict[str, Tuple[float, float]]]:
    if "pelvis" not in kps or "thorax" not in kps:
        return None
    px, py = kps["pelvis"][:2]
    tx, ty = kps["thorax"][:2]
    scale = float(np.hypot(tx - px, ty - py))
    if scale < 1e-6:
        return None
    return {name: ((v[0] - px) / scale, (v[1] - py) / scale) for name, v in kps.items()}

def mirror_keypoints(norm_kps: dict) -> dict:
    mirrored = {}
    for name, (x, y) in norm_kps.items():
        swapped = name.replace("l_", "r_") if name.startswith("l_") else (
            name.replace("r_", "l_") if name.startswith("r_") else name
        )
        mirrored[swapped] = (-x, y)
    return mirrored

def pose_similarity(kps_a: dict, kps_b: dict, decay_rate: float = 2.0) -> float:
    common_joints = set(kps_a.keys()) & set(kps_b.keys())
    if len(common_joints) < 4:
        return 0.0
    dists = [np.hypot(kps_a[j][0] - kps_b[j][0], kps_a[j][1] - kps_b[j][1]) for j in common_joints]
    mean_dist = float(np.mean(dists))
    return float(np.exp(-mean_dist * decay_rate))

def best_similarity_with_mirror(live_norm: dict, ref_norm: dict, decay_rate: float = 2.0) -> float:
    direct = pose_similarity(live_norm, ref_norm, decay_rate)
    mirrored = pose_similarity(mirror_keypoints(live_norm), ref_norm, decay_rate)
    return max(direct, mirrored)

@dataclass
class SceneInfo:
    location: str
    num_people: int
    lighting: str = "any"

@dataclass
class Recommendation:
    template_id: str
    pose_name: str
    caption: str
    delta_hint: str
    similarity_score: float
    is_close_match: bool
    confidence: float
    scene_filter_level: str

class PoseRecommender:
    def __init__(self, template_path: str, close_match_threshold: float = 0.85, decay_rate: float = 2.0, visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD):
        with open(template_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.templates = data["templates"]
        self.close_match_threshold = close_match_threshold
        self.decay_rate = decay_rate
        self.visibility_threshold = visibility_threshold
        for t in self.templates:
            ref_filtered = filter_low_confidence(t["keypoints_reference"], visibility_threshold)
            t["_normalized_ref"] = normalize_keypoints(ref_filtered)

    def _filter_by_scene(self, scene: SceneInfo) -> Tuple[List[dict], str]:
        def matches(t, ignore_lighting=False, ignore_people=False, ignore_location=False):
            cond = t.get("scene_condition", {})
            if not ignore_location and cond.get("location") not in (scene.location, "any", None):
                return False
            if not ignore_people and cond.get("num_people") not in (scene.num_people, "any", None):
                return False
            if not ignore_lighting and cond.get("lighting") not in (scene.lighting, "any", None):
                return False
            return True
        levels = [
            ("exact", dict(ignore_lighting=False, ignore_people=False, ignore_location=False)),
            ("relaxed_lighting", dict(ignore_lighting=True, ignore_people=False, ignore_location=False)),
            ("relaxed_lighting_people", dict(ignore_lighting=True, ignore_people=True, ignore_location=False)),
            ("relaxed_all", dict(ignore_lighting=True, ignore_people=True, ignore_location=True)),
        ]
        for level_name, kwargs in levels:
            matched = [t for t in self.templates if matches(t, **kwargs)]
            if matched:
                return matched, level_name
        return [], "relaxed_all"

    def recommend(self, live_landmarks: Dict[str, Tuple[float, float, float]], scene: SceneInfo, top_k: int = 3) -> List[Recommendation]:
        common_kps = mediapipe_to_common_format(live_landmarks)
        common_kps_filtered = filter_low_confidence(common_kps, self.visibility_threshold)
        confidence = average_confidence(common_kps_filtered)
        live_norm = normalize_keypoints(common_kps_filtered)
        if live_norm is None:
            return []
        candidates, filter_level = self._filter_by_scene(scene)
        if not candidates:
            return []
        scored = []
        for t in candidates:
            ref_norm = t["_normalized_ref"]
            score = 0.0 if ref_norm is None else best_similarity_with_mirror(live_norm, ref_norm, self.decay_rate)
            scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            Recommendation(
                template_id=t["template_id"], pose_name=t["pose_name"],
                caption=t["caption"], delta_hint=t["delta_hint"],
                similarity_score=round(score, 3), is_close_match=score >= self.close_match_threshold,
                confidence=round(confidence, 3), scene_filter_level=filter_level,
            )
            for score, t in scored[:top_k]
        ]

def run_pose_recommender_demo():
    if not os.path.exists(TEMPLATE_PATH):
        print(f"ERROR: Không tìm thấy {TEMPLATE_PATH}. Cần hoàn tất tuyển chọn pose_templates_v1.json trước.")
        return
    fake_live_landmarks = {
        "left_shoulder": (0.45, 0.30, 0.99), "right_shoulder": (0.55, 0.30, 0.99),
        "left_hip": (0.46, 0.55, 0.95), "right_hip": (0.54, 0.55, 0.95),
        "left_elbow": (0.44, 0.42, 0.90), "right_elbow": (0.56, 0.42, 0.90),
        "left_wrist": (0.43, 0.60, 0.85), "right_wrist": (0.57, 0.60, 0.85),
        "left_knee": (0.47, 0.75, 0.90), "right_knee": (0.53, 0.75, 0.90),
        "left_ankle": (0.47, 0.95, 0.85), "right_ankle": (0.53, 0.95, 0.85),
        "nose": (0.50, 0.20, 0.99),
    }
    scene = SceneInfo(location="outdoor", num_people=1, lighting="any")
    recommender = PoseRecommender(TEMPLATE_PATH)
    recs = recommender.recommend(fake_live_landmarks, scene, top_k=3)
    for r in recs:
        print(f"[{r.similarity_score}] {r.pose_name} — {r.caption} (confidence={r.confidence}, scene_filter={r.scene_filter_level})")
        if not r.is_close_match:
            print(f"    Gợi ý chỉnh: {r.delta_hint}")

if __name__ == "__main__":
    print("=" * 70)
    print("MODEL 1/3: PoseEstimator benchmark (COCO Keypoints)")
    print("=" * 70)
    run_pose_estimator_benchmark()

    print("\n" + "=" * 70)
    print("MODEL 2/3: SceneAnalyzer benchmark (Places365)")
    print("=" * 70)
    run_scene_analyzer_benchmark()

    print("\n" + "=" * 70)
    print("MODEL 3/3: PoseRecommender demo")
    print("=" * 70)
    run_pose_recommender_demo()
