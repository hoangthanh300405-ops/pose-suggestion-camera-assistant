# -*- coding: utf-8 -*-
"""
06_export_coreml.py — Export SceneAnalyzer (MobileNetV3-Small, indoor/outdoor) sang Core ML cho iOS.

Chạy được trên Colab (chỉ CONVERT) hoặc trên Mac (CONVERT + kiểm tra độ lệch so với PyTorch;
Core ML chỉ chạy predict được trên macOS).

    pip install coremltools torch torchvision pillow
    python 06_export_coreml.py \
        --ckpt /content/drive/MyDrive/pose-assistant/checkpoints/scene_analyzer.pt \
        --out  /content/drive/MyDrive/pose-assistant/checkpoints/SceneAnalyzer.mlpackage

Thiết kế:
  - Chuẩn hoá ImageNet (mean/std) được NHÚNG vào model -> phía Swift chỉ cần đưa ảnh RGB 224x224
    (ImageType scale = 1/255), không phải tự normalize -> tránh sai lệch tiền xử lý giữa Python và Swift.
  - Softmax nhúng vào model, nhãn ["indoor","outdoor"] gắn qua ClassifierConfig.
  - minimum_deployment_target = iOS15 để chạy được cả iPhone X (iOS 16 là bản cuối).
  - Lưu ý: trong Colab phải TÁI LẬP đúng tiền xử lý lúc train: Resize((224,224)) (KHÔNG giữ tỉ lệ).
    Khi đưa frame camera vào Core ML (Vision), đặt imageCropAndScaleOption = .scaleFill để khớp.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

LABELS = ["indoor", "outdoor"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_model(ckpt: str) -> nn.Module:
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Không thấy checkpoint: {ckpt}")
    m = mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.eval()


class Wrapped(nn.Module):
    """Input: ảnh RGB trong [0,1], NCHW. Output: xác suất [indoor, outdoor]."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return torch.softmax(self.base(x), dim=1)


def convert(ckpt: str, out: str):
    import coremltools as ct

    model = Wrapped(load_model(ckpt)).eval()
    example = torch.rand(1, 3, 224, 224)
    traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1 / 255.0,
                             color_layout=ct.colorlayout.RGB)],
        classifier_config=ct.ClassifierConfig(LABELS),
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.ALL,   # CPU + GPU + Neural Engine
    )
    mlmodel.short_description = "SceneAnalyzer: indoor/outdoor (MobileNetV3-Small, fine-tuned head, Places365)"
    mlmodel.save(out)
    print(f"Đã lưu Core ML model: {out}")
    return model


def parity_check(torch_model: nn.Module, out: str, n: int = 20) -> None:
    """Chỉ chạy được trên macOS. So xác suất PyTorch vs Core ML trên ảnh ngẫu nhiên."""
    if sys.platform != "darwin":
        print("[SKIP] parity check cần macOS (Core ML predict). Chạy lại bước này trên Mac.")
        return
    import coremltools as ct
    from PIL import Image

    ml = ct.models.MLModel(out)
    rng = np.random.default_rng(0)
    max_diff, label_mismatch = 0.0, 0
    for _ in range(n):
        arr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        with torch.no_grad():
            p_torch = torch_model(torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255).numpy()[0]
        res = ml.predict({"image": Image.fromarray(arr)})
        probs = res[[k for k in res if k != "classLabel"][0]]
        p_ml = np.array([probs[l] for l in LABELS])
        max_diff = max(max_diff, float(np.abs(p_torch - p_ml).max()))
        label_mismatch += int(p_torch.argmax() != p_ml.argmax())
    print(f"Parity: max |Δprob| = {max_diff:.4f}, lệch nhãn {label_mismatch}/{n}")
    print("  -> OK" if max_diff < 0.02 and label_mismatch == 0 else "  -> CẦN KIỂM TRA (lệch quá ngưỡng)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/content/drive/MyDrive/pose-assistant/checkpoints/scene_analyzer.pt")
    ap.add_argument("--out", default="SceneAnalyzer.mlpackage")
    a = ap.parse_args()
    tm = convert(a.ckpt, a.out)
    parity_check(tm, a.out)
