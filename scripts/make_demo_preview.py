# Tạo docs/assets/overlay_preview_synthetic.png từ DỮ LIỆU TỔNG HỢP bằng đúng code render của src/05_upgrade_v2.py.
# Đây chỉ minh hoạ giao diện overlay; KHÔNG phải kết quả phát hiện trên ảnh thật.
# Chạy: python scripts/make_demo_preview.py   (cần numpy, opencv-python, pillow)
import sys, json, tempfile, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, cv2, importlib
m = importlib.import_module("05_upgrade_v2")

W, H = 540, 960
def bg():
    img = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        t = y / H
        img[y, :] = (int(70 + 40 * t), int(62 + 30 * t), int(55 + 20 * t))
    cv2.rectangle(img, (0, 770), (W, H), (52, 58, 52), -1)   # "mặt đất"
    return img

# Template: chỉ 1 tư thế "chống hông"
sk = m._skeleton("hip")
tpl = {"templates": [{"template_id": "demo_hip", "pose_name": "hand_on_hip", "caption": "Thử chống một tay lên hông",
       "delta_hint": "", "keypoints_reference": {n: [x * 180 + 400, y * 180 + 300] for n, (x, y) in sk.items()},
       "scene_condition": {"location": "any"}}]}
f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8"); json.dump(tpl, f); f.close()
rec_engine = m.PoseRecommenderV2(f.name)

def panel(kind, title):
    img = bg()
    lm, w, h = m._live(kind, w=W, h=H, scale=215, off=(270, 400))
    kps = m.landmarks_to_px(lm, w, h)
    scene = m.SceneInfo("outdoor", 1)
    rec = rec_engine.recommend(kps, scene)[0]
    img = m.draw_skeleton(img, kps)
    hx, hy = kps["thorax"][0], kps["thorax"][1] - 0.5 * 215
    cv2.circle(img, (int(hx), int(hy)), 36, (0, 255, 0), 3)
    img = m.draw_ghost(img, rec, kps)
    img = m.draw_caption(img, rec, scene)
    print(title, "score", rec.similarity_score, "close", rec.is_close_match, "hint", rec.delta_hint)
    return img

left = panel("stand", "A")
right = panel("hip", "B")
sep = np.full((H, 8, 3), 255, np.uint8)
row = np.hstack([left, sep, right])
bar = np.full((70, row.shape[1], 3), 24, np.uint8)
row = np.vstack([bar, row])
row = m.draw_texts(row, [("Trái: chưa khớp → có gợi ý chỉnh", (14, 6), 18, (255, 255, 255)),
                         ("Phải: đã khớp tư thế mẫu (✓)", (W + 22, 6), 18, (255, 255, 255)),
                         ("Xanh lá = người dùng · xanh dương = tư thế mẫu (ghost)", (14, 30), 15, (200, 200, 200)),
                         ("ẢNH MINH HOẠ — dữ liệu tổng hợp, không phải kết quả phát hiện trên ảnh thật", (14, 49), 15, (0, 215, 255))])
os.makedirs(os.path.join(ROOT, "docs", "assets"), exist_ok=True)
out = os.path.join(ROOT, "docs", "assets", "overlay_preview_synthetic.png")
cv2.imwrite(out, row); print(out, row.shape)
