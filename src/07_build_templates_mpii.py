# -*- coding: utf-8 -*-
"""
07_build_templates_mpii.py — Dựng pose_templates_v1.json THẬT từ MPII Human Pose.

Quy trình 3 bước (mỗi bước chạy độc lập, kết quả lưu ra Drive nên không mất khi Colab reset):

  1) scan   : tự dò file .mat + thư mục ảnh MPII trên Drive, lọc ứng viên (ảnh 1 người, đủ thân
              và chân, activity liên quan tư thế đứng/tạo dáng) -> lưu candidates.json (ID ổn định).
  2) sheet  : vẽ "contact sheet" (lưới ảnh cắt quanh người + khung xương + số ID) -> bạn xem và chọn.
  3) build  : đọc template_selection.json do BẠN điền (ID + tên + caption + điều kiện scene)
              -> ghi pose_templates_v1.json. Từ chối ghi đè nếu số template mới ít hơn file hiện có;
              luôn sao lưu bản cũ.

Dùng trong notebook (khuyên dùng) — import rồi gọi hàm, KHÔNG dán code vào ô:
    import sys, importlib; sys.path.insert(0, '/content/drive/MyDrive/pose-assistant/code')
    m07 = importlib.import_module('07_build_templates_mpii')
    m07.scan(); m07.sheet(0); m07.build(SELECTION)
Hoặc dòng lệnh:
    !python 07_build_templates_mpii.py scan | sheet --page 0 | build | --selftest

LƯU Ý LICENSE: ảnh MPII chỉ dùng cho nghiên cứu (non-commercial). Template chỉ lưu TOẠ ĐỘ khớp,
không lưu ảnh, nhưng vẫn ghi rõ nguồn trong README.
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime
from types import SimpleNamespace

import numpy as np

BASE_DRIVE = "/content/drive/MyDrive/pose-assistant"
TEMPLATE_DIR = f"{BASE_DRIVE}/templates"
WORK_DIR = f"{TEMPLATE_DIR}/mpii_work"
OUT_TEMPLATES = f"{TEMPLATE_DIR}/pose_templates_v1.json"
CAND_FILE = f"{WORK_DIR}/candidates.json"
SELECTION_FILE = f"{TEMPLATE_DIR}/template_selection.json"

MAT_NAME = "mpii_human_pose_v1_u12_1.mat"
MPII_JOINTS = ["r_ankle", "r_knee", "r_hip", "l_hip", "l_knee", "l_ankle",
               "pelvis", "thorax", "upper_neck", "head_top",
               "r_wrist", "r_elbow", "r_shoulder", "l_shoulder", "l_elbow", "l_wrist"]
REQUIRED = {"l_shoulder", "r_shoulder", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
            "pelvis", "thorax"}
MIN_JOINTS = 14
# Dùng gốc từ: MPII ghi "dancing"/"ballroom dancing" nên từ khoá "dance" (bản notebook cũ) KHÔNG khớp.
DEFAULT_KEYWORDS = ["stand", "pose", "posing", "danc", "yoga", "walk"]
SKELETON = [("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
            ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"),
            ("r_shoulder", "r_hip"), ("l_hip", "r_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
            ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("thorax", "upper_neck"), ("upper_neck", "head_top")]


class TemplateError(RuntimeError):
    """Lỗi có thông báo rõ ràng cho người dùng (notebook hiện thông báo, CLI thoát mã 1)."""


# --------------------------------------------------------------------------
# Tiện ích đọc cấu trúc .mat (struct hoặc mảng struct, có thể thiếu field)
# --------------------------------------------------------------------------
def as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return list(x.ravel())
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def get(o, name, default=None):
    return getattr(o, name, default)


def _num(v, default=None):
    try:
        if isinstance(v, np.ndarray) and v.size == 0:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def extract_person(rect):
    """-> {joint: [x, y, vis]} hoặc None nếu không có annopoints."""
    ap = get(rect, "annopoints")
    if ap is None or isinstance(ap, np.ndarray):  # mảng rỗng = không có nhãn khớp
        return None
    kps = {}
    for p in as_list(get(ap, "point")):
        jid = _num(get(p, "id"))
        x, y = _num(get(p, "x")), _num(get(p, "y"))
        if jid is None or x is None or y is None or not (0 <= int(jid) < 16):
            continue
        vis = _num(get(p, "is_visible"), 1.0)
        kps[MPII_JOINTS[int(jid)]] = [x, y, vis]
    return kps or None


def person_box(rect):
    """Hộp cắt quanh người: (cx, cy, half_size) theo quy ước MPII (scale*200 ≈ chiều cao người)."""
    pos, scale = get(rect, "objpos"), _num(get(rect, "scale"))
    if pos is None or scale is None or isinstance(pos, np.ndarray):
        return None
    cx, cy = _num(get(pos, "x")), _num(get(pos, "y"))
    if cx is None or cy is None:
        return None
    return cx, cy, 0.75 * 200 * scale


def scan_candidates(release, keywords=None, any_activity=False):
    keywords = [k.lower() for k in (keywords or DEFAULT_KEYWORDS)]
    annolist, act, img_train = release.annolist, release.act, release.img_train
    cands, stats = [], dict(total=len(annolist), not_train=0, not_single=0, no_kps=0, incomplete=0, no_activity=0)
    for i in range(len(annolist)):
        if not int(img_train[i]):
            stats["not_train"] += 1
            continue
        rects = as_list(get(annolist[i], "annorect"))
        if len(rects) != 1:
            stats["not_single"] += 1
            continue
        kps = extract_person(rects[0])
        if kps is None:
            stats["no_kps"] += 1
            continue
        if len(kps) < MIN_JOINTS or not REQUIRED.issubset(kps):
            stats["incomplete"] += 1
            continue
        a = act[i]
        cat = get(a, "cat_name", "") if not isinstance(a, np.ndarray) else ""
        name = get(a, "act_name", "") if not isinstance(a, np.ndarray) else ""
        cat, name = (cat if isinstance(cat, str) else ""), (name if isinstance(name, str) else "")
        if not any_activity and not any(k in f"{cat} {name}".lower() for k in keywords):
            stats["no_activity"] += 1
            continue
        cands.append({"id": len(cands), "index": i, "image_name": annolist[i].image.name,
                      "category": cat, "activity": name, "box": person_box(rects[0]), "keypoints": kps})
    return cands, stats


# --------------------------------------------------------------------------
# Tìm dữ liệu trên Drive
# --------------------------------------------------------------------------
def find_mpii_paths(root=f"{BASE_DRIVE}/mpii_dataset"):
    if not os.path.isdir(root):
        root = BASE_DRIVE
    mat, img_dir = None, None
    for r, dirs, files in os.walk(root):
        if mat is None and MAT_NAME in files:
            mat = os.path.join(r, MAT_NAME)
        if img_dir is None and sum(f.lower().endswith(".jpg") for f in files[:200]) >= 50 \
                and "val2017" not in r and "places365" not in r.lower():
            img_dir = r
        if mat and img_dir:
            break
    return mat, img_dir


def scan(keywords=None, any_activity=False, max_candidates=240, seed=42):
    """Dò MPII trên Drive, lọc ứng viên, lưu candidates.json (ID ổn định cho các bước sau)."""
    import random
    import scipy.io as sio
    mat, img_dir = find_mpii_paths()
    print(f"File .mat    : {mat}\nThư mục ảnh  : {img_dir}")
    if mat is None:
        raise TemplateError(f"Không thấy {MAT_NAME} dưới {BASE_DRIVE}/mpii_dataset. Giải nén mpii_human_pose_v1_u12_2.zip vào đó.")
    if img_dir is None:
        raise TemplateError("Không thấy thư mục ảnh MPII (cần ≥50 file .jpg). Giải nén mpii_human_pose_v1.tar.gz vào mpii_dataset/images.")
    release = sio.loadmat(mat, struct_as_record=False, squeeze_me=True)["RELEASE"]
    cands, stats = scan_candidates(release, keywords, any_activity)
    print("Thống kê lọc:", stats, f"-> {len(cands)} ứng viên")
    if not cands:
        raise TemplateError("Không có ứng viên — thử scan(any_activity=True).")
    if len(cands) > max_candidates:
        rng = random.Random(seed)
        cands = sorted(rng.sample(cands, max_candidates), key=lambda c: c["index"])
        for new_id, c in enumerate(cands):
            c["id"] = new_id
        print(f"Lấy mẫu ngẫu nhiên {max_candidates} ứng viên (seed={seed}) để duyệt cho gọn; ID đã đánh lại 0..{max_candidates-1}.")
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(CAND_FILE, "w", encoding="utf-8") as f:
        json.dump({"mat": mat, "img_dir": img_dir, "candidates": cands}, f, ensure_ascii=False)
    print(f"Đã lưu {CAND_FILE} | số trang contact sheet: {(len(cands) + 11) // 12} (đánh số từ 0)")
    return cands


# --------------------------------------------------------------------------
# Contact sheet
# --------------------------------------------------------------------------
def render_tile(img, cand):
    import cv2
    h, w = img.shape[:2]
    cx, cy, half = cand["box"] or (w / 2, h / 2, max(w, h) / 2)
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    if x1 - x0 < 10 or y1 - y0 < 10:
        x0, y0, x1, y1 = 0, 0, w, h
    crop = img[y0:y1, x0:x1].copy()
    k = cand["keypoints"]
    pt = lambda n: (int(k[n][0] - x0), int(k[n][1] - y0))
    for a, b in SKELETON:
        if a in k and b in k:
            cv2.line(crop, pt(a), pt(b), (0, 255, 0), 2)
    for n in k:
        cv2.circle(crop, pt(n), 3, (0, 0, 255), -1)
    tile = cv2.resize(crop, (300, 300))
    cv2.rectangle(tile, (0, 0), (300, 26), (0, 0, 0), -1)
    cv2.putText(tile, f"#{cand['id']}  {cand['activity'][:22]}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return tile


def make_sheet(cands, img_dir, page, per_page=12, cols=4):
    import cv2
    chunk = cands[page * per_page:(page + 1) * per_page]
    if not chunk:
        return None
    tiles = []
    for c in chunk:
        img = cv2.imread(os.path.join(img_dir, c["image_name"]))
        tiles.append(render_tile(img, c) if img is not None else np.zeros((300, 300, 3), np.uint8))
    while len(tiles) % cols:
        tiles.append(np.zeros((300, 300, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def sheet(page=0):
    """Vẽ + hiển thị 1 trang contact sheet (12 ảnh). Lặp: for p in range(5): m07.sheet(p)"""
    import cv2
    if not os.path.exists(CAND_FILE):
        raise TemplateError("Chưa có candidates.json — chạy scan() trước.")
    with open(CAND_FILE, encoding="utf-8") as f:
        data = json.load(f)
    total_pages = (len(data["candidates"]) + 11) // 12
    img = make_sheet(data["candidates"], data["img_dir"], page)
    if img is None:
        raise TemplateError(f"Không có trang {page} (tổng {total_pages} trang, đánh số từ 0).")
    os.makedirs(f"{WORK_DIR}/sheets", exist_ok=True)
    out = f"{WORK_DIR}/sheets/page_{page:03d}.png"
    cv2.imwrite(out, img)
    print(f"Trang {page}/{total_pages - 1} -> {out}")
    try:
        from IPython.display import Image as IPImage, display
        display(IPImage(filename=out))
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def validate_selection(sel, cand_by_id):
    errs = []
    seen = set()
    for n, s in enumerate(sel):
        for key in ("candidate_id", "pose_name", "caption"):
            if key not in s:
                errs.append(f"Mục {n}: thiếu '{key}'")
        cid = s.get("candidate_id")
        if cid not in cand_by_id:
            errs.append(f"Mục {n}: candidate_id={cid} không tồn tại")
        if cid in seen:
            errs.append(f"Mục {n}: candidate_id={cid} bị trùng")
        seen.add(cid)
        if s.get("location", "any") not in ("any", "indoor", "outdoor"):
            errs.append(f"Mục {n}: location phải là any/indoor/outdoor")
    return errs


def build_payload(sel, cand_by_id):
    templates = []
    for s in sel:
        c = cand_by_id[s["candidate_id"]]
        templates.append({
            "template_id": f"mpii_{c['index']}", "pose_name": s["pose_name"],
            "source_image": c["image_name"], "mpii_activity": c["activity"],
            "keypoints_reference": c["keypoints"],
            "scene_condition": {"location": s.get("location", "any"), "num_people": s.get("num_people", 1),
                                "lighting": s.get("lighting", "any")},
            "caption": s["caption"],
            "delta_hint": s.get("delta_hint", "Điều chỉnh theo khung xương mờ")})
    return {"version": "1.1", "created_at": datetime.now().isoformat(), "num_templates": len(templates),
            "source": "MPII Human Pose (research/non-commercial) — chỉ lưu toạ độ khớp", "templates": templates}


def write_templates(payload, out_path, force=False):
    old_n = 0
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                old_n = len(json.load(f).get("templates", []))
        except Exception:
            old_n = 0
        if payload["num_templates"] < old_n and not force:
            raise TemplateError(f"[TỪ CHỐI] File hiện có {old_n} template, bản mới chỉ {payload['num_templates']}. "
                                "Dùng force=True nếu thực sự muốn thay.")
        bak = f"{out_path}.bak_{datetime.now():%Y%m%d_%H%M%S}"
        shutil.copy2(out_path, bak)
        print(f"Đã sao lưu bản cũ: {bak}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Đã ghi {payload['num_templates']} template -> {out_path}")


def build(selection=None, force=False):
    """selection: list dict [{'candidate_id':..,'pose_name':..,'caption':..,'location':..}, ...]
    Nếu None thì đọc template_selection.json."""
    if not os.path.exists(CAND_FILE):
        raise TemplateError("Chưa có candidates.json — chạy scan() trước.")
    with open(CAND_FILE, encoding="utf-8") as f:
        cand_by_id = {c["id"]: c for c in json.load(f)["candidates"]}
    if selection is None:
        if not os.path.exists(SELECTION_FILE):
            raise TemplateError(f"Chưa có {SELECTION_FILE}. Truyền selection=[...] hoặc tạo file đó.")
        with open(SELECTION_FILE, encoding="utf-8") as f:
            selection = json.load(f)
    errs = validate_selection(selection, cand_by_id)
    if errs:
        raise TemplateError("Lỗi file chọn:\n  " + "\n  ".join(errs))
    if len(selection) < 10:
        print(f"[CẢNH BÁO] Mới có {len(selection)} template (mục tiêu 15–20).")
    with open(SELECTION_FILE, "w", encoding="utf-8") as f:  # lưu lại lựa chọn để không mất khi Colab reset
        json.dump(selection, f, ensure_ascii=False, indent=2)
    payload = build_payload(selection, cand_by_id)
    write_templates(payload, OUT_TEMPLATES, force)
    return payload


# --------------------------------------------------------------------------
# Selftest (dữ liệu giả mô phỏng cấu trúc scipy.io.loadmat)
# --------------------------------------------------------------------------
def _fake_release(n=6):
    def pt(i, x, y, vis=1):
        return SimpleNamespace(id=i, x=x, y=y, is_visible=vis)
    full = lambda dx: [pt(j, 100 + dx + 5 * j, 50 + 10 * j) for j in range(16)]
    annolist, act, train = [], [], []
    for i in range(n):
        rect = SimpleNamespace(annopoints=SimpleNamespace(point=np.array(full(i), dtype=object)),
                               objpos=SimpleNamespace(x=150, y=150), scale=0.9)
        if i == 3:
            rect = [rect, rect]                 # 2 người -> loại
        if i == 4:
            rect = SimpleNamespace(annopoints=np.array([]))  # không nhãn -> loại
        annolist.append(SimpleNamespace(image=SimpleNamespace(name=f"{i:05d}.jpg"), annorect=rect))
        act.append(SimpleNamespace(cat_name="dancing", act_name="ballroom dancing") if i != 5 else np.array([]))
        train.append(1)
    return SimpleNamespace(annolist=annolist, act=act, img_train=train)


def selftest():
    import tempfile
    fails = []

    def check(name, ok):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            fails.append(name)

    cands, stats = scan_candidates(_fake_release())
    check("scan: loại ảnh 2 người, ảnh không nhãn, ảnh không activity (còn 3/6)", len(cands) == 3)
    check("scan: ID liên tục 0..n-1", [c["id"] for c in cands] == [0, 1, 2])
    by_id = {c["id"]: c for c in cands}
    sel = [{"candidate_id": 0, "pose_name": "a", "caption": "A"}, {"candidate_id": 2, "pose_name": "b", "caption": "B", "location": "indoor"}]
    check("validate: hợp lệ", validate_selection(sel, by_id) == [])
    check("validate: bắt ID sai/trùng/location sai",
          len(validate_selection([{"candidate_id": 99, "pose_name": "x", "caption": "x"},
                                  {"candidate_id": 0, "pose_name": "x", "caption": "x", "location": "xx"},
                                  {"candidate_id": 0, "pose_name": "x", "caption": "x"}], by_id)) >= 3)
    out = os.path.join(tempfile.mkdtemp(), "t.json")
    write_templates(build_payload(sel, by_id), out)
    try:
        write_templates(build_payload(sel[:1], by_id), out)
        check("guard: từ chối ghi đè bằng bản ít template hơn", False)
    except TemplateError:
        check("guard: từ chối ghi đè bằng bản ít template hơn", True)
    check("guard: có file .bak khi ghi đè hợp lệ", (write_templates(build_payload(sel, by_id), out) or True)
          and any(f.startswith("t.json.bak_") for f in os.listdir(os.path.dirname(out))))
    # Tương thích với PoseRecommenderV2 trong 05_upgrade_v2.py (nếu cùng thư mục)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    try:
        import importlib
        v2 = importlib.import_module("05_upgrade_v2")
        rec = v2.PoseRecommenderV2(out)
        ok = all(t["_ref_norm"] is not None and len(t["_ref_norm"]) >= 14 for t in rec.templates)
        check("tương thích PoseRecommenderV2 (đủ ≥14 khớp sau chuẩn hoá)", ok)
    except ImportError:
        print("[SKIP] không tìm thấy 05_upgrade_v2.py cùng thư mục")
    try:
        import cv2
        d = tempfile.mkdtemp()
        for c in cands:
            cv2.imwrite(os.path.join(d, c["image_name"]), np.full((300, 300, 3), 90, np.uint8))
        sheet = make_sheet(cands, d, 0)
        check("contact sheet render được", sheet is not None and sheet.shape[2] == 3)
    except ImportError:
        print("[SKIP] không có cv2")
    print("\nTẤT CẢ TEST PASS" if not fails else f"\nFAIL: {fails}")
    return 1 if fails else 0


def _clean_argv(argv):
    """Colab/Jupyter chèn '-f <kernel.json>' vào sys.argv -> loại bỏ."""
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
    ap = argparse.ArgumentParser(description="Dựng pose_templates_v1.json từ MPII")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("scan")
    p.add_argument("--keywords", nargs="*")
    p.add_argument("--any-activity", action="store_true")
    p.add_argument("--max-candidates", type=int, default=240)
    p = sub.add_parser("sheet")
    p.add_argument("--page", type=int, default=0)
    p = sub.add_parser("build")
    p.add_argument("--force", action="store_true")
    a = ap.parse_args(_clean_argv(sys.argv[1:] if argv is None else argv))
    try:
        if a.selftest:
            return selftest()
        if a.cmd == "scan":
            scan(a.keywords, a.any_activity, a.max_candidates)
        elif a.cmd == "sheet":
            sheet(a.page)
        elif a.cmd == "build":
            build(None, a.force)
        else:
            ap.print_help()
    except TemplateError as e:
        print(f"[LỖI] {e}")
        return 1
    return 0


if __name__ == "__main__":
    rc = main()
    if rc and "ipykernel" not in sys.modules:
        sys.exit(rc)
