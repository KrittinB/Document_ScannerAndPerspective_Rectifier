"""
test_geometry.py
Unit tests สำหรับโมดูลคำนวณเรขาคณิตและการแปลงภาพ (src/geometry.py)
ครอบคลุม F-11:
  - order_corners ถูกต้องและ invariant ต่อการสลับลำดับ input
  - get_a4_dimensions ได้สัดส่วน A4 (1 : 1.414 +- 0.01) ทั้งแนวตั้งและแนวนอน
  - quad_is_landscape ตรวจสอบทิศทางเอกสารถูกต้อง
  - validate_quad ตรวจจับจุดซ้ำ, จุด collinear, พื้นที่เล็กเกิน, และ quad เว้า
  - rectify_from_corners กู้คืนภาพเรขาคณิตที่รู้คำตอบ (Ground Truth Recovery)
  - rectify_from_reference ดัดมุมมองด้วย SIFT + RANSAC และปฏิเสธคู่ภาพที่ไม่เกี่ยวข้องกัน
"""

import itertools
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geometry import (
    A4_RATIO,
    get_a4_dimensions,
    quad_is_landscape,
    corners_to_homography,
    compute_homography,
    warp_perspective,
    rectify_from_corners,
    rectify_from_reference,
    _is_usable,
)
from src.detection import order_corners, validate_quad
from src.features import extract_and_match


# ---------------------------------------------------------------------------
# 1. ขนาดและสัดส่วน A4 (get_a4_dimensions)
# ---------------------------------------------------------------------------

def test_get_a4_dimensions_portrait():
    """ตรวจสอบสัดส่วน A4 แนวตั้ง: height / width ต้องใกล้เคียง sqrt(2) +- 0.01"""
    for base in [800, 1000, 1200, 1600]:
        w, h = get_a4_dimensions(base=base, landscape=False)
        assert h == base
        assert w < h
        ratio = h / w
        assert abs(ratio - A4_RATIO) < 0.01, f"Portrait ratio {ratio} != {A4_RATIO}"


def test_get_a4_dimensions_landscape():
    """ตรวจสอบสัดส่วน A4 แนวนอน: width / height ต้องใกล้เคียง sqrt(2) +- 0.01"""
    for base in [800, 1000, 1200, 1600]:
        w, h = get_a4_dimensions(base=base, landscape=True)
        assert w == base
        assert w > h
        ratio = w / h
        assert abs(ratio - A4_RATIO) < 0.01, f"Landscape ratio {ratio} != {A4_RATIO}"


# ---------------------------------------------------------------------------
# 2. ตรวจสอบทิศทางเอกสาร (quad_is_landscape)
# ---------------------------------------------------------------------------

def test_quad_is_landscape():
    # กรอบแนวนอน: กว้าง 800 สูง 500
    pts_landscape = np.array([[50, 50], [850, 50], [850, 550], [50, 550]], dtype=np.float32)
    assert quad_is_landscape(pts_landscape) is True

    # กรอบแนวตั้ง: กว้าง 500 สูง 800
    pts_portrait = np.array([[50, 50], [550, 50], [550, 850], [50, 850]], dtype=np.float32)
    assert quad_is_landscape(pts_portrait) is False


# ---------------------------------------------------------------------------
# 3. จัดเรียงมุม 4 จุด (order_corners)
# ---------------------------------------------------------------------------

def test_order_corners_canonical_order():
    """ตรวจสอบว่าคืนค่าเป็น [TL, TR, BR, BL] ตามเข็มนาฬิกา"""
    # จุดสี่เหลี่ยมผืนผ้าปกติ
    pts = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32)
    ordered = order_corners(pts)

    assert np.allclose(ordered[0], [100, 100])  # TL
    assert np.allclose(ordered[1], [500, 100])  # TR
    assert np.allclose(ordered[2], [500, 400])  # BR
    assert np.allclose(ordered[3], [100, 400])  # BL


def test_order_corners_permutation_invariance():
    """สลับลำดับอินพุตทั้ง 24 แบบ (4!) ต้องได้ผลลัพธ์จัดเรียงเดียวกันเสมอ"""
    canonical = np.array([[120, 80], [600, 90], [580, 450], [100, 420]], dtype=np.float32)
    expected = order_corners(canonical)

    for p in itertools.permutations(canonical):
        perm_pts = np.array(p, dtype=np.float32)
        res = order_corners(perm_pts)
        assert np.allclose(res, expected, atol=1e-3), "order_corners ไม่ invariant ต่อการสลับลำดับอินพุต"


def test_order_corners_rotated_45_deg():
    """ตรวจสอบว่าสี่เหลี่ยมเอียง 45 องศาไม่ทำให้เกิดจุดซ้ำ (แก้บั๊ก sum/diff เดิม)"""
    # เพชร/สี่เหลี่ยมหมุน 45 องศา รอบจุด (300, 300) รัศมี 150
    pts = np.array([
        [300, 150],  # บน
        [450, 300],  # ขวา
        [300, 450],  # ล่าง
        [150, 300],  # ซ้าย
    ], dtype=np.float32)

    ordered = order_corners(pts)
    assert len(ordered) == 4
    # ตรวจสอบว่าทั้ง 4 จุดไม่ซ้ำกัน
    for i in range(4):
        for j in range(i + 1, 4):
            dist = np.linalg.norm(ordered[i] - ordered[j])
            assert dist > 10.0, f"พบจุดซ้ำหรือใกล้กันเกินไป: จุด {i} กับ {j}"


# ---------------------------------------------------------------------------
# 4. ตรวจสอบความถูกต้องของสี่เหลี่ยม (validate_quad)
# ---------------------------------------------------------------------------

def test_validate_quad_valid():
    img_shape = (800, 1000)
    pts = np.array([[100, 100], [900, 100], [900, 700], [100, 700]], dtype=np.float32)
    ok, msg = validate_quad(pts, img_shape)
    assert ok is True
    assert msg == ""


def test_validate_quad_duplicate_points():
    img_shape = (800, 1000)
    # จุด 0 กับจุด 1 ซ้ำกัน
    pts = np.array([[100, 100], [100, 101], [900, 700], [100, 700]], dtype=np.float32)
    ok, msg = validate_quad(pts, img_shape)
    assert ok is False
    assert "ซ้ำหรือใกล้กันเกินไป" in msg


def test_validate_quad_too_small():
    img_shape = (1000, 1000)
    # สี่เหลี่ยมขนาดเล็ก (50x50 px = 2,500 px^2 < 3% ของ 1,000,000) แต่ระยะห่างแต่ละจุด 50 px > 2% diag (28.3 px)
    pts = np.array([[100, 100], [150, 100], [150, 150], [100, 150]], dtype=np.float32)
    ok, msg = validate_quad(pts, img_shape)
    assert ok is False
    assert "เล็กเกินกว่า" in msg


def test_validate_quad_non_convex():
    img_shape = (800, 1000)
    # จุดรูปโบว์หรือมุมเว้า (Non-convex bowtie)
    pts = np.array([[100, 100], [900, 700], [900, 100], [100, 700]], dtype=np.float32)
    ok, msg = validate_quad(pts, img_shape)
    assert ok is False
    assert "ไขว้กันเอง" in msg


# ---------------------------------------------------------------------------
# 5. กู้คืน Homography ที่รู้คำตอบ (Ground Truth Homography Recovery)
# ---------------------------------------------------------------------------

def test_rectify_from_corners_synthetic_ground_truth():
    """สร้างภาพเอกสารจำลอง บิดด้วย homography ที่กำหนด แล้วดัดกลับ"""
    # 1. สร้างภาพเอกสารตรงขนาด 800x566
    doc_w, doc_h = 800, 566
    doc = np.ones((doc_h, doc_w, 3), dtype=np.uint8) * 255
    # วาดกรอบและแพทเทิร์น
    cv2.rectangle(doc, (40, 40), (doc_w - 40, doc_h - 40), (0, 0, 0), 4)
    cv2.circle(doc, (doc_w // 2, doc_h // 2), 100, (50, 100, 200), -1)

    # 2. บิดมุมมองสังเคราะห์ลงในภาพฉากหลังขนาด 1000x1200
    scene_h, scene_w = 1000, 1200
    scene = np.ones((scene_h, scene_w, 3), dtype=np.uint8) * 40

    src_corners = np.array([[0, 0], [doc_w, 0], [doc_w, doc_h], [0, doc_h]], dtype=np.float32)
    # พิกัดเอียงใน scene
    skewed_corners = np.array([[180, 120], [1050, 220], [920, 880], [120, 760]], dtype=np.float32)

    H_warp = cv2.getPerspectiveTransform(src_corners, skewed_corners)
    cv2.warpPerspective(doc, H_warp, (scene_w, scene_h), dst=scene, borderMode=cv2.BORDER_TRANSPARENT)

    # 3. รัน rectify_from_corners
    res = rectify_from_corners(scene, skewed_corners, scale=1.0, a4_base=doc_w)

    assert res["success"] is True
    assert res["warped"] is not None
    assert res["landscape"] is True
    out_h, out_w = res["warped"].shape[:2]
    assert (out_w, out_h) == (doc_w, doc_h)

    # 4. ตรวจสอบความถูกต้องทางภาพ (Normalized Cross-Correlation)
    gray_doc = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)
    gray_rect = cv2.cvtColor(res["warped"], cv2.COLOR_BGR2GRAY)
    res_match = cv2.matchTemplate(gray_rect, gray_doc, cv2.TM_CCORR_NORMED)
    correlation = float(res_match[0, 0])
    assert correlation > 0.96, f"Correlation {correlation} ต่ำกว่า 0.96 (กู้คืนภาพไม่ตรง)"


def test_rectify_from_reference_mode():
    """ทดสอบโหมด Reference: ภาพสองใบที่เป็นเอกสารเดียวกันบิดมุมมอง"""
    # สร้างภาพเอกสารที่มีรายละเอียด (texture) เพียงพอสำหรับ SIFT
    h, w = 600, 800
    rng = np.random.RandomState(42)
    base_tex = rng.randint(50, 220, (h, w), dtype=np.uint8)
    base_bgr = cv2.cvtColor(base_tex, cv2.COLOR_GRAY2BGR)

    # เพิ่มข้อความและรูปทรงเรขาคณิตหลาย ๆ จุดเพื่อสร้าง keypoints
    for i in range(10):
        cv2.putText(base_bgr, f"Section {i} CP461 Computer Vision", (50, 50 + i * 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.rectangle(base_bgr, (500, 30 + i * 50), (700, 60 + i * 50), (255, 255, 255), -1)

    # ภาพ 1: Flat reference
    ref_img = base_bgr.copy()

    # ภาพ 2: Skewed photo (บิดมุมมองเล็กน้อย)
    src_quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_quad = np.array([[40, 30], [w - 50, 60], [w - 80, h - 40], [60, h - 30]], dtype=np.float32)
    H_sim = cv2.getPerspectiveTransform(src_quad, dst_quad)
    skewed_photo = cv2.warpPerspective(base_bgr, H_sim, (w, h))

    # จับคู่ฟีเจอร์ SIFT
    photo_gray = cv2.cvtColor(skewed_photo, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    feat = extract_and_match(photo_gray, ref_gray, method="SIFT", matcher="FLANN", ratio=0.75)

    assert feat["n_good"] >= 15, f"จำนวน good matches {feat['n_good']} ไม่เพียงพอสำหรับทดสอบ"

    # ดัดด้วย rectify_from_reference
    res = rectify_from_reference(skewed_photo, photo_scale=1.0, reference_resized=ref_img, feature_result=feat)
    assert res["success"] is True
    assert res["used_ransac"] is True
    assert res["n_inliers"] >= 10
    assert res["warped"] is not None


def test_rectify_from_reference_reject_unrelated():
    """ทดสอบว่าปฏิเสธคู่ภาพที่ไม่เกี่ยวข้องกัน (Unrelated documents) อย่างถูกต้อง"""
    h, w = 400, 400
    img1 = np.ones((h, w, 3), dtype=np.uint8) * 128
    img2 = np.zeros((h, w, 3), dtype=np.uint8)

    feat = {
        "kp1": [],
        "kp2": [],
        "good_matches": [],
        "n_good": 0,
    }

    res = rectify_from_reference(img1, photo_scale=1.0, reference_resized=img2, feature_result=feat)
    assert res["success"] is False
    assert "ต้องการอย่างน้อย 4" in res["message"]


def test_is_usable_guard():
    assert _is_usable(None) is False
    assert _is_usable(np.zeros((3, 3))) is False
    assert _is_usable(np.eye(3)) is True
    assert _is_usable(np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]])) is False
    assert _is_usable(np.array([[np.inf, 0, 0], [0, 1, 0], [0, 0, 1]])) is False


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
