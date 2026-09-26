"""
test_features.py
Unit tests สำหรับระบบ Feature Extraction & Matching (src/features.py)
ทดสอบ: SIFT, ORB, BFMatcher, FLANN, Lowe's Ratio Test, pipeline wrapper
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import (
    extract_features,
    match_features,
    apply_ratio_test,
    extract_and_match,
)


# -----------------------------------------------------------------------
# Helper: สร้างภาพทดสอบที่มี texture เพียงพอสำหรับ feature detection
# -----------------------------------------------------------------------

def _make_textured_image(h: int = 400, w: int = 500, seed: int = 42) -> np.ndarray:
    """สร้างภาพ grayscale ที่มี pattern หลากหลายเพื่อให้ detector หาจุดได้"""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w), dtype=np.uint8)

    # วาดสี่เหลี่ยมและวงกลมหลายขนาดเพื่อสร้าง texture
    for _ in range(30):
        x1 = rng.randint(0, w - 40)
        y1 = rng.randint(0, h - 40)
        x2 = x1 + rng.randint(20, 80)
        y2 = y1 + rng.randint(20, 80)
        cv2.rectangle(img, (x1, y1), (min(x2, w - 1), min(y2, h - 1)),
                      int(rng.randint(80, 255)), rng.choice([-1, 1, 2]))

    for _ in range(20):
        cx = rng.randint(20, w - 20)
        cy = rng.randint(20, h - 20)
        r = rng.randint(5, 30)
        cv2.circle(img, (cx, cy), r, int(rng.randint(80, 255)), -1)

    # เพิ่มข้อความเพื่อสร้างขอบที่ชัดเจน
    cv2.putText(img, "FEATURE TEST", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    cv2.putText(img, "SIFT & ORB", (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 200, 2)

    return img


def _make_blank_image(h: int = 300, w: int = 400) -> np.ndarray:
    """สร้างภาพสีทึบไม่มี texture — ไม่ควรหา keypoints ได้"""
    return np.ones((h, w), dtype=np.uint8) * 128


# -----------------------------------------------------------------------
# Tests: Feature Extraction — SIFT
# -----------------------------------------------------------------------

def test_extract_features_sift_returns_keypoints():
    """SIFT ต้องหา keypoints ได้จากภาพที่มี texture"""
    img = _make_textured_image()
    kp, desc = extract_features(img, method="SIFT")

    assert len(kp) > 0
    assert desc is not None
    assert desc.shape[0] == len(kp)
    assert desc.dtype == np.float32  # SIFT descriptors เป็น float32


def test_extract_features_sift_descriptor_shape():
    """SIFT descriptor ต้องมี 128 มิติ"""
    img = _make_textured_image()
    kp, desc = extract_features(img, method="SIFT")

    assert desc.ndim == 2
    assert desc.shape[1] == 128


def test_extract_features_sift_max_keypoints():
    """จำนวน keypoints ไม่ควรเกิน nfeatures=1000"""
    img = _make_textured_image()
    kp, desc = extract_features(img, method="SIFT")

    assert len(kp) <= 1000


# -----------------------------------------------------------------------
# Tests: Feature Extraction — ORB
# -----------------------------------------------------------------------

def test_extract_features_orb_returns_keypoints():
    """ORB ต้องหา keypoints ได้จากภาพที่มี texture"""
    img = _make_textured_image()
    kp, desc = extract_features(img, method="ORB")

    assert len(kp) > 0
    assert desc is not None
    assert desc.shape[0] == len(kp)
    assert desc.dtype == np.uint8  # ORB descriptors เป็น binary (uint8)


def test_extract_features_orb_descriptor_shape():
    """ORB descriptor ต้องมี 32 bytes (256 bit)"""
    img = _make_textured_image()
    kp, desc = extract_features(img, method="ORB")

    assert desc.ndim == 2
    assert desc.shape[1] == 32


def test_extract_features_orb_max_keypoints():
    """จำนวน keypoints ของ ORB ไม่ควรเกินขีดจำกัดที่ตั้งไว้มาก (nfeatures=1000 เป็น soft limit)"""
    img = _make_textured_image()
    kp, desc = extract_features(img, method="ORB")

    # OpenCV ORB อาจคืน keypoints เกิน nfeatures เล็กน้อยได้ ขึ้นกับ image content
    assert len(kp) <= 1100


# -----------------------------------------------------------------------
# Tests: Feature Extraction — Edge Cases
# -----------------------------------------------------------------------

def test_extract_features_blank_image():
    """ภาพสีทึบไม่มี texture ควรได้ keypoints น้อยมากหรือ 0"""
    img = _make_blank_image()
    kp, desc = extract_features(img, method="SIFT")

    # ภาพสีทึบอาจได้ 0 หรือน้อยมาก
    assert len(kp) < 10


def test_extract_features_blank_orb():
    """ORB บนภาพสีทึบควรได้ keypoints น้อยมากหรือ 0"""
    img = _make_blank_image()
    kp, desc = extract_features(img, method="ORB")

    assert len(kp) < 10


# -----------------------------------------------------------------------
# Tests: Feature Matching — BFMatcher
# -----------------------------------------------------------------------

def test_match_features_bfmatcher_sift():
    """BFMatcher + SIFT ต้องคืน raw matches ได้"""
    img1 = _make_textured_image(seed=42)
    img2 = _make_textured_image(seed=42)  # ภาพเดียวกัน
    _, desc1 = extract_features(img1, "SIFT")
    _, desc2 = extract_features(img2, "SIFT")

    raw = match_features(desc1, desc2, method="SIFT", matcher="BFMatcher")
    assert len(raw) > 0
    # knnMatch k=2 ต้องคืน list of pairs
    assert len(raw[0]) == 2


def test_match_features_bfmatcher_orb():
    """BFMatcher + ORB (Hamming distance) ต้องทำงานได้"""
    img1 = _make_textured_image(seed=42)
    img2 = _make_textured_image(seed=42)
    _, desc1 = extract_features(img1, "ORB")
    _, desc2 = extract_features(img2, "ORB")

    raw = match_features(desc1, desc2, method="ORB", matcher="BFMatcher")
    assert len(raw) > 0


# -----------------------------------------------------------------------
# Tests: Feature Matching — FLANN
# -----------------------------------------------------------------------

def test_match_features_flann_sift():
    """FLANN KD-Tree + SIFT ต้องคืน raw matches ได้"""
    img1 = _make_textured_image(seed=42)
    img2 = _make_textured_image(seed=42)
    _, desc1 = extract_features(img1, "SIFT")
    _, desc2 = extract_features(img2, "SIFT")

    raw = match_features(desc1, desc2, method="SIFT", matcher="FLANN")
    assert len(raw) > 0


def test_match_features_flann_orb():
    """FLANN LSH + ORB ต้องคืน raw matches ได้"""
    img1 = _make_textured_image(seed=42)
    img2 = _make_textured_image(seed=42)
    _, desc1 = extract_features(img1, "ORB")
    _, desc2 = extract_features(img2, "ORB")

    raw = match_features(desc1, desc2, method="ORB", matcher="FLANN")
    assert len(raw) > 0


# -----------------------------------------------------------------------
# Tests: Feature Matching — Edge Cases
# -----------------------------------------------------------------------

def test_match_features_returns_empty_when_none_descriptors():
    """ถ้า descriptor เป็น None ต้องคืน list ว่าง"""
    result = match_features(None, None, method="SIFT", matcher="BFMatcher")
    assert result == []


def test_match_features_returns_empty_when_too_few_descriptors():
    """ถ้า descriptor มีน้อยกว่า 2 ต้องคืน list ว่าง"""
    desc1 = np.random.rand(1, 128).astype(np.float32)
    desc2 = np.random.rand(1, 128).astype(np.float32)

    result = match_features(desc1, desc2, method="SIFT", matcher="BFMatcher")
    assert result == []


# -----------------------------------------------------------------------
# Tests: Lowe's Ratio Test
# -----------------------------------------------------------------------

def test_apply_ratio_test_filters_matches():
    """ratio test ต้องลดจำนวน matches ลง"""
    img1 = _make_textured_image(seed=42)
    img2 = _make_textured_image(seed=99)  # ภาพต่างกันเล็กน้อย
    _, desc1 = extract_features(img1, "SIFT")
    _, desc2 = extract_features(img2, "SIFT")

    raw = match_features(desc1, desc2, method="SIFT", matcher="BFMatcher")
    good = apply_ratio_test(raw, ratio=0.75)

    assert len(good) <= len(raw)


def test_apply_ratio_test_strict_threshold():
    """ratio ที่ต่ำมาก (เข้มงวด) ต้องได้ good matches น้อยกว่า ratio ที่สูง (ผ่อนปรน)"""
    img1 = _make_textured_image(seed=42)
    img2 = _make_textured_image(seed=99)
    _, desc1 = extract_features(img1, "SIFT")
    _, desc2 = extract_features(img2, "SIFT")

    raw = match_features(desc1, desc2, method="SIFT", matcher="BFMatcher")
    good_strict = apply_ratio_test(raw, ratio=0.5)
    good_relaxed = apply_ratio_test(raw, ratio=0.9)

    assert len(good_strict) <= len(good_relaxed)


def test_apply_ratio_test_empty_input():
    """ratio test กับ input ว่างต้องคืน list ว่าง"""
    assert apply_ratio_test([], ratio=0.75) == []


def test_apply_ratio_test_identical_images():
    """ภาพเดียวกันทั้งคู่ต้องได้ good matches จำนวนมาก"""
    img = _make_textured_image(seed=42)
    _, desc1 = extract_features(img, "SIFT")
    _, desc2 = extract_features(img, "SIFT")

    raw = match_features(desc1, desc2, method="SIFT", matcher="BFMatcher")
    good = apply_ratio_test(raw, ratio=0.75)

    # ภาพเหมือนกัน — good matches ต้องมากกว่า 50% ของ raw matches
    assert len(good) > len(raw) * 0.5


# -----------------------------------------------------------------------
# Tests: extract_and_match pipeline
# -----------------------------------------------------------------------

def test_extract_and_match_sift_bfmatcher():
    """pipeline wrapper ต้องคืน dict ที่มี key ครบ"""
    img1 = _make_textured_image(seed=42)
    img2 = _make_textured_image(seed=42)

    result = extract_and_match(img1, img2, method="SIFT", matcher="BFMatcher")

    expected_keys = {
        "kp1", "kp2", "desc1", "desc2",
        "raw_matches", "good_matches",
        "n_keypoints_1", "n_keypoints_2", "n_good",
    }
    assert set(result.keys()) == expected_keys
    assert result["n_keypoints_1"] > 0
    assert result["n_keypoints_2"] > 0
    assert result["n_good"] >= 0


def test_extract_and_match_orb_flann():
    """pipeline wrapper ทำงานได้กับ ORB + FLANN"""
    img1 = _make_textured_image(seed=42)
    img2 = _make_textured_image(seed=42)

    result = extract_and_match(img1, img2, method="ORB", matcher="FLANN")

    assert result["n_keypoints_1"] > 0
    assert result["n_keypoints_2"] > 0


def test_extract_and_match_identical_image_high_matches():
    """ภาพเดียวกัน ต้องได้ good matches จำนวนมาก"""
    img = _make_textured_image(seed=42)

    result = extract_and_match(img, img, method="SIFT", matcher="BFMatcher")

    assert result["n_good"] > 20


def test_extract_and_match_different_images_fewer_matches():
    """ภาพต่างกันมาก ต้องได้ good matches น้อยกว่าภาพเดียวกัน"""
    img1 = _make_textured_image(seed=42)
    img2 = _make_textured_image(seed=999)

    result_same = extract_and_match(img1, img1, method="SIFT", matcher="BFMatcher")
    result_diff = extract_and_match(img1, img2, method="SIFT", matcher="BFMatcher")

    assert result_diff["n_good"] <= result_same["n_good"]


def test_extract_and_match_custom_ratio():
    """ปรับ ratio parameter ต้องมีผลต่อจำนวน good matches"""
    img1 = _make_textured_image(seed=42)
    img2 = _make_textured_image(seed=99)

    result_strict = extract_and_match(img1, img2, method="SIFT",
                                      matcher="BFMatcher", ratio=0.5)
    result_relaxed = extract_and_match(img1, img2, method="SIFT",
                                       matcher="BFMatcher", ratio=0.9)

    assert result_strict["n_good"] <= result_relaxed["n_good"]


# -----------------------------------------------------------------------
# Tests: Real sample images
# -----------------------------------------------------------------------

def test_features_real_sample():
    """ทดสอบ feature extraction กับภาพจริง test1.webp"""
    sample_path = os.path.join(os.path.dirname(__file__), "sample_images", "test1.webp")
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None

        kp_sift, desc_sift = extract_features(img, "SIFT")
        assert len(kp_sift) > 0
        assert desc_sift is not None

        kp_orb, desc_orb = extract_features(img, "ORB")
        assert len(kp_orb) > 0
        assert desc_orb is not None


def test_reference_pair_matching():
    """ทดสอบ matching ระหว่างภาพ reference pair จริง"""
    dir_path = os.path.join(os.path.dirname(__file__), "sample_images")
    ref_path = os.path.join(dir_path, "ref1_flat_reference.jpg")
    skew_path = os.path.join(dir_path, "ref1_skewed_photo.jpg")

    if os.path.exists(ref_path) and os.path.exists(skew_path):
        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        skew = cv2.imread(skew_path, cv2.IMREAD_GRAYSCALE)
        assert ref is not None and skew is not None

        result = extract_and_match(skew, ref, method="SIFT", matcher="FLANN")

        # reference pair ที่เป็นภาพเดียวกันถ่ายต่างมุม ต้องได้ good matches
        assert result["n_good"] >= 4


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
