"""
test_preprocessing.py
Unit tests สำหรับโมดูลเตรียมภาพ (src/preprocessing.py)
ทดสอบ: resize, grayscale, Gaussian blur, output dict structure
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess, MAX_DIM


# -----------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------

def _make_bgr(h: int, w: int, color=(128, 128, 128)) -> np.ndarray:
    """สร้างภาพ BGR จำลองขนาดที่กำหนด"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


# -----------------------------------------------------------------------
# Tests: Output dict structure
# -----------------------------------------------------------------------

def test_preprocess_returns_correct_keys():
    """ตรวจสอบว่า preprocess คืน dict ที่มี key ครบ 4 ตัว"""
    img = _make_bgr(400, 300)
    result = preprocess(img)

    expected_keys = {"resized", "gray", "blurred", "scale"}
    assert set(result.keys()) == expected_keys


def test_preprocess_output_types():
    """ตรวจสอบชนิดข้อมูลของผลลัพธ์แต่ละ key"""
    img = _make_bgr(400, 300)
    result = preprocess(img)

    assert isinstance(result["resized"], np.ndarray)
    assert isinstance(result["gray"], np.ndarray)
    assert isinstance(result["blurred"], np.ndarray)
    assert isinstance(result["scale"], float)


# -----------------------------------------------------------------------
# Tests: Resize behavior
# -----------------------------------------------------------------------

def test_resize_large_image():
    """ภาพที่ใหญ่กว่า MAX_DIM ต้องถูก resize ลงโดยรักษาสัดส่วน"""
    h, w = 2400, 3200  # ใหญ่กว่า MAX_DIM
    img = _make_bgr(h, w)
    result = preprocess(img)

    rh, rw = result["resized"].shape[:2]
    # ด้านที่ยาวที่สุดต้อง <= MAX_DIM
    assert max(rh, rw) <= MAX_DIM
    # scale ต้อง < 1.0
    assert result["scale"] < 1.0
    # ตรวจสอบ aspect ratio ยังใกล้เคียงเดิม
    original_ratio = w / h
    resized_ratio = rw / rh
    assert abs(original_ratio - resized_ratio) < 0.02


def test_resize_preserves_aspect_ratio_portrait():
    """ภาพ portrait ที่ใหญ่เกิน MAX_DIM ต้องรักษา aspect ratio"""
    h, w = 3000, 1500  # portrait
    img = _make_bgr(h, w)
    result = preprocess(img)

    rh, rw = result["resized"].shape[:2]
    assert rh > rw  # ยังคงเป็น portrait
    assert max(rh, rw) <= MAX_DIM
    assert abs((w / h) - (rw / rh)) < 0.02


def test_no_resize_when_small():
    """ภาพที่เล็กกว่า MAX_DIM ต้องไม่ถูก resize — scale = 1.0"""
    h, w = 600, 800  # เล็กกว่า MAX_DIM
    img = _make_bgr(h, w)
    result = preprocess(img)

    rh, rw = result["resized"].shape[:2]
    assert rh == h
    assert rw == w
    assert result["scale"] == 1.0


def test_no_resize_at_exact_max_dim():
    """ภาพที่มีขนาดพอดี MAX_DIM ต้องไม่ถูก resize"""
    img = _make_bgr(MAX_DIM, 800)
    result = preprocess(img)

    rh, rw = result["resized"].shape[:2]
    assert rh == MAX_DIM
    assert rw == 800
    assert result["scale"] == 1.0


def test_scale_factor_correctness():
    """ค่า scale ต้องตรงกับ MAX_DIM / max(h, w) เมื่อภาพใหญ่เกิน"""
    h, w = 2000, 3000
    img = _make_bgr(h, w)
    result = preprocess(img)

    expected_scale = MAX_DIM / max(h, w)
    assert abs(result["scale"] - expected_scale) < 1e-6


# -----------------------------------------------------------------------
# Tests: Grayscale conversion
# -----------------------------------------------------------------------

def test_grayscale_output_is_single_channel():
    """ภาพ gray ต้องเป็น single-channel (2D array)"""
    img = _make_bgr(400, 300)
    result = preprocess(img)

    assert result["gray"].ndim == 2
    assert result["gray"].dtype == np.uint8


def test_grayscale_shape_matches_resized():
    """ขนาดของ gray ต้องตรงกับ resized"""
    img = _make_bgr(2000, 3000)
    result = preprocess(img)

    rh, rw = result["resized"].shape[:2]
    gh, gw = result["gray"].shape[:2]
    assert gh == rh
    assert gw == rw


def test_grayscale_values_reasonable():
    """ภาพสีที่แปลงเป็น grayscale ต้องมีค่าที่สมเหตุสมผล"""
    # ภาพสีแดงบริสุทธิ์
    img = _make_bgr(100, 100, color=(0, 0, 255))  # BGR: แดง
    result = preprocess(img)

    gray = result["gray"]
    # ค่า grayscale ของสีแดงบริสุทธิ์ ≈ 0.299 * 255 ≈ 76
    mean_val = gray.mean()
    assert 50 < mean_val < 100


# -----------------------------------------------------------------------
# Tests: Gaussian blur
# -----------------------------------------------------------------------

def test_blurred_output_is_single_channel():
    """ภาพ blurred ต้องเป็น single-channel (2D array)"""
    img = _make_bgr(400, 300)
    result = preprocess(img)

    assert result["blurred"].ndim == 2
    assert result["blurred"].dtype == np.uint8


def test_blurred_shape_matches_gray():
    """ขนาดของ blurred ต้องตรงกับ gray"""
    img = _make_bgr(2000, 3000)
    result = preprocess(img)

    assert result["blurred"].shape == result["gray"].shape


def test_blur_reduces_noise():
    """Gaussian blur ต้องลดความแปรปรวน (variance) ของภาพที่มี noise"""
    img = _make_bgr(400, 400)
    # เพิ่ม noise
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    noisy_img = cv2.add(img, noise)

    result = preprocess(noisy_img)
    gray = result["gray"]
    blurred = result["blurred"]

    # variance ของ blurred ต้องน้อยกว่า gray (noise ถูกลด)
    assert np.var(blurred.astype(float)) <= np.var(gray.astype(float))


def test_blur_does_not_change_mean_significantly():
    """Gaussian blur ไม่ควรเปลี่ยนค่าเฉลี่ยของภาพอย่างมีนัยสำคัญ"""
    img = _make_bgr(400, 300, color=(100, 150, 200))
    result = preprocess(img)

    gray_mean = result["gray"].mean()
    blurred_mean = result["blurred"].mean()
    # ค่าเฉลี่ยต้องต่างกันน้อยมาก
    assert abs(gray_mean - blurred_mean) < 5.0


# -----------------------------------------------------------------------
# Tests: Real sample image
# -----------------------------------------------------------------------

def test_preprocess_real_sample():
    """ทดสอบ preprocess กับภาพจริง test1.webp"""
    sample_path = os.path.join(os.path.dirname(__file__), "sample_images", "test1.webp")
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        assert img is not None

        result = preprocess(img)

        # ตรวจ output keys
        assert set(result.keys()) == {"resized", "gray", "blurred", "scale"}

        # ด้านยาวสุดของ resized ต้อง <= MAX_DIM
        rh, rw = result["resized"].shape[:2]
        assert max(rh, rw) <= MAX_DIM

        # gray และ blurred ต้อง single-channel
        assert result["gray"].ndim == 2
        assert result["blurred"].ndim == 2

        # ไม่มี NaN
        assert not np.isnan(result["gray"].astype(float)).any()
        assert not np.isnan(result["blurred"].astype(float)).any()


# -----------------------------------------------------------------------
# Tests: Input does not get mutated
# -----------------------------------------------------------------------

def test_preprocess_does_not_mutate_input():
    """preprocess ต้องไม่แก้ไขภาพ input ต้นฉบับ"""
    img = _make_bgr(400, 300, color=(50, 100, 150))
    original_copy = img.copy()

    preprocess(img)
    assert np.array_equal(img, original_copy)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
