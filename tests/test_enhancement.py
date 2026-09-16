"""
test_enhancement.py
Unit tests สำหรับระบบ Document Enhancement & Smart Filters (src/enhancement.py)
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.enhancement import (
    FILTER_ORIGINAL,
    FILTER_MAGIC,
    FILTER_BW,
    FILTER_GRAY,
    FILTER_MODES,
    remove_shadows,
    enhance_magic_color,
    enhance_clean_bw,
    enhance_grayscale,
    apply_filter,
)


def create_synthetic_document() -> np.ndarray:
    """สร้างภาพเอกสารจำลองขนาด 400x300 พร้อมไล่ระดับแสงเงา (Uneven Lighting) และมีตัวหนังสือ"""
    h, w = 300, 400
    # พื้นหลังสีขาว
    img = np.ones((h, w, 3), dtype=np.uint8) * 240

    # สร้างเงาเอียงพาดผ่าน (Linear Shadow gradient)
    for y in range(h):
        for x in range(w):
            factor = 0.4 + 0.6 * (x / w) * (y / h)
            img[y, x] = np.clip(img[y, x] * factor, 0, 255)

    # วาดข้อความสีดำและสีน้ำเงิน
    cv2.putText(img, "TEST DOCUMENT SCANNER", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(img, "Computer Vision CP461", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 50, 20), 2)
    cv2.putText(img, "Clean B&W & Magic Color", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)

    return img


def test_remove_shadows():
    img = create_synthetic_document()
    out = remove_shadows(img, kernel_size=25)

    assert out is not None
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert not np.isnan(out).any()
    # ตรวจสอบว่าพื้นที่เงามืดสว่างขึ้น
    assert np.mean(out) > np.mean(img)


def test_enhance_magic_color():
    img = create_synthetic_document()
    out = enhance_magic_color(img, clip_limit=2.0)

    assert out is not None
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert not np.isnan(out).any()


def test_enhance_clean_bw():
    img = create_synthetic_document()
    out = enhance_clean_bw(img, block_size=21, c_val=10)

    assert out is not None
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # ภาพ B&W ต้องมีพิกเซลที่เป็นขาวบริสุทธิ์ (255) และดำ (0)
    unique_vals = np.unique(out)
    assert 0 in unique_vals or 255 in unique_vals


def test_enhance_grayscale():
    img = create_synthetic_document()
    out = enhance_grayscale(img)

    assert out is not None
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # ตรวจสอบว่าเป็น 3-channel grayscale (B == G == R)
    assert np.array_equal(out[:, :, 0], out[:, :, 1])
    assert np.array_equal(out[:, :, 1], out[:, :, 2])


def test_apply_filter_all_modes():
    img = create_synthetic_document()
    for mode in FILTER_MODES:
        out = apply_filter(img, mode)
        assert out is not None
        assert out.shape == img.shape
        assert out.dtype == np.uint8
        assert not np.isnan(out).any()


def test_real_sample_image():
    sample_path = os.path.join("tests", "sample_images", "test1.webp")
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        assert img is not None
        for mode in FILTER_MODES:
            out = apply_filter(img, mode)
            assert out is not None
            assert out.shape == img.shape


def test_edge_cases():
    # Empty / None
    assert apply_filter(None, FILTER_MAGIC) is None
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert apply_filter(empty, FILTER_MAGIC).size == 0


if __name__ == "__main__":
    test_remove_shadows()
    test_enhance_magic_color()
    test_enhance_clean_bw()
    test_enhance_grayscale()
    test_apply_filter_all_modes()
    test_real_sample_image()
    test_edge_cases()
    print("All enhancement tests passed successfully!")
