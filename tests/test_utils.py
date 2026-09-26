"""
test_utils.py
Unit tests สำหรับ Utility functions (src/utils.py)
ทดสอบ: format converters (PIL↔NumPy, bytes↔image),
        visualization helpers (draw_corners, draw_keypoints, draw_inlier_outlier),
        numpy_to_bytes_png
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import (
    pil_to_numpy_bgr,
    numpy_bgr_to_pil,
    numpy_gray_to_pil,
    bytes_to_numpy_bgr,
    draw_corners,
    draw_matches_visualization,
    draw_keypoints,
    draw_inlier_outlier,
    numpy_to_bytes_png,
)


# -----------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------

def _make_bgr(h: int = 300, w: int = 400) -> np.ndarray:
    """สร้างภาพ BGR จำลอง"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (350, 250), (100, 150, 200), -1)
    return img


def _make_pil_rgb(h: int = 300, w: int = 400) -> Image.Image:
    """สร้าง PIL Image RGB จำลอง"""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[50:250, 50:350] = [200, 150, 100]  # RGB
    return Image.fromarray(arr, mode="RGB")


# -----------------------------------------------------------------------
# Tests: pil_to_numpy_bgr
# -----------------------------------------------------------------------

def test_pil_to_numpy_bgr_shape():
    """แปลง PIL→NumPy BGR ต้องได้ shape เดิม (H, W, 3)"""
    pil = _make_pil_rgb(300, 400)
    bgr = pil_to_numpy_bgr(pil)

    assert bgr.shape == (300, 400, 3)
    assert bgr.dtype == np.uint8


def test_pil_to_numpy_bgr_channel_order():
    """ตรวจสอบว่า channel order ถูก swap จาก RGB → BGR"""
    # สร้างภาพ PIL สีแดง (R=255, G=0, B=0)
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[:, :] = [255, 0, 0]  # RGB: แดง
    pil = Image.fromarray(arr, mode="RGB")

    bgr = pil_to_numpy_bgr(pil)
    # BGR: ช่อง 0=Blue ต้องเป็น 0, ช่อง 2=Red ต้องเป็น 255
    assert bgr[5, 5, 0] == 0    # Blue
    assert bgr[5, 5, 1] == 0    # Green
    assert bgr[5, 5, 2] == 255  # Red


def test_pil_to_numpy_bgr_rgba_input():
    """รองรับ RGBA input ต้องตัดช่อง alpha ออก"""
    arr = np.zeros((10, 10, 4), dtype=np.uint8)
    arr[:, :] = [255, 128, 64, 200]
    pil = Image.fromarray(arr, mode="RGBA")

    bgr = pil_to_numpy_bgr(pil)
    # .convert("RGB") ต้องตัด alpha ออก → 3 channels
    assert bgr.shape == (10, 10, 3)


# -----------------------------------------------------------------------
# Tests: numpy_bgr_to_pil
# -----------------------------------------------------------------------

def test_numpy_bgr_to_pil_type():
    """แปลง NumPy BGR→PIL ต้องได้ PIL Image"""
    bgr = _make_bgr()
    pil = numpy_bgr_to_pil(bgr)

    assert isinstance(pil, Image.Image)
    assert pil.mode == "RGB"
    assert pil.size == (400, 300)  # PIL size = (width, height)


def test_numpy_bgr_to_pil_channel_order():
    """ตรวจสอบว่า BGR→RGB channel order ถูกต้อง"""
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr[:, :] = [0, 0, 255]  # BGR: แดง

    pil = numpy_bgr_to_pil(bgr)
    r, g, b = pil.getpixel((5, 5))
    assert r == 255  # Red
    assert g == 0
    assert b == 0


def test_roundtrip_bgr_pil_bgr():
    """แปลง BGR→PIL→BGR ต้องได้ภาพเดิม"""
    original = _make_bgr()
    pil = numpy_bgr_to_pil(original)
    restored = pil_to_numpy_bgr(pil)

    assert np.array_equal(original, restored)


# -----------------------------------------------------------------------
# Tests: numpy_gray_to_pil
# -----------------------------------------------------------------------

def test_numpy_gray_to_pil():
    """แปลง grayscale NumPy→PIL ต้องได้ mode 'L' หรือ grayscale ที่ถูกต้อง"""
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    pil = numpy_gray_to_pil(gray)

    assert isinstance(pil, Image.Image)
    assert pil.size == (100, 100)


def test_numpy_gray_to_pil_values():
    """ค่าพิกเซลต้องตรงกัน"""
    gray = np.ones((10, 10), dtype=np.uint8) * 42
    pil = numpy_gray_to_pil(gray)

    assert pil.getpixel((5, 5)) == 42


# -----------------------------------------------------------------------
# Tests: bytes_to_numpy_bgr
# -----------------------------------------------------------------------

def test_bytes_to_numpy_bgr():
    """แปลง PNG bytes → NumPy BGR ต้องได้ภาพที่ถูกต้อง"""
    # สร้าง PNG bytes จากภาพจำลอง
    original = _make_bgr(200, 300)
    _, buf = cv2.imencode(".png", original)
    png_bytes = buf.tobytes()

    decoded = bytes_to_numpy_bgr(png_bytes)
    assert decoded is not None
    assert decoded.shape == (200, 300, 3)
    assert decoded.dtype == np.uint8


def test_bytes_to_numpy_bgr_jpeg():
    """แปลง JPEG bytes → NumPy BGR ต้องทำงานได้"""
    original = _make_bgr(200, 300)
    _, buf = cv2.imencode(".jpg", original)
    jpg_bytes = buf.tobytes()

    decoded = bytes_to_numpy_bgr(jpg_bytes)
    assert decoded is not None
    assert decoded.shape == (200, 300, 3)


# -----------------------------------------------------------------------
# Tests: draw_corners
# -----------------------------------------------------------------------

def test_draw_corners_returns_annotated_image():
    """draw_corners ต้องคืนภาพที่วาดจุดมุมแล้ว ขนาดเท่าเดิม"""
    img = _make_bgr(400, 600)
    corners = np.array([[100, 80], [500, 80], [500, 350], [100, 350]], dtype=np.float32)

    result = draw_corners(img, corners)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_draw_corners_does_not_mutate_input():
    """draw_corners ต้องไม่แก้ไขภาพ input"""
    img = _make_bgr(400, 600)
    original_copy = img.copy()
    corners = np.array([[100, 80], [500, 80], [500, 350], [100, 350]], dtype=np.float32)

    draw_corners(img, corners)
    assert np.array_equal(img, original_copy)


def test_draw_corners_produces_different_image():
    """ภาพที่วาดจุดมุมแล้วต้องต่างจากภาพเดิม"""
    img = _make_bgr(400, 600)
    corners = np.array([[100, 80], [500, 80], [500, 350], [100, 350]], dtype=np.float32)

    result = draw_corners(img, corners)
    assert not np.array_equal(img, result)


# -----------------------------------------------------------------------
# Tests: draw_keypoints
# -----------------------------------------------------------------------

def test_draw_keypoints_returns_image():
    """draw_keypoints ต้องคืนภาพที่วาด keypoints แล้ว"""
    img = _make_bgr()
    keypoints = [
        cv2.KeyPoint(100, 100, 10),
        cv2.KeyPoint(200, 150, 15),
        cv2.KeyPoint(300, 200, 20),
    ]

    result = draw_keypoints(img, keypoints)
    assert result is not None
    assert result.shape[:2] == img.shape[:2]
    assert result.dtype == np.uint8


def test_draw_keypoints_empty_list():
    """draw_keypoints กับ keypoints ว่างต้องไม่ crash"""
    img = _make_bgr()
    result = draw_keypoints(img, [])

    assert result is not None
    assert result.shape[:2] == img.shape[:2]


def test_draw_keypoints_max_draw():
    """max_draw ต้องจำกัดจำนวน keypoints ที่วาด"""
    img = _make_bgr()
    keypoints = [cv2.KeyPoint(float(i * 10), float(i * 10), 5) for i in range(100)]

    # ไม่ crash แม้ keypoints > max_draw
    result = draw_keypoints(img, keypoints, max_draw=10)
    assert result is not None


# -----------------------------------------------------------------------
# Tests: draw_matches_visualization
# -----------------------------------------------------------------------

def test_draw_matches_visualization():
    """draw_matches_visualization ต้องคืนภาพ concat ที่ถูกต้อง"""
    img1 = _make_bgr(200, 300)
    img2 = _make_bgr(200, 300)

    kp1 = [cv2.KeyPoint(50, 50, 10), cv2.KeyPoint(100, 100, 10)]
    kp2 = [cv2.KeyPoint(60, 60, 10), cv2.KeyPoint(110, 110, 10)]

    # สร้าง DMatch จำลอง
    matches = [cv2.DMatch(0, 0, 10.0), cv2.DMatch(1, 1, 20.0)]

    result = draw_matches_visualization(img1, kp1, img2, kp2, matches)
    assert result is not None
    assert result.ndim == 3
    # ภาพ output ต้องกว้างกว่า input เดี่ยว (concat 2 ภาพ)
    assert result.shape[1] >= img1.shape[1]


def test_draw_matches_visualization_empty():
    """draw_matches_visualization กับ matches ว่างต้องไม่ crash"""
    img1 = _make_bgr(200, 300)
    img2 = _make_bgr(200, 300)

    result = draw_matches_visualization(img1, [], img2, [], [])
    assert result is not None


# -----------------------------------------------------------------------
# Tests: draw_inlier_outlier
# -----------------------------------------------------------------------

def test_draw_inlier_outlier_with_valid_mask():
    """draw_inlier_outlier ต้องวาดจุดสีเขียว (inlier) และแดง (outlier)"""
    img = _make_bgr()
    kp = [cv2.KeyPoint(50, 50, 10), cv2.KeyPoint(100, 100, 10),
          cv2.KeyPoint(200, 200, 10)]
    matches = [cv2.DMatch(0, 0, 5.0), cv2.DMatch(1, 1, 10.0),
               cv2.DMatch(2, 2, 50.0)]
    mask = np.array([1, 1, 0])  # 2 inliers, 1 outlier

    result = draw_inlier_outlier(img, kp, matches, mask)
    assert result is not None
    assert result.shape == img.shape


def test_draw_inlier_outlier_returns_none_when_no_mask():
    """ถ้า mask เป็น None ต้องคืน None"""
    img = _make_bgr()
    kp = [cv2.KeyPoint(50, 50, 10)]
    matches = [cv2.DMatch(0, 0, 5.0)]

    result = draw_inlier_outlier(img, kp, matches, None)
    assert result is None


def test_draw_inlier_outlier_returns_none_when_mask_mismatch():
    """ถ้า mask มีขนาดไม่ตรงกับ good_matches ต้องคืน None"""
    img = _make_bgr()
    kp = [cv2.KeyPoint(50, 50, 10)]
    matches = [cv2.DMatch(0, 0, 5.0)]
    mask = np.array([1, 0, 1])  # ขนาดไม่ตรง

    result = draw_inlier_outlier(img, kp, matches, mask)
    assert result is None


def test_draw_inlier_outlier_does_not_mutate_input():
    """draw_inlier_outlier ต้องไม่แก้ไขภาพ input"""
    img = _make_bgr()
    original_copy = img.copy()
    kp = [cv2.KeyPoint(50, 50, 10)]
    matches = [cv2.DMatch(0, 0, 5.0)]
    mask = np.array([1])

    draw_inlier_outlier(img, kp, matches, mask)
    assert np.array_equal(img, original_copy)


# -----------------------------------------------------------------------
# Tests: numpy_to_bytes_png
# -----------------------------------------------------------------------

def test_numpy_to_bytes_png():
    """แปลง NumPy BGR → PNG bytes ต้องได้ bytes ที่ valid"""
    img = _make_bgr()
    result = numpy_to_bytes_png(img)

    assert isinstance(result, bytes)
    assert len(result) > 0
    # PNG magic bytes: \x89PNG
    assert result[:4] == b"\x89PNG"


def test_numpy_to_bytes_png_roundtrip():
    """แปลง BGR→PNG→BGR ต้องได้ภาพที่เหมือนเดิม (lossless)"""
    original = _make_bgr(100, 150)
    png_bytes = numpy_to_bytes_png(original)

    # decode กลับมา
    arr = np.frombuffer(png_bytes, np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    assert decoded is not None
    assert decoded.shape == original.shape
    # PNG เป็น lossless — ค่าพิกเซลต้องเหมือนเดิมทุกจุด
    assert np.array_equal(original, decoded)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
