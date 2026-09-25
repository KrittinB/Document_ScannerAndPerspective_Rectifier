"""
test_detection.py
Unit tests สำหรับโมดูลการตรวจจับขอบและ 4 มุมเอกสาร (src/detection.py)
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection import (
    detect_edges,
    find_document_contour,
    detect_document,
)
from src.preprocessing import preprocess


def test_detect_edges():
    """ตรวจสอบว่า detect_edges คืนภาพ edge map ขนาดเท่ากับอินพุต"""
    img = np.zeros((300, 400), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (350, 250), 255, -1)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    edges = detect_edges(blurred)
    assert edges is not None
    assert edges.shape == (300, 400)
    assert edges.dtype == np.uint8
    assert np.any(edges > 0)


def test_find_document_contour_synthetic():
    """ตรวจจับเอกสารบนภาพจำลองที่มีขอบชัดเจน"""
    h, w = 600, 800
    canvas = np.zeros((h, w), dtype=np.uint8)
    # วาดสี่เหลี่ยมสีขาวตรงกลาง (ตัวแทนแผ่นกระดาษ)
    cv2.rectangle(canvas, (100, 80), (700, 520), 255, -1)
    blurred = cv2.GaussianBlur(canvas, (5, 5), 0)
    edges = detect_edges(blurred)

    corners, method = find_document_contour(edges, (h, w))
    assert corners is not None
    assert corners.shape == (4, 2)
    assert method in ("contour", "minarearect")


def test_detect_document_synthetic_document():
    """ตรวจจับเอกสารและตรวจสอบผลลัพธ์ผ่าน detect_document"""
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 40
    # ใส่กระดาษสีขาว
    cv2.rectangle(img, (120, 90), (680, 510), (240, 240, 240), -1)
    blurred = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    res = detect_document(blurred, (h, w))
    assert res["success"] is True
    assert res["corners"] is not None
    assert res["corners"].shape == (4, 2)
    assert res["method"] in ("contour", "minarearect")


def test_detect_document_failure_handling():
    """ทดสอบกรณีภาพสีทึบล้วน/ไม่มีเอกสาร ระบบต้องไม่ crash และคืน success=False พร้อมข้อความอธิบาย"""
    blank = np.ones((500, 500), dtype=np.uint8) * 128
    blurred = cv2.GaussianBlur(blank, (5, 5), 0)

    res = detect_document(blurred, (500, 500))
    assert res["success"] is False
    assert len(res["message"]) > 0


def test_detect_document_real_sample():
    """ทดสอบกับภาพตัวอย่างจริง test1.webp"""
    sample_path = os.path.join(os.path.dirname(__file__), "sample_images", "test1.webp")
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        assert img is not None
        prep = preprocess(img)
        res = detect_document(prep["blurred"], prep["resized"].shape[:2])
        assert res["success"] is True
        assert res["corners"] is not None
        assert res["corners"].shape == (4, 2)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
