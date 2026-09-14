"""
preprocessing.py
ขั้นตอนเตรียมภาพก่อนประมวลผล:
  - resize ให้ขนาดไม่เกิน MAX_DIM เพื่อความเร็ว
  - แปลงเป็น grayscale
  - Gaussian blur เพื่อลด noise
"""

import cv2
import numpy as np

MAX_DIM = 1200  # pixel


def preprocess(image: np.ndarray) -> dict:
    """
    รับ BGR image (numpy array) คืน dict ที่มี:
      'resized'  : ภาพ BGR ย่อ/ขยาย
      'gray'     : grayscale
      'blurred'  : grayscale + Gaussian blur
      'scale'    : อัตราส่วน resize (สำหรับ map กลับไปขนาดเดิม)
    """
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Gaussian blur kernel ขนาด 5x5 — ลด noise ก่อน edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return {
        "resized": resized,
        "gray": gray,
        "blurred": blurred,
        "scale": scale,
    }
