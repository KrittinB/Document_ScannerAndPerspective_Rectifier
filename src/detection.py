"""
detection.py
ตรวจจับเอกสารและหา 4 มุม:
  1. Canny edge detection + morphological closing
  2. หา contour ที่ ใหญ่ที่สุด ที่ approx เป็น 4 จุด (เลือกตาม area ไม่ใช่ตัวแรก)
  3. Fallback: minAreaRect บน largest contour
  4. Fallback สุดท้าย: ใช้มุมภาพทั้งหมด
  5. จัดเรียงมุม: TL, TR, BR, BL
"""

import cv2
import numpy as np


def detect_edges(blurred: np.ndarray) -> np.ndarray:
    """
    Canny edge detection พร้อม adaptive threshold
    ใช้ median intensity เพื่อปรับ threshold อัตโนมัติ
    เพิ่ม morphological closing เพื่อปิดช่องว่างของขอบกระดาษ
    """
    median = np.median(blurred)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(blurred, lower, upper)

    # Closing: ปิดช่องว่างขอบกระดาษ (dilate → erode)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return edges


def _score_quad(approx: np.ndarray, img_shape: tuple) -> float:
    """
    ให้คะแนน quadrilateral — ยิ่งใหญ่และสัดส่วนใกล้ A4 ยิ่งได้คะแนนสูง
    """
    area = cv2.contourArea(approx)
    img_area = img_shape[0] * img_shape[1]
    area_ratio = area / img_area  # 0..1

    # คำนวณ bounding box เพื่อดู aspect ratio
    x, y, w, h = cv2.boundingRect(approx)
    if h == 0 or w == 0:
        return 0.0
    aspect = max(w, h) / min(w, h)

    # A4 = 1.414, ให้ bonus ถ้าสัดส่วนอยู่ใน 1.0 - 2.0
    aspect_score = 1.0 if 1.0 <= aspect <= 2.5 else 0.5

    return area_ratio * aspect_score


def find_document_contour(edges: np.ndarray, img_shape: tuple) -> np.ndarray | None:
    """
    หา contour ที่น่าจะเป็นกระดาษ/เอกสาร
    - เก็บเฉพาะ contour ที่ approxPolyDP ได้ 4 จุด
    - เลือกอันที่มี score (area × aspect_ratio) สูงสุด ไม่ใช่แค่ตัวแรก
    คืน array (4,2) หรือ None
    """
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    img_area = img_shape[0] * img_shape[1]
    min_area = img_area * 0.05  # ต้องใหญ่กว่า 5% ของภาพ

    best_approx = None
    best_score = -1.0

    # ลอง epsilon หลายค่า — บางภาพต้องการ epsilon ต่างกัน
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps_factor in [0.01, 0.02, 0.03, 0.05]:
            approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
            if len(approx) == 4:
                score = _score_quad(approx, img_shape)
                if score > best_score:
                    best_score = score
                    best_approx = approx
                break  # ได้ 4 จุดแล้ว ไม่ต้องลอง epsilon อื่น

    if best_approx is not None:
        return best_approx.reshape(4, 2).astype(np.float32)

    # ---- Fallback 1: minAreaRect บน largest contour ----
    large_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if large_contours:
        largest = max(large_contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32)

    # ---- Fallback 2: ใช้มุมภาพทั้งหมด (full-frame) ----
    h, w = img_shape[:2]
    margin = 10
    return np.array(
        [[margin, margin], [w - margin, margin],
         [w - margin, h - margin], [margin, h - margin]],
        dtype=np.float32,
    )


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    จัดเรียง 4 จุดให้เป็น: [TL, TR, BR, BL]
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL: ผลรวม x+y น้อยสุด
    rect[2] = pts[np.argmax(s)]   # BR: ผลรวม x+y มากสุด

    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # TR: y-x น้อยสุด
    rect[3] = pts[np.argmax(diff)]  # BL: y-x มากสุด

    return rect


def detect_document(blurred: np.ndarray, img_shape: tuple) -> dict:
    """
    Pipeline หลักสำหรับ detect เอกสาร
    คืน dict:
      'edges'   : ภาพ edge map
      'corners' : array (4,2) [TL,TR,BR,BL] (ไม่เป็น None อีกต่อไป — มี fallback)
      'success' : bool
      'message' : ข้อความสถานะ
    """
    edges = detect_edges(blurred)
    corners_raw = find_document_contour(edges, img_shape)

    if corners_raw is None:
        return {
            "edges": edges,
            "corners": None,
            "success": False,
            "message": "ไม่พบขอบเอกสารในภาพ — ลองปรับแสง หรือให้กระดาษตัดกับพื้นหลังชัดขึ้น",
        }

    corners = order_corners(corners_raw)
    return {
        "edges": edges,
        "corners": corners,
        "success": True,
        "message": "พบเอกสาร — ตรวจจับ 4 มุมสำเร็จ",
    }
