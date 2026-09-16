"""
detection.py
ตรวจจับเอกสารและหา 4 มุม:
  1. Canny edge detection + morphological closing
  2. หา contour ที่ใหญ่ที่สุด ที่ approx เป็น 4 จุด (เลือกตาม score ไม่ใช่ตัวแรก)
  3. Fallback: minAreaRect บน largest contour
  4. Fallback สุดท้าย: ใช้มุมภาพทั้งหมด — แต่รายงานว่า "ไม่สำเร็จ" ไม่ใช่ "สำเร็จ"
  5. จัดเรียงมุมด้วยมุมรอบจุดศูนย์ถ่วง (atan2) → TL, TR, BR, BL
  6. ตรวจความสมเหตุสมผลของ quad ก่อนส่งต่อให้ geometry
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

    # A4 = 1.414, ให้ bonus ถ้าสัดส่วนอยู่ใน 1.0 - 2.5
    aspect_score = 1.0 if 1.0 <= aspect <= 2.5 else 0.5

    return area_ratio * aspect_score


def find_document_contour(edges: np.ndarray, img_shape: tuple) -> tuple[np.ndarray | None, str]:
    """
    หา contour ที่น่าจะเป็นกระดาษ/เอกสาร
    - เก็บเฉพาะ contour ที่ approxPolyDP ได้ 4 จุด
    - เลือกอันที่มี score (area × aspect_ratio) สูงสุด ไม่ใช่แค่ตัวแรก

    คืน (corners (4,2), method) โดย method บอกว่ามาจากทางไหน:
      'contour'     ตรวจเจอสี่เหลี่ยมจริง — เชื่อถือได้
      'minarearect' เดาจากกรอบรอบ contour ใหญ่สุด — พอใช้
      'fullframe'   เดาไม่ออก ใช้ทั้งภาพแทน — ถือว่าตรวจไม่สำเร็จ
    """
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, "none"

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
        return best_approx.reshape(4, 2).astype(np.float32), "contour"

    # ---- Fallback 1: minAreaRect บน largest contour ----
    large_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if large_contours:
        largest = max(large_contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32), "minarearect"

    # ---- Fallback 2: ใช้มุมภาพทั้งหมด (full-frame) ----
    # ไม่ใช่การตรวจจับสำเร็จ — เป็นแค่ค่าตั้งต้นให้ผู้ใช้เห็นว่าระบบเดาอะไรอยู่
    h, w = img_shape[:2]
    margin = 10
    full = np.array(
        [[margin, margin], [w - margin, margin],
         [w - margin, h - margin], [margin, h - margin]],
        dtype=np.float32,
    )
    return full, "fullframe"


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    จัดเรียง 4 จุดให้เป็น: [TL, TR, BR, BL]

    ใช้มุมรอบจุดศูนย์ถ่วง (atan2) แทนวิธี sum/diff แบบเดิม
    เพราะ sum/diff จะจับจุดเดียวกันซ้ำสองช่องเมื่อกระดาษเอียงเข้าใกล้ 45°
    (เจอบ่อยกับผลจาก cv2.boxPoints) แล้วทำให้ homography degenerate จนภาพออกมาดำ
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    centre = pts.mean(axis=0)

    # ในระบบพิกัดภาพ (y ชี้ลง) การเรียงมุมจากน้อยไปมากได้ลำดับตามเข็มนาฬิกาบนจอ
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]

    # หมุน list ให้จุดซ้ายบนสุด (x+y น้อยสุด) มาอยู่ตำแหน่งแรก → TL, TR, BR, BL
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0).astype(np.float32)


def validate_quad(corners: np.ndarray, img_shape: tuple, min_area_ratio: float = 0.03) -> tuple[bool, str]:
    """
    ตรวจว่า quad ที่ได้เอาไปคำนวณ homography ได้จริงไหม
    คืน (ok, เหตุผลถ้าไม่ผ่าน)
    """
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        return False, "ตรวจจับมุมได้ไม่ครบ 4 จุด"

    h, w = img_shape[:2]
    diag = float(np.hypot(h, w))

    # จุดซ้ำหรือใกล้กันเกินไป → getPerspectiveTransform ได้เมทริกซ์ degenerate แล้วภาพออกมาดำ
    for i in range(4):
        for j in range(i + 1, 4):
            if float(np.linalg.norm(pts[i] - pts[j])) < diag * 0.02:
                return False, "มุมที่ตรวจได้ซ้ำหรือใกล้กันเกินไป จนคำนวณ perspective ไม่ได้"

    area = abs(cv2.contourArea(pts))
    if area < h * w * min_area_ratio:
        return False, "กรอบที่ตรวจได้เล็กเกินกว่าจะเป็นเอกสาร"

    # ไม่นูน = มุมไขว้กัน (เช่น ลำดับจุดผิด หรือ 3 จุดเกือบอยู่บนเส้นตรงเดียวกัน)
    if not cv2.isContourConvex(pts.astype(np.int32)):
        return False, "กรอบที่ตรวจได้ไขว้กันเอง (ไม่เป็นรูปสี่เหลี่ยมนูน)"

    return True, ""


def detect_document(blurred: np.ndarray, img_shape: tuple) -> dict:
    """
    Pipeline หลักสำหรับ detect เอกสาร

    คืน dict:
      'edges'   : ภาพ edge map
      'corners' : array (4,2) [TL,TR,BR,BL] หรือ None
      'method'  : 'contour' | 'minarearect' | 'fullframe' | 'none'
      'success' : True เฉพาะเมื่อตรวจเจอขอบเอกสารจริงและ quad ใช้งานได้
      'message' : ข้อความสถานะ

    หมายเหตุ: กรณี fullframe จะคืน corners มาด้วยเพื่อให้ UI แสดงให้ดูได้
    ว่าระบบเดาอะไรอยู่ แต่ success เป็น False เพราะยังไม่ถือว่าตรวจเจอเอกสาร
    """
    edges = detect_edges(blurred)
    corners_raw, method = find_document_contour(edges, img_shape)

    if corners_raw is None:
        return {
            "edges": edges,
            "corners": None,
            "method": method,
            "success": False,
            "message": "ไม่พบขอบใด ๆ ในภาพ — ลองปรับแสง หรือให้กระดาษตัดกับพื้นหลังชัดขึ้น",
        }

    corners = order_corners(corners_raw)
    ok, reason = validate_quad(corners, img_shape)

    if not ok:
        return {
            "edges": edges,
            "corners": corners,
            "method": method,
            "success": False,
            "message": f"ตรวจจับเอกสารไม่สำเร็จ: {reason}",
        }

    if method == "fullframe":
        return {
            "edges": edges,
            "corners": corners,
            "method": method,
            "success": False,
            "message": (
                "ไม่พบขอบเอกสารที่ชัดเจน — ระบบกำลังเดาด้วยกรอบเต็มภาพ "
                "ลองถ่ายให้กระดาษตัดกับพื้นหลังมากขึ้น หรือใช้โหมด Reference"
            ),
        }

    label = "ตรวจจับ 4 มุมจากขอบเอกสารสำเร็จ" if method == "contour" \
        else "ตรวจจับด้วยกรอบรอบวัตถุที่ใหญ่ที่สุด (ความมั่นใจปานกลาง)"

    return {
        "edges": edges,
        "corners": corners,
        "method": method,
        "success": True,
        "message": f"พบเอกสาร — {label}",
    }
