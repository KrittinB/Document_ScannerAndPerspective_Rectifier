"""
geometry.py
Geometric transformation pipeline — มี 2 เส้นทางที่แยกกันชัดเจน:

  rectify_from_corners()    โหมด Auto (ภาพเดียว)
                            H มาจาก 4 มุมที่ detection หาได้ → getPerspectiveTransform
                            ไม่มี feature matching เข้ามาเกี่ยวข้อง และไม่อ้างว่ามี

  rectify_from_reference()  โหมด Reference (2 ภาพ)
                            H มาจาก SIFT/ORB matches ที่ผ่าน Lowe's ratio test
                            แล้วประมาณด้วย RANSAC จริง ๆ → ใช้ H ตัวนั้น warp จริง

ทั้งสองโหมด warp จาก "ภาพต้นฉบับความละเอียดเต็ม" ไม่ใช่ภาพที่ย่อไว้ตอน detect
โดยรับ scale มาแปลงพิกัดกลับ (ดู preprocessing.preprocess)
"""

import cv2
import numpy as np

A4_RATIO = 1.4142135623730951  # sqrt(2)


# -----------------------------------------------------------------------
# ขนาดผลลัพธ์
# -----------------------------------------------------------------------

def get_a4_dimensions(base: int = 800, landscape: bool = False) -> tuple[int, int]:
    """
    คืน (width, height) ที่สัดส่วน A4 (1:sqrt(2))
    base = ด้านยาวของ output เป็น pixel
    landscape=True สำหรับเอกสารแนวนอน (ไม่งั้นภาพแนวนอนจะถูกบีบลงกรอบแนวตั้ง)
    """
    short_side = int(round(base / A4_RATIO))
    if landscape:
        return base, short_side
    return short_side, base


def quad_is_landscape(corners: np.ndarray) -> bool:
    """
    ดูจากความยาวด้านของ quad ว่าเอกสารเป็นแนวนอนหรือแนวตั้ง
    corners ต้องเรียงเป็น [TL, TR, BR, BL] แล้ว
    """
    tl, tr, br, bl = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return bool(width > height)


# -----------------------------------------------------------------------
# Homography helpers
# -----------------------------------------------------------------------

def compute_homography(
    src_pts: np.ndarray, dst_pts: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    คำนวณ homography matrix ด้วย RANSAC
    Args:
        src_pts : (N,2) จุดใน source image
        dst_pts : (N,2) จุดใน destination image
    Returns:
        (H, mask) — H คือ 3x3 homography, mask คือ inlier mask (N,1)
        (None, None) ถ้า compute ไม่ได้
    """
    if src_pts is None or dst_pts is None or len(src_pts) < 4:
        return None, None

    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        cv2.RANSAC,
        ransacReprojThreshold=5.0,
        confidence=0.995,
        maxIters=2000,
    )
    return H, mask


def corners_to_homography(corners: np.ndarray, dst_size: tuple[int, int]) -> np.ndarray:
    """
    สร้าง exact 4-point transform จาก 4 มุม ไปยังกรอบขนาด dst_size = (w, h)
    corners ต้องเรียง [TL, TR, BR, BL] มาแล้ว
    """
    w, h = dst_size
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(src, dst)


def warp_perspective(image: np.ndarray, H: np.ndarray, dst_size: tuple[int, int]) -> np.ndarray:
    """
    Apply perspective warp
    Args:
        image    : BGR source image
        H        : 3x3 homography matrix
        dst_size : (width, height) ของ output
    Returns:
        warped BGR image
    """
    return cv2.warpPerspective(image, H, dst_size, flags=cv2.INTER_CUBIC)


def _scale_matrix(s: float) -> np.ndarray:
    """เมทริกซ์ย่อ/ขยายพิกัดแบบ homogeneous"""
    return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _is_usable(H: np.ndarray | None) -> bool:
    """กัน homography ที่ degenerate หรือมี inf/nan ไม่ให้หลุดไปถึง warpPerspective"""
    if H is None or H.shape != (3, 3):
        return False
    if not np.all(np.isfinite(H)):
        return False
    return abs(float(np.linalg.det(H))) > 1e-8


# -----------------------------------------------------------------------
# โหมด Auto — ภาพเดียว, H จาก 4 มุม
# -----------------------------------------------------------------------

def rectify_from_corners(
    original: np.ndarray,
    corners: np.ndarray,
    scale: float = 1.0,
    a4_base: int = 800,
) -> dict:
    """
    ดัด perspective จากมุมที่ detection หาได้

    Args:
        original : ภาพ BGR ต้นฉบับ (ความละเอียดเต็ม ไม่ใช่ภาพที่ย่อแล้ว)
        corners  : (4,2) [TL,TR,BR,BL] ในพิกัดของ "ภาพที่ย่อแล้ว"
        scale    : อัตราส่วนย่อจาก preprocess (พิกัดย่อ = scale × พิกัดเต็ม)
        a4_base  : ด้านยาวของ output เป็น pixel
    """
    landscape = quad_is_landscape(corners)
    w, h = get_a4_dimensions(a4_base, landscape)

    # แปลงพิกัดมุมกลับไปที่ภาพต้นฉบับ เพื่อให้ warp ได้รายละเอียดเต็มความละเอียดจริง
    src_full = np.asarray(corners, dtype=np.float32).reshape(4, 2) / float(scale or 1.0)
    H = corners_to_homography(src_full, (w, h))

    if not _is_usable(H):
        return {
            "warped": None, "H": None, "mask": None, "dst_size": (w, h),
            "n_inliers": 0, "used_ransac": False, "landscape": landscape,
            "mode": "auto", "success": False,
            "message": "คำนวณ Homography ไม่ได้ — มุมที่ตรวจได้ไม่สมเหตุสมผล",
        }

    warped = warp_perspective(original, H, (w, h))
    orientation = "แนวนอน" if landscape else "แนวตั้ง"
    return {
        "warped": warped,
        "H": H,
        "mask": None,          # โหมดนี้ไม่มี RANSAC จึงไม่มี inlier mask — อย่าปลอมเป็น dummy
        "dst_size": (w, h),
        "n_inliers": 0,
        "used_ransac": False,
        "landscape": landscape,
        "mode": "auto",
        "success": True,
        "message": f"ดัดมุมมองสำเร็จจาก 4 มุมที่ตรวจจับได้ — A4 {orientation} {w}×{h} px",
    }


# -----------------------------------------------------------------------
# โหมด Reference — 2 ภาพ, H จาก feature matching + RANSAC จริง
# -----------------------------------------------------------------------

def rectify_from_reference(
    photo_original: np.ndarray,
    photo_scale: float,
    reference_resized: np.ndarray,
    feature_result: dict,
    a4_base: int = 800,
    min_inliers: int = 10,
) -> dict:
    """
    ดัด perspective ด้วย homography ที่ได้จากการจับคู่ feature ระหว่าง
    ภาพถ่ายเอียง (photo) กับภาพเอกสารอ้างอิงที่แบนราบ (reference)

    เส้นทางคำนวณ (พิกัดภาพถ่ายเต็ม → A4):
        H_total = S · H_ransac · K
        โดย K ย่อพิกัดเต็ม→พิกัดที่ใช้หา keypoint, H_ransac คือผลจาก RANSAC,
        และ S ยืดกรอบภาพอ้างอิงให้เป็นสัดส่วน A4

    Args:
        photo_original    : ภาพถ่าย BGR ความละเอียดเต็ม
        photo_scale       : scale จาก preprocess ของภาพถ่าย
        reference_resized : ภาพอ้างอิง BGR ขนาดเดียวกับที่ใช้หา keypoint
        feature_result    : ผลจาก features.extract_and_match(photo_gray, ref_gray)
    """
    kp1 = feature_result.get("kp1") or []
    kp2 = feature_result.get("kp2") or []
    good = feature_result.get("good_matches") or []

    ref_h, ref_w = reference_resized.shape[:2]
    landscape = ref_w > ref_h
    w, h = get_a4_dimensions(a4_base, landscape)

    def _fail(msg: str) -> dict:
        return {
            "warped": None, "H": None, "mask": None, "dst_size": (w, h),
            "n_inliers": 0, "used_ransac": False, "landscape": landscape,
            "mode": "reference", "success": False, "message": msg,
        }

    if len(good) < 4:
        return _fail(
            f"จับคู่จุดเด่นได้เพียง {len(good)} คู่ (ต้องการอย่างน้อย 4) — "
            "ภาพอ้างอิงกับภาพถ่ายอาจไม่ใช่เอกสารหน้าเดียวกัน หรือภาพเบลอเกินไป"
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])   # พิกัดในภาพถ่าย (ย่อแล้ว)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])   # พิกัดในภาพอ้างอิง

    H_ransac, mask = compute_homography(src_pts, dst_pts)
    if not _is_usable(H_ransac) or mask is None:
        return _fail("RANSAC หา Homography ที่ใช้ได้ไม่เจอ — ลองลด Lowe's ratio หรือเปลี่ยนภาพอ้างอิง")

    n_inliers = int(mask.sum())
    if n_inliers < min_inliers:
        return _fail(
            f"RANSAC ได้ inlier เพียง {n_inliers} จุด จาก {len(good)} คู่ (ต้องการ ≥ {min_inliers}) — "
            "ยังไม่น่าเชื่อถือพอจะใช้ดัดภาพ"
        )

    # S: ยืดกรอบภาพอ้างอิงให้เป็นสัดส่วน A4 · K: แปลงพิกัดเต็ม → พิกัดที่ใช้หา keypoint
    S = np.array([[w / ref_w, 0.0, 0.0], [0.0, h / ref_h, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    K = _scale_matrix(float(photo_scale or 1.0))
    H_total = S @ H_ransac @ K

    if not _is_usable(H_total):
        return _fail("Homography รวมออกมาใช้ไม่ได้ — ลองภาพอ้างอิงที่คมชัดกว่านี้")

    warped = warp_perspective(photo_original, H_total, (w, h))
    ratio = n_inliers / max(len(good), 1) * 100.0
    return {
        "warped": warped,
        "H": H_ransac,          # เมทริกซ์ที่ RANSAC ประมาณได้ — ตัวที่เอาไปใช้จริง
        "H_total": H_total,     # ตัวที่ส่งเข้า warpPerspective (รวม scale แล้ว)
        "mask": mask,
        "dst_size": (w, h),
        "n_inliers": n_inliers,
        "n_good": len(good),
        "used_ransac": True,
        "landscape": landscape,
        "mode": "reference",
        "success": True,
        "message": (
            f"ดัดมุมมองด้วย Homography จาก RANSAC สำเร็จ — "
            f"{n_inliers}/{len(good)} inliers ({ratio:.0f}%)"
        ),
    }
