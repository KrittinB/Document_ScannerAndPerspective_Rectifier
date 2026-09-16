"""
enhancement.py
ระบบปรับแต่งภาพเอกสารหลัง Perspective Rectification (Post-Processing Pipeline):
  1. Original Color  — ภาพสีต้นฉบับตามจริง
  2. Magic Color     — ปรับความคมชัด ลบเงา (Shadow Removal) และเร่งสีด้วย CLAHE ใน LAB Space
  3. Clean B&W       — แปลงเป็นเอกสารขาว-ดำคมชัด (Binarization) พื้นหลังขาวบริสุทธิ์แบบเครื่องสแกน
  4. Grayscale Scan  — เฉดสีเทา ปรับสมดุลแสงและยืด Contrast เหมาะกับเอกสารตัวเขียนหรือภาพพิมพ์
"""

import cv2
import numpy as np

FILTER_ORIGINAL = "Original Color"
FILTER_MAGIC = "Magic Color (Auto Enhance)"
FILTER_BW = "Clean B&W (Scanner)"
FILTER_GRAY = "Grayscale Scan"

FILTER_MODES = [FILTER_ORIGINAL, FILTER_MAGIC, FILTER_BW, FILTER_GRAY]


def _estimate_background_channel(channel: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    """
    ประมาณระนาบแสงพื้นหลัง (Illumination Map) ของ 1 channel
    ด้วย Morphological Dilation ขนาดใหญ่ร่วมกับ Median Blur
    เพื่อเกลี่ยสีของตัวหนังสือออก เหลือเฉพาะแสงของผิวกระดาษ
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dilated = cv2.dilate(channel, kernel)
    # ใช้ Median Blur เพื่อเกลี่ยรอยต่อให้เรียบเนียน
    blur_k = max(3, (k // 2) * 2 + 1)
    bg = cv2.medianBlur(dilated, blur_k)
    return bg


def remove_shadows(image_bgr: np.ndarray, kernel_size: int = 35) -> np.ndarray:
    """
    กำจัดเงาและปรับแสงให้สม่ำเสมอทั่วทั้งแผ่น (Illumination Normalization)
    โดยใช้หลักการ Division Normalization: I_norm = (I / B) * 255
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    channels = cv2.split(image_bgr)
    norm_channels = []

    for ch in channels:
        bg = _estimate_background_channel(ch, kernel_size=kernel_size)
        # ป้องกันการหารด้วยศูนย์ด้วย np.maximum
        bg_f = np.maximum(bg.astype(np.float32), 1.0)
        ch_f = ch.astype(np.float32)
        # ปรับค่าให้พื้นหลังสว่างเป็นสีขาว (255)
        norm = np.clip((ch_f / bg_f) * 255.0, 0, 255).astype(np.uint8)
        norm_channels.append(norm)

    return cv2.merge(norm_channels)


def enhance_magic_color(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    brightness: int = 0,
    contrast: float = 1.05,
    unsharp: bool = True,
) -> np.ndarray:
    """
    Magic Color Mode:
      1. ลบเงาและปรับระนาบแสงด้วย Division Normalization
      2. แปลงเข้าสู่ระบบสี LAB เพื่อปรับ CLAHE บนเฉพาะช่อง L (Luminance)
      3. ปรับ Brightness / Contrast
      4. ทำ Unsharp Masking เสริมความคมชัดของขอบตัวอักษร
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # ขั้นตอนที่ 1: ลบเงามืด
    de_shadowed = remove_shadows(image_bgr, kernel_size=35)

    # ขั้นตอนที่ 2: ปรับความคมชัดของแสงในช่อง L ด้วย CLAHE
    lab = cv2.cvtColor(de_shadowed, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)

    # รวมกลับเป็น BGR
    lab_merged = cv2.merge([l_enhanced, a_ch, b_ch])
    enhanced = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

    # ขั้นตอนที่ 3: ปรับ Brightness & Contrast
    if contrast != 1.0 or brightness != 0:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=brightness)

    # ขั้นตอนที่ 4: Unsharp Masking เพิ่มความคมกริบของลายเส้น/ตัวหนังสือ
    if unsharp:
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.35, gaussian, -0.35, 0)

    return enhanced


def enhance_clean_bw(
    image_bgr: np.ndarray,
    block_size: int = 21,
    c_val: int = 11,
    denoise: bool = True,
) -> np.ndarray:
    """
    Clean B&W (Scanner / Binary Mode):
      1. แปลงเป็น Grayscale และลบเงามืดของผิวกระดาษ
      2. ทำ Adaptive Gaussian Thresholding เพื่อคัดแยกข้อความออกจากพื้นหลัง
      3. กรองจุด Noise เล็กๆ ด้วย Median Filter
      4. คืนค่าเป็นภาพ 3 Channels (BGR) เพื่อความเข้ากันได้กับ UI
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # ขั้นตอนที่ 1: แปลงเป็น Grayscale แล้วลบเงามืดก่อนทำ Binarization
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bg = _estimate_background_channel(gray, kernel_size=35)
    bg_f = np.maximum(bg.astype(np.float32), 1.0)
    gray_norm = np.clip((gray.astype(np.float32) / bg_f) * 255.0, 0, 255).astype(np.uint8)

    # ขั้นตอนที่ 2: Adaptive Gaussian Thresholding
    bs = max(3, block_size if block_size % 2 == 1 else block_size + 1)
    binary = cv2.adaptiveThreshold(
        gray_norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bs,
        c_val,
    )

    # ขั้นตอนที่ 3: กรองเกล็ดหมึกและจุดรบกวนขนาดเล็ก (Salt & Pepper Noise)
    if denoise:
        binary = cv2.medianBlur(binary, 3)

    # แปลงกลับเป็น 3 channels สำหรับแสดงผลบน Web App
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def enhance_grayscale(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    brightness: int = 5,
    contrast: float = 1.1,
) -> np.ndarray:
    """
    Grayscale Scan Mode:
      1. แปลงเป็น Grayscale และลบเงาพื้นหลัง
      2. ปรับ Contrast ด้วย CLAHE
      3. คืนค่าเป็นภาพ 3 Channels (BGR)
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bg = _estimate_background_channel(gray, kernel_size=35)
    bg_f = np.maximum(bg.astype(np.float32), 1.0)
    gray_norm = np.clip((gray.astype(np.float32) / bg_f) * 255.0, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray_norm)

    if contrast != 1.0 or brightness != 0:
        enhanced_gray = cv2.convertScaleAbs(enhanced_gray, alpha=contrast, beta=brightness)

    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)


def apply_filter(
    image_bgr: np.ndarray,
    filter_mode: str,
    brightness: int = 0,
    contrast: float = 1.0,
    bw_threshold_c: int = 11,
) -> np.ndarray:
    """
    Router ฟังก์ชันหลักสำหรับเรียกใช้ Filter ตามโหมดที่ผู้ใช้เลือก:
      - FILTER_ORIGINAL : ไม่แต่งภาพ คืนภาพเดิม
      - FILTER_MAGIC    : โหมด Magic Color ปรับความสดใสและลบเงา
      - FILTER_BW       : โหมด Clean B&W เอกสารขาวดำคมชัด
      - FILTER_GRAY     : โหมด Grayscale ปรับแสงสม่ำเสมอ
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    try:
        if filter_mode == FILTER_MAGIC:
            return enhance_magic_color(
                image_bgr,
                brightness=brightness,
                contrast=max(0.5, contrast),
            )
        elif filter_mode == FILTER_BW:
            return enhance_clean_bw(
                image_bgr,
                c_val=bw_threshold_c,
            )
        elif filter_mode == FILTER_GRAY:
            return enhance_grayscale(
                image_bgr,
                brightness=brightness,
                contrast=max(0.5, contrast),
            )
        else:
            # FILTER_ORIGINAL หรือค่าอื่นๆ
            if contrast != 1.0 or brightness != 0:
                return cv2.convertScaleAbs(image_bgr, alpha=contrast, beta=brightness)
            return image_bgr.copy()

    except Exception:
        # หากเกิดข้อผิดพลาด ให้คืนภาพดั้งเดิมอย่างปลอดภัย
        return image_bgr.copy()
