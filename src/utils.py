"""
utils.py
Helper functions สำหรับ:
  - วาด visualization (corners, matches, inlier/outlier)
  - แปลง numpy BGR → PIL Image สำหรับ Streamlit
  - แปลงไฟล์ upload → numpy array
"""

import cv2
import numpy as np
from PIL import Image
import io


# -----------------------------------------------------------------------
# Image format conversion
# -----------------------------------------------------------------------

def pil_to_numpy_bgr(pil_image: Image.Image) -> np.ndarray:
    """PIL Image (RGB) → numpy BGR"""
    arr = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def numpy_bgr_to_pil(img: np.ndarray) -> Image.Image:
    """numpy BGR → PIL Image (RGB)"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def numpy_gray_to_pil(img: np.ndarray) -> Image.Image:
    """numpy grayscale → PIL Image"""
    return Image.fromarray(img)


def bytes_to_numpy_bgr(file_bytes: bytes) -> np.ndarray:
    """bytes (จาก st.file_uploader) → numpy BGR"""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# -----------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------

def draw_corners(
    image: np.ndarray,
    corners: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 3,
    dot_radius: int = 8,
) -> np.ndarray:
    """
    วาดกรอบเอกสาร + จุดมุมบนภาพ
    corners: (4,2) [TL,TR,BR,BL]
    """
    out = image.copy()
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)

    labels = ["TL", "TR", "BR", "BL"]
    colors_dot = [
        (0, 255, 0),    # TL - เขียว
        (255, 165, 0),  # TR - ส้ม
        (0, 0, 255),    # BR - แดง
        (255, 0, 255),  # BL - ม่วง
    ]
    for i, (pt, label, c) in enumerate(zip(corners.astype(int), labels, colors_dot)):
        cv2.circle(out, tuple(pt), dot_radius, c, -1)
        cv2.putText(
            out, label,
            (pt[0] + 10, pt[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, c, 2,
        )
    return out


def draw_matches_visualization(
    img1: np.ndarray,
    kp1: list,
    img2: np.ndarray,
    kp2: list,
    good_matches: list,
    max_draw: int = 50,
) -> np.ndarray:
    """
    วาดเส้น feature matching ระหว่างสองภาพ
    ใช้ cv2.drawMatchesKnn (แปลง good_matches เป็น list of list ก่อน)
    """
    # drawMatchesKnn ต้องการ list[list[DMatch]]
    matches_knn = [[m] for m in good_matches[:max_draw]]

    out = cv2.drawMatchesKnn(
        img1, kp1,
        img2, kp2,
        matches_knn,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return out


def draw_inlier_outlier(
    image: np.ndarray,
    keypoints: list,
    good_matches: list,
    mask: np.ndarray | None,
    max_draw: int = 200,
) -> np.ndarray:
    """
    วาด keypoints ที่ผ่าน ratio test บนภาพ
      - inlier (mask=1): สีเขียว
      - outlier (mask=0): สีแดง
    ถ้า mask เป็น None → วาดทุก good_matches เป็นสีเขียว
    """
    out = image.copy()

    if mask is None or len(mask) == 0:
        # วาดทุก good_matches เป็นสีเขียว
        for i, m in enumerate(good_matches[:max_draw]):
            if m.queryIdx < len(keypoints):
                pt = tuple(map(int, keypoints[m.queryIdx].pt))
                cv2.circle(out, pt, 5, (0, 255, 0), -1)
        return out

    mask_flat = mask.flatten()
    for i, m in enumerate(good_matches[:max_draw]):
        if m.queryIdx >= len(keypoints):
            continue
        pt = tuple(map(int, keypoints[m.queryIdx].pt))
        if i < len(mask_flat) and mask_flat[i]:
            cv2.circle(out, pt, 5, (0, 255, 0), -1)   # inlier: เขียว
        else:
            cv2.circle(out, pt, 5, (0, 0, 255), -1)   # outlier: แดง

    # legend
    cv2.putText(out, "Inlier", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(out, "Outlier", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def numpy_to_bytes_png(img: np.ndarray) -> bytes:
    """numpy BGR → PNG bytes (สำหรับ st.download_button)"""
    pil = numpy_bgr_to_pil(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()
