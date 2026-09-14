"""
features.py
Feature extraction และ matching สำหรับ rubric ของวิชา:
  - SIFT หรือ ORB (keypoints + descriptors)
  - BFMatcher หรือ FLANN matching
  - Lowe's ratio test เพื่อกรอง good matches
"""

import cv2
import numpy as np


# -----------------------------------------------------------------------
# Feature Extraction
# -----------------------------------------------------------------------

def extract_features(
    img_gray: np.ndarray, method: str = "SIFT"
) -> tuple[list, np.ndarray | None]:
    """
    สกัด keypoints และ descriptors จากภาพ grayscale
    Args:
        img_gray : grayscale image
        method   : 'SIFT' หรือ 'ORB'
    Returns:
        (keypoints, descriptors)
    """
    if method == "SIFT":
        detector = cv2.SIFT_create(nfeatures=1000)
    else:  # ORB
        detector = cv2.ORB_create(nfeatures=1000)

    keypoints, descriptors = detector.detectAndCompute(img_gray, None)
    return keypoints, descriptors


# -----------------------------------------------------------------------
# Feature Matching
# -----------------------------------------------------------------------

def match_features(
    desc1: np.ndarray,
    desc2: np.ndarray,
    method: str = "SIFT",
    matcher: str = "BFMatcher",
) -> list:
    """
    Match descriptors ระหว่างสองภาพ
    Args:
        desc1, desc2 : descriptors จาก extract_features
        method       : 'SIFT' หรือ 'ORB' (ใช้กำหนด norm type)
        matcher      : 'BFMatcher' หรือ 'FLANN'
    Returns:
        raw_matches (list of list[DMatch]) — knnMatch k=2
    """
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return []

    if matcher == "BFMatcher":
        norm = cv2.NORM_L2 if method == "SIFT" else cv2.NORM_HAMMING
        bf = cv2.BFMatcher(norm, crossCheck=False)
        raw_matches = bf.knnMatch(desc1, desc2, k=2)

    else:  # FLANN
        if method == "SIFT":
            index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
        else:  # ORB — ใช้ LSH
            index_params = dict(
                algorithm=6,          # FLANN_INDEX_LSH
                table_number=6,
                key_size=12,
                multi_probe_level=1,
            )
            search_params = dict(checks=50)

        # desc ต้องเป็น float32 สำหรับ FLANN KD-Tree
        d1 = desc1.astype(np.float32) if method == "SIFT" else desc1
        d2 = desc2.astype(np.float32) if method == "SIFT" else desc2

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = flann.knnMatch(d1, d2, k=2)

    return raw_matches


# -----------------------------------------------------------------------
# Lowe's Ratio Test
# -----------------------------------------------------------------------

def apply_ratio_test(raw_matches: list, ratio: float = 0.75) -> list:
    """
    Lowe's ratio test: เก็บเฉพาะ match ที่ระยะของ best match
    น้อยกว่า ratio เท่าของ second-best match
    Returns:
        good_matches (list[DMatch])
    """
    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


# -----------------------------------------------------------------------
# Pipeline wrapper
# -----------------------------------------------------------------------

def extract_and_match(
    img1_gray: np.ndarray,
    img2_gray: np.ndarray,
    method: str = "SIFT",
    matcher: str = "BFMatcher",
    ratio: float = 0.75,
) -> dict:
    """
    รัน feature pipeline ครบ: extract → match → ratio test
    Returns dict:
      'kp1', 'kp2'         : keypoints
      'desc1', 'desc2'     : descriptors
      'raw_matches'        : matches ก่อน ratio test
      'good_matches'       : matches หลัง ratio test
      'n_keypoints_1/2'    : จำนวน keypoints
      'n_good'             : จำนวน good matches
    """
    kp1, desc1 = extract_features(img1_gray, method)
    kp2, desc2 = extract_features(img2_gray, method)
    raw_matches = match_features(desc1, desc2, method, matcher)
    good_matches = apply_ratio_test(raw_matches, ratio)

    return {
        "kp1": kp1,
        "kp2": kp2,
        "desc1": desc1,
        "desc2": desc2,
        "raw_matches": raw_matches,
        "good_matches": good_matches,
        "n_keypoints_1": len(kp1),
        "n_keypoints_2": len(kp2),
        "n_good": len(good_matches),
    }
