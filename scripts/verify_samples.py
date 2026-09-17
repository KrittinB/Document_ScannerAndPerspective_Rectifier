"""
verify_samples.py
ทดสอบรัน pipeline กับทุกไฟล์ภาพใน tests/sample_images/
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
from src.preprocessing import preprocess
from src.detection import detect_document
from src.features import extract_and_match
from src.geometry import rectify_from_reference, rectify_from_corners

samples_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "sample_images")

print("================================================================")
print("1. ทดสอบโหมด Auto กับชุดภาพตัวอย่างทั้งหมด (รวม Edge Cases)")
print("================================================================")

test_files = [
    "test1.webp",
    "test2.webp",
    "edge1_extreme_perspective.jpg",
    "edge2_heavy_shadow.jpg",
    "edge3_cluttered_background.jpg",
    "edge4_corner_occluded.jpg",
    "edge5_low_contrast.jpg",
    "ref1_skewed_photo.jpg",
]

for fname in test_files:
    fpath = os.path.join(samples_dir, fname)
    if not os.path.exists(fpath):
        print(f"Skipping {fname} (not found)")
        continue
    img = cv2.imread(fpath)
    prep = preprocess(img)
    det = detect_document(prep["blurred"], prep["blurred"].shape)

    rect_info = ""
    if det["success"]:
        rect = rectify_from_corners(img, det["corners"], prep["scale"], 800)
        rect_info = f" -> Warped A4: {rect['dst_size']}, landscape={rect['landscape']}"

    print(f"[{fname:30s}] Success: {str(det['success']):5s} | Method: {det['method']:11s} | Msg: {det['message']}{rect_info}")

print("\n================================================================")
print("2. ทดสอบโหมด Reference (SIFT + RANSAC) กับคู่ภาพ ref1")
print("================================================================")
photo_path = os.path.join(samples_dir, "ref1_skewed_photo.jpg")
ref_path = os.path.join(samples_dir, "ref1_flat_reference.jpg")

photo = cv2.imread(photo_path)
ref = cv2.imread(ref_path)

p_prep = preprocess(photo)
r_prep = preprocess(ref)

feat = extract_and_match(p_prep["gray"], r_prep["gray"], "SIFT", "BFMatcher", 0.75)
print(f"Matches: raw={len(feat['raw_matches'])}, good={feat['n_good']}")

res = rectify_from_reference(photo, p_prep["scale"], r_prep["resized"], feat, 800)
ratio = res["n_inliers"] / max(feat["n_good"], 1) * 100
print(f"Reference rectify: success={res['success']}, inliers={res['n_inliers']}/{feat['n_good']} ({ratio:.1f}%), dst_size={res['dst_size']}")
print(f"Message: {res['message']}")

print("\nVerification completed successfully!")
