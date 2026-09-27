"""
verify_demo_suite.py
สคริปต์ตรวจสอบความสมบูรณ์ของชุดภาพ Demo ทั้งหมดในโฟลเดอร์ demo/
รันตรวจสอบการทำงานของ Pipeline จริงทั้งโหมด Auto และ Reference
"""

import os
import sys
import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.preprocessing import preprocess
from src.detection import detect_document
from src.features import extract_and_match
from src.geometry import rectify_from_corners, rectify_from_reference
from src.enhancement import apply_filter, FILTER_MAGIC

DEMO_DIR = os.path.dirname(__file__)
STD_DIR = os.path.join(DEMO_DIR, "01_Standard_Documents")
REF_DIR = os.path.join(DEMO_DIR, "02_Reference_Pairs")
EDGE_DIR = os.path.join(DEMO_DIR, "03_Edge_Cases")


def test_standard_documents():
    print("\n" + "=" * 70)
    print("1. ทดสอบหมวด 01_Standard_Documents (โหมด Auto: Contour -> Rectify)")
    print("=" * 70)
    files = sorted(os.listdir(STD_DIR))
    for f in files:
        if not f.lower().endswith((".jpg", ".png", ".webp")):
            continue
        p = os.path.join(STD_DIR, f)
        img = cv2.imread(p)
        if img is None:
            print(f"[FAIL] Cannot read {f}")
            continue

        prep = preprocess(img)
        det = detect_document(prep["blurred"], prep["blurred"].shape)
        status = "PASS" if det["success"] else "FALLBACK"
        
        info = ""
        if det["success"]:
            rect = rectify_from_corners(img, det["corners"], prep["scale"], 800)
            enh = apply_filter(rect["warped"], FILTER_MAGIC)
            info = f"-> Warped A4: {rect['dst_size']}, Landscape: {rect['landscape']}, Enhanced: {enh.shape}"
        
        print(f"[{status:8s}] {f:32s} | Method: {det['method']:11s} | {info}")


def test_reference_pairs():
    print("\n" + "=" * 70)
    print("2. ทดสอบหมวด 02_Reference_Pairs (โหมด Reference: SIFT + RANSAC)")
    print("=" * 70)
    pairs = [
        ("pair1_thai_tax_invoice_flat.jpg", "pair1_thai_tax_invoice_skewed.jpg", "Thai Tax Invoice Pair"),
        ("pair2_aurora_invoice_flat.jpg", "pair2_aurora_invoice_skewed.jpg", "Aurora Tax Invoice Pair"),
        ("pair3_apex_invoice_flat.jpg", "pair3_apex_invoice_skewed.jpg", "Apex Consulting Invoice Pair"),
        ("pair4_cp461_spec_report_flat.jpg", "pair4_cp461_spec_report_skewed.jpg", "CP461 Specification Report Pair"),
    ]

    for flat_f, skew_f, label in pairs:
        flat_p = os.path.join(REF_DIR, flat_f)
        skew_p = os.path.join(REF_DIR, skew_f)
        if not os.path.exists(flat_p) or not os.path.exists(skew_p):
            print(f"[SKIP] Pair not found: {label}")
            continue

        ref_img = cv2.imread(flat_p)
        photo_img = cv2.imread(skew_p)

        p_prep = preprocess(photo_img)
        r_prep = preprocess(ref_img)

        feat = extract_and_match(p_prep["gray"], r_prep["gray"], "SIFT", "BFMatcher", 0.75)
        res = rectify_from_reference(photo_img, p_prep["scale"], r_prep["resized"], feat, 800)
        
        inlier_pct = (res["n_inliers"] / max(feat["n_good"], 1)) * 100
        status = "PASS" if res["success"] else "FAIL"
        print(f"[{status:4s}] {label:32s} | Good Matches: {feat['n_good']:4d} | Inliers: {res['n_inliers']:3d} ({inlier_pct:5.1f}%) | Result Size: {res['dst_size']}")


def test_edge_cases():
    print("\n" + "=" * 70)
    print("3. ทดสอบหมวด 03_Edge_Cases (ทดสอบการตอบสนองต่อกรณีขอบเขต & Fallback)")
    print("=" * 70)
    files = sorted(os.listdir(EDGE_DIR))
    for f in files:
        if not f.lower().endswith((".jpg", ".png", ".webp")):
            continue
        p = os.path.join(EDGE_DIR, f)
        img = cv2.imread(p)
        prep = preprocess(img)
        det = detect_document(prep["blurred"], prep["blurred"].shape)
        
        expected_behavior = ""
        if "shadow" in f:
            expected_behavior = "แสงเงาทับขอบ -> Fallback fullframe (แนะนำ Manual Adjustment)"
        elif "clutter" in f:
            expected_behavior = "พื้นหลังรก -> Fallback minarearect/fullframe"
        elif "occluded" in f:
            expected_behavior = "มุมถูกบดบัง -> ตรวจจับได้บางส่วน (สาธิตลากมุมแมนนวล)"
        elif "low_contrast" in f:
            expected_behavior = "สีกลืนกับพื้น -> Fallback fullframe (สาธิตโหมด Reference)"
        elif "extreme" in f:
            expected_behavior = "มุมเอียงรุนแรง -> Fallback/Skew (สาธิตโหมด Reference)"

        print(f"[INFO] {f:30s} | Method: {det['method']:11s} | Status: Success={det['success']} | {expected_behavior}")


if __name__ == "__main__":
    print("Starting Comprehensive Verification of Demo Suite...")
    test_standard_documents()
    test_reference_pairs()
    test_edge_cases()
    print("\n" + "=" * 70)
    print("Verification Completed Successfully!")
    print("=" * 70)
