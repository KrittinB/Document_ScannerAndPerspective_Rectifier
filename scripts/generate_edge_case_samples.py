"""
generate_edge_case_samples.py
สร้างชุดภาพทดสอบ Edge Cases และ Reference Pair สำหรับ CP461 Document Scanner Project
"""

import os
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "sample_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_font(size: int, bold: bool = False):
    font_names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "tahomabd.ttf" if bold else "tahoma.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in font_names:
        win_path = os.path.join("C:\\Windows\\Fonts", name)
        if os.path.exists(win_path):
            try:
                return ImageFont.truetype(win_path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def create_realistic_document(w: int = 1200, h: int = 1697) -> np.ndarray:
    """สร้างภาพเอกสาร A4 จำลองที่มีหัวข้อ ตาราง ข้อความ และกราฟิก"""
    img = Image.new("RGB", (w, h), color=(252, 252, 250))
    draw = ImageDraw.Draw(img)

    f_title = get_font(42, bold=True)
    f_subtitle = get_font(26, bold=False)
    f_heading = get_font(28, bold=True)
    f_body = get_font(20, bold=False)
    f_small = get_font(16, bold=False)

    # Header bar
    draw.rectangle([(60, 60), (w - 60, 70)], fill=(41, 65, 148))

    # Title
    draw.text((60, 90), "CP461 COMPUTER VISION TECHNICAL REPORT", fill=(20, 20, 25), font=f_title)
    draw.text((60, 145), "Perspective Rectification & Robust Feature Matching Analysis", fill=(80, 80, 90), font=f_subtitle)

    # Metadata
    draw.line([(60, 195), (w - 60, 195)], fill=(200, 200, 205), width=2)
    draw.text((60, 210), "Document ID: SPEC-2026-CV461", fill=(100, 100, 110), font=f_small)
    draw.text((w - 280, 210), "Classification: Educational", fill=(100, 100, 110), font=f_small)

    # Section 1
    draw.text((60, 250), "1. Abstract & System Architecture", fill=(41, 65, 148), font=f_heading)
    abstract_text = (
        "This document illustrates automated perspective rectification from distorted document captures.\n"
        "Planar homography transformation maps physical quad coordinates back into canonical A4 geometry.\n"
        "Robust estimation using RANSAC filters out outlier correspondences arising from repetitive text patterns.\n"
        "Edge detectors coupled with convex polygon scoring localize document boundaries in cluttered settings."
    )
    y = 295
    for line in abstract_text.split("\n"):
        draw.text((60, y), line, fill=(40, 40, 45), font=f_body)
        y += 28

    # Section 2: Table
    y += 20
    draw.text((60, y), "2. Performance Evaluation Table", fill=(41, 65, 148), font=f_heading)
    y += 45

    # Draw Table
    tx, ty, tw, th = 60, y, w - 120, 220
    draw.rectangle([(tx, ty), (tx + tw, ty + th)], outline=(180, 180, 190), width=2)
    draw.rectangle([(tx, ty), (tx + tw, ty + 40)], fill=(235, 240, 250))
    draw.line([(tx, ty + 40), (tx + tw, ty + 40)], fill=(180, 180, 190), width=2)

    cols = [tx, tx + 240, tx + 500, tx + 750, tx + tw]
    for c in cols[1:-1]:
        draw.line([(c, ty), (c, ty + th)], fill=(180, 180, 190), width=1)

    headers = ["Feature Extractor", "Keypoints Detected", "Good Matches", "RANSAC Inliers"]
    for i, h_text in enumerate(headers):
        draw.text((cols[i] + 15, ty + 10), h_text, fill=(30, 40, 70), font=f_heading if i == 0 else f_body)

    rows = [
        ["SIFT (Scale-Invariant)", "1,248 pts", "482 pairs", "436 inliers (90.5%)"],
        ["ORB (Oriented FAST)", "890 pts", "312 pairs", "278 inliers (89.1%)"],
        ["AKAZE (Non-linear Scale)", "1,054 pts", "395 pairs", "351 inliers (88.9%)"],
        ["Canny Contour Fallback", "4 corners", "N/A (Geometric)", "100% Deterministic"],
    ]
    for r_idx, row in enumerate(rows):
        ry = ty + 45 + r_idx * 42
        if r_idx % 2 == 1:
            draw.rectangle([(tx, ry - 3), (tx + tw, ry + 39)], fill=(248, 250, 254))
        for c_idx, cell in enumerate(row):
            draw.text((cols[c_idx] + 15, ry + 6), cell, fill=(50, 50, 55), font=f_body)

    # Section 3: Detailed Description
    y = ty + th + 40
    draw.text((60, y), "3. Algorithmic Formulation & Inlier Filtering", fill=(41, 65, 148), font=f_heading)
    y += 40

    sec3_text = (
        "Let X = (u, v, 1)^T be the homogeneous coordinates of a point in the source camera image,\n"
        "and X' = (u', v', 1)^T denote the canonical target position on the corrected A4 reference plane.\n"
        "The projective transformation satisfies s * X' = H * X, where H in R^(3x3) denotes the 8-DOF matrix.\n"
        "RANSAC estimates H iteratively from candidate matches satisfying Lowe's nearest-neighbor distance ratio:\n"
        "                        d(f_i, g_j) / d(f_i, g_k) < theta (where theta = 0.75)\n"
        "Singular value decomposition (SVD) solves the overdetermined linear system Ah = 0 across inliers."
    )
    for line in sec3_text.split("\n"):
        draw.text((60, y), line, fill=(40, 40, 45), font=f_body)
        y += 28

    # Section 4: Diagram / Box representation
    y += 20
    draw.rectangle([(60, y), (w - 60, y + 260)], outline=(200, 210, 225), fill=(245, 248, 255), width=2)
    draw.text((80, y + 15), "[System Flowchart Diagram]", fill=(60, 80, 130), font=f_heading)

    # Draw diagram blocks
    blocks = [
        ("Input Photo", 90, y + 70, 180, 60),
        ("Preprocess & Blur", 320, y + 70, 200, 60),
        ("Feature / Edge", 570, y + 70, 200, 60),
        ("Homography", 820, y + 70, 180, 60),
    ]
    for b_title, bx, by, bw, bh in blocks:
        draw.rectangle([(bx, by), (bx + bw, by + bh)], fill=(255, 255, 255), outline=(41, 65, 148), width=2)
        draw.text((bx + 15, by + 18), b_title, fill=(20, 20, 30), font=f_body)

    for i in range(len(blocks) - 1):
        x_start = blocks[i][1] + blocks[i][3]
        x_end = blocks[i + 1][1]
        mid_y = blocks[i][2] + 30
        draw.line([(x_start, mid_y), (x_end, mid_y)], fill=(41, 65, 148), width=3)
        draw.polygon([(x_end, mid_y), (x_end - 10, mid_y - 6), (x_end - 10, mid_y + 6)], fill=(41, 65, 148))

    # Diagram caption
    draw.text((80, y + 180), "Figure 1: Complete end-to-end perspective rectification architecture for CP461.", fill=(100, 100, 115), font=f_small)

    # Footer
    draw.line([(60, h - 80), (w - 60, h - 80)], fill=(200, 200, 205), width=1)
    draw.text((60, h - 65), "Document Scanner & Perspective Rectifier | Group Project Demonstration", fill=(120, 120, 130), font=f_small)
    draw.text((w - 140, h - 65), "Page 1 of 1", fill=(120, 120, 130), font=f_small)

    # Convert to OpenCV BGR
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return cv_img


def create_wood_texture(w: int, h: int, base_color=(65, 95, 145)) -> np.ndarray:
    """สร้างพื้นหลังลายไม้หรือโต๊ะทำงาน"""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :] = base_color
    noise = np.random.normal(0, 8, (h, w)).astype(np.float32)
    # Grain streaks
    streaks = np.sin(np.linspace(0, 30 * math.pi, w)) * 12
    streaks = np.tile(streaks, (h, 1)).astype(np.float32)
    combined = noise + streaks

    for c in range(3):
        channel = bg[:, :, c].astype(np.float32) + combined
        bg[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
    return bg


def warp_into_canvas(doc_bgr: np.ndarray, canvas_w: int, canvas_h: int,
                     src_quad: np.ndarray, dst_quad: np.ndarray,
                     bg_bgr: np.ndarray) -> np.ndarray:
    """Warp เอกสารลงบนผืนผ้าใบพื้นหลัง พร้อมสร้างเงาธรรมชาติรอบกระดาษ"""
    H = cv2.getPerspectiveTransform(src_quad.astype(np.float32), dst_quad.astype(np.float32))

    # Warp doc
    warped_doc = cv2.warpPerspective(doc_bgr, H, (canvas_w, canvas_h), flags=cv2.INTER_LANCZOS4)

    # Mask for paper
    doc_mask = np.ones((doc_bgr.shape[0], doc_bgr.shape[1]), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(doc_mask, H, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST)

    # Drop shadow
    shadow_mask = cv2.dilate(warped_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    shadow_mask = cv2.GaussianBlur(shadow_mask, (45, 45), 0)

    # Composite: background -> shadow -> paper
    result = bg_bgr.copy().astype(np.float32)
    shadow_alpha = (shadow_mask.astype(np.float32) / 255.0)[:, :, np.newaxis] * 0.45
    result = result * (1.0 - shadow_alpha) + np.array([20, 20, 20], dtype=np.float32) * shadow_alpha

    paper_alpha = (warped_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    result = result * (1.0 - paper_alpha) + warped_doc.astype(np.float32) * paper_alpha

    return np.clip(result, 0, 255).astype(np.uint8)


def generate_all_samples():
    print("Generating base realistic document...")
    doc = create_realistic_document(1200, 1697)
    dh, dw = doc.shape[:2]
    src_corners = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float32)

    # 1. Reference Pair
    print("1. Generating Reference Pair (Flat & Skewed)...")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ref1_flat_reference.jpg"), doc, [cv2.IMWRITE_JPEG_QUALITY, 95])

    cw, ch = 1440, 1920
    wood_bg = create_wood_texture(cw, ch, base_color=(50, 70, 110))
    dst_quad_ref = np.array([
        [220, 260],
        [1180, 190],
        [1320, 1750],
        [120, 1680]
    ], dtype=np.float32)
    skewed_ref = warp_into_canvas(doc, cw, ch, src_corners, dst_quad_ref, wood_bg)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ref1_skewed_photo.jpg"), skewed_ref, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 2. Edge Case 1: Extreme Perspective
    print("2. Generating Edge Case 1: Extreme Perspective...")
    bg1 = create_wood_texture(cw, ch, base_color=(45, 55, 75))
    dst_quad_extreme = np.array([
        [680, 110],
        [1240, 95],
        [1840, 1190],
        [80, 1220]
    ], dtype=np.float32)
    edge1 = warp_into_canvas(doc, cw, ch, src_corners, dst_quad_extreme, bg1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "edge1_extreme_perspective.jpg"), edge1, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 3. Edge Case 2: Heavy Shadow
    print("3. Generating Edge Case 2: Heavy Shadow...")
    bg2 = create_wood_texture(cw, ch, base_color=(60, 85, 120))
    dst_quad_shadow = np.array([
        [380, 160],
        [1540, 140],
        [1620, 1150],
        [320, 1170]
    ], dtype=np.float32)
    edge2 = warp_into_canvas(doc, cw, ch, src_corners, dst_quad_shadow, bg2)

    shadow_overlay = np.zeros((ch, cw), dtype=np.float32)
    pts_shadow = np.array([[600, 0], [1920, 0], [1920, 1100], [900, 1280], [300, 600]], dtype=np.int32)
    cv2.fillPoly(shadow_overlay, [pts_shadow], 1.0)
    shadow_overlay = cv2.GaussianBlur(shadow_overlay, (151, 151), 0)
    shadow_multiplier = 1.0 - (shadow_overlay * 0.65)
    for c in range(3):
        edge2[:, :, c] = np.clip(edge2[:, :, c].astype(np.float32) * shadow_multiplier, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "edge2_heavy_shadow.jpg"), edge2, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 4. Edge Case 3: Cluttered Background
    print("4. Generating Edge Case 3: Cluttered Background...")
    bg3 = create_wood_texture(cw, ch, base_color=(55, 75, 105))
    cv2.rectangle(bg3, (120, 100), (320, 300), (80, 210, 240), -1)
    cv2.rectangle(bg3, (120, 100), (320, 300), (50, 160, 190), 2)
    cv2.rectangle(bg3, (1600, 150), (1820, 370), (180, 130, 245), -1)
    cv2.rectangle(bg3, (1500, 800), (1920, 1280), (120, 60, 50), -1)
    cv2.line(bg3, (150, 750), (280, 1080), (30, 30, 30), 18)
    cv2.line(bg3, (150, 750), (165, 785), (200, 200, 200), 16)

    dst_quad_clutter = np.array([
        [420, 200],
        [1480, 170],
        [1560, 1120],
        [350, 1140]
    ], dtype=np.float32)
    edge3 = warp_into_canvas(doc, cw, ch, src_corners, dst_quad_clutter, bg3)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "edge3_cluttered_background.jpg"), edge3, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 5. Edge Case 4: Corner Occluded
    print("5. Generating Edge Case 4: Corner Occluded...")
    bg4 = create_wood_texture(cw, ch, base_color=(50, 70, 100))
    dst_quad_occl = np.array([
        [360, 160],
        [1520, 140],
        [1600, 1160],
        [280, 1170]
    ], dtype=np.float32)
    edge4 = warp_into_canvas(doc, cw, ch, src_corners, dst_quad_occl, bg4)
    cv2.ellipse(edge4, (1580, 1150), (110, 80), 35, 0, 360, (140, 165, 215), -1)
    cv2.ellipse(edge4, (1580, 1150), (110, 80), 35, 0, 360, (100, 120, 170), 3)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "edge4_corner_occluded.jpg"), edge4, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 6. Edge Case 5: Low Contrast
    print("6. Generating Edge Case 5: Low Contrast...")
    bg5 = np.zeros((ch, cw, 3), dtype=np.uint8)
    bg5[:, :] = (238, 238, 236)
    noise_pale = np.random.normal(0, 3, (ch, cw)).astype(np.float32)
    for c in range(3):
        bg5[:, :, c] = np.clip(bg5[:, :, c].astype(np.float32) + noise_pale, 0, 255).astype(np.uint8)

    dst_quad_low_c = np.array([
        [410, 170],
        [1500, 150],
        [1580, 1130],
        [330, 1140]
    ], dtype=np.float32)
    edge5 = warp_into_canvas(doc, cw, ch, src_corners, dst_quad_low_c, bg5)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "edge5_low_contrast.jpg"), edge5, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print("All sample images generated successfully in:", OUTPUT_DIR)


if __name__ == "__main__":
    generate_all_samples()
