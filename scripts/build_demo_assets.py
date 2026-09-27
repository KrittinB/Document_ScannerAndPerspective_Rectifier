"""
build_demo_assets.py
สคริปต์สร้างและรวบรวมชุดภาพเอกสารสำหรับ Demo & Presentation ของโปรเจกต์
CP461 Document Scanner & Perspective Rectifier
จัดเก็บไว้ในโฟลเดอร์ demo/ พร้อมจัดหมวดหมู่อย่างเป็นระเบียบ
"""

import os
import sys
import shutil
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Root paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.preprocessing import preprocess
from src.detection import detect_document
from src.features import extract_and_match
from src.geometry import rectify_from_corners, rectify_from_reference

DEMO_DIR = os.path.join(ROOT_DIR, "demo")
STD_DIR = os.path.join(DEMO_DIR, "01_Standard_Documents")
REF_DIR = os.path.join(DEMO_DIR, "02_Reference_Pairs")
EDGE_DIR = os.path.join(DEMO_DIR, "03_Edge_Cases")

for d in [DEMO_DIR, STD_DIR, REF_DIR, EDGE_DIR]:
    os.makedirs(d, exist_ok=True)

BRAIN_DIR = r"C:\Users\narat\.gemini\antigravity-ide\brain\0597b174-90ca-4d0d-b731-2466266091ed"

# Wood texture generator
def create_wood_texture(w: int, h: int, base_color=(50, 75, 115)) -> np.ndarray:
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :] = base_color
    noise = np.random.normal(0, 8, (h, w)).astype(np.float32)
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
    H = cv2.getPerspectiveTransform(src_quad.astype(np.float32), dst_quad.astype(np.float32))
    warped_doc = cv2.warpPerspective(doc_bgr, H, (canvas_w, canvas_h), flags=cv2.INTER_LANCZOS4)
    doc_mask = np.ones((doc_bgr.shape[0], doc_bgr.shape[1]), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(doc_mask, H, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST)

    shadow_mask = cv2.dilate(warped_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    shadow_mask = cv2.GaussianBlur(shadow_mask, (45, 45), 0)

    result = bg_bgr.copy().astype(np.float32)
    shadow_alpha = (shadow_mask.astype(np.float32) / 255.0)[:, :, np.newaxis] * 0.45
    result = result * (1.0 - shadow_alpha) + np.array([20, 20, 20], dtype=np.float32) * shadow_alpha

    paper_alpha = (warped_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    result = result * (1.0 - paper_alpha) + warped_doc.astype(np.float32) * paper_alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def create_thai_tax_invoice() -> np.ndarray:
    """สร้างใบเสร็จรับเงิน / ใบกำกับภาษีไทยแบบ A4 คมชัดสูง"""
    w, h = 1200, 1697
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    f_title = ImageFont.truetype("C:\\Windows\\Fonts\\tahomabd.ttf", 36)
    f_header = ImageFont.truetype("C:\\Windows\\Fonts\\tahomabd.ttf", 23)
    f_sub = ImageFont.truetype("C:\\Windows\\Fonts\\tahoma.ttf", 19)
    f_body = ImageFont.truetype("C:\\Windows\\Fonts\\tahoma.ttf", 18)
    f_small = ImageFont.truetype("C:\\Windows\\Fonts\\tahoma.ttf", 15)

    # Top banner & Company details
    draw.rectangle([(50, 50), (w - 50, 56)], fill=(20, 80, 160))
    draw.text((50, 75), "บริษัท สยาม วิชั่น อินโนเวชั่น จำกัด (มหาชน)", fill=(20, 40, 80), font=f_header)
    draw.text((50, 110), "99/1 อาคารไอทีทาวเวอร์ ชั้น 18 ถนนพหลโยธิน แขวงลาดยาว เขตจตุจักร กรุงเทพฯ 10900", fill=(70, 70, 70), font=f_sub)
    draw.text((50, 140), "เลขประจำตัวผู้เสียภาษีอากร: 0-1055-63098-76-5 | สำนักงานใหญ่ โทร: 02-555-8900", fill=(80, 80, 80), font=f_small)

    # Title box
    draw.rectangle([(w - 440, 70), (w - 50, 175)], outline=(20, 80, 160), width=2, fill=(240, 245, 255))
    draw.text((w - 420, 80), "ใบเสร็จรับเงิน / ใบกำกับภาษี", fill=(20, 50, 120), font=f_title)
    draw.text((w - 400, 130), "TAX INVOICE / RECEIPT", fill=(70, 80, 110), font=f_sub)

    draw.line([(50, 185), (w - 50, 185)], fill=(180, 190, 210), width=2)

    # Customer and document info
    draw.text((50, 205), "ข้อมูลลูกค้า (Customer Information):", fill=(20, 80, 160), font=f_header)
    draw.text((50, 240), "ชื่อ: ภาควิชาวิทยาการคอมพิวเตอร์ คณะวิทยาศาสตร์", fill=(30, 30, 30), font=f_body)
    draw.text((50, 270), "ที่อยู่: มหาวิทยาลัยเชียงใหม่ 239 ถนนห้วยแก้ว ต.สุเทพ อ.เมือง จ.เชียงใหม่ 50200", fill=(50, 50, 50), font=f_body)
    draw.text((50, 300), "เลขประจำตัวผู้เสียภาษี: 0-9940-00164-90-2", fill=(50, 50, 50), font=f_body)

    draw.text((w - 450, 205), "รายละเอียดเอกสาร (Invoice Details):", fill=(20, 80, 160), font=f_header)
    draw.text((w - 450, 240), "เลขที่เอกสาร: INV-2026/0927-CP461", fill=(30, 30, 30), font=f_body)
    draw.text((w - 450, 270), "วันที่: 27 กันยายน 2569", fill=(50, 50, 50), font=f_body)
    draw.text((w - 450, 300), "วิธีการชำระ: เงินสด / สแกน QR Code", fill=(50, 50, 50), font=f_body)

    # Item Table
    ty = 350
    draw.rectangle([(50, ty), (w - 50, ty + 50)], fill=(225, 235, 250), outline=(150, 170, 210), width=2)
    cols = [50, 130, 680, 780, 930, w - 50]
    headers = ["ลำดับ", "รายการสินค้า / บริการ", "จำนวน", "ราคา/หน่วย", "จำนวนเงิน (บาท)"]
    for i, h_text in enumerate(headers):
        draw.text((cols[i] + 12, ty + 12), h_text, fill=(20, 40, 90), font=f_body)

    rows = [
        ["1", "ระบบประมวลผล Computer Vision Document Scanner", "1 ชุด", "35,000.00", "35,000.00"],
        ["2", "โมดูล Perspective Rectification & Homography Matrix", "1 ระบบ", "22,000.00", "22,000.00"],
        ["3", "การจัดทำ Feature Matching Visualization (SIFT/ORB)", "1 งาน", "12,000.00", "12,000.00"],
        ["4", "ชุดทดสอบความทนทานต่อกรณีขอบเขต (Robustness QA)", "1 ชุด", "8,000.00", "8,000.00"],
        ["5", "บริการ Cloud Deployment และคู่มือการใช้งานระบบ", "1 สัญญา", "5,000.00", "5,000.00"],
    ]

    for r_idx, row in enumerate(rows):
        row_y = ty + 50 + r_idx * 45
        bg_col = (250, 252, 255) if r_idx % 2 == 1 else (255, 255, 255)
        draw.rectangle([(50, row_y), (w - 50, row_y + 45)], fill=bg_col, outline=(220, 225, 235), width=1)
        for c_idx, cell in enumerate(row):
            draw.text((cols[c_idx] + 12, row_y + 10), cell, fill=(40, 40, 40), font=f_body)

    for c in cols[1:-1]:
        draw.line([(c, ty), (c, ty + 50 + len(rows) * 45)], fill=(180, 195, 220), width=1)
    draw.rectangle([(50, ty), (w - 50, ty + 50 + len(rows) * 45)], outline=(150, 170, 210), width=2)

    # Summary box
    sy = ty + 50 + len(rows) * 45 + 30
    draw.rectangle([(w - 480, sy), (w - 50, sy + 160)], fill=(245, 248, 255), outline=(150, 170, 210), width=2)
    draw.text((w - 460, sy + 15), "รวมเป็นเงิน (Subtotal):", fill=(50, 50, 50), font=f_body)
    draw.text((w - 180, sy + 15), "82,000.00", fill=(30, 30, 30), font=f_header)
    draw.text((w - 460, sy + 55), "ภาษีมูลค่าเพิ่ม (VAT 7%):", fill=(50, 50, 50), font=f_body)
    draw.text((w - 180, sy + 55), "5,740.00", fill=(30, 30, 30), font=f_header)
    draw.line([(w - 470, sy + 100), (w - 60, sy + 100)], fill=(180, 190, 210), width=1)
    draw.text((w - 460, sy + 115), "ยอดเงินสุทธิ (TOTAL):", fill=(20, 80, 160), font=f_header)
    draw.text((w - 180, sy + 115), "87,740.00", fill=(180, 30, 30), font=f_header)

    # Baht Text Box
    draw.rectangle([(50, sy), (w - 510, sy + 60)], fill=(240, 245, 252), outline=(180, 195, 220), width=1)
    draw.text((65, sy + 18), "จำนวนเงินตัวอักษร: (แปดหมื่นเจ็ดพันเจ็ดร้อยสี่สิบบาทถ้วน)", fill=(30, 60, 120), font=f_body)

    # Signatures & Official Stamp
    sig_y = sy + 220
    draw.line([(100, sig_y + 80), (400, sig_y + 80)], fill=(150, 150, 150), width=1)
    draw.text((160, sig_y + 90), "ลงชื่อ ผู้มีอำนาจลงนาม", fill=(60, 60, 60), font=f_body)
    draw.text((170, sig_y + 120), "นายสมศักดิ์ ธรรมรัตน์", fill=(100, 100, 100), font=f_small)

    draw.line([(w - 450, sig_y + 80), (w - 150, sig_y + 80)], fill=(150, 150, 150), width=1)
    draw.text((w - 380, sig_y + 90), "ลงชื่อ ผู้รับเงิน", fill=(60, 60, 60), font=f_body)
    draw.text((w - 390, sig_y + 120), "นางสาวนภาพร สดใส", fill=(100, 100, 100), font=f_small)

    # Red Stamp
    draw.ellipse([(220, sig_y + 10), (330, sig_y + 120)], outline=(190, 40, 40), width=3)
    draw.text((238, sig_y + 45), "SIAM VISION\n  OFFICIAL", fill=(190, 40, 40), font=f_small)

    draw.text((50, h - 60), "เอกสารนี้ออกโดยระบบอัตโนมัติ | CP461 Computer Vision Project Demonstration Suite", fill=(120, 120, 130), font=f_small)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    print("==================================================")
    print("Building and Verifying Demo Suite Assets...")
    print("==================================================")

    # 1. Standard Documents
    # Aurora Tax Invoice photo
    aurora_src = os.path.join(BRAIN_DIR, "demo_thai_receipt_1790487673584.jpg")
    aurora_dst = os.path.join(STD_DIR, "01_aurora_tax_invoice.jpg")
    shutil.copy2(aurora_src, aurora_dst)
    shutil.copy2(aurora_src, os.path.join(DEMO_DIR, "01_standard_aurora_tax_invoice.jpg"))

    # Apex Solutions Consulting Invoice photo
    apex_src = os.path.join(BRAIN_DIR, "demo_invoice_photo_1790487302684.jpg")
    apex_dst = os.path.join(STD_DIR, "02_apex_consulting_invoice.jpg")
    shutil.copy2(apex_src, apex_dst)
    shutil.copy2(apex_src, os.path.join(DEMO_DIR, "02_standard_apex_invoice.jpg"))

    # Business Contract photo
    contract_src = os.path.join(BRAIN_DIR, "demo_contract_desk_1790487364607.jpg")
    contract_dst = os.path.join(STD_DIR, "03_business_contract.jpg")
    shutil.copy2(contract_src, contract_dst)
    shutil.copy2(contract_src, os.path.join(DEMO_DIR, "03_standard_business_contract.jpg"))

    # Academic Research Paper photo
    research_src = os.path.join(BRAIN_DIR, "demo_research_paper_1790487388671.jpg")
    research_dst = os.path.join(STD_DIR, "04_academic_research_paper.jpg")
    shutil.copy2(research_src, research_dst)
    shutil.copy2(research_src, os.path.join(DEMO_DIR, "04_standard_research_paper.jpg"))

    # Thai University Certificate photo
    cert_src = os.path.join(BRAIN_DIR, "demo_thai_doc_1790487568114.jpg")
    cert_dst = os.path.join(STD_DIR, "05_thai_cmu_certificate.jpg")
    shutil.copy2(cert_src, cert_dst)
    shutil.copy2(cert_src, os.path.join(DEMO_DIR, "05_standard_thai_certificate.jpg"))

    # Synthetic Thai Tax Invoice & Skewed photo on wood
    print("Generating Synthetic Thai Tax Invoice...")
    thai_inv = create_thai_tax_invoice()
    thai_flat_path = os.path.join(REF_DIR, "pair1_thai_tax_invoice_flat.jpg")
    cv2.imwrite(thai_flat_path, thai_inv, [cv2.IMWRITE_JPEG_QUALITY, 95])
    shutil.copy2(thai_flat_path, os.path.join(DEMO_DIR, "ref1_thai_invoice_flat.jpg"))

    cw, ch = 1440, 1920
    wood_bg = create_wood_texture(cw, ch, base_color=(50, 70, 110))
    dh, dw = thai_inv.shape[:2]
    src_corners = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float32)
    dst_quad = np.array([
        [240, 220],
        [1180, 180],
        [1280, 1720],
        [160, 1680]
    ], dtype=np.float32)
    thai_skewed = warp_into_canvas(thai_inv, cw, ch, src_corners, dst_quad, wood_bg)
    thai_skewed_path = os.path.join(REF_DIR, "pair1_thai_tax_invoice_skewed.jpg")
    cv2.imwrite(thai_skewed_path, thai_skewed, [cv2.IMWRITE_JPEG_QUALITY, 92])
    shutil.copy2(thai_skewed_path, os.path.join(DEMO_DIR, "ref1_thai_invoice_skewed.jpg"))
    shutil.copy2(thai_skewed_path, os.path.join(STD_DIR, "06_thai_tax_invoice_photo.jpg"))
    shutil.copy2(thai_skewed_path, os.path.join(DEMO_DIR, "06_standard_thai_tax_invoice.jpg"))

    # 2. Reference Pairs
    # Pair 2: Aurora Tax Invoice
    aurora_img = cv2.imread(aurora_src)
    aurora_prep = preprocess(aurora_img)
    aurora_det = detect_document(aurora_prep["blurred"], aurora_prep["blurred"].shape)
    aurora_rect = rectify_from_corners(aurora_img, aurora_det["corners"], aurora_prep["scale"], 800)
    aurora_flat_path = os.path.join(REF_DIR, "pair2_aurora_invoice_flat.jpg")
    cv2.imwrite(aurora_flat_path, aurora_rect["warped"], [cv2.IMWRITE_JPEG_QUALITY, 95])
    shutil.copy2(aurora_flat_path, os.path.join(DEMO_DIR, "ref2_aurora_flat.jpg"))
    shutil.copy2(aurora_src, os.path.join(REF_DIR, "pair2_aurora_invoice_skewed.jpg"))
    shutil.copy2(aurora_src, os.path.join(DEMO_DIR, "ref2_aurora_skewed.jpg"))

    # Pair 3: Apex Solutions Invoice
    apex_img = cv2.imread(apex_src)
    apex_prep = preprocess(apex_img)
    apex_det = detect_document(apex_prep["blurred"], apex_prep["blurred"].shape)
    apex_rect = rectify_from_corners(apex_img, apex_det["corners"], apex_prep["scale"], 800)
    apex_flat_path = os.path.join(REF_DIR, "pair3_apex_invoice_flat.jpg")
    cv2.imwrite(apex_flat_path, apex_rect["warped"], [cv2.IMWRITE_JPEG_QUALITY, 95])
    shutil.copy2(apex_flat_path, os.path.join(DEMO_DIR, "ref3_apex_flat.jpg"))
    shutil.copy2(apex_src, os.path.join(REF_DIR, "pair3_apex_invoice_skewed.jpg"))
    shutil.copy2(apex_src, os.path.join(DEMO_DIR, "ref3_apex_skewed.jpg"))

    # Pair 4: CP461 Specification Report from tests/sample_images
    ref_spec_flat = os.path.join(ROOT_DIR, "tests", "sample_images", "ref1_flat_reference.jpg")
    ref_spec_skewed = os.path.join(ROOT_DIR, "tests", "sample_images", "ref1_skewed_photo.jpg")
    if os.path.exists(ref_spec_flat) and os.path.exists(ref_spec_skewed):
        shutil.copy2(ref_spec_flat, os.path.join(REF_DIR, "pair4_cp461_spec_report_flat.jpg"))
        shutil.copy2(ref_spec_skewed, os.path.join(REF_DIR, "pair4_cp461_spec_report_skewed.jpg"))
        shutil.copy2(ref_spec_flat, os.path.join(DEMO_DIR, "ref4_cp461_spec_flat.jpg"))
        shutil.copy2(ref_spec_skewed, os.path.join(DEMO_DIR, "ref4_cp461_spec_skewed.jpg"))

    # 3. Edge Cases
    # Edge 1: Heavy Shadow
    shadow_src = os.path.join(BRAIN_DIR, "demo_heavy_shadow_1790487470980.jpg")
    shutil.copy2(shadow_src, os.path.join(EDGE_DIR, "edge1_heavy_shadow.jpg"))
    shutil.copy2(shadow_src, os.path.join(DEMO_DIR, "edge1_heavy_shadow.jpg"))

    # Edge 2: Cluttered Desk
    clutter_src = os.path.join(BRAIN_DIR, "demo_cluttered_desk_1790487449308.jpg")
    shutil.copy2(clutter_src, os.path.join(EDGE_DIR, "edge2_cluttered_desk.jpg"))
    shutil.copy2(clutter_src, os.path.join(DEMO_DIR, "edge2_cluttered_desk.jpg"))

    # Edge 3: Corner Occluded
    occl_src = os.path.join(BRAIN_DIR, "demo_corner_occluded_1790487508910.jpg")
    shutil.copy2(occl_src, os.path.join(EDGE_DIR, "edge3_corner_occluded.jpg"))
    shutil.copy2(occl_src, os.path.join(DEMO_DIR, "edge3_corner_occluded.jpg"))

    # Edge 4: Low Contrast
    lowc_src = os.path.join(BRAIN_DIR, "demo_low_contrast_1790487533221.jpg")
    shutil.copy2(lowc_src, os.path.join(EDGE_DIR, "edge4_low_contrast.jpg"))
    shutil.copy2(lowc_src, os.path.join(DEMO_DIR, "edge4_low_contrast.jpg"))

    # Edge 5: Extreme Perspective (from sample_images)
    ext_src = os.path.join(ROOT_DIR, "tests", "sample_images", "edge1_extreme_perspective.jpg")
    if os.path.exists(ext_src):
        shutil.copy2(ext_src, os.path.join(EDGE_DIR, "edge5_extreme_perspective.jpg"))
        shutil.copy2(ext_src, os.path.join(DEMO_DIR, "edge5_extreme_perspective.jpg"))

    print("All demo assets successfully assembled!")


if __name__ == "__main__":
    main()
