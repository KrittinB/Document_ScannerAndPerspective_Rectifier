"""
app.py — Smart Document Scanner
Streamlit Web App สำหรับ CP461 Group Project

Pipeline:
  Upload → Preprocess → Detect → Features → Homography → A4 Output
"""

import io
import streamlit as st
import numpy as np
import cv2
from PIL import Image

from src.preprocessing import preprocess
from src.detection import detect_document
from src.features import extract_and_match
from src.geometry import full_pipeline, get_a4_dimensions
from src.utils import (
    pil_to_numpy_bgr,
    numpy_bgr_to_pil,
    numpy_gray_to_pil,
    bytes_to_numpy_bgr,
    draw_corners,
    draw_matches_visualization,
    draw_inlier_outlier,
    numpy_to_bytes_png,
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Document Scanner | CP461",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS Styling
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark header gradient */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .main-header p {
        font-size: 1rem;
        color: rgba(255,255,255,0.75);
        margin: 0.4rem 0 0 0;
    }

    /* Stepper */
    .stepper-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0;
        margin: 1.2rem 0 1.8rem 0;
        flex-wrap: wrap;
    }
    .step {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 18px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #6b7280;
        background: #f3f4f6;
        border: 2px solid transparent;
        transition: all 0.3s;
    }
    .step.active {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-color: transparent;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .step.done {
        background: #d1fae5;
        color: #065f46;
        border-color: #6ee7b7;
    }
    .step-sep {
        width: 32px;
        height: 2px;
        background: #e5e7eb;
        margin: 0 4px;
    }
    .step-sep.done { background: #6ee7b7; }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4f46e5;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 2px;
    }

    /* Image captions */
    .img-caption {
        text-align: center;
        font-size: 0.82rem;
        color: #6b7280;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Section header */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f2937;
        border-left: 4px solid #667eea;
        padding-left: 10px;
        margin: 1.5rem 0 0.8rem 0;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green { background: #d1fae5; color: #065f46; }
    .badge-blue  { background: #dbeafe; color: #1e40af; }
    .badge-purple{ background: #ede9fe; color: #5b21b6; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #c4b5fd;
        border-radius: 12px;
        padding: 1rem;
        background: #faf5ff;
    }

    /* Run button */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        box-shadow: 0 4px 15px rgba(102,126,234,0.35);
        transition: transform 0.15s;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "uploaded_img": None,
        "result": None,
        "step": 0,  # 0=upload, 1=detect, 2=feature, 3=done
        "uploader_key": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
# Stepper HTML
# ─────────────────────────────────────────────
STEPS = ["Upload", "Detect", "Match", "Result"]

def render_stepper(current: int):
    html = '<div class="stepper-container">'
    for i, label in enumerate(STEPS):
        if i < current:
            cls = "step done"
        elif i == current:
            cls = "step active"
        else:
            cls = "step"
        html += f'<div class="{cls}">{label}</div>'
        if i < len(STEPS) - 1:
            sep_cls = "step-sep done" if i < current else "step-sep"
            html += f'<div class="{sep_cls}"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>Document Scanner & Perspective Rectifier</h1>
        <p>Perspective Rectification using SIFT/ORB · Feature Matching · Homography · RANSAC</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Sidebar: Project Info
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About Project")
    st.markdown(
        """
        **CP461 Computer Vision Project**  
        Document Scanner & Perspective Rectifier

        **Pipeline Architecture:**
        1. Preprocessing (Resize, Grayscale, Gaussian Blur)
        2. Edge Detection (Canny)
        3. Contour & 4-Corner Localization
        4. Feature Extraction (SIFT / ORB)
        5. Feature Matching (BFMatcher / FLANN)
        6. Lowe's Ratio Test Filtering
        7. Robust Estimation (RANSAC + Homography)
        8. Perspective Warping to Standard A4
        """
    )

render_stepper(st.session_state.step)

# ─────────────────────────────────────────────
# Section 1: Upload & Settings
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Section 1: Upload & Settings</div>', unsafe_allow_html=True)

col_upload, col_settings = st.columns([1.1, 0.9], gap="large")

with col_upload:
    st.markdown('<div class="subsection-title">Upload Document Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "วาง หรือ คลิกเพื่อเลือกภาพถ่ายเอกสาร (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"file_upload_{st.session_state.uploader_key}",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        img_bgr = bytes_to_numpy_bgr(file_bytes)
        if img_bgr is None:
            st.error("ไม่สามารถอ่านไฟล์ภาพได้ กรุณาลองใหม่อีกครั้ง")
            st.stop()

        st.session_state.uploaded_img = img_bgr
        if st.session_state.step == 0:
            st.session_state.step = 1

    # Action Buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1.8, 1.5, 1.2])

    with col_btn1:
        run_btn = st.button(
            "Run Scan Pipeline",
            type="primary",
            disabled=(st.session_state.uploaded_img is None),
            use_container_width=True,
        )

    with col_btn2:
        download_ready = (
            st.session_state.result is not None
            and st.session_state.result.get("success")
            and st.session_state.result.get("warped") is not None
        )
        if download_ready:
            png_bytes = numpy_to_bytes_png(st.session_state.result["warped"])
            st.download_button(
                "Download A4",
                data=png_bytes,
                file_name="scanned_document.png",
                mime="image/png",
                use_container_width=True,
            )
        else:
            st.button("Download A4", disabled=True, use_container_width=True)

    with col_btn3:
        if st.button("Reset", use_container_width=True):
            st.session_state.uploaded_img = None
            st.session_state.result = None
            st.session_state.step = 0
            st.session_state.uploader_key += 1
            st.rerun()

with col_settings:
    st.markdown('<div class="subsection-title">Pipeline Settings</div>', unsafe_allow_html=True)
    with st.container(border=True):
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            feature_method = st.selectbox(
                "Feature Extractor",
                ["SIFT", "ORB"],
                help="SIFT ให้ผลดีและทนทานกว่า ORB เร็วกว่าแต่เหมาะกับภาพทั่วไป",
            )
        with s_col2:
            matcher_method = st.selectbox(
                "Feature Matcher",
                ["BFMatcher", "FLANN"],
                help="BFMatcher แม่นยำ, FLANN เร็วกว่าสำหรับ dataset ขนาดใหญ่",
            )

        ratio_threshold = st.slider(
            "Lowe's Ratio Threshold", 0.50, 0.95, 0.75, 0.05,
            help="ค่ามาตรฐาน 0.70 - 0.75 สำหรับกรอง ambiguous matches",
        )
        a4_height = st.slider(
            "Output Resolution (height px)", 600, 1600, 800, 100,
            help="ความสูงของภาพ A4 ผลลัพธ์ (ความกว้างจะปรับตามสัดส่วน A4)",
        )


# ─────────────────────────────────────────────
# Pipeline Execution Logic
# ─────────────────────────────────────────────
if run_btn and st.session_state.uploaded_img is not None:
    img = st.session_state.uploaded_img

    with st.spinner("กำลังประมวลผล pipeline..."):
        try:
            # Step 1: Preprocess
            st.session_state.step = 1
            prep = preprocess(img)
            img_resized = prep["resized"]
            img_gray = prep["gray"]
            img_blur = prep["blurred"]

            # Step 2: Detect document corners
            st.session_state.step = 2
            det = detect_document(img_blur, img_resized.shape[:2])
            corners = det["corners"]
            edges = det["edges"]

            if not det["success"] or corners is None:
                st.session_state.result = {
                    "success": False,
                    "message": det["message"],
                    "img_resized": img_resized,
                    "edges": edges,
                }
                st.session_state.step = 1
            else:
                # Step 3: Feature extraction + matching
                from src.geometry import corners_to_homography, warp_perspective as warp_tmp

                H_tmp, _, sz_tmp = corners_to_homography(corners, a4_height)
                warped_tmp = warp_tmp(img_resized, H_tmp, sz_tmp)
                gray_warped = cv2.cvtColor(warped_tmp, cv2.COLOR_BGR2GRAY)

                feat = extract_and_match(
                    img_gray,
                    gray_warped,
                    method=feature_method,
                    matcher=matcher_method,
                    ratio=ratio_threshold,
                )

                # Step 4: Geometry / RANSAC / Homography
                st.session_state.step = 3
                geo = full_pipeline(
                    img_resized,
                    corners,
                    feature_result=feat,
                    a4_base=a4_height,
                )

                st.session_state.result = {
                    "success": geo["success"],
                    "message": geo["message"],
                    "img_resized": img_resized,
                    "edges": edges,
                    "corners": corners,
                    "warped": geo["warped"],
                    "H": geo["H"],
                    "mask": geo["mask"],
                    "n_inliers": geo["n_inliers"],
                    "used_ransac": geo["used_ransac"],
                    "feat": feat,
                    "gray_warped": gray_warped,
                    "warped_tmp": warped_tmp,
                }
                st.session_state.step = 3 if geo["success"] else 2

        except Exception as e:
            st.session_state.result = {
                "success": False,
                "message": f"เกิดข้อผิดพลาด: {str(e)}",
            }

    st.rerun()


# ─────────────────────────────────────────────
# Section 2: Main Results
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Section 2: Main Results</div>', unsafe_allow_html=True)

result = st.session_state.result

if result is None:
    if st.session_state.uploaded_img is not None:
        st.info("ภาพเอกสารถูกอัปโหลดแล้ว — กรุณากดปุ่ม 'Run Scan Pipeline' ใน Section 1 เพื่อเริ่มประมวลผล")
        preview_pil = numpy_bgr_to_pil(st.session_state.uploaded_img)
        col_prev, _ = st.columns([1, 2])
        with col_prev:
            st.image(preview_pil, caption="ภาพต้นฉบับ (Preview ก่อนประมวลผล)", use_container_width=True)
    else:
        st.info("กรุณาอัปโหลดภาพถ่ายเอกสารใน Section 1 ด้านบนเพื่อเริ่มต้นใช้งาน")

elif result is not None:
    if not result["success"]:
        st.error(f"{result['message']}")
        if result.get("img_resized") is not None:
            st.image(
                numpy_bgr_to_pil(result["img_resized"]),
                caption="ภาพต้นฉบับ",
                use_container_width=True,
            )
        if result.get("edges") is not None:
            st.image(
                numpy_gray_to_pil(result["edges"]),
                caption="Edge Map",
                use_container_width=True,
            )
    else:
        # Status Badges
        ransac_badge = (
            '<span class="badge badge-green">RANSAC from features</span>'
            if result.get("used_ransac")
            else '<span class="badge badge-blue">Corner-based homography</span>'
        )
        method_badge = f'<span class="badge badge-purple">{feature_method} + {matcher_method}</span>'
        st.markdown(f"{ransac_badge} &nbsp; {method_badge}", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)

        # 3-Column Main Results: Original | 4 Corners Detection | Final Result A4 Corrected
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            orig_pil = numpy_bgr_to_pil(result["img_resized"])
            st.image(orig_pil, caption="Original Image", use_container_width=True)
            st.markdown('<p class="img-caption">ภาพถ่ายต้นฉบับ (Resized)</p>', unsafe_allow_html=True)

        with res_col2:
            corners_img = draw_corners(result["img_resized"], result["corners"])
            st.image(numpy_bgr_to_pil(corners_img), caption="4 Corners Detection", use_container_width=True)
            st.markdown('<p class="img-caption">ตรวจจับกรอบและ 4 มุมเอกสาร (TL/TR/BR/BL)</p>', unsafe_allow_html=True)

        with res_col3:
            warped_pil = numpy_bgr_to_pil(result["warped"])
            st.image(warped_pil, caption="Final Result (A4 Corrected)", use_container_width=True)
            w, h = result["warped"].shape[1], result["warped"].shape[0]
            st.markdown(f'<p class="img-caption">ผลลัพธ์ A4 มองตรง ({w}×{h} px)</p>', unsafe_allow_html=True)

        # ── Pipeline Metrics & Confidence Row (ใต้ภาพผลลัพธ์) ──
        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
        feat = result.get("feat", {})
        n_kp1 = feat.get("n_keypoints_1", 0)
        n_kp2 = feat.get("n_keypoints_2", 0)
        n_good = feat.get("n_good", 0)
        n_inliers = result.get("n_inliers", 0)
        confidence = min(100, int(n_inliers / max(n_good, 1) * 100)) if n_good > 0 else 0

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            st.markdown(
                f'<div class="metric-card"><div class="value">{n_kp1}</div>'
                f'<div class="label">Keypoints (src)</div></div>',
                unsafe_allow_html=True,
            )
        with mc2:
            st.markdown(
                f'<div class="metric-card"><div class="value">{n_kp2}</div>'
                f'<div class="label">Keypoints (dst)</div></div>',
                unsafe_allow_html=True,
            )
        with mc3:
            st.markdown(
                f'<div class="metric-card"><div class="value">{n_good}</div>'
                f'<div class="label">Good Matches</div></div>',
                unsafe_allow_html=True,
            )
        with mc4:
            st.markdown(
                f'<div class="metric-card"><div class="value">{n_inliers}</div>'
                f'<div class="label">RANSAC Inliers</div></div>',
                unsafe_allow_html=True,
            )
        with mc5:
            st.markdown(
                f'<div class="metric-card"><div class="value">{confidence}%</div>'
                f'<div class="label">Confidence</div></div>',
                unsafe_allow_html=True,
            )

        st.progress(confidence / 100, text=f"Pipeline Confidence: {confidence}%")


# ─────────────────────────────────────────────
# Section 3: Technical Pipeline Details (Collapsible)
# ─────────────────────────────────────────────
if result is not None and result.get("success"):
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("Technical Details", expanded=False):
        # 2-Column Technical Layout
        tech_col1, tech_col2 = st.columns(2, gap="medium")

        with tech_col1:
            # [Preprocessing] Canny Edge Map
            with st.container(border=True):
                st.markdown("#### [Preprocessing] Canny Edge Map")
                edges_pil = numpy_gray_to_pil(result["edges"])
                st.image(edges_pil, caption="Canny Edge Map", use_container_width=True)
                st.caption("การตรวจจับขอบเขตเอกสารด้วย Canny Edge Detection หลังผ่าน Gaussian Blur 5×5 เพื่อลดสัญญาณรบกวน (noise)")

            # [Feature Matching] Matches After Lowe's Ratio Test
            with st.container(border=True):
                st.markdown("#### [Feature Matching] Matches After Lowe's Ratio Test")
                good_matches = feat.get("good_matches", [])
                kp1 = feat.get("kp1", [])
                kp2 = feat.get("kp2", [])

                if len(good_matches) >= 1 and len(kp1) > 0 and len(kp2) > 0:
                    match_vis = draw_matches_visualization(
                        result["img_resized"], kp1,
                        result["warped_tmp"], kp2,
                        good_matches,
                    )
                    st.image(numpy_bgr_to_pil(match_vis), caption="Feature Matching Visualization", use_container_width=True)
                else:
                    st.image(numpy_bgr_to_pil(result["img_resized"]), caption="Feature Matching (no matches)", use_container_width=True)
                st.caption(f"จับคู่จุดเด่นด้วย {matcher_method} และกรองจุดกำกวมด้วย Lowe's Ratio Test (Threshold = {ratio_threshold:.2f}) ได้ {len(good_matches)} good matches")

        with tech_col2:
            # [RANSAC] Inliers / Outliers Visualization
            with st.container(border=True):
                st.markdown("#### [RANSAC] Inliers / Outliers Visualization")
                inlier_img = draw_inlier_outlier(
                    result["img_resized"],
                    kp1,
                    good_matches,
                    result.get("mask"),
                )
                st.image(numpy_bgr_to_pil(inlier_img), caption="Inlier (Green) vs Outlier (Red)", use_container_width=True)
                st.caption("RANSAC ทำหน้าที่กำจัดจุดคู่ที่สอดคล้องผิดพลาด (Outliers สีแดง) และคงไว้เฉพาะจุดคู่ที่สอดคล้องกับระนาบจริง (Inliers สีเขียว)")

            # [Homography] Transformation Information
            with st.container(border=True):
                st.markdown("#### [Homography] Transformation Information")
                H = result.get("H")
                if H is not None:
                    st.markdown("**Homography Matrix H (3×3):**")
                    st.latex(r"""
                    H = \begin{bmatrix}
                    %.4f & %.4f & %.4f \\
                    %.4f & %.4f & %.4f \\
                    %.4f & %.4f & %.4f
                    \end{bmatrix}
                    """ % (H[0,0], H[0,1], H[0,2], H[1,0], H[1,1], H[1,2], H[2,0], H[2,1], H[2,2]))

                corners = result.get("corners")
                if corners is not None and len(corners) == 4:
                    st.markdown("**Detected Corner Coordinates (Source):**")
                    c_data = {
                        "Corner": ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Right (BR)", "Bottom-Left (BL)"],
                        "X (px)": [f"{corners[0][0]:.1f}", f"{corners[1][0]:.1f}", f"{corners[2][0]:.1f}", f"{corners[3][0]:.1f}"],
                        "Y (px)": [f"{corners[0][1]:.1f}", f"{corners[1][1]:.1f}", f"{corners[2][1]:.1f}", f"{corners[3][1]:.1f}"],
                    }
                    st.table(c_data)

                w_out = result["warped"].shape[1]
                h_out = result["warped"].shape[0]
                method_desc = "RANSAC Robust Estimation" if result.get("used_ransac") else "Direct Corner Perspective Transform"
                st.caption(f"ขนาดภาพผลลัพธ์: {w_out} × {h_out} px (สัดส่วน A4 1 : 1.414) | วิธีการคำนวณ: {method_desc}")


