"""
app.py — Smart Document Scanner
Streamlit Web App สำหรับ CP461 Group Project

มี 2 โหมดที่ใช้คนละวิธีหา Homography และบอกผู้ใช้ตรง ๆ ว่ากำลังใช้วิธีไหน:

  Auto (ภาพเดียว)     Upload → Preprocess → Canny → Contour → 4 มุม
                      → getPerspectiveTransform → A4
                      สกัด SIFT/ORB keypoints ให้ดูด้วย แต่ไม่ได้เอาไปประมาณ H

  Reference (2 ภาพ)   Upload ภาพถ่าย + ภาพเอกสารอ้างอิงที่แบนราบ
                      → SIFT/ORB → BFMatcher/FLANN → Lowe's ratio test
                      → RANSAC + findHomography → warpPerspective → A4
                      โหมดนี้ H มาจาก feature matching จริง
"""

import streamlit as st
import numpy as np
import cv2

from src.preprocessing import preprocess
from src.detection import detect_document
from src.features import extract_features, extract_and_match
from src.geometry import rectify_from_corners, rectify_from_reference
from src.utils import (
    numpy_bgr_to_pil,
    numpy_gray_to_pil,
    bytes_to_numpy_bgr,
    draw_corners,
    draw_keypoints,
    draw_matches_visualization,
    draw_inlier_outlier,
    numpy_to_bytes_png,
)
from src.enhancement import (
    apply_filter,
    FILTER_MODES,
    FILTER_ORIGINAL,
    FILTER_MAGIC,
    FILTER_BW,
    FILTER_GRAY,
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
    .badge-amber { background: #fef3c7; color: #92400e; }

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
MODE_AUTO = "Auto — ภาพเดียว (Contour)"
MODE_REF = "Reference — 2 ภาพ (SIFT + RANSAC)"


def _init_state():
    defaults = {
        "uploaded_img": None,
        "reference_img": None,
        "result": None,
        "step": 0,  # 0=upload, 1=detect/match, 2=estimate, 3=done
        "uploader_key": 0,
        "current_enhanced_img": None,
        "current_filter_name": "original",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────
# Stepper HTML
# ─────────────────────────────────────────────
STEPS_BY_MODE = {
    MODE_AUTO: ["Upload", "Detect", "Rectify", "Result"],
    MODE_REF: ["Upload", "Match", "RANSAC", "Result"],
}


def render_stepper(current: int, steps: list):
    html = '<div class="stepper-container">'
    for i, label in enumerate(steps):
        if i < current:
            cls = "step done"
        elif i == current:
            cls = "step active"
        else:
            cls = "step"
        html += f'<div class="{cls}">{label}</div>'
        if i < len(steps) - 1:
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

        **โหมด Auto (ภาพเดียว)**
        1. Preprocessing (Resize, Grayscale, Gaussian Blur)
        2. Edge Detection (Canny + Morphological Closing)
        3. Contour & 4-Corner Localization
        4. `getPerspectiveTransform` → Warp เป็น A4

        **โหมด Reference (2 ภาพ)**
        1. Preprocessing ทั้งภาพถ่ายและภาพอ้างอิง
        2. Feature Extraction (SIFT / ORB)
        3. Feature Matching (BFMatcher / FLANN)
        4. Lowe's Ratio Test Filtering
        5. RANSAC + `findHomography`
        6. Perspective Warping ด้วย H จาก RANSAC → A4

        ---
        โหมด Auto ใช้ contour หามุม จึง**ไม่ได้**ใช้ feature matching
        ประมาณ Homography — ถ้าต้องการเส้นทาง SIFT + RANSAC เต็มรูปแบบ
        ให้ใช้โหมด Reference
        """
    )

# ─────────────────────────────────────────────
# Section 1: Upload & Settings
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Section 1: Upload & Settings</div>', unsafe_allow_html=True)

col_upload, col_settings = st.columns([1.1, 0.9], gap="large")

with col_settings:
    st.markdown('<div class="subsection-title">Pipeline Settings</div>', unsafe_allow_html=True)
    with st.container(border=True):
        mode = st.radio(
            "Rectification Mode",
            [MODE_AUTO, MODE_REF],
            help=(
                "Auto: หามุมกระดาษจาก contour แล้วดัดมุมมองตรง ๆ ใช้ภาพเดียว\n\n"
                "Reference: จับคู่ feature ระหว่างภาพถ่ายกับภาพเอกสารอ้างอิง "
                "แล้วประมาณ Homography ด้วย RANSAC — เป็นเส้นทางที่ใช้ SIFT/ORB จริง"
            ),
        )

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
                disabled=(mode == MODE_AUTO),
                help="BFMatcher แม่นยำ, FLANN เร็วกว่าสำหรับ dataset ขนาดใหญ่ (ใช้เฉพาะโหมด Reference)",
            )

        ratio_threshold = st.slider(
            "Lowe's Ratio Threshold", 0.50, 0.95, 0.75, 0.05,
            disabled=(mode == MODE_AUTO),
            help="ค่ามาตรฐาน 0.70 - 0.75 สำหรับกรอง ambiguous matches (ใช้เฉพาะโหมด Reference)",
        )
        a4_height = st.slider(
            "Output Resolution (ด้านยาว, px)", 600, 1600, 800, 100,
            help="ด้านยาวของภาพ A4 ผลลัพธ์ (อีกด้านคำนวณจากสัดส่วน 1:1.414 และสลับอัตโนมัติถ้าเอกสารเป็นแนวนอน)",
        )

with col_upload:
    st.markdown('<div class="subsection-title">Upload Document Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "ภาพถ่ายเอกสารที่เอียง (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"file_upload_{st.session_state.uploader_key}",
    )

    if uploaded_file is not None:
        img_bgr = bytes_to_numpy_bgr(uploaded_file.read())
        if img_bgr is None:
            st.error("ไม่สามารถอ่านไฟล์ภาพได้ กรุณาลองไฟล์อื่น")
            st.stop()
        st.session_state.uploaded_img = img_bgr
        if st.session_state.step == 0:
            st.session_state.step = 1

    if mode == MODE_REF:
        ref_file = st.file_uploader(
            "ภาพเอกสารอ้างอิงที่แบนราบ — หน้าเดียวกัน ถ่ายตรง ๆ หรือสแกนมา",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"ref_upload_{st.session_state.uploader_key}",
        )
        if ref_file is not None:
            ref_bgr = bytes_to_numpy_bgr(ref_file.read())
            if ref_bgr is None:
                st.error("ไม่สามารถอ่านไฟล์ภาพอ้างอิงได้ กรุณาลองไฟล์อื่น")
                st.stop()
            st.session_state.reference_img = ref_bgr

needs_ref = (mode == MODE_REF) and st.session_state.reference_img is None
if needs_ref and st.session_state.uploaded_img is not None:
    st.info("โหมด Reference ต้องใช้ 2 ภาพ — กรุณาอัปโหลดภาพเอกสารอ้างอิงด้วย")

# ปุ่มวางเป็นแถวเต็มความกว้างใต้ทั้งสองคอลัมน์ เพื่อให้ข้อความบนปุ่มไม่โดนตัดบนจอแคบ
col_btn1, col_btn2, col_btn3, _ = st.columns([1.4, 1.2, 1.0, 2.4])

with col_btn1:
    run_btn = st.button(
        "Run Scan Pipeline",
        type="primary",
        disabled=(st.session_state.uploaded_img is None or needs_ref),
        use_container_width=True,
    )

with col_btn2:
    result_state = st.session_state.result
    download_ready = (
        result_state is not None
        and result_state.get("success")
        and result_state.get("warped") is not None
    )
    if download_ready:
        dl_img = st.session_state.get("current_enhanced_img")
        if dl_img is None:
            dl_img = result_state["warped"]
        filter_tag = st.session_state.get("current_filter_name", "a4")
        st.download_button(
            "Download A4",
            data=numpy_to_bytes_png(dl_img),
            file_name=f"scanned_document_{filter_tag}.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.button("Download A4", disabled=True, use_container_width=True)

with col_btn3:
    if st.button("Reset", use_container_width=True):
        st.session_state.uploaded_img = None
        st.session_state.reference_img = None
        st.session_state.result = None
        st.session_state.step = 0
        st.session_state.uploader_key += 1
        st.session_state.current_enhanced_img = None
        st.session_state.current_filter_name = "original"
        st.rerun()

render_stepper(st.session_state.step, STEPS_BY_MODE[mode])


# ─────────────────────────────────────────────
# Pipeline Execution Logic
# ─────────────────────────────────────────────
if run_btn and st.session_state.uploaded_img is not None:
    photo = st.session_state.uploaded_img

    # เก็บค่า setting ที่ใช้ตอนรันไว้ในผลลัพธ์ เพื่อให้คำอธิบายใต้ภาพตรงกับรอบที่รันจริง
    settings = {
        "mode": mode,
        "feature_method": feature_method,
        "matcher_method": matcher_method,
        "ratio": ratio_threshold,
        "a4_height": a4_height,
    }

    with st.spinner("กำลังประมวลผล pipeline..."):
        try:
            prep = preprocess(photo)
            resized, gray, blurred, scale = prep["resized"], prep["gray"], prep["blurred"], prep["scale"]

            # ── โหมด Auto: contour → 4 มุม → warp ────────────────────────
            if mode == MODE_AUTO:
                st.session_state.step = 1
                det = detect_document(blurred, resized.shape[:2])

                if not det["success"]:
                    st.session_state.result = {
                        "success": False,
                        "message": det["message"],
                        "img_resized": resized,
                        "edges": det["edges"],
                        "corners": det.get("corners"),
                        "settings": settings,
                    }
                    st.session_state.step = 1
                else:
                    st.session_state.step = 2
                    geo = rectify_from_corners(photo, det["corners"], scale, a4_height)

                    # สกัด keypoints ไว้แสดงให้เห็นขั้นตอน feature extraction
                    # (โหมดนี้ไม่ได้เอาไปประมาณ H — เขียนกำกับไว้ใต้ภาพแล้ว)
                    kp, _ = extract_features(gray, feature_method)

                    st.session_state.result = {
                        **geo,
                        "img_resized": resized,
                        "edges": det["edges"],
                        "corners": det["corners"],
                        "detect_method": det["method"],
                        "detect_message": det["message"],
                        "keypoints": kp,
                        "settings": settings,
                    }
                    st.session_state.step = 3 if geo["success"] else 2

            # ── โหมด Reference: SIFT → match → ratio → RANSAC → warp ─────
            else:
                reference = st.session_state.reference_img
                ref_prep = preprocess(reference)

                st.session_state.step = 1
                feat = extract_and_match(
                    gray, ref_prep["gray"],
                    method=feature_method,
                    matcher=matcher_method,
                    ratio=ratio_threshold,
                )

                st.session_state.step = 2
                geo = rectify_from_reference(
                    photo, scale, ref_prep["resized"], feat, a4_height
                )

                st.session_state.result = {
                    **geo,
                    "img_resized": resized,
                    "ref_resized": ref_prep["resized"],
                    "feat": feat,
                    "settings": settings,
                }
                st.session_state.step = 3 if geo["success"] else 1

        except Exception as e:
            st.session_state.result = {
                "success": False,
                "message": f"เกิดข้อผิดพลาด: {e}",
                "settings": settings,
            }

    st.rerun()


# ─────────────────────────────────────────────
# Section 2: Main Results
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Section 2: Main Results</div>', unsafe_allow_html=True)

result = st.session_state.result

if result is None:
    if st.session_state.uploaded_img is not None:
        st.info("อัปโหลดภาพแล้ว — กดปุ่ม 'Run Scan Pipeline' ใน Section 1 เพื่อเริ่มประมวลผล")
        col_prev, col_prev2, _ = st.columns([1, 1, 1])
        with col_prev:
            st.image(
                numpy_bgr_to_pil(st.session_state.uploaded_img),
                caption="ภาพถ่ายเอกสาร (Preview)",
                use_container_width=True,
            )
        if st.session_state.reference_img is not None:
            with col_prev2:
                st.image(
                    numpy_bgr_to_pil(st.session_state.reference_img),
                    caption="ภาพอ้างอิง (Preview)",
                    use_container_width=True,
                )
    else:
        st.info("กรุณาอัปโหลดภาพถ่ายเอกสารใน Section 1 ด้านบนเพื่อเริ่มต้นใช้งาน")

elif not result.get("success"):
    # ── กรณีล้มเหลว: บอกเหตุผลตรง ๆ พร้อมภาพประกอบให้ผู้ใช้เห็นว่าระบบเห็นอะไร ──
    st.error(result.get("message", "ประมวลผลไม่สำเร็จ"))

    # เก็บเฉพาะภาพที่มีจริงในรอบนี้ แล้วค่อยแบ่งคอลัมน์ตามจำนวน — กันคอลัมน์ว่าง
    panels = []
    if result.get("img_resized") is not None:
        panels.append((numpy_bgr_to_pil(result["img_resized"]), "ภาพถ่ายต้นฉบับ"))
    if result.get("ref_resized") is not None:
        panels.append((numpy_bgr_to_pil(result["ref_resized"]), "ภาพอ้างอิงที่ใช้จับคู่"))
    if result.get("edges") is not None:
        panels.append((numpy_gray_to_pil(result["edges"]), "Canny Edge Map — สิ่งที่ระบบมองเห็น"))
    if result.get("corners") is not None and result.get("img_resized") is not None:
        panels.append((
            numpy_bgr_to_pil(draw_corners(result["img_resized"], result["corners"], color=(0, 165, 255))),
            "กรอบที่ระบบเดาได้ (ยังไม่ผ่านเกณฑ์)",
        ))

    if panels:
        for col, (img, cap) in zip(st.columns(len(panels)), panels):
            with col:
                st.image(img, caption=cap, use_container_width=True)

    if result.get("settings", {}).get("mode") == MODE_REF:
        st.caption(
            "ลองใช้ภาพอ้างอิงที่เป็นเอกสารหน้าเดียวกันจริง ๆ ถ่ายให้คมชัดและเห็นเต็มหน้า "
            "หรือเพิ่มค่า Lowe's Ratio Threshold เพื่อให้ผ่าน ratio test มากขึ้น"
        )
    else:
        st.caption(
            "ลองถ่ายใหม่ให้กระดาษตัดกับพื้นหลังชัดขึ้น หลีกเลี่ยงเงาทับขอบกระดาษ "
            "หรือเปลี่ยนไปใช้โหมด Reference ที่ไม่ต้องพึ่งขอบกระดาษ"
        )

else:
    settings = result.get("settings", {})
    run_mode = settings.get("mode", MODE_AUTO)
    is_ref = run_mode == MODE_REF

    # ── Status Badges ──
    if result.get("used_ransac"):
        est_badge = '<span class="badge badge-green">Homography จาก RANSAC + feature matching</span>'
    else:
        est_badge = '<span class="badge badge-amber">Homography จาก 4 มุม (contour) — ไม่ได้ใช้ feature</span>'
    method_badge = (
        f'<span class="badge badge-purple">{settings.get("feature_method")} '
        f'+ {settings.get("matcher_method")}</span>'
        if is_ref else
        f'<span class="badge badge-purple">{settings.get("feature_method")} keypoints</span>'
    )
    orient_badge = (
        f'<span class="badge badge-blue">A4 '
        f'{"แนวนอน" if result.get("landscape") else "แนวตั้ง"}</span>'
    )
    st.markdown(f"{est_badge} &nbsp; {method_badge} &nbsp; {orient_badge}", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)

    # ── Document Enhancement & Smart Filters ──
    with st.container(border=True):
        f_col_select, f_col_tune = st.columns([2.8, 1.2], vertical_alignment="center")
        with f_col_select:
            selected_filter = st.radio(
                "✨ Document Filter (ปรับปรุงคุณภาพและลบเงา):",
                FILTER_MODES,
                index=0,
                horizontal=True,
                help=(
                    "Original Color: ภาพสีต้นฉบับคมชัดเต็มพิกเซล\n"
                    "Magic Color: ลบเงาด้วย Morphological Division + ปรับสีสดใสด้วย CLAHE ใน LAB Space\n"
                    "Clean B&W: เอกสารขาว-ดำคมกริบ พื้นหลังขาวบริสุทธิ์แบบเครื่องสแกน\n"
                    "Grayscale Scan: เฉดสีเทา ปรับสมดุลแสงสม่ำเสมอทั่วทั้งแผ่น"
                ),
            )
        with f_col_tune:
            with st.popover("⚙️ ปรับแต่งฟิลเตอร์ละเอียด"):
                f_bright = st.slider("ความสว่าง (Brightness)", -50, 50, 0, 5)
                f_contrast = st.slider("คอนทราสต์ (Contrast)", 0.5, 2.0, 1.05, 0.05)
                f_bw_c = st.slider("ความไวขาวดำ (B&W Sensitivity)", 3, 31, 11, 2)
                show_compare = st.checkbox("เปรียบเทียบ ก่อน/หลัง แต่งภาพ", value=False)

    enhanced_warped = apply_filter(
        result["warped"],
        selected_filter,
        brightness=f_bright,
        contrast=f_contrast,
        bw_threshold_c=f_bw_c,
    )
    st.session_state["current_enhanced_img"] = enhanced_warped
    st.session_state["current_filter_name"] = selected_filter.split()[0].lower()

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.image(numpy_bgr_to_pil(result["img_resized"]), caption="Original Photo", use_container_width=True)
        st.markdown('<p class="img-caption">ภาพถ่ายต้นฉบับ (แสดงแบบย่อ)</p>', unsafe_allow_html=True)

    with res_col2:
        if is_ref:
            st.image(numpy_bgr_to_pil(result["ref_resized"]), caption="Reference Document", use_container_width=True)
            st.markdown('<p class="img-caption">ภาพอ้างอิงที่ใช้เป็นเป้าหมายของ Homography</p>', unsafe_allow_html=True)
        else:
            st.image(
                numpy_bgr_to_pil(draw_corners(result["img_resized"], result["corners"])),
                caption="4 Corners Detection", use_container_width=True,
            )
            st.markdown('<p class="img-caption">ตรวจจับกรอบและ 4 มุมเอกสาร (TL/TR/BR/BL)</p>', unsafe_allow_html=True)

    with res_col3:
        out_h, out_w = enhanced_warped.shape[:2]
        if show_compare and selected_filter != FILTER_ORIGINAL:
            tab_enhanced, tab_raw = st.tabs(["✨ ปรับแต่งแล้ว (Enhanced)", "📷 ก่อนปรับ (Raw Warped)"])
            with tab_enhanced:
                st.image(numpy_bgr_to_pil(enhanced_warped), caption=f"Final Result ({selected_filter})", use_container_width=True)
            with tab_raw:
                st.image(numpy_bgr_to_pil(result["warped"]), caption="Raw Warped (Original)", use_container_width=True)
        else:
            st.image(numpy_bgr_to_pil(enhanced_warped), caption=f"Final Result ({selected_filter})", use_container_width=True)

        st.markdown(
            f'<p class="img-caption">ผลลัพธ์ A4 ({out_w}×{out_h} px) — {selected_filter}</p>',
            unsafe_allow_html=True,
        )
        st.download_button(
            f"📥 Download ({selected_filter.split()[0]})",
            data=numpy_to_bytes_png(enhanced_warped),
            file_name=f"scanned_document_{st.session_state['current_filter_name']}.png",
            mime="image/png",
            use_container_width=True,
        )

    # ── Metrics ──
    st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

    def _metric(col, value, label):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="value">{value}</div>'
                f'<div class="label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    if is_ref:
        feat = result.get("feat", {})
        n_good = feat.get("n_good", 0)
        n_inliers = result.get("n_inliers", 0)
        inlier_ratio = int(round(n_inliers / max(n_good, 1) * 100))

        cols = st.columns(5)
        _metric(cols[0], feat.get("n_keypoints_1", 0), "Keypoints (ภาพถ่าย)")
        _metric(cols[1], feat.get("n_keypoints_2", 0), "Keypoints (อ้างอิง)")
        _metric(cols[2], n_good, "Good Matches (หลัง ratio test)")
        _metric(cols[3], n_inliers, "RANSAC Inliers")
        _metric(cols[4], f"{inlier_ratio}%", "Inlier Ratio")

        st.progress(
            min(inlier_ratio, 100) / 100,
            text=f"Inlier Ratio: {n_inliers} จาก {n_good} คู่ที่ผ่าน ratio test ({inlier_ratio}%)",
        )
    else:
        out_h, out_w = result["warped"].shape[:2]
        detect_label = {
            "contour": "Contour (4 จุด)",
            "minarearect": "minAreaRect",
        }.get(result.get("detect_method"), result.get("detect_method", "-"))

        cols = st.columns(4)
        _metric(cols[0], len(result.get("keypoints", [])), f'{settings.get("feature_method")} Keypoints')
        _metric(cols[1], detect_label, "วิธีหามุม")
        _metric(cols[2], "แนวนอน" if result.get("landscape") else "แนวตั้ง", "ทิศทางเอกสาร")
        _metric(cols[3], f"{out_w}×{out_h}", "ขนาดผลลัพธ์ (px)")

        st.caption(
            "โหมด Auto ไม่มีค่า inlier เพราะไม่ได้ใช้ RANSAC ประมาณ Homography — "
            "ถ้าต้องการตัวเลข inlier จาก RANSAC จริง ให้สลับไปโหมด Reference"
        )

    st.success(result.get("message", ""))


# ─────────────────────────────────────────────
# Section 3: Technical Pipeline Details
# ─────────────────────────────────────────────
if result is not None and result.get("success"):
    settings = result.get("settings", {})
    is_ref = settings.get("mode") == MODE_REF

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("Technical Details", expanded=False):
        tech_col1, tech_col2 = st.columns(2, gap="medium")

        with tech_col1:
            if is_ref:
                with st.container(border=True):
                    st.markdown("#### [Feature Matching] Matches After Lowe's Ratio Test")
                    feat = result.get("feat", {})
                    good_matches = feat.get("good_matches", [])
                    kp1, kp2 = feat.get("kp1", []), feat.get("kp2", [])

                    if good_matches and kp1 and kp2:
                        match_vis = draw_matches_visualization(
                            result["img_resized"], kp1, result["ref_resized"], kp2, good_matches,
                        )
                        st.image(numpy_bgr_to_pil(match_vis), caption="Feature Matching Visualization",
                                 use_container_width=True)
                    else:
                        st.warning("ไม่มีคู่จุดที่ผ่าน ratio test ให้แสดง")
                    st.caption(
                        f'จับคู่จุดเด่นด้วย {settings.get("matcher_method")} (knnMatch k=2) '
                        f'แล้วกรองจุดกำกวมด้วย Lowe\'s Ratio Test ที่ threshold '
                        f'{settings.get("ratio", 0):.2f} เหลือ {len(good_matches)} คู่'
                    )
            else:
                with st.container(border=True):
                    st.markdown("#### [Preprocessing] Canny Edge Map")
                    st.image(numpy_gray_to_pil(result["edges"]), caption="Canny Edge Map",
                             use_container_width=True)
                    st.caption(
                        "ตรวจจับขอบด้วย Canny แบบ adaptive threshold (อิงค่า median) "
                        "หลัง Gaussian Blur 5×5 แล้วปิดรอยขาดของขอบด้วย Morphological Closing"
                    )

                with st.container(border=True):
                    st.markdown(f'#### [Feature Extraction] {settings.get("feature_method")} Keypoints')
                    kp = result.get("keypoints", [])
                    st.image(
                        numpy_bgr_to_pil(draw_keypoints(result["img_resized"], kp)),
                        caption=f"{len(kp)} keypoints", use_container_width=True,
                    )
                    st.caption(
                        f'สกัด keypoints ด้วย {settings.get("feature_method")} ได้ {len(kp)} จุด '
                        "(ขนาดวงกลม = scale, เส้นในวงกลม = orientation) — "
                        "**โหมด Auto ไม่ได้นำ keypoints เหล่านี้ไปประมาณ Homography** "
                        "แสดงไว้เพื่อให้เห็นขั้นตอน feature extraction เท่านั้น"
                    )

        with tech_col2:
            if is_ref:
                with st.container(border=True):
                    st.markdown("#### [RANSAC] Inliers / Outliers")
                    feat = result.get("feat", {})
                    inlier_img = draw_inlier_outlier(
                        result["img_resized"], feat.get("kp1", []),
                        feat.get("good_matches", []), result.get("mask"),
                    )
                    if inlier_img is not None:
                        st.image(numpy_bgr_to_pil(inlier_img), caption="Inlier (เขียว) vs Outlier (แดง)",
                                 use_container_width=True)
                        st.caption(
                            "RANSAC สุ่มเลือกชุดจุด 4 คู่ซ้ำ ๆ เพื่อหา Homography ที่มีจุดสอดคล้องมากที่สุด "
                            "จุดเขียวคือคู่ที่เข้ากับระนาบเดียวกัน (inlier) จุดแดงคือคู่ที่ถูกตัดทิ้ง (outlier)"
                        )
                    else:
                        st.info("ไม่มี inlier mask สำหรับรอบนี้")

            with st.container(border=True):
                st.markdown("#### [Homography] Transformation Information")
                H = result.get("H")
                if H is not None:
                    st.markdown("**Homography Matrix H (3×3):**")
                    st.latex(
                        r"H = \begin{bmatrix}"
                        r"%.4f & %.4f & %.4f \\ %.4f & %.4f & %.4f \\ %.4f & %.4f & %.4f"
                        r"\end{bmatrix}"
                        % (H[0, 0], H[0, 1], H[0, 2],
                           H[1, 0], H[1, 1], H[1, 2],
                           H[2, 0], H[2, 1], H[2, 2])
                    )

                corners = result.get("corners")
                if corners is not None and len(corners) == 4:
                    st.markdown("**Detected Corner Coordinates (พิกัดในภาพย่อ):**")
                    st.table({
                        "Corner": ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Right (BR)", "Bottom-Left (BL)"],
                        "X (px)": [f"{c[0]:.1f}" for c in corners],
                        "Y (px)": [f"{c[1]:.1f}" for c in corners],
                    })

                out_h, out_w = result["warped"].shape[:2]
                method_desc = (
                    "RANSAC Robust Estimation จาก feature matches"
                    if result.get("used_ransac")
                    else "Direct 4-Point Perspective Transform จาก contour"
                )
                st.caption(
                    f"ขนาดผลลัพธ์: {out_w} × {out_h} px (สัดส่วน A4 1 : 1.414) · วิธีคำนวณ: {method_desc}"
                )

            with st.container(border=True):
                st.markdown("#### [Post-Processing] Document Enhancement & Filters")
                if selected_filter == FILTER_MAGIC:
                    st.markdown(
                        "**Magic Color Mode (Auto-Enhance & Shadow Removal):**\n"
                        "- **Shadow Removal:** ประมาณระนาบแสงพื้นหลังด้วย Morphological Dilation/Closing (Kernel 35×35) ร่วมกับ Median Blur แล้วทำการ Division Normalization ($I_{norm} = I / B \\times 255$) เพื่อลบเงามือถือและปรับแสงให้สม่ำเสมอ\n"
                        "- **LAB Contrast Enhancement:** แปลงเข้าสู่ระบบสี LAB แล้วประยุกต์ใช้ CLAHE บน L-Channel (Luminance) โดยเฉพาะ เพื่อเร่งคอนทราสต์โดยไม่เพี้ยนสี\n"
                        "- **Unsharp Masking:** เพิ่มความคมชัดของลายเส้นตัวอักษรด้วย Gaussian Blur Weighted Difference"
                    )
                elif selected_filter == FILTER_BW:
                    st.markdown(
                        "**Clean B&W Mode (Document Scanner Binarization):**\n"
                        "- **Pre-Binarization Illumination Flattening:** เกลี่ยแสงพื้นหลังกระดาษให้เรียบเท่ากันทั่วทั้งหน้า เพื่อป้องกันไม่ให้บริเวณเงามืดกลายเป็นปื้นดำ\n"
                        "- **Adaptive Gaussian Thresholding:** คำนวณค่า Threshold แบบ Local สำหรับแต่ละพิกเซลโดยอิงเกาส์เซียนรอบจุด\n"
                        "- **Denoising:** กรองสัญญาณรบกวนและเกล็ดหมึก (Salt & Pepper Noise) ด้วย Median Filter (3×3)"
                    )
                elif selected_filter == FILTER_GRAY:
                    st.markdown(
                        "**Grayscale Scan Mode:**\n"
                        "- แปลงเป็นเฉดสีเทา ลบเงามืดทั่วทั้งแผ่น และขยายช่วงไดนามิก (Contrast Stretching) ด้วย CLAHE เหมาะกับเอกสารลายมือหรือเอกสารที่มีภาพประกอบ"
                    )
                else:
                    st.markdown(
                        "**Original Color Mode:**\n"
                        "- แสดงผลลัพธ์ภาพสีดั้งเดิมที่ได้จากการ Warp Perspective ระดับ Full Resolution โดยตรงจากภาพต้นฉบับ"
                    )
