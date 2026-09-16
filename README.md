# Document Scanner & Perspective Rectifier

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?logo=numpy&logoColor=white)](https://numpy.org/)

**CP461 Computer Vision Group Project**  
ระบบสแกนและปรับมุมมองเอกสารอัตโนมัติ (Perspective Rectification) จากภาพถ่ายเอกสารที่มีมุมเอียงหรือ perspective ผิดเพี้ยน แปลงกลับเป็นภาพเอกสารสัดส่วนมาตรฐาน A4 (1 : 1.414) มุมมองตรงจากด้านบน (Top-Down Flat View) พร้อมการวิเคราะห์ฟีเจอร์เชิงลึกด้วย **SIFT/ORB**, **Feature Matching**, **RANSAC**, และ **Homography Transformation**

## Streamlit App
https://documentscannerandperspectiverectifier-ceuurr4pfg8wzjqzvrygv6.streamlit.app/



## สารบัญ (Table of Contents)
- [ภาพรวมของระบบ (Overview)](#ภาพรวมของระบบ-overview)
- [ฟีเจอร์หลักและการออกแบบ UI (Features & UI Design)](#ฟีเจอร์หลักและการออกแบบ-ui-features--ui-design)
- [กระบวนการทำงาน (End-to-End Pipeline)](#กระบวนการทำงาน-end-to-end-pipeline)
- [โครงสร้างโปรเจกต์ (Project Structure)](#โครงสร้างโปรเจกต์-project-structure)
- [เทคโนโลยีที่ใช้ (Tech Stack)](#เทคโนโลยีที่ใช้-tech-stack)
- [การติดตั้งและรัน Local (Installation & Local Run)](#การติดตั้งและรัน-local-installation--local-run)
- [การนำไปขึ้นคลาวด์ (Deployment)](#การนำไปขึ้นคลาวด์-deployment)
- [ชุดข้อมูลทดสอบ (Test Dataset)](#ชุดข้อมูลทดสอบ-test-dataset)
- [การแบ่งบทบาทหน้าที่ (Task Allocation)](#การแบ่งบทบาทหน้าที่-task-allocation)
- [ข้อมูลรายวิชาและลิขสิทธิ์ (Course Information & License)](#ข้อมูลรายวิชาและลิขสิทธิ์-course-information--license)

---

## ภาพรวมของระบบ (Overview)

ในการถ่ายภาพเอกสารในชีวิตประจำวัน มักประสบปัญหาหลัก:
1. มุมกล้องไม่ได้ระนาบขนานกับระนาบเอกสาร ทำให้เกิด **Perspective Distortion**
2. ปัญหาสภาพแสง เงาสะท้อน และพื้นหลังที่มีวัตถุรบกวน (Background Clutter)
3. สัดส่วนและขนาดของเอกสารผิดเพี้ยนไปจากมาตรฐาน ส่งผลให้อ่านหรือนำไปประมวลผลต่อ (เช่น OCR) ได้ยาก

โปรเจกต์นี้ผสานรวมเทคนิคทาง **Computer Vision** เพื่อแก้ไขปัญหาดังกล่าวแบบอัตโนมัติตั้งแต่ต้นจนจบ (End-to-End):
- ระบบตรวจจับเส้นขอบเอกสารด้วย Canny Edge Detection ร่วมกับ Morphological Closing เพื่อปิดรอยขาดของขอบกระดาษ
- ค้นหาและคัดเลือก Contour ที่สอดคล้องกับระนาบเอกสาร พร้อมระบบให้คะแนน (Scoring System) ตามขนาดและอัตราส่วน A4
- เรียงลำดับพิกัดมุม 4 จุด (TL, TR, BR, BL) ด้วยมุมรอบจุดศูนย์ถ่วง (`atan2`) ซึ่งทนต่อกระดาษที่เอียงมากกว่าวิธี sum/diff แบบเดิม พร้อมตรวจความสมเหตุสมผลของสี่เหลี่ยมก่อนนำไปคำนวณ
- วิเคราะห์จุดเด่นด้วย SIFT หรือ ORB จับคู่ด้วย BFMatcher หรือ FLANN และคัดกรองจุดกำกวมด้วย Lowe's Ratio Test
- ประมาณค่า Homography Matrix ($H$) ด้วย RANSAC เพื่อขจัด Outliers แล้ว Perspective Warp เข้าสู่สัดส่วนมาตรฐาน A4

### โหมดการทำงาน 2 แบบ

แอปแยกวิธีหา Homography ออกเป็น 2 เส้นทางที่ชัดเจน และแสดงให้ผู้ใช้เห็นเสมอว่ากำลังใช้เส้นทางไหน

| | **Auto — ภาพเดียว** | **Reference — 2 ภาพ** |
|---|---|---|
| อินพุต | ภาพถ่ายเอกสารที่เอียง | ภาพถ่ายเอียง + ภาพเอกสารหน้าเดียวกันที่แบนราบ |
| ที่มาของ $H$ | 4 มุมจาก contour → `getPerspectiveTransform` | SIFT/ORB matches → Lowe's ratio test → `findHomography` + **RANSAC** |
| Feature matching | สกัด keypoints ให้ดู แต่**ไม่ได้**ใช้ประมาณ $H$ | ใช้จริง เป็นตัวกำหนด $H$ ทั้งหมด |
| ค่า inlier | ไม่มี (และไม่แสดงตัวเลขปลอม) | จำนวน inlier จริงจาก RANSAC mask |
| เหมาะกับ | ถ่ายเร็ว ๆ กระดาษตัดกับพื้นหลังชัด | เอกสารที่มีต้นฉบับอยู่แล้ว ขอบกระดาษไม่ชัด หรือมุมถูกบัง |

> โหมด Auto ไม่ได้ใช้ feature matching ประมาณ $H$ และแอปเขียนกำกับไว้ตรง ๆ ทั้งบน badge และใต้ภาพ
> ถ้าต้องการดูเส้นทาง SIFT → Matching → Ratio Test → RANSAC → Homography ครบทุกขั้น ให้ใช้โหมด Reference

---

## ฟีเจอร์หลักและการออกแบบ UI (Features & UI Design)

เว็บแอปพลิเคชันถูกออกแบบตามมาตรฐาน Modern Web Application ให้ความสำคัญกับความเรียบหรู ใช้งานง่าย และรองรับการตรวจสอบขั้นตอนทางเทคนิค (Technical Inspection):

### Section 1: Upload & Settings
- **Drag & Drop Upload:** รองรับไฟล์รูปภาพเอกสารนามสกุล JPG, JPEG, PNG, WEBP
- **Pipeline Controls (Settings Panel):**
  - **Rectification Mode:** เลือกระหว่าง `Auto` (ภาพเดียว หา H จาก contour) หรือ `Reference` (2 ภาพ หา H จาก feature matching + RANSAC) โดยโหมด Reference จะเปิดช่องอัปโหลดภาพอ้างอิงเพิ่มให้อัตโนมัติ
  - **Feature Extractor:** เลือกระหว่าง `SIFT` (เน้นความแม่นยำและทนทานต่อสเกล/มุมหมุน) หรือ `ORB` (เน้นความเร็วในการประมวลผล)
  - **Feature Matcher:** เลือกระหว่าง `BFMatcher` (Brute-Force Matcher) หรือ `FLANN` (Fast Library for Approximate Nearest Neighbors)
  - **Lowe's Ratio Threshold:** ปรับเกณฑ์คัดกรองจุดจับคู่ที่กำกวม (ค่าเริ่มต้น 0.75, ปรับได้ระหว่าง 0.50 – 0.95)
  - **Output Resolution:** กำหนดด้านยาวของผลลัพธ์ภาพ A4 (600 – 1600 px โดยอีกด้านคำนวณตามสัดส่วน 1:1.414 และสลับแนวตั้ง/แนวนอนให้อัตโนมัติตามทิศทางของเอกสาร)

  ช่อง Feature Matcher และ Lowe's Ratio Threshold จะถูก disable ในโหมด Auto เพราะโหมดนั้นไม่มีขั้นตอน matching
- **Control Buttons:**
  - `Run Scan Pipeline` — เริ่มต้นกระบวนการสแกนและประมวลผลอัตโนมัติ
  - `Download A4` — บันทึกภาพเอกสารผลลัพธ์ความละเอียดสูงเป็นไฟล์ PNG
  - `Reset` — ล้างข้อมูลภาพ สถานะการทำงาน และรีเซ็ตเซสชันทั้งหมด

### Section 2: Main Results
แสดงผลลัพธ์ 3 ช่องแบบเคียงข้างกัน (Side-by-Side) โดยช่องกลางเปลี่ยนตามโหมดที่เลือก:
1. **Original Photo:** ภาพถ่ายต้นฉบับ (แสดงแบบย่อที่ Max Dimension 1200 px)
2. **4 Corners Detection** (โหมด Auto) หรือ **Reference Document** (โหมด Reference)
3. **Final Result (A4 Corrected):** ภาพผลลัพธ์ที่ warp มาจากภาพต้นฉบับความละเอียดเต็ม ไม่ใช่จากภาพที่ย่อแล้ว
- **Pipeline Metrics** — แสดงเฉพาะค่าที่มีอยู่จริงในโหมดนั้น:
  - โหมด **Reference**: Keypoints ทั้งสองภาพ, Good Matches, RANSAC Inliers และ Inlier Ratio (%) พร้อม Progress Bar
  - โหมด **Auto**: จำนวน Keypoints, วิธีที่ใช้หามุม (contour / minAreaRect), ทิศทางเอกสาร และขนาดผลลัพธ์ — **ไม่แสดงค่า inlier เพราะโหมดนี้ไม่ได้ใช้ RANSAC**
  - Method Badges บอกตรง ๆ ว่า Homography รอบนี้มาจาก RANSAC หรือมาจาก contour

### Section 3: Technical Details (Collapsible Expander)
ส่วนแสดงผลเชิงลึกทางเทคนิคสำหรับตรวจสอบและประเมินผลขั้นตอนการทำงานของอัลกอริทึม (เปิด/ปิดได้เพื่อความสบายตา):
- **โหมด Auto** — **[Preprocessing] Canny Edge Map** และ **[Feature Extraction] Keypoints** ที่วาดขนาด/ทิศทางของแต่ละจุด พร้อมคำอธิบายกำกับว่าโหมดนี้ไม่ได้นำ keypoints ไปประมาณ Homography
- **โหมด Reference** — **[Feature Matching] Matches After Lowe's Ratio Test** แสดงเส้นโยงระหว่างภาพถ่ายกับภาพอ้างอิง และ **[RANSAC] Inliers / Outliers** แสดงจุดเขียว (inlier) เทียบจุดแดง (outlier) จาก mask จริงของ RANSAC
- **[Homography] Transformation Information:** แสดงเมทริกซ์การแปลง $H_{3 \times 3}$ ในรูปแบบสมการคณิตศาสตร์ LaTeX พร้อมตารางพิกัด 4 มุมจริงในหน่วยพิกเซล

---

## กระบวนการทำงาน (End-to-End Pipeline)

```text
                        [Input Photo]
                              │
                              ▼
        Preprocessing (Resize <=1200px / Gray / Gaussian Blur 5x5)
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
        โหมด Auto                        โหมด Reference
   (ภาพเดียว, H จาก contour)        (2 ภาพ, H จาก feature matching)
              │                                │
              ▼                                ▼
   Adaptive Canny Edge Detection     Feature Extraction (SIFT / ORB)
              │                                │
              ▼                                ▼
   Morphological Closing (5x5)       Feature Matching (BFMatcher / FLANN)
              │                                │
              ▼                                ▼
   Contour & 4-Corner Localization   Lowe's Ratio Test Filtering
   (Scoring: Area & Aspect Ratio)              │
              │                                ▼
              ▼                       RANSAC + cv2.findHomography()
   Corner Ordering (atan2)                     │
   + Quad Validation                           │
              │                                │
              ▼                                ▼
   cv2.getPerspectiveTransform()      H_total = S · H_ransac · K
              │                                │
              └───────────────┬────────────────┘
                              ▼
          Warp จาก "ภาพต้นฉบับความละเอียดเต็ม"
              (cv2.warpPerspective, INTER_CUBIC)
                              │
                              ▼
      [Final A4 Output — สลับแนวตั้ง/แนวนอนอัตโนมัติ]
```

### รายละเอียดฟังก์ชันใน Pipeline

| ขั้นตอน | รายละเอียดการทำงาน | ฟังก์ชัน | ไฟล์ต้นทาง |
|---|---|---|---|
| 1. Preprocess | ปรับขนาดภาพ (Max Dim 1200), แปลง Grayscale, ลดสัญญาณรบกวนด้วย Gaussian Blur (5x5) และคืนค่า `scale` ไว้ map พิกัดกลับไปภาพเต็ม | `preprocess()` | `src/preprocessing.py` |
| 2. Edge Detection | ตรวจจับขอบภาพด้วย Canny แบบ Adaptive Threshold อิงค่า Median และทำ Morphological Closing | `detect_edges()` | `src/detection.py` |
| 3. Corner Detection | ค้นหา Contour เอกสาร คัดเลือกด้วยคะแนนพื้นที่และสัดส่วน A4 พร้อม Fallback สองระดับ และรายงานว่าใช้วิธีไหน | `detect_document()` | `src/detection.py` |
| 4. Point Ordering | จัดเรียงพิกัด 4 จุดเป็น TL, TR, BR, BL ด้วยมุมรอบจุดศูนย์ถ่วง (`atan2`) | `order_corners()` | `src/detection.py` |
| 5. Quad Validation | ปฏิเสธสี่เหลี่ยมที่มีจุดซ้ำ เล็กเกินไป หรือไขว้กันเอง ก่อนส่งไปคำนวณ Homography | `validate_quad()` | `src/detection.py` |
| 6. Feature Extraction | สกัด Keypoints และคำนวณ Descriptors ด้วย SIFT หรือ ORB | `extract_features()` | `src/features.py` |
| 7. Feature Matching | จับคู่จุดเด่นด้วย BFMatcher หรือ FLANN ผ่าน KnnMatch (k=2) | `match_features()` | `src/features.py` |
| 8. Ratio Test | กรองคู่จุดที่กำกวมออกด้วยเกณฑ์ Lowe's Ratio Test ตามค่า Threshold ที่กำหนด | `apply_ratio_test()` | `src/features.py` |
| 9. Robust Estimation | คำนวณ Homography Matrix พร้อมขจัด Outliers ด้วย RANSAC (Threshold 5.0, Confidence 0.995) | `compute_homography()` | `src/geometry.py` |
| 10a. Rectify (Auto) | สร้าง Transform จาก 4 มุม แล้ว Warp จากภาพต้นฉบับความละเอียดเต็ม | `rectify_from_corners()` | `src/geometry.py` |
| 10b. Rectify (Reference) | ใช้ $H$ จาก RANSAC ต่อกับเมทริกซ์ scale แล้ว Warp เป็น A4 พร้อมเช็คจำนวน inlier ขั้นต่ำ | `rectify_from_reference()` | `src/geometry.py` |
| 11. Output Sizing | คำนวณขนาด A4 (1:1.414) และสลับแนวตั้ง/แนวนอนตามสัดส่วนของเอกสารที่ตรวจเจอ | `get_a4_dimensions()`, `quad_is_landscape()` | `src/geometry.py` |

---

## โครงสร้างโปรเจกต์ (Project Structure)

```text
document-scanner/
├── app.py                         # Web Application หลักพัฒนาด้วย Streamlit (UI, Session State, Pipeline Runner)
├── requirements.txt               # รายการ Python dependencies (opencv-python-headless, numpy, Pillow, streamlit)
├── packages.txt                   # รายการ System packages สำหรับ Cloud Linux Environment (libgl1, libglib2.0-0)
├── README.md                      # เอกสารคู่มือการใช้งานและรายละเอียดโปรเจกต์
├── CP461_document_scanner_spec.md    # ข้อกำหนดทางเทคนิคและเกณฑ์โครงงาน CP461
├── src/
│   ├── __init__.py                # Source package initialization
│   ├── preprocessing.py           # ฟังก์ชันปรับขนาดรูป, แปลง Grayscale, และ Gaussian Blur
│   ├── detection.py               # ฟังก์ชัน Canny Edge, Morphological Closing, Contour Scoring, Corner Fallback
│   ├── features.py                # ฟังก์ชันสกัดจุดเด่น (SIFT/ORB), การจับคู่ (BF/FLANN), และ Lowe's Ratio Test
│   ├── geometry.py                # ฟังก์ชัน Homography, RANSAC, ขนาด A4 ตามทิศทาง และ Warp ทั้งสองโหมด
│   └── utils.py                   # ฟังก์ชันวาดเส้นกรอบมุม, แสดงคู่จุด Match, Inliers/Outliers, และแปลง Format ภาพ
└── tests/
    └── sample_images/             # ชุดภาพตัวอย่างสำหรับทดสอบระบบ
        ├── test1.webp             # ภาพเอกสารมุมเอียงทั่วไป
        └── test2.webp             # ภาพเอกสารมุมเอียงองศาสูง (Perspective จัด)
```

---

## เทคโนโลยีที่ใช้ (Tech Stack)

- **ภาษาหลัก:** Python 3.10+
- **Computer Vision Library:** OpenCV (`opencv-python-headless` เวอร์ชัน 4.x)
- **การประมวลผลเชิงตัวเลขและเมทริกซ์:** NumPy
- **การประมวลผลและจัดการไฟล์ภาพ:** Pillow (PIL)
- **Web Application Framework:** Streamlit
- **ระบบปฏิบัติการเป้าหมาย:** Cross-platform (Windows, macOS, Linux / Docker)
- **Version Control:** Git & GitHub

---

## การติดตั้งและรัน Local (Installation & Local Run)

### 1. โคลนคลังโค้ด (Clone Repository)
```bash
git clone https://github.com/KrittinB/Document_ScannerAndPerspective_Rectifier.git
cd Document_ScannerAndPerspective_Rectifier
```
*(หากโฟลเดอร์ในเครื่องอยู่ในโฟลเดอร์ย่อย ให้ใช้คำสั่ง `cd document-scanner`)*

### 2. สร้างและเปิดใช้งาน Virtual Environment (แนะนำ)
```bash
# สำหรับ Windows
python -m venv venv
venv\Scripts\activate

# สำหรับ macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 4. รัน Web Application
```bash
streamlit run app.py
```
เปิดเว็บบราวเซอร์แล้วเข้าไปที่ URL: `http://localhost:8501`



---

## ชุดข้อมูลทดสอบ (Test Dataset)

ในโฟลเดอร์ `tests/sample_images/` ได้จัดเตรียมภาพเอกสารตัวอย่างที่มีสภาพแวดล้อมและความยากแตกต่างกันเพื่อการทดสอบ:
1. `test1.webp`: ภาพเอกสารบนระนาบที่มีมุมเอียงเล็กน้อยถึงปานกลาง
2. `test2.webp`: ภาพเอกสารที่มีมุมมองเฉียงองศาสูง ทดสอบความทนทานของ Perspective Rectification

> **TODO (ก่อนส่งงาน):** ถ่ายภาพ edge case ด้วยมือถือเองเพิ่ม — แสงเงาทับขอบ, พื้นหลังรก,
> มุมกระดาษถูกมือบัง, กระดาษสีกลืนกับโต๊ะ — rubric ให้ 1.0 pt กับการ demo edge case ด้วยภาพของตัวเอง



---

## ข้อมูลรายวิชาและลิขสิทธิ์ (Course Information & License)
- **รายวิชา:** CP461 Computer Vision
- **ลิขสิทธิ์:** สำหรับใช้ประกอบการศึกษาและการประเมินผลโครงงานในรายวิชา CP461
