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
- มีกลไก Fallback ที่แข็งแกร่งด้วย Minimum Area Bounding Box และ Full-frame Boundary เพื่อป้องกันข้อผิดพลาดกรณีตรวจจับไม่พบ
- ตรวจสอบและเรียงลำดับพิกัดมุม 4 จุด (Top-Left, Top-Right, Bottom-Right, Bottom-Left) พร้อมระบบทดสอบทิศทางการหมุน (Orientation Candidates) ป้องกันปัญหาภาพผลลัพธ์ม้วนกลับด้านหรือมืดดำ
- วิเคราะห์จุดเด่นด้วย SIFT หรือ ORB และจับคู่จุดเด่นด้วย BFMatcher หรือ FLANN พร้อมคัดกรองจุดกำกวมด้วย Lowe's Ratio Test
- ประมาณค่า Homography Matrix ($H$) ด้วย RANSAC เพื่อขจัด Outliers และทำการ Perspective Warp เข้าสู่สัดส่วนมาตรฐาน A4

---

## ฟีเจอร์หลักและการออกแบบ UI (Features & UI Design)

เว็บแอปพลิเคชันถูกออกแบบตามมาตรฐาน Modern Web Application ให้ความสำคัญกับความเรียบหรู ใช้งานง่าย และรองรับการตรวจสอบขั้นตอนทางเทคนิค (Technical Inspection):

### Section 1: Upload & Settings
- **Drag & Drop Upload:** รองรับไฟล์รูปภาพเอกสารนามสกุล JPG, JPEG, PNG, WEBP
- **Pipeline Controls (Sidebar / Settings Panel):**
  - **Feature Extractor:** เลือกระหว่าง `SIFT` (เน้นความแม่นยำและทนทานต่อสเกล/มุมหมุน) หรือ `ORB` (เน้นความเร็วในการประมวลผล)
  - **Feature Matcher:** เลือกระหว่าง `BFMatcher` (Brute-Force Matcher) หรือ `FLANN` (Fast Library for Approximate Nearest Neighbors)
  - **Lowe's Ratio Threshold:** ปรับเกณฑ์คัดกรองจุดจับคู่ที่กำกวม (ค่าเริ่มต้น 0.75, ปรับได้ระหว่าง 0.50 – 0.95)
  - **Output Resolution:** กำหนดความสูงของผลลัพธ์ภาพ A4 (600 – 1600 px โดยความกว้างจะถูกคำนวณตามสัดส่วน 1:1.414 อัตโนมัติ)
- **Control Buttons:**
  - `Run Scan Pipeline` — เริ่มต้นกระบวนการสแกนและประมวลผลอัตโนมัติ
  - `Download A4` — บันทึกภาพเอกสารผลลัพธ์ความละเอียดสูงเป็นไฟล์ PNG
  - `Reset` — ล้างข้อมูลภาพ สถานะการทำงาน และรีเซ็ตเซสชันทั้งหมด

### Section 2: Main Results
แสดงผลลัพธ์เปรียบเทียบ 3 ขั้นตอนหลักแบบเคียงข้างกัน (Side-by-Side):
1. **Original Image:** ภาพถ่ายต้นฉบับหลังผ่านการปรับสเกลขนาดมาตรฐาน (Max Dimension 1280 px)
2. **4 Corners Detection:** ภาพแสดงเส้นกรอบเอกสารและพิกัดมุมทั้ง 4 จุด (TL, TR, BR, BL) พร้อมมาร์กเกอร์สี
3. **Final Result (A4 Corrected):** ภาพเอกสารที่ผ่านการปรับมุมมองระนาบตรง (Top-Down A4 Flat View)
- **Pipeline Metrics & Confidence Indicators:**
  - Keypoints (Source & Destination)
  - Good Matches (จำนวนคู่จุดเด่นที่ผ่าน Lowe's Ratio Test)
  - RANSAC Inliers (จำนวนจุดที่สอดคล้องกับระนาบจริง)
  - Pipeline Confidence (%) คำนวณจากสัดส่วน Inliers ต่อ Good Matches พร้อมแถบ Progress Bar
  - Method Badges แสดงประเภทอัลกอริทึมและวิธีการแปลงที่ระบบเลือกใช้

### Section 3: Technical Details (Collapsible Expander)
ส่วนแสดงผลเชิงลึกทางเทคนิคสำหรับตรวจสอบและประเมินผลขั้นตอนการทำงานของอัลกอริทึม (เปิด/ปิดได้เพื่อความสบายตา):
- **[Preprocessing] Canny Edge Map:** แสดงผลลัพธ์แผนที่ขอบภาพหลังผ่าน Gaussian Blur และ Morphological Closing
- **[Feature Matching] Matches Visualization:** แสดงเส้นโยงการจับคู่จุดเด่นระหว่างภาพต้นฉบับกับภาพผลลัพธ์ระนาบ
- **[RANSAC] Inlier / Outlier Visualization:** แสดงจุดสีเขียว (Inliers ที่ยอมรับ) เปรียบเทียบกับจุดสีแดง (Outliers ที่ถูกคัดทิ้ง)
- **[Homography] Transformation Information:** แสดงเมทริกซ์การแปลง $H_{3 \times 3}$ ในรูปแบบสมการคณิตศาสตร์ LaTeX พร้อมตารางพิกัด 4 มุมจริงในหน่วยพิกเซล

---

## กระบวนการทำงาน (End-to-End Pipeline)

```text
[Input Image] ───> Preprocessing (Resize / Gray / Gaussian Blur 5x5)
                         │
                         ▼
                   Adaptive Canny Edge Detection
                         │
                         ▼
                   Morphological Closing (5x5)
                         │
                         ▼
            Contour & 4-Corner Localization
         (Scoring: Area Ratio & A4 Aspect Ratio)
         (Fallback: minAreaRect / Full Frame)
                         │
                         ▼
               Corner Point Ordering
              (TL, TR, BR, BL Sorting)
                         │
                         ▼
         Feature Extraction (SIFT / ORB)
                         │
                         ▼
         Feature Matching (BFMatcher / FLANN)
                         │
                         ▼
            Lowe's Ratio Test Filtering
                         │
                         ▼
      RANSAC & Homography Estimation (H Matrix)
      + Multi-Rotation Orientation Validation
                         │
                         ▼
       Perspective Warping (cv2.warpPerspective)
                         │
                         ▼
             [Final A4 Rectified Output]
```

### รายละเอียดฟังก์ชันใน Pipeline

| ขั้นตอน | รายละเอียดการทำงาน | ฟังก์ชัน | ไฟล์ต้นทาง |
|---|---|---|---|
| 1. Preprocess | ปรับขนาดภาพ (Max Dim 1280), แปลง Grayscale, ลดสัญญาณรบกวนด้วย Gaussian Blur (5x5) | `preprocess()` | `src/preprocessing.py` |
| 2. Edge Detection | ตรวจจับขอบภาพด้วย Canny แบบ Adaptive Threshold อิงค่า Median และทำ Morphological Closing | `detect_edges()` | `src/detection.py` |
| 3. Corner Detection | ค้นหา Contour เอกสาร คัดเลือกด้วยคะแนนพื้นที่และสัดส่วน A4 พร้อมระบบ Fallback สองระดับ | `detect_document()` | `src/detection.py` |
| 4. Point Ordering | จัดเรียงพิกัด 4 จุดให้อยู่ในลำดับสากล: Top-Left, Top-Right, Bottom-Right, Bottom-Left | `order_points()` | `src/detection.py` |
| 5. Feature Extraction | สกัด Keypoints และคำนวณ Descriptors ด้วย SIFT หรือ ORB | `extract_features()` | `src/features.py` |
| 6. Feature Matching | จับคู่จุดเด่นด้วย BFMatcher หรือ FLANN ผ่าน KnnMatch (k=2) | `match_features()` | `src/features.py` |
| 7. Ratio Test | กรองคู่จุดที่กำกวมออกด้วยเกณฑ์ Lowe's Ratio Test ตามค่า Threshold ที่กำหนด | `apply_ratio_test()` | `src/features.py` |
| 8. Robust Estimation | คำนวณ Homography Matrix พร้อมขจัด Outliers ด้วย RANSAC (Threshold 5.0, Confidence 0.995) | `compute_homography()` | `src/geometry.py` |
| 9. Exact Transform & Orientation Check | คำนวณ Transformation Matrix 4 จุด ตรวจสอบทิศทางการหมุนเพื่อป้องกันภาพกลับด้านหรือมืดดำ | `corners_to_homography()` | `src/geometry.py` |
| 10. Perspective Warp | ทำ Perspective Transform ภาพเข้าสู่ขนาดและสัดส่วนมาตรฐาน A4 | `warp_perspective()` | `src/geometry.py` |

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
│   ├── geometry.py                # ฟังก์ชันคำนวณ Homography, RANSAC, Multi-Rotation Check, และ Warp Perspective
│   └── utils.py                   # ฟังก์ชันวาดเส้นกรอบมุม, แสดงคู่จุด Match, Inliers/Outliers, และแปลง Format ภาพ
└── tests/
    └── sample_images/             # ชุดภาพตัวอย่างสำหรับทดสอบระบบ
        ├── test1_angled.jpg       # ภาพเอกสารมุมเอียงทั่วไป
        ├── test2_extreme_angle.jpg# ภาพเอกสารมุมเอียงองศาสูง (Perspective จัด)
        └── test3_cluttered_bg.jpg # ภาพเอกสารวางบนโต๊ะที่มีสิ่งของรบกวนในฉากหลัง
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
1. `test1_angled.jpg`: ภาพเอกสารบนระนาบที่มีมุมเอียงเล็กน้อยถึงปานกลาง
2. `test2_extreme_angle.jpg`: ภาพเอกสารที่มีมุมมองเฉียงองศาสูง ทดสอบความทนทานของ Perspective Rectification
3. `test3_cluttered_bg.jpg`: ภาพเอกสารที่มีวัตถุ พื้นผิว และสิ่งของรบกวนในฉากหลัง ทดสอบความแม่นยำของ Contour Scoring และ RANSAC Inlier Filtering



---

## ข้อมูลรายวิชาและลิขสิทธิ์ (Course Information & License)
- **รายวิชา:** CP461 Computer Vision
- **ลิขสิทธิ์:** สำหรับใช้ประกอบการศึกษาและการประเมินผลโครงงานในรายวิชา CP461
