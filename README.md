<<<<<<< HEAD
# Document_ScannerAndPerspective_Rectifier
 Document Scanner &amp; Perspective Rectifier: Detect tilted paper corners and rectify to a flat top‐down A4 perspective.
=======
# Smart Document Scanner — CP461

## ภาพรวม

Document Scanner ที่ใช้ Computer Vision แปลงภาพถ่ายเอกสาร (เอียง/perspective ผิดเพี้ยน) ให้เป็นภาพ A4 มองตรงจากด้านบน

**Pipeline:** Preprocessing → Edge Detection → Corner Detection → SIFT/ORB → Feature Matching → RANSAC → Homography → Perspective Warp → A4 Output

---

## Tech Stack

| ส่วน | เทคโนโลยี |
|---|---|
| ภาษา | Python 3.10+ |
| Computer Vision | OpenCV |
| คำนวณ | NumPy |
| ภาพ | Pillow |
| Web App | Streamlit |

---

## โครงสร้างโปรเจกต์

```
document-scanner/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # resize, grayscale, Gaussian blur
│   ├── detection.py        # Canny edge, contour, 4-corner + fallback
│   ├── features.py         # SIFT/ORB + BFMatcher/FLANN + ratio test
│   ├── geometry.py         # RANSAC + homography + warpPerspective
│   └── utils.py            # visualization helpers
└── tests/
    └── sample_images/
```

---

## วิธีรัน Local

```bash
# 1. Clone repo
git clone <repo-url>
cd document-scanner

# 2. ติดตั้ง dependencies
pip install -r requirements.txt

# 3. รัน app
streamlit run app.py
```

จะเปิดที่ `http://localhost:8501` อัตโนมัติ

---

## วิธี Deploy บน Streamlit Community Cloud

1. Push โค้ดขึ้น GitHub (public repo)
2. ไปที่ [share.streamlit.io](https://share.streamlit.io)
3. เชื่อมต่อ GitHub account
4. เลือก repo → branch → `app.py`
5. กด **Deploy** → ได้ public URL

## วิธี Deploy บน Hugging Face Spaces

1. สร้าง Space ใหม่ที่ [huggingface.co/spaces](https://huggingface.co/spaces)
2. เลือก SDK = **Streamlit**
3. อัปโหลดไฟล์ทั้งหมด หรือ push ผ่าน Git:
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git push space main
   ```

---

## Pipeline รายละเอียด

| Step | Function | ไฟล์ |
|---|---|---|
| 1. Resize + Grayscale + Blur | `preprocess()` | `src/preprocessing.py` |
| 2. Canny Edge Detection | `detect_edges()` | `src/detection.py` |
| 3. Contour → 4 Corners | `find_document_contour()` | `src/detection.py` |
| 4. SIFT/ORB Extraction | `extract_features()` | `src/features.py` |
| 5. BFMatcher/FLANN | `match_features()` | `src/features.py` |
| 6. Lowe's Ratio Test | `apply_ratio_test()` | `src/features.py` |
| 7. RANSAC + Homography | `compute_homography()` | `src/geometry.py` |
| 8. Perspective Warp | `warp_perspective()` | `src/geometry.py` |
| 9. A4 Output | `full_pipeline()` | `src/geometry.py` |

---

## Features ของ App

- **Stepper UI**: แสดงขั้นตอน Upload → Detect → Match → Result
- **Settings Sidebar**: เลือก SIFT/ORB, BFMatcher/FLANN, ratio threshold, output resolution
- **4-column visualization**: ต้นฉบับ | Corner detection | Feature matching | A4 result
- **Inlier/Outlier plot**: เขียว = inlier, แดง = outlier
- **Metrics**: จำนวน keypoints, good matches, inliers, confidence %
- **Download**: ดาวน์โหลดผล A4 เป็น PNG
- **Error handling**: แสดงข้อความชัดเจน ไม่ crash

---

## การแบ่งงาน

1. **Image Preprocessing** → `src/preprocessing.py`
2. **Document Detection** → `src/detection.py`
3. **Feature Matching** → `src/features.py`
4. **Homography & Transformation** → `src/geometry.py`
5. **Web App / Integration** → `app.py` + `src/utils.py`
>>>>>>> da0099a (Initial commit: smart document scanner and perspective rectifier)
