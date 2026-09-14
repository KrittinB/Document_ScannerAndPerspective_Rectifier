# CP461 Group Project Spec — Document Scanner & Perspective Rectifier

## 1. โจทย์และเป้าหมาย

สร้าง **end-to-end Computer Vision pipeline** ที่รับภาพถ่ายเอกสาร (กระดาษที่เอียง/มี perspective ผิดเพี้ยน) แล้วแปลงให้เป็นภาพเอกสารที่มองตรงจากด้านบน สัดส่วน A4 (1:1.414) เหมือนสแกน

**ต้องมีครบตามข้อกำหนดวิชา (บังคับ ห้ามข้าม):**
- Keypoint extraction: **SIFT หรือ ORB**
- Feature matching: **BFMatcher หรือ FLANN**
- Geometric transformation: **Homography + RANSAC**

**ผลลัพธ์สุดท้ายต้องเป็น Streamlit Web App ที่ deploy ขึ้น public URL** (Hugging Face Spaces หรือ Streamlit Cloud) เพื่อให้เข้าเกณฑ์ Tier 3 (คะแนนเต็ม engineering 3/3, total max 10/10)

---

## 2. Tech stack

| ส่วน | เทคโนโลยี |
|---|---|
| ภาษา | Python 3.10+ |
| Computer Vision | OpenCV (`opencv-python`) |
| คำนวณ/เมทริกซ์ | NumPy |
| จัดการภาพ | Pillow |
| Web App | Streamlit |
| เก็บโค้ด | Git + GitHub (public repo) |
| Deploy | Streamlit Community Cloud หรือ Hugging Face Spaces |

`requirements.txt`:
```
opencv-python
numpy
Pillow
streamlit
```

---

## 3. Pipeline ที่ต้อง implement (ตามลำดับ)

```
1. Upload Image
2. Image Preprocessing        → resize, grayscale, Gaussian blur
3. Edge Detection              → Canny
4. Contour Detection           → หา contour ที่เป็นกระดาษ
5. Detect 4 Corners            → order เป็น TL, TR, BR, BL
6. Feature Extraction          → SIFT หรือ ORB (keypoints + descriptors)
7. Feature Matching            → BFMatcher / FLANN
8. Lowe's Ratio Test           → กรอง good matches ออกจาก matches ที่ไม่ดี
9. RANSAC                      → กำจัด outliers, หา inlier matches
10. Homography                 → cv2.findHomography()
11. Perspective Transformation → cv2.warpPerspective()
12. A4 Output                  → resize ให้สัดส่วน 1:1.414
```

**หมายเหตุสำคัญ:** อย่าทำแค่ contour → warp เฉยๆ โดยข้าม step 6-9 เพราะ rubric ให้คะแนนหลักตรงส่วน SIFT/ORB + ratio test + RANSAC (ดู section 4)

### รายละเอียดทางเทคนิคที่ต้องมี
- **Corner detection ต้อง robust**: รองรับกรณีขอบกระดาษไม่ชัด, แสงไม่สม่ำเสมอ, พื้นหลังซับซ้อน — ถ้า contour approximation หา 4 จุดไม่ได้ ต้องมี fallback (เช่น convex hull + minAreaRect หรือให้ผู้ใช้ลาก corner เอง)
- **Feature matching ต้องแสดงภาพ visualization** (`cv2.drawMatches` หรือ `cv2.drawMatchesKnn`) ให้เห็น keypoints และเส้นจับคู่จริง ไม่ใช่แค่ใช้ homography จาก corner points เฉยๆ
- **ต้อง handle failure case**: ภาพที่ไม่มีเอกสารชัดเจน, มุมกระดาษถูกบัง, แสงจ้า/มืด ต้องมี error message หรือ fallback ไม่ crash
- **ต้องมี inlier plot**: แสดงว่าจุดไหนถูก RANSAC เลือกว่าเป็น inlier (สีเขียว) vs outlier (สีแดง/เอาออก)

---

## 4. เกณฑ์การให้คะแนน (ต้องเช็คให้ครบ)

| หัวข้อ | คะแนน | สิ่งที่ต้องมี |
|---|---:|---|
| Algorithmic Correctness & Robustness | 4 pts | ใช้ SIFT/ORB ถูกต้อง, descriptor ratio testing, RANSAC, handle failure case/outliers, output คมชัด |
| Engineering & UI Implementation | 3 pts | โค้ดเป็นระเบียบ, UX ดี — **ถูก cap ตาม tier**: Colab only = 1/3, Local app = 2/3, **Deployed public app = 3/3** |
| 10-Minute Presentation & Demo | 3 pts | อธิบาย pipeline ชัดเจน (1.5), live demo กับภาพ edge-case ของจริง (1.0), พูดครบทุกคนในกลุ่ม (0.5) |

**Submission package ที่ต้องส่ง:**
- GitHub repo (clean, มี `requirements.txt` และคำสั่งรันชัดเจน)
- Executable link: Colab notebook link หรือ live deployment URL (จำเป็นถ้าจะได้ Tier 3)
- วิดีโอสาธิต **ยาวพอดี 10 นาที** (ห้ามเกิน) อัปโหลด YouTube (Unlisted) หรือ Google Drive มี voiceover อธิบาย technical decisions + live demo จริง

---

## 5. สเปก Streamlit UI

หน้าตาแอปควรมี flow แบบนี้:

1. **Header**: ชื่อแอป "Smart document scanner" + คำอธิบายสั้นๆ
2. **Stepper/ตัวบอกขั้นตอน**: Upload → Detect → Match → Result (ให้ผู้ใช้เห็นว่าอยู่ขั้นไหน)
3. **Upload area**: `st.file_uploader` รับภาพ, มี drag & drop
4. **แสดงผลแบบ side-by-side**: ใช้ `st.columns` แบ่ง 2-4 ช่อง
   - ภาพต้นฉบับ
   - ภาพที่ตรวจจับ 4 มุมแล้ว (วาดเส้นขอบ + จุดมุม)
   - ภาพ feature matching visualization (เส้นจับคู่ keypoints)
   - ภาพผลลัพธ์ A4 หลัง warp
5. **แสดง confidence / metric**: เช่น จำนวน inlier matches, % confidence ของการหามุม (progress bar)
6. **ปุ่ม action**: "Run scan pipeline" (ปุ่มหลัก เด่นสุด) และ "Download result" / "Reset" (ปุ่มรอง)
7. **จัดการ error**: ถ้าตรวจจับเอกสารไม่ได้ ให้ขึ้นข้อความแจ้งเตือนที่ชัดเจน ไม่ crash

---

## 6. โครงสร้างโปรเจกต์ที่แนะนำ

```
document-scanner/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── README.md                # วิธีรัน + คำอธิบาย pipeline
├── src/
│   ├── preprocessing.py     # resize, grayscale, blur
│   ├── detection.py         # edge, contour, 4-corner detection
│   ├── features.py          # SIFT/ORB extraction + matching + ratio test
│   ├── geometry.py          # RANSAC + homography + warpPerspective
│   └── utils.py             # helper functions, A4 aspect ratio handling
├── tests/
│   └── sample_images/       # ภาพทดสอบ รวม edge case (แสงไม่ดี, มุมบัง ฯลฯ)
└── notebook/
    └── pipeline_demo.ipynb  # เวอร์ชัน Colab (ทางเลือก/สำรอง)
```

---

## 7. การแบ่งงาน 5 คน (อ้างอิงจากที่กลุ่มตกลงไว้)

1. **Image Preprocessing** — resize, grayscale, blur, edge detection
2. **Document Detection** — contour detection, หา 4 corners, robust fallback
3. **Feature Matching** — SIFT/ORB, keypoints/descriptors, matching, ratio test
4. **Homography & Transformation** — RANSAC, homography, perspective transform, A4 output
5. **Web App / Integration / Testing** — Streamlit UI, รวมโค้ดทุกคน, ทดสอบ edge case, เตรียม deploy

---

## 8. Checklist ก่อนส่งงาน

- [ ] Pipeline ครบทุก step ตาม section 3 (ไม่ข้าม SIFT/ORB + ratio test + RANSAC)
- [ ] มี visualization ของ keypoints/matches และ inlier/outlier ใน UI
- [ ] ทดสอบกับภาพ edge case อย่างน้อย 3-5 แบบ (แสงไม่ดี, มุมเอียงมาก, พื้นหลังรก)
- [ ] Deploy ขึ้น public URL สำเร็จ (Streamlit Cloud / Hugging Face Spaces)
- [ ] GitHub repo สะอาด มี `requirements.txt` + README วิธีรัน
- [ ] อัดวิดีโอ 10 นาทีพอดี มี voiceover + live demo + ทุกคนพูด
