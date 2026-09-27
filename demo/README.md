# 📁 CP461 Document Scanner & Perspective Rectifier — Demo Suite & Test Dataset

ชุดภาพเอกสารและข้อมูลตัวอย่างความละเอียดสูง สำหรับทดสอบและใช้ในการนำเสนอ **Live Demo (10 นาที)** ของโปรเจกต์ End-to-End Computer Vision Document Scanner & Perspective Rectifier

---

## 📂 โครงสร้างโฟลเดอร์ (Folder Structure)

```
demo/
├── 01_Standard_Documents/          # เอกสารสำหรับการสาธิตโหมด Auto (Contour Detection)
│   ├── 01_aurora_tax_invoice.jpg       # ใบกำกับภาษี Aurora Tech บนโต๊ะไม้ ถ่ายมุมเฉียงธรรมชาติ
│   ├── 02_apex_consulting_invoice.jpg  # ใบแจ้งหนี้ Apex Consulting บนโต๊ะไม้โบราณ
│   ├── 03_business_contract.jpg        # สัญญาข้อตกลงทางธุรกิจ A4 บนโต๊ะสีเข้ม (Contrast สูง)
│   ├── 04_academic_research_paper.jpg  # เปเปอร์วิชาการ AI/Time-series พร้อมกราฟและสูตรคณิตศาสตร์
│   ├── 05_thai_cmu_certificate.jpg     # หนังสือรับรองการปฏิบัติงาน มช. พร้อมตราครุฑและตรายาง
│   └── 06_thai_tax_invoice_photo.jpg   # ใบเสร็จรับเงิน/ใบกำกับภาษีไทย Siam Vision (A4 คมชัดสูง)
│
├── 02_Reference_Pairs/             # คู่ภาพสำหรับการสาธิตโหมด Reference (SIFT/ORB + RANSAC)
│   ├── pair1_thai_tax_invoice_flat.jpg     # [ภาพตรงอ้างอิง] ใบเสร็จภาษีไทย Siam Vision
│   ├── pair1_thai_tax_invoice_skewed.jpg   # [ภาพถ่ายเอียง] ภาพถ่ายมุมมองเอียงบนโต๊ะไม้
│   ├── pair2_aurora_invoice_flat.jpg       # [ภาพตรงอ้างอิง] Aurora Tax Invoice
│   ├── pair2_aurora_invoice_skewed.jpg     # [ภาพถ่ายเอียง] Aurora Tax Invoice ถ่ายจากกล้องมือถือ
│   ├── pair3_apex_invoice_flat.jpg         # [ภาพตรงอ้างอิง] Apex Consulting Invoice
│   ├── pair3_apex_invoice_skewed.jpg       # [ภาพถ่ายเอียง] Apex Consulting Invoice จากกล้อง
│   ├── pair4_cp461_spec_report_flat.jpg    # [ภาพตรงอ้างอิง] รายงานเทคนิค CP461
│   └── pair4_cp461_spec_report_skewed.jpg  # [ภาพถ่ายเอียง] รายงานเทคนิค CP461 ถ่ายมุมมองเอียง
│
├── 03_Edge_Cases/                  # ชุดภาพทดสอบกรณีขอบเขต & Fallback Mechanisms
│   ├── edge1_heavy_shadow.jpg          # แสงเงาทอดทับเฉียงข้ามตัวเอกสารอย่างรุนแรง (Heavy Shadow)
│   ├── edge2_cluttered_desk.jpg        # โต๊ะทำงานรก มีแก้วกาแฟ แล็ปท็อป ปากกา โพสต์อิท (Clutter)
│   ├── edge3_corner_occluded.jpg       # มุมกระดาษขวาล่างถูกแก้วกาแฟวางทับ (Corner Occlusion)
│   ├── edge4_low_contrast.jpg          # กระดาษสีขาววางบนโต๊ะหินอ่อนสีขาว (Low Contrast)
│   └── edge5_extreme_perspective.jpg   # ภาพถ่ายมุมมองเอียงรุนแรงมาก (>60 องศา)
│
├── verify_demo_suite.py            # สคริปต์อัตโนมัติสำหรับรัน Pipeline ตรวจสอบทุกรูป
└── README.md                       # คู่มือการใช้งานชุดภาพเดโม (เอกสารฉบับนี้)
```

> **หมายเหตุ:** ไฟล์ภาพเดโมทั้งหมดถูกทำสำเนาไว้ที่รูทของโฟลเดอร์ `demo/` ด้วย เพื่อความสะดวกในการคลิกเลือกไฟล์ผ่าน File Dialog บนหน้า Streamlit Web App ในคลิกเดียว

---

## 🎯 ตารางสรุปภาพและการนำไปใช้ในการสาธิต (Demo Matrix)

### 1. หมวดเอกสารมาตรฐาน (Standard Documents — Auto Mode)

| ชื่อไฟล์ | ประเภทเอกสาร | จุดเด่นในการเดโม | ผลลัพธ์ที่คาดหวัง |
|---|---|---|---|
| `01_aurora_tax_invoice.jpg` | ใบกำกับภาษีไอที | ภาพถ่ายมือถือธรรมชาติ มีตารางและตัวเลขชัดเจน | ตรวจจับ 4 มุมอัตโนมัติ ดัดตรงเป็นสัดส่วน A4 |
| `02_apex_consulting_invoice.jpg` | ใบแจ้งหนี้ที่ปรึกษา | ขอบกระดาษชัดเจนบนโต๊ะไม้ มีโลโก้สีฟ้า | Contour แม่นยำ ดัดมุมมองเป็น A4 แนวนอน/ตั้ง |
| `03_business_contract.jpg` | สัญญาธุรกิจ A4 | ความเปรียบต่างสูง (High Contrast) บนโต๊ะสีเข้ม | ตรวจจับ 4 มุมด้วย Contour 100% |
| `04_academic_research_paper.jpg` | บทความวิจัยวิชาการ | เอกสาร 2 คอลัมน์ มีกราฟเส้น Loss และสูตรคณิตศาสตร์ | ดัดตรงและทดสอบฟิลเตอร์ B&W ได้ตัวหนังสือคมกริบ |
| `06_thai_tax_invoice_photo.jpg` | ใบเสร็จภาษีภาษาไทย | ภาษาไทยล้วน มีตาราง 5 แถว ตรายาง และลายเซ็น | ตรวจจับ Contour สมบูรณ์ ดัดเป็น A4 แนวตั้ง |

### 2. หมวดคู่ภาพอ้างอิง (Reference Pairs — SIFT + BFMatcher + RANSAC)

| คู่ภาพ (Pair) | เอกสาร | Good Matches | RANSAC Inliers | คำแนะนำในการพรีเซนต์ |
|---|---|---|---|---|
| **Pair 1: Thai Tax Invoice** | ใบเสร็จภาษีไทย Siam Vision | ~147 คู่ | **120 inliers (81.6%)** | แสดงการจับคู่ภาษาไทยและตารางอย่างแม่นยำ |
| **Pair 2: Aurora Tax Invoice** | ใบกำกับภาษี Aurora | ~180 คู่ | **127 inliers (70.6%)** | แสดง Inlier Matching ข้ามมุมมองกล้องจริง |
| **Pair 3: Apex Consulting** | ใบแจ้งหนี้ Apex | ~313 คู่ | **247 inliers (78.9%)** | จุด Match หนาแน่น Visualization ชัดเจนมาก |
| **Pair 4: CP461 Spec Report** | รายงานเทคนิค CP461 | ~217 คู่ | **171 inliers (78.8%)** | ครอบคลุมไดอะแกรม Flowchart และตารางสเปก |

### 3. หมวดกรณีขอบเขตและระบบสำรอง (Edge Cases & Fallbacks)

| ชื่อไฟล์ | สภาพแวดล้อมที่ท้าทาย | พฤติกรรมของระบบ | ประเด็นที่ใช้พูดในการนำเสนอ |
|---|---|---|---|
| `edge1_heavy_shadow.jpg` | แสงเงาทอดทับเฉียงครึ่งแผ่น | ระบบตรวจพบเงาตัดขอบ จึงสลับเป็น **Fallback** พร้อมแจ้งเตือน | สาธิตระบบแจ้งเตือนที่เป็นมิตร และการปรับมุมแบบ Manual Adjustment |
| `edge2_cluttered_desk.jpg` | โต๊ะทำงานรก มีสิ่งของรอบข้าง | ใช้ **Fallback 1 (minAreaRect)** ตรวจจับด้วยความมั่นใจปานกลาง | แสดงความฉลาดของอัลกอริทึมในการเลือก Bounding Quad |
| `edge3_corner_occluded.jpg` | มุมขวาล่างถูกแก้วกาแฟทับ | ตรวจจับมุมกระดาษส่วนที่เหลือ | สาธิตฟีเจอร์ **Manual Corner Adjustment** ให้ผู้ใช้ลากจุดมุมเอง |
| `edge4_low_contrast.jpg` | กระดาษขาวบนโต๊ะหินอ่อนขาว | Canny Edge ตรวจขอบได้ยาก | สาธิตการใช้ **โหมด Reference (SIFT)** แก้ปัญหาภาพที่ขอบไม่ชัด |
| `edge5_extreme_perspective.jpg` | มุมมองเอียงรุนแรงมาก (>60°) | Perspective Distortion สูง | สาธิตการใช้ SIFT Feature Matching คำนวณ Homography แทน Contour |

---

## 🚀 ลำดับขั้นตอนการนำเสนอ Live Demo 10 นาที (Presentation Flow)

หากต้องการนำเสนอโปรเจกต์ให้ประทับใจและได้คะแนนเต็มในเกณฑ์ 3/3 (Engineering & Demo):

1. **เปิดด้วยโหมด Auto (1.5 นาที):**
   - อัปโหลด `01_aurora_tax_invoice.jpg` หรือ `06_thai_tax_invoice_photo.jpg`
   - ชี้ให้เห็นขั้นตอน 4 ขั้นตอนบน Stepper: Preprocessing → Edge/Contour → Warp → A4
   - แสดงผลลัพธ์ภาพที่ Rectify แล้ว พร้อมปรับฟิลเตอร์ **Magic Color** หรือ **Clean B&W**

2. **สาธิตหัวใจสำคัญของวิชา: โหมด Reference ด้วย SIFT + RANSAC (2.5 นาที):**
   - สลับไปที่โหมด **Reference (2 ภาพ)**
   - เลือกคู่ภาพ `pair1` (ภาษาไทย) หรือ `pair3` (Apex Consulting)
   - แสดงหน้าต่าง **Feature Matching Visualization** (`cv2.drawMatchesKnn`)
   - อธิบายว่าเส้นสีเขียวคือ Inliers จาก RANSAC (มากกว่า 75%) และเส้นสีแดง/จุดที่ตัดทิ้งคือ Outliers จาก Lowe's Ratio Test
   - ชี้ให้เห็น Homography Matrix $H \in \mathbb{R}^{3 \times 3}$ ที่คำนวณได้

3. **สาธิตความทนทานต่อ Edge Cases & Fallback (2 นาที):**
   - อัปโหลด `edge2_cluttered_desk.jpg` แสดงว่าระบบไม่ Crash และตรวจจับเอกสารได้ด้วย minAreaRect
   - อัปโหลด `edge3_corner_occluded.jpg` หรือ `edge1_heavy_shadow.jpg`
   - เปิดโหมด **Manual Corner Adjustment** เพื่อแสดงว่าผู้ใช้สามารถลากเลื่อนพิกัดมุมทั้ง 4 ได้อย่างอิสระตามเกณฑ์ Rubric

4. **ดาวน์โหลดผลลัพธ์ (0.5 นาที):**
   - กดปุ่ม **Download Scanned Document** ดาวน์โหลดผลลัพธ์ความละเอียดสูงกลับมาใช้งาน

---

## 🧪 การรันคำสั่งตรวจสอบอัตโนมัติ (Verification Command)

สามารถทดสอบความพร้อมของภาพทุกรูปในโฟลเดอร์ `demo/` ด้วยคำสั่งเดียวผ่านเทอร์มินัล:

```bash
python -X utf8 demo/verify_demo_suite.py
```

ระบบจะประมวลผลทุกภาพและพิมพ์รายงานสรุปความถูกต้องของทั้งโหมด Auto, โหมด Reference และโหมด Edge Cases ให้ทันที
