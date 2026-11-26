# VISION SWINE — Dataset-2: Pig Weight Estimation Model Development

**คู่มือเริ่มต้นและเอกสารประกอบ Source Code สำหรับส่งมอบงาน**  
เวอร์ชันเอกสาร: 1.0 | 26 พฤศจิกายน 2569  
ตำแหน่งที่แนะนำ: `Estimation/Dataset-2/README.md`

> **ขอบเขตของงานนี้:** การเตรียมข้อมูล การฝึก YOLO Segmentation การประมวลผลภาพ การสกัดคุณลักษณะ การฝึกและประเมินโมเดลทำนายน้ำหนักสุกรจาก **Dataset-2** เท่านั้น **ไม่รวม** Dataset-1, การนำโมเดลไปติดตั้งบน Jetson Nano, การเชื่อมกล้อง, Dashboard, Backend หรือ Deployment
>
> README นี้อ้างอิง Source Code/โครงสร้างไฟล์จาก `Estimation_Small_20261123_115720.zip` และรายการ Libraries ที่ตรวจจาก `.venv` ของ Dataset-2 เมื่อวันที่ 23 พฤศจิกายน 2569 ไฟล์ ZIP ดังกล่าวเก็บภาพ *ตัวอย่าง* เพียงบางส่วน ไม่ใช่ชุดข้อมูลเต็มสำหรับฝึกซ้ำ

## สารบัญ

1. [ภาพรวมโครงการ](#1-ภาพรวมโครงการ)
2. [ข้อมูลและจำนวนภาพ](#2-ข้อมูลและจำนวนภาพ)
3. [โครงสร้างโฟลเดอร์](#3-โครงสร้างโฟลเดอร์)
4. [Python และ Libraries](#4-python-และ-libraries)
5. [ขั้นตอน A — ฝึก YOLO Segmentation](#5-ขั้นตอน-a--ฝึก-yolo-segmentation)
6. [ขั้นตอน B — สร้างชุดข้อมูลประเมินน้ำหนัก](#6-ขั้นตอน-b--สร้างชุดข้อมูลประเมินน้ำหนัก)
7. [ขั้นตอน C — ฝึกและประเมินโมเดลน้ำหนัก](#7-ขั้นตอน-c--ฝึกและประเมินโมเดลน้ำหนัก)
8. [การรันโปรแกรมบน Windows](#8-การรันโปรแกรมบน-windows)
9. [ข้อควรตรวจสอบและปัญหาที่อาจพบ](#9-ข้อควรตรวจสอบและปัญหาที่อาจพบ)

---

## 1. ภาพรวมโครงการ

งาน Dataset-2 มี **สองสายงาน** ซึ่งใช้ข้อมูลต่างกันแต่เชื่อมโยงกัน:

**A. ฝึกโมเดลแยกพื้นที่สุกร (YOLO Segmentation)**

```text
RGB_3394 + MASK_3394 (Ground Truth)
    → ตรวจสอบภาพ/Mask → สร้าง Polygon Labels
    → แบ่ง Train / Val / Test → Transfer Learning YOLO
    → best.pt + ผลการประเมิน Segmentation
```

**B. พัฒนาโมเดลประเมินน้ำหนักสุกร (Weight Estimation)**

```text
RGB_9579 + ข้อมูลน้ำหนักจากชื่อโฟลเดอร์
    → YOLO Segmentation (best.pt)
    → Predicted Masks → Refined Masks → Target Mask
    → QC / ตัดภาพที่ไม่เหมาะสม
    → Raw Skeleton → Main Path Skeleton
    → สกัด 8 Features + จับคู่ Ground Truth
    → model_dataset.xlsx
    → Linear / Polynomial / ANN / Polynomial ANN
    → ผลเปรียบเทียบ Split / GroupKFold / LOGO
```

> ภาพ RGB สำหรับฝึก YOLO และภาพ RGB ที่ใช้สร้างข้อมูลประเมินน้ำหนักเป็น **คนละกลุ่มข้อมูลภายใน Dataset-2** ไม่ควรนำจำนวนภาพมาบวกรวมแล้วเรียกว่า “จำนวนภาพที่ฝึกโมเดลน้ำหนัก”

## 2. ข้อมูลและจำนวนภาพ

| รายการ | จำนวนที่บันทึก | ความหมาย |
|---|---:|---|
| `RGB_9579` | 9,578 ภาพ | ภาพ RGB ที่ใช้เป็นต้นทางของสายงานประเมินน้ำหนัก |
| `RGB_3394` | 3,073 ภาพ | ภาพ RGB ที่จับคู่กับ Mask สำหรับเตรียม YOLO Segmentation |
| `MASK_3394` | 3,073 ภาพ | Ground Truth Mask คู่กับภาพ RGB |
| YOLO Train | 2,159 ภาพ | ชุดฝึก Segmentation |
| YOLO Validation | 453 ภาพ | ชุดตรวจสอบระหว่างพัฒนา |
| YOLO Test | 461 ภาพ | ชุดทดสอบ Segmentation |
| `output/segmentation/masks` | 9,559 ภาพ | Mask ที่ทำนายได้และถูกบันทึก |
| `output/segmentation/refined_masks` | 9,559 ภาพ | Mask หลังปรับให้เหลือ Contour หลัก |
| `output/segmentation/target_mask` | 9,556 ภาพ | Mask ของสุกรเป้าหมายที่เลือกไว้ |
| `raw_skeleton` | 9,313 ภาพ | Skeleton ที่สร้างจาก Mask ของภาพที่ผ่านขั้นตอนตรวจสอบ |
| `main_path_skeleton` | 9,313 ภาพ | Branch หลักที่เลือกจาก Skeleton |
| `model_dataset.xlsx` | 9,313 แถว / 73 `Pig_ID` | ข้อมูลคุณลักษณะ + น้ำหนักจริงสำหรับทดลองโมเดล |

จำนวนข้างต้นมาจากการตรวจไฟล์และบันทึกจำนวนภาพ **ก่อนลดขนาด ZIP เพื่ออัปโหลด** (`dataset_counts.csv`) ไม่ใช่การนับภาพที่เหลือใน ZIP ตัวอย่าง ปริมาณภาพลดลงในแต่ละช่วง **ไม่ได้ยืนยันสาเหตุการคัดออกแบบรายภาพ** ต้องอ้างรายงาน QC/ชื่อไฟล์จริงประกอบ

ยอดรวมภาพในโฟลเดอร์ของ Dataset-2 ที่ตรวจนับได้คือ **80,463 ไฟล์ภาพ** ซึ่งมีภาพต้นฉบับและผลลัพธ์ระหว่างทางหลายสำเนา **ไม่ใช่ 80,463 ภาพที่ไม่ซ้ำ** หากต้องดูจำนวนภาพรายโฟลเดอร์ย่อย ให้ใช้ `Dataset2_Folder_Inventory.csv` (บัญชี 694 โฟลเดอร์จากชุดตรวจนับเดิม) ซึ่งเป็นเอกสารแนบแยกจาก README

## 3. โครงสร้างโฟลเดอร์

```text
Dataset-2/
├── README.md                           # ไฟล์นี้
├── requirements_environment.txt        # pip freeze จาก .venv เดิม (แนบจากเครื่องผู้พัฒนา)
├── .venv/                              # Python virtual environment บนเครื่องพัฒนา; ไม่แนะนำให้ส่งผ่าน Git
├── dataset/
│   ├── PigRGB-Weight/
│   │   ├── RGB_9579/                   # ภาพ RGB สำหรับสร้างชุดข้อมูลน้ำหนัก (fold*/กลุ่มสุกร/ภาพ)
│   │   ├── RGB_MASK_3394/
│   │   │   ├── RGB_3394/               # ภาพ RGB สำหรับฝึก Segmentation
│   │   │   └── MASK_3394/              # Ground Truth Mask คู่กับ RGB
│   │   └── transfer_learning_dataset/
│   │       ├── raw_labels/            # YOLO Polygon Labels ที่สร้างจาก Mask
│   │       ├── segmentation_visualization/  # ภาพตรวจ Polygon Labels
│   │       ├── images/{train,val,test}/
│   │       ├── labels/{train,val,test}/
│   │       └── data.yaml              # อ้างอิงตำแหน่งข้อมูลและชื่อคลาส pig
│   ├── segmentation_training/RGB/      # ภาพอีกกลุ่มหนึ่งที่เก็บใน Dataset-2
│   ├── ภาพผิดหนัก/                      # ตัวอย่าง/รายการภาพที่ใช้แยกออกในขั้น QC
│   └── Weight_Ground_Truth.xlsx        # ตารางภาพ / Pig_ID / น้ำหนักจริง
├── model/
│   ├── weights22k.pt                   # โมเดลตั้งต้นที่สคริปต์ Transfer Learning โหลด
│   ├── best.pt                         # โมเดล Segmentation ที่สายงานประเมินน้ำหนักโหลด
│   └── ...                             # ไฟล์โมเดลอื่นที่พบในโฟลเดอร์ (ไม่ใช่ทุกไฟล์ถูกใช้โดยสคริปต์หลัก)
├── source/
│   ├── transfer_learning/              # สคริปต์ 00–09 สำหรับฝึกและทดสอบ YOLO
│   └── weight_estimation/
│       ├── config.py                   # Path, Features, Hyperparameters
│       ├── 01_create_weight_ground_truth.py
│       ├── 02_segmentation.py
│       ├── 03_segmentation_qc.py
│       ├── 04_feature_extraction.py
│       ├── 05_merge_datase.py          # ชื่อไฟล์สะกดแบบนี้ในโปรเจกต์จริง
│       └── *_model_*.py                # สคริปต์ทดลอง Regression/ANN
└── output/
    ├── transfer_learning/             # ผลการฝึก YOLO / checkpoints ตามการตั้งค่า
    ├── segmentation/
    │   ├── masks/                     # Predicted Masks
    │   ├── refined_masks/             # Masks หลังเลือก Contour หลัก
    │   ├── target_mask/               # Masks ที่เลือกเป็นสุกรเป้าหมาย
    │   ├── segmentation_review/       # ภาพ overlay สำหรับตรวจสอบด้วยสายตา
    │   └── segmentation_usable_dataset.xlsx
    ├── feature_extraction/
    │   ├── raw_skeleton/              # Skeleton ทั้งโครงร่าง
    │   ├── main_path_skeleton/        # ภาพ Branch ที่เลือกเป็นเส้นหลัก
    │   ├── feature_dataset.xlsx       # 8 Features ของแต่ละภาพ
    │   └── model_dataset.xlsx         # Features + Weight + Pig_ID
    └── estimation_model/              # โมเดล Scaler Transformer และผลทดลอง แยกตามวิธี
```

**ตัวอย่างชนิดภาพตามลำดับ**

- `RGB_9579`: ภาพสุกรต้นฉบับ
- `masks`: ภาพ Binary Mask ที่ทำนายจาก YOLO
- `refined_masks`: เก็บบริเวณ Contour ภายนอกที่ใหญ่ที่สุดของแต่ละ Mask เพื่อลดส่วนย่อยรบกวน
- `target_mask`: เลือก Mask ของสุกรเป้าหมายหนึ่งรายจากหลาย Mask โดยใช้คะแนน
- `segmentation_review`: ภาพสำหรับตรวจผลการแยกพื้นที่
- `raw_skeleton`: ลด Mask ให้เหลือเส้นโครงกระดูก
- `main_path_skeleton`: เลือก Branch ยาวที่สุดจาก Skeleton เพื่อใช้คำนวณคุณลักษณะ

**ข้อควรระวังในการดูภาพ:** อย่านำภาพจากคนละชื่อไฟล์/คนละเฟรมไปจัดวางเป็น “ก่อน–หลัง” โดยไม่มีหมายเหตุ ภาพ Mask บางรายการอาจเหมือนกันระหว่างขั้นตอน เพราะ Refinement ไม่ได้เปลี่ยนทุกภาพ

## 4. Python และ Libraries

ตรวจสอบจาก Environment ปัจจุบันของ `Dataset-2/.venv` เมื่อ 23 พฤศจิกายน 2569:

- **Python:** `3.13.14`
- **ผล `pip check`:** `No broken requirements found.`

| Library | Version ที่ตรวจพบ | ใช้สำหรับ |
|---|---|---|
| `ultralytics` | `8.4.115` | YOLO Segmentation / Training / Validation |
| `torch` | `2.11.0+cu128` | Backend ของ YOLO/PyTorch |
| `torchvision` | `0.26.0+cu128` | เครื่องมือร่วมกับ PyTorch |
| `opencv-python` | `4.10.0.84` | อ่านภาพ, Contour, Mask |
| `numpy` | `2.4.6` | คำนวณอาร์เรย์ภาพและข้อมูลตัวเลข |
| `pandas` | `3.0.5` | Excel / DataFrame / รวมชุดข้อมูล |
| `openpyxl` | `3.1.5` | อ่านและเขียน Excel |
| `scikit-image` | `0.26.0` | `skeletonize` |
| `skan` | `0.13.1` | วิเคราะห์ Skeleton/Branch |
| `scikit-learn` | `1.9.0` | Regression, Scaling, Split, Evaluation |
| `tensorflow` | `2.21.0` | ANN |
| `keras` | `3.15.1` | Neural-network API ที่มีใน Environment |
| `joblib` | `1.5.3` | บันทึกโมเดลและตัวแปลงข้อมูล |
| `matplotlib` | `3.11.1` | กราฟ |
| `seaborn` | `0.13.2` | กราฟและการแสดงผลการทดลอง |

เวอร์ชันข้างต้นเป็น **เวอร์ชันที่ติดตั้ง ณ วันที่ตรวจสอบ** ไม่สามารถพิสูจน์ย้อนหลังว่าเหมือนเวอร์ชันวันฝึกเดิมทั้งหมด ควรแนบ `requirements_environment.txt` จริงจากเครื่องผู้พัฒนาเพื่อเก็บข้อมูลแพ็กเกจทั้งหมด (รวม Dependencies ทางอ้อม)

### ติดตั้ง Environment ใหม่ (Windows PowerShell)

เปิด PowerShell ในโฟลเดอร์ `Dataset-2` และตรวจว่ามี Python 3.13 ก่อน:

```powershell
py -3.13 --version
py -3.13 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r ".\requirements_environment.txt"
& $py -m pip check
```

หากมี `.venv` เดิมที่ใช้งานได้ **ไม่จำเป็นต้องสร้างใหม่** ให้กำหนด `$py` แล้วตรวจเวอร์ชันได้เลย:

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py --version
& $py -m pip check
& $py -m pip freeze > ".\requirements_environment.txt"
```

> `requirements_environment.txt` ต้องคัดลอกมาจาก Environment ของ Dataset-2 จริงก่อนสั่งติดตั้งบนเครื่องใหม่ การติดตั้งแพ็กเกจ CUDA เช่น `torch==...+cu128` ขึ้นอยู่กับ OS, GPU, CUDA และแหล่ง wheel ที่รองรับ ไม่รับประกันว่าจะติดตั้งได้ทุกเครื่อง ควรตรวจ GPU และแพ็กเกจให้เข้ากันก่อน

## 5. ขั้นตอน A — ฝึก YOLO Segmentation

| ลำดับ | Source Code | Input และหลักการ | Output / สิ่งที่ตรวจ |
|---|---|---|---|
| 00 | `00_check_base_model.py` | เปิด `model/weights22k.pt` เพื่อตรวจชนิดโมเดลและคลาส | สรุปโมเดลตั้งต้นใน Console |
| 01 | `01_check_dataset.py` | ตรวจความถูกต้องของภาพ RGB และคู่ Mask | รายงานจำนวน/ภาพผิดปกติ |
| 02 | `02_prepare_segmentation_dataset.py` | Threshold Ground Truth Mask → Contour → Normalized Polygon | `raw_labels/`, `segmentation_visualization/` |
| 03 | `03_check_segmentation_label.py` | ตรวจโครงสร้าง Polygon Label และพิกัด | รายงาน Label Validation |
| 04 | `04_split_dataset.py` | แบ่งกลุ่มโฟลเดอร์ด้วย seed `42` สัดส่วน `70/15/15` | `images/` และ `labels/` ของ train, val, test |
| 05 | `05_check_dataset_split.py` | ตรวจคู่ Image/Label และข้อมูลหลัง Split | สรุปความถูกต้องแต่ละ Split |
| 06 | `06_create_data_yaml.py` | เขียน Path ข้อมูลและคลาส `0: pig` | `transfer_learning_dataset/data.yaml` |
| 07 | `07_train_model.py` | YOLO Transfer Learning จาก `weights22k.pt` | Checkpoint และผลฝึก |
| 08 | `08_check_trained_model.py` | ตรวจโมเดลที่ฝึกแล้ว | ชนิด Task/Classes |
| 09 | `09_test_model.py` | ประเมินโมเดลกับ `test` split | ค่าประเมิน Segmentation |

ค่าที่กำหนดไว้ใน `07_train_model.py`:

| Parameter | ค่าในโค้ด |
|---|---|
| `epochs` | 20 |
| `imgsz` | 960 |
| `batch` | 2 |
| `device` | 0 (CUDA GPU) |
| `workers` | 0 |
| `rect` | `True` |
| `pretrained` | `True` |

**หมายเหตุ:** `06_create_data_yaml.py` เขียน `path` แบบ Absolute Path หากย้ายเครื่อง/โฟลเดอร์ให้รันใหม่ สคริปต์ `08_check_trained_model.py` คาดหวัง `output/transfer_learning/pig_segmentation_test/weights/best.pt` แต่ `09_test_model.py` ระบุ `runs/segment/output/transfer_learning/pig_segmentation_test/weights/best.pt` จึง **ต้องตรวจและทำให้ Path ตรงกับตำแหน่งไฟล์ที่มีจริงก่อนทดสอบ** ไม่ควรคาดเดาว่าสคริปต์ทั้งสองชี้ไฟล์เดียวกันโดยอัตโนมัติ

## 6. ขั้นตอน B — สร้างชุดข้อมูลประเมินน้ำหนัก

ไฟล์ `source/weight_estimation/config.py` กำหนดเส้นทางของข้อมูล, `YOLO_CONFIDENCE=0.90`, `YOLO_IMAGE_SIZE=960`, เกณฑ์เลือก Mask และค่าการฝึกโมเดล

| ลำดับ | Source Code | กระบวนการ | Output |
|---|---|---|---|
| 01 | `01_create_weight_ground_truth.py` | อ่าน `fold*`/ชื่อโฟลเดอร์สุกรใน `RGB_9579` แล้วแยกน้ำหนักจากชื่อโฟลเดอร์ สร้าง `Pig_ID` และจับคู่ชื่อภาพ | `dataset/Weight_Ground_Truth.xlsx` |
| 02 | `02_segmentation.py` | โหลด `model/best.pt` → ทำนาย Mask → เลือก Contour หลัก → คำนวณคะแนนเลือก Mask เป้าหมาย → สร้าง Review | `masks/`, `refined_masks/`, `target_mask/`, `segmentation_review/` |
| 03 | `03_segmentation_qc.py` | จับคู่ RGB/Target Mask และตัดภาพที่อยู่ในรายการ `ภาพผิดหนัก` ออก | `segmentation_usable_dataset.xlsx` |
| 04 | `04_feature_extraction.py` | สกัด Features จาก Target Mask ของภาพที่ผ่าน QC พร้อมสร้าง Skeleton | `raw_skeleton/`, `main_path_skeleton/`, `feature_dataset.xlsx` |
| 05 | `05_merge_datase.py` | ทำชื่อภาพให้ตรงรูปแบบ แล้ว `inner join` Features กับ Ground Truth | `model_dataset.xlsx` |

### 6.1 การเลือก Target Mask จากโค้ดจริง

`02_segmentation.py` เลือก Mask หลัง Refinement ด้วยขั้นตอนดังนี้:

1. แปลง Mask เป็น Binary แล้วเก็บ Contour ภายนอกที่มีพื้นที่มากที่สุดของ **แต่ละ Mask**
2. ไม่พิจารณา Mask ที่มีพื้นที่น้อยกว่า `MIN_MASK_AREA = 3000` พิกเซล
3. คำนวณคะแนน = **Area (สูงสุด 40) + Center (สูงสุด 30) + Border (20 หรือ 0) + Completeness (10 หรือ 0)**
4. เลือก Mask ที่มีคะแนนสูงสุดเป็น `target_mask`

ค่าที่เกี่ยวข้องใน `config.py`: `BORDER_MARGIN=20`, `AREA_SCORE_SCALE=600`, `MAX_AREA_SCORE=40`, `COMPLETENESS_THRESHOLD=0.60` หากไม่มี Mask ผ่านเกณฑ์ อาจไม่เกิด Target Mask สำหรับภาพนั้น

### 6.2 คุณลักษณะ (Features) ทั้ง 8 ตัวแปร

| Feature | วิธีคำนวณจากโค้ด | หน่วย/ข้อสังเกต |
|---|---|---|
| `Area` | นับพิกเซลที่เป็น Mask | pixel² (จำนวนพิกเซล) |
| `Perimeter` | ความยาว Contour ของ Mask | pixel |
| `Bounding_Box_Length` | ด้านยาวของกรอบ `minAreaRect` | pixel |
| `Bounding_Box_Width` | ด้านสั้นของกรอบ `minAreaRect` | pixel |
| `Deviation` | ระยะจาก Centroid ของสุกรถึงกึ่งกลางภาพ หารด้วยระยะสูงสุดจากกึ่งกลางถึงมุมภาพ | ไม่มีหน่วย |
| `Skeleton_Length` | ความยาว `branch-distance` ของ Branch ที่เลือก | pixel |
| `Straight_Distance` | ระยะเส้นตรงระหว่างปลายของ Branch ที่เลือก | pixel |
| `Straightness` | `Straight_Distance / Skeleton_Length` | ไม่มีหน่วย |

**ลำดับ Skeleton:** `skeletonize(binary_mask)` → `skan.Skeleton` + `summarize` → เลือก **Branch ที่มี `branch-distance` ยาวที่สุด** แล้วบันทึกเป็น `main_path_skeleton` ซึ่งไม่ใช่การหาเส้นทางยาวที่สุดของกราฟ Skeleton ทั้งหมดโดยตรง และไม่รับประกันว่าตรงกับแนวแกนลำตัวของสุกรทุกภาพ

`model_dataset.xlsx` ควรมีคอลัมน์ Features ทั้ง 8 ร่วมกับ `Image`, `Pig_ID` และ `Weight` เพื่อให้สคริปต์ทดลองสามารถอ่านต่อได้

## 7. ขั้นตอน C — ฝึกและประเมินโมเดลน้ำหนัก

ข้อมูลเข้า: `output/feature_extraction/model_dataset.xlsx`  
`X` = Features 8 ตัวแปร, `y` = `Weight` (กิโลกรัม), `Pig_ID` = กลุ่มสำหรับประเมินแบบแยกสุกร

| โมเดล | Random Split | GroupKFold | LOGO |
|---|---|---|---|
| Linear Regression | `linear_regression_model__split.py` | `linear_regression_model_kfold.py` | `linear_regression_model_logo.py` |
| Polynomial Regression | `polynomial_regression_model__split.py` | `polynomial_regression_model_kfold.py` | `polynomial_regression_model__logo.py` |
| ANN | `ann_model_split.py` | `ann_model_kfold.py` | `ann_model_logo.py` |
| Polynomial ANN | `polynomial_ann_model_split.py` | `polynomial_ann_model_kfold.py` | `polynomial_ann_model_logo.py` |

- **Random Split:** แบ่งระดับรายการภาพแบบสุ่ม (`TEST_SIZE=0.15`, `RANDOM_STATE=42`) ใช้ประเมินในชุดแบ่งสุ่ม แต่ภาพของสุกรตัวเดียวกันอาจอยู่คนละส่วน จึงไม่ควรสรุปว่าเป็นการทดสอบกับสุกรที่ไม่เคยเห็น
- **GroupKFold:** แบ่งกลุ่มโดย `Pig_ID` (`KFOLD_SPLITS=5`) เพื่อประเมินกับกลุ่มสุกรที่แยกกันในแต่ละ Fold
- **LOGO:** Leave-One-Group-Out โดยแยกสุกรเป้าหมายออกเป็นชุดทดสอบทีละกลุ่ม

ค่าตั้งต้น ANN ใน `config.py`: `HIDDEN_LAYER_1=16`, `HIDDEN_LAYER_2=8`, `LEARNING_RATE=0.001`, `EPOCHS=3000`, `BATCH_SIZE=4` โดยรายละเอียด EarlyStopping และการแบ่ง Validation ให้ตรวจในสคริปต์ของการทดลองนั้น ๆ

ผลการทดลองจัดเก็บภายใต้ `output/estimation_model/` แยกเป็นโฟลเดอร์ `linear_regression_*`, `polynomial_regression_*`, `ann_*`, `polynomial_ann_*` ตาม Split/KFold/LOGO ภายในอาจมี `*.xlsx`, `*.joblib`, `*.keras`, `fold_models/` และกราฟ; **โมเดลที่ใช้คู่กับ Scaler/Polynomial Transformer ต้องเก็บไฟล์ประกอบของการทดลองเดียวกัน** ห้ามสลับข้าม Fold หรือข้ามโมเดล

**ตัวชี้วัด:** MAE และ RMSE มีหน่วยกิโลกรัม (ยิ่งต่ำยิ่งดี); R² เป็นตัวชี้วัดความสามารถอธิบายความแปรปรวน ไม่ใช่ “เปอร์เซ็นต์ความแม่นยำ” โดยตรง และผล Random Split / GroupKFold / LOGO ต้องรายงานแยกกัน

## 8. การรันโปรแกรมบน Windows

**รันจากโฟลเดอร์ `Dataset-2` เสมอ** เพราะสคริปต์ Transfer Learning ใช้ Relative Paths เช่น `dataset/...` และ `model/...`:

```powershell
cd 'C:\Users\jewwl\Desktop\project\Estimation\Dataset-2'
$py = '.\.venv\Scripts\python.exe'
& $py --version
& $py -m pip check
```

> คำสั่งต่อไปเป็น **ตัวอย่างสำหรับรันกับ Dataset เต็ม** ไม่ควรรันทั้งหมดกับ ZIP ย่อที่มีเพียงภาพตัวอย่าง และสคริปต์บางตัวจะสร้าง/เขียนทับไฟล์ผลลัพธ์ แนะนำสำรอง `dataset/`, `model/`, `output/` ก่อนรันซ้ำ

### 8.1 ฝึก YOLO Segmentation ตั้งแต่ต้น

เรียกทีละคำสั่งและตรวจผลของขั้นนั้นก่อนเริ่มขั้นถัดไป:

```powershell
& $py 'source\transfer_learning\00_check_base_model.py'
& $py 'source\transfer_learning\01_check_dataset.py'
& $py 'source\transfer_learning\02_prepare_segmentation_dataset.py'
& $py 'source\transfer_learning\03_check_segmentation_label.py'
& $py 'source\transfer_learning\04_split_dataset.py'
& $py 'source\transfer_learning\05_check_dataset_split.py'
& $py 'source\transfer_learning\06_create_data_yaml.py'
& $py 'source\transfer_learning\07_train_model.py'
& $py 'source\transfer_learning\08_check_trained_model.py'
# ตรวจ/แก้ MODEL_PATH ใน 09_test_model.py ให้ตรงกับ checkpoint จริงก่อนใช้
& $py 'source\transfer_learning\09_test_model.py'
```

### 8.2 สร้าง Weight Estimation Dataset

หากมี `model/best.pt` ที่ใช้ได้อยู่แล้ว **ไม่จำเป็นต้องฝึก YOLO ใหม่ก่อนทุกครั้ง**:

```powershell
& $py 'source\weight_estimation\01_create_weight_ground_truth.py'
& $py 'source\weight_estimation\02_segmentation.py'
& $py 'source\weight_estimation\03_segmentation_qc.py'
& $py 'source\weight_estimation\04_feature_extraction.py'
& $py 'source\weight_estimation\05_merge_datase.py'
```

**การใช้ GPU:** โค้ด `02_segmentation.py` เรียก YOLO ด้วย `device=0` และ `07_train_model.py`/`09_test_model.py` กำหนด `DEVICE=0` ต้องใช้เครื่องที่มี GPU/CUDA ที่พร้อม หรือแก้ค่าการเลือกอุปกรณ์และทดสอบความเข้ากันได้ก่อนรัน ไม่ใช่โค้ดที่สลับ CPU อัตโนมัติในทุกขั้น

### 8.3 ทดลองโมเดลน้ำหนัก

เลือก **หนึ่งไฟล์** เพื่อทำการทดลอง ไม่จำเป็นต้องรันครบ 12 สคริปต์ในครั้งเดียว:

```powershell
# ตัวอย่าง Linear Regression: Random Split
& $py 'source\weight_estimation\linear_regression_model__split.py'

# ตัวอย่าง Polynomial ANN: GroupKFold
& $py 'source\weight_estimation\polynomial_ann_model_kfold.py'
```

ดูผลแต่ละการทดลองใน `output/estimation_model/<ชื่อการทดลอง>/` แล้วบันทึกค่าประเมินพร้อมวิธี Split ที่ใช้เสมอ

## 9. ข้อควรตรวจสอบและปัญหาที่อาจพบ

| ปัญหาหรือข้อควรระวัง | แนวทางตรวจสอบ |
|---|---|
| `python` ไม่พบใน PowerShell | ใช้ `py -3.13` หรือ `& '.\.venv\Scripts\python.exe'` จาก Dataset-2 |
| ไม่พบไฟล์ภาพ/Mask หลังย้ายโปรเจกต์ | เริ่มรันจากโฟลเดอร์ `Dataset-2` และตรวจ Path ใน `config.py`, `data.yaml` |
| `data.yaml` ยังชี้เครื่องเก่า | รัน `06_create_data_yaml.py` ใหม่หลังย้ายโฟลเดอร์ |
| `best.pt` ไม่พบในขั้น 08/09 | ตรวจ Path จริง; โค้ด 08 และ 09 อ้างพาธไม่ตรงกันตาม Source Code ที่ตรวจ |
| YOLO/CUDA ไม่พร้อม | ตรวจไดรเวอร์ GPU, CUDA และเวอร์ชัน PyTorch; บางสคริปต์ระบุ `device=0` |
| จำนวนภาพไม่เท่าชื่อโฟลเดอร์ (`9579`, `3394`) | ใช้จำนวนที่ตรวจจากไฟล์/ตารางจริง ไม่ใช้ตัวเลขในชื่อโฟลเดอร์แทนจำนวน |
| จำนวน Mask/Features ลดลงจาก RGB | ตรวจ `segmentation_review`, `ภาพผิดหนัก`, QC Excel และรายชื่อภาพที่จับคู่ได้ก่อนสรุปสาเหตุ |
| `pip check` ผ่าน แต่โค้ดรันไม่สำเร็จ | `pip check` ตรวจ Dependencies ไม่ได้ทดสอบ Pipeline; ต้องรัน Smoke Test กับข้อมูลจริง |
| นำภาพประกอบคนละเฟรมมาวางเทียบกัน | ตรวจชื่อภาพ/โฟลเดอร์/fold และ frame ให้ตรงกัน |
| โหลด `.joblib` จากแหล่งไม่รู้จัก | ไม่โหลดไฟล์ที่ไม่ได้รับความเชื่อถือ เพราะการโหลดอาจเรียกใช้โค้ดที่ฝังอยู่ในไฟล์ได้ |

---

**สถานะเอกสาร:** สรุปจากไฟล์ที่ตรวจสอบได้ ไม่ได้ยืนยันการรันฝึก/ทดสอบซ้ำใน Environment ใหม่ และไม่รวมการ Deployment ส่วนที่ยังต้องแนบจากเครื่องพัฒนาคือ `requirements_environment.txt` ฉบับจริงและ Dataset เต็มชุด
