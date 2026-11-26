# VISION SWINE
## AI-Based Pig Counting and Weight Estimation System

**ระบบนับจำนวนและประเมินน้ำหนักสุกรด้วยกล้อง AI**

**Project Type:** Prototype  
**Hardware:** NVIDIA Jetson Nano  
**Platform:** Jetson Linux / Ubuntu  
**Last Verified:** 25 November 2026

---

## 1. Project Overview

VISION SWINE เป็นระบบต้นแบบสำหรับตรวจจับ ติดตาม นับจำนวน และประเมินน้ำหนักสุกรด้วยเทคโนโลยี Artificial Intelligence (AI) โดยประมวลผลภาพจากกล้อง USB บน NVIDIA Jetson Nano

ระบบสามารถแสดงภาพสดและข้อมูลการประมวลผลผ่าน Web Dashboard รวมถึงส่งข้อมูลน้ำหนัก เหตุการณ์ และวิดีโอไปจัดเก็บบน NAS เพื่อเรียกดูข้อมูลย้อนหลัง

**ฟังก์ชันหลัก**

1. ตรวจจับและติดตามสุกรจากภาพกล้อง
2. นับจำนวนสุกรด้วยระบบ AI
3. ประเมินน้ำหนักสุกรจากคุณลักษณะของภาพ
4. แสดงภาพสดและสถานะระบบผ่าน Web Dashboard
5. ส่งข้อมูลการประเมินน้ำหนักไปจัดเก็บใน PostgreSQL
6. บันทึกและอัปโหลดวิดีโอไปยัง NAS
7. เชื่อมต่อกล้อง USB ใหม่โดยอัตโนมัติเมื่อกล้องหลุด
8. เริ่ม AI Service อัตโนมัติเมื่อเปิด Jetson Nano

## 2. System Architecture

ระบบแบ่งออกเป็น 3 ส่วนหลัก

**1. Edge AI — Jetson Nano**

รับภาพจากกล้อง USB ประมวลผลด้วย TensorRT และ OpenCV เพื่อตรวจจับ ติดตาม นับจำนวน และประเมินน้ำหนักสุกร

**2. Backend & Storage — NAS**

รับข้อมูลจาก Jetson Nano ผ่าน API จัดเก็บข้อมูลเชิงโครงสร้างใน PostgreSQL และจัดเก็บไฟล์วิดีโอสำหรับเรียกดูย้อนหลัง

**3. Frontend — React Dashboard**

แสดงภาพสด จำนวนสุกร ข้อมูล Hardware สถานะเครือข่าย การตั้งค่า AI และข้อมูลย้อนหลัง

**Data Flow**

`USB Camera → Jetson Nano → AI Processing → API / Video Upload → NAS → Dashboard`

## 3. Hardware and Software Environment

เวอร์ชันต่อไปนี้ตรวจสอบจาก Jetson Nano ที่ใช้งานจริง

| Component | Version |
|---|---|
| Device | NVIDIA Jetson Nano |
| Processor | NVIDIA Tegra X1 |
| Architecture | ARM64 (aarch64) |
| L4T | R32.7.1 |
| JetPack | สอดคล้องกับ 4.6.1 |
| CUDA Runtime | 10.2.300 |
| TensorRT | 8.2.1.8 |
| Python | 3.8.0 |
| pip | 24.0 |
| OpenCV | 4.10.0 |
| NumPy | 1.23.5 |
| PyCUDA | 2022.1 |
| scikit-learn | 1.3.2 |
| scikit-image | 0.21.0 |
| joblib | 1.3.2 |

รายละเอียด Python Packages ทั้งหมดอยู่ในไฟล์ `requirements_environment.txt`

**Important:** TensorRT, CUDA และ Python Binding จำเป็นต้องเข้ากันได้กับ Jetson Linux และ Python ที่ใช้งานอยู่ การติดตั้งจาก `requirements_environment.txt` เพียงอย่างเดียวอาจไม่เพียงพอสำหรับการสร้าง Environment ใหม่

## 4. Project Structure

โครงสร้างหลักของโปรเจกต์

```text
vision_swine_v2/
│
├── src/
│   └── vision_swine/
│       ├── main.py
│       ├── api_client.py
│       ├── config.py
│       ├── detector.py
│       ├── features.py
│       ├── runtime_config.py
│       ├── stream_server.py
│       ├── tracking.py
│       ├── video_processor.py
│       ├── video_recorder.py
│       ├── video_uploader.py
│       └── weight_predictor.py
│
├── config/
│   └── gate.json
│
├── models/
│   ├── weights22k.engine
│   ├── weights22k.onnx
│   ├── weights22k.pt
│   └── regression/
│       ├── linear_regression_split.joblib
│       ├── linear_regression_split_portable.json
│       └── linear_regression_split_scaler.joblib
│
├── web/
│   ├── src/
│   ├── public/
│   ├── dist/
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── System_Setup/
│   ├── AI_Auto_Start/
│   ├── USB_WiFi_Auto_Connect/
│   └── Web_Dashboard_Server/
│
├── test_images/
├── test_videos/
├── outputs/
└── requirements_environment.txt
```

### Source Code Description

| File | Description |
|---|---|
| `main.py` | โปรแกรมหลัก ควบคุมการทำงานของระบบ AI |
| `detector.py` | ตรวจจับสุกรด้วยโมเดล AI |
| `tracking.py` | ติดตามและนับจำนวนสุกร |
| `features.py` | สกัดคุณลักษณะจากภาพ |
| `weight_predictor.py` | ประเมินน้ำหนักสุกร |
| `api_client.py` | ติดต่อ API ของ NAS |
| `stream_server.py` | ให้บริการภาพสดและ API บน Jetson |
| `video_recorder.py` | บันทึกวิดีโอ |
| `video_processor.py` | ประมวลผลและแปลงวิดีโอ |
| `video_uploader.py` | ส่งวิดีโอไป NAS |
| `config.py` | การตั้งค่าระบบ |
| `runtime_config.py` | การจัดการค่าตั้งต้นระหว่างทำงาน |

## 5. Running the AI System

### 5.1 Automatic Startup

ระบบใช้ systemd Service ชื่อ

`vision-swine-ai.service`

เมื่อติดตั้ง Service และเปิดใช้งานแบบ `enabled` ระบบจะเริ่มทำงานอัตโนมัติหลังเปิดบอร์ด

### 5.2 Service Commands

**ตรวจสอบสถานะ**

```bash
sudo systemctl status vision-swine-ai.service --no-pager
```

**เริ่มระบบ AI**

```bash
sudo systemctl start vision-swine-ai.service
```

**หยุดระบบ AI**

```bash
sudo systemctl stop vision-swine-ai.service
```

**Restart ระบบ AI**

```bash
sudo systemctl restart vision-swine-ai.service
```

**ตรวจสอบ Log**

```bash
sudo journalctl -u vision-swine-ai.service -n 100 --no-pager -l
```

### 5.3 Manual Execution

หากต้องการทดสอบผ่าน Terminal ให้หยุด AI Service ก่อน เพื่อป้องกันโปรแกรมสองตัวแย่งใช้งานกล้องเดียวกัน

```bash
cd ~/vision_swine_v2

PYTHONPATH=src .venv/bin/python -m vision_swine.main
```

## 6. System Setup Files

ไฟล์ตั้งค่าที่เตรียมไว้ใน `System_Setup` ประกอบด้วย

**AI Auto Start**

- `vision-swine-ai.service` — กำหนดการทำงานของ AI Service
- `10-startup-delay.conf` — หน่วงเวลาเริ่มโปรแกรม 30 วินาที
- `20-openblas.conf` — กำหนด `OPENBLAS_NUM_THREADS=1`

**USB Wi-Fi Auto Connect**

- `99-usb-wifi.rules` — กฎสำหรับจัดการอุปกรณ์ USB Wi-Fi

**Web Dashboard Server**

- `vision-swine.nginx.conf` — ไฟล์ตั้งค่า Nginx สำหรับให้บริการ Dashboard

ไฟล์เหล่านี้เป็นสำเนาการตั้งค่าที่จัดเตรียมไว้สำหรับส่งมอบ การติดตั้งบนบอร์ดใหม่จำเป็นต้องนำไปวางในตำแหน่งระบบที่ถูกต้องตามคู่มือติดตั้ง

## 7. Dashboard

Frontend พัฒนาด้วย React และ Vite โดยแบ่งเป็นหน้าหลักดังนี้

- Live Dashboard
- AI Settings
- Hardware Status
- Network & Storage
- History & Video Records

ข้อมูลภาพสดและสถานะการประมวลผลมาจาก Jetson Nano ส่วนข้อมูลประวัติและวิดีโอเรียกจากระบบ NAS

ไฟล์หลักของ Frontend อยู่ใน `web/src` และชุดเว็บไซต์ที่ Build แล้วอยู่ใน `web/dist`

## 8. Testing Summary

การทดสอบระบบต้นแบบที่ดำเนินการแล้วประกอบด้วย

- การตรวจจับและประเมินน้ำหนักสุกร
- การแสดงภาพสดบน Dashboard
- การทำงานของ AI Service ต่อเนื่องประมาณ 1 ชั่วโมง
- การเปลี่ยนช่อง USB ของกล้องขณะหยุด Service
- การเชื่อมต่อกล้องอัตโนมัติหลังกล้องหลุด
- การบันทึกและอัปโหลดวิดีโอ H.264 ไป NAS
- การเรียกดูวิดีโอผ่าน NAS API
- การตรวจสอบการบันทึกน้ำหนักใน PostgreSQL

ผลการทดสอบข้างต้นเป็นผลในสภาพแวดล้อมต้นแบบที่ใช้งานอยู่ ไม่ใช่การรับรองความเสถียรตลอด 24 ชั่วโมง

## 9. Prototype Limitations

ระบบปัจจุบันยังมีข้อจำกัดที่ควรพิจารณาก่อนนำไปใช้งานจริง

**การส่งข้อมูลน้ำหนัก**

กำหนด API Timeout ไว้ที่ 10 วินาที และไม่ Retry คำขอเขียนรายการเดิมโดยอัตโนมัติเมื่อเกิด Timeout เพื่อหลีกเลี่ยงความเสี่ยงในการสร้างข้อมูลซ้ำ

จึงยังไม่สามารถรับประกันได้ว่าข้อมูลน้ำหนักทุกเหตุการณ์จะถูกบันทึกครบถ้วน

**การนำไปทดลองใช้งานจริง**

ควรพิจารณาพัฒนาเพิ่มเติม ได้แก่ ระบบ Retry ที่ป้องกันข้อมูลซ้ำ การเก็บข้อมูลรอส่ง การตรวจสอบความสมเหตุสมผลของน้ำหนัก และการทดสอบความเสถียรระยะยาว

**การย้ายระบบไปยังบอร์ดใหม่**

TensorRT Engine และ Python Environment อาจขึ้นอยู่กับ Hardware, CUDA, JetPack และ Python Binding ของเครื่องที่ใช้งาน จึงต้องตรวจสอบความเข้ากันได้ก่อนนำไปติดตั้งใหม่

## 10. Handover Notes

ไฟล์สำหรับส่งมอบควรประกอบด้วย Source Code, โมเดล AI, Config, System Setup, รายการ Libraries และคู่มือการใช้งาน

ก่อนจัดส่ง ควรตรวจสอบและไม่นำรหัสผ่าน Token หรือไฟล์ตั้งค่าการเชื่อมต่อที่มีข้อมูลลับรวมอยู่ในชุดส่งมอบโดยไม่ได้รับอนุญาต

**เอกสารประกอบเพิ่มเติม**

1. Jetson Nano Installation & Environment Setup Guide
2. VISION SWINE User Manual
3. USB Wi-Fi, Camera and Network Troubleshooting Guide

---

**VISION SWINE — Prototype Handover Documentation**

*เอกสารฉบับนี้จัดทำจากโครงสร้างโปรเจกต์และสภาพแวดล้อมที่ตรวจสอบ ณ วันที่ 25 พฤศจิกายน 2026*