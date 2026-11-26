# VISION SWINE — Backend API (FastAPI)

**Developer README · สำหรับผู้รับช่วงพัฒนา Backend**  
**อ้างอิง:** Source Code ใน `Backend_NAS.zip` ชุดส่งมอบ (ตรวจสอบ 26 พฤศจิกายน 2026) · คู่มือหลัก **VISION SWINE NAS Developer Handover Manual v0.8.6**

> เอกสารนี้อธิบายสิ่งที่พบจาก Source Code และ Config ในชุดส่งมอบ ไม่ใช่การรับรองสถานะบริการ ณ เวลาปัจจุบัน ก่อนเปลี่ยนฐานข้อมูลหรือ Deploy ให้ตรวจบริการจริงบน NAS อีกครั้ง  
> **ห้ามใส่รหัสผ่าน, `.env`, Database Dump หรือ Token จริงใน Git, README หรือภาพหน้าจอที่เผยแพร่**

## 1. ขอบเขตและหน้าที่

Backend นี้เป็น **FastAPI** ที่รับ/ส่งข้อมูลจาก Edge AI และให้ Frontend อ่านข้อมูลจาก NAS โดยมีหน้าที่หลักดังนี้

- จัดการรอบประมวลผล AI (`processing_sessions`) ของอุปกรณ์และกล้อง
- บันทึกและอ่านค่าน้ำหนักสุกร (`weight_estimations`)
- บันทึกและอ่านเหตุการณ์นับสุกรเข้า–ออก (`counting_events`) พร้อมเชื่อมผลประเมินน้ำหนัก
- รับไฟล์วิดีโอ MP4, บันทึก Metadata ใน PostgreSQL และส่งไฟล์ให้หน้า Video Recordings Web
- ตรวจสถานะ API / PostgreSQL และพื้นที่จัดเก็บ NAS

**ไม่รวมใน Repository ส่วนนี้:** โมเดล YOLO/การประมวลผลบน Jetson Nano และ Source Code ของ Dashboard หลัก (เป็นส่วนที่พัฒนาแยกกัน)

### ภาพรวมการไหลของข้อมูล

```text
Jetson Nano / Edge AI
   │ POST /sessions, /weights, /counting-events, /videos
   ▼
FastAPI :8002 ───────────► PostgreSQL :5432
   │                         └── session / count / weight / video metadata
   └── MP4 ──────────────► /home/dev/vision-swine/storage/videos/
                                  │
Video Recordings Web :8090 ─ Nginx /videos proxy ─► FastAPI GET /videos, /videos/{id}/file
```

พอร์ตด้านบนเป็นค่าที่กำหนดในไฟล์สคริปต์และ Config ของชุดส่งมอบ; API และ PostgreSQL ตั้งค่าให้ฟังบน `127.0.0.1` ของสภาพแวดล้อม NAS/Container ไม่ได้หมายความว่าทุกพอร์ตเปิดให้เข้าถึงจากภายนอกโดยตรง

## 2. ตำแหน่งโครงการและโครงสร้างไฟล์

**โฟลเดอร์จริงบน NAS:**

```text
/home/dev/vision-swine/
├── management_system/
│   ├── README.md                         # ภาพรวม Backend + Frontend + NAS
│   ├── backend/                           # โฟลเดอร์ที่กำลังอ่าน
│   │   ├── README.md                     # คู่มือนี้
│   │   ├── requirements.txt              # Python dependencies (ดูข้อจำกัดด้านล่าง)
│   │   ├── .env                          # Secret/config ห้ามเผยแพร่
│   │   ├── venv/                         # Environment ที่เคยใช้งานบน NAS
│   │   └── vision_swine_api/
│   │       ├── __init__.py
│   │       ├── main.py                   # FastAPI, CORS, health, include_router
│   │       ├── database.py               # อ่าน .env และสร้าง psycopg connection
│   │       └── routes/
│   │           ├── sessions.py           # เปิด/อ่าน/ปิด Session
│   │           ├── weights.py            # ค่าน้ำหนัก
│   │           ├── counting.py           # เหตุการณ์ IN / OUT
│   │           ├── videos.py             # Upload, list, serve MP4
│   │           └── storage.py            # ขนาดและพื้นที่ NAS
│   └── frontend/video-recordings-web/   # React/Vite; มี README แยก
├── storage/videos/                        # MP4 ที่ระบบจัดเก็บ
├── database/sql/schema_v2_current.sql    # Schema snapshot สำหรับอ้างอิง
├── tools/start-management-api.sh         # สคริปต์เริ่ม API
├── tools/start-all.sh                    # เริ่ม pgweb + API
├── tools/start-pgweb.sh                  # เริ่ม pgweb
└── logs/management-api.log               # Log ของการรัน API ผ่านสคริปต์
```

อ่านต่อ: [README ระบบหลัก](../README.md) · [README Frontend](../frontend/video-recordings-web/README.md) · [Database schema](../../database/sql/schema_v2_current.sql)

**จุดแก้โค้ดที่พบบ่อย:** เพิ่ม API ที่ `routes/*.py` แล้วผูก Router ใน `main.py`; แก้การเชื่อมต่อ DB ใน `database.py`; ตรวจรูปแบบตารางใน SQL Schema ก่อนแก้ Query; แก้การเปิดให้ Frontend เรียก API ใน `main.py` (CORS) และ Config Nginx ของ Frontend ตามกรณี

## 3. เทคโนโลยีและ Dependencies

จาก `requirements.txt` ใน ZIP:

| Package | เวอร์ชันตามไฟล์ | หน้าที่ |
|---|---|---|
| `fastapi` | `0.141.1` | API, validation, routing |
| `uvicorn` | `0.52.4` | ASGI server |
| `psycopg[binary]` | `3.3.5` | PostgreSQL driver |
| `python-dotenv` | `1.2.3` | โหลด Environment Variables |

- Environment ที่บันทึกใน ZIP เป็น **Python 3.12** (`backend/venv/lib/python3.12/`); การ Deploy ใหม่ควรสร้าง `venv` บนเครื่องเป้าหมาย ไม่คัดลอก Environment ข้าม OS/สถาปัตยกรรม
- **ข้อจำกัดที่พบ:** `routes/videos.py` ใช้ `Form` และ `UploadFile` ซึ่งต้องมี `python-multipart`; ใน `requirements.txt` ยังไม่ได้ระบุแพ็กเกจนี้ แต่ใน `venv` ที่ส่งมอบพบ `python-multipart 0.0.32` จึงต้องติดตั้งเพิ่มเมื่อสร้าง Environment ใหม่ และควรเพิ่มลง `requirements.txt` ในรอบแก้ Source Code ถัดไป
- ยังไม่พบชุด Automated Tests ของโค้ดโครงการใน ZIP; ควรเพิ่มทดสอบก่อนขยาย API

## 4. การตั้งค่าฐานข้อมูลและ `.env`

ไฟล์ `vision_swine_api/database.py` ใช้ `load_dotenv()` กับ **Absolute Path** ต่อไปนี้:

```text
/home/dev/vision-swine/management_system/backend/.env
```

ตัวแปรที่โค้ดใช้จริงมี **5 ค่า**:

```dotenv
DATABASE_NAME=<database_name>
DATABASE_USER=<postgres_user>
DATABASE_PASSWORD=<secret_from_admin>
DATABASE_HOST=127.0.0.1
DATABASE_PORT=5432
```

> บรรทัดนี้เป็น **ตัวอย่างค่า placeholder ไม่ใช่ไฟล์ `.env` พร้อมใช้งาน** อย่านำ placeholder ไปทับข้อมูลจริงบน NAS โดยไม่ตรวจสอบก่อน และอย่าเผยแพร่ `.env` ในชุด Source Code สาธารณะ

**เรื่องชื่อฐานข้อมูลที่ต้องระวัง:** `.env` ที่อยู่ใน `Backend_NAS.zip` ซึ่งใช้ตรวจเอกสารระบุ `DATABASE_NAME=vision_swine_v2` ขณะที่ข้อมูลการเชื่อมต่อ pgweb ที่ผู้ส่งมอบให้ภายหลังใช้ชื่อ `vision_swine` — **ห้ามสมมติว่าเป็นฐานเดียวกัน** ตรวจจาก API `GET /health/database` และใน pgweb รันคำสั่งอ่านข้อมูลนี้ก่อนแก้ไขหรือลบข้อมูล:

```sql
SELECT current_database(), current_user, version();
```

การเชื่อมต่อ PostgreSQL ที่ตั้งค่าใน Backend ใช้ `psycopg.connect(**DATABASE_CONFIG)` และสร้าง Connection ต่อการเรียก Route ตามโค้ดปัจจุบัน การแก้ Schema ต้องตรวจผลกระทบต่อ Foreign Key ก่อนเสมอ

## 5. การติดตั้ง / เตรียม Environment สำหรับพัฒนาต่อ

**Prerequisites:** Linux/Ubuntu หรือ Container ที่เข้าถึงโปรเจกต์และ PostgreSQL ได้, Python 3.12, `python3.12-venv`, `pip`, สิทธิ์อ่าน Config และเขียน `storage/videos/` ตามงานที่ทดสอบ

ใน Terminal บน NAS หรือ Environment ทดสอบที่เตรียม Path/DB ให้ถูกต้อง:

```bash
cd /home/dev/vision-swine/management_system/backend

# ถ้าจำเป็นต้องสร้าง venv ใหม่ ให้สำรองของเดิมก่อน ห้ามลบทับ venv ที่กำลังใช้ใน Production
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

# จำเป็นต่อ POST /videos; มีใน venv เดิม แต่ยังไม่ถูก pin ใน requirements.txt
python -m pip install 'python-multipart==0.0.32'

python -m pip check
```

**สำคัญ:** ตัวอย่างนี้สร้าง/ปรับ `venv` จึงไม่ควรรันบน Production ที่กำลังให้บริการโดยไม่มีแผนสำรองและ Downtime; หาก `venv` มีอยู่แล้วให้ตรวจ `venv/bin/python -V` และ `venv/bin/python -m pip list` ก่อน ส่วนไฟล์ `.env` ที่ใช้ทดสอบต้องเป็น Credential ของ Environment นั้น ๆ และเนื่องจากโค้ดโหลด `.env` ด้วย Absolute Path ต้องจัด Path ให้ตรงหรือปรับโค้ดก่อนรันบนเครื่องอื่น

## 6. การเริ่มและตรวจสอบ API

### 6.1 เริ่มด้วยสคริปต์ของโครงการ (บน NAS)

```bash
bash /home/dev/vision-swine/tools/start-management-api.sh
```

สคริปต์ใน ZIP ใช้ `backend/venv/bin/uvicorn` รัน `vision_swine_api.main:app` แบบ `nohup` ที่ `127.0.0.1:8002` และส่ง Log ไป `/home/dev/vision-swine/logs/management-api.log`; มีการตรวจว่ามีโปรเซสที่ตรงเงื่อนไขทำงานอยู่หรือไม่ก่อนเริ่มใหม่

สคริปต์ `tools/start-all.sh` เรียก `start-pgweb.sh` และ `start-management-api.sh` เท่านั้น **ไม่ได้เริ่ม Nginx หน้า Video Recordings Web** โดยอัตโนมัติจากสคริปต์นี้

### 6.2 รันแบบ Foreground เพื่อ Debug (ใช้กับ Environment ทดสอบ)

```bash
cd /home/dev/vision-swine/management_system/backend
source venv/bin/activate
uvicorn vision_swine_api.main:app --host 127.0.0.1 --port 8002
```

> อย่าเปิด Uvicorn ซ้ำทับพอร์ต `8002` ที่ให้บริการอยู่แล้ว หากต้อง Debug แบบแยก ให้ใช้พอร์ตอื่นใน Environment ทดสอบที่เชื่อมต่อฐานข้อมูลทดสอบเท่านั้น

### 6.3 ตรวจสถานะโดยไม่แก้ข้อมูล

```bash
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8002/health/database
curl -fsS http://127.0.0.1:8002/system/storage
curl -fsS http://127.0.0.1:8002/videos

ss -ltnp | grep ':8002'
tail -n 80 /home/dev/vision-swine/logs/management-api.log
```

- `/health` ตรวจ API; **ไม่ได้ตรวจฐานข้อมูล**
- `/health/database` ตรวจ PostgreSQL และส่งชื่อฐานที่ API ใช้จริง
- `/system/storage` ตรวจพื้นที่ Disk และปริมาณไฟล์ใน Video Directory
- `/docs` และ `/redoc` เป็นเอกสาร API อัตโนมัติของ FastAPI (เข้าจากฝั่งที่เข้าถึง `127.0.0.1:8002` ได้)

กรณีเข้า NAS ผ่าน VS Code Remote-SSH: หากต้องเปิด `/docs` บน Browser เครื่องตนเอง ให้ Forward Port `8002` ผ่าน SSH ก่อน ไม่ต้องเปลี่ยน API เป็น `0.0.0.0` โดยไม่ประเมินความปลอดภัย

## 7. API Reference จาก Source Code ปัจจุบัน

ค่าที่ระบุด้านล่างยืนยันจาก `main.py` และ `routes/*.py` ใน ZIP; API ที่อ่านรายการจำกัด **100 แถวล่าสุดต่อคำขอ** โดยยังไม่มี Pagination

| Method | Endpoint | รายละเอียด |
|---|---|---|
| `GET` | `/health` | สถานะ FastAPI |
| `GET` | `/health/database` | สถานะ DB, ชื่อ DB, เวอร์ชัน PostgreSQL |
| `POST` | `/sessions` | สร้างรอบประมวลผล (`device_id`, `camera_id?`, `source_type`) |
| `GET` | `/sessions` | 100 Session ล่าสุดพร้อมชื่ออุปกรณ์ |
| `PATCH` | `/sessions/{session_id}/end` | ปิด Session; 404 ถ้าไม่พบ, 409 ถ้าปิดแล้ว |
| `POST` | `/weights` | บันทึกค่าน้ำหนักตาม `session_id` + `track_id` |
| `GET` | `/weights` | 100 ผลประเมินล่าสุด |
| `POST` | `/counting-events` | บันทึกเหตุการณ์ `IN` / `OUT`; เชื่อม `weight_estimation_id` ได้ |
| `GET` | `/counting-events` | 100 เหตุการณ์ล่าสุด พร้อมน้ำหนักที่เชื่อมไว้ (ถ้ามี) |
| `POST` | `/videos` | อัปโหลด MP4 + `session_id` แบบ `multipart/form-data`; ตอบ `201` |
| `GET` | `/videos` | 100 รายการวิดีโอล่าสุดจาก DB |
| `GET` | `/videos/{video_id}/file` | เล่น/ดาวน์โหลด MP4 ผ่าน `FileResponse` |
| `GET` | `/system/storage` | Disk total/used/free และ Video Storage Usage |

### 7.1 ข้อมูลสำคัญใน Request

**Session** (`POST /sessions`, JSON):

```json
{
  "device_id": 1,
  "camera_id": null,
  "source_type": "LIVE"
}
```

`device_id` ต้องมีอยู่ในตาราง `devices`; `camera_id` เป็น `null` ได้ แต่ถ้าระบุหมายเลขจะต้องมีอยู่ใน `cameras`; `source_type` ต้องเป็น `LIVE` หรือ `VIDEO` ตัวอย่างนี้แสดงรูปแบบข้อมูลเท่านั้น **อย่ารัน POST กับฐานข้อมูลจริงโดยไม่ตั้งใจ**

**Weight** (`POST /weights`, JSON):

```json
{
  "session_id": "<valid-session-uuid>",
  "track_id": 1,
  "weight_kg": 118.50,
  "frame_number": 120
}
```

`session_id` ต้องเป็น UUID ที่มีจริง, `track_id` มากกว่า 0, `weight_kg` มากกว่า 0 (รองรับทศนิยม 2 ตำแหน่ง), `frame_number` ตั้งแต่ 0

**Counting** (`POST /counting-events`, JSON):

```json
{
  "session_id": "<valid-session-uuid>",
  "track_id": 1,
  "direction": "IN",
  "frame_number": 120,
  "weight_estimation_id": null
}
```

`direction` ต้องเป็น `IN` หรือ `OUT`; `event_time` ไม่ส่งก็ได้ (Backend จะใช้เวลาปัจจุบันแบบ UTC); หากระบุ `weight_estimation_id` ระบบจะตรวจว่าเป็น `session_id` และ `track_id` เดียวกัน

**Video upload** (`POST /videos`): เป็น `multipart/form-data` ประกอบด้วย `session_id` และไฟล์ชื่อฟิลด์ `file`; อนุญาตเฉพาะชื่อไฟล์ลงท้าย `.mp4` ไม่รับไฟล์ว่าง, จำกัดขนาด **2 GiB** และอ่านไฟล์เป็น Chunk ขนาด **1 MiB** (ไม่ได้ยืนยันเนื้อไฟล์ว่าเป็น MP4 จริงนอกจากตรวจนามสกุล)

### 7.2 การอ่านวิดีโอ

1. `POST /videos`: ตรวจ Session ก่อนสร้างชื่อสุ่ม `<uuid>.mp4` และบันทึกลง `storage/videos/`
2. บันทึกแถวใน `public.video_recordings` โดยเก็บ `file_name` ต้นฉบับ, `file_path` แบบ `videos/<uuid>.mp4`, `file_size_bytes`, `status='READY'`
3. `GET /videos`: Frontend ใช้แสดงรายการล่าสุด (ยังไม่ใช่ผลลัพธ์แบบ Pagination)
4. `GET /videos/{video_id}/file`: ตรวจว่ามี Metadata, สถานะเป็น `READY`, Path อยู่ภายใต้ `storage/videos/` และมีไฟล์จริงก่อนส่ง `FileResponse` สำหรับการเล่นวิดีโอ

**หากลบเฉพาะ MP4 โดยไม่ลบ Metadata:** รายการอาจยังปรากฏใน `GET /videos` แต่เมื่อเปิดไฟล์จะได้รับ `404` (`Video file is missing from storage`); หากลบ Metadata แต่เก็บ MP4 ไว้จะเกิด Orphan File ควรพัฒนา Cleanup Workflow ที่ทำงานร่วมกันอย่างปลอดภัย

## 8. ตาราง PostgreSQL ที่สัมพันธ์กับ Backend

อ้างอิง `database/sql/schema_v2_current.sql` และ SQL ใน Routes:

| ตาราง | ใช้ทำอะไร |
|---|---|
| `devices`, `cameras` | ตรวจสอบอุปกรณ์/กล้องก่อนเปิด Session |
| `processing_sessions` | รอบการประมวลผล, เวลาเริ่ม/สิ้นสุด, อุปกรณ์และกล้อง |
| `weight_estimations` | `session_id`, `track_id`, `weight_kg`, `frame_number`, `estimated_at` |
| `counting_events` | `session_id`, `track_id`, `direction`, `frame_number`, `event_time`, การอ้างอิงน้ำหนัก |
| `video_recordings` | `session_id`, ชื่อและ Path ไฟล์, ขนาด, สถานะ, เวลาสร้าง |
| `farms`, `houses` | โครงสร้างสถานที่ใน Schema แต่ยังไม่มี Endpoint สำหรับ Filter ใน Router ชุดนี้ |

มี Foreign Key ระหว่างตารางหลายจุด จึง **ไม่ควรใช้ `TRUNCATE ... CASCADE` หรือลบ `processing_sessions` โดยไม่ตรวจความสัมพันธ์ก่อน** การบันทึกข้อมูลใหม่ควรเริ่มจาก Session ที่มี Device อยู่จริงก่อน

**pgweb:** ใช้ดูและ Query PostgreSQL ได้ แต่การกด `Disconnect` เป็นการตัดการเชื่อมต่อใน pgweb เท่านั้น ไม่ได้ปิด DB ผู้ใช้ต้อง Connect ใหม่ด้วย Credential ที่ผู้ดูแลจัดให้ โดยไม่ควรบันทึกรหัสผ่านจริงใน README

## 9. การเชื่อมต่อ Frontend / Network

Frontend `management_system/frontend/video-recordings-web/` ใช้ React/Vite และเชื่อม API รายการวิดีโอด้วย `/videos` ส่วน Nginx ที่ `nginx-recordings.conf` ฟังบนพอร์ต `8090` และ Reverse Proxy เส้นทาง `/videos` ไปที่ `127.0.0.1:8002`; ไฟล์ Build ของหน้าเว็บอยู่ที่ `dist/`

- ฝั่ง Backend เปิด CORS ด้วย Origin ที่กำหนดไว้แบบเจาะจงใน `main.py` (เช่น Dashboard ที่พัฒนาแยกบนอีกเครื่อง) ถ้าเปลี่ยน URL ของ Dashboard ต้องตรวจ CORS ก่อน
- Nginx Config ของ Video Recordings ปัจจุบัน Proxy **เฉพาะ `/videos`**; การเรียก `/health` จาก Browser ผ่านพอร์ต `8090` ไม่ใช่เส้นทาง API ที่ Config นี้รับรอง
- `127.0.0.1` หมายถึง Loopback ของเครื่อง/Container ที่รันบริการ ไม่ใช่ IP ที่ใช้จาก Windows; การเข้าถึงจากภายนอกควรใช้ SSH Port Forward / Network Routing ตามนโยบายระบบ
- การเริ่ม Nginx อัตโนมัติหลัง Container Restart ยังไม่ได้ยืนยันว่าถูกจัดการถาวร; `tools/start-all.sh` ไม่ได้เริ่ม Nginx

## 10. Troubleshooting

| ปัญหา | วิธีตรวจ/แนวทาง |
|---|---|
| `curl /health` ไม่ได้ / Connection refused | `ss -ltnp \| grep :8002`; ตรวจสคริปต์เริ่ม API และ `logs/management-api.log` |
| `/health` ตอบ OK แต่ `/health/database` Error | ตรวจ PostgreSQL, `.env` และชื่อ DB จริง; ดู Log ขณะเชื่อม; อย่าเปิดเผย Password |
| `RuntimeError: Form data requires "python-multipart"` | ติดตั้ง `python-multipart` ใน `venv` เดียวกับที่รัน Uvicorn; เพิ่มลง requirements ในรอบแก้โค้ด |
| `POST /sessions` ได้ 404 | ตรวจ `device_id` / `camera_id` ว่ามีในตารางจริง |
| `POST /counting-events` ได้ 422 | ตรวจรูปแบบฟิลด์ และถ้ามี `weight_estimation_id` ต้องตรง Session + Tracking ID |
| `GET /videos/{id}/file` ได้ 404 | ตรวจ Metadata และไฟล์จริงใต้ `storage/videos/`; กรณีลบ MP4 แล้ว DB ยังอยู่จะพบ 404 |
| `GET /videos/{id}/file` ได้ 409 | สถานะรายการใน DB ยังไม่ใช่ `READY` |
| `POST /videos` ได้ 413 | ไฟล์เกิน 2 GiB ตามข้อจำกัดใน `routes/videos.py` |
| `GET /system/storage` ได้ 503 | ตรวจว่ามี `storage/videos/` และสิทธิ์อ่าน Directory/Disk |
| หน้า React เปิดได้แต่รายการวิดีโอไม่มา | ตรวจ API `:8002/videos`, Config Nginx `:8090`, Error Log และการเข้าถึงเครือข่าย |
| ย้าย Backend ไปเครื่องใหม่แล้วหา `.env` ไม่เจอ | `database.py` ใช้ Absolute Path เดิม ต้องเตรียมตำแหน่งใหม่ให้ตรงหรือปรับโค้ด |

### ตรวจสอบก่อนและหลังแก้โค้ด

```bash
cd /home/dev/vision-swine/management_system/backend
./venv/bin/python -m compileall -q vision_swine_api
./venv/bin/python -m pip check
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8002/health/database
```

คำสั่ง `compileall` ตรวจได้เฉพาะ Syntax เบื้องต้น ไม่ได้แทนการทดสอบ API หรือ Database จริง; ควรทดสอบการเขียนข้อมูลกับฐานข้อมูลทดสอบแยกจาก Production

## 11. แนวทางพัฒนาต่อยอด (รายการที่ยังไม่ควรถือว่าพร้อมใช้งาน)

1. **เพิ่ม Automated Tests และ Test Database** เพื่อทดสอบ CRUD/Validation, Foreign Key, File Upload และ Range Request ก่อนปล่อยจริง
2. **จัดการ Dependencies ให้ครบ** โดยเพิ่ม `python-multipart` ลง `requirements.txt` และทดสอบสร้าง Environment ใหม่จากศูนย์
3. **จัดการ Configuration ให้ย้ายระบบได้ง่าย** ลด Absolute Path ใน `database.py`, ทำ `.env.example` ที่ไม่มี Secret และแยก Development/Production
4. **เพิ่ม Pagination / Filter API** โดยเฉพาะรายการวิดีโอ, ฟาร์ม, โรงเรือน, กล้อง; โค้ดปัจจุบัน `GET` Lists จำกัด 100 แถวและยังไม่มี Query Parameters สำหรับ Filter
5. **ออกแบบกลไกลบวิดีโอให้ DB และ Storage ตรงกัน** รวมการจัดการไฟล์ขาดหาย/Orphan, Log และการรับมือกรณีลบล้มเหลว
6. **เพิ่ม Security ก่อนเปิดบริการภายนอก** เช่น Authentication/Authorization, Secret Rotation, Rate Limit, Upload Validation; ในโค้ด Route ชุดนี้ยังไม่เห็นกลไก Login/API Token
7. **ทำ Deployment/Restart ที่ดูแลได้จริง** เช่นกำหนดบริการระดับ Host/Container, Health Check, Log Rotation และขั้นตอน Backup/Restore PostgreSQL ที่ทดสอบแล้ว
8. **เพิ่ม Monitoring** ของ Storage และ DB; ตรวจสอบข้อจำกัดด้านทรัพยากรของ Container และรองรับเวลาที่ API ไม่พร้อมใช้งาน

## 12. Handover Checklist สำหรับผู้รับช่วง

- [ ] เปิด `management_system/README.md` และอ่านคู่มือ v0.8.6 ประกอบ
- [ ] ยืนยัน **ชื่อ Database จริง** ผ่าน `/health/database` เทียบกับ pgweb ก่อนจัดการข้อมูล
- [ ] ตรวจ `backend/.env` โดยไม่คัดลอก Secret เข้าเอกสารหรือ Git
- [ ] ตรวจ Python Version, venv, requirements และ `python-multipart`
- [ ] ทดสอบ `GET /health`, `/health/database`, `/system/storage`, `/videos`
- [ ] ตรวจสถานะ PostgreSQL และ `logs/management-api.log`
- [ ] ตรวจ `storage/videos/` พร้อมสิทธิ์อ่าน/เขียน และตรวจ Metadata ใน `video_recordings`
- [ ] ตรวจการเชื่อมจาก Frontend ผ่าน Nginx พอร์ต `8090`
- [ ] ทดสอบ API ที่เขียนข้อมูลใน **Test Environment** ก่อน Production
- [ ] วางแผน Backup/Restore, ป้องกัน Secret และการ Restart บริการ

---

**ที่มาของข้อมูล:** `management_system/backend/requirements.txt`, `vision_swine_api/{main.py,database.py,routes/*.py}`, `tools/{start-management-api.sh,start-all.sh,start-pgweb.sh}`, `database/sql/schema_v2_current.sql` และ `frontend/video-recordings-web/nginx-recordings.conf` ภายใน ZIP ส่งมอบ ระบบจริงอาจเปลี่ยนภายหลังได้
