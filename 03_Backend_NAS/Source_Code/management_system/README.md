# VISION SWINE — Management System (Backend & NAS)

ระบบ Backend สำหรับรับ จัดการ และเรียกดูข้อมูลจาก **Vision Swine** ซึ่งใช้ Edge AI ตรวจนับจำนวนและประเมินน้ำหนักสุกร โดยมี **FastAPI** ให้บริการ API, **PostgreSQL** จัดเก็บข้อมูล และพื้นที่ **NAS Storage** สำหรับไฟล์วิดีโอ

> **ขอบเขตเอกสาร:** README นี้เป็นคู่มือเริ่มต้นของระบบฝั่ง NAS ไม่ใช่คู่มือฉบับเต็ม และไม่ครอบคลุมโค้ด Jetson Nano หรือ Frontend Dashboard ที่พัฒนาแยกจากโฟลเดอร์นี้  
> **ที่มาของข้อมูล:** ตรวจจากสำเนาโปรเจกต์ `Backend_NAS.zip` และผลตรวจบริการ NAS ที่บันทึกไว้เมื่อ 22 พฤศจิกายน 2026 รายละเอียดการทำงานจริงอาจเปลี่ยนไปเมื่ออัปเดตระบบ

## 1. ภาพรวมการทำงาน

```text
Jetson Nano / Edge AI
         |
         | ข้อมูลการนับ / น้ำหนัก / วิดีโอ (ผ่าน API)
         v
FastAPI Backend (ภายใน NAS)
         |                       |
         v                       v
PostgreSQL 18                NAS Storage
ข้อมูลและ Metadata           ไฟล์วิดีโอ MP4
         ^                       ^
         |                       |
         +------ API สำหรับ Dashboard ------+
```

- **FastAPI** จัดการ Session, ค่าน้ำหนัก, เหตุการณ์นับ, วิดีโอ และข้อมูลพื้นที่จัดเก็บ
- **PostgreSQL** เก็บข้อมูลฟาร์ม โรงเรือน อุปกรณ์ กล้อง และผลการประมวลผล
- **Storage** เก็บไฟล์ MP4 จริง โดยบันทึกข้อมูลอ้างอิงของไฟล์ใน PostgreSQL
- **Frontend Dashboard** เรียก API เพื่อแสดงข้อมูล แต่ Source Code ของ Dashboard หลักอยู่ในโปรเจกต์แยก

แผนภาพนี้เป็นภาพรวมตาม Source Code; การเชื่อมต่อครบวงจรต้องตรวจสอบในระบบที่ใช้งานจริงอีกครั้ง

## 2. โครงสร้างโฟลเดอร์

ตำแหน่งโปรเจกต์บน NAS: `/home/dev/vision-swine`

```text
vision-swine/
├── management_system/
│   ├── README.md                   # เอกสารนี้
│   ├── backend/
│   │   ├── .env                    # ตัวแปรการเชื่อมต่อ (ข้อมูลลับ)
│   │   ├── requirements.txt        # ไลบรารี Python
│   │   ├── venv/                   # Python virtual environment
│   │   └── vision_swine_api/
│   │       ├── main.py             # FastAPI application / health checks
│   │       ├── database.py         # การเชื่อมต่อ PostgreSQL
│   │       └── routes/
│   │           ├── sessions.py     # รอบการประมวลผล AI
│   │           ├── weights.py      # ผลการประเมินน้ำหนัก
│   │           ├── counting.py     # เหตุการณ์นับ IN / OUT
│   │           ├── videos.py       # อัปโหลดและเรียกดู MP4
│   │           └── storage.py      # ตรวจสอบพื้นที่จัดเก็บ
│   └── frontend/                   # ไม่พบ Source Code Dashboard ใน ZIP ที่ตรวจ
├── database/
│   ├── sql/                        # Schema ปัจจุบันที่ส่งออกไว้
│   ├── scripts/                    # สคริปต์ส่งออก Schema
│   ├── backups/                    # Database dumps จากหลายช่วงเวลา
│   └── postgres18/                 # PostgreSQL data directory
├── storage/
│   ├── images/
│   │   ├── raw/
│   │   └── processed/
│   └── videos/                     # ไฟล์วิดีโอที่จัดเก็บ
├── logs/                           # บันทึกการทำงาน
├── tools/                          # สคริปต์และโปรแกรมสำหรับเริ่มบริการ
└── api-backup-before-rename-20261014-163204/
                                    # โค้ดสำรอง API รุ่นเก่า (ไม่ใช่ตัวหลัก)
```

> โครงสร้างด้านบนสรุปเฉพาะส่วนที่เกี่ยวข้องกับการดูแลระบบ ไม่แสดงไลบรารีใน `venv/` และไฟล์ข้อมูลทุกไฟล์

## 3. บริการและพอร์ตที่ใช้

| บริการ | ที่อยู่ภายใน NAS | หมายเหตุ |
|---|---|---|
| FastAPI | `127.0.0.1:8002` | API และ Swagger (`/docs`) |
| PostgreSQL 18 | `127.0.0.1:5432` | ฐานข้อมูลหลัก `vision_swine_v2` |
| pgweb | `127.0.0.1:8082` | ต้องใช้ SSH Tunnel เพื่อเปิดจากโน้ตบุ๊ก |

`127.0.0.1` เป็นที่อยู่ของ **เครื่องที่รันบริการนั้น** ไม่ใช่ IP ที่เปิดจากคอมพิวเตอร์เครื่องอื่นได้โดยตรง

### ตรวจสอบสถานะ (คำสั่งอ่านอย่างเดียว)

เปิด Terminal ที่เชื่อมต่อเข้า NAS แล้วรัน:

```bash
pg_lsclusters
ps -ef | grep -Ei 'postgres|pgweb|uvicorn' | grep -v grep
ss -ltnp | grep -E ':(5432|8002|8082)\b'
curl -i http://127.0.0.1:8002/health
curl -i http://127.0.0.1:8002/health/database
```

- หาก `/health` ตอบ `200` แสดงว่า API ตอบสนองได้ แต่ **ยังไม่ได้ยืนยันว่าทุกฟังก์ชันทำงานครบ**
- หาก `/health/database` ตอบ `200` แสดงว่าคำขอเชื่อมฐานข้อมูลสำเร็จ
- ถ้าไม่พบ Listener หรือคำสั่ง `curl` ล้มเหลว ให้ตรวจ Process และ Log ก่อนเริ่มบริการใหม่
- สภาพแวดล้อม NAS ที่ตรวจล่าสุดไม่มี `systemd` และไม่พบคำสั่ง `docker` ใน Shell จึงใช้ `ps`, `ss`, `pg_lsclusters` ในการตรวจสอบเบื้องต้น

## 4. API ที่มีใน Source Code

| Method | Endpoint | คำอธิบาย |
|---|---|---|
| GET | `/health` | ตรวจสอบ Backend |
| GET | `/health/database` | ตรวจสอบการเชื่อมต่อ PostgreSQL |
| POST | `/sessions` | สร้าง Processing Session |
| GET | `/sessions` | อ่าน Session ล่าสุด |
| PATCH | `/sessions/{session_id}/end` | ปิด Session |
| POST | `/weights` | บันทึกน้ำหนักที่ AI ประเมิน |
| GET | `/weights` | อ่านข้อมูลน้ำหนักล่าสุด |
| POST | `/counting-events` | บันทึกเหตุการณ์ IN / OUT |
| GET | `/counting-events` | อ่านเหตุการณ์นับพร้อมข้อมูลน้ำหนักที่เชื่อมโยง |
| POST | `/videos` | อัปโหลดไฟล์ MP4 และบันทึก Metadata |
| GET | `/videos` | อ่านรายการวิดีโอ |
| GET | `/videos/{video_id}/file` | เปิดไฟล์วิดีโอตาม ID |
| GET | `/system/storage` | ตรวจสอบพื้นที่ดิสก์และพื้นที่วิดีโอ |

ดูรายละเอียด Schema ของคำขอและคำตอบที่ `http://127.0.0.1:8002/docs` **จากภายใน NAS หรือผ่าน SSH Tunnel ที่ตั้งค่าไว้แล้ว** (เมื่อบริการทำงานอยู่)

## 5. PostgreSQL และไฟล์วิดีโอ

ฐานข้อมูลหลัก: `vision_swine_v2` มีตารางที่ระบุใน Schema ทั้งหมด 8 ตาราง

`farms` · `houses` · `devices` · `cameras` · `processing_sessions` · `counting_events` · `weight_estimations` · `video_recordings`

- `processing_sessions` ใช้เชื่อมเหตุการณ์นับ น้ำหนัก และวิดีโอในรอบประมวลผลเดียวกัน
- `video_recordings` เก็บข้อมูลไฟล์และพาธอ้างอิง ส่วนไฟล์ MP4 จริงอยู่ภายใต้ `storage/videos/`
- `videos.py` กำหนดพาธวิดีโอเป็น `/home/dev/vision-swine/storage/videos` ในโค้ด ส่วน `storage.py` รองรับตัวแปร `VISION_SWINE_VIDEO_DIR` หากเปลี่ยนค่าตัวแปรนี้ ต้องตรวจสอบให้ทั้งสองส่วนชี้พื้นที่เดียวกัน
- `database/scripts/export_schema_v2.py` ส่งออกเฉพาะโครงสร้างฐานข้อมูล (**ไม่ใช่** การสำรองข้อมูลทั้งหมด)
- ห้ามคัดลอก `database/postgres18/` ขณะ PostgreSQL ทำงาน แล้วถือว่าเป็น Backup ที่กู้คืนได้แน่นอน ควรใช้กระบวนการสำรอง/กู้คืน PostgreSQL ที่ถูกต้องและทดสอบในฐานข้อมูลทดสอบ

## 6. วิธีเข้า pgweb จากโน้ตบุ๊ก (SSH Tunnel)

pgweb ทำงานบน `127.0.0.1:8082` ภายใน NAS ดังนั้น **การ SSH เข้า PuTTY ตามปกติไม่ทำให้เปิดหน้าเว็บได้ทันที** ต้องทำ Port Forwarding เพิ่ม

### วิธี A — Windows PowerShell

เปิด PowerShell **บนโน้ตบุ๊ก Windows** แล้วใช้คำสั่ง:

```powershell
ssh -N -L 18082:127.0.0.1:8082 <SSH_USER>@<NAS_TAILSCALE_IP>
```

แทน `<SSH_USER>` และ `<NAS_TAILSCALE_IP>` ด้วยบัญชีและ IP ที่ได้รับอนุญาต แล้วเปิดหน้าต่าง PowerShell นี้ค้างไว้ จากนั้นเปิดเบราว์เซอร์ที่:

**http://127.0.0.1:18082**

### วิธี B — PuTTY

1. เปิด **PuTTY Configuration** แล้วตั้งค่า Host ของ NAS ตามปกติ
2. ไปที่ **Connection → SSH → Tunnels**
3. ใส่ **Source port:** `18082` และ **Destination:** `127.0.0.1:8082`
4. เลือก **Local** แล้วกด **Add** (จะเห็นรายการลักษณะ `L18082 127.0.0.1:8082`)
5. กลับหน้า **Session** บันทึกค่าการเชื่อมต่อหากต้องการ แล้วกด **Open**
6. เมื่อ SSH เชื่อมต่อแล้ว เปิด **http://127.0.0.1:18082** บนเบราว์เซอร์ของโน้ตบุ๊ก

หากเข้าไม่ได้ ให้ตรวจว่า pgweb ยังทำงานที่ `127.0.0.1:8082`, SSH Tunnel เปิดอยู่ และพอร์ต `18082` บนโน้ตบุ๊กไม่ถูกโปรแกรมอื่นใช้งาน

## 7. การเริ่มบริการและ Log

สคริปต์ที่พบใน `tools/`:

| ไฟล์ | หน้าที่ |
|---|---|
| `start-management-api.sh` | เริ่ม FastAPI ผ่าน Uvicorn ที่ `127.0.0.1:8002` |
| `start-pgweb.sh` | เริ่ม pgweb ที่ `127.0.0.1:8082` |
| `start-all.sh` | เรียกสคริปต์เริ่ม pgweb และ Backend |
| `container-start.sh` | เรียก `start-all.sh` แล้วคง Process หลักไว้ |
| `pgweb_vision_swine` | โปรแกรม pgweb แบบ Executable |

หากจำเป็นต้องเริ่มบริการใหม่ **ต้องตรวจ Process/Port ก่อน และได้รับอนุญาตจากผู้ดูแลระบบ** จึงค่อยใช้:

```bash
bash /home/dev/vision-swine/tools/start-all.sh
```

สคริปต์เริ่มบริการไม่ได้ยืนยันความสมบูรณ์ของ API หรือฐานข้อมูลทุกส่วน ต้องตรวจ `/health`, `/health/database` และ Log หลังเรียกใช้งาน ทั้งนี้ยังไม่ได้ยืนยันกลไกเริ่มอัตโนมัติหลัง Restart จากหลักฐานใน ZIP เพียงอย่างเดียว

Log ที่เกี่ยวข้อง ได้แก่:

```text
/home/dev/vision-swine/logs/management-api.log
/home/dev/vision-swine/logs/pgweb.log
/home/dev/vision-swine/logs/fastapi.log    # อาจมีประวัติจากการทำงานก่อนหน้า
```

ดู Log ล่าสุดโดยไม่แก้ไขไฟล์:

```bash
tail -n 50 /home/dev/vision-swine/logs/management-api.log
tail -n 50 /home/dev/vision-swine/logs/pgweb.log
```

## 8. การตั้งค่าและความปลอดภัย

ไฟล์ `management_system/backend/.env` ใช้ตัวแปรต่อไปนี้:

```text
DATABASE_NAME
DATABASE_USER
DATABASE_PASSWORD
DATABASE_HOST
DATABASE_PORT
```

- **ห้ามเผยแพร่** `.env`, รหัสผ่าน, Token, ไฟล์ Dump ที่มีข้อมูลจริง หรือข้อมูลฟาร์มที่ไม่ได้รับอนุญาต
- หากจะนำ Source Code ขึ้น GitHub ให้ตรวจไฟล์ก่อนเผยแพร่ และแยก `venv/`, ไฟล์ฐานข้อมูล, Log และวิดีโอออกจากโค้ดที่ต้องเก็บเวอร์ชัน
- CORS ใน `main.py` ไม่ใช่ระบบ Authentication/Authorization จึงควรตรวจสอบนโยบายการเข้าถึง API ก่อนเปิดให้บุคคลอื่นใช้งาน
- `main.py` และ `database.py` ใช้ค่าที่ระบุไว้สำหรับระบบบน NAS นี้ หากย้ายเครื่องต้องตรวจสอบพาธและ Environment ใหม่
- ใน ZIP ที่ตรวจ `requirements.txt` ไม่ได้ระบุ `python-multipart` โดยตรง ทั้งที่ API อัปโหลดวิดีโอใช้ `Form` และ `File`; ควรตรวจ Dependency ที่ติดตั้งจริงก่อนติดตั้งระบบใหม่

## 9. สถานะและการตรวจรับทางเทคนิค

**ยืนยันจาก Source Code และไฟล์สำรอง:** มี Router สำหรับ Sessions, Weights, Counting, Videos, Storage และ Schema หลัก 8 ตาราง

**มีหลักฐานจาก NAS ที่ตรวจล่าสุด:** PostgreSQL Cluster `18/main` อยู่สถานะ `online` ที่พอร์ต 5432 และพบ Process pgweb ที่ใช้ `127.0.0.1:8082`

**ยังต้องยืนยันก่อนถือว่าส่งมอบพร้อมใช้งานครบวงจร:**

- การเริ่มบริการอัตโนมัติหลัง Restart / ไฟดับ
- การกู้คืน PostgreSQL จากไฟล์ Backup บนสภาพแวดล้อมทดสอบ
- การทำงาน Jetson Nano → API → PostgreSQL/Storage → Dashboard แบบ End-to-End
- การกำหนดสิทธิ์เข้าถึง API และวิธีจัดการข้อมูลสำคัญ

## 10. คู่มือฉบับเต็ม

คู่มือเชิงเทคนิคประกอบสำหรับผู้ดูแลระบบคือ **VISION SWINE — Backend & NAS Manual v0.8.1** ซึ่งครอบคลุม Source Code, Database, Storage, Tools, Logs, การแก้ไขปัญหา และรายการตรวจสอบก่อนส่งมอบ

> **ข้อเสนอในการจัดไฟล์:** สามารถนำคู่มือ Word ไปเก็บในโฟลเดอร์ `Documentation/` ที่สร้างใหม่ภายใต้โปรเจกต์หลัก แล้วเพิ่มลิงก์ไฟล์ที่นี่เมื่อจัดเก็บเรียบร้อย (ยังไม่มีโฟลเดอร์นี้ใน ZIP ต้นฉบับ)

---

**หมายเหตุ:** README นี้จัดทำจากสำเนาโปรเจกต์ ไม่ได้ปรับเปลี่ยน Source Code หรือบริการที่รันอยู่บน NAS กรุณาอัปเดตข้อมูลให้ตรงกับระบบจริงทุกครั้งที่เปลี่ยนพอร์ต โครงสร้างฐานข้อมูล API หรือวิธีเริ่มบริการ
