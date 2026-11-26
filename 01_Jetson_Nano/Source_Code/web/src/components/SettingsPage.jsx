import { useState } from 'react'
import Icon from './Icon'

// [REACT v6 SETTINGS] This backend currently does NOT expose /api/config/model.
// Sliders are local previews only; never imply that a new value was saved.
export default function SettingsPage() {
  const [confidence, setConfidence] = useState(0.45)
  const [iou, setIou] = useState(0.5)
  const reset = () => { setConfidence(0.45); setIou(0.5) }
  return (
    <div className="v6-settings-stack">
      <section className="panel v6-settings-hero">
        <div className="v6-title-icon"><span className="v6-icon green"><Icon name="settings" size={22} /></span><div><h2>ตั้งค่าระบบ AI <em>/ AI Settings</em></h2><p>ตรวจสอบข้อมูลโมเดล และทดลองปรับเกณฑ์ในการตรวจจับสุกรของ Edge AI</p></div></div>
        <span className="v6-status neutral">ยังไม่ได้อ่านค่าจาก Jetson</span>
      </section>
      <div className="v6-settings-grid">
        <section className="panel v6-card v6-model-card">
          <div className="v6-card-header"><h3>ข้อมูลโมเดลที่ใช้งาน</h3><span className="v6-status neutral">อ่านอย่างเดียว</span></div>
          <div className="v6-model-highlight"><small>AI DETECTION ENGINE</small><strong>YOLOv8 / TensorRT</strong><p>ข้อมูลจากโครงการ ยังไม่ได้ยืนยันไฟล์ Engine ที่โหลดจริงผ่าน API</p></div>
          <dl className="v6-description-list">
            <div><dt>อุปกรณ์ประมวลผล</dt><dd>Jetson Nano</dd></div>
            <div><dt>Weight Estimation</dt><dd>Linear Regression (8 Features)</dd></div>
            <div><dt>Inference Precision</dt><dd>—</dd></div>
            <div><dt>Model File / Engine</dt><dd>—</dd></div>
          </dl>
          <p className="v6-muted">ไม่เปิดให้สลับ Model Architecture หรือ FP32 / FP16 / INT8 จากหน้าเว็บ จนกว่าจะมี Backend รองรับจริง</p>
        </section>
        <section className="panel v6-card v6-thresholds">
          <div className="v6-card-header"><h3>Detection Thresholds</h3><span className="v6-status demo">LOCAL PREVIEW</span></div>
          <p className="v6-muted">ค่าต่อไปนี้ใช้ทดลองการจัดหน้าจอ ยังไม่ใช่ค่าที่อ่านจาก Jetson และไม่ส่งไปยังโมเดล AI</p>
          <label className="v6-range-field"><span className="v6-range-title">Confidence Threshold <span>(ความมั่นใจขั้นต่ำ)</span><b>{confidence.toFixed(2)}</b></span><input type="range" min="0.1" max="0.95" step="0.01" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} /><span className="v6-range-ends"><small>0.10 รับการตรวจจับมากขึ้น</small><small>0.95 เข้มงวดมากขึ้น</small></span><p>คะแนนต่ำกว่าเกณฑ์จะถูกกรองออก ค่าสูงขึ้นอาจลดการตรวจจับผิด แต่เพิ่มโอกาสพลาดสุกร</p></label>
          <label className="v6-range-field blue"><span className="v6-range-title">IoU Threshold <span>(การซ้อนทับของกรอบ)</span><b>{iou.toFixed(2)}</b></span><input type="range" min="0.1" max="0.95" step="0.01" value={iou} onChange={(e) => setIou(Number(e.target.value))} /><span className="v6-range-ends"><small>0.10 กรองซ้ำมากขึ้น</small><small>0.95 ยอมให้ซ้อนมากขึ้น</small></span><p>ใช้ควบคุม NMS ไม่ใช่ค่าความแม่นยำของโมเดล</p></label>
          <div className="v6-settings-actions"><span className="v6-muted">ยังไม่มี GET/POST /api/config/model บน Backend นี้</span><button type="button" className="v6-button subtle" onClick={reset}>คืนค่าตัวอย่าง</button><button type="button" className="v6-button primary" disabled>บันทึก Thresholds</button></div>
        </section>
      </div>
      <aside className="v6-info-bar"><Icon name="info" size={19}/><div><b>การตั้งค่านี้มีผลกับอะไร?</b><p>Confidence และ IoU เกี่ยวข้องกับการตรวจจับ (Detection) ซึ่งอาจส่งผลทางอ้อมต่อ Tracking การนับเข้า–ออก และจำนวนสุกรที่ประเมินน้ำหนักได้ แต่ไม่ใช่การฝึกโมเดลหรือปรับโมเดล Linear Regression โดยตรง</p></div></aside>
    </div>
  )
}
