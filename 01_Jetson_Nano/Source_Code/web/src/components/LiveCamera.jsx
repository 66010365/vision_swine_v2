import { useEffect, useRef, useState } from 'react'
import Icon from './Icon'
import { jetsonAPI, STREAM_URL, unwrapData } from '../services/api'

const REF_WIDTH = 640
const REF_HEIGHT = 480
const initialGate = { x_left: 130, x_right: 490, line_y: 178, margin: 20 }

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, Math.round(value)))
}

export default function LiveCamera({ realtimeStatus, fps }) {
  const [version, setVersion] = useState(0)
  const [imageStatus, setImageStatus] = useState('connecting')
  const [gate, setGate] = useState(initialGate)
  const [gateError, setGateError] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveMessage, setSaveMessage] = useState('')
  const dragRef = useRef(null)

  useEffect(() => {
    const controller = new AbortController()

    jetsonAPI.getGate(controller.signal)
      .then((response) => {
        const data = unwrapData(response)
        const raw = data.gate || data
        setGate({
          x_left: Number(raw.x_left ?? initialGate.x_left),
          x_right: Number(raw.x_right ?? initialGate.x_right),
          line_y: Number(raw.line_y ?? raw.main_gate_y ?? initialGate.line_y),
          margin: Number(raw.margin ?? raw.margin_offset ?? initialGate.margin),
        })
        setGateError('')
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setGateError('ยังโหลดตำแหน่ง Gate จาก Jetson ไม่ได้')
        }
      })

    return () => controller.abort()
  }, [])

  function toCoords(event) {
    const rect = event.currentTarget.getBoundingClientRect()
    return {
      x: clamp((event.clientX - rect.left) / rect.width * REF_WIDTH, 0, REF_WIDTH),
      y: clamp((event.clientY - rect.top) / rect.height * REF_HEIGHT, 0, REF_HEIGHT),
    }
  }

  function startDrag(event) {
    const part = event.target.getAttribute('data-drag')
    if (!part) return
    dragRef.current = part
    event.currentTarget.setPointerCapture(event.pointerId)
    event.preventDefault()
  }

  function moveDrag(event) {
    if (!dragRef.current) return
    const { x, y } = toCoords(event)
    const part = dragRef.current
    setGate((old) => {
      if (part === 'left') return { ...old, x_left: Math.min(x, old.x_right - 10), line_y: y }
      if (part === 'right') return { ...old, x_right: Math.max(x, old.x_left + 10), line_y: y }
      return { ...old, line_y: y }
    })
  }

  function stopDrag(event) {
    dragRef.current = null
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }

  async function saveGate() {
    setSaveMessage('')
    if (gate.x_left >= gate.x_right || gate.margin < 0) {
      setSaveMessage('ตำแหน่ง Gate ไม่ถูกต้อง')
      return
    }

    setSaving(true)
    try {
      const response = await jetsonAPI.saveGate(gate)
      if (response?.success === false || response?.error) {
        throw new Error(response.error || 'บันทึกไม่สำเร็จ')
      }
      setGateError('')
      setSaveMessage('บันทึกตำแหน่ง Gate บน Jetson แล้ว')
    } catch (error) {
      setSaveMessage(`บันทึกไม่สำเร็จ: ${error.message}`)
    } finally {
      setSaving(false)
    }
  }

  function reloadStream() {
    setImageStatus('connecting')
    setVersion((value) => value + 1)
  }

  return (
    <section className="panel camera-panel">
      <div className="panel-heading">
        <div className="title-with-icon">
          <span className="icon-square"><Icon name="video" /></span>
          <h2>กล้องตรวจจับและนับจำนวนหมู (AI Object Detection Feed)</h2>
        </div>
        <div className="heading-actions">
          <span className="pill pill-green">MJPEG Stream</span>
          <button type="button" className="button-icon" onClick={reloadStream} title="เชื่อมต่อสตรีมใหม่" aria-label="เชื่อมต่อใหม่">
            <Icon name="refresh" size={16} />
          </button>
        </div>
      </div>

      <div className="camera-stage">
        <div className="camera-frame">
          <img
            key={version}
            src={`${STREAM_URL}?v=${version}`}
            onLoad={() => setImageStatus('live')}
            onError={() => setImageStatus('error')}
            alt="สตรีมภาพจากกล้อง AI บน Jetson Nano"
          />

          {/* Gate is in the same 640×480 coordinates as Jetson's camera */}
          <svg
            className="gate-overlay"
            viewBox={`0 0 ${REF_WIDTH} ${REF_HEIGHT}`}
            preserveAspectRatio="none"
            onPointerDown={startDrag}
            onPointerMove={moveDrag}
            onPointerUp={stopDrag}
            onPointerCancel={stopDrag}
            aria-label="ปรับตำแหน่งเส้นนับสุกร"
          >
            <line className="gate-margin" x1={gate.x_left} x2={gate.x_right} y1={gate.line_y - gate.margin} y2={gate.line_y - gate.margin} />
            <line className="gate-margin" x1={gate.x_left} x2={gate.x_right} y1={gate.line_y + gate.margin} y2={gate.line_y + gate.margin} />
            <line data-drag="line" className="gate-line" x1={gate.x_left} x2={gate.x_right} y1={gate.line_y} y2={gate.line_y} />
            <circle data-drag="left" className="gate-handle" cx={gate.x_left} cy={gate.line_y} r="11" />
            <circle data-drag="right" className="gate-handle" cx={gate.x_right} cy={gate.line_y} r="11" />
          </svg>

          {imageStatus !== 'live' && (
            <div className="camera-placeholder">
              <Icon name={imageStatus === 'error' ? 'alert' : 'video'} size={34} />
              <strong>{imageStatus === 'error' ? 'ไม่สามารถรับภาพสดได้' : 'กำลังเชื่อมต่อกล้อง AI...'}</strong>
              <small>{imageStatus === 'error' ? 'ตรวจสอบว่า AI และ MJPEG Server พอร์ต 8080 ทำงานอยู่' : 'กำลังรอเฟรมภาพจาก Jetson Nano'}</small>
            </div>
          )}

          <div className="camera-badges">
            <span className="fps-badge">{realtimeStatus === 'live' ? Math.round(fps ?? 0) : '—'} <small>FPS</small></span>
            <span className={`stream-badge ${imageStatus === 'live' ? 'is-live' : ''}`}>
              <span className="live-led" /> {imageStatus === 'live' ? 'LIVE' : 'OFFLINE'}
            </span>
          </div>
        </div>
      </div>

      <div className="camera-footer">
        <div className="camera-hint"><Icon name="info" size={15} /> ลากเส้นสีแดงหรือจุดสีขาวเพื่อปรับตำแหน่งนับสุกร</div>
        <button type="button" className="button-primary button-blue" onClick={saveGate} disabled={saving}>
          <Icon name="save" size={15} /> {saving ? 'กำลังบันทึก...' : 'บันทึกตำแหน่งเส้น'}
        </button>
      </div>
      {(saveMessage || gateError) && (
        <div className={`form-feedback ${gateError ? 'is-error' : ''}`} role="status">{saveMessage || gateError}</div>
      )}
    </section>
  )
}
