import { useEffect, useState } from 'react'
import Icon from './Icon'

// Saved only in the browser. Clear when a new tracking session starts,
// since the AI may reuse tracking IDs after restarting.
const WEIGHT_STORAGE_KEY = 'vision-swine:latest-pig-weights:v1'

function displayPercent(value) {
  return value == null ? '—' : `${Math.round(Number(value))}%`
}

function temp(value) {
  return value == null ? '—' : `${Math.round(Number(value))}°C`
}

function formatBytes(value) {
  if (value == null) return '—'
  const amount = Number(value)
  if (amount >= 1024 ** 3) return `${(amount / 1024 ** 3).toFixed(1)} GB`
  if (amount >= 1024 ** 2) return `${(amount / 1024 ** 2).toFixed(1)} MB`
  return `${(amount / 1024).toFixed(1)} KB`
}

function loadSavedWeights() {
  try {
    const saved = window.localStorage.getItem(WEIGHT_STORAGE_KEY)
    if (!saved) return {}
    const parsed = JSON.parse(saved)
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {}

    const valid = {}
    for (const [id, pig] of Object.entries(parsed)) {
      const weight = Number(pig?.weight)
      const seenAt = Number(pig?.seenAt)
      if (
        id &&
        pig?.weight != null &&
        Number.isFinite(weight) &&
        weight >= 0 &&
        Number.isFinite(seenAt) &&
        seenAt > 0
      ) {
        valid[id] = { id, weight, seenAt }
      }
    }
    return valid
  } catch {
    // Private browsing or browser storage restrictions should not break the UI.
    return {}
  }
}

function formatWeightTime(timestamp) {
  if (!timestamp) return 'ไม่ทราบเวลา'
  return new Date(timestamp).toLocaleString('th-TH', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

// Keep the API mapping compatible with the existing Jetson backend.
export function normalizeAIPigs(data) {
  const items = data?.pigs ?? data?.pig_weights ?? data?.weight_estimates ?? data?.detections ?? []
  if (!Array.isArray(items)) return []

  return items.filter((item) => item && typeof item === 'object').map((item, index) => {
    const rawId = item.track_id ?? item.tracking_id ?? item.pig_id ?? item.id
    const id = rawId == null || String(rawId).trim() === ''
      ? `รายการ ${index + 1}`
      : /^\d+$/.test(String(rawId)) ? `PIG-${String(rawId).padStart(3, '0')}` : String(rawId)
    const rawWeight = item.weight_kg ?? item.estimated_weight_kg ?? item.predicted_weight ?? item.weight
    const weight = rawWeight == null || rawWeight === '' ? null : Number(rawWeight)
    return { id, weight: weight != null && Number.isFinite(weight) && weight >= 0 ? weight : null }
  })
}

const DEMO_AI = {
  pig_counts: { total: 2, in: 7, out: 3 },
  pigs: [
    { track_id: 1, weight_kg: 118.5 },
    { track_id: 2, weight_kg: 125.0 },
  ],
}

// Count is net crossings since reset, NOT the number physically in the pen.
export function PigCounter({ data, status }) {
  const [demo, setDemo] = useState(false)
  const [weightMemory, setWeightMemory] = useState(loadSavedWeights)

  // Remember only real, valid AI measurements that have a tracking ID.
  // No automatic 10-second expiry: each new prediction updates its ID.
  useEffect(() => {
    if (status !== 'live' || !data) return

    const measured = normalizeAIPigs(data).filter(
      (pig) => pig.weight != null && !pig.id.startsWith('รายการ ')
    )
    if (measured.length === 0) return

    const seenAt = Date.now()
    setWeightMemory((previous) => {
      const next = { ...previous }
      for (const pig of measured) {
        next[pig.id] = { id: pig.id, weight: pig.weight, seenAt }
      }
      return next
    })
  }, [data, status])

  // Keep the latest real measurements even if the browser page is refreshed.
  useEffect(() => {
    try {
      if (Object.keys(weightMemory).length === 0) {
        window.localStorage.removeItem(WEIGHT_STORAGE_KEY)
      } else {
        window.localStorage.setItem(WEIGHT_STORAGE_KEY, JSON.stringify(weightMemory))
      }
    } catch {
      // Continue using in-memory data if localStorage is not available.
    }
  }, [weightMemory])

  const source = demo ? DEMO_AI : status === 'live' ? data : null
  const counts = source?.pig_counts
  const incoming = counts?.in == null ? null : Number(counts.in)
  const outgoing = counts?.out == null ? null : Number(counts.out)
  const validIn = Number.isFinite(incoming)
  const validOut = Number.isFinite(outgoing)
  const net = validIn && validOut ? incoming - outgoing : null
  const observed = counts?.total == null ? null : Number(counts.total)
  const frameCount = Number.isFinite(observed) ? observed : null

  const livePigs = normalizeAIPigs(source)
  const liveIds = new Set(livePigs.map((pig) => pig.id))

  const liveRows = livePigs.map((pig) => {
    const previous = demo ? null : weightMemory[pig.id]
    const isLatest = !demo && pig.weight == null && previous != null
    return {
      id: pig.id,
      weight: isLatest ? previous.weight : pig.weight,
      isLatest,
      isDemo: demo,
      seenAt: isLatest ? previous.seenAt : null,
    }
  })

  const savedRows = demo ? [] : Object.values(weightMemory)
    .filter((pig) => !liveIds.has(pig.id))
    .sort((a, b) => b.seenAt - a.seenAt)
    .map((pig) => ({ ...pig, isLatest: true, isDemo: false }))

  const displayedPigs = [...liveRows, ...savedRows]
  const savedCount = Object.keys(weightMemory).length

  return (
    <article className="panel count-panel ai-v6-panel" aria-label="ผลลัพธ์โมเดล AI">
      <div className="panel-heading compact ai-v6-heading">
        <div className="title-with-icon">
          <span className="icon-square green"><Icon name="dashboard" /></span>
          <h3>ผลลัพธ์โมเดล AI</h3>
        </div>
        <span className={`pill ${demo ? 'ai-v6-demo-badge' : status === 'live' ? 'pill-green' : 'pill-muted'}`}>
          {demo ? 'DEMO' : status === 'live' ? 'Counting...' : 'รอข้อมูล...'}
        </span>
      </div>

      <div className="ai-v6-net-summary">
        <div className="ai-v6-caption">จำนวนสุกรสุทธิ (เข้า − ออก)</div>
        <div className="ai-v6-net-number">{net ?? '—'} <small>ตัว</small></div>
        <div className="ai-v6-explain">ผลต่างสะสมของการเดินผ่านเส้นนับ</div>
      </div>

      <div className="ai-v6-direction-grid">
        <div className="ai-v6-direction in">
          <span>↑ จำนวนขาเข้า</span>
          <strong>{validIn ? incoming : '—'} <small>ตัว</small></strong>
        </div>
        <div className="ai-v6-direction out">
          <span>↓ จำนวนขาออก</span>
          <strong>{validOut ? outgoing : '—'} <small>ตัว</small></strong>
        </div>
      </div>

      <div className="ai-v6-weight-top">
        <h4>⚖️ น้ำหนักสุกรรายตัว</h4>
        <span className="ai-v6-frame-badge">{frameCount == null ? '—' : frameCount} ตัวในเฟรม</span>
      </div>

      <div className="ai-v6-weight-list" aria-live="polite">
        {displayedPigs.length === 0 ? (
          <div className="ai-v6-weight-empty">
            {!source
              ? 'ยังไม่มีน้ำหนักที่บันทึกไว้ / กำลังรอข้อมูลจาก AI'
              : frameCount === 0
                ? 'ไม่พบสุกรในเฟรมปัจจุบัน และยังไม่มีน้ำหนักล่าสุด'
                : 'ยังไม่มีข้อมูลน้ำหนักรายตัวจาก API'}
          </div>
        ) : displayedPigs.map((pig, index) => (
          <div className="ai-v6-pig-row" key={`${pig.id}-${index}`}>
            <span className="ai-v6-pig-emoji" aria-hidden="true">🐷</span>
            <div className="ai-v6-pig-meta">
              <strong>{pig.id}</strong>
              <small>
                {pig.isDemo
                  ? 'DEMO · ข้อมูลตัวอย่าง'
                  : pig.isLatest
                    ? `ค่าล่าสุด · ${formatWeightTime(pig.seenAt)}`
                    : 'Live · น้ำหนักจาก AI ปัจจุบัน'}
              </small>
            </div>
            <div className="ai-v6-pig-weight">
              {pig.weight == null ? 'รอน้ำหนัก' : <>{pig.weight.toFixed(1)} <small>กก.</small></>}
            </div>
          </div>
        ))}
      </div>

      {!demo && savedCount > 0 && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 10 }}>
          <button
            type="button"
            className="ai-v6-demo-button"
            onClick={() => setWeightMemory({})}
            title="ลบเฉพาะน้ำหนักล่าสุดที่เก็บไว้ใน Dashboard ไม่ลบข้อมูลใน NAS หรือ PostgreSQL"
          >
            ล้างน้ำหนักล่าสุด ({savedCount})
          </button>
        </div>
      )}

      <p className="ai-v6-note">
        Live = ค่าจาก API ปัจจุบัน · ค่าล่าสุด = น้ำหนักที่เคยประเมินได้
        และบันทึกไว้ในเบราว์เซอร์จนกว่าจะมีการประเมินใหม่หรือกดล้างข้อมูล
      </p>
      <p className="ai-v6-note">
        จำนวนสุทธิ = เข้า − ออก นับตั้งแต่รีเซ็ต ไม่ใช่จำนวนสุกรในเล้าจริง
        หากไม่ได้กำหนดจำนวนเริ่มต้น และไม่ใช่จำนวนที่พบในเฟรม
      </p>
      <div className="ai-v6-actions">
        <button type="button" className="ai-v6-demo-button" aria-pressed={demo} onClick={() => setDemo((v) => !v)}>
          {demo ? 'กลับไปข้อมูลจริง' : 'ดูข้อมูลตัวอย่าง'}
        </button>
        <button type="button" className="ai-v6-reset-button" disabled title="การรีเซ็ตตัวนับผ่าน Backend ยังไม่ได้ยืนยันว่ารองรับ">
          ↻ รีเซ็ตตัวนับ · ยังไม่รองรับ
        </button>
      </div>
    </article>
  )
}

export function HardwareMini({ data, status }) {
  const hw = status === 'live' ? data?.hardware : null
  const metrics = [
    ['CPU', displayPercent(hw?.cpu_usage), 'blue'],
    ['GPU', displayPercent(hw?.gpu_usage), 'purple'],
    ['RAM', displayPercent(hw?.ram_usage), 'indigo'],
    ['TEMP', temp(hw?.cpu_temp), 'orange'],
  ]
  return (
    <article className="panel monitor-panel">
      <div className="panel-heading compact"><div className="title-with-icon"><span className="icon-square"><Icon name="cpu" /></span><h3>Jetson Nano Hardware Monitor</h3></div></div>
      <div className="load-card"><span>ภาระงานเฉลี่ย (4 Cores)</span><strong>{hw?.load_avg?.length >= 3 ? hw.load_avg.join(' / ') : '— / — / —'}</strong></div>
      <div className="mini-metrics">
        {metrics.map(([label, value, color]) => (
          <div className={`mini-metric ${color}`} key={label}>
            <span>{label}</span><strong>{value}</strong>
            <div className="progress-track"><div className="progress-fill" style={{ width: label === 'TEMP' ? `${Math.min(100, Number(hw?.cpu_temp || 0) / 85 * 100)}%` : `${Math.max(0, Math.min(100, Number(hw?.[label === 'CPU' ? 'cpu_usage' : label === 'GPU' ? 'gpu_usage' : 'ram_usage'] || 0)))}%` }}/></div>
          </div>
        ))}
      </div>
      {hw?.gpu_usage == null && <small className="muted-note">GPU Usage: Jetson API ยังไม่ส่งค่านี้</small>}
    </article>
  )
}

export function NetworkMini({ data, status, latency }) {
  const net = status === 'live' ? data?.network : null
  const speed = net ? (Number(net.rx_speed || 0) + Number(net.tx_speed || 0)) / 1024 : null
  return (
    <article className="panel monitor-panel">
      <div className="panel-heading compact"><div className="title-with-icon"><span className="icon-square"><Icon name="globe" /></span><h3>Network Status</h3></div></div>
      <div className="network-grid">
        <div className="network-tile"><span>Speed</span><strong className="green-text">{speed == null ? '—' : speed.toFixed(1)} <small>KB/s</small></strong></div>
        <div className="network-tile"><span>Latency</span><strong className="orange-text">{latency ?? '—'} <small>ms</small></strong></div>
        <div className="network-tile"><span>ส่งออก (TX)</span><strong>{formatBytes(net?.tx_bytes)}</strong></div>
        <div className="network-tile"><span>รับเข้า (RX)</span><strong>{formatBytes(net?.rx_bytes)}</strong></div>
      </div>
    </article>
  )
}

export { displayPercent, temp, formatBytes }
