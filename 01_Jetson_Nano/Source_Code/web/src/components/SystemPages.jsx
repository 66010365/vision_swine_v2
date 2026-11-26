import { useEffect, useState } from 'react'
import Icon from './Icon'
import TrendChart from './TrendChart'
import { NAS_API, requestJSON } from '../services/api'

const DEMO_HARDWARE = { cpu_usage: 44.2, gpu_usage: null, ram_usage: 56.1, cpu_temp: 37.5, gpu_temp: 39, chip_temp: 51, load_avg: [1.79, 0.61, 0.21] }
const DEMO_NETWORK = { rx_speed: 9302, tx_speed: 154148, rx_bytes: 30895924, tx_bytes: 26879618 }
const demoPoints = Array.from({length: 20}, (_, i) => ({
  time: `${String(Math.floor(i / 2)).padStart(2,'0')}:${i % 2 ? '30' : '00'}`,
  cpu: 32 + Math.sin(i / 2) * 9 + i * 0.4, ram: 51 + i * 0.28,
  gpu: null, cpuTemp: 37 + Math.sin(i / 3) * 3, gpuTemp: 39 + Math.cos(i / 4) * 3,
  chipTemp: 49 + Math.sin(i / 3) * 3, rx: 8 + Math.abs(Math.sin(i)) * 13, tx: 90 + Math.abs(Math.cos(i / 3)) * 70,
}))

const valid = (v) => v != null && Number.isFinite(Number(v))
const percent = (v) => valid(v) ? `${Number(v).toFixed(1)}%` : '—'
const degrees = (v) => valid(v) ? `${Number(v).toFixed(1)}°C` : '—'
const number = (v, n=1) => valid(v) ? Number(v).toFixed(n) : '—'
const formatGB = (v) => valid(v) ? `${Number(v).toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 2 })} GB` : '—'

function Metric({ label, value, tone, fraction, note }) {
  const width = valid(fraction) ? `${Math.max(0, Math.min(100, Number(fraction)))}%` : '0%'
  return <div className={`v6-hw-metric ${tone}`}><span>{label}</span><strong>{value}</strong><div className="v6-progress"><i style={{width}} /></div><small>{note || (value === '—' ? 'ไม่มีข้อมูลจากอุปกรณ์' : 'ข้อมูลล่าสุด')}</small></div>
}
function StatusHeader({ title, subtitle, icon, status, demo, onToggle, lastUpdated }) {
  return <div className="v6-page-heading"><div className="v6-title-icon"><span className="v6-icon"><Icon name={icon} size={22}/></span><div><h2>{title}</h2><p>{subtitle}</p></div></div><div className="v6-heading-actions"><span className={`v6-status ${demo?'demo':status==='live'?'online':'neutral'}`}>{demo?'DEMO · ข้อมูลจำลอง':status==='live'?'API Connected':'รอข้อมูลจาก API'}</span><small>{lastUpdated ? `อัปเดต ${lastUpdated.toLocaleTimeString('th-TH')}` : 'อัปเดตล่าสุด: —'}</small><button type="button" className="v6-button subtle" aria-pressed={demo} onClick={onToggle}>{demo?'ใช้ข้อมูลจริง':'ดูข้อมูลตัวอย่าง'}</button></div></div>
}

export function HardwarePage({data, status, history, lastUpdated}) {
  const [demo, setDemo] = useState(false)
  const hw = demo ? DEMO_HARDWARE : status === 'live' ? data?.hardware : null
  const points = demo ? demoPoints : history || []
  const loading = !demo && status !== 'live'
  const specs = [
    ['CPU Usage', hw?.cpu_usage, 'cpu', percent, hw?.cpu_usage],
    ['GPU Usage', hw?.gpu_usage, 'gpu', percent, hw?.gpu_usage],
    ['RAM Usage', hw?.ram_usage, 'ram', percent, hw?.ram_usage],
    ['CPU Temperature', hw?.cpu_temp, 'temperature', degrees, valid(hw?.cpu_temp) ? Number(hw.cpu_temp)/90*100:null],
    ['GPU Temperature', hw?.gpu_temp, 'gpu', degrees, valid(hw?.gpu_temp)?Number(hw.gpu_temp)/90*100:null],
    ['Chip Temperature', hw?.chip_temp, 'temperature', degrees, valid(hw?.chip_temp)?Number(hw.chip_temp)/90*100:null],
  ]
  return <div className="v6-page-stack">
    <section className="panel v6-card"><StatusHeader title="Hardware Monitor" subtitle="Jetson Nano · System Performance" icon="cpu" status={status} demo={demo} onToggle={()=>setDemo(v=>!v)} lastUpdated={demo?null:lastUpdated}/>
      <div className="v6-hw-grid">{specs.map(([label,value,tone,formatter,fraction])=><Metric key={label} label={label} value={formatter(value)} tone={tone} fraction={fraction}/>)}</div>
      <div className="v6-load"><div><h3>Load Average</h3><p>ค่าเฉลี่ยภาระงานของระบบ ไม่ใช่เปอร์เซ็นต์ CPU</p></div><div className="v6-load-grid">{['1 นาที','5 นาที','15 นาที'].map((label,i)=><div key={label}><b>{number(hw?.load_avg?.[i],2)}</b><small>{label}</small></div>)}</div></div>
      <p className="v6-muted">ค่าที่ API ไม่ส่งมาจะแสดง — ไม่ใช่ 0 และเกณฑ์เตือนอุณหภูมิต้องสอบเทียบตามอุปกรณ์ก่อนนำไปใช้จริง {loading?'· กำลังรอการเชื่อมต่อ Jetson':''}</p>
    </section>
    <div className="v6-chart-grid"><TrendChart title="CPU & RAM Usage · ย้อนหลัง" points={points} min={0} max={100} unit="%" lines={[{field:'cpu', label:'CPU', color:'#2563eb'},{field:'ram',label:'RAM',color:'#f97316'}]}/><TrendChart title="Hardware Temperature · ย้อนหลัง" points={points} min={0} max={100} unit="°C" lines={[{field:'cpuTemp',label:'CPU',color:'#2563eb'},{field:'gpuTemp',label:'GPU',color:'#a855f7'}]}/></div>
  </div>
}

export function NetworkPage({data, status, history, latency, lastUpdated}) {
  const [demo, setDemo] = useState(false)
  const [nasStorage, setNasStorage] = useState(null)
  const [nasLoading, setNasLoading] = useState(true)
  const [nasError, setNasError] = useState(false)
  const [nasUpdatedAt, setNasUpdatedAt] = useState(null)

  // NAS storage is independent of Jetson's main.py / realtime API.
  // Refresh every 30 seconds while this page is mounted.
  useEffect(() => {
    const controller = new AbortController()

    async function loadNasStorage() {
      try {
        const payload = await requestJSON(`${NAS_API}/system/storage`, {
          signal: controller.signal,
        })
        if (controller.signal.aborted) return
        const result = payload?.data ?? payload
        if (!result || typeof result !== 'object' || !valid(result.total_gb)) {
          throw new Error('Invalid NAS storage response')
        }
        setNasStorage(result)
        setNasError(false)
        setNasUpdatedAt(new Date())
      } catch (error) {
        if (!controller.signal.aborted) setNasError(true)
      } finally {
        if (!controller.signal.aborted) setNasLoading(false)
      }
    }

    loadNasStorage()
    const timer = setInterval(loadNasStorage, 30000)
    return () => {
      clearInterval(timer)
      controller.abort()
    }
  }, [])

  const net = demo ? DEMO_NETWORK : status === 'live' ? data?.network : null
  const hw = demo ? {disk_free:41.2}: status === 'live' ? data?.hardware : null
  const points = demo ? demoPoints : history || []
  const free = valid(hw?.disk_free) ? Number(hw.disk_free) : null
  const total = valid(hw?.disk_total) && Number(hw.disk_total)>0 ? Number(hw.disk_total) : null
  const used = valid(hw?.disk_used) ? Number(hw.disk_used) : total != null && free != null ? Math.max(0,total-free) : null
  const usedPercent = total != null && used != null ? used/total*100 : null
  const metrics = [
    ['Download Speed (RX)', valid(net?.rx_speed) ? `${number(net.rx_speed/1024)} KB/s`: '—', 'download'],
    ['Upload Speed (TX)', valid(net?.tx_speed) ? `${number(net.tx_speed/1024)} KB/s`: '—', 'upload'],
    ['API Response Time', demo?'12 ms':latency == null || status !== 'live'?'—':`${latency} ms`, 'api'],
    ['Disk Free (Jetson)', free == null ? '—':`${free.toFixed(1)} GB`, 'disk'],
  ]
  const nasPercent = valid(nasStorage?.usage_percent) ? Number(nasStorage.usage_percent) : null
  const nasVideoSize = valid(nasStorage?.videos_used_mb) ? `${number(nasStorage.videos_used_mb)} MB` : valid(nasStorage?.videos_used_gb) ? formatGB(nasStorage.videos_used_gb) : '—'

  return <div className="v6-page-stack">
    <section className="panel v6-card"><StatusHeader title="Network & Storage Monitor" subtitle="ติดตามการรับส่งข้อมูลและพื้นที่จัดเก็บของ Jetson Nano" icon="globe" status={status} demo={demo} onToggle={()=>setDemo(v=>!v)} lastUpdated={demo?null:lastUpdated}/><div className="v6-network-metrics">{metrics.map(([label,value,tone])=><div className={`v6-network-metric ${tone}`} key={label}><small>{label}</small><strong>{value}</strong><span>{tone==='api'?'เวลาตอบกลับ HTTP ไม่ใช่ค่า Ping':tone==='disk'?'พื้นที่ว่างบนอุปกรณ์ Edge':'อัตราข้อมูลรับส่ง (KB/s)'}</span></div>)}</div></section>
    <section className="panel v6-card"><div className="v6-card-header"><div><h3>Network Traffic</h3><p>เปรียบเทียบ Download (RX) และ Upload (TX) จากค่าความเร็ว Bytes/s</p></div><span className="v6-status neutral">ล่าสุดสูงสุด 35 จุด (~70 วินาที)</span></div><TrendChart title="ความเร็วเครือข่าย (KB/s)" points={points} lines={[{field:'rx',label:'Download RX',color:'#2563eb'},{field:'tx',label:'Upload TX',color:'#10b981'}]} /><p className="v6-muted">ไม่ใช้ยอดข้อมูลสะสม rx_bytes/tx_bytes เป็นความเร็ว</p></section>
    <section className="panel v6-card">
      <div className="v6-card-header"><div><h3>Storage Information</h3><p>แยกพื้นที่จัดเก็บบน Jetson Nano กับ Server/NAS</p></div></div>
      <div className="v6-storage-grid">
        <div className="v6-storage-card">
          <div className="v6-card-header"><h4>Jetson Nano · Local Storage</h4><span className="v6-status neutral">{total==null?'ยังไม่มี disk_total':'ข้อมูลจาก API'}</span></div>
          <strong>{usedPercent==null?'—':`${usedPercent.toFixed(1)}%`} <small>Used</small></strong>
          <div className="v6-progress"><i style={{width:usedPercent==null?'0%':`${Math.max(0,Math.min(100,usedPercent))}%`}}/></div>
          <div className="v6-storage-sub"><span>Used: {used==null?'—':used.toFixed(1)} GB</span><span>Free: {free==null?'—':free.toFixed(1)} GB</span></div>
          <p>{total == null ? 'ยังไม่ได้รับข้อมูลพื้นที่ทั้งหมดจาก Jetson API' : `Total: ${formatGB(total)} · ข้อมูลพื้นที่จาก Jetson API`}</p>
        </div>
        <div className="v6-storage-card">
          <div className="v6-card-header"><h4>Server / NAS · Recording Storage</h4><span className={`v6-status ${nasStorage && !nasError ? 'online' : 'neutral'}`}>{nasLoading && !nasStorage ? 'กำลังโหลด...' : nasError ? 'เชื่อมต่อ API ไม่สำเร็จ' : nasStorage ? 'ข้อมูลจาก NAS API' : 'รอข้อมูลจาก API'}</span></div>
          <strong>{nasPercent==null?'—':`${nasPercent.toFixed(1)}%`} <small>Used</small></strong>
          <div className="v6-progress"><i style={{width:nasPercent==null?'0%':`${Math.max(0,Math.min(100,nasPercent))}%`}}/></div>
          <div className="v6-storage-sub"><span>Used: {formatGB(nasStorage?.used_gb)}</span><span>Free: {formatGB(nasStorage?.free_gb)}</span></div>
          <p>Total: {formatGB(nasStorage?.total_gb)}</p>
          <p>พื้นที่เก็บวิดีโอ: <strong>{nasVideoSize}</strong></p>
          <p>{nasError ? 'ไม่สามารถอัปเดตข้อมูลจาก NAS ได้ (อาจยังแสดงค่าล่าสุด)' : nasUpdatedAt ? `อัปเดตล่าสุด ${nasUpdatedAt.toLocaleTimeString('th-TH')} · รีเฟรชทุก 30 วินาที` : 'กำลังรอข้อมูลพื้นที่จาก NAS API'}</p>
        </div>
      </div>
    </section>
  </div>
}
