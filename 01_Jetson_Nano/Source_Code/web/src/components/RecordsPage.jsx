import { useMemo, useState } from 'react'
import Icon from './Icon'
import { nasAPI } from '../services/api'

const demoEvents = [
  {event_id:'demo1',event_time:'2026-10-29T10:24:31',direction:'IN',track_id:1,weight_kg:118.5,frame_number:240,video_id:'demo-1'},
  {event_id:'demo2',event_time:'2026-10-29T10:21:05',direction:'OUT',track_id:2,weight_kg:125,frame_number:220,video_id:'demo-2'},
  {event_id:'demo3',event_time:'2026-10-29T10:18:42',direction:'IN',track_id:3,weight_kg:121.4,frame_number:190,video_id:'demo-1'},
]
const demoVideos = [
  {video_id:'demo-1',file_name:'record_102431.mp4',created_at:'2026-10-29T10:24:31',duration_seconds:180},
  {video_id:'demo-2',file_name:'record_102105.mp4',created_at:'2026-10-29T10:21:05',duration_seconds:180},
]
function timeLabel(value) {
  if (!value) return '—'
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString('th-TH', {hour12:false})
}
function weightLabel(value) { return value == null || !Number.isFinite(Number(value)) ? '—' : `${Number(value).toFixed(1)} กก.` }
function videoID(item) { return item?.video_id ?? item?.recording_id ?? item?.id }
function videoName(item) { return item?.file_name ?? item?.filename ?? item?.original_filename ?? `Video #${videoID(item) ?? '—'}` }
function eventTime(item) { return item?.event_time ?? item?.timestamp ?? item?.created_at }
function videoTime(item) { return item?.created_at ?? item?.recorded_at ?? item?.started_at }
function clipRef(item) { return item?.video_id ?? item?.recording_id ?? null }

export default function RecordsPage({records}) {
  const {events, videos, weights, loading, error, refresh, lastUpdated} = records
  const [tab,setTab] = useState('events')
  const [demo,setDemo] = useState(false)
  const [query,setQuery] = useState('')
  const [direction,setDirection] = useState('ALL')
  const [date,setDate] = useState('')
  const [selectedVideoId,setSelectedVideoId] = useState(null)
  const allEvents = demo ? demoEvents : events
  const allVideos = demo ? demoVideos : videos
  const filteredEvents = useMemo(()=> [...allEvents].filter((e)=>{
    const search = query.trim().toLowerCase()
    const id = String(e.track_id ?? e.pig_id ?? '')
    const dir = String(e.direction ?? '').toUpperCase()
    const tm = eventTime(e)
    const searchMatch = !search || id.toLowerCase().includes(search)
    const directionMatch = direction==='ALL' || dir === direction
    const dateMatch = !date || (tm && !Number.isNaN(new Date(tm).getTime()) && new Date(tm).toLocaleDateString('en-CA')===date)
    return searchMatch && directionMatch && dateMatch
  }).sort((a,b)=>new Date(eventTime(b)||0)-new Date(eventTime(a)||0)).slice(0,100), [allEvents,query,direction,date])
  const sortedVideos = useMemo(()=>[...allVideos].sort((a,b)=>new Date(videoTime(b)||0)-new Date(videoTime(a)||0)).slice(0,100),[allVideos])
  const selectedVideo = sortedVideos.find((v)=>String(videoID(v))===String(selectedVideoId))
  function showVideo(id) { if(id==null) return; setSelectedVideoId(id); setTab('videos') }
  return <div className="v6-page-stack">
    <section className="panel v6-card v6-records-header">
      <div className="v6-card-header"><div><small className="v6-eyebrow">VISION SWINE / RECORDS</small><h2>ประวัติและวิดีโอบันทึก</h2><p>ตรวจสอบเหตุการณ์นับสุกรและไฟล์บันทึกจาก NAS</p></div><span className={`v6-status ${demo?'demo':error?'neutral':'online'}`}>{demo?'DEMO · ข้อมูลตัวอย่าง':error?'เชื่อม API บางส่วนไม่ได้':loading?'กำลังโหลด':'ข้อมูลจาก NAS'}</span></div>
      <div className="v6-records-toolbar"><div className="v6-tabs" role="tablist" aria-label="ประเภทประวัติ"><button role="tab" type="button" aria-selected={tab==='events'} className={tab==='events'?'active':''} onClick={()=>setTab('events')}><Icon name="history" size={16}/> ประวัติการนับ</button><button role="tab" type="button" aria-selected={tab==='videos'} className={tab==='videos'?'active':''} onClick={()=>setTab('videos')}><Icon name="video" size={16}/> วิดีโอบันทึก</button></div><div className="v6-records-actions"><button type="button" className="v6-button subtle" aria-pressed={demo} onClick={()=>{setDemo(v=>!v);setSelectedVideoId(null)}}>{demo?'กลับไปข้อมูลจริง':'ดูข้อมูลตัวอย่าง'}</button><button type="button" className="v6-button subtle" disabled={loading || demo} onClick={()=>refresh()}><Icon name="refresh" size={15}/> อัปเดต</button></div></div>
      {demo && <p className="v6-demo-note">DEMO — Tracking ID, น้ำหนัก และชื่อไฟล์เป็นข้อมูลสมมติ ไม่ใช่ข้อมูลจาก NAS และไม่สามารถเล่นไฟล์ DEMO ได้</p>}
      {error && !demo && <p role="alert" className="records-error">{error} — ตรวจสอบ API บน NAS และ CORS</p>}
    </section>
    {tab==='events' ? <section className="panel v6-card">
      <div className="v6-card-header"><div><h3>Event History</h3><p>ตรวจสอบการนับเข้า–ออกและน้ำหนักที่สัมพันธ์กับเหตุการณ์</p></div><span className="v6-status neutral">{filteredEvents.length} รายการ</span></div>
      <div className="v6-filters"><label><span>ค้นหา Tracking ID</span><input value={query} onChange={(e)=>setQuery(e.target.value)} placeholder="เช่น 12" /></label><label><span>ประเภทเหตุการณ์</span><select value={direction} onChange={(e)=>setDirection(e.target.value)}><option value="ALL">ทั้งหมด</option><option value="IN">ขาเข้า</option><option value="OUT">ขาออก</option></select></label><label><span>วันที่</span><input type="date" value={date} onChange={(e)=>setDate(e.target.value)} /></label><button type="button" className="v6-button subtle" onClick={()=>{setQuery('');setDirection('ALL');setDate('')}}>ล้างตัวกรอง</button></div>
      <div className="table-scroll"><table className="records-table v6-records-table"><thead><tr><th>วัน–เวลา</th><th>เหตุการณ์</th><th>Tracking ID</th><th>น้ำหนัก (กก.)</th><th>Frame</th><th>วิดีโอ</th></tr></thead><tbody>{filteredEvents.map((event,index)=><tr key={event.event_id??index}><td>{timeLabel(eventTime(event))}</td><td><span className={`direction-label ${String(event.direction).toUpperCase()==='IN'?'in':'out'}`}>{event.direction==='IN'?'↑ เข้า':event.direction==='OUT'?'↓ ออก':event.direction??'—'}</span></td><td>PIG-{String(event.track_id??'—').padStart(3,'0')}</td><td>{weightLabel(event.weight_kg)}</td><td>{event.frame_number??'—'}</td><td>{clipRef(event)!=null&&sortedVideos.some(v=>String(videoID(v))===String(clipRef(event)))?<button type="button" className="v6-link-button" onClick={()=>showVideo(clipRef(event))}>ดูคลิป</button>:'—'}</td></tr>)}{filteredEvents.length===0&&<tr><td colSpan="6" className="empty-table">{loading&&!demo?'กำลังโหลดประวัติจาก NAS...':'ไม่มีข้อมูลตามตัวกรอง หรือยังไม่ได้เชื่อมต่อ NAS'}</td></tr>}</tbody></table></div>
      <div className="records-footnote"><span>แสดงสูงสุด 100 รายการล่าสุด · /counting-events</span><span>{lastUpdated&&!demo?`อัปเดต ${lastUpdated.toLocaleTimeString('th-TH')}`:'—'}</span></div>
    </section> : <section className="panel v6-card"><div className="v6-card-header"><div><h3>Video Records</h3><p>เลือกคลิป MP4 จาก NAS และตรวจสอบรายละเอียดการบันทึก</p></div><span className="v6-status neutral">{sortedVideos.length} คลิป</span></div><div className="v6-video-grid"><div className="v6-video-list">{sortedVideos.map((video,index)=>{const id=videoID(video);return <button key={id??index} type="button" className={`video-list-item ${String(selectedVideoId)===String(id)?'selected':''}`} onClick={()=>setSelectedVideoId(id)} disabled={id==null}><span className="video-list-icon"><Icon name="play" size={17}/></span><span className="video-list-info"><strong>{videoName(video)}</strong><small>{timeLabel(videoTime(video))}</small></span><span className="video-list-action">เลือก</span></button>})}{sortedVideos.length===0&&<p className="empty-records">{loading&&!demo?'กำลังโหลดวิดีโอ...':'ไม่มีวิดีโอ หรือยังไม่สามารถเชื่อม NAS ได้'}</p>}</div><div className="v6-video-player">{selectedVideo&&!demo&&videoID(selectedVideo)!=null?<><video key={String(videoID(selectedVideo))} controls playsInline preload="metadata" src={nasAPI.videoFile(videoID(selectedVideo))}>เบราว์เซอร์ไม่รองรับวิดีโอ</video><div className="v6-video-meta"><b>{videoName(selectedVideo)}</b><span>{timeLabel(videoTime(selectedVideo))}</span><small>เล่นผ่าน NAS API · ต้องเข้าถึงไฟล์และรองรับ Codec</small></div></>:<div className="video-empty"><Icon name="video" size={36}/><strong>{demo?'ข้อมูลตัวอย่างไม่มีไฟล์วิดีโอจริง':'เลือกวิดีโอจากรายการ'}</strong><small>{demo?'สลับกลับเป็นข้อมูลจริงเพื่อเปิดไฟล์จาก NAS':'พื้นที่เล่นไฟล์ MP4 จาก NAS เมื่อเลือกคลิป'}</small></div>}</div></div></section>}
    <p className="v6-muted">ข้อมูลน้ำหนักที่ดึงได้จาก NAS: {demo?'ตัวอย่าง':weights.length} รายการ · หน้านี้ไม่ได้ลบหรือแก้ข้อมูลใน PostgreSQL</p>
  </div>
}
