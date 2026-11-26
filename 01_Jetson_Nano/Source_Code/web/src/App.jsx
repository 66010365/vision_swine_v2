import { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import LiveCamera from './components/LiveCamera'
import { PigCounter, HardwareMini } from './components/LiveOverview'
import { HardwarePage, NetworkPage } from './components/SystemPages'
import SettingsPage from './components/SettingsPage'
import RecordsPage from './components/RecordsPage'
import { useRealtime } from './hooks/useRealtime'
import { useRecords } from './hooks/useRecords'
import './App.css'

function LiveDashboard({ realtime, records }) {
  const { data, status } = realtime

  return (
    <section className="live-dashboard">
      <LiveCamera realtimeStatus={status} fps={data?.fps} />
      <div className="overview-column">
        <PigCounter data={data} status={status} />
        <HardwareMini data={data} status={status} />
        <div className="overview-footnote">
          <span>ประวัติบน NAS: {records.events.length} เหตุการณ์</span>
          <span>การเชื่อมต่อจริงขึ้นอยู่กับสถานะอุปกรณ์</span>
        </div>
      </div>
    </section>
  )
}

function PageHeading({ activeTab, realtimeStatus }) {
  const [clock, setClock] = useState(() => new Date())

  useEffect(() => {
    const id = setInterval(() => setClock(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  const pageNames = {
    dashboard: 'กล้องสดเรียลไทม์',
    'ai-settings': 'ตั้งค่าระบบ AI',
    'board-status': 'สถานะบอร์ด Hardware',
    'network-status': 'เครือข่ายและการใช้ดิสก์',
    'history-logs': 'ประวัติและวิดีโอ',
  }

  return (
    <header className="page-header">
      <div className="heading-brand">
        <span className="heading-pig">🐷</span>
        <div>
          <h1>PigVision Studio</h1>
          <p>Jetson Nano Edge AI Inference System</p>
        </div>
      </div>
      <div className="header-right">
        <div className="system-caption">
          <small>System Status</small>
          <strong className={realtimeStatus === 'live' ? 'online-text' : 'offline-text'}>
            <span className={`status-dot ${realtimeStatus === 'live' ? 'online' : ''}`} />
            {realtimeStatus === 'live' ? 'Jetson API เชื่อมต่อแล้ว' : 'กำลังรอการเชื่อมต่อ Jetson'}
          </strong>
          <span className="current-section">{pageNames[activeTab]}</span>
        </div>
        <div className="clock">{clock.toLocaleTimeString('th-TH', { hour12: false })}</div>
      </div>
    </header>
  )
}

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const realtime = useRealtime()
  const records = useRecords()

  return (
    <div className="app-shell">
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        liveStatus={realtime.status}
      />
      <main className="main-content">
        <PageHeading activeTab={activeTab} realtimeStatus={realtime.status} />

        {activeTab === 'dashboard' && <LiveDashboard realtime={realtime} records={records} />}
        {activeTab === 'ai-settings' && <SettingsPage />}
        {activeTab === 'board-status' && <HardwarePage data={realtime.data} status={realtime.status} history={realtime.history} lastUpdated={realtime.lastUpdated} />}
        {activeTab === 'network-status' && <NetworkPage data={realtime.data} status={realtime.status} history={realtime.history} latency={realtime.latency} lastUpdated={realtime.lastUpdated} />}
        {activeTab === 'history-logs' && <RecordsPage records={records} />}

        <footer className="page-footer">
          <span>VISION SWINE · REACT v6 FULL UI</span>
          <span>Jetson Nano · React / Vite · NAS PostgreSQL</span>
        </footer>
      </main>
    </div>
  )
}

export default App
