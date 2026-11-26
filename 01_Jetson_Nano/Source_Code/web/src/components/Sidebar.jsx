import Icon from './Icon'

const navItems = [
  { id: 'dashboard', text: 'กล้องสดเรียลไทม์', icon: 'video' },
  { id: 'ai-settings', text: 'ตั้งค่าระบบ AI', icon: 'settings' },
  { id: 'board-status', text: 'สถานะบอร์ด Hardware', icon: 'cpu' },
  { id: 'network-status', text: 'เครือข่ายและการใช้ดิสก์', icon: 'globe' },
  { id: 'history-logs', text: 'ประวัติและวิดีโอ', icon: 'history' },
]

export default function Sidebar({ activeTab, onTabChange, liveStatus }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-logo">🐷</div>
        <div className="brand-text">
          <strong>PigVision</strong>
          <small>JETSON NANO AI SYSTEM</small>
        </div>
      </div>

      <div className="nav-label">WORKSPACE</div>
      <nav className="sidebar-nav" aria-label="เมนูหลัก">
        {navItems.map(({ id, text, icon }) => (
          <button
            key={id}
            type="button"
            className={`nav-item ${activeTab === id ? 'active' : ''}`}
            onClick={() => onTabChange(id)}
            aria-current={activeTab === id ? 'page' : undefined}
            title={text}
          >
            <Icon name={icon} size={18} />
            <span>{text}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span className={`status-dot ${liveStatus === 'live' ? 'online' : ''}`} />
        <div>
          <strong>JETSON NANO</strong>
          <small>{liveStatus === 'live' ? 'Realtime API Online' : 'Waiting for Edge AI'}</small>
        </div>
      </div>
    </aside>
  )
}
