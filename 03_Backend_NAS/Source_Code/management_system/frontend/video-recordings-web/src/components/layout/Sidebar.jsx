export default function Sidebar() {
  return (
    <aside className="vs-sidebar">
      <div className="vs-brand">
        <div className="vs-brand-mark">VS</div>
        <div>
          <strong>VISION SWINE</strong>
          <span>AI Monitoring System</span>
        </div>
      </div>

      <nav className="vs-nav">
        <div className="vs-nav-title">MAIN MENU</div>

        <a href="/" className="vs-nav-item active" aria-current="page">
          <span className="vs-nav-icon">▶</span>
          <span>Video Recordings</span>
        </a>
      </nav>

      <div className="vs-sidebar-footer">
        <div className="vs-system-dot" />
        <div>
          <strong>UI Preview</strong>
          <span>Vision Swine</span>
        </div>
      </div>
      <style>{`

        .vs-sidebar {
          position: fixed;
          inset: 0 auto 0 0;
          z-index: 50;
          width: 230px;
          display: flex;
          flex-direction: column;
          background: linear-gradient(180deg,#ffffff 0%,#f9fbfe 100%);
          border-right: 1px solid #e8edf5;
          box-shadow: 5px 0 25px rgba(29,57,95,.025);
        }
        .vs-brand {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 27px 18px 25px;
          border-bottom: 1px solid #edf1f7;
        }
        .vs-brand-mark {
          width: 40px;
          height: 40px;
          display: grid;
          place-items: center;
          flex: 0 0 auto;
          border-radius: 13px;
          background: linear-gradient(145deg,#7fa9d7,#4f7fb7);
          color: #fff;
          font-size: 13px;
          font-weight: 750;
          box-shadow: 0 6px 16px rgba(84,132,191,.17);
        }
        .vs-brand strong {display: block;color:#294364;font-size:13px;font-weight:750;letter-spacing:.35px;}
        .vs-brand span {display:block;margin-top:3px;color:#93a0b2;font-size:10px;}
        .vs-nav {flex:1;padding:26px 13px;}
        .vs-nav-title {
          padding:0 11px 11px;
          color:#9babc0;
          font-size:10px;
          font-weight:650;
          letter-spacing:1px;
        }
        .vs-nav-item {
          display:flex;
          align-items:center;
          gap:11px;
          padding:12px 13px;
          margin-bottom:5px;
          border:1px solid transparent;
          border-radius:13px;
          color:#6a7e97;
          text-decoration:none;
          font-size:13px;
          font-weight:550;
          transition:background .2s ease,color .2s ease,border-color .2s ease;
        }
        .vs-nav-item:hover {background:#f5f8fd;color:#3e658f;}
        .vs-nav-item.active {
          background:linear-gradient(115deg,#eef5fe,#f3f8ff);
          border-color:#e0ebfa;
          color:#4477b3;
          font-weight:650;
        }
        .vs-nav-icon {width:22px;text-align:center;font-size:14px;line-height:1;color:#5f90c9;}
        .vs-sidebar-footer {
          display:flex;
          align-items:center;
          gap:11px;
          margin:13px;
          padding:13px 12px;
          border:1px solid #e9eff7;
          border-radius:14px;
          background:#f7fafd;
        }
        .vs-system-dot {
          width:8px;height:8px;flex:0 0 auto;border-radius:50%;
          background:#81a9d1;box-shadow:0 0 0 4px #e7f0fa;
        }
        .vs-sidebar-footer strong {display:block;font-size:11px;font-weight:650;color:#58718f;}
        .vs-sidebar-footer span {display:block;margin-top:3px;font-size:10px;color:#94a3b8;}
        @media (max-width:800px) {
          .vs-sidebar {width:68px;}
          .vs-brand {justify-content:center;padding:18px 8px;}
          .vs-brand > div:last-child,
          .vs-nav-title,
          .vs-nav-item span:last-child,
          .vs-sidebar-footer > div:last-child {display:none;}
          .vs-nav {padding:20px 8px;}
          .vs-nav-item {justify-content:center;padding:12px 8px;}
          .vs-sidebar-footer {justify-content:center;padding:12px 8px;}
        }
        @media (prefers-reduced-motion:reduce) { .vs-nav-item {transition:none;} }
      `}</style>
    </aside>
  );
}
