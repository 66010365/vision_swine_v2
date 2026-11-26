import Sidebar from "./Sidebar.jsx";

export default function DashboardLayout({ children }) {
  return (
    <div className="vs-layout">
      <Sidebar />
      <main className="vs-main">{children}</main>
      <style>{`

        .vs-main {
          min-height: 100vh;
          margin-left: 230px;
          padding: 0 24px 42px;
          background:
            radial-gradient(ellipse 530px 350px at 95% 0%,rgba(206,227,250,.30),transparent 95%),
            linear-gradient(180deg,#f7f9fd 0%,#f3f7fb 52%,#f4f7fb 100%);
        }
        @media(max-width:800px) {
          .vs-main {margin-left:68px;padding:0 16px 30px;}
        }
      `}</style>
    </div>
  );
}
