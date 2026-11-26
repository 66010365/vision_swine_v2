import DashboardLayout from "./components/layout/DashboardLayout.jsx";
import VideoRecordings from "./pages/VideoRecordings.jsx";

// Standalone frontend: only Video Recordings, no other dashboard routes.
export default function App() {
  return (
    <DashboardLayout>
      <VideoRecordings />
    </DashboardLayout>
  );
}
