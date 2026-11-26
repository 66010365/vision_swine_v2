import { useEffect, useMemo, useState } from "react";
import PageHeader from "../components/ui/PageHeader.jsx";
import PageBoard, { Block } from "../components/board/PageBoard.jsx";
import Card from "../components/ui/Card.jsx";
import "./VideoRecordings.css";

// The backend returns video metadata, but not farm/barn/camera or duration.
// Never fill missing metadata with the former mock values.
const displayDate = new Intl.DateTimeFormat("th-TH", {
  day: "2-digit",
  month: "short",
  year: "numeric",
  timeZone: "Asia/Bangkok",
});
const displayTime = new Intl.DateTimeFormat("th-TH", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
  timeZone: "Asia/Bangkok",
});

function formatBytes(bytes) {
  const size = Number(bytes);
  if (!Number.isFinite(size) || size < 0) return "ไม่ทราบ";
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const n = Math.floor(seconds);
  const hours = Math.floor(n / 3600);
  const minutes = Math.floor((n % 3600) / 60);
  const remaining = String(n % 60).padStart(2, "0");
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${remaining}`
    : `${String(minutes).padStart(2, "0")}:${remaining}`;
}

function normalizeVideo(item) {
  if (item == null || item.video_id == null) return null;
  const dateValue = item.created_at ? new Date(item.created_at) : null;
  const validDate = dateValue && !Number.isNaN(dateValue.getTime());
  const filename = typeof item.file_name === "string" && item.file_name.trim()
    ? item.file_name
    : `Video #${item.video_id}`;

  return {
    id: String(item.video_id),
    title: filename,
    filename,
    date: validDate ? displayDate.format(dateValue) : "ไม่ทราบ",
    time: validDate ? displayTime.format(dateValue) : "ไม่ทราบ",
    size: formatBytes(item.file_size_bytes),
    status: String(item.status || "UNKNOWN"),
    sessionId: item.session_id || "ไม่ทราบ",
  };
}

export default function VideoRecordings() {
  const [videos, setVideos] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [playbackError, setPlaybackError] = useState("");
  const [durationById, setDurationById] = useState({});
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setLoadError("");

    async function getVideos() {
      try {
        const response = await fetch("/videos", {
          signal: controller.signal,
          cache: "no-store",
          headers: { Accept: "application/json" },
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!Array.isArray(data)) throw new Error("API ส่งข้อมูลที่ไม่ใช่รายการวิดีโอ");
        const items = data.map(normalizeVideo).filter(Boolean);
        if (controller.signal.aborted) return;
        setVideos(items);
        setSelectedId((current) =>
          items.some((video) => video.id === current)
            ? current
            : (items.find((video) => video.status === "READY") || items[0])?.id ?? null,
        );
      } catch (error) {
        if (!controller.signal.aborted) {
          setLoadError(error instanceof Error ? error.message : "ไม่สามารถเชื่อมต่อ API ได้");
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    getVideos();
    return () => controller.abort();
  }, [refresh]);

  const selectedVideo = useMemo(
    () => videos.find((video) => video.id === selectedId) || null,
    [videos, selectedId],
  );

  useEffect(() => {
    setPlaybackError("");
  }, [selectedId]);

  function reloadVideos() {
    setRefresh((value) => value + 1);
  }

  function selectVideo(video) {
    setSelectedId(video.id);
  }

  return (
    <>
      <PageHeader
        title="Video Recordings"
        subtitle="คลังวิดีโอบันทึกจากระบบ VISION SWINE · เลือกดูและเล่นวิดีโอย้อนหลัง"
        action={
          <div
            className={`vr-nas-status ${loadError ? "is-error" : loading ? "is-loading" : "is-connected"}`}
            role="status"
            aria-live="polite"
            title={loadError ? `เชื่อมต่อ /videos ไม่สำเร็จ: ${loadError}` : "สถานะ API รายการวิดีโอผ่าน /videos"}
          >
            <span className="vr-nas-status-dot" aria-hidden="true" />
            <span>NAS {loadError ? "· เชื่อมต่อไม่ได้" : loading ? "· กำลังเชื่อมต่อ" : "· เชื่อมต่อแล้ว"}</span>
          </div>
        }
      />

      <PageBoard>
        <Block id="wall" label="ค้นหาวิดีโอย้อนหลัง">
          <Card>
            <div className="vr-filters">
              <UnavailableFilter label="ฟาร์ม" id="farm" />
              <UnavailableFilter label="โรงเรือน" id="barn" />
              <UnavailableFilter label="กล้อง" id="camera" />

              <button type="button" className="vr-reload" onClick={reloadVideos} disabled={loading}>
                <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M20 7v5h-5"/><path d="M4 17v-5h5"/>
                  <path d="M5.6 9a7.4 7.4 0 0 1 12.7-2L20 12M4 12l1.7 5a7.4 7.4 0 0 0 12.7-2"/>
                </svg>
                <span>รีโหลดรายการ</span>
              </button>
              <div className="vr-result-count" title="จำนวนรายการที่ API ส่งกลับมา (อาจจำกัดจำนวน)">
                <strong>{videos.length}</strong><span>ไฟล์</span>
              </div>
            </div>
            <div className="vr-filter-note">
              ตัวกรองฟาร์ม โรงเรือน และกล้องยังไม่พร้อมใช้งาน เพราะ API /videos ยังไม่ส่งข้อมูลสำหรับตัวกรองเหล่านี้
            </div>
          </Card>
        </Block>

        <Block id="detected">
          <div className="vr-main-grid">
            <Card
              title="เล่นวิดีโอย้อนหลัง"
              subtitle="เลือกไฟล์จากรายการด้านขวาเพื่อรับชม"
            >
              <div className="vr-player">
                <div className="vr-player-top">
                  <span className="vr-player-id">{selectedVideo ? `VIDEO #${selectedVideo.id}` : "VIDEO PLAYER"}</span>
                  <span className={`vr-file-status ${selectedVideo?.status === "READY" ? "is-ready" : ""}`}>
                    {selectedVideo?.status || (loading ? "กำลังโหลด" : "รอข้อมูล")}
                  </span>
                </div>
                <div className="vr-player-screen">
                  {selectedVideo?.status === "READY" ? (
                    <>
                      <video
                        key={selectedVideo.id}
                        className="vr-native-video"
                        src={`/videos/${encodeURIComponent(selectedVideo.id)}/file`}
                        controls
                        playsInline
                        preload="metadata"
                        onLoadedMetadata={(event) => {
                          const duration = formatDuration(event.currentTarget.duration);
                          setDurationById((current) => ({ ...current, [selectedVideo.id]: duration }));
                          setPlaybackError("");
                        }}
                        onError={() => setPlaybackError("ไม่สามารถเล่นวิดีโอได้ โปรดตรวจสอบไฟล์ MP4 หรือการเชื่อมต่อ API")}
                      />
                      {playbackError && <div className="vr-player-alert">{playbackError}</div>}
                    </>
                  ) : (
                    <div className="vr-player-placeholder">
                      {loading ? "กำลังโหลดรายการวิดีโอ..." :
                        loadError ? "เชื่อมต่อ API ไม่สำเร็จ" :
                        selectedVideo ? "ไฟล์นี้ยังไม่พร้อมเล่น" : "ยังไม่มีวิดีโอให้แสดง"}
                    </div>
                  )}
                </div>
                <div className="vr-player-meta">
                  <div className="vr-player-file">
                    <span className="vr-meta-label">ไฟล์ที่เลือก</span>
                    <strong title={selectedVideo?.filename || ""}>{selectedVideo?.filename || "—"}</strong>
                    <span className="vr-meta-time">{selectedVideo ? `${selectedVideo.date} · ${selectedVideo.time}` : "—"}</span>
                  </div>
                  <div className="vr-player-file-size">
                    <span className="vr-meta-label">ขนาดไฟล์</span>
                    <strong>{selectedVideo?.size || "—"}</strong>
                    {selectedVideo && <span className="vr-meta-time">{durationById[selectedVideo.id] || "—"}</span>}
                  </div>
                </div>
              </div>
            </Card>

            <Card
              title="รายการวิดีโอ"
              subtitle={`${videos.length} ไฟล์ที่ได้รับจาก NAS${videos.length === 100 ? " · API อาจจำกัด 100 รายการ" : ""}`}
            >
              <div className="vr-list">
                {loadError && <div className="vr-message-error">เชื่อมต่อ API ไม่สำเร็จ: {loadError}</div>}
                {loading && videos.length === 0 ? (
                  <div className="vr-empty">กำลังโหลดรายการวิดีโอ...</div>
                ) : videos.length === 0 ? (
                  <div className="vr-empty">{loadError ? "กดรีเฟรชเพื่อลองอีกครั้ง" : "ไม่พบรายการวิดีโอ"}</div>
                ) : (
                  videos.map((video) => (
                    <button
                      type="button"
                      key={video.id}
                      className={`vr-row ${selectedVideo?.id === video.id ? "selected" : ""}`}
                      aria-pressed={selectedVideo?.id === video.id}
                      onClick={() => selectVideo(video)}
                    >
                      <span className="vr-thumb" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M9 6.5 18 12l-9 5.5z"/></svg>
                      </span>
                      <span className="vr-info">
                        <strong title={video.filename}>{video.filename}</strong>
                        <span>{video.date} · {video.time}</span>
                        <small>VIDEO #{video.id} <span aria-hidden="true">·</span> {video.status}</small>
                      </span>
                      <span className="vr-extra"><span>{video.size}</span><span>{durationById[video.id] || "—"}</span></span>
                    </button>
                  ))
                )}
              </div>
            </Card>
          </div>
        </Block>
      </PageBoard>
    </>
  );
}

// Keep the original three-filter layout until the API provides farm/barn/camera fields.
// Do not invent filter options or imply that filtering is operational.
function UnavailableFilter({ label, id }) {
  return (
    <div className="vr-filter">
      <label htmlFor={`vr-${id}`}>{label}</label>
      <select id={`vr-${id}`} disabled value="unavailable" aria-label={`${label}: ยังไม่มีข้อมูลจาก API`}>
        <option value="unavailable">ยังไม่มีข้อมูลจาก API</option>
      </select>
    </div>
  );
}
