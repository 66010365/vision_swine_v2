
/**
 * VISION SWINE V2 - API Services
 *
 * Jetson Nano (8080):
 * - MJPEG Streaming
 * - Real-time Monitoring
 * - Gate Configuration
 *
 * NAS FastAPI (8002):
 * - Weight Records
 * - Counting Events
 * - Processing Sessions
 * - Recorded Videos
 *
 * API requests use Vite Proxy to avoid browser CORS issues.
 */

// ======================================================
// 01. API Configuration
// ======================================================

const withoutTrailingSlash = (url) => url.replace(/\/+$/, '')

export const JETSON_API = withoutTrailingSlash(
  import.meta.env.VITE_JETSON_API || '/jetson-api'
)

export const NAS_API = withoutTrailingSlash(
  import.meta.env.VITE_NAS_API || '/nas-api'
)

export const STREAM_URL = `${JETSON_API}/stream`

// ======================================================
// 02. HTTP JSON Request
// ======================================================

export async function requestJSON(url, options = {}) {
  const response = await fetch(url, {
    cache: 'no-store',
    ...options,
  })

  const raw = await response.text()

  let result = null

  if (raw) {
    try {
      result = JSON.parse(raw)
    } catch {
      throw new Error(`Invalid JSON from ${url}`)
    }
  }

  if (!response.ok) {
    throw new Error(
      result?.detail ||
      result?.error ||
      `HTTP ${response.status}`
    )
  }

  return result
}

// ======================================================
// 03. Jetson Nano API
// ======================================================

export const jetsonAPI = {
  realtime: (signal) =>
    requestJSON(`${JETSON_API}/api/system/realtime`, { signal }),

  getGate: (signal) =>
    requestJSON(`${JETSON_API}/api/config/gate`, { signal }),

  saveGate: (gate) =>
    requestJSON(`${JETSON_API}/api/config/gate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(gate),
    }),
}

// ======================================================
// 04. NAS Backend API
// ======================================================

export const nasAPI = {
  weights: (signal) =>
    requestJSON(`${NAS_API}/weights`, { signal }),

  events: (signal) =>
    requestJSON(`${NAS_API}/counting-events`, { signal }),

  sessions: (signal) =>
    requestJSON(`${NAS_API}/sessions`, { signal }),

  videos: (signal) =>
    requestJSON(`${NAS_API}/videos`, { signal }),

  videoFile: (videoId) =>
    `${NAS_API}/videos/${encodeURIComponent(videoId)}/file`,
}

// ======================================================
// 05. Response Helpers
// ======================================================

export function unwrapData(response) {
  return response?.data ?? response
}

export function rowsFrom(response) {
  const data = unwrapData(response)

  return Array.isArray(data)
    ? data
    : Array.isArray(data?.items)
      ? data.items
      : []
}
