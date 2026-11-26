import { useCallback, useEffect, useState } from 'react'
import { nasAPI, rowsFrom } from '../services/api'

export function useRecords() {
  const [weights, setWeights] = useState([])
  const [events, setEvents] = useState([])
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  const refresh = useCallback(async (signal) => {
    setLoading(true)
    try {
      // AllSettled lets the page display any sections that are available.
      const results = await Promise.allSettled([
        nasAPI.weights(signal), nasAPI.events(signal), nasAPI.videos(signal),
      ])
      if (signal?.aborted) return

      const [w, e, v] = results
      if (w.status === 'fulfilled') setWeights(rowsFrom(w.value))
      if (e.status === 'fulfilled') setEvents(rowsFrom(e.value))
      if (v.status === 'fulfilled') setVideos(rowsFrom(v.value))

      const failures = results.filter((result) => result.status === 'rejected')
      setError(failures.length ? `เชื่อมต่อข้อมูลบางส่วนไม่ได้ (${failures.length}/3 API)` : null)
      setLastUpdated(new Date())
    } catch (err) {
      if (!signal?.aborted) setError(err.message)
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    refresh(controller.signal)
    return () => controller.abort()
  }, [refresh])

  return { weights, events, videos, loading, error, lastUpdated, refresh }
}
