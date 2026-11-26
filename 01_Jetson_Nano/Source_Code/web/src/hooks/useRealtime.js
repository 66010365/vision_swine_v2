import { useEffect, useState } from 'react'
import { jetsonAPI, unwrapData } from '../services/api'

const MAX_POINTS = 35

export function useRealtime() {
  const [data, setData] = useState(null)
  const [status, setStatus] = useState('connecting')
  const [history, setHistory] = useState([])
  const [latency, setLatency] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  useEffect(() => {
    let active = true
    let inFlight = false
    let controller = null

    async function refresh() {
      if (inFlight) return
      inFlight = true
      controller = new AbortController()
      const started = performance.now()

      try {
        const response = await jetsonAPI.realtime(controller.signal)
        const next = unwrapData(response)
        if (!next || typeof next !== 'object' || !next.pig_counts) {
          throw new Error('The realtime response is missing pig_counts')
        }

        if (!active) return
        const stamp = new Date()
        setData(next)
        setStatus('live')
        setLatency(Math.round(performance.now() - started))
        setLastUpdated(stamp)
        setHistory((points) => [...points, {
          time: stamp.toLocaleTimeString('th-TH', { hour12: false }),
          cpu: next.hardware?.cpu_usage ?? null,
          ram: next.hardware?.ram_usage ?? null,
          gpu: next.hardware?.gpu_usage ?? null,
          cpuTemp: next.hardware?.cpu_temp ?? null,
          gpuTemp: next.hardware?.gpu_temp ?? null,
          rx: next.network?.rx_speed == null ? null : next.network.rx_speed / 1024,
          tx: next.network?.tx_speed == null ? null : next.network.tx_speed / 1024,
        }].slice(-MAX_POINTS))
      } catch (error) {
        if (active && error.name !== 'AbortError') {
          setStatus('error')
          setData(null) // do not display stale values as live data
        }
      } finally {
        inFlight = false
      }
    }

    refresh()
    const intervalId = setInterval(refresh, 2000)

    return () => {
      active = false
      clearInterval(intervalId)
      controller?.abort()
    }
  }, [])

  return { data, status, history, latency, lastUpdated }
}
