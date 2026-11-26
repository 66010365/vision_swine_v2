/** Lightweight SVG chart: no Chart.js dependency and safe for React re-renders. */
const COLORS = ['#2563eb', '#10b981', '#a855f7', '#f59e0b']
const WIDTH = 640
const HEIGHT = 210
const PAD = { left: 42, right: 15, top: 15, bottom: 29 }

function toFixed(value) {
  if (value >= 100) return value.toFixed(0)
  if (value >= 10) return value.toFixed(1)
  return value.toFixed(2)
}

export default function TrendChart({ title, points, lines, min = 0, max = null, unit = '' }) {
  const observed = lines.flatMap((line) => points.map((point) => point[line.field]).filter((v) => Number.isFinite(Number(v)) && v != null).map(Number))
  const maximum = max ?? Math.max(min + 1, ...observed.map((v) => v * 1.15), 1)
  const yMax = Math.max(maximum, min + 1)
  const innerW = WIDTH - PAD.left - PAD.right
  const innerH = HEIGHT - PAD.top - PAD.bottom
  const x = (index) => PAD.left + (points.length <= 1 ? innerW / 2 : index / (points.length - 1) * innerW)
  const y = (value) => PAD.top + innerH * (1 - (Number(value) - min) / (yMax - min))

  return (
    <section className="panel chart-card">
      <div className="chart-heading">
        <h3>{title}</h3>
        <div className="chart-legend">
          {lines.map((line, index) => (
            <span key={line.field}><i style={{ background: line.color || COLORS[index] }} /> {line.label}</span>
          ))}
        </div>
      </div>
      <svg className="trend-chart" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`กราฟ ${title}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const yy = PAD.top + fraction * innerH
          return <g key={fraction}>
            <line x1={PAD.left} y1={yy} x2={WIDTH - PAD.right} y2={yy} stroke="#e9eef0" strokeDasharray="3 4" />
            <text x={PAD.left - 8} y={yy + 4} textAnchor="end" fill="#94a3b8" fontSize="11">{toFixed(yMax - fraction * (yMax - min))}</text>
          </g>
        })}
        {lines.map((line, index) => {
          // Split on missing values; don't display missing GPU readings as zero.
          const segments = []
          let current = []
          points.forEach((point, idx) => {
            const val = point[line.field]
            if (val == null || !Number.isFinite(Number(val))) {
              if (current.length) segments.push(current)
              current = []
            } else {
              current.push(`${x(idx)},${y(val)}`)
            }
          })
          if (current.length) segments.push(current)
          return segments.map((segment, segmentIndex) => (
            <polyline key={`${line.field}-${segmentIndex}`} points={segment.join(' ')} fill="none" stroke={line.color || COLORS[index]} strokeWidth="2.7" strokeLinecap="round" strokeLinejoin="round" />
          ))
        })}
        <text x={PAD.left} y={HEIGHT - 8} fill="#94a3b8" fontSize="11">{points[0]?.time || '—'}</text>
        <text x={WIDTH - PAD.right} y={HEIGHT - 8} textAnchor="end" fill="#94a3b8" fontSize="11">{points[points.length - 1]?.time || '—'} {unit}</text>
      </svg>
      {points.length < 2 && <p className="chart-hint">รอข้อมูล Real-time อย่างน้อย 2 รอบเพื่อวาดกราฟ</p>}
    </section>
  )
}
