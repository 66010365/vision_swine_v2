export function Block({ id, label, children }) {
  return (
    <section className={`page-block page-block-${id}`}>
      {label && <div className="page-block-label">{label}</div>}
      {children}
    </section>
  )
}

export default function PageBoard({ children }) {
  return (
    <div className="page-board">
      {children}
    </div>
  )
}