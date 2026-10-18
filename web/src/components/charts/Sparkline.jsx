import { colorVar, downsample, extent } from '../../lib/chart.js'

/** Tiny inline trend. Decorative: the number next to it carries the meaning. */
export function Sparkline({ values = [], color = 'chart-1', width = 96, height = 28, className }) {
  const pts = downsample(
    values.map((y, x) => ({ x, y })).filter((p) => Number.isFinite(p.y)),
    120,
  )
  if (pts.length < 2) return <svg width={width} height={height} aria-hidden className={className} />
  const [lo, hi] = extent(pts.map((p) => p.y))
  const span = hi - lo || 1
  const xMax = pts[pts.length - 1].x || 1
  const d = pts
    .map(
      (p, i) =>
        `${i ? 'L' : 'M'}${((p.x / xMax) * (width - 2) + 1).toFixed(1)} ${(height - 2 - ((p.y - lo) / span) * (height - 4)).toFixed(1)}`,
    )
    .join('')
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden className={className}>
      <path d={d} fill="none" stroke={colorVar(color)} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}
