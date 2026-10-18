/** Pure helpers for the SVG chart. Kept free of React so they are easy to test. */

/** Exponential moving average; `alpha` in [0,1) — 0 means no smoothing. */
export function ema(values, alpha) {
  if (!alpha) return values
  let prev = null
  return values.map((v) => {
    if (v === null || v === undefined || !Number.isFinite(v)) return v
    prev = prev === null ? v : alpha * prev + (1 - alpha) * v
    return prev
  })
}

/** "Nice" axis ticks covering [min, max]. */
export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return []
  if (min === max) {
    const pad = Math.abs(min) * 0.1 || 1
    min -= pad
    max += pad
  }
  const span = max - min
  const step0 = span / Math.max(1, count)
  const mag = 10 ** Math.floor(Math.log10(step0))
  const err = step0 / mag
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag
  const start = Math.floor(min / step) * step
  const ticks = []
  for (let v = start; v <= max + step * 0.5; v += step) ticks.push(Number(v.toPrecision(12)))
  return ticks
}

/** Keep at most ~`max` points while preserving local extremes (min/max per bucket). */
export function downsample(points, max = 800) {
  if (points.length <= max) return points
  const bucket = Math.ceil(points.length / (max / 2))
  const out = []
  for (let i = 0; i < points.length; i += bucket) {
    const slice = points.slice(i, i + bucket)
    let lo = slice[0]
    let hi = slice[0]
    for (const p of slice) {
      if (p.y < lo.y) lo = p
      if (p.y > hi.y) hi = p
    }
    if (lo === hi) out.push(lo)
    else if (lo.x < hi.x) out.push(lo, hi)
    else out.push(hi, lo)
  }
  return out
}

/** Index of the point whose x is closest to `x` (points sorted by x). */
export function nearestIndex(points, x) {
  let lo = 0
  let hi = points.length - 1
  if (hi < 0) return -1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (points[mid].x < x) lo = mid
    else hi = mid
  }
  return Math.abs(points[lo].x - x) <= Math.abs(points[hi].x - x) ? lo : hi
}

export function extent(values) {
  let min = Infinity
  let max = -Infinity
  for (const v of values) {
    if (v === null || v === undefined || !Number.isFinite(v)) continue
    if (v < min) min = v
    if (v > max) max = v
  }
  return [min, max]
}

/** Compact tick label. */
export function tickLabel(v) {
  const a = Math.abs(v)
  if (a >= 1e6) return `${(v / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`
  if (a >= 1e4) return `${(v / 1e3).toFixed(a >= 1e5 ? 0 : 1)}k`
  if (a > 0 && a < 0.01) return v.toExponential(0)
  return String(Number(v.toPrecision(4)))
}

export const SERIES_COLORS = ['chart-1', 'chart-2', 'chart-3', 'chart-4', 'chart-5', 'chart-6']
export const colorVar = (token) => `rgb(var(--${token}))`
