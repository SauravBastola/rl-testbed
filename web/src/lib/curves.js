/** Turn run histories into chart series. */
import { SERIES_COLORS } from './chart.js'
import { metricLabel } from './format.js'

/** Metric keys present in a history, ordered by the lab's palette first. */
export function availableMetrics(history = [], preferred = []) {
  const keys = new Set()
  for (const row of history.slice(-50)) {
    for (const k of Object.keys(row)) {
      if (k === 'episode' || k === 'eval') continue
      if (typeof row[k] === 'number') keys.add(k)
    }
  }
  if (history.some((r) => r.eval)) keys.add('eval')
  const ordered = preferred.filter((k) => keys.has(k))
  const rest = [...keys].filter((k) => !ordered.includes(k)).sort()
  return [...ordered, ...rest].map((k) => ({ value: k, label: metricLabel(k) }))
}

export function seriesFromHistory(history = [], key = 'reward', opts = {}) {
  const points =
    key === 'eval'
      ? history.filter((r) => r.eval).map((r) => ({ x: r.episode, y: r.eval.episode_mean }))
      : history.filter((r) => typeof r[key] === 'number').map((r) => ({ x: r.episode, y: r[key] }))
  return {
    id: opts.id || key,
    label: opts.label || metricLabel(key),
    color: opts.color || 'chart-1',
    points,
    noSmooth: key === 'eval',
  }
}

/** Aggregated group from /api/compare/aggregate → series with a ±1σ band. */
export function seriesFromGroup(group, i) {
  const color = SERIES_COLORS[i % SERIES_COLORS.length]
  const points = group.points.map((p) => ({ x: p.x, y: p.mean }))
  const hasBand = group.n_runs > 1 && group.points.some((p) => p.std !== null)
  const band = hasBand
    ? group.points.filter((p) => p.std !== null).map((p) => ({ x: p.x, lo: p.mean - p.std, hi: p.mean + p.std }))
    : undefined
  const label = group.n_runs > 1 ? `${group.label} (${group.n_runs} seeds)` : group.label
  return { id: group.key, label, color, points, band }
}
