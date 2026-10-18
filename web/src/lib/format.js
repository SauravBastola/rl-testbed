/** Shared formatting. Every page uses these; none re-implements them. */

export function fmtNum(x, opts = {}) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  if (typeof x !== 'number') return String(x)
  const { digits } = opts
  const abs = Math.abs(x)
  if (digits !== undefined) return x.toFixed(digits)
  if (abs !== 0 && (abs < 0.001 || abs >= 1e6)) return x.toExponential(2)
  if (Number.isInteger(x)) return x.toLocaleString()
  if (abs >= 100) return x.toFixed(1)
  if (abs >= 1) return x.toFixed(2)
  return x.toFixed(3)
}

export function fmtCompact(x) {
  if (x === null || x === undefined) return '—'
  return Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(x)
}

export function fmtBytes(n) {
  if (n === null || n === undefined) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let v = n
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${units[i]}`
}

export function fmtDuration(sec) {
  if (sec === null || sec === undefined || !Number.isFinite(sec)) return '—'
  if (sec < 1) return '<1s'
  if (sec < 60) return `${Math.round(sec)}s`
  const m = Math.floor(sec / 60)
  if (m < 60) return `${m}m ${Math.round(sec % 60)}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

export function fmtRelative(ts, now = Date.now() / 1000) {
  if (!ts) return '—'
  const d = now - ts
  if (d < 45) return 'just now'
  if (d < 3600) return `${Math.round(d / 60)} min ago`
  if (d < 86400) return `${Math.round(d / 3600)} h ago`
  if (d < 86400 * 7) return `${Math.round(d / 86400)} d ago`
  return fmtDate(ts)
}

export function fmtDate(ts) {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function plural(n, word, pluralWord) {
  return `${n.toLocaleString()} ${n === 1 ? word : pluralWord || `${word}s`}`
}

/** Status vocabulary shared by every surface. */
export const STATUS = {
  queued: { label: 'Queued', tone: 'neutral', live: false },
  created: { label: 'Created', tone: 'neutral', live: false },
  running: { label: 'Running', tone: 'accent', live: true },
  stopping: { label: 'Stopping', tone: 'warning', live: true },
  completed: { label: 'Completed', tone: 'success', live: false },
  stopped: { label: 'Stopped', tone: 'neutral', live: false },
  failed: { label: 'Failed', tone: 'danger', live: false },
  interrupted: { label: 'Interrupted', tone: 'warning', live: false },
}
export const ACTIVE_STATUSES = ['queued', 'created', 'running', 'stopping']
export const isActive = (status) => ACTIVE_STATUSES.includes(status)

/** Human label for history keys such as `metric_td_error`. */
export function metricLabel(key) {
  const fixed = {
    reward: 'Episode return',
    mean_reward: 'Mean return (last 100)',
    steps: 'Episode length',
    eval: 'Greedy evaluation',
  }
  if (fixed[key]) return fixed[key]
  const k = key.replace(/^metric_/, '').replace(/_/g, ' ')
  return k.charAt(0).toUpperCase() + k.slice(1)
}

/** Progress of a run, using live progress when the run is active. */
export function runProgress(run) {
  const total = run.progress?.total_episodes || run.episodes || 0
  const done = isActive(run.status) ? (run.progress?.episode ?? run.episodes_done ?? 0) : (run.episodes_done ?? 0)
  return { done, total, frac: total ? Math.min(1, done / total) : 0 }
}
