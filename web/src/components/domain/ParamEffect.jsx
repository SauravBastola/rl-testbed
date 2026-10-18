import { extent, niceTicks, tickLabel } from '../../lib/chart.js'
import { fmtNum } from '../../lib/format.js'
import { cn } from '../../lib/utils.js'
import { Card } from '../ui/Card.jsx'

/** Mean score for each value of a parameter, as horizontal bars. */
function DiscreteEffect({ rows }) {
  const scored = rows.filter((r) => r.mean !== null)
  if (!scored.length) return <p className="text-body-s text-muted">No finished configurations yet.</p>
  const [lo, hi] = extent(scored.map((r) => r.mean))
  const best = Math.max(...scored.map((r) => r.mean))
  // Bars start at 12% so the worst value is still visible; the scale is relative.
  const width = (m) => (hi === lo ? 100 : 12 + ((m - lo) / (hi - lo)) * 88)
  return (
    <ul className="space-y-2">
      {rows.map((r) => (
        <li
          key={String(r.value)}
          className="grid grid-cols-[minmax(3.5rem,auto)_minmax(0,1fr)_auto] items-center gap-3"
        >
          <span className="num truncate text-right text-caption text-muted">{String(r.value)}</span>
          <span className="h-5 overflow-hidden rounded-sm bg-surface-3">
            {r.mean !== null && (
              <span
                className={cn('block h-full rounded-sm', r.mean === best ? 'bg-accent' : 'bg-accent/40')}
                style={{ width: `${width(r.mean)}%` }}
              />
            )}
          </span>
          <span className="num min-w-24 text-right text-caption text-text">
            {r.mean === null ? '—' : fmtNum(r.mean)}
            {r.std !== null && r.std !== undefined && <span className="text-subtle"> ±{fmtNum(r.std)}</span>}
          </span>
        </li>
      ))}
    </ul>
  )
}

/** One dot per configuration, for parameters sampled from a range. */
function ContinuousEffect({ points, log }) {
  if (!points.length) return <p className="text-body-s text-muted">No finished configurations yet.</p>
  const W = 320
  const H = 150
  const P = { l: 40, r: 10, t: 8, b: 22 }
  const tx = (v) => (log ? Math.log10(v) : v)
  const [x0, x1] = extent(points.map((p) => tx(p.value)))
  const yTicks = niceTicks(...extent(points.map((p) => p.score)), 3)
  const y0 = yTicks[0]
  const y1 = yTicks[yTicks.length - 1]
  const sx = (v) => P.l + ((tx(v) - x0) / (x1 - x0 || 1)) * (W - P.l - P.r)
  const sy = (v) => P.t + (1 - (v - y0) / (y1 - y0 || 1)) * (H - P.t - P.b)
  const best = Math.max(...points.map((p) => p.score))
  const xTicks = log
    ? niceTicks(x0, x1, 3)
        .filter((t) => Number.isInteger(t))
        .map((t) => 10 ** t)
    : niceTicks(x0, x1, 3)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Score against parameter value">
      {yTicks.map((t) => (
        <g key={t}>
          <line
            x1={P.l}
            x2={W - P.r}
            y1={sy(t)}
            y2={sy(t)}
            style={{ stroke: 'rgb(var(--chart-grid))', strokeOpacity: 'var(--chart-grid-alpha)' }}
          />
          <text x={P.l - 6} y={sy(t)} dy="0.32em" textAnchor="end" className="num fill-subtle text-micro">
            {tickLabel(t)}
          </text>
        </g>
      ))}
      {xTicks
        .filter((t) => tx(t) >= x0 - 1e-9 && tx(t) <= x1 + 1e-9)
        .map((t) => (
          <text key={t} x={sx(t)} y={H - 6} textAnchor="middle" className="num fill-subtle text-micro">
            {tickLabel(t)}
          </text>
        ))}
      {points.map((p) => (
        <circle
          key={p.config}
          cx={sx(p.value)}
          cy={sy(p.score)}
          r={p.score === best ? 5 : 3.5}
          fill="rgb(var(--accent))"
          fillOpacity={p.score === best ? 1 : 0.5}
        >
          <title>{`Config ${p.config}: ${fmtNum(p.value)} → ${fmtNum(p.score)}`}</title>
        </circle>
      ))}
    </svg>
  )
}

export function ParamEffect({ name, effect }) {
  return (
    <Card pad="sm" className="min-w-0">
      <div className="mb-3 flex items-baseline justify-between gap-2">
        <h3 className="truncate text-body-s font-semibold">
          <code>{name}</code>
        </h3>
        <span className="text-caption text-subtle">
          {effect.kind === 'discrete' ? 'mean per value' : effect.log ? 'log scale' : 'per configuration'}
        </span>
      </div>
      {effect.kind === 'discrete' ? (
        <DiscreteEffect rows={effect.values} />
      ) : (
        <ContinuousEffect points={effect.points} log={effect.log} />
      )}
    </Card>
  )
}
