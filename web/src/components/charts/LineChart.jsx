import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { colorVar, downsample, extent, nearestIndex, niceTicks, tickLabel } from '../../lib/chart.js'
import { fmtNum } from '../../lib/format.js'

const PAD = { top: 14, right: 16, bottom: 30, left: 52 }

function useWidth(ref) {
  const [w, setW] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return undefined
    setW(el.getBoundingClientRect().width)
    if (typeof ResizeObserver === 'undefined') return undefined
    const ro = new ResizeObserver(([entry]) => setW(entry.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return w
}

const pathFrom = (pts, sx, sy) => {
  let d = ''
  let pen = false
  for (const p of pts) {
    if (p.y === null || p.y === undefined || !Number.isFinite(p.y)) {
      pen = false
      continue
    }
    d += `${pen ? 'L' : 'M'}${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`
    pen = true
  }
  return d
}

/**
 * Multi-series SVG line chart.
 *
 * series: [{ id, label, color: token, points: [{x, y}], raw?: [{x,y}], band?: [{x, lo, hi}], dashed? }]
 * `raw` is drawn faintly behind the (smoothed) points. `band` is a shaded ±1σ region.
 * Draws once on mount; data updates never animate (values the user is reading stay still).
 */
export function LineChart({ series, height = 260, threshold, xLabel = 'Episode', yLabel, yDomain, ariaLabel, svgRef }) {
  const wrap = useRef(null)
  const width = useWidth(wrap)
  const uid = useId().replace(/:/g, '')
  const [hover, setHover] = useState(null)
  const [revealed, setRevealed] = useState(false)
  useEffect(() => {
    const t = requestAnimationFrame(() => setRevealed(true))
    return () => cancelAnimationFrame(t)
  }, [])

  const prepared = useMemo(
    () =>
      series.map((s) => ({
        ...s,
        points: downsample(
          s.points.filter((p) => Number.isFinite(p.x)),
          900,
        ),
        raw: s.raw ? downsample(s.raw, 900) : null,
      })),
    [series],
  )

  const { xs, ys } = useMemo(() => {
    const allX = []
    const allY = []
    for (const s of prepared) {
      for (const p of s.points) {
        allX.push(p.x)
        allY.push(p.y)
      }
      for (const p of s.raw || []) allY.push(p.y)
      for (const b of s.band || []) {
        allY.push(b.lo)
        allY.push(b.hi)
      }
    }
    const [lo, hi] = extent(allY)
    // Include the threshold in the domain only when it is near the data;
    // otherwise a far-off "solved" line would flatten the curve.
    const t = threshold?.value
    if (t !== undefined && t !== null && Number.isFinite(lo)) {
      const span = Math.max(hi - lo, Math.abs(hi) * 0.1, 1e-9)
      if (t <= hi + span * 1.5 && t >= lo - span * 1.5) allY.push(t)
    }
    return { xs: extent(allX), ys: yDomain || extent(allY) }
  }, [prepared, threshold, yDomain])

  const innerW = Math.max(10, width - PAD.left - PAD.right)
  const innerH = Math.max(10, height - PAD.top - PAD.bottom)
  const valid = Number.isFinite(xs[0]) && Number.isFinite(ys[0])
  const yTicks = valid ? niceTicks(ys[0], ys[1], Math.max(3, Math.round(innerH / 56))) : []
  const y0 = yTicks.length ? Math.min(yTicks[0], ys[0]) : 0
  const y1 = yTicks.length ? Math.max(yTicks[yTicks.length - 1], ys[1]) : 1
  const x0 = xs[0]
  const x1 = xs[1] === xs[0] ? xs[0] + 1 : xs[1]
  const sx = (x) => PAD.left + ((x - x0) / (x1 - x0)) * innerW
  const sy = (y) => PAD.top + innerH - ((y - y0) / (y1 - y0 || 1)) * innerH
  const xTicks = valid ? niceTicks(x0, x1, Math.max(2, Math.round(innerW / 90))).filter((t) => t >= x0 && t <= x1) : []

  const onMove = (e) => {
    if (!valid) return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - rect.left
    if (px < PAD.left || px > PAD.left + innerW) return setHover(null)
    const x = x0 + ((px - PAD.left) / innerW) * (x1 - x0)
    setHover({ x, px })
  }

  const hoverRows = hover
    ? prepared
        .map((s) => {
          const i = nearestIndex(s.points, hover.x)
          if (i < 0) return null
          const p = s.points[i]
          const band = s.band ? s.band[nearestIndex(s.band, hover.x)] : null
          return { s, p, band }
        })
        .filter(Boolean)
    : []
  const hoverX = hoverRows[0]?.p.x

  return (
    <div ref={wrap} className="relative w-full" style={{ height }}>
      {width > 0 && valid && (
        <svg
          ref={svgRef}
          width={width}
          height={height}
          role="img"
          aria-label={ariaLabel || `${yLabel || 'Value'} by ${xLabel.toLowerCase()}`}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
          className="block select-none"
        >
          <defs>
            <clipPath id={`clip-${uid}`}>
              <rect
                x={PAD.left}
                y={0}
                height={height}
                width={revealed ? innerW + 2 : 0}
                style={{ transition: revealed ? 'width 400ms cubic-bezier(.2,0,0,1)' : 'none' }}
              />
            </clipPath>
          </defs>

          {/* grid + axes */}
          {yTicks.map((t) => (
            <g key={`y${t}`}>
              <line
                x1={PAD.left}
                x2={PAD.left + innerW}
                y1={sy(t)}
                y2={sy(t)}
                style={{ stroke: 'rgb(var(--chart-grid))', strokeOpacity: 'var(--chart-grid-alpha)' }}
              />
              <text x={PAD.left - 8} y={sy(t)} dy="0.32em" textAnchor="end" className="num fill-subtle text-micro">
                {tickLabel(t)}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={`x${t}`} x={sx(t)} y={height - 10} textAnchor="middle" className="num fill-subtle text-micro">
              {tickLabel(t)}
            </text>
          ))}

          <g clipPath={`url(#clip-${uid})`}>
            {prepared.map((s) =>
              s.band?.length ? (
                <path
                  key={`band-${s.id}`}
                  d={
                    pathFrom(
                      s.band.map((b) => ({ x: b.x, y: b.hi })),
                      sx,
                      sy,
                    ) +
                    pathFrom(
                      [...s.band].reverse().map((b) => ({ x: b.x, y: b.lo })),
                      sx,
                      sy,
                    ).replace(/^M/, 'L') +
                    'Z'
                  }
                  fill={colorVar(s.color)}
                  fillOpacity={0.14}
                  stroke="none"
                />
              ) : null,
            )}
            {prepared.map((s) =>
              s.raw?.length ? (
                <path
                  key={`raw-${s.id}`}
                  d={pathFrom(s.raw, sx, sy)}
                  fill="none"
                  stroke={colorVar(s.color)}
                  strokeOpacity={0.22}
                  strokeWidth={1}
                />
              ) : null,
            )}
            {prepared.map((s) => (
              <path
                key={`line-${s.id}`}
                d={pathFrom(s.points, sx, sy)}
                fill="none"
                stroke={colorVar(s.color)}
                strokeWidth={s.width || 2}
                strokeDasharray={s.dashed ? '5 4' : undefined}
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity={hover && s.dim ? 0.4 : 1}
              />
            ))}
          </g>

          {threshold?.value !== undefined &&
            threshold?.value !== null &&
            (threshold.value < y0 || threshold.value > y1) && (
              <text
                x={PAD.left + innerW - 4}
                y={PAD.top + 4}
                textAnchor="end"
                className="fill-success text-micro font-medium"
              >
                {threshold.label || 'Solved'} at {tickLabel(threshold.value)},{' '}
                {threshold.value > y1 ? 'above' : 'below'} this range
              </text>
            )}
          {threshold?.value !== undefined &&
            threshold?.value !== null &&
            threshold.value >= y0 &&
            threshold.value <= y1 && (
              <g>
                <line
                  x1={PAD.left}
                  x2={PAD.left + innerW}
                  y1={sy(threshold.value)}
                  y2={sy(threshold.value)}
                  stroke="rgb(var(--success))"
                  strokeDasharray="4 5"
                  strokeOpacity={0.8}
                />
                <text
                  x={PAD.left + innerW - 4}
                  y={sy(threshold.value) - 6}
                  textAnchor="end"
                  className="fill-success text-micro font-medium"
                >
                  {threshold.label || 'Solved'} {tickLabel(threshold.value)}
                </text>
              </g>
            )}

          {hoverX !== undefined && (
            <g pointerEvents="none">
              <line
                x1={sx(hoverX)}
                x2={sx(hoverX)}
                y1={PAD.top}
                y2={PAD.top + innerH}
                stroke="rgb(var(--text-subtle))"
                strokeOpacity={0.5}
              />
              {hoverRows.map(({ s, p }) =>
                Number.isFinite(p.y) ? (
                  <circle
                    key={s.id}
                    cx={sx(p.x)}
                    cy={sy(p.y)}
                    r={3.5}
                    fill={colorVar(s.color)}
                    stroke="rgb(var(--surface))"
                    strokeWidth={2}
                  />
                ) : null,
              )}
            </g>
          )}
        </svg>
      )}

      {hoverX !== undefined && (
        <div
          className="pointer-events-none absolute top-2 z-10 min-w-40 rounded border border-border bg-surface-3/95 px-3 py-2 text-caption shadow-overlay backdrop-blur"
          style={hover.px > width / 2 ? { right: width - hover.px + 12 } : { left: hover.px + 12 }}
        >
          <div className="mb-1 text-subtle">
            {xLabel} <span className="num text-text">{fmtNum(Math.round(hoverX))}</span>
          </div>
          {hoverRows.map(({ s, p, band }) => (
            <div key={s.id} className="flex items-center justify-between gap-4">
              <span className="flex min-w-0 items-center gap-1.5 text-muted">
                <span className="h-0.5 w-3 shrink-0 rounded" style={{ background: colorVar(s.color) }} />
                <span className="truncate">{s.label}</span>
              </span>
              <span className="num text-text">
                {fmtNum(p.y)}
                {band && Number.isFinite(band.hi - band.lo) && band.hi !== band.lo && (
                  <span className="text-subtle"> ±{fmtNum((band.hi - band.lo) / 2)}</span>
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
