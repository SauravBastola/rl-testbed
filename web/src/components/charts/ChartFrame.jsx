import { useMemo, useRef, useState } from 'react'
import { Download, LineChart as LineIcon } from 'lucide-react'
import { cn } from '../../lib/utils.js'
import { colorVar, ema } from '../../lib/chart.js'
import { Card } from '../ui/Card.jsx'
import { IconButton } from '../ui/Button.jsx'
import { Segmented } from '../ui/Segmented.jsx'
import { Select } from '../ui/Field.jsx'
import { EmptyState, Skeleton } from '../ui/Feedback.jsx'
import { LineChart } from './LineChart.jsx'

export const SMOOTHING = [
  { value: 0, label: 'Raw' },
  { value: 0.6, label: 'Light' },
  { value: 0.9, label: 'Heavy' },
]

/** Resolve CSS custom properties so an exported image keeps the theme colours. */
function exportPng(svg, filename) {
  const root = getComputedStyle(document.documentElement)
  const clone = svg.cloneNode(true)
  const resolve = (value) =>
    value.replace(/var\((--[\w-]+)\)/g, (_, name) => root.getPropertyValue(name).trim() || '0 0 0')
  const walk = (src, dst) => {
    const cs = getComputedStyle(src)
    for (const prop of ['fill', 'stroke', 'stroke-opacity', 'fill-opacity', 'font-family', 'font-size', 'opacity']) {
      dst.style.setProperty(prop, resolve(cs.getPropertyValue(prop)))
    }
    for (const attr of ['fill', 'stroke']) {
      const v = dst.getAttribute(attr)
      if (v && v.includes('var(')) dst.setAttribute(attr, resolve(v))
    }
    ;[...src.children].forEach((c, i) => walk(c, dst.children[i]))
  }
  walk(svg, clone)
  const { width, height } = svg.getBoundingClientRect()
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  const bg = `rgb(${root.getPropertyValue('--surface').trim()})`
  const data = new XMLSerializer().serializeToString(clone)
  const img = new Image()
  img.onload = () => {
    const scale = 2
    const canvas = document.createElement('canvas')
    canvas.width = width * scale
    canvas.height = height * scale
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = bg
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.scale(scale, scale)
    ctx.drawImage(img, 0, 0)
    const a = document.createElement('a')
    a.download = `${filename}.png`
    a.href = canvas.toDataURL('image/png')
    a.click()
  }
  img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(data)}`
}

/**
 * Every chart in the app sits in one of these. It owns the title, legend,
 * metric switcher, smoothing control, PNG export, and the empty/loading states.
 *
 * series: [{ id, label, color, points: [{x,y}], band?, dashed? }] — raw values;
 * smoothing is applied here so the raw line can be drawn faintly behind.
 */
export function ChartFrame({
  title,
  description,
  series = [],
  loading,
  threshold,
  metric,
  metrics,
  onMetricChange,
  height = 260,
  smoothing: smoothingProp,
  defaultSmoothing = 0.6,
  emptyTitle = 'No data yet',
  emptyDescription,
  emptyAction,
  className,
  actions,
  xLabel,
  legend = true,
  filename = 'chart',
  compact,
}) {
  const [smoothingState, setSmoothing] = useState(defaultSmoothing)
  const smoothing = smoothingProp ?? smoothingState
  const svgRef = useRef(null)
  const hasData = series.some((s) => s.points?.length > 1)

  const shown = useMemo(
    () =>
      series.map((s) => {
        if (!smoothing || s.noSmooth) return s
        const ys = ema(
          s.points.map((p) => p.y),
          smoothing,
        )
        const band = s.band
          ? (() => {
              const lo = ema(
                s.band.map((b) => b.lo),
                smoothing,
              )
              const hi = ema(
                s.band.map((b) => b.hi),
                smoothing,
              )
              return s.band.map((b, i) => ({ x: b.x, lo: lo[i], hi: hi[i] }))
            })()
          : undefined
        return { ...s, raw: s.points, points: s.points.map((p, i) => ({ x: p.x, y: ys[i] })), band }
      }),
    [series, smoothing],
  )

  return (
    <Card pad={compact ? 'sm' : 'md'} className={cn('min-w-0', className)}>
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-title-s font-semibold text-text">{title}</h3>
          {description && <p className="mt-0.5 text-body-s text-muted">{description}</p>}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {metrics?.length > 1 && (
            <Select
              aria-label="Metric"
              value={metric}
              onChange={(e) => onMetricChange(e.target.value)}
              className="h-8 w-auto min-w-40 text-body-s"
            >
              {metrics.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </Select>
          )}
          {smoothingProp === undefined && (
            <Segmented size="sm" label="Smoothing" value={smoothing} onChange={setSmoothing} options={SMOOTHING} />
          )}
          {actions}
          <IconButton
            icon={Download}
            label="Export as PNG"
            size="sm"
            disabled={!hasData}
            onClick={() => svgRef.current && exportPng(svgRef.current, filename)}
          />
        </div>
      </div>

      {legend && hasData && series.length > 1 && (
        <ul className="mb-2 flex flex-wrap gap-x-4 gap-y-1" aria-label="Legend">
          {series.map((s) => (
            <li key={s.id} className="flex min-w-0 items-center gap-1.5 text-caption text-muted">
              <span
                className="h-0.5 w-4 shrink-0 rounded"
                style={{ background: colorVar(s.color), opacity: s.dashed ? 0.7 : 1 }}
              />
              <span className="max-w-64 truncate">{s.label}</span>
            </li>
          ))}
        </ul>
      )}

      {loading ? (
        <Skeleton className="w-full" style={{ height }} />
      ) : hasData ? (
        <LineChart
          series={shown}
          height={height}
          threshold={threshold}
          svgRef={svgRef}
          xLabel={xLabel}
          yLabel={title}
        />
      ) : (
        <div style={{ minHeight: height }} className="flex">
          <EmptyState
            compact
            icon={LineIcon}
            title={emptyTitle}
            description={emptyDescription}
            action={emptyAction}
            className="flex-1"
          />
        </div>
      )}
    </Card>
  )
}
