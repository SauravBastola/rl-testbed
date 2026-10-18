import { cn } from '../../lib/utils.js'

/** Every page starts with this: title, one-line purpose, actions on the right. */
export function PageHeader({ title, description, actions, meta, icon, className }) {
  return (
    <header className={cn('mb-8 flex flex-wrap items-end justify-between gap-x-6 gap-y-4', className)}>
      <div className="flex min-w-0 items-start gap-4">
        {icon}
        <div className="min-w-0">
          <h1 className="text-display font-semibold text-text">{title}</h1>
          {description && <p className="mt-1.5 max-w-[68ch] text-body text-muted">{description}</p>}
          {meta && <div className="mt-3 flex flex-wrap items-center gap-2">{meta}</div>}
        </div>
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  )
}

/** Compact metric: label above, value below. Values never animate. */
export function Stat({ label, value, hint, tone, className, children }) {
  return (
    <div className={cn('min-w-0', className)}>
      <div className="label truncate">{label}</div>
      <div
        className={cn(
          'num mt-1 truncate text-title font-medium',
          tone === 'accent' && 'text-accent',
          tone === 'success' && 'text-success',
          tone === 'danger' && 'text-danger',
          tone === 'warning' && 'text-warning',
        )}
      >
        {value}
      </div>
      {hint && <div className="mt-0.5 truncate text-caption text-subtle">{hint}</div>}
      {children}
    </div>
  )
}

/** A row of stats inside one bordered strip, divided rather than boxed. */
export function StatStrip({ children, className }) {
  return (
    <div
      className={cn(
        'grid grid-cols-2 divide-border rounded-lg border border-border bg-surface md:grid-cols-4 md:divide-x',
        '[&>*]:px-5 [&>*]:py-4',
        className,
      )}
    >
      {children}
    </div>
  )
}

export function KeyValue({ rows, className }) {
  return (
    <dl className={cn('grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)] gap-x-4 gap-y-2 text-body-s', className)}>
      {rows
        .filter((r) => r && r[1] !== undefined && r[1] !== null && r[1] !== '')
        .map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-subtle">{k}</dt>
            <dd className="min-w-0 break-words text-text">{v}</dd>
          </div>
        ))}
    </dl>
  )
}

export function Kbd({ children, className }) {
  return (
    <kbd
      className={cn(
        'inline-flex h-5 min-w-5 items-center justify-center rounded-sm border border-border bg-surface-3 px-1 font-sans text-micro text-muted',
        className,
      )}
    >
      {children}
    </kbd>
  )
}

/** Thin determinate progress bar. */
export function Progress({ value, className, tone = 'accent' }) {
  const pct = Math.max(0, Math.min(1, value || 0)) * 100
  return (
    <div
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
      className={cn('h-1 w-full overflow-hidden rounded-full bg-surface-3', className)}
    >
      <div
        className={cn(
          'h-full rounded-full transition-[width] duration-slow ease-standard',
          tone === 'accent' ? 'bg-accent' : tone === 'success' ? 'bg-success' : 'bg-subtle',
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}
