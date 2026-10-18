import { AlertTriangle, Info, RotateCw } from 'lucide-react'
import { cn } from '../../lib/utils.js'
import { Button } from './Button.jsx'

/** Shimmering placeholder in the exact footprint of the content it stands in for. */
export function Skeleton({ className, ...props }) {
  return (
    <div
      aria-hidden
      {...props}
      className={cn(
        'relative overflow-hidden rounded bg-surface-3/70',
        'after:absolute after:inset-0 after:-translate-x-full after:animate-shimmer',
        'after:bg-gradient-to-r after:from-transparent after:via-text/[.05] after:to-transparent',
        className,
      )}
    />
  )
}

export function SkeletonRows({ rows = 5, className }) {
  return (
    <div className={cn('space-y-2', className)} role="status" aria-label="Loading">
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-10 w-full" />
      ))}
    </div>
  )
}

/** Empty is an invitation: say what is missing and offer the action that fixes it. */
export function EmptyState({ icon: Icon, art, title, description, action, className, compact }) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center rounded-lg border border-dashed border-border-strong/70 text-center',
        compact ? 'gap-2 px-6 py-8' : 'gap-3 px-8 py-14',
        className,
      )}
    >
      {art}
      {Icon && !art && (
        <span className="grid h-10 w-10 place-items-center rounded-lg bg-surface-3 text-muted">
          <Icon size={18} aria-hidden />
        </span>
      )}
      <div className="max-w-md">
        <p className="text-title-s font-semibold text-text">{title}</p>
        {description && <p className="mt-1 text-body-s text-muted">{description}</p>}
      </div>
      {action && <div className="mt-1 flex flex-wrap justify-center gap-2">{action}</div>}
    </div>
  )
}

export function ErrorState({ error, onRetry, title = 'Could not load this', className }) {
  const message = error?.message || String(error || 'Unknown error')
  return (
    <div
      role="alert"
      className={cn('flex items-start gap-3 rounded-lg border border-danger/30 bg-danger/[.06] p-4', className)}
    >
      <AlertTriangle size={16} className="mt-0.5 shrink-0 text-danger" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="text-body-s font-semibold text-text">{title}</p>
        <p className="mt-0.5 break-words text-body-s text-muted">{message}</p>
      </div>
      {onRetry && (
        <Button size="sm" variant="secondary" icon={RotateCw} onClick={() => onRetry()}>
          Retry
        </Button>
      )}
    </div>
  )
}

/** A non-blocking message inside a page. tone: info | warning | danger | success */
export function Notice({ tone = 'info', title, children, icon: IconOverride, className, action }) {
  const Icon = IconOverride || (tone === 'info' ? Info : AlertTriangle)
  const tones = {
    info: 'border-info/25 bg-info/[.06] text-info',
    warning: 'border-warning/30 bg-warning/[.07] text-warning',
    danger: 'border-danger/30 bg-danger/[.06] text-danger',
    success: 'border-success/30 bg-success/[.06] text-success',
  }
  return (
    <div className={cn('flex items-start gap-3 rounded-lg border px-4 py-3', tones[tone], className)}>
      <Icon size={16} className="mt-0.5 shrink-0" aria-hidden />
      <div className="min-w-0 flex-1 text-body-s text-muted">
        {title && <p className="font-semibold text-text">{title}</p>}
        {children}
      </div>
      {action}
    </div>
  )
}

/** Wraps a query: renders skeleton, error (with retry) or children(data). */
export function QueryState({ query, skeleton, children, errorTitle }) {
  if (query.isLoading) return skeleton || <SkeletonRows />
  if (query.error) return <ErrorState error={query.error} onRetry={query.refetch} title={errorTitle} />
  return children(query.data)
}
