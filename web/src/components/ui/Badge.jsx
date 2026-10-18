import { cn } from '../../lib/utils.js'
import { STATUS } from '../../lib/format.js'

const TONES = {
  neutral: 'bg-surface-3 text-muted border-border',
  accent: 'bg-accent/12 text-accent border-accent/30',
  success: 'bg-success/12 text-success border-success/30',
  warning: 'bg-warning/12 text-warning border-warning/30',
  danger: 'bg-danger/12 text-danger border-danger/30',
  info: 'bg-info/12 text-info border-info/30',
  outline: 'bg-transparent text-muted border-border-strong',
}

export function Badge({ tone = 'neutral', className, children, ...props }) {
  return (
    <span
      className={cn(
        'inline-flex h-6 items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 text-caption font-medium',
        TONES[tone],
        className,
      )}
      {...props}
    >
      {children}
    </span>
  )
}

export function Dot({ tone = 'neutral', pulse, className }) {
  const color = {
    neutral: 'bg-subtle',
    accent: 'bg-accent',
    success: 'bg-success',
    warning: 'bg-warning',
    danger: 'bg-danger',
    info: 'bg-info',
  }[tone]
  return (
    <span
      aria-hidden
      className={cn('inline-block h-1.5 w-1.5 shrink-0 rounded-full', color, pulse && 'animate-pulse-dot', className)}
    />
  )
}

export function StatusBadge({ status, className }) {
  const meta = STATUS[status] || { label: status || 'Unknown', tone: 'neutral' }
  return (
    <Badge tone={meta.tone} className={className}>
      <Dot tone={meta.tone} pulse={meta.live} />
      {meta.label}
    </Badge>
  )
}

const DIFFICULTY_TONE = { easy: 'success', medium: 'info', hard: 'warning', expert: 'danger', experimental: 'outline' }
export function DifficultyBadge({ value }) {
  if (!value) return null
  return <Badge tone={DIFFICULTY_TONE[value] || 'neutral'}>{value.charAt(0).toUpperCase() + value.slice(1)}</Badge>
}
