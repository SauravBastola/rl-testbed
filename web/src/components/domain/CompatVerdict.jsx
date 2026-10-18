import { CheckCircle2, Info, OctagonX, TriangleAlert } from 'lucide-react'
import { cn } from '../../lib/utils.js'

/** Renders the server's compatibility verdict. Never re-derives it. */
export function CompatVerdict({ verdict, compact, className }) {
  if (!verdict) return null
  const meta = {
    ok: { icon: CheckCircle2, tone: 'text-success', border: 'border-success/30 bg-success/[.05]', title: 'Compatible' },
    warning: {
      icon: TriangleAlert,
      tone: 'text-warning',
      border: 'border-warning/30 bg-warning/[.06]',
      title: 'Trainable, with caveats',
    },
    blocked: {
      icon: OctagonX,
      tone: 'text-danger',
      border: 'border-danger/30 bg-danger/[.05]',
      title: 'Cannot train this pairing',
    },
  }[verdict.level]
  const Icon = meta.icon
  const lines = [
    ...verdict.blockers.map((t) => ({ t, kind: 'blocker' })),
    ...verdict.warnings.map((t) => ({ t, kind: 'warning' })),
    ...(compact ? [] : verdict.notes.map((t) => ({ t, kind: 'note' }))),
  ]
  return (
    <div
      className={cn('rounded-lg border p-3', meta.border, className)}
      role={verdict.level === 'blocked' ? 'alert' : undefined}
    >
      <div className={cn('flex items-center gap-2 text-body-s font-semibold', meta.tone)}>
        <Icon size={16} aria-hidden /> {meta.title}
      </div>
      {lines.length > 0 && (
        <ul className="mt-2 space-y-1.5">
          {lines.map(({ t, kind }) => (
            <li key={t} className="flex gap-2 text-body-s text-muted">
              {kind === 'note' ? (
                <Info size={14} className="mt-0.5 shrink-0 text-subtle" aria-hidden />
              ) : (
                <span
                  className={cn('mt-2 h-1 w-1 shrink-0 rounded-full', kind === 'blocker' ? 'bg-danger' : 'bg-warning')}
                />
              )}
              <span>{t}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** The small dot used in pickers. */
export function CompatDot({ level }) {
  const color = { ok: 'bg-success', warning: 'bg-warning', blocked: 'bg-danger/70' }[level] || 'bg-subtle'
  const label = { ok: 'Compatible', warning: 'Compatible with caveats', blocked: 'Incompatible' }[level] || 'Unknown'
  return (
    <span
      className={cn('inline-block h-2 w-2 shrink-0 rounded-full', color)}
      title={label}
      aria-label={label}
      role="img"
    />
  )
}
