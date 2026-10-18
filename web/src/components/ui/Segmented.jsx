import { cn } from '../../lib/utils.js'

/** Mutually exclusive options. `options`: [{value, label, count?}] */
export function Segmented({ value, onChange, options, size = 'md', label, className }) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn('inline-flex rounded border border-border bg-surface-2 p-0.5', className)}
    >
      {options.map((o) => {
        const active = o.value === value
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.value)}
            className={cn(
              'inline-flex items-center gap-1.5 whitespace-nowrap rounded-sm px-3 font-medium transition-colors duration-fast ease-standard',
              size === 'sm' ? 'h-7 text-caption' : 'h-8 text-body-s',
              active ? 'bg-surface text-text shadow-raised' : 'text-muted hover:text-text',
            )}
          >
            {o.icon && <o.icon size={14} aria-hidden />}
            {o.label}
            {o.count !== undefined && <span className="num text-caption text-subtle">{o.count}</span>}
          </button>
        )
      })}
    </div>
  )
}

/** Underline tabs for switching panels within a page. */
export function Tabs({ value, onChange, tabs, className }) {
  return (
    <div role="tablist" className={cn('flex gap-1 overflow-x-auto border-b border-border scrollbar-none', className)}>
      {tabs.map((t) => {
        const active = t.value === value
        return (
          <button
            key={t.value}
            role="tab"
            type="button"
            aria-selected={active}
            onClick={() => onChange(t.value)}
            className={cn(
              'relative -mb-px inline-flex h-10 items-center gap-2 whitespace-nowrap border-b-2 px-3 text-body-s font-medium transition-colors duration-fast',
              active ? 'border-accent text-text' : 'border-transparent text-muted hover:text-text',
            )}
          >
            {t.icon && <t.icon size={14} aria-hidden />}
            {t.label}
            {t.count !== undefined && <span className="num text-caption text-subtle">{t.count}</span>}
          </button>
        )
      })}
    </div>
  )
}

/** Filter chip that toggles. */
export function Chip({ active, onClick, children, count, className }) {
  return (
    <button
      type="button"
      aria-pressed={!!active}
      onClick={onClick}
      className={cn(
        'inline-flex h-7 items-center gap-1.5 whitespace-nowrap rounded-full border px-3 text-caption font-medium transition-colors duration-fast ease-standard',
        active
          ? 'border-accent/50 bg-accent/12 text-text'
          : 'border-border bg-surface-2 text-muted hover:border-border-strong hover:text-text',
        className,
      )}
    >
      {children}
      {count !== undefined && <span className={cn('num', active ? 'text-accent' : 'text-subtle')}>{count}</span>}
    </button>
  )
}
