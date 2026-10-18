import { cn } from '../../lib/utils.js'

/** The one card. `pad` = 'md' (24px) | 'sm' (16px) | 'none'. `raised` adds elevation. */
export function Card({ as: Tag = 'div', pad = 'md', raised, interactive, className, children, ...props }) {
  return (
    <Tag
      className={cn(
        'rounded-lg border border-border bg-surface',
        pad === 'md' && 'p-6',
        pad === 'sm' && 'p-4',
        raised && 'shadow-raised',
        interactive &&
          'transition-[border-color,background-color,transform] duration-fast ease-standard hover:border-border-strong hover:bg-surface-2',
        className,
      )}
      {...props}
    >
      {children}
    </Tag>
  )
}

export function CardHeader({ title, description, actions, icon: Icon, className }) {
  return (
    <div className={cn('mb-4 flex items-start justify-between gap-4', className)}>
      <div className="flex min-w-0 items-start gap-3">
        {Icon && (
          <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded bg-accent/10 text-accent">
            <Icon size={16} aria-hidden />
          </span>
        )}
        <div className="min-w-0">
          <h2 className="text-title-s font-semibold text-text">{title}</h2>
          {description && <p className="mt-0.5 text-body-s text-muted">{description}</p>}
        </div>
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  )
}

/** A titled region of a page without a container — for content that is not a card. */
export function Section({ title, description, actions, className, children, id }) {
  return (
    <section className={cn('min-w-0', className)} id={id} aria-labelledby={id ? `${id}-title` : undefined}>
      {(title || actions) && (
        <div className="mb-3 flex items-end justify-between gap-4">
          <div className="min-w-0">
            {title && (
              <h2 id={id ? `${id}-title` : undefined} className="text-title-s font-semibold text-text">
                {title}
              </h2>
            )}
            {description && <p className="mt-0.5 text-body-s text-muted">{description}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  )
}
