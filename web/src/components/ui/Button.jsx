import { forwardRef } from 'react'
import { Link } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { cn } from '../../lib/utils.js'

const VARIANTS = {
  primary:
    'bg-accent text-accent-contrast border-transparent hover:brightness-110 active:brightness-95 shadow-[0_6px_18px_-8px_rgb(var(--accent)/.7)]',
  secondary: 'bg-surface-2 text-text border-border hover:bg-surface-3 hover:border-border-strong active:bg-surface-2',
  ghost: 'bg-transparent text-muted border-transparent hover:bg-surface-2 hover:text-text active:bg-surface-3',
  danger: 'bg-danger/10 text-danger border-danger/30 hover:bg-danger/15 active:bg-danger/20',
}
const SIZES = {
  sm: 'h-8 px-3 text-body-s gap-1.5',
  md: 'h-9 px-4 text-body gap-2',
  lg: 'h-11 px-5 text-body gap-2',
}

export function buttonClass({ variant = 'secondary', size = 'md', className } = {}) {
  return cn(
    'inline-flex shrink-0 select-none items-center justify-center whitespace-nowrap rounded border font-medium',
    'transition-[background-color,border-color,color,filter,transform] duration-fast ease-standard',
    'active:translate-y-px disabled:pointer-events-none disabled:opacity-45',
    VARIANTS[variant],
    SIZES[size],
    className,
  )
}

export const Button = forwardRef(function Button(
  { variant, size, className, loading, icon: Icon, children, type = 'button', ...props },
  ref,
) {
  const iconSize = size === 'sm' ? 14 : 16
  return (
    <button
      ref={ref}
      type={type}
      className={buttonClass({ variant, size, className })}
      disabled={loading || props.disabled}
      {...props}
    >
      {loading ? (
        <Loader2 size={iconSize} className="animate-spin" aria-hidden />
      ) : Icon ? (
        <Icon size={iconSize} aria-hidden />
      ) : null}
      {children}
    </button>
  )
})

export function ButtonLink({ to, variant, size, className, icon: Icon, children, ...props }) {
  const iconSize = size === 'sm' ? 14 : 16
  return (
    <Link to={to} className={buttonClass({ variant, size, className })} {...props}>
      {Icon ? <Icon size={iconSize} aria-hidden /> : null}
      {children}
    </Link>
  )
}

/** Icon-only button. `label` is required: it becomes the accessible name and tooltip. */
export const IconButton = forwardRef(function IconButton(
  { icon: Icon, label, size = 'md', variant = 'ghost', className, ...props },
  ref,
) {
  const dim = size === 'sm' ? 'h-8 w-8' : 'h-9 w-9'
  return (
    <button
      ref={ref}
      type="button"
      aria-label={label}
      title={label}
      className={cn(buttonClass({ variant, size }), 'px-0', dim, className)}
      {...props}
    >
      <Icon size={size === 'sm' ? 14 : 16} aria-hidden />
    </button>
  )
})
