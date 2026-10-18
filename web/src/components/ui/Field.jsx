import { forwardRef, useId } from 'react'
import { ChevronDown, Search, X } from 'lucide-react'
import { cn } from '../../lib/utils.js'

const control =
  'w-full rounded border border-border bg-surface-2 px-3 text-body text-text placeholder:text-subtle ' +
  'transition-[border-color,box-shadow] duration-fast ease-standard hover:border-border-strong ' +
  'focus:border-accent focus:shadow-focus focus:outline-none disabled:cursor-not-allowed disabled:opacity-50'

/** Label + control + help/error. Children receive the generated id via render prop or `id`. */
export function Field({ label, help, error, hint, htmlFor, className, children, inline }) {
  const auto = useId()
  const id = htmlFor || auto
  return (
    <div className={cn(inline ? 'flex items-center justify-between gap-4' : 'flex flex-col gap-1.5', className)}>
      {label && (
        <div className="flex items-baseline justify-between gap-3">
          <label htmlFor={id} className="text-body-s font-medium text-text">
            {label}
          </label>
          {hint && <span className="text-caption text-subtle">{hint}</span>}
        </div>
      )}
      {typeof children === 'function' ? children(id) : children}
      {error ? (
        <p className="text-caption text-danger" role="alert">
          {error}
        </p>
      ) : help ? (
        <p className="text-caption text-subtle">{help}</p>
      ) : null}
    </div>
  )
}

export const Input = forwardRef(function Input({ className, ...props }, ref) {
  return <input ref={ref} className={cn(control, 'h-9', className)} {...props} />
})

export const Textarea = forwardRef(function Textarea({ className, ...props }, ref) {
  return <textarea ref={ref} className={cn(control, 'min-h-20 py-2', className)} {...props} />
})

export const Select = forwardRef(function Select({ className, children, ...props }, ref) {
  return (
    <div className="relative">
      <select ref={ref} className={cn(control, 'h-9 appearance-none pr-9', className)} {...props}>
        {children}
      </select>
      <ChevronDown
        size={14}
        aria-hidden
        className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-subtle"
      />
    </div>
  )
})

export function SearchInput({ value, onChange, placeholder = 'Search', className, inputRef, ...props }) {
  return (
    <div className={cn('relative', className)}>
      <Search
        size={14}
        aria-hidden
        className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-subtle"
      />
      <input
        ref={inputRef}
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        aria-label={props['aria-label'] || placeholder}
        className={cn(control, 'h-9 pl-9 pr-8 [&::-webkit-search-cancel-button]:hidden')}
        {...props}
      />
      {value && (
        <button
          type="button"
          aria-label="Clear search"
          onClick={() => onChange('')}
          className="absolute right-2 top-1/2 grid h-6 w-6 -translate-y-1/2 place-items-center rounded text-subtle hover:bg-surface-3 hover:text-text"
        >
          <X size={14} aria-hidden />
        </button>
      )}
    </div>
  )
}

export function Switch({ checked, onChange, label, id, disabled }) {
  return (
    <button
      id={id}
      type="button"
      role="switch"
      aria-checked={!!checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        'relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors duration-fast ease-standard',
        checked ? 'border-accent bg-accent' : 'border-border-strong bg-surface-3',
        disabled && 'opacity-50',
      )}
    >
      <span
        className={cn(
          'inline-block h-3.5 w-3.5 rounded-full bg-text shadow transition-transform duration-fast ease-standard',
          checked ? 'translate-x-[18px] bg-accent-contrast' : 'translate-x-[2px]',
        )}
      />
    </button>
  )
}
