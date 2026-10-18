import { ArrowDown, ArrowUp, ChevronsUpDown } from 'lucide-react'
import { cn } from '../../lib/utils.js'

/** Horizontally scrollable table shell with a sticky header. */
export function Table({ children, className, maxHeight }) {
  return (
    <div
      className={cn('overflow-auto rounded-lg border border-border bg-surface', className)}
      style={maxHeight ? { maxHeight } : undefined}
    >
      <table className="w-full border-collapse text-left text-body-s">{children}</table>
    </div>
  )
}

export function THead({ children }) {
  return (
    <thead className="sticky top-0 z-10 bg-surface-2/95 backdrop-blur [&_th]:border-b [&_th]:border-border">
      <tr>{children}</tr>
    </thead>
  )
}

export function Th({ children, className, align = 'left', sort, onSort, field, width }) {
  const active = sort?.field === field && field
  const Icon = !field ? null : active ? (sort.dir === 'asc' ? ArrowUp : ArrowDown) : ChevronsUpDown
  const content = (
    <span className={cn('inline-flex items-center gap-1', align === 'right' && 'flex-row-reverse')}>
      {children}
      {Icon && <Icon size={12} aria-hidden className={active ? 'text-accent' : 'text-subtle/60'} />}
    </span>
  )
  return (
    <th
      scope="col"
      style={width ? { width } : undefined}
      aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined}
      className={cn(
        'h-9 whitespace-nowrap px-4 text-caption font-medium text-subtle',
        align === 'right' && 'text-right',
        className,
      )}
    >
      {field ? (
        <button type="button" onClick={() => onSort(field)} className="rounded hover:text-text">
          {content}
        </button>
      ) : (
        content
      )}
    </th>
  )
}

export function Tr({ children, className, onClick, selected }) {
  return (
    <tr
      onClick={onClick}
      className={cn(
        'border-b border-border/70 last:border-0 transition-colors duration-fast',
        onClick && 'cursor-pointer hover:bg-surface-2',
        selected && 'bg-accent/[.06]',
        className,
      )}
    >
      {children}
    </tr>
  )
}

export function Td({ children, className, align = 'left', mono }) {
  return (
    <td className={cn('h-11 px-4 align-middle', align === 'right' && 'text-right', mono && 'num', className)}>
      {children}
    </td>
  )
}

/** Generic client-side sort helper used by tables. */
export function sortRows(rows, sort, accessors = {}) {
  if (!sort?.field) return rows
  const get = accessors[sort.field] || ((r) => r[sort.field])
  const dir = sort.dir === 'asc' ? 1 : -1
  return [...rows].sort((a, b) => {
    const va = get(a)
    const vb = get(b)
    if (va === vb) return 0
    if (va === null || va === undefined) return 1
    if (vb === null || vb === undefined) return -1
    return va > vb ? dir : -dir
  })
}

export function nextSort(sort, field) {
  if (sort?.field !== field) return { field, dir: 'desc' }
  return { field, dir: sort.dir === 'desc' ? 'asc' : 'desc' }
}
