import { useEffect, useRef } from 'react'
import { cn } from '../../lib/utils.js'

/**
 * Line-numbered source view. `startLine` offsets numbering (code from
 * /api/agents/{name}/code starts at the class definition); `highlight` is a
 * predicate on the line text used to mark the update rule.
 */
export function CodeBlock({ code, startLine = 1, highlight, maxHeight = 520, className, label }) {
  const ref = useRef(null)
  const lines = (code || '').replace(/\n$/, '').split('\n')
  const marked = lines.map((l) => (highlight ? highlight(l) : false))
  const first = marked.indexOf(true)
  useEffect(() => {
    if (first < 0 || !ref.current) return
    const row = ref.current.querySelector(`[data-line="${first}"]`)
    if (row) ref.current.scrollTop = Math.max(0, row.offsetTop - 80)
  }, [first, code])
  return (
    <div className={cn('overflow-hidden rounded-lg border border-border bg-bg-soft', className)}>
      {label && <div className="border-b border-border px-4 py-2 text-caption text-subtle">{label}</div>}
      <div ref={ref} className="overflow-auto" style={{ maxHeight }}>
        <pre className="min-w-max py-3 text-caption leading-5">
          {lines.map((line, i) => (
            <div
              key={i}
              data-line={i}
              className={cn('flex pr-4', marked[i] && 'bg-accent/10 shadow-[inset_2px_0_0_rgb(var(--accent))]')}
            >
              <span className="num w-12 shrink-0 select-none pr-4 text-right text-subtle/70">{startLine + i}</span>
              <code className={cn('whitespace-pre', marked[i] ? 'text-text' : 'text-muted')}>{line || ' '}</code>
            </div>
          ))}
        </pre>
      </div>
    </div>
  )
}
