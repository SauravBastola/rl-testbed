import { CheckCircle2, XCircle } from 'lucide-react'
import { fmtNum } from '../../lib/format.js'
import { cn } from '../../lib/utils.js'

/**
 * Sampled answers from a language policy. Each generated character is shaded by
 * the probability the model gave it, so sharpening is visible at a glance.
 */
export function SampleList({ samples, showProbs = true }) {
  return (
    <ul className="divide-y divide-border rounded-lg border border-border bg-surface">
      {samples.map((s, i) => (
        <li key={i} className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5">
          <span className="num w-20 shrink-0 text-body-s text-muted">{s.prompt}</span>
          <span className="flex min-w-0 flex-1 flex-wrap gap-0.5" aria-label={`answer ${s.completion}`}>
            {(s.tokens || [...s.completion].map((c) => ({ text: c }))).map((t, k) => (
              <span
                key={k}
                title={t.prob !== undefined ? `p = ${t.prob.toFixed(2)}` : undefined}
                className={cn(
                  'num grid h-7 min-w-6 place-items-center rounded-sm px-1 text-body-s',
                  t.text === '⏎' ? 'text-subtle' : 'text-text',
                )}
                style={
                  showProbs && t.prob !== undefined
                    ? { background: `rgb(var(--accent) / ${(0.08 + 0.5 * t.prob).toFixed(3)})` }
                    : undefined
                }
              >
                {t.text}
              </span>
            ))}
          </span>
          <span className="flex shrink-0 items-center gap-2 text-caption">
            {!s.correct && (
              <span className="text-subtle">
                expected <span className="num text-muted">{s.answer}</span>
              </span>
            )}
            <span className="num text-muted">{fmtNum(s.reward)}</span>
            {s.correct ? (
              <CheckCircle2 size={16} className="text-success" aria-label="correct" />
            ) : (
              <XCircle size={16} className="text-danger" aria-label="wrong" />
            )}
          </span>
        </li>
      ))}
    </ul>
  )
}
