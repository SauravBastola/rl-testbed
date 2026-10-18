import { Link } from 'react-router-dom'
import { CheckCircle2, Play } from 'lucide-react'
import { useCatalogue, useLearningPath, useRuns } from '../lib/queries.js'
import { ButtonLink } from '../components/ui/Button.jsx'
import { ErrorState, SkeletonRows } from '../components/ui/Feedback.jsx'
import { PageHeader } from '../components/ui/Page.jsx'
import { cn } from '../lib/utils.js'

/** The cross-lab curriculum. Each step links straight into a pre-filled run. */
export default function Learn() {
  const path = useLearningPath()
  const { envBy, agentBy, labs } = useCatalogue()
  const runs = useRuns({ limit: 500 })
  const done = (s) =>
    (runs.data || []).some((r) => r.env === s.env && r.agent === s.algorithm && r.status === 'completed')
  const labFor = (env) =>
    envBy[env]?.lab && labs.some((l) => l.id === envBy[env].lab) ? envBy[env].lab : 'foundations'

  return (
    <div>
      <PageHeader
        title="Learning path"
        description="From a random baseline to comparing modern methods across seeds. Each step is one experiment; finishing a run marks it done."
      />
      {path.error ? (
        <ErrorState error={path.error} onRetry={path.refetch} />
      ) : path.isLoading ? (
        <SkeletonRows rows={6} />
      ) : (
        <ol className="relative space-y-3 before:absolute before:bottom-6 before:left-[19px] before:top-6 before:w-px before:bg-border">
          {path.data.map((s) => {
            const complete = done(s)
            const lab = labFor(s.env)
            return (
              <li key={s.step} className="relative flex gap-5">
                <span
                  className={cn(
                    'num relative z-10 grid h-10 w-10 shrink-0 place-items-center rounded-full border text-body font-semibold',
                    complete ? 'border-success/40 bg-success/15 text-success' : 'border-border-strong bg-bg text-muted',
                  )}
                >
                  {complete ? <CheckCircle2 size={18} aria-label="Done" /> : s.step}
                </span>
                <div className="flex min-w-0 flex-1 flex-wrap items-start justify-between gap-4 rounded-lg border border-border bg-surface p-5">
                  <div className="min-w-0 max-w-[64ch]">
                    <h2 className="text-title-s font-semibold">{s.title}</h2>
                    <p className="mt-1 text-body-s text-muted">{s.why}</p>
                    <p className="mt-2 text-caption text-subtle">
                      <Link to={`/algorithms/${s.algorithm}`} className="text-text hover:text-accent">
                        {agentBy[s.algorithm]?.label || s.algorithm}
                      </Link>{' '}
                      on{' '}
                      <Link to={`/envs/${s.env}`} className="text-text hover:text-accent">
                        {envBy[s.env]?.label || s.env}
                      </Link>
                      {s.done_when && <> · Done when: {s.done_when}</>}
                    </p>
                  </div>
                  <ButtonLink
                    to={`/lab/${lab}/new?agent=${s.algorithm}&env=${s.env}`}
                    size="sm"
                    variant={complete ? 'secondary' : 'primary'}
                    icon={Play}
                  >
                    {complete ? 'Run again' : 'Start'}
                  </ButtonLink>
                </div>
              </li>
            )
          })}
        </ol>
      )}
    </div>
  )
}
