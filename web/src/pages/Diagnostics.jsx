import { Link, useNavigate } from 'react-router-dom'
import { Check, Circle, Loader2, Minus, PlayCircle, X } from 'lucide-react'
import { api } from '../lib/api.js'
import { useDiagnostics, useRunMutation } from '../lib/queries.js'
import { fmtNum, plural } from '../lib/format.js'
import { cn } from '../lib/utils.js'
import { Button } from '../components/ui/Button.jsx'
import { Card } from '../components/ui/Card.jsx'
import { ErrorState, Skeleton } from '../components/ui/Feedback.jsx'
import { PageHeader, Stat, StatStrip } from '../components/ui/Page.jsx'
import { Tooltip } from '../components/ui/Overlay.jsx'
import { useToast } from '../components/ui/Toast.jsx'
import { LabSpecimen } from '../components/art/LabArt.jsx'

const VERDICT = {
  pass: { icon: Check, cls: 'bg-success/15 text-success', label: 'Passed' },
  fail: { icon: X, cls: 'bg-danger/15 text-danger', label: 'Failed' },
  running: { icon: Loader2, cls: 'bg-accent/15 text-accent', label: 'Running', spin: true },
  untested: { icon: Circle, cls: 'text-subtle', label: 'Not run yet' },
  'n/a': { icon: Minus, cls: 'text-subtle/50', label: 'Not applicable' },
}

function Cell({ cell, onRun, bench, agentLabel }) {
  const v = VERDICT[cell.verdict] || VERDICT.untested
  const Icon = v.icon
  const tip = `${agentLabel} on ${bench.title}: ${v.label}${cell.best_eval !== null && cell.best_eval !== undefined ? ` (best eval ${fmtNum(cell.best_eval)}, needs ${bench.pass_threshold})` : ''}${cell.expected ? '' : '. Not expected to pass.'}`
  const content = (
    <span
      className={cn(
        'grid h-8 w-8 place-items-center rounded transition-transform duration-fast',
        v.cls,
        cell.expected && cell.verdict !== 'n/a' && 'ring-1 ring-inset ring-border-strong',
      )}
    >
      <Icon size={14} aria-hidden className={v.spin ? 'animate-spin' : undefined} />
    </span>
  )
  if (cell.run_id)
    return (
      <Tooltip content={tip}>
        <Link to={`/runs/${cell.run_id}`} aria-label={tip} className="rounded hover:scale-110">
          {content}
        </Link>
      </Tooltip>
    )
  if (cell.verdict === 'untested')
    return (
      <Tooltip content={`${tip}. Click to run.`}>
        <button
          type="button"
          aria-label={`Run ${agentLabel} on ${bench.title}`}
          onClick={onRun}
          className="rounded hover:scale-110"
        >
          {content}
        </button>
      </Tooltip>
    )
  return (
    <Tooltip content={tip}>
      <span aria-label={tip}>{content}</span>
    </Tooltip>
  )
}

export default function Diagnostics() {
  const q = useDiagnostics()
  const navigate = useNavigate()
  const { toast } = useToast()
  const run = useRunMutation(api.runDiagnostics, {
    onSuccess: (res) =>
      toast({
        title: res.queued ? `Queued ${plural(res.queued, 'probe run')}` : 'Nothing to run',
        description: res.queued
          ? 'They start as training slots free up.'
          : 'Every expected check is passing or already running.',
        tone: res.queued ? 'success' : 'info',
      }),
  })

  if (q.error) return <ErrorState error={q.error} onRetry={q.refetch} title="Could not load diagnostics" />
  const data = q.data
  const cells = data ? data.benchmarks.flatMap((b) => Object.values(b.cells).filter((c) => c.expected)) : []
  const passed = cells.filter((c) => c.verdict === 'pass').length
  const failed = cells.filter((c) => c.verdict === 'fail').length
  const running = cells.filter((c) => c.verdict === 'running').length
  // Only show algorithms that are expected to pass at least one probe, plus any with results.
  const agents = data
    ? data.agents.filter((a) => data.benchmarks.some((b) => b.cells[a.name].expected || b.cells[a.name].run_id))
    : []

  return (
    <div className="space-y-8">
      <PageHeader
        title="Diagnostics"
        description="Five tiny environments with known answers. If an algorithm fails one, there is a bug in its implementation or its settings, not a hard problem."
        actions={
          <Button
            variant="primary"
            icon={PlayCircle}
            loading={run.isPending}
            onClick={() => run.mutate({ only_missing: true })}
          >
            Run missing checks
          </Button>
        }
      />

      <StatStrip>
        <Stat
          label="Expected checks"
          value={q.isLoading ? '—' : cells.length}
          hint="Algorithm × probe pairs that should pass"
        />
        <Stat
          label="Passing"
          value={q.isLoading ? '—' : passed}
          tone={passed ? 'success' : undefined}
          hint={cells.length ? `${Math.round((passed / cells.length) * 100)}% healthy` : undefined}
        />
        <Stat
          label="Failing"
          value={q.isLoading ? '—' : failed}
          tone={failed ? 'danger' : undefined}
          hint={failed ? 'Open a red cell to read its run' : 'None'}
        />
        <Stat
          label="Running"
          value={q.isLoading ? '—' : running}
          tone={running ? 'accent' : undefined}
          hint={running ? 'Updates automatically' : 'Idle'}
        />
      </StatStrip>

      {q.isLoading ? (
        <Skeleton className="h-80" />
      ) : (
        <Card pad="none" className="overflow-x-auto">
          <table className="w-auto min-w-max border-collapse text-body-s">
            <thead>
              <tr className="border-b border-border">
                <th
                  scope="col"
                  className="sticky left-0 z-10 bg-surface px-5 py-3 text-left text-caption font-medium text-subtle"
                >
                  Probe
                </th>
                {agents.map((a) => (
                  <th key={a.name} scope="col" className="h-32 w-11 px-0 align-bottom">
                    <Link
                      to={`/algorithms/${a.name}`}
                      className="mx-auto mb-2 block w-6 whitespace-nowrap text-caption font-medium text-muted [writing-mode:vertical-rl] rotate-180 hover:text-text"
                    >
                      {a.label}
                    </Link>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.benchmarks.map((b) => (
                <tr key={b.id} className="border-b border-border/70 last:border-0">
                  <th scope="row" className="sticky left-0 z-10 bg-surface px-5 py-3 text-left font-normal">
                    <span className="block font-medium text-text">{b.title}</span>
                    <span className="num block text-caption text-subtle">
                      {b.probe} · pass at {b.pass_threshold}
                    </span>
                  </th>
                  {agents.map((a) => (
                    <td key={a.name} className="px-1.5 py-2 text-center">
                      <div className="flex justify-center">
                        <Cell
                          cell={b.cells[a.name]}
                          bench={b}
                          agentLabel={a.label}
                          onRun={() => run.mutate({ agents: [a.name], benchmarks: [b.id], only_missing: false })}
                        />
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex flex-wrap items-center gap-4 border-t border-border px-5 py-3 text-caption text-subtle">
            {Object.entries(VERDICT).map(([k, v]) => (
              <span key={k} className="flex items-center gap-1.5">
                <span className={cn('grid h-5 w-5 place-items-center rounded-sm', v.cls)}>
                  <v.icon size={11} aria-hidden />
                </span>
                {v.label}
              </span>
            ))}
            <span className="flex items-center gap-1.5">
              <span className="h-5 w-5 rounded-sm ring-1 ring-inset ring-border-strong" /> Expected to pass
            </span>
          </div>
        </Card>
      )}

      {data && (
        <section className="space-y-3" aria-label="What each probe checks">
          <h2 className="text-title-s font-semibold">What each probe checks</h2>
          {data.benchmarks.map((b) => (
            <Card key={b.id} className="grid gap-6 md:grid-cols-[120px_minmax(0,1fr)]">
              <div className="hidden h-24 w-24 md:block">
                <LabSpecimen visual="probe" className="h-full w-full" />
              </div>
              <div className="min-w-0">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h3 className="text-title-s font-semibold">{b.title}</h3>
                  <button
                    type="button"
                    onClick={() => navigate(`/envs/${b.probe}`)}
                    className="num text-caption text-accent hover:underline"
                  >
                    {b.probe}
                  </button>
                </div>
                <p className="mt-1 text-body-s text-muted">{b.tests}</p>
                <dl className="mt-4 grid gap-4 text-body-s sm:grid-cols-3">
                  <div>
                    <dt className="label">Why it matters</dt>
                    <dd className="mt-1 text-muted">{b.why}</dd>
                  </div>
                  <div>
                    <dt className="label">Healthy result</dt>
                    <dd className="mt-1 text-muted">{b.expected}</dd>
                  </div>
                  <div>
                    <dt className="label">A failure suggests</dt>
                    <dd className="mt-1 text-muted">{b.failure_suggests}</dd>
                  </div>
                </dl>
              </div>
            </Card>
          ))}
        </section>
      )}
    </div>
  )
}
