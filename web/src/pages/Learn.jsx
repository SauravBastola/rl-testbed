import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { CheckCircle2, ChevronDown, CircleSlash, FileCode2, GitCompareArrows, Loader2, Play } from 'lucide-react'
import { api } from '../lib/api.js'
import { useCatalogue, useCurriculum, useGlossary } from '../lib/queries.js'
import { fmtNum, plural } from '../lib/format.js'
import { cn } from '../lib/utils.js'
import { Button, ButtonLink } from '../components/ui/Button.jsx'
import { ErrorState, Notice, SkeletonRows } from '../components/ui/Feedback.jsx'
import { PageHeader, Progress } from '../components/ui/Page.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { useToast } from '../components/ui/Toast.jsx'

const DRAFT_KEY = 'rltestbed.experiment-draft'

function StatusMark({ state, n }) {
  const base = 'grid h-9 w-9 shrink-0 place-items-center rounded-full border text-body-s font-semibold'
  if (state === 'done')
    return (
      <span className={cn(base, 'border-success/40 bg-success/15 text-success')}>
        <CheckCircle2 size={18} aria-label="Done" />
      </span>
    )
  if (state === 'running')
    return (
      <span className={cn(base, 'border-accent/40 bg-accent/10 text-accent')}>
        <Loader2 size={16} className="animate-spin" aria-label="Running" />
      </span>
    )
  if (state === 'unavailable')
    return (
      <span className={cn(base, 'border-border text-subtle')}>
        <CircleSlash size={16} aria-label="Needs an optional install" />
      </span>
    )
  return <span className={cn(base, 'num border-border-strong bg-bg text-muted')}>{n}</span>
}

/** One line per experiment entry: "PPO on CartPole · 3 seeds × 300 episodes". */
function ExperimentSummary({ experiment, agentBy, envBy }) {
  const seeds = (e) => plural((e.seeds || [0]).length, 'seed')
  return (
    <ul className="space-y-1 text-body-s">
      {(experiment.runs || []).map((r, i) => (
        <li key={`r${i}`} className="text-muted">
          <span className="text-text">{r.name || agentBy[r.agent]?.label || r.agent}</span> on{' '}
          {envBy[r.env]?.label || r.env}
          <span className="text-subtle">
            {' '}
            · {seeds(r)} × {r.episodes} episodes
            {r.params &&
              Object.keys(r.params).length > 0 &&
              ` · ${Object.entries(r.params)
                .map(([k, v]) => `${k}=${v}`)
                .join(', ')}`}
          </span>
        </li>
      ))}
      {(experiment.sweeps || []).map((s, i) => (
        <li key={`s${i}`} className="text-muted">
          <span className="text-text">Sweep</span> of {agentBy[s.agent]?.label || s.agent} on{' '}
          {envBy[s.env]?.label || s.env}
          <span className="text-subtle">
            {' '}
            ·{' '}
            {Object.entries(s.space)
              .map(([k, d]) => `${k} ∈ {${(d.values || []).join(', ')}}`)
              .join(', ')}{' '}
            · {seeds(s)} each
          </span>
        </li>
      ))}
    </ul>
  )
}

function Step({ step, n, open, onToggle, termBy, agentBy, envBy }) {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const { toast } = useToast()
  const st = step.status
  const launch = useMutation({
    mutationFn: () => api.launchExperiment(step.experiment_text),
    onSuccess: (out) => {
      qc.invalidateQueries({ queryKey: ['curriculum'] })
      qc.invalidateQueries({ queryKey: ['runs'] })
      toast({ title: `Started “${step.title}”`, description: plural(out.run_ids.length, 'run') })
      if (out.sweeps.length) navigate(`/sweeps/${out.sweeps[0].sweep_id}`)
      else navigate(`/compare?runs=${out.run_ids.join(',')}&group_by=config`)
    },
  })
  const editAsFile = () => {
    try {
      window.localStorage.setItem(DRAFT_KEY, step.experiment_text)
    } catch {
      /* storage disabled: the page opens with its previous draft */
    }
    navigate('/experiments')
  }
  const hasSweep = (step.experiment.sweeps || []).length > 0

  return (
    <li className="relative flex gap-4">
      <StatusMark state={st.state} n={n} />
      <div className="min-w-0 flex-1 rounded-lg border border-border bg-surface">
        <button
          type="button"
          aria-expanded={open}
          onClick={onToggle}
          className="flex w-full items-start gap-3 px-5 py-4 text-left"
        >
          <span className="min-w-0 flex-1">
            <span className="flex flex-wrap items-center gap-2">
              <span className="text-title-s font-semibold text-text">{step.title}</span>
              {st.state === 'done' && <Badge tone="success">Done</Badge>}
              {st.state === 'running' && <Badge tone="accent">Running</Badge>}
              {st.state === 'unavailable' && <Badge tone="outline">Needs an optional install</Badge>}
            </span>
            <span className="mt-0.5 block text-body-s text-muted">{step.summary}</span>
          </span>
          <ChevronDown
            size={18}
            aria-hidden
            className={cn('mt-1 shrink-0 text-subtle transition-transform duration-fast', open && 'rotate-180')}
          />
        </button>
        {open && (
          <div className="space-y-5 border-t border-border px-5 py-4">
            <div>
              <div className="label mb-1.5">Concepts</div>
              <div className="flex flex-wrap gap-1.5">
                {step.concepts.map((c) => (
                  <Link
                    key={c}
                    to={`/glossary#${c}`}
                    className="inline-flex h-7 items-center rounded-full border border-border bg-surface-2 px-3 text-caption text-muted hover:border-border-strong hover:text-text"
                  >
                    {termBy[c]?.term || c}
                  </Link>
                ))}
              </div>
            </div>
            <div>
              <div className="label mb-1.5">The experiment</div>
              <ExperimentSummary experiment={step.experiment} agentBy={agentBy} envBy={envBy} />
            </div>
            <div>
              <div className="label mb-1.5">What to look for</div>
              <p className="text-body-s text-muted">{step.watch}</p>
              {step.target && (
                <p className="mt-1 text-caption text-subtle">
                  Done when the best greedy evaluation reaches{' '}
                  <span className="num text-text">{fmtNum(step.target.at_least)}</span>
                  {st.best_eval !== null && st.best_eval !== undefined && (
                    <>
                      {' '}
                      (best so far <span className="num text-text">{fmtNum(st.best_eval)}</span>)
                    </>
                  )}
                  .
                </p>
              )}
            </div>
            <details className="group rounded-lg border border-border bg-surface-2 px-4 py-3">
              <summary className="cursor-pointer list-none text-body-s font-medium text-text">
                <span className="text-accent">Check your understanding: </span>
                {step.question}
                <span className="ml-2 text-caption text-subtle group-open:hidden">Show answer</span>
              </summary>
              <p className="mt-2 text-body-s text-muted">{step.answer}</p>
            </details>
            {st.state === 'unavailable' && (
              <Notice tone="warning">
                Needs {st.missing_envs.map((e) => envBy[e]?.label || e).join(', ')}, which is not installed. The
                environment page explains how to install it.
              </Notice>
            )}
            {launch.error && <Notice tone="danger">{launch.error.message}</Notice>}
            <div className="flex flex-wrap gap-2">
              <Button
                variant={st.state === 'done' ? 'secondary' : 'primary'}
                icon={Play}
                loading={launch.isPending}
                disabled={st.state === 'unavailable'}
                onClick={() => launch.mutate()}
              >
                {st.state === 'done' ? 'Run it again' : 'Run this step'}
              </Button>
              {st.run_ids.length > 0 &&
                (hasSweep ? (
                  <ButtonLink
                    to={`/lab/${envBy[step.experiment.sweeps[0].env]?.lab || 'foundations'}/sweeps`}
                    icon={GitCompareArrows}
                  >
                    See the sweep
                  </ButtonLink>
                ) : (
                  <ButtonLink to={`/compare?runs=${st.run_ids.join(',')}&group_by=config`} icon={GitCompareArrows}>
                    See results
                  </ButtonLink>
                ))}
              <Button variant="ghost" icon={FileCode2} onClick={editAsFile}>
                Edit as experiment file
              </Button>
            </div>
          </div>
        )}
      </div>
    </li>
  )
}

/** The curriculum: modules of steps, each an experiment with a purpose. */
export default function Learn() {
  const q = useCurriculum()
  const glossary = useGlossary()
  const { agentBy, envBy } = useCatalogue()
  const [openId, setOpenId] = useState(null)
  const termBy = Object.fromEntries((glossary.data || []).map((g) => [g.id, g]))

  if (q.error) return <ErrorState error={q.error} onRetry={q.refetch} />
  const data = q.data
  const firstOpen = data?.modules.flatMap((m) => m.steps).find((s) => s.status.state !== 'done')?.id
  const current = openId ?? firstOpen
  let n = 0

  return (
    <div>
      <PageHeader
        title="Learning path"
        description="Fifteen experiments from a random baseline to reading results across seeds. Each step teaches named concepts, runs a ready-made comparison, and marks itself done from your runs."
      />
      {q.isLoading ? (
        <SkeletonRows rows={8} />
      ) : (
        <>
          <div className="mb-8 flex items-center gap-4 rounded-lg border border-border bg-surface px-5 py-4">
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between">
                <span className="text-body-s font-medium text-text">Your progress</span>
                <span className="num text-caption text-muted">
                  {data.done} of {data.total} steps
                </span>
              </div>
              <Progress value={data.total ? data.done / data.total : 0} className="mt-2 h-1.5" tone="success" />
            </div>
          </div>
          <div className="space-y-10">
            {data.modules.map((m, mi) => (
              <section key={m.id} aria-labelledby={`mod-${m.id}`}>
                <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
                  <div>
                    <h2 id={`mod-${m.id}`} className="text-title font-semibold">
                      <span className="num mr-2 text-accent">{mi + 1}</span>
                      {m.title}
                    </h2>
                    <p className="mt-0.5 text-body-s text-muted">{m.summary}</p>
                  </div>
                  <span className="num text-caption text-subtle">
                    {m.done}/{m.steps.length} done
                  </span>
                </div>
                <ol className="space-y-3">
                  {m.steps.map((s) => {
                    n += 1
                    return (
                      <Step
                        key={s.id}
                        step={s}
                        n={n}
                        open={current === s.id}
                        onToggle={() => setOpenId(current === s.id ? '' : s.id)}
                        termBy={termBy}
                        agentBy={agentBy}
                        envBy={envBy}
                      />
                    )
                  })}
                </ol>
              </section>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
