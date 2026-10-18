import { Link } from 'react-router-dom'
import { ArrowUpRight, Map, Play, Plus } from 'lucide-react'
import { useRouteLab } from '../app/AppContext.jsx'
import { useCatalogue, useLiveRun, useRuns } from '../lib/queries.js'
import { fmtNum, fmtRelative, isActive, plural } from '../lib/format.js'
import { seriesFromHistory } from '../lib/curves.js'
import { LabSpecimen } from '../components/art/LabArt.jsx'
import { ButtonLink } from '../components/ui/Button.jsx'
import { Card, CardHeader, Section } from '../components/ui/Card.jsx'
import { EmptyState, ErrorState, Skeleton, SkeletonRows } from '../components/ui/Feedback.jsx'
import { Stat, StatStrip } from '../components/ui/Page.jsx'
import { StatusBadge } from '../components/ui/Badge.jsx'
import { RunTable } from '../components/domain/RunTable.jsx'
import { categoryLabel } from '../components/domain/CatalogCards.jsx'
import { ChartFrame } from '../components/charts/ChartFrame.jsx'
import NotFound from './NotFound.jsx'

function composeLink(labId, step) {
  const q = new URLSearchParams({ agent: step.agent, env: step.env, episodes: step.episodes })
  return `/lab/${labId}/new?${q}`
}

/** The most recent run, with its real curve. */
function ContinueCard({ run, labId, envBy, threshold }) {
  const { history } = useLiveRun(run.id)
  const series = history.data ? [seriesFromHistory(history.data, 'reward', { label: 'Episode return' })] : []
  return (
    <ChartFrame
      title={
        <span className="flex flex-wrap items-center gap-2">
          <Link to={`/runs/${run.id}`} className="hover:text-accent">
            {run.name}
          </Link>
          <StatusBadge status={run.status} />
        </span>
      }
      description={`${run.agent} on ${envBy[run.env]?.label || run.env} · started ${fmtRelative(run.created_at)}`}
      series={series}
      loading={history.isLoading}
      threshold={threshold != null ? { value: threshold } : undefined}
      height={200}
      filename={`run-${run.id}`}
      actions={
        <ButtonLink to={`/runs/${run.id}`} size="sm" variant="ghost" icon={ArrowUpRight}>
          Open
        </ButtonLink>
      }
      emptyTitle={isActive(run.status) ? 'Waiting for the first episode' : 'This run recorded no episodes'}
      emptyDescription={
        isActive(run.status)
          ? 'The curve appears as soon as the worker reports progress.'
          : 'Open the run to read its log.'
      }
      emptyAction={
        !isActive(run.status) && (
          <ButtonLink to={`/lab/${labId}/new`} size="sm" variant="primary" icon={Plus}>
            New run
          </ButtonLink>
        )
      }
    />
  )
}

export default function LabOverview() {
  const { labId, lab, unknown, isLoading, error } = useRouteLab()
  const { envBy, agentBy, envs } = useCatalogue()
  const runs = useRuns({ lab: labId })

  if (unknown) return <NotFound what={`a lab called “${labId}”`} />
  if (error) return <ErrorState error={error} title="Could not load labs" />
  if (isLoading || !lab)
    return (
      <div className="space-y-6">
        <Skeleton className="h-52" />
        <Skeleton className="h-20" />
        <SkeletonRows rows={6} />
      </div>
    )

  const list = runs.data || []
  const training = list.filter((r) => ['running', 'stopping'].includes(r.status))
  const queued = list.filter((r) => r.status === 'queued')
  const attention = list.filter((r) => ['failed', 'interrupted'].includes(r.status))
  const best = list
    .filter((r) => r.best_eval_reward !== null && r.best_eval_reward !== undefined)
    .sort((a, b) => b.best_eval_reward - a.best_eval_reward)[0]
  // The run worth showing: one that is training, else the newest that has data.
  const latest = training[0] || list.find((r) => !['queued', 'created'].includes(r.status)) || list[0]
  const labEnvs = envs.filter((e) => e.lab === labId)
  const families = Object.entries(
    labEnvs.reduce((acc, e) => ({ ...acc, [e.category]: (acc[e.category] || 0) + 1 }), {}),
  )
  const unavailable = labEnvs.filter((e) => e.available === false)
  const first = lab.curriculum?.[0]

  return (
    <div className="space-y-8">
      {/* Hero: the lab's identity and its single primary action. */}
      <section className="relative overflow-hidden rounded-xl border border-border bg-surface">
        <div className="grid items-center gap-6 p-6 md:grid-cols-[minmax(0,1fr)_220px] md:p-8 lg:grid-cols-[minmax(0,1fr)_260px]">
          <div className="min-w-0">
            <p className="text-body-s font-medium text-accent">{lab.tagline}</p>
            <h1 className="mt-2 text-hero font-semibold tracking-tight text-text">{lab.label}</h1>
            <p className="mt-3 max-w-[62ch] text-body text-muted">{lab.description}</p>
            <div className="mt-6 flex flex-wrap gap-2">
              <ButtonLink to={`/lab/${labId}/new`} variant="primary" size="lg" icon={Play}>
                New run
              </ButtonLink>
              <ButtonLink to={`/lab/${labId}/envs`} size="lg" icon={Map}>
                {plural(labEnvs.length, 'environment')}
              </ButtonLink>
            </div>
          </div>
          <div className="mx-auto hidden aspect-square w-full max-w-[260px] rounded-lg border border-border bg-bg-soft p-5 md:block">
            <LabSpecimen visual={lab.visual} className="h-full w-full" />
          </div>
        </div>
      </section>

      <StatStrip>
        <Stat
          label="Runs in this lab"
          value={runs.isLoading ? '—' : list.length}
          hint={`${new Set(list.map((r) => r.group_id).filter(Boolean)).size} seed groups`}
        />
        <Stat
          label="Training now"
          value={runs.isLoading ? '—' : training.length}
          tone={training.length ? 'accent' : undefined}
          hint={
            queued.length
              ? `${queued.length} queued`
              : training.length
                ? training
                    .map((r) => r.agent)
                    .slice(0, 3)
                    .join(', ')
                : 'Nothing running'
          }
        />
        <Stat
          label="Best evaluation"
          value={best ? fmtNum(best.best_eval_reward) : '—'}
          hint={best ? `${best.agent} on ${envBy[best.env]?.label || best.env}` : 'No evaluated runs yet'}
        />
        <Stat
          label="Needs attention"
          value={runs.isLoading ? '—' : attention.length}
          tone={attention.length ? 'danger' : undefined}
        >
          {attention.length > 0 ? (
            <Link
              to={`/lab/${labId}/runs?status=failed`}
              className="mt-0.5 block text-caption text-accent hover:underline"
            >
              Review failed runs
            </Link>
          ) : (
            <div className="mt-0.5 text-caption text-subtle">No failed or interrupted runs</div>
          )}
        </Stat>
      </StatStrip>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-6">
          {runs.error ? (
            <ErrorState error={runs.error} onRetry={runs.refetch} title="Could not load runs" />
          ) : runs.isLoading ? (
            <Skeleton className="h-72" />
          ) : latest ? (
            <ContinueCard run={latest} labId={labId} envBy={envBy} threshold={envBy[latest.env]?.reward_threshold} />
          ) : (
            <EmptyState
              art={<LabSpecimen visual={lab.visual} className="h-24 w-24 opacity-80" />}
              title={`No runs in ${lab.label} yet`}
              description={
                first
                  ? `Start with “${first.title}”: ${first.why}`
                  : 'Pick an environment and an algorithm to launch your first experiment.'
              }
              action={
                <>
                  {first && (
                    <ButtonLink to={composeLink(labId, first)} variant="primary" icon={Play}>
                      {agentBy[first.agent]?.label || first.agent} on {envBy[first.env]?.label || first.env}
                    </ButtonLink>
                  )}
                  <ButtonLink to={`/lab/${labId}/new`} icon={Plus}>
                    Choose my own
                  </ButtonLink>
                </>
              }
            />
          )}

          {list.length > 0 && (
            <Section
              title="Recent runs"
              actions={
                <ButtonLink to={`/lab/${labId}/runs`} size="sm" variant="ghost" icon={ArrowUpRight}>
                  All {list.length}
                </ButtonLink>
              }
            >
              <RunTable runs={list.slice(0, 6)} compact envBy={envBy} />
            </Section>
          )}
        </div>

        <aside className="space-y-6">
          <Card>
            <CardHeader title="Curriculum" description="Experiments that build on each other, in order." />
            <ol className="space-y-1">
              {(lab.curriculum || []).map((step, i) => {
                const done = list.some((r) => r.agent === step.agent && r.env === step.env && r.status === 'completed')
                const env = envBy[step.env]
                const blocked = env?.available === false
                return (
                  <li key={`${step.agent}-${step.env}`}>
                    <Link
                      to={composeLink(labId, step)}
                      className="group flex gap-3 rounded p-2 hover:bg-surface-2"
                      aria-disabled={blocked}
                    >
                      <span
                        className={
                          done
                            ? 'num grid h-6 w-6 shrink-0 place-items-center rounded-full bg-success/15 text-caption text-success'
                            : 'num grid h-6 w-6 shrink-0 place-items-center rounded-full border border-border-strong text-caption text-muted'
                        }
                      >
                        {i + 1}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block text-body-s font-medium text-text group-hover:text-accent">
                          {step.title}
                        </span>
                        <span className="block text-caption text-muted">{step.why}</span>
                        <span className="mt-0.5 block text-caption text-subtle">
                          {agentBy[step.agent]?.label || step.agent} · {env?.label || step.env}
                          {blocked && ' · not installed'}
                        </span>
                      </span>
                    </Link>
                  </li>
                )
              })}
            </ol>
          </Card>

          <Card>
            <CardHeader title="In this lab" />
            <div className="space-y-4">
              <div>
                <div className="label mb-2">Environment families</div>
                <div className="flex flex-wrap gap-1.5">
                  {families.map(([cat, n]) => (
                    <Link
                      key={cat}
                      to={`/lab/${labId}/envs?category=${cat}`}
                      className="inline-flex h-7 items-center gap-1.5 rounded-full border border-border bg-surface-2 px-3 text-caption text-muted hover:border-border-strong hover:text-text"
                    >
                      {categoryLabel(cat)} <span className="num text-subtle">{n}</span>
                    </Link>
                  ))}
                </div>
                {unavailable.length > 0 && (
                  <p className="mt-2 text-caption text-subtle">
                    {plural(unavailable.length, 'environment')} need an optional install.
                  </p>
                )}
              </div>
              <div>
                <div className="label mb-2">Suited algorithms</div>
                <div className="flex flex-wrap gap-1.5">
                  {(lab.algorithms || [])
                    .filter((a) => agentBy[a]?.status === 'implemented')
                    .map((a) => (
                      <Link
                        key={a}
                        to={`/algorithms/${a}`}
                        className="inline-flex h-7 items-center rounded-full border border-border bg-surface-2 px-3 text-caption text-muted hover:border-border-strong hover:text-text"
                      >
                        {agentBy[a]?.label || a}
                      </Link>
                    ))}
                </div>
              </div>
            </div>
          </Card>
        </aside>
      </div>
    </div>
  )
}
