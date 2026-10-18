import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Check, Download, FileCode2, GitCompareArrows, Layers, Link2, Plus, X } from 'lucide-react'
import { api } from '../lib/api.js'
import { useCatalogue, useRuns } from '../lib/queries.js'
import { fmtDuration, fmtNum, metricLabel, plural } from '../lib/format.js'
import { colorVar, SERIES_COLORS } from '../lib/chart.js'
import { seriesFromGroup } from '../lib/curves.js'
import { cn } from '../lib/utils.js'
import { Button, buttonClass } from '../components/ui/Button.jsx'
import { Card, CardHeader, Section } from '../components/ui/Card.jsx'
import { SearchInput } from '../components/ui/Field.jsx'
import { EmptyState, ErrorState, Notice, Skeleton } from '../components/ui/Feedback.jsx'
import { PageHeader } from '../components/ui/Page.jsx'
import { Segmented } from '../components/ui/Segmented.jsx'
import { StatusBadge } from '../components/ui/Badge.jsx'
import { Table, THead, Th, Tr, Td } from '../components/ui/Table.jsx'
import { useToast } from '../components/ui/Toast.jsx'
import { ChartFrame } from '../components/charts/ChartFrame.jsx'

const BASE_METRICS = ['reward', 'mean_reward', 'eval', 'steps']
const signature = (r) => `${r.agent}|${r.env}|${JSON.stringify(r.params || {}, Object.keys(r.params || {}).sort())}`
const meanStd = (s) =>
  s?.mean === null || s?.mean === undefined
    ? '—'
    : s.std !== null && s.std !== undefined
      ? `${fmtNum(s.mean)} ± ${fmtNum(s.std)}`
      : fmtNum(s.mean)

function Picker({ runs, selected, onToggle, onAddSeeds }) {
  const [q, setQ] = useState('')
  const tokens = q.toLowerCase().split(/\s+/).filter(Boolean)
  const shown = runs.filter((r) =>
    tokens.every((t) => `${r.name} ${r.id} ${r.agent} ${r.env}`.toLowerCase().includes(t)),
  )
  const selectedRuns = runs.filter((r) => selected.includes(r.id))
  return (
    <Card pad="sm" className="flex h-full flex-col">
      <div className="mb-2 flex items-center justify-between px-1">
        <h2 className="text-body-s font-semibold">Runs</h2>
        <span className="num text-caption text-subtle">{selected.length} selected</span>
      </div>
      <SearchInput value={q} onChange={setQ} placeholder="Filter runs" />
      {selectedRuns.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {selectedRuns.map((r) => (
            <span
              key={r.id}
              className="inline-flex h-7 items-center gap-1 rounded-full border border-border bg-surface-2 pl-2.5 pr-1 text-caption text-text"
            >
              <span className="max-w-36 truncate">{r.name}</span>
              <button
                type="button"
                aria-label={`Remove ${r.name}`}
                onClick={() => onToggle(r.id)}
                className="rounded-full p-0.5 text-subtle hover:bg-surface-3 hover:text-text"
              >
                <X size={12} aria-hidden />
              </button>
            </span>
          ))}
        </div>
      )}
      <div
        role="listbox"
        aria-multiselectable="true"
        aria-label="Runs to compare"
        className="mt-2 max-h-[460px] min-h-0 flex-1 space-y-0.5 overflow-y-auto"
      >
        {shown.map((r) => {
          const on = selected.includes(r.id)
          const siblings = runs.filter(
            (x) => x.id !== r.id && signature(x) === signature(r) && !selected.includes(x.id),
          )
          return (
            <div
              key={r.id}
              className={cn(
                'group flex items-center gap-2 rounded px-2 py-1.5',
                on ? 'bg-accent/10' : 'hover:bg-surface-2',
              )}
            >
              <button
                type="button"
                role="option"
                aria-selected={on}
                onClick={() => onToggle(r.id)}
                className="flex min-w-0 flex-1 items-center gap-2 text-left"
              >
                <span
                  className={cn(
                    'grid h-4 w-4 shrink-0 place-items-center rounded-sm border',
                    on ? 'border-accent bg-accent text-accent-contrast' : 'border-border-strong',
                  )}
                >
                  {on && <Check size={11} aria-hidden />}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-body-s text-text">{r.name}</span>
                  <span className="block truncate text-caption text-subtle">
                    {r.agent} · {r.env} · seed <span className="num">{r.seed}</span>
                  </span>
                </span>
              </button>
              {on && siblings.length > 0 && (
                <button
                  type="button"
                  onClick={() => onAddSeeds(siblings.map((s) => s.id))}
                  className="shrink-0 rounded px-1.5 py-0.5 text-caption text-accent hover:bg-accent/10"
                  title="Add runs with the same configuration but different seeds"
                >
                  +{siblings.length} seeds
                </button>
              )}
              <StatusBadge status={r.status} className="hidden 2xl:inline-flex" />
            </div>
          )
        })}
        {!shown.length && <p className="p-4 text-center text-body-s text-muted">No runs match.</p>}
      </div>
    </Card>
  )
}

/** Shown before anything is selected: what a good comparison needs, and pairs worth comparing. */
function CompareGuide({ runs, onPick }) {
  const groups = Object.values(
    runs.reduce((acc, r) => {
      const k = signature(r)
      ;(acc[k] = acc[k] || []).push(r)
      return acc
    }, {}),
  )
  const byEnv = {}
  groups.forEach((g) => {
    ;(byEnv[g[0].env] = byEnv[g[0].env] || []).push(g)
  })
  const suggestions = Object.entries(byEnv)
    .filter(([, gs]) => gs.length > 1)
    .slice(0, 4)
    .map(([env, gs]) => ({ env, groups: gs.slice(0, 3) }))
  const seedGroups = groups.filter((g) => g.length > 1).slice(0, 3)
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader title="Make the comparison count" icon={Layers} />
        <ul className="space-y-2.5 text-body-s text-muted">
          <li>
            <span className="text-text">Use several seeds.</span> One run per configuration is an anecdote. Three or
            more seeds show whether a difference is real; the shaded band is ±1 standard deviation.
          </li>
          <li>
            <span className="text-text">Change one thing.</span> The configuration diff below the chart lists every
            setting that differs. If it lists more than one, you cannot tell which mattered.
          </li>
          <li>
            <span className="text-text">Match the budget.</span> Compare at the same number of episodes, and read greedy
            evaluations, not only noisy training returns.
          </li>
        </ul>
      </Card>
      <Card>
        <CardHeader title="Worth comparing" description="Found in your run history." />
        {suggestions.length || seedGroups.length ? (
          <ul className="space-y-1">
            {seedGroups.map((g) => (
              <li key={g[0].id}>
                <button
                  type="button"
                  onClick={() => onPick(g.map((r) => r.id))}
                  className="flex w-full items-center gap-3 rounded p-2 text-left hover:bg-surface-2"
                >
                  <Layers size={14} className="text-accent" aria-hidden />
                  <span className="min-w-0 flex-1 truncate text-body-s text-text">
                    {g[0].agent} on {g[0].env}
                  </span>
                  <span className="text-caption text-subtle">{plural(g.length, 'seed')}</span>
                </button>
              </li>
            ))}
            {suggestions.map((s) => (
              <li key={s.env}>
                <button
                  type="button"
                  onClick={() => onPick(s.groups.flatMap((g) => g.map((r) => r.id)))}
                  className="flex w-full items-center gap-3 rounded p-2 text-left hover:bg-surface-2"
                >
                  <GitCompareArrows size={14} className="text-accent" aria-hidden />
                  <span className="min-w-0 flex-1 truncate text-body-s text-text">
                    {s.groups.map((g) => g[0].agent).join(' vs ')}
                  </span>
                  <span className="truncate text-caption text-subtle">{s.env}</span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <EmptyState
            compact
            icon={GitCompareArrows}
            title="Nothing to pair up yet"
            description="Train two algorithms on the same environment, or launch one configuration with 3 seeds."
          />
        )}
      </Card>
    </div>
  )
}

export default function Compare() {
  const [params, setParams] = useSearchParams()
  const { toast } = useToast()
  const { agents } = useCatalogue()
  const runs = useRuns({ limit: 300 })
  const groupParam = params.get('group')
  const groupRuns = useRuns(groupParam ? { group_id: groupParam } : { limit: 0 })
  const selected = useMemo(() => {
    const fromUrl = (params.get('runs') || '').split(',').filter(Boolean)
    const fromGroup = groupParam ? (groupRuns.data || []).map((r) => r.id) : []
    return [...new Set([...fromUrl, ...fromGroup])]
  }, [params, groupParam, groupRuns.data])
  const metric = params.get('metric') || 'reward'
  const groupBy = params.get('group_by') || 'config'

  const update = (patch) => {
    const next = new URLSearchParams(params)
    Object.entries(patch).forEach(([k, v]) => (v ? next.set(k, v) : next.delete(k)))
    setParams(next, { replace: true })
  }
  const setSelected = (ids) => update({ runs: ids.join(','), group: '' })
  const toggle = (id) => setSelected(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id])

  const agg = useQuery({
    queryKey: ['aggregate', selected, metric, groupBy],
    queryFn: () => api.aggregate(selected, metric, groupBy),
    enabled: selected.length > 0,
    refetchInterval: (q) =>
      q.state.data?.groups?.some((g) => g.runs.some((r) => ['running', 'queued'].includes(r.status))) ? 3000 : false,
  })

  const metricOptions = useMemo(() => {
    const extra = new Set()
    agents.forEach((a) => (a.docs?.expected_metrics || []).forEach((m) => extra.add(`metric_${m.key}`)))
    return [...BASE_METRICS, ...[...extra].sort()].map((v) => ({ value: v, label: metricLabel(v) }))
  }, [agents])

  const groups = agg.data?.groups || []
  const series = groups.map(seriesFromGroup)
  const singleSeedOnly = groups.length > 0 && groups.every((g) => g.n_runs === 1)
  const diff = agg.data?.diff || {}
  const allRuns = runs.data || []

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href)
      toast({ title: 'Link copied', description: 'Anyone with access to this testbed sees the same comparison.' })
    } catch {
      toast({ title: 'Could not copy', description: 'Copy the address bar instead.', tone: 'danger' })
    }
  }

  return (
    <div>
      <PageHeader
        title="Compare"
        description="Overlay runs. Seeds of the same configuration are pooled into a mean with a ±1σ band."
        actions={
          selected.length > 0 && (
            <>
              <Button icon={Link2} onClick={copyLink}>
                Copy link
              </Button>
              <a href={api.exportUrl({ runIds: selected.join(',') })} download className={buttonClass()}>
                <FileCode2 size={16} aria-hidden /> Experiment file
              </a>
              <a href={api.compareCsvUrl(selected.join(','), metric, groupBy)} download className={buttonClass()}>
                <Download size={16} aria-hidden /> CSV
              </a>
              <Button variant="ghost" onClick={() => setParams({}, { replace: true })}>
                Clear
              </Button>
            </>
          )
        }
      />
      <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
        <div className="xl:sticky xl:top-20 xl:self-start">
          {runs.isLoading ? (
            <Skeleton className="h-96" />
          ) : runs.error ? (
            <ErrorState error={runs.error} onRetry={runs.refetch} />
          ) : !allRuns.length ? (
            <EmptyState
              icon={GitCompareArrows}
              title="No runs to compare"
              description="Launch at least two runs first."
              action={
                <Link to="/" className="text-body-s text-accent hover:underline">
                  Go to the lab overview
                </Link>
              }
            />
          ) : (
            <Picker
              runs={allRuns}
              selected={selected}
              onToggle={toggle}
              onAddSeeds={(ids) => setSelected([...selected, ...ids])}
            />
          )}
        </div>

        <div className="min-w-0 space-y-6">
          {!selected.length ? (
            <CompareGuide runs={allRuns} onPick={setSelected} />
          ) : (
            <>
              {singleSeedOnly && (
                <Notice tone="warning" title="Each configuration has one seed">
                  Differences between single runs are often just luck. Select more seeds of the same configuration, or
                  launch it with 3 seeds from New run.
                </Notice>
              )}
              {agg.error && <ErrorState error={agg.error} onRetry={agg.refetch} />}
              <ChartFrame
                title={metricLabel(metric)}
                description={
                  groupBy === 'config'
                    ? 'Mean across seeds; shaded band is ±1 standard deviation.'
                    : 'Each run on its own.'
                }
                series={series}
                loading={agg.isLoading}
                threshold={agg.data?.threshold != null ? { value: agg.data.threshold } : undefined}
                metric={metric}
                metrics={metricOptions}
                onMetricChange={(v) => update({ metric: v })}
                height={340}
                filename="comparison"
                actions={
                  <Segmented
                    size="sm"
                    label="Grouping"
                    value={groupBy}
                    onChange={(v) => update({ group_by: v })}
                    options={[
                      { value: 'config', label: 'Pool seeds' },
                      { value: 'run', label: 'Each run' },
                    ]}
                  />
                }
                emptyTitle={`No ${metricLabel(metric).toLowerCase()} recorded`}
                emptyDescription="These runs have not reported this metric. Try episode return."
              />

              {groups.length > 0 && (
                <Section title="Summary" description="Final is the mean training return over the last 10% of episodes.">
                  <Table>
                    <THead>
                      <Th>Configuration</Th>
                      <Th align="right">Seeds</Th>
                      <Th align="right">Final</Th>
                      <Th align="right">Best eval</Th>
                      <Th align="right">Episodes to solve</Th>
                      <Th align="right">Wall clock</Th>
                    </THead>
                    <tbody>
                      {groups.map((g, i) => (
                        <Tr key={g.key}>
                          <Td>
                            <span className="flex items-center gap-2">
                              <span
                                className="h-0.5 w-4 shrink-0 rounded"
                                style={{ background: colorVar(SERIES_COLORS[i % SERIES_COLORS.length]) }}
                              />
                              <span className="truncate font-medium text-text">{g.label}</span>
                            </span>
                          </Td>
                          <Td align="right" mono>
                            {g.n_runs}
                          </Td>
                          <Td align="right" mono>
                            {meanStd(g.summary.final)}
                          </Td>
                          <Td align="right" mono>
                            {meanStd(g.summary.best_eval)}
                          </Td>
                          <Td align="right" mono className="text-muted">
                            {agg.data?.threshold == null
                              ? 'no threshold'
                              : g.summary.reached_threshold
                                ? `${meanStd(g.summary.episodes_to_threshold)} (${g.summary.reached_threshold}/${g.n_runs})`
                                : `not reached`}
                          </Td>
                          <Td align="right" mono className="text-muted">
                            {fmtDuration(g.summary.wall_clock.mean)}
                          </Td>
                        </Tr>
                      ))}
                    </tbody>
                  </Table>
                </Section>
              )}

              {groups.some((g) => g.n_runs > 1) && (
                <Section title="Per seed">
                  <Table maxHeight={360}>
                    <THead>
                      <Th>Run</Th>
                      <Th>Configuration</Th>
                      <Th align="right">Seed</Th>
                      <Th align="right">Episodes</Th>
                      <Th align="right">Final</Th>
                      <Th align="right">Best eval</Th>
                      <Th>Status</Th>
                    </THead>
                    <tbody>
                      {groups.flatMap((g) =>
                        g.runs.map((r) => (
                          <Tr key={r.run_id}>
                            <Td mono>
                              <Link to={`/runs/${r.run_id}`} className="text-accent hover:underline">
                                {r.run_id}
                              </Link>
                            </Td>
                            <Td className="text-muted">{g.label}</Td>
                            <Td align="right" mono>
                              {r.seed}
                            </Td>
                            <Td align="right" mono className="text-muted">
                              {r.episodes}
                            </Td>
                            <Td align="right" mono>
                              {fmtNum(r.final)}
                            </Td>
                            <Td align="right" mono>
                              {fmtNum(r.best_eval)}
                            </Td>
                            <Td>
                              <StatusBadge status={r.status} />
                            </Td>
                          </Tr>
                        )),
                      )}
                    </tbody>
                  </Table>
                </Section>
              )}

              <Section title="What differs" description="Settings that are not identical across the selected runs.">
                {Object.keys(diff).length ? (
                  <Table>
                    <THead>
                      <Th>Setting</Th>
                      <Th>Values</Th>
                    </THead>
                    <tbody>
                      {Object.entries(diff).map(([k, vals]) => (
                        <Tr key={k}>
                          <Td mono className="text-text">
                            {k}
                          </Td>
                          <Td className="text-muted">
                            <span className="num">
                              {[...new Set(vals.map((v) => JSON.stringify(v)))].join('  ·  ')}
                            </span>
                          </Td>
                        </Tr>
                      ))}
                    </tbody>
                  </Table>
                ) : (
                  <p className="text-body-s text-muted">Nothing: every selected run used identical settings.</p>
                )}
              </Section>
            </>
          )}
          {!selected.length && allRuns.length > 0 && (
            <p className="text-body-s text-subtle">
              <Plus size={14} className="mr-1 inline" aria-hidden />
              Pick runs on the left to start.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
