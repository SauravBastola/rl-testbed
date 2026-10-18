import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { GitCompareArrows, Play, Square, Trash2, Trophy } from 'lucide-react'
import { useAdoptLab } from '../app/AppContext.jsx'
import { api } from '../lib/api.js'
import { useCatalogue, useSweep } from '../lib/queries.js'
import { fmtNum, plural } from '../lib/format.js'
import { SERIES_COLORS, colorVar } from '../lib/chart.js'
import { Button, ButtonLink } from '../components/ui/Button.jsx'
import { Card, Section } from '../components/ui/Card.jsx'
import { ErrorState, Notice, Skeleton } from '../components/ui/Feedback.jsx'
import { Modal } from '../components/ui/Overlay.jsx'
import { Progress, Stat, StatStrip } from '../components/ui/Page.jsx'
import { Segmented } from '../components/ui/Segmented.jsx'
import { StatusBadge } from '../components/ui/Badge.jsx'
import { Table, THead, Th, Tr, Td } from '../components/ui/Table.jsx'
import { useToast } from '../components/ui/Toast.jsx'
import { ChartFrame } from '../components/charts/ChartFrame.jsx'
import { ParamEffect } from '../components/domain/ParamEffect.jsx'
import { sweepStatus } from './Sweeps.jsx'
import NotFound from './NotFound.jsx'

const paramText = (params) =>
  Object.entries(params)
    .map(([k, v]) => `${k}=${v}`)
    .join(', ')

const meanStd = (s) =>
  s?.mean === null || s?.mean === undefined
    ? '—'
    : s.std !== null && s.std !== undefined
      ? `${fmtNum(s.mean)} ± ${fmtNum(s.std)}`
      : fmtNum(s.mean)

export default function SweepDetail() {
  const { sweepId } = useParams()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const { toast } = useToast()
  const { envBy, agentBy } = useCatalogue()
  const [rankBy, setRankBy] = useState('final')
  const [shown, setShown] = useState(3)
  const [confirm, setConfirm] = useState(false)
  const q = useSweep(sweepId, rankBy)
  const sw = q.data
  useAdoptLab(sw?.lab)

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ['sweep'] })
    qc.invalidateQueries({ queryKey: ['sweeps'] })
    qc.invalidateQueries({ queryKey: ['runs'] })
  }
  const stop = useMutation({
    mutationFn: () => api.stopSweep(sweepId),
    onSuccess: (r) => {
      refresh()
      toast({ title: `Stopping ${plural(r.stopped, 'run')}`, tone: 'info' })
    },
  })
  const del = useMutation({
    mutationFn: () => api.deleteSweep(sweepId),
    onSuccess: () => {
      refresh()
      toast({ title: 'Sweep deleted' })
      navigate(`/lab/${sw?.lab || 'foundations'}/sweeps`)
    },
  })

  if (q.error?.status === 404) return <NotFound what={`a sweep with id “${sweepId}”`} />
  if (q.error) return <ErrorState error={q.error} onRetry={q.refetch} title="Could not load this sweep" />
  if (q.isLoading || !sw)
    return (
      <div className="space-y-6">
        <Skeleton className="h-16" />
        <Skeleton className="h-24" />
        <Skeleton className="h-80" />
      </div>
    )

  const c = sw.counts
  const status = sweepStatus({ ...c, n_runs: sw.n_runs })
  const finished = c.completed + c.failed + c.stopped
  const active = c.running + c.queued > 0
  const swept = Object.keys(sw.request.space || {}).sort()
  const scored = sw.configs.filter((x) => x.score.mean !== null)
  const best = scored[0]
  const top = sw.configs.slice(0, shown)
  const series = top
    .filter((g) => g.points.length)
    .map((g, i) => ({
      id: g.group_id,
      label: `#${g.rank} ${paramText(g.params)}`,
      color: SERIES_COLORS[i % SERIES_COLORS.length],
      points: g.points.map((p) => ({ x: p.x, y: p.mean })),
      band:
        g.n_runs > 1
          ? g.points.filter((p) => p.std !== null).map((p) => ({ x: p.x, lo: p.mean - p.std, hi: p.mean + p.std }))
          : undefined,
    }))
  const scoreLabel = rankBy === 'final' ? 'Final return' : 'Best evaluation'
  const labId = sw.lab && sw.lab !== 'diagnostics' ? sw.lab : 'foundations'

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="truncate text-display font-semibold">{sw.name}</h1>
            <StatusBadge status={status} />
          </div>
          <p className="mt-1.5 text-body text-muted">
            <Link to={`/algorithms/${sw.agent}`} className="text-text hover:text-accent">
              {agentBy[sw.agent]?.label || sw.agent}
            </Link>{' '}
            on{' '}
            <Link to={`/envs/${sw.env}`} className="text-text hover:text-accent">
              {envBy[sw.env]?.label || sw.env}
            </Link>
            <span className="text-subtle">
              {' '}
              · {sw.strategy} search over {swept.join(', ')} · {sw.episodes} episodes per run
            </span>
          </p>
        </div>
        <div className="flex gap-2">
          {active && (
            <Button icon={Square} loading={stop.isPending} onClick={() => stop.mutate()}>
              Stop sweep
            </Button>
          )}
          <Button variant="danger" icon={Trash2} onClick={() => setConfirm(true)}>
            Delete
          </Button>
        </div>
      </header>

      <StatStrip>
        <Stat label="Configurations" value={sw.configs.length} hint={`${sw.request.seeds?.length || 1} seeds each`} />
        <Stat
          label="Runs finished"
          value={`${finished} / ${sw.n_runs}`}
          hint={active ? `${c.running} running, ${c.queued} queued` : c.failed ? `${c.failed} failed` : 'All done'}
        >
          <Progress
            value={sw.n_runs ? finished / sw.n_runs : 0}
            className="mt-2"
            tone={active ? 'accent' : 'success'}
          />
        </Stat>
        <Stat
          label={`Best ${scoreLabel.toLowerCase()}`}
          value={best ? meanStd(best.score) : '—'}
          tone={best ? 'accent' : undefined}
          hint={best ? paramText(best.params) : 'Waiting for results'}
        />
        <Stat
          label="Solved at"
          value={sw.threshold != null ? fmtNum(sw.threshold) : '—'}
          hint={sw.threshold != null ? 'Environment threshold' : 'No threshold defined'}
        />
      </StatStrip>

      {active && finished === 0 && (
        <Notice tone="info" title="Results appear as runs finish">
          Rankings use the runs that have reported so far and update every few seconds.
        </Notice>
      )}

      {best && (
        <Card className="flex flex-wrap items-center gap-x-6 gap-y-4">
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-accent/12 text-accent">
            <Trophy size={18} aria-hidden />
          </span>
          <div className="min-w-0 flex-1">
            <div className="label">Best configuration{active ? ' so far' : ''}</div>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {Object.entries(best.params).map(([k, v]) => (
                <code key={k} className="rounded bg-surface-3 px-2 py-0.5 text-body-s text-text">
                  {k} = {String(v)}
                </code>
              ))}
            </div>
            <p className="mt-1.5 text-caption text-subtle">
              {scoreLabel} {meanStd(best.score)} over {plural(best.score.n, 'seed')}
              {best.score.n < 2 && ' — one seed only, treat with caution'}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <ButtonLink
              to={`/compare?runs=${sw.configs
                .slice(0, 3)
                .flatMap((g) => g.run_ids)
                .join(',')}&group_by=config`}
              icon={GitCompareArrows}
            >
              Compare top 3
            </ButtonLink>
            <ButtonLink
              variant="primary"
              icon={Play}
              to={`/lab/${labId}/new?agent=${sw.agent}&env=${sw.env}&episodes=${sw.episodes}&params=${encodeURIComponent(JSON.stringify(best.all_params))}`}
            >
              Train this configuration
            </ButtonLink>
          </div>
        </Card>
      )}

      <ChartFrame
        title="Learning curves"
        description={`The top ${Math.min(shown, sw.configs.length)} configurations by ${scoreLabel.toLowerCase()}; bands are ±1σ across seeds.`}
        series={series}
        threshold={sw.threshold != null ? { value: sw.threshold } : undefined}
        height={300}
        filename={`sweep-${sw.id}`}
        defaultSmoothing={0.9}
        actions={
          sw.configs.length > 3 && (
            <Segmented
              size="sm"
              label="Configurations shown"
              value={shown}
              onChange={setShown}
              options={[
                { value: 3, label: 'Top 3' },
                { value: 6, label: 'Top 6' },
              ]}
            />
          )
        }
        emptyTitle="No episodes yet"
        emptyDescription="Curves appear as soon as the first runs report progress."
      />

      {swept.length > 0 && (
        <Section
          title="What each parameter does"
          description={`${scoreLabel} by value, averaged over the other parameters and seeds.`}
        >
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {swept.map((k) => (
              <ParamEffect key={k} name={k} effect={sw.effects[k]} />
            ))}
          </div>
        </Section>
      )}

      <Section
        title="Ranking"
        actions={
          <Segmented
            size="sm"
            label="Rank by"
            value={rankBy}
            onChange={setRankBy}
            options={[
              { value: 'final', label: 'Final return' },
              { value: 'best_eval', label: 'Best eval' },
            ]}
          />
        }
      >
        <Table>
          <THead>
            <Th align="right">Rank</Th>
            {swept.map((k) => (
              <Th key={k}>{k}</Th>
            ))}
            <Th align="right">{scoreLabel}</Th>
            <Th align="right">{rankBy === 'final' ? 'Best eval' : 'Final return'}</Th>
            <Th>Seeds done</Th>
            <Th align="right"> </Th>
          </THead>
          <tbody>
            {sw.configs.map((g, i) => (
              <Tr key={g.group_id}>
                <Td align="right" mono className="text-subtle">
                  <span className="inline-flex items-center gap-2">
                    {i < 6 && (
                      <span
                        className="h-0.5 w-3 rounded"
                        style={{ background: colorVar(SERIES_COLORS[i % SERIES_COLORS.length]) }}
                      />
                    )}
                    {g.rank}
                  </span>
                </Td>
                {swept.map((k) => (
                  <Td key={k} mono className="text-text">
                    {String(g.params[k])}
                  </Td>
                ))}
                <Td align="right" mono className={g.rank === 1 && g.score.mean !== null ? 'text-accent' : 'text-text'}>
                  {meanStd(g.score)}
                </Td>
                <Td align="right" mono className="text-muted">
                  {meanStd(rankBy === 'final' ? g.summary.best_eval : g.summary.final)}
                </Td>
                <Td>
                  <div className="flex min-w-24 items-center gap-2">
                    <Progress
                      value={g.n_runs ? g.done / g.n_runs : 0}
                      className="w-12"
                      tone={g.done === g.n_runs ? 'success' : 'accent'}
                    />
                    <span className="num text-caption text-muted">
                      {g.done}/{g.n_runs}
                    </span>
                  </div>
                </Td>
                <Td align="right">
                  <Link
                    to={`/compare?runs=${g.run_ids.join(',')}`}
                    className="text-caption text-accent hover:underline"
                  >
                    Seeds
                  </Link>
                </Td>
              </Tr>
            ))}
          </tbody>
        </Table>
      </Section>

      <Modal
        open={confirm}
        onClose={() => setConfirm(false)}
        title="Delete this sweep?"
        description={`Its ${plural(sw.n_runs, 'run')} and their artifacts are removed. Kept models are not affected.`}
        size="sm"
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirm(false)}>
              Cancel
            </Button>
            <Button variant="danger" icon={Trash2} loading={del.isPending} onClick={() => del.mutate()}>
              Delete sweep
            </Button>
          </>
        }
      >
        <p className="text-body-s text-muted">
          {active ? 'Running runs are stopped first.' : 'This cannot be undone.'}
        </p>
      </Modal>
    </div>
  )
}
