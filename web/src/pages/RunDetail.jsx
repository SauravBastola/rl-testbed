import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Archive,
  Copy,
  Download,
  FileCode2,
  FileText,
  GitCompareArrows,
  Pencil,
  Play,
  RotateCw,
  Square,
  Trash2,
} from 'lucide-react'
import { useAdoptLab, useLab } from '../app/AppContext.jsx'
import { api } from '../lib/api.js'
import { useCatalogue, useLiveRun, useRunMutation } from '../lib/queries.js'
import { fmtBytes, fmtDate, fmtDuration, fmtNum, isActive, metricLabel } from '../lib/format.js'
import { availableMetrics, seriesFromHistory } from '../lib/curves.js'
import { Button, IconButton, buttonClass } from '../components/ui/Button.jsx'
import { Card, CardHeader } from '../components/ui/Card.jsx'
import { EmptyState, ErrorState, Notice, Skeleton, SkeletonRows } from '../components/ui/Feedback.jsx'
import { Input, Textarea, Field } from '../components/ui/Field.jsx'
import { KeyValue, Progress, Stat, StatStrip } from '../components/ui/Page.jsx'
import { Modal } from '../components/ui/Overlay.jsx'
import { Tabs } from '../components/ui/Segmented.jsx'
import { Table, THead, Th, Tr, Td } from '../components/ui/Table.jsx'
import { Badge, StatusBadge } from '../components/ui/Badge.jsx'
import { useToast } from '../components/ui/Toast.jsx'
import { ChartFrame, SMOOTHING } from '../components/charts/ChartFrame.jsx'
import { Segmented } from '../components/ui/Segmented.jsx'
import { CompatVerdict } from '../components/domain/CompatVerdict.jsx'
import { SampleList } from '../components/domain/SampleList.jsx'
import NotFound from './NotFound.jsx'

function MetricChart({ history, metric, onMetric, metrics, smoothing, compact }) {
  return (
    <ChartFrame
      compact={compact}
      title={metricLabel(metric)}
      series={[seriesFromHistory(history, metric, { color: 'chart-2' })]}
      metric={metric}
      metrics={metrics}
      onMetricChange={onMetric}
      smoothing={smoothing}
      height={180}
      legend={false}
      filename={metric}
      emptyTitle="Not reported yet"
      emptyDescription="Most algorithms only report this once they start updating."
    />
  )
}

function LogPanel({ runId, live }) {
  const logs = useQuery({
    queryKey: ['logs', runId],
    queryFn: () => api.logs(runId),
    refetchInterval: live ? 2000 : false,
  })
  if (logs.isLoading) return <SkeletonRows rows={6} />
  if (logs.error) return <ErrorState error={logs.error} onRetry={logs.refetch} />
  if (!logs.data.exists)
    return (
      <EmptyState compact icon={FileText} title="No log yet" description="The worker writes its log once it starts." />
    )
  return (
    <pre className="max-h-[480px] overflow-auto rounded-lg border border-border bg-bg-soft p-4 text-caption leading-5 text-muted">
      {logs.data.lines.map((l, i) => (
        <div
          key={i}
          className={/ERROR|failed|Traceback/i.test(l) ? 'text-danger' : /WARNING/.test(l) ? 'text-warning' : undefined}
        >
          {l}
        </div>
      ))}
    </pre>
  )
}

/** Latest evaluation answers of a language run (written by SequenceTrainer). */
function SamplesPanel({ runId, live }) {
  const q = useQuery({
    queryKey: ['samples', runId],
    queryFn: async () => {
      const res = await fetch(api.artifactUrl(runId, 'samples.json'))
      return res.ok ? res.json() : []
    },
    refetchInterval: live ? 4000 : false,
  })
  if (q.isLoading) return <SkeletonRows rows={4} />
  if (!q.data?.length)
    return (
      <EmptyState
        compact
        icon={FileText}
        title="No samples yet"
        description="They appear after the first evaluation."
      />
    )
  return (
    <div className="space-y-2">
      <p className="text-body-s text-muted">
        Answers to held-out prompts at the latest evaluation, sampled at temperature 1.
      </p>
      <SampleList samples={q.data} showProbs={false} />
    </div>
  )
}

function ArtifactsPanel({ runId }) {
  const q = useQuery({ queryKey: ['artifacts', runId], queryFn: () => api.artifacts(runId) })
  if (q.isLoading) return <SkeletonRows rows={4} />
  if (q.error) return <ErrorState error={q.error} onRetry={q.refetch} />
  if (!q.data.files.length) return <EmptyState compact icon={Archive} title="No artifacts yet" />
  const describe = (p) =>
    ({
      'best.pth': 'Checkpoint with the best evaluation score',
      'checkpoint.pth': 'Latest checkpoint',
      'history.jsonl': 'One JSON object per episode',
      'progress.json': 'Live progress snapshot',
      'worker.log': 'Worker log',
    })[p] || (p.startsWith('tb/') ? 'TensorBoard event file' : '')
  return (
    <Table>
      <THead>
        <Th>File</Th>
        <Th>Contents</Th>
        <Th align="right">Size</Th>
        <Th align="right"> </Th>
      </THead>
      <tbody>
        {q.data.files.map((f) => (
          <Tr key={f.path}>
            <Td mono className="text-text">
              {f.path}
            </Td>
            <Td className="text-muted">{describe(f.path)}</Td>
            <Td align="right" mono className="text-muted">
              {fmtBytes(f.size)}
            </Td>
            <Td align="right">
              <a
                href={api.artifactUrl(runId, f.path)}
                download
                aria-label={`Download ${f.path}`}
                className="inline-flex h-8 w-8 items-center justify-center rounded text-muted hover:bg-surface-2 hover:text-text"
              >
                <Download size={14} aria-hidden />
              </a>
            </Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}

function EditModal({ run, open, onClose }) {
  const { toast } = useToast()
  const [name, setName] = useState(run.name)
  const [notes, setNotes] = useState(run.notes || '')
  const [tags, setTags] = useState((run.tags || []).join(', '))
  const save = useRunMutation((body) => api.updateRun(run.id, body), {
    onSuccess: () => {
      toast({ title: 'Saved' })
      onClose()
    },
  })
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Edit run"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            loading={save.isPending}
            onClick={() =>
              save.mutate({
                name,
                notes,
                tags: tags
                  .split(',')
                  .map((t) => t.trim())
                  .filter(Boolean),
              })
            }
          >
            Save
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <Field label="Name">{(id) => <Input id={id} value={name} onChange={(e) => setName(e.target.value)} />}</Field>
        <Field label="Tags" hint="Comma separated">
          {(id) => <Input id={id} value={tags} onChange={(e) => setTags(e.target.value)} />}
        </Field>
        <Field label="Notes" help="What were you testing? What happened?">
          {(id) => <Textarea id={id} rows={5} value={notes} onChange={(e) => setNotes(e.target.value)} />}
        </Field>
        {save.error && <Notice tone="danger">{save.error.message}</Notice>}
      </div>
    </Modal>
  )
}

export default function RunDetail() {
  const { runId } = useParams()
  const navigate = useNavigate()
  const { toast } = useToast()
  const { envBy, agentBy } = useCatalogue()
  const { lab } = useLab()
  const { run, history, live } = useLiveRun(runId)
  const r = run.data
  useAdoptLab(r?.lab)
  const [tab, setTab] = useState('episodes')
  const [smoothing, setSmoothing] = useState(0.6)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [editing, setEditing] = useState(false)
  const [picked, setPicked] = useState({})

  const hist = useMemo(() => history.data || [], [history.data])
  const metrics = useMemo(() => availableMetrics(hist, lab?.metrics || []), [hist, lab])
  const secondary = metrics.filter((m) => !['reward', 'mean_reward', 'eval'].includes(m.value))
  const tableExtra = secondary.filter((m) => m.value !== 'steps').slice(0, 3)
  const slots = [0, 1, 2, 3].map((i) => picked[i] || secondary[i]?.value).filter(Boolean)

  const stop = useRunMutation(() => api.stopRun(runId), {
    onSuccess: () =>
      toast({
        title: 'Stopping',
        description: 'The current episode finishes and a checkpoint is written.',
        tone: 'info',
      }),
  })
  const restart = useRunMutation(() => api.startRun(runId), { onSuccess: () => toast({ title: 'Run restarted' }) })
  const clone = useRunMutation(() => api.cloneRun(runId), {
    onSuccess: (res) => {
      toast({ title: 'Cloned', description: 'Same configuration, new run.' })
      navigate(`/runs/${res.run_id}`)
    },
  })
  const keep = useRunMutation((source) => api.keepModel(runId, { source }), {
    onSuccess: (m) => toast({ title: 'Model kept', description: m.name }),
    onError: (e) => toast({ title: 'Could not keep model', description: e.message, tone: 'danger' }),
  })
  const del = useRunMutation(() => api.deleteRun(runId), {
    onSuccess: () => {
      toast({ title: 'Run deleted' })
      navigate(`/lab/${r?.lab && r.lab !== 'diagnostics' ? r.lab : 'foundations'}/runs`)
    },
  })

  if (run.error?.status === 404) return <NotFound what={`a run with id “${runId}”`} />
  if (run.error) return <ErrorState error={run.error} onRetry={run.refetch} title="Could not load this run" />
  if (run.isLoading || !r)
    return (
      <div className="space-y-6">
        <Skeleton className="h-16" />
        <Skeleton className="h-20" />
        <Skeleton className="h-80" />
      </div>
    )

  const env = envBy[r.env]
  const agent = agentBy[r.agent]
  const p = r.progress || {}
  const total = p.total_episodes || r.episodes
  const done = live ? p.episode || 0 : r.episodes_done || hist.length
  const meanReturn = live ? p.running_reward : (r.final_mean_reward ?? p.running_reward)
  const threshold = env?.reward_threshold
  const rewardSeries = [
    seriesFromHistory(hist, 'reward', { label: 'Training return', color: 'chart-1' }),
    { ...seriesFromHistory(hist, 'eval', { label: 'Greedy evaluation', color: 'chart-3' }), dashed: true },
  ].filter((s) => s.points.length)

  return (
    <div className="space-y-6">
      {/* header */}
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="truncate text-display font-semibold">{r.name}</h1>
            <StatusBadge status={r.status} />
            {r.group_id && (
              <Link to={`/compare?group=${r.group_id}`} className="text-caption text-accent hover:underline">
                Seed group
              </Link>
            )}
          </div>
          <p className="mt-1.5 text-body text-muted">
            <Link to={`/algorithms/${r.agent}`} className="text-text hover:text-accent">
              {agent?.label || r.agent}
            </Link>{' '}
            on{' '}
            <Link to={`/envs/${r.env}`} className="text-text hover:text-accent">
              {env?.label || r.env}
            </Link>
            <span className="text-subtle">
              {' '}
              · seed <span className="num">{r.seed}</span> · <span className="num">{r.id}</span>
            </span>
          </p>
          {(r.tags || []).length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {r.tags.map((t) => (
                <Badge key={t}>{t}</Badge>
              ))}
            </div>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {isActive(r.status) && r.status !== 'stopping' && (
            <Button icon={Square} loading={stop.isPending} onClick={() => stop.mutate()}>
              Stop
            </Button>
          )}
          {['stopped', 'failed', 'interrupted', 'created'].includes(r.status) && (
            <Button icon={RotateCw} loading={restart.isPending} onClick={() => restart.mutate()}>
              Restart
            </Button>
          )}
          {!isActive(r.status) && (
            <Button icon={Archive} loading={keep.isPending} onClick={() => keep.mutate('best')} disabled={r.model_kept}>
              {r.model_kept ? 'Model kept' : 'Keep model'}
            </Button>
          )}
          <Button icon={GitCompareArrows} onClick={() => navigate(`/compare?runs=${r.id}`)}>
            Compare
          </Button>
          <a
            href={api.exportUrl({ runIds: r.id })}
            download
            aria-label="Export as an experiment file"
            title="Export as an experiment file"
            className={buttonClass({ variant: 'secondary', className: 'w-9 px-0' })}
          >
            <FileCode2 size={16} aria-hidden />
          </a>
          <IconButton
            icon={Copy}
            label="Clone with the same settings"
            variant="secondary"
            onClick={() => clone.mutate()}
          />
          <IconButton
            icon={Pencil}
            label="Edit name, tags and notes"
            variant="secondary"
            onClick={() => setEditing(true)}
          />
          <IconButton icon={Trash2} label="Delete run" variant="danger" onClick={() => setConfirmDelete(true)} />
        </div>
      </header>

      {p.error && (
        <Notice tone="danger" title="The run failed">
          <code className="break-words">{p.error}</code>
          <button type="button" className="ml-2 text-accent hover:underline" onClick={() => setTab('log')}>
            Read the log
          </button>
        </Notice>
      )}
      {r.status === 'queued' && (
        <Notice tone="info" title="Waiting for a free slot">
          Other runs are using every training slot. This one starts automatically when one finishes.
        </Notice>
      )}

      <StatStrip>
        <Stat label="Episodes" value={`${(done || 0).toLocaleString()} / ${(total || 0).toLocaleString()}`}>
          <Progress
            value={total ? done / total : 0}
            className="mt-2"
            tone={r.status === 'completed' ? 'success' : 'accent'}
          />
        </Stat>
        <Stat
          label="Mean return, last 100"
          value={fmtNum(meanReturn)}
          hint={threshold != null ? `Solved at ${fmtNum(threshold)}` : undefined}
          tone={threshold != null && meanReturn >= threshold ? 'success' : undefined}
        />
        <Stat
          label="Best evaluation"
          value={fmtNum(r.best_eval_reward ?? p.best_eval_reward)}
          hint={`Every ${r.config?.eval_freq || 25} episodes`}
        />
        <Stat
          label={live ? 'Speed' : 'Duration'}
          value={
            live
              ? `${fmtNum(p.episodes_per_sec)} ep/s`
              : fmtDuration(r.finished_at && r.started_at ? r.finished_at - r.started_at : p.elapsed_sec)
          }
          hint={
            p.total_env_steps
              ? `${p.total_env_steps.toLocaleString()} env steps${live ? ` · ${fmtNum(p.fps)} steps/s` : ''}`
              : undefined
          }
        />
      </StatStrip>

      <ChartFrame
        title="Learning curve"
        description="Return per training episode, with greedy evaluations dashed."
        series={rewardSeries}
        loading={history.isLoading}
        threshold={threshold != null ? { value: threshold } : undefined}
        height={300}
        smoothing={smoothing}
        filename={`run-${r.id}-return`}
        actions={
          <Segmented size="sm" label="Smoothing" value={smoothing} onChange={setSmoothing} options={SMOOTHING} />
        }
        emptyTitle={live ? 'Waiting for the first episode' : 'No episodes recorded'}
        emptyDescription={
          live
            ? 'The curve draws itself as episodes finish.'
            : 'The run ended before completing an episode. Check the log.'
        }
      />

      {slots.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-2">
          {slots.map((m, i) => (
            <MetricChart
              key={i}
              compact
              history={hist}
              metric={m}
              metrics={metrics}
              smoothing={smoothing}
              onMetric={(v) => setPicked((s) => ({ ...s, [i]: v }))}
            />
          ))}
        </div>
      )}

      <div>
        <Tabs
          value={tab}
          onChange={setTab}
          tabs={[
            ...(env?.category === 'language' ? [{ value: 'samples', label: 'Samples' }] : []),
            { value: 'episodes', label: env?.category === 'language' ? 'Steps' : 'Episodes', count: hist.length },
            { value: 'config', label: 'Configuration' },
            { value: 'artifacts', label: 'Artifacts' },
            { value: 'log', label: 'Log' },
          ]}
        />
        <div className="pt-4">
          {tab === 'episodes' &&
            (hist.length ? (
              <Table maxHeight={420}>
                <THead>
                  <Th>Episode</Th>
                  <Th align="right">Return</Th>
                  <Th align="right">Mean 100</Th>
                  <Th align="right">Steps</Th>
                  <Th align="right">Eval</Th>
                  {tableExtra.map((m) => (
                    <Th key={m.value} align="right">
                      {m.label}
                    </Th>
                  ))}
                </THead>
                <tbody>
                  {hist
                    .slice(-50)
                    .reverse()
                    .map((h) => (
                      <Tr key={h.episode}>
                        <Td mono>{h.episode}</Td>
                        <Td mono align="right">
                          {fmtNum(h.reward)}
                        </Td>
                        <Td mono align="right" className="text-muted">
                          {fmtNum(h.mean_reward)}
                        </Td>
                        <Td mono align="right" className="text-muted">
                          {h.steps}
                        </Td>
                        <Td mono align="right" className="text-muted">
                          {h.eval ? fmtNum(h.eval.episode_mean) : ''}
                        </Td>
                        {tableExtra.map((m) => (
                          <Td key={m.value} mono align="right" className="text-muted">
                            {fmtNum(h[m.value])}
                          </Td>
                        ))}
                      </Tr>
                    ))}
                </tbody>
              </Table>
            ) : (
              <EmptyState compact icon={Play} title="No episodes yet" />
            ))}
          {tab === 'config' && (
            <div className="grid gap-4 lg:grid-cols-2">
              <Card>
                <CardHeader title="Hyperparameters" description="Resolved values the worker used." />
                <KeyValue
                  rows={Object.entries(r.params || {}).map(([k, v]) => [
                    k,
                    <span key={k} className="num">
                      {String(v)}
                    </span>,
                  ])}
                />
              </Card>
              <Card>
                <CardHeader
                  title="Run"
                  actions={
                    <Button size="sm" variant="ghost" icon={Pencil} onClick={() => setEditing(true)}>
                      Edit
                    </Button>
                  }
                />
                <KeyValue
                  rows={[
                    ['Episodes', r.episodes],
                    ['Seed', r.seed],
                    ['Device', r.device],
                    ['Evaluation', `${r.config?.eval_episodes || 10} episodes every ${r.config?.eval_freq || 25}`],
                    ['Preset', r.config?.preset],
                    ['Created', fmtDate(r.created_at)],
                    ['Started', fmtDate(r.started_at)],
                    ['Finished', fmtDate(r.finished_at)],
                    [
                      'Parent run',
                      r.parent_run_id && (
                        <Link to={`/runs/${r.parent_run_id}`} className="num text-accent">
                          {r.parent_run_id}
                        </Link>
                      ),
                    ],
                    ['Git commit', r.git_commit],
                    ['Notes', r.notes],
                  ]}
                />
              </Card>
              {r.config?.compatibility && (
                <div className="lg:col-span-2">
                  <CompatVerdict verdict={r.config.compatibility} />
                </div>
              )}
            </div>
          )}
          {tab === 'samples' && <SamplesPanel runId={r.id} live={live} />}
          {tab === 'artifacts' && (
            <div className="space-y-3">
              <a href={api.historyCsvUrl(r.id)} download className={buttonClass({ size: 'sm' })}>
                <Download size={14} aria-hidden /> Episode history as CSV
              </a>
              <ArtifactsPanel runId={r.id} />
            </div>
          )}
          {tab === 'log' && <LogPanel runId={r.id} live={live} />}
        </div>
      </div>

      <Modal
        open={confirmDelete}
        onClose={() => setConfirmDelete(false)}
        title="Delete this run?"
        description="Its history, logs and checkpoints are removed from disk. Kept models stay."
        size="sm"
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirmDelete(false)}>
              Cancel
            </Button>
            <Button variant="danger" icon={Trash2} loading={del.isPending} onClick={() => del.mutate()}>
              Delete run
            </Button>
          </>
        }
      >
        <p className="text-body-s text-muted">
          {isActive(r.status) ? 'The run is still training; it will be stopped first.' : 'This cannot be undone.'}
        </p>
      </Modal>
      {editing && <EditModal run={r} open={editing} onClose={() => setEditing(false)} />}
    </div>
  )
}
