import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { Eye, Info, Pause, Play, SkipBack, SkipForward } from 'lucide-react'
import { useLab } from '../app/AppContext.jsx'
import { api } from '../lib/api.js'
import { useCatalogue, useModels } from '../lib/queries.js'
import { fmtNum } from '../lib/format.js'
import { Button, IconButton } from '../components/ui/Button.jsx'
import { Card, CardHeader } from '../components/ui/Card.jsx'
import { Field, Input, Select } from '../components/ui/Field.jsx'
import { EmptyState, ErrorState, Notice, Skeleton } from '../components/ui/Feedback.jsx'
import { PageHeader, Stat } from '../components/ui/Page.jsx'
import { Segmented } from '../components/ui/Segmented.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { LineChart } from '../components/charts/LineChart.jsx'
import { EnvSchematic } from '../components/domain/EnvSchematic.jsx'
import { spacesSummary } from '../components/domain/CatalogCards.jsx'

const SPEEDS = [
  { value: 2, label: '0.5×' },
  { value: 4, label: '1×' },
  { value: 12, label: '3×' },
]

const SOURCE_BADGE = {
  pixels: 'Rendered frames',
  frames: "The agent's own input",
  schematic: 'Drawing of the state',
  observation: 'Observation only',
}

/** The observation the agent received, labelled when the backend knows the labels. */
function Observation({ obs, labels, obsType }) {
  if (!Array.isArray(obs)) return null
  if (obsType === 'onehot' && obs.length > 1) {
    const idx = obs.findIndex((v) => v === 1)
    return (
      <p className="text-body-s text-muted">
        One-hot vector: state <span className="num text-text">{idx >= 0 ? idx : '—'}</span> of{' '}
        <span className="num text-text">{obs.length}</span>. The agent sees which state it is in, and nothing else.
      </p>
    )
  }
  const shown = obs.slice(0, 64)
  if (labels) {
    return (
      <dl className="grid max-h-80 grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 overflow-y-auto text-caption">
        {shown.map((v, k) => (
          <div key={k} className="contents">
            <dt className="truncate text-muted" title={labels[k]}>
              {labels[k]}
            </dt>
            <dd className="num text-right text-text">{fmtNum(Number(v), { digits: 3 })}</dd>
          </div>
        ))}
      </dl>
    )
  }
  return (
    <div className="grid max-h-80 grid-cols-4 gap-1 overflow-y-auto">
      {shown.map((v, k) => (
        <span
          key={k}
          className="num truncate rounded-sm bg-surface-2 px-1.5 py-1 text-right text-caption text-muted"
          title={`obs[${k}] = ${v}`}
        >
          {fmtNum(Number(v), { digits: 2 })}
        </span>
      ))}
    </div>
  )
}

/**
 * The action at this step, with a fixed footprint so playback never reflows:
 * a name for discrete actions; one row per dimension (bar + number) for
 * continuous ones.
 */
function ActionPanel({ step, data }) {
  if (step.final) return <p className="h-6 text-body-s text-muted">None: this is the final state.</p>
  const labels = data.action_labels
  if (data.action_type !== 'continuous') {
    const name = labels?.[step.action]
    return (
      <p className="flex h-6 min-w-0 items-baseline gap-2 text-body-s">
        <span className="truncate font-medium text-text">{name || `Action ${step.action}`}</span>
        {name && <span className="num shrink-0 text-caption text-subtle">#{step.action}</span>}
      </p>
    )
  }
  const values = Array.isArray(step.action) ? step.action : [step.action]
  const low = data.action_bounds?.low || values.map(() => -1)
  const high = data.action_bounds?.high || values.map(() => 1)
  return (
    <ul className="space-y-1.5">
      {values.map((v, k) => {
        const lo = low[k] ?? -1
        const hi = high[k] ?? 1
        const zero = Math.min(1, Math.max(0, (0 - lo) / (hi - lo || 1)))
        const pos = Math.min(1, Math.max(0, (v - lo) / (hi - lo || 1)))
        const left = Math.min(zero, pos)
        return (
          <li key={k} className="grid h-5 grid-cols-[minmax(0,7.5rem)_minmax(0,1fr)_3.5rem] items-center gap-2">
            <span className="truncate text-caption text-muted" title={labels?.[k]}>
              {labels?.[k] || `dim ${k + 1}`}
            </span>
            <span className="relative h-1.5 rounded-full bg-surface-3">
              <span className="absolute inset-y-[-2px] w-px bg-border-strong" style={{ left: `${zero * 100}%` }} />
              <span
                className="absolute inset-y-0 rounded-full bg-accent"
                style={{ left: `${left * 100}%`, width: `${Math.abs(pos - zero) * 100}%` }}
              />
            </span>
            <span className="num text-right text-caption text-text">{fmtNum(Number(v), { digits: 2 })}</span>
          </li>
        )
      })}
    </ul>
  )
}

function Player({ data, traj }) {
  const [i, setI] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [fps, setFps] = useState(4)
  const steps = traj.steps
  const step = steps[Math.min(i, steps.length - 1)]
  const source = data.render.source

  useEffect(() => {
    setI(0)
    setPlaying(true)
  }, [traj])
  useEffect(() => {
    if (!playing) return undefined
    const t = setInterval(() => setI((x) => (x + 1 >= steps.length ? (setPlaying(false), x) : x + 1)), 1000 / fps)
    return () => clearInterval(t)
  }, [playing, fps, steps.length])

  const rewardSeries = useMemo(
    () => [
      {
        id: 'cum',
        label: 'Return so far',
        color: 'chart-1',
        points: steps.map((s) => ({ x: s.step, y: s.total_reward })),
      },
      {
        id: 'r',
        label: 'Step reward',
        color: 'chart-3',
        points: steps.filter((s) => !s.final).map((s) => ({ x: s.step, y: s.reward })),
      },
    ],
    [steps],
  )
  if (!step) return null

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
      <Card>
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <Badge tone="accent">{SOURCE_BADGE[source]}</Badge>
          {step.final && (
            <Badge tone={step.terminated ? 'success' : 'neutral'}>
              {step.terminated ? 'Episode over' : 'Step limit reached'}
            </Badge>
          )}
        </div>
        <div className="grid min-h-80 place-items-center overflow-hidden rounded-lg border border-border bg-bg-soft p-2">
          {step.frame ? (
            <img
              src={`data:${data.render.mime || 'image/png'};base64,${step.frame}`}
              alt={`State at step ${step.step}`}
              className="max-h-[440px] w-auto max-w-full [image-rendering:pixelated]"
            />
          ) : step.view ? (
            <div className="w-full max-w-2xl p-4">
              <EnvSchematic view={step.view} />
            </div>
          ) : (
            <div className="max-w-md p-8 text-center">
              <p className="text-body-s font-medium text-text">Nothing to draw</p>
              <p className="mt-1 text-body-s text-muted">The observation on the right is the whole state.</p>
            </div>
          )}
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <IconButton icon={SkipBack} label="Restart" variant="secondary" onClick={() => setI(0)} />
          <IconButton
            icon={playing ? Pause : Play}
            label={playing ? 'Pause' : 'Play'}
            variant="secondary"
            onClick={() => setPlaying((p) => !p)}
          />
          <IconButton
            icon={SkipForward}
            label="Next step"
            variant="secondary"
            onClick={() => {
              setPlaying(false)
              setI((x) => Math.min(steps.length - 1, x + 1))
            }}
          />
          <input
            type="range"
            min={0}
            max={steps.length - 1}
            value={i}
            onChange={(e) => {
              setPlaying(false)
              setI(Number(e.target.value))
            }}
            aria-label="Step"
            className="min-w-32 flex-1 accent-[rgb(var(--accent))]"
          />
          <span className="num w-20 text-right text-caption text-muted">
            {step.step} / {traj.length}
          </span>
          <Segmented size="sm" label="Playback speed" value={fps} onChange={setFps} options={SPEEDS} />
        </div>
      </Card>

      <div className="space-y-4">
        <Card pad="sm">
          <div className="grid grid-cols-2 gap-4">
            <Stat label="Return so far" value={fmtNum(step.total_reward)} />
            <Stat label="Episode return" value={fmtNum(traj.total_reward)} />
          </div>
          <div className="mt-4 border-t border-border pt-3">
            <div className="mb-2 flex items-baseline justify-between">
              <span className="label">Action chosen</span>
              {!step.final && (
                <span className="text-caption text-subtle">
                  reward <span className="num text-text">{fmtNum(step.reward)}</span>
                </span>
              )}
            </div>
            <ActionPanel step={step} data={data} />
          </div>
        </Card>
        <Card pad="sm">
          <CardHeader
            className="mb-2"
            title="What the agent observes"
            description={`${step.obs.length} number${step.obs.length === 1 ? '' : 's'}${step.obs.length > 64 ? ', first 64 shown' : ''}`}
          />
          <Observation obs={step.obs} labels={data.obs_labels} obsType={data.obs_type} />
        </Card>
        <Card pad="sm">
          <LineChart series={rewardSeries} height={140} xLabel="Step" yLabel="Reward" />
        </Card>
      </div>
    </div>
  )
}

export default function Visualize() {
  const [params, setParams] = useSearchParams()
  const { labId } = useLab()
  const { envs, envBy, isLoading, error } = useCatalogue()
  const models = useModels()
  const [envName, setEnvName] = useState(params.get('env') || '')
  const [source, setSource] = useState(params.get('model') ? 'model' : 'random')
  const [modelId, setModelId] = useState(params.get('model') || '')
  const [seed, setSeed] = useState(0)
  const [maxSteps, setMaxSteps] = useState(300)
  const [episode, setEpisode] = useState(0)

  const usable = envs.filter((e) => e.available !== false)
  const defaultEnv = usable.find((e) => e.lab === labId)?.name || usable[0]?.name || ''
  const env = envBy[envName || defaultEnv]
  const model = (models.data || []).find((m) => m.id === modelId)
  const envModels = (models.data || []).filter((m) => m.env === env?.name)
  const slowRenderer = env?.category === 'mujoco'

  const roll = useMutation({ mutationFn: api.rollout })
  const go = () => {
    setEpisode(0)
    setParams({ env: env.name, ...(source === 'model' && modelId ? { model: modelId } : {}) }, { replace: true })
    roll.mutate({
      env: env.name,
      model_id: source === 'model' ? modelId : undefined,
      agent: source === 'model' ? undefined : 'random',
      seed: Number(seed),
      max_steps: Number(maxSteps),
      episodes: 1,
    })
  }

  if (error) return <ErrorState error={error} />
  const data = roll.data
  const trajs = data?.trajectories || []
  const modelNote =
    source === 'model'
      ? !envModels.length
        ? `No kept models for ${env?.label || 'this environment'} yet. Keep one from a finished run's page.`
        : !modelId
          ? 'Choose which kept model to watch.'
          : null
      : null

  return (
    <div className="space-y-6">
      <PageHeader
        title="Visualize"
        description="Play one episode and step through it, with a random policy or a kept model."
      />
      <Card>
        {/* Every control keeps a fixed footprint; notes go below the row, never inside it. */}
        <div className="grid items-end gap-4 md:grid-cols-2 xl:grid-cols-[minmax(0,2fr)_auto_minmax(0,1.6fr)_7rem_7rem_auto]">
          <Field label="Environment">
            {(id) =>
              isLoading ? (
                <Skeleton className="h-9" />
              ) : (
                <Select
                  id={id}
                  value={env?.name || ''}
                  onChange={(e) => {
                    setEnvName(e.target.value)
                    setModelId('')
                  }}
                >
                  {usable.map((e) => (
                    <option key={e.name} value={e.name}>
                      {e.label}
                    </option>
                  ))}
                </Select>
              )
            }
          </Field>
          <Field label="Policy">
            <Segmented
              value={source}
              onChange={setSource}
              label="Policy source"
              options={[
                { value: 'random', label: 'Random' },
                { value: 'model', label: 'Kept model' },
              ]}
            />
          </Field>
          <Field label="Model">
            {(id) => (
              <Select
                id={id}
                value={modelId}
                disabled={source !== 'model' || !envModels.length}
                onChange={(e) => setModelId(e.target.value)}
              >
                <option value="">
                  {source !== 'model' ? '—' : envModels.length ? 'Choose a model' : 'None kept yet'}
                </option>
                {envModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </Select>
            )}
          </Field>
          <Field label="Seed">
            {(id) => (
              <Input
                id={id}
                type="number"
                min={0}
                className="num"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
              />
            )}
          </Field>
          <Field label="Max steps">
            {(id) => (
              <Input
                id={id}
                type="number"
                min={1}
                max={2000}
                className="num"
                value={maxSteps}
                onChange={(e) => setMaxSteps(e.target.value)}
              />
            )}
          </Field>
          <Button
            variant="primary"
            icon={Play}
            loading={roll.isPending}
            disabled={!env || (source === 'model' && !modelId)}
            onClick={go}
          >
            Play episode
          </Button>
        </div>
        <p className="mt-3 min-h-[18px] text-caption text-subtle">
          {modelNote ? (
            <span className="text-warning">{modelNote}</span>
          ) : env ? (
            <>
              {spacesSummary(env)} · up to {env.max_steps.toLocaleString()} steps
              {model ? ` · ${model.agent} model` : ''}
            </>
          ) : null}
        </p>
      </Card>

      {roll.error && <ErrorState error={roll.error} onRetry={go} title="Could not play the episode" />}
      {roll.isPending && (
        <div className="space-y-3">
          {slowRenderer && (
            <Notice tone="info">Starting the MuJoCo renderer in its own process; this takes a few seconds.</Notice>
          )}
          <Skeleton className="h-96" />
        </div>
      )}
      {!roll.isPending && !trajs.length && !roll.error && (
        <EmptyState
          icon={Eye}
          title="Choose an environment and press Play episode"
          description="Environments with a renderer show real frames from the same instance the agent acts in. Worlds without one are drawn from their state, and every step lists exactly what the agent observes."
        />
      )}
      {!roll.isPending && trajs.length > 0 && (
        <>
          <Notice tone={data.render.error ? 'warning' : 'info'} icon={Info} title="What you're seeing">
            {data.render.explanation}
            {data.agent === 'random' && ` The policy is random: a baseline for what no learning looks like.`}
          </Notice>
          {trajs.length > 1 && (
            <Segmented
              value={episode}
              onChange={setEpisode}
              options={trajs.map((_, k) => ({ value: k, label: `Episode ${k + 1}` }))}
            />
          )}
          <Player data={data} traj={trajs[episode]} />
        </>
      )}
    </div>
  )
}
