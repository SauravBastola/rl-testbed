import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { Eye, Pause, Play, SkipBack, SkipForward } from 'lucide-react'
import { useLab } from '../app/AppContext.jsx'
import { api } from '../lib/api.js'
import { useCatalogue, useModels } from '../lib/queries.js'
import { fmtNum } from '../lib/format.js'
import { Button, IconButton } from '../components/ui/Button.jsx'
import { Card, CardHeader } from '../components/ui/Card.jsx'
import { Field, Input, Select } from '../components/ui/Field.jsx'
import { EmptyState, ErrorState, Notice, Skeleton } from '../components/ui/Feedback.jsx'
import { KeyValue, PageHeader, Stat } from '../components/ui/Page.jsx'
import { Segmented } from '../components/ui/Segmented.jsx'
import { LineChart } from '../components/charts/LineChart.jsx'
import { spacesSummary } from '../components/domain/CatalogCards.jsx'

const SPEEDS = [
  { value: 2, label: '0.5×' },
  { value: 4, label: '1×' },
  { value: 12, label: '3×' },
]

function fmtAction(a) {
  if (Array.isArray(a)) return `[${a.map((x) => fmtNum(x, { digits: 2 })).join(', ')}]`
  return String(a)
}

function Player({ traj }) {
  const [i, setI] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [fps, setFps] = useState(4)
  const steps = traj.steps
  const step = steps[Math.min(i, steps.length - 1)]

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
        label: 'Cumulative return',
        color: 'chart-1',
        points: steps.map((s) => ({ x: s.step + 1, y: s.total_reward })),
      },
      { id: 'r', label: 'Step reward', color: 'chart-3', points: steps.map((s) => ({ x: s.step + 1, y: s.reward })) },
    ],
    [steps],
  )
  if (!step) return null
  const obs = Array.isArray(step.obs) ? step.obs : step.obs?.obs

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
      <Card>
        <div
          className="grid place-items-center overflow-hidden rounded-lg border border-border bg-bg-soft"
          style={{ minHeight: 320 }}
        >
          {step.frame ? (
            <img
              src={`data:image/png;base64,${step.frame}`}
              alt={`Frame ${step.step + 1}`}
              className="max-h-[420px] w-auto max-w-full [image-rendering:pixelated]"
            />
          ) : (
            <p className="p-8 text-center text-body-s text-muted">
              This environment has no renderer; the observation is shown on the right.
            </p>
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
          <span className="num text-caption text-muted">
            {step.step + 1} / {steps.length}
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
          <KeyValue
            className="mt-4"
            rows={[
              [
                'Action',
                <span key="a" className="num">
                  {fmtAction(step.action)}
                </span>,
              ],
              [
                'Reward',
                <span key="r" className="num">
                  {fmtNum(step.reward)}
                </span>,
              ],
              ['Ended', step.terminated ? 'Terminated' : step.truncated ? 'Time limit' : null],
            ]}
          />
        </Card>
        {Array.isArray(obs) && (
          <Card pad="sm">
            <CardHeader
              title="Observation"
              description={`${obs.length} values${obs.length > 24 ? ', first 24 shown' : ''}`}
              className="mb-2"
            />
            <div className="grid grid-cols-4 gap-1">
              {obs.slice(0, 24).map((v, k) => (
                <span
                  key={k}
                  className="num truncate rounded-sm bg-surface-2 px-1.5 py-1 text-right text-caption text-muted"
                  title={String(v)}
                >
                  {fmtNum(Number(v), { digits: 2 })}
                </span>
              ))}
            </div>
          </Card>
        )}
        <Card pad="sm">
          <LineChart series={rewardSeries} height={150} xLabel="Step" yLabel="Reward" />
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
  const trajs = roll.data?.trajectories || []

  return (
    <div className="space-y-6">
      <PageHeader
        title="Visualize"
        description="Play one episode and step through it frame by frame, with a random policy or a kept model."
      />
      <Card>
        <div className="grid items-end gap-4 md:grid-cols-2 xl:grid-cols-[minmax(0,2fr)_minmax(0,1.2fr)_minmax(0,1.6fr)_7rem_7rem_auto]">
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
          <Field
            label="Model"
            help={source === 'model' && !envModels.length ? 'No kept models for this environment yet.' : undefined}
          >
            {(id) => (
              <Select
                id={id}
                value={modelId}
                disabled={source !== 'model' || !envModels.length}
                onChange={(e) => setModelId(e.target.value)}
              >
                <option value="">{envModels.length ? 'Choose a model' : 'None available'}</option>
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
        {env && (
          <p className="mt-3 text-caption text-subtle">
            {spacesSummary(env)} · up to {env.max_steps.toLocaleString()} steps{model ? ` · ${model.agent} model` : ''}
          </p>
        )}
      </Card>

      {roll.error && <ErrorState error={roll.error} onRetry={go} title="Could not play the episode" />}
      {roll.isPending && <Skeleton className="h-96" />}
      {!roll.isPending && !trajs.length && !roll.error && (
        <EmptyState
          icon={Eye}
          title="Choose an environment and press Play episode"
          description="Rendered environments show real frames. Those without a renderer show the observation vector, action and reward for every step, so nothing is unviewable."
        />
      )}
      {!roll.isPending && trajs.length > 0 && (
        <>
          {trajs.length > 1 && (
            <Segmented
              value={episode}
              onChange={setEpisode}
              options={trajs.map((_, k) => ({ value: k, label: `Episode ${k + 1}` }))}
            />
          )}
          {source === 'random' && (
            <Notice tone="info">
              This is a random policy: a baseline for what no learning looks like on {env?.label}.
            </Notice>
          )}
          <Player traj={trajs[episode]} />
        </>
      )}
    </div>
  )
}
