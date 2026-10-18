import { useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Layers, Play } from 'lucide-react'
import { useRouteLab } from '../app/AppContext.jsx'
import { api } from '../lib/api.js'
import { useCatalogue, useRunMutation } from '../lib/queries.js'
import { fmtCompact, metricLabel } from '../lib/format.js'
import { cn } from '../lib/utils.js'
import { Button } from '../components/ui/Button.jsx'
import { Card } from '../components/ui/Card.jsx'
import { Field, Input, Switch } from '../components/ui/Field.jsx'
import { Chip, Segmented } from '../components/ui/Segmented.jsx'
import { ErrorState, Notice, Skeleton } from '../components/ui/Feedback.jsx'
import { PageHeader } from '../components/ui/Page.jsx'
import { useToast } from '../components/ui/Toast.jsx'
import { CompatVerdict } from '../components/domain/CompatVerdict.jsx'
import { AllLabsToggle, ExperimentPicker, StepHeader } from '../components/domain/ExperimentPicker.jsx'
import { ParamForm, defaultsOf, paramsValid } from '../components/domain/ParamForm.jsx'

const SEED_OPTIONS = [
  { value: 1, label: '1' },
  { value: 3, label: '3' },
  { value: 5, label: '5' },
]

export default function Composer() {
  const { labId, lab } = useRouteLab()
  const [search, setSearch] = useSearchParams()
  const navigate = useNavigate()
  const { toast } = useToast()
  const cat = useCatalogue()

  const [envName, setEnvName] = useState(search.get('env') || '')
  const [agentName, setAgentName] = useState(search.get('agent') || '')
  const [allLabs, setAllLabs] = useState(false)
  const [params, setParams] = useState({})
  const [episodes, setEpisodes] = useState(Number(search.get('episodes')) || 300)
  const [seeds, setSeeds] = useState(1)
  const [baseSeed, setBaseSeed] = useState(42)
  const [name, setName] = useState('')
  const [tags, setTags] = useState('')
  const [evalFreq, setEvalFreq] = useState(25)
  const [earlyStop, setEarlyStop] = useState(false)
  const [preset, setPreset] = useState('custom')

  const env = cat.envBy[envName]
  const agent = cat.agentBy[agentName]

  // Hyperparameters passed in the URL (e.g. "train the best sweep configuration")
  // apply once, to the algorithm they were given for.
  const [initial] = useState(() => {
    try {
      return { agent: search.get('agent'), params: JSON.parse(search.get('params') || '{}') }
    } catch {
      return { agent: null, params: {} }
    }
  })

  // reset params when the algorithm changes
  useEffect(() => {
    if (agent) {
      const known = new Set(agent.params.map((p) => p.name))
      const fromUrl =
        agentName === initial.agent
          ? Object.fromEntries(Object.entries(initial.params).filter(([k]) => known.has(k)))
          : {}
      setParams({ ...defaultsOf(agent.params), ...fromUrl })
    }
    setPreset('custom')
  }, [agentName, agent, initial])

  // keep the URL in sync so a composed experiment is shareable
  useEffect(() => {
    const next = new URLSearchParams()
    if (envName) next.set('env', envName)
    if (agentName) next.set('agent', agentName)
    if (episodes) next.set('episodes', String(episodes))
    setSearch(next, { replace: true })
  }, [envName, agentName, episodes, setSearch])

  const verdictQ = useQuery({
    queryKey: ['compat', agentName, envName],
    queryFn: () => api.compatibility(agentName, envName),
    enabled: !!agentName && !!envName,
  })

  const create = useRunMutation(api.createRun, {
    onSuccess: (res) => {
      if (res.run_ids?.length > 1) {
        toast({
          title: `Launched ${res.run_ids.length} seeds`,
          description: 'They run under one group so you can see mean ± std.',
        })
        navigate(`/compare?runs=${res.run_ids.join(',')}&group_by=config`)
      } else {
        toast({ title: 'Run launched', description: `${agent?.label} on ${env?.label}` })
        navigate(`/runs/${res.run_id}`)
      }
    },
  })

  const verdict = verdictQ.data
  const chosen = !!env && !!agent
  const valid = chosen && paramsValid(agent.params, params) && episodes >= 1
  const canLaunch = valid && verdict?.compatible
  const typical = env?.typical_episode_length || env?.max_steps || 0
  const totalSteps = typical * episodes * seeds

  const applyPreset = (p) => {
    setPreset(p.name)
    setParams({ ...defaultsOf(agent.params), ...(p.overrides || {}) })
    if (p.episodes) setEpisodes(p.episodes)
  }

  const launch = () => {
    const seedList = Array.from({ length: seeds }, (_, i) => Number(baseSeed) + i)
    create.mutate({
      agent: agentName,
      env: envName,
      name: name.trim() || undefined,
      tags: tags
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean),
      params,
      episodes: Number(episodes),
      seed: seedList[0],
      seeds: seeds > 1 ? seedList : undefined,
      eval_freq: Number(evalFreq) || 25,
      early_stopping_patience: earlyStop ? 8 : null,
      preset,
      mode: 'advanced',
    })
  }

  if (cat.error) return <ErrorState error={cat.error} onRetry={cat.refetch} title="Could not load the catalogue" />

  return (
    <div>
      <PageHeader
        title="New run"
        description={`Choose an environment and an algorithm, adjust the settings, then launch. Pairings are checked before anything starts.`}
      />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-6">
          {/* ------------------------------------------------ step 1 */}
          <Card>
            <StepHeader n={1} title="Choose the experiment" done={chosen}>
              <AllLabsToggle value={allLabs} onChange={setAllLabs} />
            </StepHeader>
            <ExperimentPicker
              labId={labId}
              lab={lab}
              envName={envName}
              agentName={agentName}
              onEnv={setEnvName}
              onAgent={setAgentName}
              allLabs={allLabs}
            />
          </Card>

          {/* ------------------------------------------------ step 2 */}
          <Card className={cn(!chosen && 'opacity-60')} aria-disabled={!chosen}>
            <StepHeader n={2} title="Configure" done={chosen && valid} />
            {!chosen ? (
              <p className="text-body-s text-muted">
                Settings appear once you have chosen both an environment and an algorithm.
              </p>
            ) : (
              <div className="space-y-6">
                {agent.presets?.length > 0 && (
                  <div>
                    <div className="label mb-2">Start from a preset</div>
                    <div className="flex flex-wrap gap-2">
                      {agent.presets.map((p) => (
                        <Chip key={p.name} active={preset === p.name} onClick={() => applyPreset(p)}>
                          <span title={p.description}>{p.label}</span>
                        </Chip>
                      ))}
                    </div>
                  </div>
                )}

                <div className="grid gap-x-5 gap-y-4 sm:grid-cols-2 lg:grid-cols-4">
                  <Field label="Episodes" help={`About ${typical.toLocaleString()} steps each`}>
                    {(id) => (
                      <Input
                        id={id}
                        type="number"
                        min={1}
                        className="num"
                        value={episodes}
                        onChange={(e) => setEpisodes(e.target.value === '' ? '' : Number(e.target.value))}
                      />
                    )}
                  </Field>
                  <Field label="Seeds" help="3+ give a mean ± std">
                    <Segmented
                      value={seeds}
                      onChange={setSeeds}
                      options={SEED_OPTIONS}
                      label="Number of seeds"
                      size="sm"
                    />
                  </Field>
                  <Field
                    label="First seed"
                    help={seeds > 1 ? `Seeds ${baseSeed}–${Number(baseSeed) + seeds - 1}` : 'For reproducibility'}
                  >
                    {(id) => (
                      <Input
                        id={id}
                        type="number"
                        min={0}
                        className="num"
                        value={baseSeed}
                        onChange={(e) => setBaseSeed(e.target.value)}
                      />
                    )}
                  </Field>
                  <Field label="Evaluate every" help="Episodes between greedy evaluations">
                    {(id) => (
                      <Input
                        id={id}
                        type="number"
                        min={1}
                        className="num"
                        value={evalFreq}
                        onChange={(e) => setEvalFreq(e.target.value)}
                      />
                    )}
                  </Field>
                </div>

                <div className="grid gap-x-5 gap-y-4 sm:grid-cols-2">
                  <Field label="Name" hint="Optional">
                    {(id) => (
                      <Input
                        id={id}
                        value={name}
                        placeholder={`${agentName}-${envName}`}
                        onChange={(e) => setName(e.target.value)}
                      />
                    )}
                  </Field>
                  <Field label="Tags" hint="Comma separated">
                    {(id) => (
                      <Input
                        id={id}
                        value={tags}
                        placeholder="baseline, sweep-1"
                        onChange={(e) => setTags(e.target.value)}
                      />
                    )}
                  </Field>
                </div>
                <label className="flex items-center gap-3 text-body-s text-muted">
                  <Switch checked={earlyStop} onChange={setEarlyStop} label="Stop early when evaluation plateaus" />
                  Stop early if evaluation has not improved for 8 checks
                </label>

                <div>
                  <h3 className="mb-3 text-body-s font-semibold">Hyperparameters</h3>
                  <ParamForm
                    params={agent.params}
                    values={params}
                    onChange={(v) => {
                      setParams(v)
                      setPreset('custom')
                    }}
                  />
                </div>
              </div>
            )}
          </Card>
        </div>

        {/* ------------------------------------------------ step 3: summary rail */}
        <aside className="xl:sticky xl:top-20 xl:self-start">
          <Card raised>
            <StepHeader n={3} title="Review and launch" done={false} />
            {!env && !agent && (
              <div className="space-y-3 text-body-s text-muted">
                <p>
                  Pick from the two lists on the left. The rail fills in as you go: the compatibility check, the
                  workload and what to watch for.
                </p>
                {lab?.curriculum?.[0] && (
                  <Notice tone="info" title="Not sure where to start?">
                    <Link
                      className="text-accent hover:underline"
                      to={`?agent=${lab.curriculum[0].agent}&env=${lab.curriculum[0].env}&episodes=${lab.curriculum[0].episodes}`}
                      onClick={() => {
                        setAgentName(lab.curriculum[0].agent)
                        setEnvName(lab.curriculum[0].env)
                        setEpisodes(lab.curriculum[0].episodes)
                      }}
                    >
                      {lab.curriculum[0].title}
                    </Link>{' '}
                    is the first step of this lab’s curriculum.
                  </Notice>
                )}
              </div>
            )}

            {(env || agent) && (
              <div className="space-y-5">
                <dl className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-3 gap-y-2 text-body-s">
                  <dt className="text-subtle">Environment</dt>
                  <dd className="truncate font-medium text-text">
                    {env ? (
                      <Link to={`/envs/${env.name}`} className="hover:text-accent">
                        {env.label}
                      </Link>
                    ) : (
                      '—'
                    )}
                  </dd>
                  <dt className="text-subtle">Algorithm</dt>
                  <dd className="truncate font-medium text-text">
                    {agent ? (
                      <Link to={`/algorithms/${agent.name}`} className="hover:text-accent">
                        {agent.label}
                      </Link>
                    ) : (
                      '—'
                    )}
                  </dd>
                  {chosen && (
                    <>
                      <dt className="text-subtle">Workload</dt>
                      <dd className="text-text">
                        <span className="num">{Number(episodes).toLocaleString()}</span> episodes
                        {seeds > 1 && (
                          <>
                            {' '}
                            × <span className="num">{seeds}</span> seeds
                          </>
                        )}
                        <span className="block text-caption text-subtle">
                          ≈ {fmtCompact(totalSteps)} environment steps
                        </span>
                      </dd>
                    </>
                  )}
                </dl>

                {chosen && verdictQ.isLoading && <Skeleton className="h-16" />}
                {verdict && <CompatVerdict verdict={verdict} />}

                {chosen && agent.docs?.expected_metrics?.length > 0 && (
                  <div>
                    <div className="label mb-1.5">You will see</div>
                    <ul className="space-y-1 text-body-s">
                      <li className="text-muted">
                        <span className="text-text">Episode return</span> and its 100-episode mean
                      </li>
                      {agent.docs.expected_metrics.slice(0, 4).map((m) => (
                        <li key={m.key} className="text-muted">
                          <span className="text-text">{metricLabel(m.key)}</span>: {m.meaning}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {env?.beginner_exercise && (
                  <div>
                    <div className="label mb-1.5">Suggested exercise for this environment</div>
                    <p className="text-body-s text-muted">{env.beginner_exercise}</p>
                  </div>
                )}

                {create.error && (
                  <Notice tone="danger" title="Launch failed">
                    {create.error.message}
                  </Notice>
                )}

                <Button
                  variant="primary"
                  size="lg"
                  className="w-full"
                  icon={seeds > 1 ? Layers : Play}
                  disabled={!canLaunch}
                  loading={create.isPending}
                  onClick={launch}
                >
                  {seeds > 1 ? `Launch ${seeds} seeds` : 'Launch run'}
                </Button>
                {!chosen && <p className="text-center text-caption text-subtle">Choose both to continue.</p>}
                {chosen && verdict && !verdict.compatible && (
                  <p className="text-center text-caption text-subtle">Resolve the blocker above to launch.</p>
                )}
              </div>
            )}
          </Card>
        </aside>
      </div>
    </div>
  )
}
