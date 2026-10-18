import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Check, CircleSlash, Layers, Play, Sparkles } from 'lucide-react'
import { useRouteLab } from '../app/AppContext.jsx'
import { api } from '../lib/api.js'
import { useCatalogue, useCompatMatrix, useRunMutation } from '../lib/queries.js'
import { fmtCompact, metricLabel } from '../lib/format.js'
import { cn } from '../lib/utils.js'
import { Button } from '../components/ui/Button.jsx'
import { Card } from '../components/ui/Card.jsx'
import { Field, Input, SearchInput, Switch } from '../components/ui/Field.jsx'
import { Chip, Segmented } from '../components/ui/Segmented.jsx'
import { ErrorState, Notice, Skeleton } from '../components/ui/Feedback.jsx'
import { PageHeader } from '../components/ui/Page.jsx'
import { Badge, DifficultyBadge } from '../components/ui/Badge.jsx'
import { useToast } from '../components/ui/Toast.jsx'
import { CompatDot, CompatVerdict } from '../components/domain/CompatVerdict.jsx'
import { ParamForm, defaultsOf, paramsValid } from '../components/domain/ParamForm.jsx'
import { familyLabel, spacesSummary } from '../components/domain/CatalogCards.jsx'

const SEED_OPTIONS = [
  { value: 1, label: '1' },
  { value: 3, label: '3' },
  { value: 5, label: '5' },
]

/** Keep the chosen item first so it never scrolls out of view. */
const pinFirst = (list, name) => {
  const i = list.findIndex((x) => x.name === name)
  return i > 0 ? [list[i], ...list.slice(0, i), ...list.slice(i + 1)] : list
}

function StepHeader({ n, title, done, children }) {
  return (
    <div className="mb-4 flex items-center gap-3">
      <span
        className={cn(
          'num grid h-7 w-7 shrink-0 place-items-center rounded-full text-body-s font-semibold',
          done ? 'bg-accent text-accent-contrast' : 'border border-border-strong text-muted',
        )}
        aria-hidden
      >
        {done ? <Check size={14} /> : n}
      </span>
      <h2 className="text-title-s font-semibold">{title}</h2>
      <div className="ml-auto flex items-center gap-2">{children}</div>
    </div>
  )
}

function PickerRow({ selected, onClick, title, subtitle, right, disabled, dot }) {
  return (
    <button
      type="button"
      role="option"
      aria-selected={selected}
      onClick={onClick}
      className={cn(
        'flex w-full items-center gap-3 rounded px-3 py-2 text-left transition-colors duration-fast',
        selected ? 'bg-accent/12 ring-1 ring-inset ring-accent/40' : 'hover:bg-surface-2',
        disabled && 'opacity-55',
      )}
    >
      {dot}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-body-s font-medium text-text">{title}</span>
        <span className="block truncate text-caption text-subtle">{subtitle}</span>
      </span>
      {right}
    </button>
  )
}

export default function Composer() {
  const { labId, lab } = useRouteLab()
  const [search, setSearch] = useSearchParams()
  const navigate = useNavigate()
  const { toast } = useToast()
  const cat = useCatalogue()
  const compat = useCompatMatrix()

  const [envName, setEnvName] = useState(search.get('env') || '')
  const [agentName, setAgentName] = useState(search.get('agent') || '')
  const [envQuery, setEnvQuery] = useState('')
  const [agentQuery, setAgentQuery] = useState('')
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

  // reset params when the algorithm changes
  useEffect(() => {
    if (agent) setParams(defaultsOf(agent.params))
    setPreset('custom')
  }, [agentName, agent])

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

  const envList = useMemo(() => {
    const tokens = envQuery.toLowerCase().split(/\s+/).filter(Boolean)
    return cat.envs
      .filter((e) => e.category !== 'probe' && (allLabs || e.lab === labId))
      .filter((e) =>
        tokens.every((t) => `${e.label} ${e.name} ${e.category} ${(e.tags || []).join(' ')}`.toLowerCase().includes(t)),
      )
      .sort((a, b) => (b.available !== false) - (a.available !== false))
  }, [cat.envs, envQuery, allLabs, labId])

  const agentList = useMemo(() => {
    const tokens = agentQuery.toLowerCase().split(/\s+/).filter(Boolean)
    const priority = lab?.algorithms || []
    const rank = (a) => {
      const level = envName ? compat.lookup(a.name, envName)?.level : null
      const levelRank = level === 'ok' ? 0 : level === 'warning' ? 1 : level === 'blocked' ? 3 : 2
      const p = priority.indexOf(a.name)
      return [a.status === 'implemented' ? 0 : 1, levelRank, p < 0 ? 99 : p]
    }
    return cat.agents
      .filter((a) => tokens.every((t) => `${a.label} ${a.name} ${a.category}`.toLowerCase().includes(t)))
      .sort((a, b) => {
        const ra = rank(a)
        const rb = rank(b)
        for (let i = 0; i < ra.length; i += 1) if (ra[i] !== rb[i]) return ra[i] - rb[i]
        return a.label.localeCompare(b.label)
      })
  }, [cat.agents, agentQuery, envName, compat, lab])

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
              <label className="flex items-center gap-2 text-caption text-muted">
                <Switch checked={allLabs} onChange={setAllLabs} label="Show environments from all labs" />
                All labs
              </label>
            </StepHeader>
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="min-w-0">
                <div className="mb-2 flex items-baseline justify-between">
                  <span className="text-body-s font-medium">Environment</span>
                  <span className="num text-caption text-subtle">{envList.length}</span>
                </div>
                <SearchInput value={envQuery} onChange={setEnvQuery} placeholder="Filter environments" />
                <div role="listbox" aria-label="Environments" className="mt-2 h-80 space-y-0.5 overflow-y-auto pr-1">
                  {cat.isLoading && Array.from({ length: 7 }).map((_, i) => <Skeleton key={i} className="h-11" />)}
                  {pinFirst(envList, envName).map((e) => (
                    <PickerRow
                      key={e.name}
                      selected={e.name === envName}
                      onClick={() => setEnvName(e.name)}
                      title={e.label}
                      subtitle={spacesSummary(e)}
                      disabled={e.available === false}
                      right={
                        e.available === false ? (
                          <CircleSlash size={14} className="text-subtle" aria-label="Not installed" />
                        ) : (
                          <DifficultyBadge value={e.difficulty} />
                        )
                      }
                    />
                  ))}
                  {!cat.isLoading && !envList.length && (
                    <p className="p-4 text-center text-body-s text-muted">
                      No environments match. {!allLabs && 'Try “All labs”.'}
                    </p>
                  )}
                </div>
              </div>
              <div className="min-w-0">
                <div className="mb-2 flex items-baseline justify-between">
                  <span className="text-body-s font-medium">Algorithm</span>
                  <span className="text-caption text-subtle">
                    {envName ? 'Sorted by fit' : 'Pick an environment to check fit'}
                  </span>
                </div>
                <SearchInput value={agentQuery} onChange={setAgentQuery} placeholder="Filter algorithms" />
                <div role="listbox" aria-label="Algorithms" className="mt-2 h-80 space-y-0.5 overflow-y-auto pr-1">
                  {cat.isLoading && Array.from({ length: 7 }).map((_, i) => <Skeleton key={i} className="h-11" />)}
                  {pinFirst(agentList, agentName).map((a) => {
                    const level = envName ? compat.lookup(a.name, envName)?.level : null
                    const planned = a.status !== 'implemented'
                    return (
                      <PickerRow
                        key={a.name}
                        selected={a.name === agentName}
                        onClick={() => setAgentName(a.name)}
                        title={a.label}
                        subtitle={`${familyLabel(a.category)} · ${a.type}`}
                        disabled={planned || level === 'blocked'}
                        dot={<CompatDot level={planned ? 'blocked' : level} />}
                        right={
                          planned ? (
                            <Badge tone="outline">Planned</Badge>
                          ) : lab?.algorithms?.includes(a.name) ? (
                            <Sparkles size={14} className="text-accent" aria-label="Suited to this lab" />
                          ) : null
                        }
                      />
                    )
                  })}
                </div>
              </div>
            </div>
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
