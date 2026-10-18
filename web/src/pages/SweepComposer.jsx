import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { SlidersHorizontal } from 'lucide-react'
import { useRouteLab } from '../app/AppContext.jsx'
import { api } from '../lib/api.js'
import { useCatalogue, useSystem } from '../lib/queries.js'
import { fmtCompact } from '../lib/format.js'
import { cn } from '../lib/utils.js'
import { Button } from '../components/ui/Button.jsx'
import { Card } from '../components/ui/Card.jsx'
import { ErrorState, Notice, Skeleton } from '../components/ui/Feedback.jsx'
import { Field, Input } from '../components/ui/Field.jsx'
import { PageHeader } from '../components/ui/Page.jsx'
import { Segmented } from '../components/ui/Segmented.jsx'
import { Table, THead, Th, Tr, Td } from '../components/ui/Table.jsx'
import { useToast } from '../components/ui/Toast.jsx'
import { AllLabsToggle, ExperimentPicker, StepHeader } from '../components/domain/ExperimentPicker.jsx'
import { SpaceEditor, suggestValues } from '../components/domain/SpaceEditor.jsx'

const SEEDS = [1, 2, 3, 5].map((n) => ({ value: n, label: String(n) }))

/** Debounce a value so the preview endpoint is not called on every keystroke. */
function useDebounced(value, ms = 300) {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}

export default function SweepComposer() {
  const { labId, lab } = useRouteLab()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const { toast } = useToast()
  const cat = useCatalogue()
  const sys = useSystem(true)

  const [envName, setEnvName] = useState('')
  const [agentName, setAgentName] = useState('')
  const [allLabs, setAllLabs] = useState(false)
  const [strategy, setStrategy] = useState('grid')
  const [trials, setTrials] = useState(12)
  const [space, setSpace] = useState({})
  const [episodes, setEpisodes] = useState(200)
  const [seeds, setSeeds] = useState(2)
  const [firstSeed, setFirstSeed] = useState(0)
  const [evalFreq, setEvalFreq] = useState(25)
  const [name, setName] = useState('')

  const env = cat.envBy[envName]
  const agent = cat.agentBy[agentName]
  const chosen = !!env && !!agent

  useEffect(() => setSpace({}), [agentName])

  // Grid search cannot use ranges: turn any range back into a value list.
  const changeStrategy = (next) => {
    setStrategy(next)
    if (next === 'grid' && agent) {
      const byName = Object.fromEntries(agent.params.map((p) => [p.name, p]))
      setSpace((s) =>
        Object.fromEntries(
          Object.entries(s).map(([k, d]) => [k, 'min' in d ? { values: suggestValues(byName[k]) } : d]),
        ),
      )
    }
  }

  const body = useMemo(
    () => ({
      agent: agentName,
      env: envName,
      name: name.trim() || undefined,
      strategy,
      n_trials: Number(trials) || 1,
      space,
      seeds: Array.from({ length: seeds }, (_, i) => Number(firstSeed) + i),
      episodes: Number(episodes) || 1,
      eval_freq: Number(evalFreq) || 25,
    }),
    [agentName, envName, name, strategy, trials, space, seeds, firstSeed, episodes, evalFreq],
  )
  const debounced = useDebounced(body)
  const preview = useQuery({
    queryKey: ['sweep-preview', debounced],
    queryFn: () => api.previewSweep(debounced),
    enabled: chosen && Object.keys(debounced.space).length > 0,
  })

  const create = useMutation({
    mutationFn: api.createSweep,
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['sweeps'] })
      qc.invalidateQueries({ queryKey: ['runs'] })
      toast({ title: `Sweep launched: ${res.run_ids.length} runs`, description: `${res.n_configs} configurations` })
      navigate(`/sweeps/${res.sweep_id}`)
    },
  })

  const p = preview.data
  const stepsPerRun = (env?.typical_episode_length || env?.max_steps || 0) * (Number(episodes) || 0)
  const limit = sys.data?.jobs?.max_concurrent

  if (cat.error) return <ErrorState error={cat.error} onRetry={cat.refetch} />

  return (
    <div>
      <PageHeader
        title="New sweep"
        description="Search over hyperparameters: every configuration runs with several seeds, then configurations are ranked by mean result."
      />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-6">
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

          <Card className={cn(!chosen && 'opacity-60')}>
            <StepHeader n={2} title="Search space" done={chosen && p?.valid}>
              {chosen && (
                <Segmented
                  size="sm"
                  label="Search strategy"
                  value={strategy}
                  onChange={changeStrategy}
                  options={[
                    { value: 'grid', label: 'Grid' },
                    { value: 'random', label: 'Random' },
                  ]}
                />
              )}
            </StepHeader>
            {!chosen ? (
              <p className="text-body-s text-muted">Pick an environment and an algorithm to see its parameters.</p>
            ) : (
              <div className="space-y-4">
                <p className="text-body-s text-muted">
                  {strategy === 'grid'
                    ? 'Grid search tries every combination of the values you list. It is exhaustive and grows multiplicatively.'
                    : 'Random search samples configurations: lists by uniform choice, ranges uniformly or on a log scale. Usually finds good settings with fewer runs than a grid.'}
                </p>
                {strategy === 'random' && (
                  <Field label="Configurations to sample" help="Duplicates are dropped, so small lists may give fewer.">
                    {(id) => (
                      <Input
                        id={id}
                        type="number"
                        min={1}
                        className="num w-40"
                        value={trials}
                        onChange={(e) => setTrials(e.target.value)}
                      />
                    )}
                  </Field>
                )}
                <SpaceEditor agent={agent} space={space} onChange={setSpace} strategy={strategy} />
              </div>
            )}
          </Card>

          <Card className={cn(!chosen && 'opacity-60')}>
            <StepHeader n={3} title="Budget" done={chosen && Number(episodes) > 0} />
            <div className="grid gap-x-5 gap-y-4 sm:grid-cols-2 lg:grid-cols-4">
              <Field label="Episodes per run">
                {(id) => (
                  <Input
                    id={id}
                    type="number"
                    min={1}
                    className="num"
                    value={episodes}
                    onChange={(e) => setEpisodes(e.target.value)}
                  />
                )}
              </Field>
              <Field label="Seeds per configuration" help="2+ to see variance">
                <Segmented
                  size="sm"
                  label="Seeds per configuration"
                  value={seeds}
                  onChange={setSeeds}
                  options={SEEDS}
                />
              </Field>
              <Field label="First seed">
                {(id) => (
                  <Input
                    id={id}
                    type="number"
                    min={0}
                    className="num"
                    value={firstSeed}
                    onChange={(e) => setFirstSeed(e.target.value)}
                  />
                )}
              </Field>
              <Field label="Evaluate every" help="Episodes">
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
            <Field label="Name" hint="Optional" className="mt-4">
              {(id) => (
                <Input
                  id={id}
                  value={name}
                  placeholder={agentName ? `${agentName}-${envName} sweep` : 'My sweep'}
                  onChange={(e) => setName(e.target.value)}
                />
              )}
            </Field>
          </Card>
        </div>

        <aside className="xl:sticky xl:top-20 xl:self-start">
          <Card raised>
            <StepHeader n={4} title="Review and launch" done={false} />
            {!chosen ? (
              <p className="text-body-s text-muted">
                The plan appears here: how many configurations, how many runs, and any setting the server would reject.
              </p>
            ) : !Object.keys(space).length ? (
              <p className="text-body-s text-muted">
                Switch on a parameter in step 2 to start building the search space.
              </p>
            ) : preview.isLoading ? (
              <Skeleton className="h-40" />
            ) : p ? (
              <div className="space-y-4">
                {p.valid ? (
                  <>
                    <div className="rounded-lg border border-border bg-surface-2 p-4">
                      <div className="num text-display font-semibold text-text">{p.n_runs}</div>
                      <div className="text-body-s text-muted">
                        runs: {p.n_configs} configuration{p.n_configs === 1 ? '' : 's'} × {seeds} seed
                        {seeds === 1 ? '' : 's'}
                      </div>
                      <div className="mt-2 text-caption text-subtle">
                        ≈ {fmtCompact(stepsPerRun * p.n_runs)} environment steps
                        {limit ? `, ${limit} at a time` : ''}
                      </div>
                    </div>
                    <div className="max-h-56 overflow-auto">
                      <Table>
                        <THead>
                          <Th>#</Th>
                          {Object.keys(space)
                            .sort()
                            .map((k) => (
                              <Th key={k}>{k}</Th>
                            ))}
                        </THead>
                        <tbody>
                          {p.configs.slice(0, 12).map((c, i) => (
                            <Tr key={i}>
                              <Td mono className="text-subtle">
                                {i + 1}
                              </Td>
                              {Object.keys(space)
                                .sort()
                                .map((k) => (
                                  <Td key={k} mono>
                                    {String(c[k])}
                                  </Td>
                                ))}
                            </Tr>
                          ))}
                        </tbody>
                      </Table>
                    </div>
                    {p.n_configs > 12 && <p className="text-caption text-subtle">First 12 of {p.n_configs} shown.</p>}
                  </>
                ) : (
                  <Notice tone="danger" title="This sweep cannot run yet">
                    <ul className="mt-1 list-disc space-y-1 pl-4">
                      {p.errors.map((e) => (
                        <li key={e}>{e}</li>
                      ))}
                    </ul>
                  </Notice>
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
                  icon={SlidersHorizontal}
                  disabled={!p.valid}
                  loading={create.isPending}
                  onClick={() => create.mutate(body)}
                >
                  Launch {p.valid ? `${p.n_runs} runs` : 'sweep'}
                </Button>
                <p className="text-center text-caption text-subtle">
                  Runs wait in the queue and start as slots free up.{' '}
                  <Link to={`/lab/${labId}/sweeps`} className="text-accent hover:underline">
                    All sweeps
                  </Link>
                </p>
              </div>
            ) : (
              preview.error && <ErrorState error={preview.error} onRetry={preview.refetch} />
            )}
          </Card>
        </aside>
      </div>
    </div>
  )
}
