import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { CheckCircle2, Download, FileUp, Play, TriangleAlert } from 'lucide-react'
import { api } from '../lib/api.js'
import { plural } from '../lib/format.js'
import { Button } from '../components/ui/Button.jsx'
import { Card, CardHeader } from '../components/ui/Card.jsx'
import { Notice, Skeleton } from '../components/ui/Feedback.jsx'
import { Select } from '../components/ui/Field.jsx'
import { PageHeader } from '../components/ui/Page.jsx'
import { useToast } from '../components/ui/Toast.jsx'

const DRAFT_KEY = 'rltestbed.experiment-draft'

const STARTER = `version: 1
name: My experiment
description: Two algorithms on CartPole, three seeds each.
runs:
  - agent: dqn
    env: cartpole
    episodes: 300
    seeds: [0, 1, 2]
  - agent: ppo
    env: cartpole
    episodes: 300
    seeds: [0, 1, 2]
`

function useDebounced(value, ms = 400) {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}

function readDraft() {
  try {
    return window.localStorage.getItem(DRAFT_KEY) || STARTER
  } catch {
    return STARTER
  }
}

/** Write, load or paste an experiment file; validate it live; launch it. */
export default function Experiments() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const { toast } = useToast()
  const [text, setText] = useState(readDraft)
  const fileInput = useRef(null)
  const examples = useQuery({ queryKey: ['experiment-examples'], queryFn: api.experimentExamples, staleTime: 300_000 })
  const debounced = useDebounced(text)
  const check = useQuery({
    queryKey: ['experiment-validate', debounced],
    queryFn: () => api.validateExperiment(debounced),
    placeholderData: (prev) => prev,
  })

  useEffect(() => {
    try {
      window.localStorage.setItem(DRAFT_KEY, text)
    } catch {
      /* storage disabled */
    }
  }, [text])

  const launch = useMutation({
    mutationFn: () => api.launchExperiment(text),
    onSuccess: (out) => {
      qc.invalidateQueries({ queryKey: ['runs'] })
      qc.invalidateQueries({ queryKey: ['sweeps'] })
      toast({ title: `Launched ${plural(out.run_ids.length, 'run')}`, description: out.name })
      if (out.sweeps.length === 1 && !out.runs.length) navigate(`/sweeps/${out.sweeps[0].sweep_id}`)
      else navigate(`/compare?runs=${out.run_ids.slice(0, 60).join(',')}&group_by=config`)
    },
  })

  const openFile = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    file.text().then(setText)
    e.target.value = ''
  }
  const download = () => {
    const name = (/^name:\s*(.+)$/m.exec(text)?.[1] || 'experiment').trim().replace(/[^\w.-]+/g, '-')
    const url = URL.createObjectURL(new Blob([text], { type: 'application/x-yaml' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `${name}.yaml`
    a.click()
    URL.revokeObjectURL(url)
  }

  const result = check.data
  const stale = debounced !== text || check.isFetching
  const lines = text.split('\n').length

  return (
    <div>
      <PageHeader
        title="Experiments"
        description="Describe runs and sweeps in a YAML file, check it, launch it. Export any run, comparison or sweep to a file, commit it with your code, and anyone can reproduce it."
      />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Card pad="none" className="min-w-0 overflow-hidden">
          <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-3">
            <Select
              aria-label="Load an example"
              value=""
              className="h-8 w-auto min-w-56 text-body-s"
              onChange={(e) => {
                const ex = (examples.data || []).find((x) => x.id === e.target.value)
                if (ex) setText(ex.text)
              }}
            >
              <option value="">Load an example…</option>
              {(examples.data || []).map((ex) => (
                <option key={ex.id} value={ex.id}>
                  {ex.label}
                </option>
              ))}
            </Select>
            <Button size="sm" icon={FileUp} onClick={() => fileInput.current?.click()}>
              Open file
            </Button>
            <input
              ref={fileInput}
              type="file"
              accept=".yaml,.yml,.json,text/yaml"
              className="hidden"
              onChange={openFile}
            />
            <Button size="sm" icon={Download} onClick={download}>
              Download
            </Button>
            <span className="num ml-auto text-caption text-subtle">{plural(lines, 'line')}</span>
          </div>
          <div className="flex min-h-[520px] bg-bg-soft">
            <div
              aria-hidden
              className="num select-none border-r border-border px-3 py-3 text-right text-caption leading-6 text-subtle/70"
            >
              {Array.from({ length: lines }, (_, i) => (
                <div key={i}>{i + 1}</div>
              ))}
            </div>
            <textarea
              aria-label="Experiment file"
              spellCheck={false}
              wrap="off"
              value={text}
              style={{ height: Math.max(520, lines * 24 + 24) }}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                // Tab inserts two spaces, as YAML expects
                if (e.key === 'Tab' && !e.shiftKey) {
                  e.preventDefault()
                  const el = e.target
                  const { selectionStart: s, selectionEnd: end } = el
                  const next = `${text.slice(0, s)}  ${text.slice(end)}`
                  setText(next)
                  requestAnimationFrame(() => el.setSelectionRange(s + 2, s + 2))
                }
              }}
              className="w-full flex-1 resize-none overflow-y-hidden bg-transparent px-4 py-3 font-mono text-caption leading-6 text-text focus:outline-none"
            />
          </div>
        </Card>

        <aside className="space-y-4 xl:sticky xl:top-20 xl:self-start">
          <Card raised>
            <CardHeader
              title="Check"
              description={stale ? 'Checking…' : 'Validated by the same rules as every launch.'}
            />
            {!result ? (
              <Skeleton className="h-32" />
            ) : result.valid ? (
              <div className="space-y-4">
                <div className="flex items-start gap-3 rounded-lg border border-success/30 bg-success/[.05] p-3">
                  <CheckCircle2 size={16} className="mt-0.5 shrink-0 text-success" aria-hidden />
                  <div className="min-w-0 text-body-s">
                    <p className="font-semibold text-text">{result.summary.name}</p>
                    <p className="text-muted">
                      {plural(result.summary.n_runs, 'run')} from {plural(result.summary.n_entries, 'entry', 'entries')}
                    </p>
                  </div>
                </div>
                <ul className="space-y-1.5">
                  {result.summary.entries.map((e) => (
                    <li key={`${e.kind}${e.index}`} className="flex items-baseline justify-between gap-3 text-body-s">
                      <span className="min-w-0 truncate text-text">
                        <span className="num text-caption text-subtle">
                          {e.kind}s[{e.index}]
                        </span>{' '}
                        {e.agent} on {e.env}
                      </span>
                      <span className="num shrink-0 text-caption text-muted">
                        {e.kind === 'sweep' ? `${e.n_configs} configs, ` : ''}
                        {e.n_runs} runs
                      </span>
                    </li>
                  ))}
                </ul>
                {launch.error && (
                  <Notice tone="danger" title="Launch failed">
                    {launch.error.message}
                  </Notice>
                )}
                <Button
                  variant="primary"
                  size="lg"
                  className="w-full"
                  icon={Play}
                  disabled={stale}
                  loading={launch.isPending}
                  onClick={() => launch.mutate()}
                >
                  Launch {plural(result.summary.n_runs, 'run')}
                </Button>
              </div>
            ) : (
              <div className="space-y-2" role="alert">
                <div className="flex items-center gap-2 text-body-s font-semibold text-danger">
                  <TriangleAlert size={16} aria-hidden /> {plural(result.errors.length, 'problem')}
                </div>
                <ul className="space-y-2">
                  {result.errors.map((e, i) => (
                    <li key={i} className="rounded border border-border bg-surface-2 px-3 py-2 text-body-s">
                      <code className="text-caption text-accent">{e.loc}</code>
                      <p className="mt-0.5 break-words text-muted">{e.message}</p>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </Card>

          <Card pad="sm">
            <details>
              <summary className="cursor-pointer text-body-s font-semibold text-text">File format</summary>
              <div className="prose-lab mt-3 space-y-2 text-body-s">
                <p>
                  Top level: <code>version: 1</code>, <code>name</code>, <code>description</code>, and lists{' '}
                  <code>runs</code> and <code>sweeps</code>.
                </p>
                <p>
                  A run takes the fields of New run: <code>agent</code>, <code>env</code>, <code>episodes</code>,{' '}
                  <code>seed</code> or <code>seeds</code>, <code>params</code>, <code>eval_freq</code>,{' '}
                  <code>tags</code>.
                </p>
                <p>
                  A sweep adds <code>strategy</code> (grid or random), <code>space</code> (each parameter as{' '}
                  <code>{'{values: [...]}'}</code> or <code>{'{min, max, log}'}</code>) and <code>n_trials</code>.
                </p>
                <p>
                  Unknown keys are errors, so typos never pass silently. Nothing is created unless the whole file is
                  valid.
                </p>
                <p>
                  From a terminal: <code>rltestbed run file.yaml --dry-run</code>.
                </p>
              </div>
            </details>
          </Card>
        </aside>
      </div>
    </div>
  )
}
