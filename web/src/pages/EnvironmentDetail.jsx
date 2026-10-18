import { Link, useParams } from 'react-router-dom'
import { ExternalLink, Eye, Play } from 'lucide-react'
import { useAdoptLab } from '../app/AppContext.jsx'
import { useCatalogue, useCompatMatrix, useEnv, useRuns } from '../lib/queries.js'
import { fmtNum } from '../lib/format.js'
import { ButtonLink } from '../components/ui/Button.jsx'
import { Card, CardHeader, Section } from '../components/ui/Card.jsx'
import { ErrorState, Notice, Skeleton } from '../components/ui/Feedback.jsx'
import { KeyValue, PageHeader } from '../components/ui/Page.jsx'
import { Badge, DifficultyBadge } from '../components/ui/Badge.jsx'
import { LabGlyph } from '../components/art/LabArt.jsx'
import { CompatDot } from '../components/domain/CompatVerdict.jsx'
import { RunTable } from '../components/domain/RunTable.jsx'
import { categoryLabel } from '../components/domain/CatalogCards.jsx'
import NotFound from './NotFound.jsx'

export default function EnvironmentDetail() {
  const { envName } = useParams()
  const q = useEnv(envName)
  const { agents, labBy, envBy } = useCatalogue()
  const compat = useCompatMatrix()
  const runs = useRuns({ env: envName })
  const env = q.data
  useAdoptLab(env?.lab)

  if (q.error?.status === 404) return <NotFound what={`an environment called “${envName}”`} />
  if (q.error) return <ErrorState error={q.error} onRetry={q.refetch} />
  if (q.isLoading || !env)
    return (
      <div className="space-y-6">
        <Skeleton className="h-24" />
        <Skeleton className="h-64" />
      </div>
    )

  const lab = labBy[env.lab]
  const labId = lab ? env.lab : 'foundations'
  const implemented = agents.filter((a) => a.status === 'implemented')
  const rated = implemented
    .map((a) => ({ a, v: compat.lookup(a.name, env.name) }))
    .filter((x) => x.v)
    .sort((x, y) => {
      const order = { ok: 0, warning: 1, blocked: 2 }
      const rec = (n) => (env.recommended_agents.includes(n) ? 0 : 1)
      return order[x.v.level] - order[y.v.level] || rec(x.a.name) - rec(y.a.name)
    })
  const firstRec = env.recommended_agents.find((n) => implemented.some((a) => a.name === n))
  const trainLink = (agent) => `/lab/${labId}/new?env=${env.name}${agent ? `&agent=${agent}` : ''}`

  return (
    <div className="space-y-8">
      <PageHeader
        icon={
          <span className="grid h-12 w-12 shrink-0 place-items-center rounded-lg bg-accent/12 text-accent">
            <LabGlyph icon={lab?.icon || 'pulse'} size={24} />
          </span>
        }
        title={env.label}
        description={env.description}
        meta={
          <>
            <Badge tone="outline">
              <span className="num">{env.name}</span>
            </Badge>
            <Badge>{categoryLabel(env.category)}</Badge>
            <DifficultyBadge value={env.difficulty} />
            {lab && (
              <Link to={`/lab/${env.lab}`}>
                <Badge tone="accent">{lab.label} lab</Badge>
              </Link>
            )}
            {env.url && (
              <a
                href={env.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-caption text-accent hover:underline"
              >
                Upstream docs <ExternalLink size={12} aria-hidden />
              </a>
            )}
          </>
        }
        actions={
          <>
            <ButtonLink to={`/visualize?env=${env.name}`} icon={Eye}>
              Watch it
            </ButtonLink>
            <ButtonLink to={trainLink(firstRec)} variant="primary" icon={Play}>
              Train here
            </ButtonLink>
          </>
        }
      />

      {env.available === false && (
        <Notice tone="warning" title="Not installed on this machine">
          {env.unavailable_reason}
        </Notice>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader title="How it works" />
          <KeyValue
            rows={[
              ['Reward', env.reward_desc],
              ['Episode ends when', env.termination_desc],
              [
                'Observation',
                `${env.obs_dim ?? '?'}-dimensional ${env.obs_type}${env.obs_type === 'image' ? ' (flattened for MLPs)' : ''}`,
              ],
              [
                'Action',
                `${env.act_dim ?? '?'} ${env.action_type}${env.action_type === 'discrete' ? ' choices' : ' dimensions'}${env.action_masked ? ', masked to legal moves' : ''}`,
              ],
              ['Step limit', `${env.max_steps.toLocaleString()} steps per episode`],
              ['Typical length', `${env.typical_episode_length.toLocaleString()} steps`],
              [
                'Solved at',
                env.reward_threshold != null ? (
                  <span key="t" className="num">
                    {fmtNum(env.reward_threshold)} mean return
                  </span>
                ) : null,
              ],
              ['Players', env.n_agents > 1 ? `${env.n_agents}` : null],
              ['Reward density', env.sparse_reward ? 'Sparse: most steps give no signal' : null],
              ['Source', env.source],
            ]}
          />
        </Card>
        <Card>
          <CardHeader title="Limitations" />
          {env.limitations.length ? (
            <ul className="space-y-2 text-body-s text-muted">
              {env.limitations.map((l) => (
                <li key={l} className="flex gap-2">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-subtle" />
                  {l}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-body-s text-muted">None documented.</p>
          )}
        </Card>
      </div>

      {(env.beginner_exercise || env.advanced_exercise) && (
        <div className="grid gap-4 md:grid-cols-2">
          {[
            ['Beginner exercise', env.beginner_exercise],
            ['Advanced exercise', env.advanced_exercise],
          ]
            .filter(([, t]) => t)
            .map(([title, text]) => (
              <Card key={title}>
                <CardHeader title={title} />
                <p className="text-body-s text-muted">{text}</p>
              </Card>
            ))}
        </div>
      )}

      <Section title="Which algorithms can train here" description="Checked by the same rules that guard every launch.">
        {compat.isLoading ? (
          <Skeleton className="h-40" />
        ) : (
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {rated.map(({ a, v }) => (
              <div key={a.name} className="flex items-start gap-3 rounded-lg border border-border bg-surface p-3">
                <span className="mt-1.5">
                  <CompatDot level={v.level} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <Link
                      to={`/algorithms/${a.name}`}
                      className="truncate text-body-s font-medium text-text hover:text-accent"
                    >
                      {a.label}
                    </Link>
                    {env.recommended_agents.includes(a.name) && <Badge tone="accent">Recommended</Badge>}
                  </div>
                  <p className="line-clamp-2 text-caption text-subtle">
                    {v.blockers[0] || v.warnings[0] || 'Compatible.'}
                  </p>
                </div>
                {v.compatible && (
                  <Link
                    to={trainLink(a.name)}
                    aria-label={`Train ${a.label} here`}
                    className="rounded p-1.5 text-muted hover:bg-surface-2 hover:text-accent"
                  >
                    <Play size={14} aria-hidden />
                  </Link>
                )}
              </div>
            ))}
          </div>
        )}
      </Section>

      <Section title="Runs on this environment">
        {runs.isLoading ? (
          <Skeleton className="h-32" />
        ) : runs.data?.length ? (
          <RunTable runs={runs.data.slice(0, 20)} compact envBy={envBy} />
        ) : (
          <p className="text-body-s text-muted">
            None yet.{' '}
            <Link to={trainLink(firstRec)} className="text-accent hover:underline">
              Train the first one
            </Link>
            .
          </p>
        )}
      </Section>
    </div>
  )
}
