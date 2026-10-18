import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { BookOpen, Code2, Play } from 'lucide-react'
import { useLab } from '../app/AppContext.jsx'
import { useAgentCode, useCatalogue, useCompatMatrix } from '../lib/queries.js'
import { metricLabel } from '../lib/format.js'
import { ButtonLink } from '../components/ui/Button.jsx'
import { Card, CardHeader, Section } from '../components/ui/Card.jsx'
import { ErrorState, Notice, Skeleton } from '../components/ui/Feedback.jsx'
import { KeyValue, PageHeader } from '../components/ui/Page.jsx'
import { Segmented, Tabs } from '../components/ui/Segmented.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { Table, THead, Th, Tr, Td } from '../components/ui/Table.jsx'
import { LabGlyph } from '../components/art/LabArt.jsx'
import { MathText } from '../components/domain/MathText.jsx'
import { CodeBlock } from '../components/domain/CodeBlock.jsx'
import { familyLabel } from '../components/domain/CatalogCards.jsx'
import NotFound from './NotFound.jsx'

const UPDATE_LINE =
  /def (update|_learn|_update|end_of_episode|_q_update)\b|loss\s*=|targets?\s*=|\btd\s*=|advantages?\s*=|ratio\s*=/

function Bullets({ items }) {
  if (!items?.length) return <p className="text-body-s text-muted">None documented.</p>
  return (
    <ul className="space-y-2 text-body-s text-muted">
      {items.map((t) => (
        <li key={t} className="flex gap-2">
          <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-subtle" />
          <span>{t}</span>
        </li>
      ))}
    </ul>
  )
}

function CodeView({ agent }) {
  const [mode, setMode] = useState('pseudo')
  const code = useAgentCode(agent.name, mode === 'real')
  const implemented = agent.status === 'implemented'
  return (
    <Card>
      <CardHeader
        title="Algorithm"
        icon={Code2}
        description={
          mode === 'real' && code.data
            ? `${code.data.file}, from line ${code.data.line}. Update logic is highlighted.`
            : 'The core loop, stripped of bookkeeping.'
        }
        actions={
          implemented && (
            <Segmented
              size="sm"
              label="Code view"
              value={mode}
              onChange={setMode}
              options={[
                { value: 'pseudo', label: 'Pseudocode' },
                { value: 'real', label: 'Source' },
              ]}
            />
          )
        }
      />
      {mode === 'pseudo' ? (
        agent.docs.pseudocode ? (
          <CodeBlock code={agent.docs.pseudocode} maxHeight={420} />
        ) : (
          <p className="text-body-s text-muted">No pseudocode documented.</p>
        )
      ) : code.isLoading ? (
        <Skeleton className="h-80" />
      ) : code.error ? (
        <ErrorState error={code.error} onRetry={code.refetch} title="Could not load source" />
      ) : (
        <CodeBlock
          code={code.data.source}
          startLine={code.data.line}
          highlight={(l) => UPDATE_LINE.test(l)}
          maxHeight={520}
        />
      )}
    </Card>
  )
}

export default function AlgorithmDetail() {
  const { agentName } = useParams()
  const { agentBy, envs, labs, isLoading, error, refetch } = useCatalogue()
  const compat = useCompatMatrix()
  const { labId } = useLab()
  const [tab, setTab] = useState('overview')
  const agent = agentBy[agentName]

  if (error) return <ErrorState error={error} onRetry={refetch} />
  if (isLoading)
    return (
      <div className="space-y-6">
        <Skeleton className="h-24" />
        <Skeleton className="h-80" />
      </div>
    )
  if (!agent) return <NotFound what={`an algorithm called “${agentName}”`} />

  const d = agent.docs || {}
  const implemented = agent.status === 'implemented'
  const homeLab =
    labs.find((l) => l.id === labId && l.algorithms.includes(agent.name)) ||
    labs.find((l) => l.algorithms.includes(agent.name))
  const perLab = labs.map((l) => {
    const members = envs.filter((e) => e.lab === l.id && e.available !== false)
    const ok = members.filter((e) => compat.lookup(agent.name, e.name)?.compatible)
    return { lab: l, ok, total: members.length }
  })
  const s = agent.supports || {}

  return (
    <div className="space-y-6">
      <PageHeader
        title={agent.label}
        description={agent.description}
        meta={
          <>
            <Badge tone="outline">
              <span className="num">{agent.name}</span>
            </Badge>
            <Badge>{familyLabel(agent.category)}</Badge>
            <Badge>{agent.type}</Badge>
            {implemented ? <Badge tone="success">Trainable</Badge> : <Badge tone="outline">Planned</Badge>}
            {agent.paper && <span className="text-caption text-subtle">{agent.paper}</span>}
          </>
        }
        actions={
          implemented && (
            <ButtonLink to={`/lab/${homeLab?.id || labId}/new?agent=${agent.name}`} variant="primary" icon={Play}>
              Train {agent.label}
            </ButtonLink>
          )
        }
      />

      {!implemented && (
        <Notice tone="info" title="Not trainable yet">
          {agent.status_reason ||
            'This algorithm is documented for reference but has no implementation in the testbed.'}
        </Notice>
      )}

      <Tabs
        value={tab}
        onChange={setTab}
        tabs={[
          { value: 'overview', label: 'Overview', icon: BookOpen },
          { value: 'theory', label: 'Theory and code', icon: Code2 },
          { value: 'params', label: 'Hyperparameters', count: agent.params.length },
          { value: 'fit', label: 'Where it fits' },
        ]}
      />

      {tab === 'overview' && (
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-2">
            {d.beginner && (
              <Card>
                <CardHeader title="The idea" />
                <p className="prose-lab">{d.beginner}</p>
                {d.intuition && <p className="prose-lab mt-3 border-l-2 border-accent/50 pl-4 italic">{d.intuition}</p>}
              </Card>
            )}
            {d.technical && (
              <Card>
                <CardHeader title="Precisely" />
                <p className="prose-lab">{d.technical}</p>
                {d.update_rule && (
                  <pre className="mt-4 overflow-x-auto rounded border border-border bg-bg-soft px-4 py-3 text-body-s text-text">
                    {d.update_rule}
                  </pre>
                )}
              </Card>
            )}
            <Card>
              <CardHeader title="Anatomy" />
              <KeyValue
                rows={[
                  ['Learns', d.learns],
                  ['Exploration', d.exploration],
                  ['Replay buffer', d.replay_buffer],
                  ['Target network', d.target_network],
                  ['Advantage', d.advantage],
                  ['Networks', d.networks],
                  [
                    'Lifecycle',
                    d.lifecycle && (
                      <span key="l" className="whitespace-pre-line">
                        {d.lifecycle}
                      </span>
                    ),
                  ],
                ]}
              />
            </Card>
          </div>
          <div className="space-y-6">
            <Card>
              <CardHeader title="Spaces" />
              <KeyValue
                rows={[
                  ['Observations', (s.obs_types || []).join(', ')],
                  ['Actions', (s.action_types || []).join(', ')],
                  ['Action masks', s.action_mask ? 'Respected' : 'Ignored'],
                  ['Multi-agent', s.multi_agent ? 'Yes' : 'No'],
                ]}
              />
            </Card>
            <Card>
              <CardHeader title="How it fails" />
              <Bullets items={d.failure_modes} />
            </Card>
            {d.expected_metrics?.length > 0 && (
              <Card>
                <CardHeader title="Metrics it reports" />
                <ul className="space-y-2 text-body-s">
                  {d.expected_metrics.map((m) => (
                    <li key={m.key}>
                      <span className="font-medium text-text">{metricLabel(m.key)}</span>
                      <span className="block text-muted">{m.meaning}</span>
                    </li>
                  ))}
                </ul>
              </Card>
            )}
            {d.related?.length > 0 && (
              <Card>
                <CardHeader title="Related" />
                <div className="flex flex-wrap gap-1.5">
                  {d.related
                    .filter((r) => r !== agent.name)
                    .map((r) => (
                      <Link key={r} to={`/algorithms/${r}`}>
                        <Badge className="hover:border-border-strong hover:text-text">{agentBy[r]?.label || r}</Badge>
                      </Link>
                    ))}
                </div>
              </Card>
            )}
          </div>
        </div>
      )}

      {tab === 'theory' && (
        <div className="grid gap-6 xl:grid-cols-2">
          <Card>
            <CardHeader title="Theory" />
            {d.math ? (
              <MathText source={d.math} className="prose-lab space-y-3" />
            ) : (
              <p className="prose-lab">{d.technical || 'No theory written for this algorithm yet.'}</p>
            )}
          </Card>
          <CodeView agent={agent} />
        </div>
      )}

      {tab === 'params' &&
        (agent.params.length ? (
          <Table>
            <THead>
              <Th>Parameter</Th>
              <Th align="right">Default</Th>
              <Th>Range</Th>
              <Th>What it does</Th>
            </THead>
            <tbody>
              {agent.params.map((p) => (
                <Tr key={p.name}>
                  <Td>
                    <span className="block font-medium text-text">{p.label}</span>
                    <span className="num text-caption text-subtle">
                      {p.name}
                      {p.advanced && ' · advanced'}
                    </span>
                  </Td>
                  <Td align="right" mono>
                    {String(p.default)}
                  </Td>
                  <Td mono className="whitespace-nowrap text-muted">
                    {p.type === 'choice'
                      ? p.choices.join(' | ')
                      : [p.min ?? '', p.max ?? ''].join(' – ').replace(/^ – $/, '—')}
                  </Td>
                  <Td className="text-muted">{p.help}</Td>
                </Tr>
              ))}
            </tbody>
          </Table>
        ) : (
          <p className="text-body-s text-muted">No hyperparameters.</p>
        ))}

      {tab === 'fit' && (
        <div className="space-y-6">
          <Section
            title="By lab"
            description="How many installed environments in each lab this algorithm can train on."
          >
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
              {perLab.map(({ lab, ok, total }) => (
                <Link
                  key={lab.id}
                  to={`/lab/${lab.id}/algorithms`}
                  className="rounded-lg border border-border bg-surface p-4 hover:border-border-strong"
                  style={{ '--accent': `var(--hue-${lab.accent})` }}
                >
                  <span className="flex items-center gap-2 text-body-s font-medium text-text">
                    <LabGlyph icon={lab.icon} size={16} className="text-accent" />
                    {lab.label}
                  </span>
                  <span className="num mt-2 block text-title text-text">
                    {ok.length}
                    <span className="text-body-s text-subtle"> / {total}</span>
                  </span>
                  <span className="block truncate text-caption text-subtle">
                    {ok
                      .slice(0, 3)
                      .map((e) => e.label)
                      .join(', ') || 'None'}
                  </span>
                </Link>
              ))}
            </div>
          </Section>
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader title="Works well on" />
              <div className="flex flex-wrap gap-1.5">
                {(d.recommended_envs || []).length ? (
                  d.recommended_envs.map((e) => (
                    <Link key={e} to={`/envs/${e}`}>
                      <Badge tone="success">{envs.find((x) => x.name === e)?.label || e}</Badge>
                    </Link>
                  ))
                ) : (
                  <p className="text-body-s text-muted">Not documented.</p>
                )}
              </div>
            </Card>
            <Card>
              <CardHeader title="Struggles on" />
              <div className="flex flex-wrap gap-1.5">
                {(d.poor_envs || []).length ? (
                  d.poor_envs.map((e) => (
                    <Link key={e} to={`/envs/${e}`}>
                      <Badge tone="warning">{envs.find((x) => x.name === e)?.label || e}</Badge>
                    </Link>
                  ))
                ) : (
                  <p className="text-body-s text-muted">Not documented.</p>
                )}
              </div>
            </Card>
          </div>
        </div>
      )}
    </div>
  )
}
