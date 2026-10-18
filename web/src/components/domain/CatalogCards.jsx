import { Link } from 'react-router-dom'
import { CircleSlash, Play } from 'lucide-react'
import { cn } from '../../lib/utils.js'
import { LabGlyph } from '../art/LabArt.jsx'
import { Badge, DifficultyBadge } from '../ui/Badge.jsx'
import { Tooltip } from '../ui/Overlay.jsx'

const CATEGORY_LABEL = {
  builtin: 'Builtin',
  'toy-text': 'Toy text',
  gym: 'Gymnasium',
  continuous: 'Continuous classic',
  mujoco: 'MuJoCo',
  minatar: 'MinAtar',
  pettingzoo: 'PettingZoo',
  probe: 'Probe',
  custom: 'Custom',
  minigrid: 'MiniGrid',
}
export const categoryLabel = (c) => CATEGORY_LABEL[c] || c

const ALGO_FAMILY = {
  'value-based': 'Value-based',
  'policy-gradient': 'Policy gradient',
  'actor-critic': 'Actor-critic',
  'model-based': 'Model-based',
  bandit: 'Bandit',
  baseline: 'Baseline',
  'multi-agent': 'Multi-agent',
  offline: 'Offline',
  exploration: 'Exploration',
}
export const familyLabel = (c) => ALGO_FAMILY[c] || c

/** One-line spaces summary: "4-dim vector → 2 discrete actions". */
export function spacesSummary(env) {
  const obs = env.obs_dim ? `${env.obs_dim}-dim ${env.obs_type}` : env.obs_type
  const act = env.act_dim
    ? `${env.act_dim} ${env.action_type}${env.action_type === 'discrete' ? ' actions' : ' action dims'}`
    : `${env.action_type} actions`
  return `${obs} → ${act}`
}

export function EnvCard({ env, labIcon, selected, onSelect, compact }) {
  const available = env.available !== false
  const body = (
    <>
      <div className="flex items-start gap-3">
        <span
          className={cn(
            'grid h-9 w-9 shrink-0 place-items-center rounded',
            available ? 'bg-accent/12 text-accent' : 'bg-surface-3 text-subtle',
          )}
        >
          <LabGlyph icon={labIcon} size={18} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="truncate text-body font-semibold text-text">{env.label}</h3>
          </div>
          <p className="num truncate text-caption text-subtle">{env.name}</p>
        </div>
        {available ? (
          <DifficultyBadge value={env.difficulty} />
        ) : (
          <Tooltip content={env.unavailable_reason}>
            <Badge tone="outline">
              <CircleSlash size={12} aria-hidden /> Not installed
            </Badge>
          </Tooltip>
        )}
      </div>
      {!compact && (
        <p className="mt-3 line-clamp-3 text-body-s text-muted" title={env.description}>
          {env.description}
        </p>
      )}
      <div className="mt-auto flex flex-wrap items-center gap-x-3 gap-y-1 pt-3 text-caption text-subtle">
        <span>{spacesSummary(env)}</span>
        <span>
          up to {env.max_steps.toLocaleString()} {env.max_steps === 1 ? 'step' : 'steps'}
        </span>
        {env.sparse_reward && <span className="text-warning">sparse reward</span>}
      </div>
    </>
  )
  const cls = cn(
    'flex h-full min-h-0 flex-col rounded-lg border bg-surface p-4 text-left transition-[border-color,background-color] duration-fast ease-standard',
    selected ? 'border-accent bg-accent/[.05]' : 'border-border hover:border-border-strong hover:bg-surface-2',
    !available && 'opacity-80',
  )
  if (onSelect)
    return (
      <button type="button" onClick={() => onSelect(env)} aria-pressed={!!selected} className={cls}>
        {body}
      </button>
    )
  return (
    <Link to={`/envs/${env.name}`} className={cls}>
      {body}
    </Link>
  )
}

export function AlgoCard({ agent, compatCount, recommended, to, actionTo }) {
  const planned = agent.status !== 'implemented'
  const s = agent.supports || {}
  return (
    <div
      className={cn(
        'group relative flex h-full flex-col rounded-lg border border-border bg-surface p-4 transition-[border-color,background-color] duration-fast hover:border-border-strong hover:bg-surface-2',
        planned && 'border-dashed',
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate text-body font-semibold text-text">
            <Link to={to || `/algorithms/${agent.name}`} className="after:absolute after:inset-0 hover:text-accent">
              {agent.label}
            </Link>
          </h3>
          <p className="text-caption text-subtle">
            {familyLabel(agent.category)} · {agent.type}
          </p>
        </div>
        {planned ? (
          <Badge tone="outline">Planned</Badge>
        ) : recommended ? (
          <Badge tone="accent">Suited to this lab</Badge>
        ) : null}
      </div>
      <p className="mt-3 line-clamp-3 text-body-s text-muted" title={agent.description}>
        {planned && agent.status_reason ? agent.status_reason : agent.description}
      </p>
      <div className="mt-auto flex flex-wrap items-center gap-1.5 pt-3">
        {(s.action_types || []).map((t) => (
          <Badge key={t} tone="neutral">
            {t} actions
          </Badge>
        ))}
        {(s.obs_types || []).includes('image') && <Badge tone="neutral">pixels</Badge>}
        {compatCount !== undefined && !planned && (
          <span className="ml-auto text-caption text-subtle">
            <span className="num text-muted">{compatCount}</span> envs here
          </span>
        )}
      </div>
      {actionTo && !planned && (
        <Link
          to={actionTo}
          className="relative z-10 mt-3 inline-flex items-center gap-1.5 self-start text-body-s font-medium text-accent hover:underline"
        >
          <Play size={14} aria-hidden /> Train
        </Link>
      )}
    </div>
  )
}
