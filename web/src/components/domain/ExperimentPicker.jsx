import { useMemo, useState } from 'react'
import { Check, CircleSlash, Sparkles } from 'lucide-react'
import { useCatalogue, useCompatMatrix } from '../../lib/queries.js'
import { cn } from '../../lib/utils.js'
import { SearchInput, Switch } from '../ui/Field.jsx'
import { Skeleton } from '../ui/Feedback.jsx'
import { Badge, DifficultyBadge } from '../ui/Badge.jsx'
import { CompatDot } from './CompatVerdict.jsx'
import { familyLabel, spacesSummary } from './CatalogCards.jsx'

/** Keep the chosen item first so it never scrolls out of view. */
const pinFirst = (list, name) => {
  const i = list.findIndex((x) => x.name === name)
  return i > 0 ? [list[i], ...list.slice(0, i), ...list.slice(i + 1)] : list
}

export function StepHeader({ n, title, done, children }) {
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

export function PickerRow({ selected, onClick, title, subtitle, right, disabled, dot }) {
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

export function AllLabsToggle({ value, onChange }) {
  return (
    <label className="flex items-center gap-2 text-caption text-muted">
      <Switch checked={value} onChange={onChange} label="Show environments from all labs" />
      All labs
    </label>
  )
}

/**
 * Environment list + algorithm list, side by side. The algorithm list is sorted
 * by the server's compatibility verdict for the chosen environment, then by the
 * lab's priority. Used by New run and New sweep.
 */
export function ExperimentPicker({ labId, lab, envName, agentName, onEnv, onAgent, allLabs }) {
  const cat = useCatalogue()
  const compat = useCompatMatrix()
  const [envQuery, setEnvQuery] = useState('')
  const [agentQuery, setAgentQuery] = useState('')
  const setEnvName = onEnv
  const setAgentName = onAgent

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

  return (
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
  )
}
