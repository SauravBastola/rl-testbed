import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Layers } from 'lucide-react'
import { fmtNum, fmtRelative, runProgress } from '../../lib/format.js'
import { cn } from '../../lib/utils.js'
import { StatusBadge } from '../ui/Badge.jsx'
import { Progress } from '../ui/Page.jsx'
import { Table, THead, Th, Tr, Td, sortRows, nextSort } from '../ui/Table.jsx'
import { Tooltip } from '../ui/Overlay.jsx'

const ACCESSORS = {
  progress: (r) => runProgress(r).frac,
  reward: (r) => r.final_mean_reward ?? r.progress?.running_reward ?? null,
}

function ProgressCell({ run }) {
  const { done, total, frac } = runProgress(run)
  return (
    <div className="flex min-w-28 items-center gap-2">
      <Progress
        value={frac}
        tone={run.status === 'completed' ? 'success' : run.status === 'running' ? 'accent' : 'neutral'}
        className="w-16"
      />
      <span className="num text-caption text-muted">
        {done.toLocaleString()}
        <span className="text-subtle">/{total.toLocaleString()}</span>
      </span>
    </div>
  )
}

/**
 * The run table used everywhere. `selectable` adds checkboxes (Runs page,
 * Compare picker); `compact` hides secondary columns for overview panels.
 */
export function RunTable({ runs, compact, selectable, selected = [], onToggle, onToggleAll, maxHeight, envBy = {} }) {
  const [sort, setSort] = useState({ field: 'created_at', dir: 'desc' })
  const navigate = useNavigate()
  const rows = sortRows(runs, sort, ACCESSORS)
  const onSort = (f) => setSort((s) => nextSort(s, f))
  const allSelected = selectable && runs.length > 0 && runs.every((r) => selected.includes(r.id))

  return (
    <Table maxHeight={maxHeight}>
      <THead>
        {selectable && (
          <th scope="col" className="w-10 px-4">
            <input
              type="checkbox"
              aria-label="Select all runs"
              checked={allSelected}
              onChange={() => onToggleAll?.(allSelected ? [] : runs.map((r) => r.id))}
              className="h-4 w-4 accent-[rgb(var(--accent))]"
            />
          </th>
        )}
        <Th field="name" sort={sort} onSort={onSort}>
          Run
        </Th>
        {!compact && (
          <Th field="agent" sort={sort} onSort={onSort}>
            Algorithm
          </Th>
        )}
        {!compact && (
          <Th field="env" sort={sort} onSort={onSort}>
            Environment
          </Th>
        )}
        {!compact && (
          <Th field="seed" sort={sort} onSort={onSort} align="right">
            Seed
          </Th>
        )}
        <Th field="progress" sort={sort} onSort={onSort}>
          Episodes
        </Th>
        <Th field="reward" sort={sort} onSort={onSort} align="right">
          <Tooltip content="Mean return over the last 100 training episodes (live while running).">
            <span>Mean return</span>
          </Tooltip>
        </Th>
        {!compact && (
          <Th field="best_eval_reward" sort={sort} onSort={onSort} align="right">
            <Tooltip content="Best greedy evaluation score. Blank if the run ended before its first evaluation.">
              <span>Best eval</span>
            </Tooltip>
          </Th>
        )}
        <Th field="status" sort={sort} onSort={onSort}>
          Status
        </Th>
        {!compact && (
          <Th field="created_at" sort={sort} onSort={onSort} align="right">
            Started
          </Th>
        )}
      </THead>
      <tbody>
        {rows.map((r) => {
          const reward = r.final_mean_reward ?? r.progress?.running_reward
          const isSel = selected.includes(r.id)
          return (
            <Tr key={r.id} onClick={() => navigate(`/runs/${r.id}`)} selected={isSel}>
              {selectable && (
                <Td className="w-10">
                  <input
                    type="checkbox"
                    aria-label={`Select ${r.name}`}
                    checked={isSel}
                    onClick={(e) => e.stopPropagation()}
                    onChange={() => onToggle?.(r.id)}
                    className="h-4 w-4 accent-[rgb(var(--accent))]"
                  />
                </Td>
              )}
              <Td className="max-w-72">
                <Link
                  to={`/runs/${r.id}`}
                  onClick={(e) => e.stopPropagation()}
                  className="block truncate font-medium text-text hover:text-accent"
                >
                  {r.name}
                </Link>
                <span className="flex items-center gap-1.5 text-caption text-subtle">
                  {compact ? (
                    <span className="truncate">
                      {r.agent} · {envBy[r.env]?.label || r.env} · {fmtRelative(r.created_at)}
                    </span>
                  ) : (
                    <span className="num">{r.id}</span>
                  )}
                  {r.group_id && (
                    <span className="inline-flex items-center gap-1" title={`Seed group ${r.group_id}`}>
                      <Layers size={11} aria-hidden /> group
                    </span>
                  )}
                  {(r.tags || []).slice(0, 2).map((t) => (
                    <span key={t} className="rounded-sm bg-surface-3 px-1">
                      {t}
                    </span>
                  ))}
                </span>
              </Td>
              {!compact && <Td className="text-muted">{r.agent}</Td>}
              {!compact && <Td className="max-w-48 truncate text-muted">{envBy[r.env]?.label || r.env}</Td>}
              {!compact && (
                <Td align="right" mono className="text-muted">
                  {r.seed}
                </Td>
              )}
              <Td>
                <ProgressCell run={r} />
              </Td>
              <Td
                align="right"
                mono
                className={cn(reward === null || reward === undefined ? 'text-subtle' : 'text-text')}
              >
                {fmtNum(reward)}
              </Td>
              {!compact && (
                <Td align="right" mono className="text-muted">
                  {fmtNum(r.best_eval_reward)}
                </Td>
              )}
              <Td>
                <StatusBadge status={r.status} />
              </Td>
              {!compact && (
                <Td align="right" className="whitespace-nowrap text-caption text-subtle">
                  {fmtRelative(r.created_at)}
                </Td>
              )}
            </Tr>
          )
        })}
      </tbody>
    </Table>
  )
}
