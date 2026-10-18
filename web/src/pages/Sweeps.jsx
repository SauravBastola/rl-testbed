import { Link, useNavigate } from 'react-router-dom'
import { Plus, SlidersHorizontal } from 'lucide-react'
import { useRouteLab } from '../app/AppContext.jsx'
import { useCatalogue, useSweeps } from '../lib/queries.js'
import { fmtNum, fmtRelative } from '../lib/format.js'
import { ButtonLink } from '../components/ui/Button.jsx'
import { EmptyState, ErrorState, SkeletonRows } from '../components/ui/Feedback.jsx'
import { PageHeader, Progress } from '../components/ui/Page.jsx'
import { StatusBadge } from '../components/ui/Badge.jsx'
import { Table, THead, Th, Tr, Td } from '../components/ui/Table.jsx'

/** A sweep's overall status, derived from its runs. */
export function sweepStatus(s) {
  if (s.running) return 'running'
  if (s.queued) return 'queued'
  if (s.failed && !s.completed) return 'failed'
  if (s.stopped && s.completed < s.n_runs) return 'stopped'
  return 'completed'
}

export default function Sweeps() {
  const { labId, lab } = useRouteLab()
  const sweeps = useSweeps(labId)
  const { envBy, agentBy } = useCatalogue()
  const navigate = useNavigate()
  const list = sweeps.data || []

  return (
    <div>
      <PageHeader
        title="Sweeps"
        description={`Hyperparameter searches on ${lab?.label || 'this lab'} environments, each configuration run over several seeds.`}
        actions={
          <ButtonLink to={`/lab/${labId}/sweeps/new`} variant="primary" icon={Plus}>
            New sweep
          </ButtonLink>
        }
      />
      {sweeps.error ? (
        <ErrorState error={sweeps.error} onRetry={sweeps.refetch} />
      ) : sweeps.isLoading ? (
        <SkeletonRows rows={5} />
      ) : !list.length ? (
        <EmptyState
          icon={SlidersHorizontal}
          title="No sweeps in this lab yet"
          description="A sweep answers “which settings work best?”: pick the parameters to vary, and every combination runs with several seeds so the ranking reflects more than luck."
          action={
            <ButtonLink to={`/lab/${labId}/sweeps/new`} variant="primary" icon={Plus}>
              Start a sweep
            </ButtonLink>
          }
        />
      ) : (
        <Table>
          <THead>
            <Th>Sweep</Th>
            <Th>Varies</Th>
            <Th>Runs</Th>
            <Th align="right">Best eval</Th>
            <Th>Status</Th>
            <Th align="right">Started</Th>
          </THead>
          <tbody>
            {list.map((s) => (
              <Tr key={s.id} onClick={() => navigate(`/sweeps/${s.id}`)}>
                <Td className="max-w-80">
                  <Link
                    to={`/sweeps/${s.id}`}
                    onClick={(e) => e.stopPropagation()}
                    className="block truncate font-medium text-text hover:text-accent"
                  >
                    {s.name}
                  </Link>
                  <span className="block truncate text-caption text-subtle">
                    {agentBy[s.agent]?.label || s.agent} on {envBy[s.env]?.label || s.env} · {s.strategy}
                  </span>
                </Td>
                <Td>
                  <span className="flex flex-wrap gap-1">
                    {s.swept.map((k) => (
                      <code key={k} className="rounded-sm bg-surface-3 px-1.5 text-caption text-muted">
                        {k}
                      </code>
                    ))}
                  </span>
                </Td>
                <Td>
                  <div className="flex min-w-32 items-center gap-2">
                    <Progress value={s.n_runs ? (s.completed + s.failed + s.stopped) / s.n_runs : 0} className="w-16" />
                    <span className="num text-caption text-muted">
                      {s.completed + s.failed + s.stopped}/{s.n_runs}
                    </span>
                  </div>
                </Td>
                <Td align="right" mono>
                  {fmtNum(s.best_eval)}
                </Td>
                <Td>
                  <StatusBadge status={sweepStatus(s)} />
                </Td>
                <Td align="right" className="whitespace-nowrap text-caption text-subtle">
                  {fmtRelative(s.created_at)}
                </Td>
              </Tr>
            ))}
          </tbody>
        </Table>
      )}
    </div>
  )
}
