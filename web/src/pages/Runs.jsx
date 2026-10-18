import { useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { GitCompareArrows, ListTree, Plus, Square, Trash2 } from 'lucide-react'
import { useRouteLab } from '../app/AppContext.jsx'
import { api } from '../lib/api.js'
import { useCatalogue, useRunMutation, useRuns } from '../lib/queries.js'
import { STATUS, isActive, plural } from '../lib/format.js'
import { Button, ButtonLink } from '../components/ui/Button.jsx'
import { SearchInput, Switch } from '../components/ui/Field.jsx'
import { Chip } from '../components/ui/Segmented.jsx'
import { EmptyState, ErrorState, SkeletonRows } from '../components/ui/Feedback.jsx'
import { PageHeader } from '../components/ui/Page.jsx'
import { Modal } from '../components/ui/Overlay.jsx'
import { useToast } from '../components/ui/Toast.jsx'
import { RunTable } from '../components/domain/RunTable.jsx'

const FILTERS = ['all', 'active', 'completed', 'failed', 'stopped']
const matches = (filter, r) =>
  filter === 'all'
    ? true
    : filter === 'active'
      ? isActive(r.status)
      : filter === 'failed'
        ? ['failed', 'interrupted'].includes(r.status)
        : r.status === filter

export default function Runs() {
  const { labId, lab } = useRouteLab()
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()
  const { toast } = useToast()
  const { envBy } = useCatalogue()
  const allLabs = params.get('all') === '1'
  const status = params.get('status') || 'all'
  const query = params.get('q') || ''
  const runs = useRuns(allLabs ? {} : { lab: labId })
  const [selected, setSelected] = useState([])
  const [confirm, setConfirm] = useState(false)

  const set = (k, v) => {
    const next = new URLSearchParams(params)
    if (v === undefined || v === '' || v === 'all' || v === false) next.delete(k)
    else next.set(k, v === true ? '1' : v)
    setParams(next, { replace: true })
  }

  const list = useMemo(() => runs.data || [], [runs.data])
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean)
  const shown = useMemo(
    () =>
      list.filter(
        (r) =>
          matches(status, r) &&
          tokens.every((t) =>
            `${r.name} ${r.id} ${r.agent} ${r.env} ${(r.tags || []).join(' ')}`.toLowerCase().includes(t),
          ),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- tokens derive from `query`
    [list, status, query],
  )
  const counts = Object.fromEntries(FILTERS.map((f) => [f, list.filter((r) => matches(f, r)).length]))

  const del = useRunMutation(async (ids) => Promise.all(ids.map((id) => api.deleteRun(id))), {
    onSuccess: (_, ids) => {
      toast({ title: `Deleted ${plural(ids.length, 'run')}` })
      setSelected([])
      setConfirm(false)
    },
    onError: (e) => toast({ title: 'Delete failed', description: e.message, tone: 'danger' }),
  })
  const stop = useRunMutation(async (ids) => Promise.all(ids.map((id) => api.stopRun(id))), {
    onSuccess: (_, ids) => toast({ title: `Stopping ${plural(ids.length, 'run')}`, tone: 'info' }),
  })
  const activeSelected = selected.filter((id) => isActive(list.find((r) => r.id === id)?.status))

  return (
    <div>
      <PageHeader
        title="Runs"
        description={allLabs ? 'Every run across all labs.' : `Runs on ${lab?.label || 'this lab'} environments.`}
        actions={
          <ButtonLink to={`/lab/${labId}/new`} variant="primary" icon={Plus}>
            New run
          </ButtonLink>
        }
      />

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by status">
          {FILTERS.map((f) => (
            <Chip key={f} active={status === f} onClick={() => set('status', f)} count={counts[f]}>
              {f === 'all' ? 'All' : f === 'active' ? 'Active' : STATUS[f]?.label || f}
            </Chip>
          ))}
        </div>
        <label className="flex items-center gap-2 text-caption text-muted">
          <Switch checked={allLabs} onChange={(v) => set('all', v)} label="Show runs from all labs" />
          All labs
        </label>
        <SearchInput
          value={query}
          onChange={(v) => set('q', v)}
          placeholder="Search name, id, tag"
          className="ml-auto w-full sm:w-72"
        />
      </div>

      {selected.length > 0 && (
        <div className="mb-3 flex flex-wrap items-center gap-2 rounded-lg border border-accent/30 bg-accent/[.06] px-4 py-2">
          <span className="text-body-s text-text">{plural(selected.length, 'run')} selected</span>
          <div className="ml-auto flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="primary"
              icon={GitCompareArrows}
              onClick={() => navigate(`/compare?runs=${selected.join(',')}`)}
            >
              Compare
            </Button>
            {activeSelected.length > 0 && (
              <Button size="sm" icon={Square} loading={stop.isPending} onClick={() => stop.mutate(activeSelected)}>
                Stop {activeSelected.length}
              </Button>
            )}
            <Button size="sm" variant="danger" icon={Trash2} onClick={() => setConfirm(true)}>
              Delete
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setSelected([])}>
              Clear
            </Button>
          </div>
        </div>
      )}

      {runs.error ? (
        <ErrorState error={runs.error} onRetry={runs.refetch} title="Could not load runs" />
      ) : runs.isLoading ? (
        <SkeletonRows rows={8} />
      ) : !list.length ? (
        <EmptyState
          icon={ListTree}
          title={allLabs ? 'No runs yet' : `No runs in ${lab?.label || 'this lab'} yet`}
          description="Runs appear here the moment they launch, with live progress."
          action={
            <ButtonLink to={`/lab/${labId}/new`} variant="primary" icon={Plus}>
              Start a run
            </ButtonLink>
          }
        />
      ) : !shown.length ? (
        <EmptyState
          compact
          icon={ListTree}
          title="No runs match these filters"
          action={
            <Button size="sm" onClick={() => setParams({}, { replace: true })}>
              Clear filters
            </Button>
          }
        />
      ) : (
        <RunTable
          runs={shown}
          envBy={envBy}
          selectable
          selected={selected}
          onToggle={(id) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]))}
          onToggleAll={setSelected}
          maxHeight="calc(100vh - 260px)"
        />
      )}

      <Modal
        open={confirm}
        onClose={() => setConfirm(false)}
        title={`Delete ${plural(selected.length, 'run')}?`}
        description="Their artifacts, logs and history are removed from disk. Kept models are not affected."
        size="sm"
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirm(false)}>
              Cancel
            </Button>
            <Button variant="danger" icon={Trash2} loading={del.isPending} onClick={() => del.mutate(selected)}>
              Delete
            </Button>
          </>
        }
      >
        <p className="text-body-s text-muted">Running runs are stopped first.</p>
      </Modal>
    </div>
  )
}
