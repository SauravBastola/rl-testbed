import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Map } from 'lucide-react'
import { useRouteLab } from '../app/AppContext.jsx'
import { useCatalogue } from '../lib/queries.js'
import { cn } from '../lib/utils.js'
import { Button } from '../components/ui/Button.jsx'
import { SearchInput, Switch } from '../components/ui/Field.jsx'
import { EmptyState, ErrorState, Skeleton } from '../components/ui/Feedback.jsx'
import { PageHeader } from '../components/ui/Page.jsx'
import { EnvCard, categoryLabel } from '../components/domain/CatalogCards.jsx'

const FACETS = [
  { key: 'category', label: 'Family', get: (e) => e.category, fmt: categoryLabel },
  {
    key: 'difficulty',
    label: 'Difficulty',
    get: (e) => e.difficulty,
    fmt: (v) => v.charAt(0).toUpperCase() + v.slice(1),
  },
  {
    key: 'obs',
    label: 'Observation',
    get: (e) => e.obs_type,
    fmt: (v) => ({ onehot: 'One-hot state', vector: 'Vector', image: 'Pixels' })[v] || v,
  },
  { key: 'action', label: 'Action', get: (e) => e.action_type, fmt: (v) => v.charAt(0).toUpperCase() + v.slice(1) },
  {
    key: 'avail',
    label: 'Availability',
    get: (e) => (e.available === false ? 'missing' : 'ready'),
    fmt: (v) => (v === 'ready' ? 'Ready to train' : 'Needs install'),
  },
]

function FacetGroup({ facet, envs, value, onChange }) {
  const counts = envs.reduce((acc, e) => {
    const v = facet.get(e)
    acc[v] = (acc[v] || 0) + 1
    return acc
  }, {})
  const options = Object.entries(counts).sort((a, b) => b[1] - a[1])
  if (options.length < 2 && !value) return null
  return (
    <fieldset>
      <legend className="label mb-1.5">{facet.label}</legend>
      <div className="flex flex-wrap gap-1 lg:flex-col lg:gap-0.5">
        {options.map(([v, n]) => {
          const on = value === v
          return (
            <button
              key={v}
              type="button"
              aria-pressed={on}
              onClick={() => onChange(on ? '' : v)}
              className={cn(
                'flex h-8 items-center justify-between gap-3 rounded px-2.5 text-body-s transition-colors duration-fast',
                on ? 'bg-accent/12 text-text' : 'text-muted hover:bg-surface-2 hover:text-text',
              )}
            >
              <span className="truncate">{facet.fmt(v)}</span>
              <span className={cn('num text-caption', on ? 'text-accent' : 'text-subtle')}>{n}</span>
            </button>
          )
        })}
      </div>
    </fieldset>
  )
}

export default function Environments() {
  const { labId, lab } = useRouteLab()
  const { envs, labBy, isLoading, error, refetch } = useCatalogue()
  const [params, setParams] = useSearchParams()
  const all = params.get('all') === '1'
  const q = params.get('q') || ''

  const set = (k, v) => {
    const next = new URLSearchParams(params)
    if (!v) next.delete(k)
    else next.set(k, v === true ? '1' : v)
    setParams(next, { replace: true })
  }

  const scope = useMemo(
    () => envs.filter((e) => e.category !== 'probe' && (all || e.lab === labId)),
    [envs, all, labId],
  )
  const tokens = q.toLowerCase().split(/\s+/).filter(Boolean)
  const shown = scope.filter(
    (e) =>
      FACETS.every((f) => !params.get(f.key) || f.get(e) === params.get(f.key)) &&
      tokens.every((t) =>
        `${e.label} ${e.name} ${e.description} ${(e.tags || []).join(' ')}`.toLowerCase().includes(t),
      ),
  )
  const sections = shown.reduce((acc, e) => {
    ;(acc[e.category] = acc[e.category] || []).push(e)
    return acc
  }, {})
  const active = FACETS.some((f) => params.get(f.key)) || q

  if (error) return <ErrorState error={error} onRetry={refetch} title="Could not load environments" />

  return (
    <div>
      <PageHeader
        title="Environments"
        description={
          all
            ? 'Every environment in the testbed.'
            : `The worlds in ${lab?.label || 'this lab'}: ${lab?.tagline?.toLowerCase() || ''}.`
        }
        actions={
          <label className="flex items-center gap-2 text-body-s text-muted">
            <Switch checked={all} onChange={(v) => set('all', v)} label="Show all labs" />
            All labs
          </label>
        }
      />
      <div className="grid gap-6 lg:grid-cols-[208px_minmax(0,1fr)]">
        <aside className="space-y-5 lg:sticky lg:top-20 lg:self-start">
          <SearchInput value={q} onChange={(v) => set('q', v)} placeholder="Search environments" />
          {FACETS.map((f) => (
            <FacetGroup
              key={f.key}
              facet={f}
              envs={scope}
              value={params.get(f.key) || ''}
              onChange={(v) => set(f.key, v)}
            />
          ))}
          {active && (
            <Button size="sm" variant="ghost" onClick={() => setParams(all ? { all: '1' } : {}, { replace: true })}>
              Clear filters
            </Button>
          )}
        </aside>

        <div className="min-w-0 space-y-8">
          <p className="text-body-s text-muted" aria-live="polite">
            <span className="num text-text">{shown.length}</span> of {scope.length} environments
          </p>
          {isLoading ? (
            <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-44" />
              ))}
            </div>
          ) : !shown.length ? (
            <EmptyState
              icon={Map}
              title="No environments match"
              description={all ? 'Try removing a filter.' : 'Try removing a filter, or include every lab.'}
              action={
                <Button size="sm" onClick={() => setParams({ all: '1' }, { replace: true })}>
                  Search all labs
                </Button>
              }
            />
          ) : (
            Object.entries(sections).map(([cat, list]) => (
              <section key={cat} aria-label={categoryLabel(cat)}>
                <h2 className="mb-3 flex items-baseline gap-2 text-title-s font-semibold">
                  {categoryLabel(cat)} <span className="num text-caption font-normal text-subtle">{list.length}</span>
                </h2>
                <div className="grid auto-rows-fr gap-4 sm:grid-cols-2 2xl:grid-cols-3">
                  {list.map((e) => (
                    <EnvCard key={e.name} env={e} labIcon={labBy[e.lab]?.icon || 'pulse'} />
                  ))}
                </div>
              </section>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
