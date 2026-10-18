import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { FlaskConical } from 'lucide-react'
import { useRouteLab } from '../app/AppContext.jsx'
import { useCatalogue, useCompatMatrix } from '../lib/queries.js'
import { SearchInput } from '../components/ui/Field.jsx'
import { Chip } from '../components/ui/Segmented.jsx'
import { EmptyState, ErrorState, Skeleton } from '../components/ui/Feedback.jsx'
import { PageHeader } from '../components/ui/Page.jsx'
import { Section } from '../components/ui/Card.jsx'
import { AlgoCard, familyLabel } from '../components/domain/CatalogCards.jsx'

export default function Algorithms() {
  const { labId, lab } = useRouteLab()
  const { agents, envs, isLoading, error, refetch } = useCatalogue()
  const compat = useCompatMatrix()
  const [params, setParams] = useSearchParams()
  const family = params.get('family') || ''
  const q = params.get('q') || ''
  const set = (k, v) => {
    const next = new URLSearchParams(params)
    if (v) next.set(k, v)
    else next.delete(k)
    setParams(next, { replace: true })
  }

  const labEnvs = useMemo(() => envs.filter((e) => e.lab === labId && e.available !== false), [envs, labId])
  const compatCount = (a) => labEnvs.filter((e) => compat.lookup(a.name, e.name)?.compatible).length
  const tokens = q.toLowerCase().split(/\s+/).filter(Boolean)
  const filtered = agents.filter(
    (a) =>
      (!family || a.category === family) &&
      tokens.every((t) => `${a.label} ${a.name} ${a.description} ${a.category}`.toLowerCase().includes(t)),
  )
  const families = Object.entries(agents.reduce((acc, a) => ({ ...acc, [a.category]: (acc[a.category] || 0) + 1 }), {}))

  const priority = lab?.algorithms || []
  const implemented = filtered.filter((a) => a.status === 'implemented')
  const suited = implemented
    .filter((a) => priority.includes(a.name))
    .sort((a, b) => priority.indexOf(a.name) - priority.indexOf(b.name))
  const alsoHere = implemented.filter((a) => !priority.includes(a.name) && compatCount(a) > 0)
  const elsewhere = implemented.filter((a) => !priority.includes(a.name) && compatCount(a) === 0)
  const planned = filtered.filter((a) => a.status !== 'implemented')

  if (error) return <ErrorState error={error} onRetry={refetch} title="Could not load algorithms" />

  const grid = (list, recommended) => (
    <div className="grid auto-rows-fr gap-4 sm:grid-cols-2 2xl:grid-cols-3">
      {list.map((a) => (
        <AlgoCard
          key={a.name}
          agent={a}
          recommended={recommended}
          compatCount={compat.isLoading ? undefined : compatCount(a)}
          actionTo={`/lab/${labId}/new?agent=${a.name}`}
        />
      ))}
    </div>
  )

  return (
    <div>
      <PageHeader
        title="Algorithms"
        description={`${agents.filter((a) => a.status === 'implemented').length} trainable methods and ${agents.filter((a) => a.status !== 'implemented').length} planned, ordered by how well they suit ${lab?.label || 'this lab'}.`}
      />
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by family">
          <Chip active={!family} onClick={() => set('family', '')} count={agents.length}>
            All
          </Chip>
          {families.map(([f, n]) => (
            <Chip key={f} active={family === f} onClick={() => set('family', family === f ? '' : f)} count={n}>
              {familyLabel(f)}
            </Chip>
          ))}
        </div>
        <SearchInput
          value={q}
          onChange={(v) => set('q', v)}
          placeholder="Search algorithms"
          className="ml-auto w-full sm:w-64"
        />
      </div>

      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-44" />
          ))}
        </div>
      ) : !filtered.length ? (
        <EmptyState
          icon={FlaskConical}
          title="No algorithms match"
          description="Clear the search or pick another family."
        />
      ) : (
        <div className="space-y-10">
          {suited.length > 0 && (
            <Section
              title={`Suited to ${lab?.label || 'this lab'}`}
              description="The methods this lab’s curriculum is built around."
            >
              {grid(suited, true)}
            </Section>
          )}
          {alsoHere.length > 0 && (
            <Section
              title="Also trainable here"
              description="Compatible with at least one environment in this lab, but not its focus."
            >
              {grid(alsoHere)}
            </Section>
          )}
          {elsewhere.length > 0 && (
            <Section
              title="Built for other labs"
              description="No installed environment in this lab fits their observation or action spaces."
            >
              {grid(elsewhere)}
            </Section>
          )}
          {planned.length > 0 && (
            <Section title="Planned" description="Documented but not yet trainable. Each card says what is missing.">
              {grid(planned)}
            </Section>
          )}
        </div>
      )}
    </div>
  )
}
