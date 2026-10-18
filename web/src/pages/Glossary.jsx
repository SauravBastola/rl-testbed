import { useEffect, useMemo, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { BookMarked } from 'lucide-react'
import { useCatalogue, useGlossary } from '../lib/queries.js'
import { cn } from '../lib/utils.js'
import { SearchInput } from '../components/ui/Field.jsx'
import { Chip } from '../components/ui/Segmented.jsx'
import { EmptyState, ErrorState, SkeletonRows } from '../components/ui/Feedback.jsx'
import { PageHeader } from '../components/ui/Page.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { MathText } from '../components/domain/MathText.jsx'

function Entry({ g, termBy, agentBy, envBy, highlighted, showCategory }) {
  const seeIt = [
    ...g.algorithms.map((a) => ({ to: `/algorithms/${a}`, label: agentBy[a]?.label || a })),
    ...g.envs.map((e) => ({ to: `/envs/${e}`, label: envBy[e]?.label || e })),
  ]
  return (
    <article
      id={g.id}
      className={cn(
        'scroll-mt-24 rounded-lg border bg-surface p-5 transition-[border-color,box-shadow] duration-slow',
        highlighted ? 'border-accent shadow-focus' : 'border-border',
      )}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-title-s font-semibold text-text">
          <a href={`#${g.id}`} className="hover:text-accent">
            {g.term}
          </a>
        </h3>
        {showCategory && <Badge>{g.category}</Badge>}
      </div>
      <p className="mt-1 text-body text-text">{g.definition}</p>
      <MathText source={g.detail} className="prose-lab mt-2 text-body-s" />
      {g.formula && (
        <div className="mt-3 overflow-x-auto rounded border border-border bg-bg-soft px-4 py-3 text-text">
          <MathText source={g.formula} />
        </div>
      )}
      {(g.related.length > 0 || seeIt.length > 0) && (
        <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2 text-caption">
          {g.related.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-subtle">Related</span>
              {g.related.map((r) => (
                <a
                  key={r}
                  href={`#${r}`}
                  className="rounded-full border border-border bg-surface-2 px-2.5 py-0.5 text-muted hover:border-border-strong hover:text-text"
                >
                  {termBy[r]?.term || r}
                </a>
              ))}
            </div>
          )}
          {seeIt.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-subtle">See it in</span>
              {seeIt.map((s) => (
                <Link key={s.to} to={s.to} className="text-accent hover:underline">
                  {s.label}
                </Link>
              ))}
            </div>
          )}
        </div>
      )}
    </article>
  )
}

/** Every term the testbed uses, grouped by topic, searchable and cross-linked. */
export default function Glossary() {
  const g = useGlossary()
  const { agentBy, envBy } = useCatalogue()
  const { hash } = useLocation()
  const [q, setQ] = useState('')
  const [category, setCategory] = useState('')
  const [highlight, setHighlight] = useState('')
  const all = useMemo(() => g.data || [], [g.data])
  const termBy = useMemo(() => Object.fromEntries(all.map((x) => [x.id, x])), [all])
  const categories = useMemo(() => [...new Set(all.map((x) => x.category))], [all])

  // Deep links (#td-error) from the learning path or a related-term chip.
  useEffect(() => {
    const id = decodeURIComponent(hash.replace('#', ''))
    if (!id || !termBy[id]) return undefined
    setQ('')
    setCategory('')
    setHighlight(id)
    const t1 = setTimeout(() => document.getElementById(id)?.scrollIntoView({ block: 'start', behavior: 'smooth' }), 50)
    const t2 = setTimeout(() => setHighlight(''), 2400)
    return () => {
      clearTimeout(t1)
      clearTimeout(t2)
    }
  }, [hash, termBy])

  const tokens = q.toLowerCase().split(/\s+/).filter(Boolean)
  const shown = all.filter(
    (x) =>
      (!category || x.category === category) &&
      tokens.every((t) => `${x.term} ${x.definition} ${x.detail}`.toLowerCase().includes(t)),
  )
  const searching = tokens.length > 0

  return (
    <div>
      <PageHeader
        title="Glossary"
        description={`${all.length || 'The'} terms used across the testbed, each with a short definition, a fuller explanation, the formula where it helps, and where to see it in action.`}
        actions={<SearchInput value={q} onChange={setQ} placeholder="Search terms" className="w-64" />}
      />
      <div className="mb-6 flex flex-wrap gap-1.5" role="group" aria-label="Filter by topic">
        <Chip active={!category} onClick={() => setCategory('')} count={all.length}>
          All
        </Chip>
        {categories.map((c) => (
          <Chip
            key={c}
            active={category === c}
            onClick={() => setCategory(category === c ? '' : c)}
            count={all.filter((x) => x.category === c).length}
          >
            {c}
          </Chip>
        ))}
      </div>
      {g.error ? (
        <ErrorState error={g.error} onRetry={g.refetch} />
      ) : g.isLoading ? (
        <SkeletonRows rows={8} />
      ) : !shown.length ? (
        <EmptyState
          icon={BookMarked}
          title={`No term matches “${q}”`}
          description="Try a shorter word, or clear the topic filter."
        />
      ) : searching ? (
        <div className="space-y-3">
          {shown.map((x) => (
            <Entry
              key={x.id}
              g={x}
              termBy={termBy}
              agentBy={agentBy}
              envBy={envBy}
              highlighted={highlight === x.id}
              showCategory
            />
          ))}
        </div>
      ) : (
        <div className="space-y-10">
          {categories
            .filter((c) => shown.some((x) => x.category === c))
            .map((c) => (
              <section key={c} aria-label={c}>
                <h2 className="mb-3 text-title font-semibold">{c}</h2>
                <div className="grid gap-3 xl:grid-cols-2">
                  {shown
                    .filter((x) => x.category === c)
                    .map((x) => (
                      <Entry
                        key={x.id}
                        g={x}
                        termBy={termBy}
                        agentBy={agentBy}
                        envBy={envBy}
                        highlighted={highlight === x.id}
                      />
                    ))}
                </div>
              </section>
            ))}
        </div>
      )}
    </div>
  )
}
