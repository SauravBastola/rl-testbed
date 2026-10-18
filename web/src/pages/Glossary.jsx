import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { BookMarked } from 'lucide-react'
import { useCatalogue, useGlossary } from '../lib/queries.js'
import { SearchInput } from '../components/ui/Field.jsx'
import { EmptyState, ErrorState, SkeletonRows } from '../components/ui/Feedback.jsx'
import { PageHeader } from '../components/ui/Page.jsx'

export default function Glossary() {
  const g = useGlossary()
  const { agentBy } = useCatalogue()
  const [q, setQ] = useState('')
  const terms = useMemo(() => {
    const t = q.toLowerCase().trim()
    return [...(g.data || [])]
      .sort((a, b) => a.term.localeCompare(b.term))
      .filter((x) => !t || `${x.term} ${x.definition}`.toLowerCase().includes(t))
  }, [g.data, q])
  const byLetter = terms.reduce((acc, x) => {
    const l = x.term[0].toUpperCase()
    ;(acc[l] = acc[l] || []).push(x)
    return acc
  }, {})

  return (
    <div>
      <PageHeader
        title="Glossary"
        description="The vocabulary used across the testbed, in plain language."
        actions={<SearchInput value={q} onChange={setQ} placeholder="Find a term" className="w-64" />}
      />
      {g.error ? (
        <ErrorState error={g.error} onRetry={g.refetch} />
      ) : g.isLoading ? (
        <SkeletonRows rows={8} />
      ) : !terms.length ? (
        <EmptyState icon={BookMarked} title={`No term matches “${q}”`} />
      ) : (
        <div className="columns-1 gap-8 lg:columns-2">
          {Object.entries(byLetter).map(([letter, list]) => (
            <section key={letter} className="mb-6 break-inside-avoid">
              <h2 className="mb-2 font-display text-title text-accent">{letter}</h2>
              <dl className="space-y-3">
                {list.map((x) => (
                  <div key={x.term} id={x.term.toLowerCase().replace(/\W+/g, '-')}>
                    <dt className="font-semibold text-text">{x.term}</dt>
                    <dd className="mt-0.5 text-body-s text-muted">
                      {x.definition}
                      {x.algorithm && agentBy[x.algorithm] && (
                        <>
                          {' '}
                          <Link to={`/algorithms/${x.algorithm}`} className="text-accent hover:underline">
                            See {agentBy[x.algorithm].label}
                          </Link>
                        </>
                      )}
                    </dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
      )}
    </div>
  )
}
