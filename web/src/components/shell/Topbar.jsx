import { Link, useLocation } from 'react-router-dom'
import { ChevronRight, Menu } from 'lucide-react'
import { useLab } from '../../app/AppContext.jsx'
import { IconButton } from '../ui/Button.jsx'
import { LabGlyph } from '../art/LabArt.jsx'

const PAGE_NAMES = {
  new: 'New run',
  runs: 'Runs',
  sweeps: 'Sweeps',
  envs: 'Environments',
  algorithms: 'Algorithms',
  compare: 'Compare',
  experiments: 'Experiments',
  models: 'Models',
  visualize: 'Visualize',
  diagnostics: 'Diagnostics',
  learn: 'Learning path',
  glossary: 'Glossary',
}

/** Breadcrumbs derived from the URL: lab segment first, then page, then entity. */
export function useCrumbs() {
  const { pathname } = useLocation()
  const { lab, labId } = useLab()
  const parts = pathname.split('/').filter(Boolean)
  const labCrumb = { label: lab?.label || 'Lab', to: `/lab/${labId}`, glyph: lab?.icon }
  if (parts[0] === 'lab') {
    const crumbs = [labCrumb]
    if (parts[2] === 'sweeps' && parts[3] === 'new') {
      crumbs.push({ label: 'Sweeps', to: `/lab/${labId}/sweeps` }, { label: 'New sweep' })
    } else if (parts[2]) crumbs.push({ label: PAGE_NAMES[parts[2]] || parts[2] })
    else crumbs.push({ label: 'Overview' })
    return crumbs
  }
  if (parts[0] === 'runs' && parts[1])
    return [labCrumb, { label: 'Runs', to: `/lab/${labId}/runs` }, { label: parts[1], mono: true }]
  if (parts[0] === 'sweeps' && parts[1])
    return [labCrumb, { label: 'Sweeps', to: `/lab/${labId}/sweeps` }, { label: parts[1], mono: true }]
  if (parts[0] === 'envs' && parts[1])
    return [labCrumb, { label: 'Environments', to: `/lab/${labId}/envs` }, { label: parts[1], mono: true }]
  if (parts[0] === 'algorithms' && parts[1])
    return [labCrumb, { label: 'Algorithms', to: `/lab/${labId}/algorithms` }, { label: parts[1], mono: true }]
  if (parts[0]) return [{ label: PAGE_NAMES[parts[0]] || 'Page' }]
  return [labCrumb]
}

export function Topbar({ onMenu }) {
  const crumbs = useCrumbs()
  return (
    <div className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b border-border/70 px-4 surface-glass lg:px-8">
      <IconButton icon={Menu} label="Open navigation" className="lg:hidden" onClick={onMenu} />
      <nav aria-label="Breadcrumb" className="min-w-0 flex-1">
        <ol className="flex min-w-0 items-center gap-1.5 text-body-s">
          {crumbs.map((c, i) => (
            <li key={`${c.label}${i}`} className="flex min-w-0 items-center gap-1.5">
              {i > 0 && <ChevronRight size={14} className="shrink-0 text-subtle" aria-hidden />}
              {c.to && i < crumbs.length - 1 ? (
                <Link to={c.to} className="flex items-center gap-1.5 truncate text-muted hover:text-text">
                  {c.glyph && <LabGlyph icon={c.glyph} size={14} className="text-accent" />}
                  {c.label}
                </Link>
              ) : (
                <span
                  className={c.mono ? 'num truncate text-text' : 'truncate text-text'}
                  aria-current={i === crumbs.length - 1 ? 'page' : undefined}
                >
                  {c.label}
                </span>
              )}
            </li>
          ))}
        </ol>
      </nav>
    </div>
  )
}
