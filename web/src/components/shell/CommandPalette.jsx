import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowRight,
  BookMarked,
  Boxes,
  CornerDownLeft,
  Eye,
  FileCode2,
  FlaskConical,
  GitCompareArrows,
  GraduationCap,
  LayoutDashboard,
  ListTree,
  Map,
  Moon,
  Play,
  Plus,
  Search,
  SlidersHorizontal,
  Stethoscope,
} from 'lucide-react'
import { useApp, useLab } from '../../app/AppContext.jsx'
import { useCatalogue, useRuns } from '../../lib/queries.js'
import { THEMES } from '../../lib/theme.js'
import { cn } from '../../lib/utils.js'
import { LabGlyph } from '../art/LabArt.jsx'
import { Kbd } from '../ui/Page.jsx'
import { StatusBadge } from '../ui/Badge.jsx'
import { useFocusTrap } from '../ui/Overlay.jsx'

/** Every whitespace-separated token must appear somewhere in the haystack. */
function score(haystack, tokens) {
  if (!tokens.length) return 1
  const h = haystack.toLowerCase()
  let s = 0
  for (const t of tokens) {
    const i = h.indexOf(t)
    if (i < 0) return 0
    s += i === 0 || h[i - 1] === ' ' ? 3 : 1
  }
  return s
}

export function CommandPalette({ open, onClose }) {
  const navigate = useNavigate()
  const { setTheme } = useApp()
  const { labId, labs } = useLab()
  const { envs, agents } = useCatalogue()
  const runs = useRuns({ limit: 60 })
  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(0)
  const ref = useRef(null)
  const list = useRef(null)
  useFocusTrap(ref, open, onClose)

  useEffect(() => {
    if (open) {
      setQuery('')
      setIndex(0)
    }
  }, [open])

  const items = useMemo(() => {
    const base = `/lab/${labId}`
    const pages = [
      ['Overview', base, LayoutDashboard],
      ['New run', `${base}/new`, Plus],
      ['Runs', `${base}/runs`, ListTree],
      ['Sweeps', `${base}/sweeps`, SlidersHorizontal],
      ['New sweep', `${base}/sweeps/new`, SlidersHorizontal],
      ['Environments', `${base}/envs`, Map],
      ['Algorithms', `${base}/algorithms`, FlaskConical],
      ['Compare runs', '/compare', GitCompareArrows],
      ['Experiment files', '/experiments', FileCode2],
      ['Models', '/models', Boxes],
      ['Visualize a policy', '/visualize', Eye],
      ['Diagnostics', '/diagnostics', Stethoscope],
      ['Learning path', '/learn', GraduationCap],
      ['Glossary', '/glossary', BookMarked],
    ].map(([label, to, icon]) => ({ group: 'Go to', label, icon, run: () => navigate(to), key: `p${to}` }))
    const labItems = labs.map((l) => ({
      group: 'Labs',
      label: `${l.label} lab`,
      hint: l.tagline,
      glyph: l.icon,
      key: `l${l.id}`,
      run: () => navigate(`/lab/${l.id}`),
    }))
    const actions = [
      ...THEMES.map((t) => ({
        group: 'Actions',
        label: `Use ${t.label} theme`,
        hint: t.description,
        icon: Moon,
        key: `t${t.id}`,
        run: () => setTheme(t.id),
      })),
    ]
    const envItems = envs.map((e) => ({
      group: 'Environments',
      label: e.label,
      hint: `${e.name} · ${e.category}`,
      icon: Map,
      key: `e${e.name}`,
      run: () => navigate(`/envs/${e.name}`),
      extra: {
        label: 'Train',
        run: () => navigate(`/lab/${e.lab === 'diagnostics' ? labId : e.lab}/new?env=${e.name}`),
      },
    }))
    const agentItems = agents.map((a) => ({
      group: 'Algorithms',
      label: a.label,
      hint: `${a.name} · ${a.category}${a.status !== 'implemented' ? ' · planned' : ''}`,
      icon: FlaskConical,
      key: `a${a.name}`,
      run: () => navigate(`/algorithms/${a.name}`),
    }))
    const runItems = (runs.data || []).map((r) => ({
      group: 'Runs',
      label: r.name,
      hint: `${r.id} · ${r.agent} on ${r.env}`,
      status: r.status,
      key: `r${r.id}`,
      run: () => navigate(`/runs/${r.id}`),
    }))
    return [...pages, ...labItems, ...actions, ...runItems, ...envItems, ...agentItems]
  }, [labId, labs, envs, agents, runs.data, navigate, setTheme])

  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean)
  const results = useMemo(() => {
    const scored = items
      .map((it) => ({ it, s: score(`${it.label} ${it.hint || ''} ${it.group}`, tokens) }))
      .filter((x) => x.s > 0)
    if (!tokens.length) {
      // Without a query, show navigation, labs and a few recent runs only.
      return scored
        .filter((x) => ['Go to', 'Labs'].includes(x.it.group) || (x.it.group === 'Runs' && x.it.key))
        .slice(0, 24)
        .map((x) => x.it)
    }
    return scored
      .sort((a, b) => b.s - a.s)
      .slice(0, 40)
      .map((x) => x.it)
  }, [items, tokens])

  // keep grouping stable: order groups by first appearance
  const grouped = useMemo(() => {
    const out = []
    const seen = {}
    results.forEach((r) => {
      if (!seen[r.group]) {
        seen[r.group] = { group: r.group, items: [] }
        out.push(seen[r.group])
      }
      seen[r.group].items.push(r)
    })
    return out
  }, [results])
  const flat = grouped.flatMap((g) => g.items)

  useEffect(() => setIndex(0), [query])
  useEffect(() => {
    list.current?.querySelector(`[data-index="${index}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [index])

  const exec = (item, alt) => {
    onClose()
    ;(alt && item.extra ? item.extra.run : item.run)()
  }
  const onKey = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setIndex((i) => Math.min(flat.length - 1, i + 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setIndex((i) => Math.max(0, i - 1))
    } else if (e.key === 'Enter' && flat[index]) {
      e.preventDefault()
      exec(flat[index], e.metaKey || e.ctrlKey)
    }
  }

  let counter = -1
  return createPortal(
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-50">
          <motion.div
            className="absolute inset-0 bg-bg/70 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.12 }}
            onClick={onClose}
          />
          <div className="pointer-events-none absolute inset-x-0 top-[10vh] flex justify-center px-4">
            <motion.div
              ref={ref}
              role="dialog"
              aria-modal="true"
              aria-label="Command palette"
              initial={{ opacity: 0, y: -6, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2, ease: [0.2, 0, 0, 1] }}
              className="pointer-events-auto flex max-h-[70vh] w-full max-w-xl flex-col overflow-hidden rounded-xl border border-border bg-surface shadow-overlay"
            >
              <div className="flex items-center gap-3 border-b border-border px-4">
                <Search size={16} className="text-subtle" aria-hidden />
                <input
                  data-autofocus
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={onKey}
                  placeholder="Search pages, labs, runs, environments, algorithms"
                  aria-label="Search"
                  role="combobox"
                  aria-expanded="true"
                  aria-controls="palette-list"
                  aria-activedescendant={flat[index] ? `pal-${flat[index].key}` : undefined}
                  className="h-12 flex-1 bg-transparent text-body text-text placeholder:text-subtle focus:outline-none"
                />
                <Kbd>Esc</Kbd>
              </div>
              <div ref={list} id="palette-list" role="listbox" className="min-h-0 flex-1 overflow-y-auto p-2">
                {!flat.length && (
                  <p className="px-3 py-8 text-center text-body-s text-muted">
                    Nothing matches “{query}”. Try an environment name like “cartpole”.
                  </p>
                )}
                {grouped.map((g) => (
                  <div key={g.group} className="mb-1">
                    <div className="px-3 pb-1 pt-2 text-caption text-subtle">{g.group}</div>
                    {g.items.map((it) => {
                      counter += 1
                      const i = counter
                      const Icon = it.icon || ArrowRight
                      return (
                        <div
                          key={it.key}
                          id={`pal-${it.key}`}
                          role="option"
                          aria-selected={i === index}
                          data-index={i}
                          onMouseMove={() => setIndex(i)}
                          onClick={() => exec(it)}
                          className={cn(
                            'flex cursor-pointer items-center gap-3 rounded px-3 py-2',
                            i === index ? 'bg-surface-3' : 'hover:bg-surface-2',
                          )}
                        >
                          {it.glyph ? (
                            <LabGlyph icon={it.glyph} size={16} className="text-accent" />
                          ) : (
                            <Icon size={16} aria-hidden className="text-subtle" />
                          )}
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-body-s text-text">{it.label}</span>
                            {it.hint && <span className="block truncate text-caption text-subtle">{it.hint}</span>}
                          </span>
                          {it.status && <StatusBadge status={it.status} />}
                          {i === index && it.extra && (
                            <span className="flex items-center gap-1 text-caption text-subtle">
                              <Play size={12} aria-hidden /> {it.extra.label} <Kbd>⌘↵</Kbd>
                            </span>
                          )}
                          {i === index && <CornerDownLeft size={14} className="text-subtle" aria-hidden />}
                        </div>
                      )
                    })}
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      )}
    </AnimatePresence>,
    document.body,
  )
}
