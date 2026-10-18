import { useEffect, useRef, useState } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import {
  BookMarked,
  Boxes,
  Check,
  ChevronsUpDown,
  Cpu,
  Eye,
  FileCode2,
  FlaskConical,
  GitCompareArrows,
  GraduationCap,
  LayoutDashboard,
  ListTree,
  Map,
  Plus,
  Search,
  SlidersHorizontal,
  Stethoscope,
} from 'lucide-react'
import { cn } from '../../lib/utils.js'
import { useLab } from '../../app/AppContext.jsx'
import { useRuns, useSystem } from '../../lib/queries.js'
import { BrandMark, Wordmark } from '../art/BrandMark.jsx'
import { LabGlyph } from '../art/LabArt.jsx'
import { Kbd } from '../ui/Page.jsx'
import { Dot } from '../ui/Badge.jsx'
import { ThemeMenu } from './ThemeMenu.jsx'
import { useFocusTrap } from '../ui/Overlay.jsx'

export const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)
export const MOD = isMac ? '⌘' : 'Ctrl'

function NavItem({ to, icon: Icon, children, end, badge, onNavigate }) {
  return (
    <NavLink
      to={to}
      end={end}
      onClick={onNavigate}
      className={({ isActive }) =>
        cn(
          'group relative flex h-8 items-center gap-2.5 rounded px-2.5 text-body-s font-medium transition-colors duration-fast',
          isActive ? 'bg-surface-3 text-text' : 'text-muted hover:bg-surface-2 hover:text-text',
        )
      }
    >
      {({ isActive }) => (
        <>
          {isActive && (
            <motion.span
              layoutId="nav-active"
              className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-accent"
              transition={{ type: 'spring', stiffness: 280, damping: 30 }}
            />
          )}
          <Icon size={16} aria-hidden className={isActive ? 'text-accent' : 'text-subtle group-hover:text-muted'} />
          <span className="truncate">{children}</span>
          {badge}
        </>
      )}
    </NavLink>
  )
}

function Group({ title, children }) {
  return (
    <div>
      <div className="mb-1 px-2.5 text-caption font-medium text-subtle">{title}</div>
      <div className="flex flex-col gap-0.5">{children}</div>
    </div>
  )
}

/** The lab switcher: which workspace you are in, and one click to change it. */
export function LabSwitcher() {
  const { lab, labs, labId } = useLab()
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const panel = useRef(null)
  useFocusTrap(panel, open, () => setOpen(false))

  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === 'l') {
        e.preventDefault()
        setOpen((o) => !o)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const choose = (id) => {
    setOpen(false)
    // Stay on the equivalent page in the new lab when there is one.
    const m = location.pathname.match(/^\/lab\/[^/]+(\/[^?]*)?/)
    navigate(`/lab/${id}${m?.[1] || ''}`)
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="flex w-full items-center gap-3 rounded-lg border border-border bg-surface-2 p-2.5 text-left transition-colors duration-fast hover:border-border-strong"
      >
        <span className="grid h-9 w-9 shrink-0 place-items-center rounded bg-accent/15 text-accent">
          <LabGlyph icon={lab?.icon} size={20} />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-caption text-subtle">Lab</span>
          <span className="block truncate text-body font-semibold text-text">{lab?.label || 'Loading labs'}</span>
        </span>
        <ChevronsUpDown size={14} className="shrink-0 text-subtle" aria-hidden />
      </button>
      <AnimatePresence>
        {open && (
          <>
            <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} aria-hidden />
            <motion.div
              ref={panel}
              role="listbox"
              aria-label="Choose a lab"
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.12 }}
              className="absolute left-0 right-0 top-full z-40 mt-2 overflow-hidden rounded-lg border border-border bg-surface shadow-overlay lg:-right-64"
            >
              <div className="flex items-center justify-between border-b border-border px-3 py-2">
                <span className="text-caption text-subtle">Switch lab</span>
                <span className="flex gap-1">
                  <Kbd>{MOD}</Kbd>
                  <Kbd>⇧</Kbd>
                  <Kbd>L</Kbd>
                </span>
              </div>
              <div className="p-1.5">
                {labs.map((l) => (
                  <button
                    key={l.id}
                    type="button"
                    role="option"
                    aria-selected={l.id === labId}
                    data-lab-option={l.id}
                    onClick={() => choose(l.id)}
                    className="flex w-full items-start gap-3 rounded p-2.5 text-left hover:bg-surface-2 focus-visible:bg-surface-2"
                  >
                    <span
                      className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded"
                      style={{ color: `rgb(var(--hue-${l.accent}))`, background: `rgb(var(--hue-${l.accent}) / .14)` }}
                    >
                      <LabGlyph icon={l.icon} size={18} />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center justify-between gap-2">
                        <span className="font-semibold text-text">{l.label}</span>
                        <span className="num text-caption text-subtle">{l.env_count} envs</span>
                      </span>
                      <span className="block text-caption text-muted">{l.tagline}</span>
                    </span>
                    {l.id === labId && <Check size={14} className="mt-1 text-accent" aria-hidden />}
                  </button>
                ))}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}

function StatusChip({ onOpen }) {
  const sys = useSystem(true)
  const running = sys.data?.jobs?.running ?? 0
  const queued = sys.data?.jobs?.queued ?? 0
  const ok = !sys.error
  return (
    <button
      type="button"
      onClick={onOpen}
      className="flex w-full items-center gap-2.5 rounded border border-border bg-surface-2 px-3 py-2 text-left hover:border-border-strong"
    >
      <Dot tone={!ok ? 'danger' : running ? 'accent' : 'success'} pulse={running > 0} />
      <span className="min-w-0 flex-1">
        <span className="block text-body-s font-medium text-text">
          {!ok ? 'Server unreachable' : running ? `${running} training` : 'Local workspace'}
        </span>
        <span className="block truncate text-caption text-subtle">
          {!ok
            ? 'Start rltestbed serve'
            : queued
              ? `${queued} queued`
              : sys.data
                ? `${sys.data.catalogue.envs_available} envs · ${sys.data.catalogue.agents_implemented} algorithms`
                : 'Connecting'}
        </span>
      </span>
      <Cpu size={14} className="text-subtle" aria-hidden />
    </button>
  )
}

export function Sidebar({ onOpenPalette, onOpenSystem, onNavigate }) {
  const { labId } = useLab()
  const runs = useRuns({ lab: labId })
  const active = (runs.data || []).filter((r) => ['running', 'queued', 'stopping'].includes(r.status)).length
  const base = `/lab/${labId}`
  return (
    <nav aria-label="Main" className="flex h-full flex-col gap-5 overflow-y-auto px-3 py-4 scrollbar-none">
      <div className="flex items-center gap-2.5 px-1.5">
        <BrandMark size={24} />
        <Wordmark />
      </div>

      <LabSwitcher />

      <button
        type="button"
        onClick={onOpenPalette}
        className="flex h-9 items-center gap-2 rounded border border-border bg-surface px-3 text-body-s text-subtle hover:border-border-strong hover:text-muted"
      >
        <Search size={14} aria-hidden />
        <span className="flex-1 text-left">Search or jump to</span>
        <Kbd>{MOD}K</Kbd>
      </button>

      <Group title="This lab">
        <NavItem to={base} end icon={LayoutDashboard} onNavigate={onNavigate}>
          Overview
        </NavItem>
        <NavItem to={`${base}/new`} icon={Plus} onNavigate={onNavigate}>
          New run
        </NavItem>
        <NavItem
          to={`${base}/runs`}
          icon={ListTree}
          onNavigate={onNavigate}
          badge={
            active ? (
              <span className="num ml-auto rounded-full bg-accent/15 px-1.5 text-micro text-accent">{active}</span>
            ) : null
          }
        >
          Runs
        </NavItem>
        <NavItem to={`${base}/sweeps`} icon={SlidersHorizontal} onNavigate={onNavigate}>
          Sweeps
        </NavItem>
        <NavItem to={`${base}/envs`} icon={Map} onNavigate={onNavigate}>
          Environments
        </NavItem>
        <NavItem to={`${base}/algorithms`} icon={FlaskConical} onNavigate={onNavigate}>
          Algorithms
        </NavItem>
      </Group>

      <Group title="Across labs">
        <NavItem to="/compare" icon={GitCompareArrows} onNavigate={onNavigate}>
          Compare
        </NavItem>
        <NavItem to="/experiments" icon={FileCode2} onNavigate={onNavigate}>
          Experiments
        </NavItem>
        <NavItem to="/models" icon={Boxes} onNavigate={onNavigate}>
          Models
        </NavItem>
        <NavItem to="/visualize" icon={Eye} onNavigate={onNavigate}>
          Visualize
        </NavItem>
        <NavItem to="/diagnostics" icon={Stethoscope} onNavigate={onNavigate}>
          Diagnostics
        </NavItem>
      </Group>

      <Group title="Learn">
        <NavItem to="/learn" icon={GraduationCap} onNavigate={onNavigate}>
          Learning path
        </NavItem>
        <NavItem to="/glossary" icon={BookMarked} onNavigate={onNavigate}>
          Glossary
        </NavItem>
      </Group>

      <div className="mt-auto flex flex-col gap-2 pt-2">
        <ThemeMenu />
        <StatusChip onOpen={onOpenSystem} />
      </div>
    </nav>
  )
}
