import { Suspense, useEffect, useRef, useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { Sidebar } from './Sidebar.jsx'
import { Topbar } from './Topbar.jsx'
import { CommandPalette } from './CommandPalette.jsx'
import { SystemDrawer } from './SystemDrawer.jsx'
import { Skeleton } from '../ui/Feedback.jsx'
import { useFocusTrap } from '../ui/Overlay.jsx'

function PageFallback() {
  return (
    <div className="space-y-6" role="status" aria-label="Loading page">
      <Skeleton className="h-9 w-72" />
      <Skeleton className="h-5 w-[28rem] max-w-full" />
      <div className="grid gap-4 md:grid-cols-3">
        <Skeleton className="h-28" />
        <Skeleton className="h-28" />
        <Skeleton className="h-28" />
      </div>
      <Skeleton className="h-72" />
    </div>
  )
}

/** One shell for every lab: sidebar, top bar, content column, palette, drawers. */
export function AppShell() {
  const [palette, setPalette] = useState(false)
  const [system, setSystem] = useState(false)
  const [drawer, setDrawer] = useState(false)
  const location = useLocation()
  const drawerRef = useRef(null)
  useFocusTrap(drawerRef, drawer, () => setDrawer(false))

  useEffect(() => {
    const onKey = (e) => {
      const typing = /input|textarea|select/i.test(e.target.tagName) || e.target.isContentEditable
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPalette((p) => !p)
      } else if (e.key === '/' && !typing) {
        e.preventDefault()
        setPalette(true)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    setDrawer(false)
    window.scrollTo({ top: 0 })
  }, [location.pathname])

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[248px_minmax(0,1fr)]">
      <a
        href="#main"
        className="sr-only z-50 rounded bg-surface px-3 py-2 focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
      >
        Skip to content
      </a>
      <aside className="sticky top-0 hidden h-screen border-r border-border/70 bg-bg-soft/60 lg:block">
        <Sidebar onOpenPalette={() => setPalette(true)} onOpenSystem={() => setSystem(true)} />
      </aside>

      <AnimatePresence>
        {drawer && (
          <div className="fixed inset-0 z-40 lg:hidden">
            <motion.div
              className="absolute inset-0 bg-bg/70 backdrop-blur-sm"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setDrawer(false)}
            />
            <motion.aside
              ref={drawerRef}
              aria-label="Navigation"
              initial={{ x: -260 }}
              animate={{ x: 0 }}
              exit={{ x: -260 }}
              transition={{ duration: 0.32, ease: [0.2, 0, 0, 1] }}
              className="absolute inset-y-0 left-0 w-[260px] border-r border-border bg-bg-soft"
            >
              <Sidebar
                onOpenPalette={() => setPalette(true)}
                onOpenSystem={() => setSystem(true)}
                onNavigate={() => setDrawer(false)}
              />
            </motion.aside>
          </div>
        )}
      </AnimatePresence>

      <div className="flex min-w-0 flex-col">
        <Topbar onMenu={() => setDrawer(true)} />
        <main id="main" className="mx-auto w-full max-w-page flex-1 px-4 pb-16 pt-8 md:px-6 xl:px-8">
          <Suspense fallback={<PageFallback />}>
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2, ease: [0.2, 0, 0, 1] }}
            >
              <Outlet />
            </motion.div>
          </Suspense>
        </main>
      </div>

      <CommandPalette open={palette} onClose={() => setPalette(false)} />
      <SystemDrawer open={system} onClose={() => setSystem(false)} />
    </div>
  )
}
