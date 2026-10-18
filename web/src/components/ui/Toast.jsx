import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { CheckCircle2, AlertTriangle, Info, X } from 'lucide-react'
import { cn } from '../../lib/utils.js'

const ToastCtx = createContext({ toast: () => {} })

export function ToastProvider({ children }) {
  const [items, setItems] = useState([])
  const dismiss = useCallback((id) => setItems((xs) => xs.filter((x) => x.id !== id)), [])
  const toast = useCallback(
    (t) => {
      const id = Math.random().toString(36).slice(2)
      setItems((xs) => [...xs.slice(-3), { id, tone: 'success', ...t }])
      setTimeout(() => dismiss(id), t.duration || 4200)
    },
    [dismiss],
  )
  const value = useMemo(() => ({ toast }), [toast])
  return (
    <ToastCtx.Provider value={value}>
      {children}
      <div aria-live="polite" className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-80 flex-col gap-2">
        <AnimatePresence initial={false}>
          {items.map((t) => {
            const Icon = t.tone === 'danger' ? AlertTriangle : t.tone === 'info' ? Info : CheckCircle2
            return (
              <motion.div
                key={t.id}
                layout
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: 16 }}
                transition={{ duration: 0.2 }}
                className="pointer-events-auto flex items-start gap-3 rounded-lg border border-border bg-surface-2 p-3 shadow-overlay"
              >
                <Icon
                  size={16}
                  aria-hidden
                  className={cn(
                    'mt-0.5 shrink-0',
                    t.tone === 'danger' ? 'text-danger' : t.tone === 'info' ? 'text-info' : 'text-success',
                  )}
                />
                <div className="min-w-0 flex-1">
                  <p className="text-body-s font-medium text-text">{t.title}</p>
                  {t.description && <p className="mt-0.5 break-words text-caption text-muted">{t.description}</p>}
                </div>
                <button
                  type="button"
                  aria-label="Dismiss"
                  onClick={() => dismiss(t.id)}
                  className="rounded p-0.5 text-subtle hover:bg-surface-3 hover:text-text"
                >
                  <X size={14} aria-hidden />
                </button>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>
    </ToastCtx.Provider>
  )
}

export const useToast = () => useContext(ToastCtx)
