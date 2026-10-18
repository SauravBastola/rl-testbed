import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { X } from 'lucide-react'
import { cn } from '../../lib/utils.js'
import { IconButton } from './Button.jsx'

const FOCUSABLE =
  'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])'

/** Traps Tab inside `ref`, closes on Escape, and restores focus on unmount. */
export function useFocusTrap(ref, active, onClose) {
  useEffect(() => {
    if (!active) return undefined
    const previous = document.activeElement
    const node = ref.current
    const first = node?.querySelector('[data-autofocus]') || node?.querySelector(FOCUSABLE)
    first?.focus()
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onClose?.()
      }
      if (e.key !== 'Tab' || !node) return
      const items = [...node.querySelectorAll(FOCUSABLE)].filter((el) => el.offsetParent !== null)
      if (!items.length) return
      const [a, b] = [items[0], items[items.length - 1]]
      if (e.shiftKey && document.activeElement === a) {
        e.preventDefault()
        b.focus()
      } else if (!e.shiftKey && document.activeElement === b) {
        e.preventDefault()
        a.focus()
      }
    }
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('keydown', onKey, true)
      if (previous && previous.focus) previous.focus()
    }
  }, [ref, active, onClose])
}

export function Modal({ open, onClose, title, description, children, footer, size = 'md', side }) {
  const ref = useRef(null)
  useFocusTrap(ref, open, onClose)
  const width = { sm: 'max-w-md', md: 'max-w-lg', lg: 'max-w-2xl' }[size]
  return createPortal(
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-50">
          <motion.div
            className="absolute inset-0 bg-bg/70 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
          />
          <div
            className={cn(
              'pointer-events-none absolute inset-0 flex p-4',
              side ? 'justify-end p-0' : 'items-start justify-center pt-[12vh]',
            )}
          >
            <motion.div
              ref={ref}
              role="dialog"
              aria-modal="true"
              aria-label={title}
              initial={side ? { x: 24, opacity: 0 } : { y: 8, opacity: 0, scale: 0.98 }}
              animate={side ? { x: 0, opacity: 1 } : { y: 0, opacity: 1, scale: 1 }}
              exit={side ? { x: 24, opacity: 0 } : { y: 8, opacity: 0 }}
              transition={{ duration: 0.32, ease: [0.2, 0, 0, 1] }}
              className={cn(
                'pointer-events-auto flex w-full flex-col border border-border bg-surface shadow-overlay',
                side ? 'h-full max-w-md' : cn('max-h-[76vh] rounded-xl', width),
              )}
            >
              <div className="flex items-start justify-between gap-4 border-b border-border px-6 py-4">
                <div>
                  <h2 className="text-title-s font-semibold">{title}</h2>
                  {description && <p className="mt-0.5 text-body-s text-muted">{description}</p>}
                </div>
                <IconButton icon={X} label="Close" size="sm" onClick={onClose} />
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">{children}</div>
              {footer && <div className="flex justify-end gap-2 border-t border-border px-6 py-4">{footer}</div>}
            </motion.div>
          </div>
        </div>
      )}
    </AnimatePresence>,
    document.body,
  )
}

/** Lightweight hover/focus tooltip. Content is also exposed to assistive tech. */
export function Tooltip({ content, children, side = 'top', className }) {
  if (!content) return children
  return (
    <span className={cn('group/tt relative inline-flex', className)}>
      {children}
      <span
        role="tooltip"
        className={cn(
          'pointer-events-none absolute left-1/2 z-40 w-max max-w-72 -translate-x-1/2 rounded border border-border bg-surface-3 px-2.5 py-1.5',
          'text-caption text-text opacity-0 shadow-overlay transition-opacity duration-fast',
          'group-focus-within/tt:opacity-100 group-hover/tt:opacity-100',
          side === 'top' ? 'bottom-full mb-2' : 'top-full mt-2',
        )}
      >
        {content}
      </span>
    </span>
  )
}
