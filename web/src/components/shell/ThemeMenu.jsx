import { Link } from 'react-router-dom'
import { Moon, Settings2, Sun } from 'lucide-react'
import { useApp } from '../../app/AppContext.jsx'
import { getTheme } from '../../lib/theme.js'
import { cn } from '../../lib/utils.js'

const MODES = [
  { id: 'night', label: 'Night', icon: Moon },
  { id: 'day', label: 'Day', icon: Sun },
]

/** Night / Day. Which theme each uses is chosen in Settings. */
export function ThemeMenu() {
  const { prefs, setMode } = useApp()
  return (
    <div className="flex items-center gap-1 rounded border border-border bg-surface p-1">
      <div role="radiogroup" aria-label="Appearance" className="grid flex-1 grid-cols-2 gap-1">
        {MODES.map(({ id, label, icon: Icon }) => {
          const active = prefs.mode === id
          return (
            <button
              key={id}
              type="button"
              role="radio"
              aria-checked={active}
              title={`${label}: ${getTheme(prefs[id]).label}`}
              onClick={() => setMode(id)}
              className={cn(
                'flex h-7 items-center justify-center gap-1.5 rounded-sm text-caption font-medium transition-colors duration-fast',
                active ? 'bg-surface-3 text-text' : 'text-subtle hover:text-text',
              )}
            >
              <Icon size={14} aria-hidden className={active ? 'text-accent' : undefined} />
              {label}
            </button>
          )
        })}
      </div>
      <Link
        to="/settings"
        aria-label="Theme settings"
        title="Choose which theme Night and Day use"
        className="grid h-7 w-7 shrink-0 place-items-center rounded-sm text-subtle hover:bg-surface-2 hover:text-text"
      >
        <Settings2 size={14} aria-hidden />
      </Link>
    </div>
  )
}
