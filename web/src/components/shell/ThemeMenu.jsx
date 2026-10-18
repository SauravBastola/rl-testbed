import { Moon, Sun, SunMoon } from 'lucide-react'
import { useApp } from '../../app/AppContext.jsx'
import { THEMES, getTheme } from '../../lib/theme.js'
import { cn } from '../../lib/utils.js'

const ICONS = { midnight: Moon, daylight: Sun, graphite: SunMoon }

/** Three themes, each a full peer: the current name plus three icon toggles. */
export function ThemeMenu() {
  const { theme, setTheme } = useApp()
  return (
    <div className="flex items-center gap-2 rounded border border-border bg-surface py-1 pl-3 pr-1">
      <span className="min-w-0 flex-1 truncate text-caption text-muted">
        Theme <span className="text-text">{getTheme(theme).label}</span>
      </span>
      <div role="radiogroup" aria-label="Theme" className="flex gap-0.5">
        {THEMES.map((t) => {
          const Icon = ICONS[t.id]
          const active = t.id === theme
          return (
            <button
              key={t.id}
              type="button"
              role="radio"
              aria-checked={active}
              aria-label={`${t.label} theme`}
              title={`${t.label}: ${t.description}`}
              onClick={() => setTheme(t.id)}
              className={cn(
                'grid h-7 w-7 place-items-center rounded-sm transition-colors duration-fast',
                active ? 'bg-surface-3 text-accent' : 'text-subtle hover:bg-surface-2 hover:text-text',
              )}
            >
              <Icon size={14} aria-hidden />
            </button>
          )
        })}
      </div>
    </div>
  )
}
