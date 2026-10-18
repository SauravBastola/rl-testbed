import { Check, Moon, Sun } from 'lucide-react'
import { useApp } from '../app/AppContext.jsx'
import { themesFor } from '../lib/theme.js'
import { cn } from '../lib/utils.js'
import { Card, CardHeader } from '../components/ui/Card.jsx'
import { PageHeader } from '../components/ui/Page.jsx'
import { Segmented } from '../components/ui/Segmented.jsx'

/**
 * A miniature of the app drawn with a theme's own tokens: the theme CSS is
 * scoped to [data-theme], so this subtree shows that theme whatever is active.
 */
function ThemePreview({ id, labId }) {
  return (
    <div data-theme={id} data-lab={labId} className="overflow-hidden rounded-lg border border-border bg-bg text-text">
      <div className="flex h-36">
        <div className="w-14 shrink-0 space-y-1.5 border-r border-border bg-bg-soft p-2">
          <div className="h-3 w-3 rounded-sm bg-accent" />
          <div className="h-1.5 w-full rounded-full bg-surface-3" />
          <div className="h-1.5 w-4/5 rounded-full bg-accent/60" />
          <div className="h-1.5 w-full rounded-full bg-surface-3" />
          <div className="h-1.5 w-3/5 rounded-full bg-surface-3" />
        </div>
        <div className="min-w-0 flex-1 space-y-2 p-2.5">
          <div className="flex items-center justify-between">
            <div className="h-2 w-16 rounded-full bg-text/80" />
            <div className="h-4 w-10 rounded bg-accent" />
          </div>
          <div className="rounded border border-border bg-surface p-1.5">
            <svg viewBox="0 0 100 34" className="h-12 w-full" aria-hidden>
              <path
                d="M0 30 C15 28 25 22 35 20 S55 10 65 9 85 5 100 4"
                fill="none"
                stroke="rgb(var(--chart-1))"
                strokeWidth="2"
              />
              <path
                d="M0 32 C20 30 30 27 45 24 S70 18 100 15"
                fill="none"
                stroke="rgb(var(--chart-2))"
                strokeWidth="1.5"
              />
              <line x1="0" x2="100" y1="6" y2="6" stroke="rgb(var(--success))" strokeDasharray="3 3" strokeWidth="1" />
            </svg>
          </div>
          <div className="flex gap-1.5">
            <span className="h-3 w-10 rounded-full bg-success/20" />
            <span className="h-3 w-8 rounded-full bg-warning/20" />
            <span className="h-3 w-9 rounded-full bg-danger/20" />
          </div>
          <div className="space-y-1">
            <div className="h-1.5 w-full rounded-full bg-muted/50" />
            <div className="h-1.5 w-2/3 rounded-full bg-subtle/40" />
          </div>
        </div>
      </div>
    </div>
  )
}

function ThemePicker({ mode }) {
  const { prefs, setTheme, labId } = useApp()
  return (
    <div
      role="radiogroup"
      aria-label={`${mode === 'night' ? 'Night' : 'Day'} theme`}
      className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"
    >
      {themesFor(mode).map((t) => {
        const selected = prefs[mode] === t.id
        return (
          <button
            key={t.id}
            type="button"
            role="radio"
            aria-checked={selected}
            onClick={() => setTheme(t.id)}
            className={cn(
              'rounded-xl border p-2 text-left transition-[border-color,box-shadow] duration-fast',
              selected ? 'border-accent shadow-focus' : 'border-border hover:border-border-strong',
            )}
          >
            <ThemePreview id={t.id} labId={labId} />
            <div className="flex items-start justify-between gap-2 px-1 pb-1 pt-3">
              <div className="min-w-0">
                <div className="text-body-s font-semibold text-text">{t.label}</div>
                <div className="text-caption text-muted">{t.description}</div>
              </div>
              {selected && (
                <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-accent text-accent-contrast">
                  <Check size={12} aria-hidden />
                </span>
              )}
            </div>
          </button>
        )
      })}
    </div>
  )
}

export default function Settings() {
  const { prefs, setMode } = useApp()
  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        description="Night and Day in the sidebar switch modes; here you choose which theme each mode uses. Previews use your current lab's accent colour."
      />
      <Card>
        <CardHeader
          title="Appearance"
          description="Choosing a theme also switches to its mode."
          actions={
            <Segmented
              label="Mode"
              value={prefs.mode}
              onChange={setMode}
              options={[
                { value: 'night', label: 'Night', icon: Moon },
                { value: 'day', label: 'Day', icon: Sun },
              ]}
            />
          }
        />
        <div className="space-y-8">
          <section aria-label="Night themes">
            <h3 className="mb-3 flex items-center gap-2 text-body-s font-semibold text-text">
              <Moon size={14} aria-hidden className="text-accent" /> Night theme
            </h3>
            <ThemePicker mode="night" />
          </section>
          <section aria-label="Day themes">
            <h3 className="mb-3 flex items-center gap-2 text-body-s font-semibold text-text">
              <Sun size={14} aria-hidden className="text-accent" /> Day theme
            </h3>
            <ThemePicker mode="day" />
          </section>
        </div>
      </Card>
    </div>
  )
}
