import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '../../lib/utils.js'
import { Input, Switch } from '../ui/Field.jsx'
import { Chip, Segmented } from '../ui/Segmented.jsx'

const sig = (x) => Number(Number(x).toPrecision(3))
const clamp = (p, v) => Math.min(p.max ?? Infinity, Math.max(p.min ?? -Infinity, v))
const uniq = (xs) => [...new Set(xs.map((x) => JSON.stringify(x)))].map((x) => JSON.parse(x))

/** A sensible starting list for a parameter: around its default, inside its bounds. */
export function suggestValues(p) {
  if (p.type === 'bool') return [true, false]
  if (p.type === 'choice') return [...p.choices]
  const d = Number(p.default)
  if (p.type === 'int') return uniq([Math.max(1, Math.round(d / 2)), d, d * 2].map((v) => Math.round(clamp(p, v))))
  if (d === 0) return uniq([p.min ?? 0, sig(((p.min ?? 0) + (p.max ?? 1)) / 2)])
  // Parameters that live just below 1 (discount factors, GAE λ) vary in their
  // distance from 1, not multiplicatively: 0.99 → 0.9, 0.97, 0.99.
  if (d >= 0.9 && d < 1 && (p.max ?? 1) <= 1) {
    return uniq([10, 3, 1].map((k) => clamp(p, Number((1 - (1 - d) * k).toFixed(4)))))
  }
  // Round first, then clamp, so rounding can never push a value out of bounds.
  return uniq([d * 0.3, d, d * 3].map((v) => clamp(p, sig(v))))
}

export function suggestRange(p) {
  const d = Number(p.default)
  if (p.type === 'int')
    return { min: Math.round(clamp(p, Math.max(1, d / 2))), max: Math.round(clamp(p, d * 2)), log: false }
  if (d > 0) return { min: sig(clamp(p, d / 10)), max: sig(clamp(p, d * 10)), log: true }
  return { min: p.min ?? 0, max: p.max ?? 1, log: false }
}

const parseList = (p, text) => {
  const tokens = text
    .split(/[,\s]+/)
    .map((t) => t.trim())
    .filter(Boolean)
  const bad = tokens.filter((t) => Number.isNaN(Number(t)))
  return { values: tokens.filter((t) => !Number.isNaN(Number(t))).map(Number), bad }
}

function ParamRow({ p, dim, onChange, strategy }) {
  const on = !!dim
  const numeric = p.type === 'int' || p.type === 'float'
  const mode = dim && 'min' in dim ? 'range' : 'list'
  const [text, setText] = useState(() => (dim?.values ? dim.values.join(', ') : suggestValues(p).join(', ')))
  const parsed = numeric ? parseList(p, text) : null

  const enable = (checked) => {
    if (!checked) return onChange(undefined)
    if (numeric && strategy === 'random') return onChange(suggestRange(p))
    const values = suggestValues(p)
    setText(values.join(', '))
    onChange({ values })
  }

  return (
    <div
      className={cn(
        'rounded-lg border px-4 py-3 transition-colors duration-fast',
        on ? 'border-accent/40 bg-accent/[.04]' : 'border-border',
      )}
    >
      <div className="flex items-center gap-3">
        <Switch checked={on} onChange={enable} label={`Sweep ${p.label}`} />
        <div className="min-w-0 flex-1">
          <span className="block truncate text-body-s font-medium text-text">{p.label}</span>
          <span className="block truncate text-caption text-subtle">
            <code>{p.name}</code> · default <span className="num">{String(p.default)}</span>
            {p.min !== null && p.min !== undefined && p.max !== null && p.max !== undefined && (
              <>
                {' '}
                · range <span className="num">{p.min}</span>–<span className="num">{p.max}</span>
              </>
            )}
          </span>
        </div>
        {on && numeric && strategy === 'random' && (
          <Segmented
            size="sm"
            label={`${p.label} search mode`}
            value={mode}
            onChange={(m) =>
              onChange(
                m === 'range' ? suggestRange(p) : { values: parsed.values.length ? parsed.values : suggestValues(p) },
              )
            }
            options={[
              { value: 'list', label: 'List' },
              { value: 'range', label: 'Range' },
            ]}
          />
        )}
      </div>

      {on && (
        <div className="mt-3 pl-12">
          {!numeric ? (
            <div className="flex flex-wrap gap-1.5">
              {(p.type === 'bool' ? [true, false] : p.choices).map((c) => {
                const active = dim.values.includes(c)
                return (
                  <Chip
                    key={String(c)}
                    active={active}
                    onClick={() => {
                      const next = active ? dim.values.filter((v) => v !== c) : [...dim.values, c]
                      onChange({ values: next })
                    }}
                  >
                    {String(c)}
                  </Chip>
                )
              })}
            </div>
          ) : mode === 'list' ? (
            <div>
              <Input
                aria-label={`${p.label} values`}
                className="num"
                value={text}
                placeholder="e.g. 0.001, 0.0003, 0.0001"
                onChange={(e) => {
                  setText(e.target.value)
                  onChange({ values: parseList(p, e.target.value).values })
                }}
              />
              <p className={cn('mt-1 text-caption', parsed.bad.length ? 'text-danger' : 'text-subtle')}>
                {parsed.bad.length
                  ? `Not a number: ${parsed.bad.join(', ')}`
                  : `${parsed.values.length} value${parsed.values.length === 1 ? '' : 's'}, separated by commas`}
              </p>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2 text-caption text-muted">
                Min
                <Input
                  type="number"
                  step="any"
                  className="num h-8 w-32"
                  value={dim.min}
                  onChange={(e) => onChange({ ...dim, min: Number(e.target.value) })}
                />
              </label>
              <label className="flex items-center gap-2 text-caption text-muted">
                Max
                <Input
                  type="number"
                  step="any"
                  className="num h-8 w-32"
                  value={dim.max}
                  onChange={(e) => onChange({ ...dim, max: Number(e.target.value) })}
                />
              </label>
              <label className="flex items-center gap-2 text-caption text-muted">
                <Switch checked={!!dim.log} onChange={(v) => onChange({ ...dim, log: v })} label="Log scale" />
                Log scale
              </label>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/** Choose which hyperparameters to sweep and over what values. */
export function SpaceEditor({ agent, space, onChange, strategy }) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const set = (name) => (dim) => {
    const next = { ...space }
    if (dim === undefined) delete next[name]
    else next[name] = dim
    onChange(next)
  }
  const core = agent.params.filter((p) => !p.advanced)
  const advanced = agent.params.filter((p) => p.advanced)
  const swept = Object.keys(space).length
  return (
    <div className="space-y-2">
      {core.map((p) => (
        <ParamRow
          key={`${agent.name}-${p.name}-${strategy}`}
          p={p}
          dim={space[p.name]}
          onChange={set(p.name)}
          strategy={strategy}
        />
      ))}
      {advanced.length > 0 && (
        <>
          <button
            type="button"
            aria-expanded={showAdvanced}
            onClick={() => setShowAdvanced((v) => !v)}
            className="flex w-full items-center justify-between rounded px-1 py-2 text-body-s font-medium text-muted hover:text-text"
          >
            <span>
              Advanced parameters <span className="text-subtle">({advanced.length})</span>
            </span>
            <ChevronDown
              size={16}
              aria-hidden
              className={cn('transition-transform duration-fast', showAdvanced && 'rotate-180')}
            />
          </button>
          {showAdvanced &&
            advanced.map((p) => (
              <ParamRow
                key={`${agent.name}-${p.name}-${strategy}`}
                p={p}
                dim={space[p.name]}
                onChange={set(p.name)}
                strategy={strategy}
              />
            ))}
        </>
      )}
      {!swept && <p className="px-1 text-caption text-subtle">Switch on at least one parameter to sweep it.</p>}
    </div>
  )
}
