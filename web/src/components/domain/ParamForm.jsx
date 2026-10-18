import { useState } from 'react'
import { ChevronDown, RotateCcw } from 'lucide-react'
import { cn } from '../../lib/utils.js'
import { Field, Input, Select, Switch } from '../ui/Field.jsx'
import { Tooltip } from '../ui/Overlay.jsx'

function boundsText(p) {
  if (p.min !== null && p.min !== undefined && p.max !== null && p.max !== undefined) return `${p.min} – ${p.max}`
  if (p.min !== null && p.min !== undefined) return `≥ ${p.min}`
  if (p.max !== null && p.max !== undefined) return `≤ ${p.max}`
  return null
}

/** Client-side check for immediate feedback; the server re-validates with validate_params. */
export function paramError(p, value) {
  if (p.type === 'int' || p.type === 'float') {
    if (value === '' || value === null || value === undefined || Number.isNaN(Number(value))) return 'Enter a number'
    const n = Number(value)
    if (p.type === 'int' && !Number.isInteger(n)) return 'Must be a whole number'
    if (p.min !== null && p.min !== undefined && n < p.min) return `Must be at least ${p.min}`
    if (p.max !== null && p.max !== undefined && n > p.max) return `Must be at most ${p.max}`
  }
  return null
}

function ParamInput({ p, value, onChange }) {
  const changed = JSON.stringify(value) !== JSON.stringify(p.default)
  const error = paramError(p, value)
  const bounds = boundsText(p)
  return (
    <Field
      label={
        <span className="flex items-center gap-2">
          {p.label}
          {changed && (
            <Tooltip content={`Reset to default (${String(p.default)})`}>
              <button
                type="button"
                aria-label={`Reset ${p.label} to default`}
                onClick={() => onChange(p.default)}
                className="rounded text-accent hover:text-text"
              >
                <RotateCcw size={12} aria-hidden />
              </button>
            </Tooltip>
          )}
        </span>
      }
      hint={<code className="text-micro">{p.name}</code>}
      help={[p.help, bounds && `Range ${bounds}`].filter(Boolean).join('. ')}
      error={error}
    >
      {(id) =>
        p.type === 'bool' ? (
          <Switch id={id} checked={!!value} onChange={onChange} label={p.label} />
        ) : p.type === 'choice' ? (
          <Select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
            {p.choices.map((c) => (
              <option key={String(c)} value={c}>
                {String(c)}
              </option>
            ))}
          </Select>
        ) : (
          <Input
            id={id}
            type="number"
            inputMode="decimal"
            step={p.type === 'int' ? 1 : 'any'}
            min={p.min ?? undefined}
            max={p.max ?? undefined}
            value={value ?? ''}
            aria-invalid={!!error}
            className={cn('num', error && 'border-danger')}
            onChange={(e) => {
              const raw = e.target.value
              onChange(raw === '' ? '' : p.type === 'int' ? Number.parseInt(raw, 10) : Number(raw))
            }}
          />
        )
      }
    </Field>
  )
}

/** Hyperparameter form generated from the algorithm's ParamSpec list. */
export function ParamForm({ params = [], values, onChange }) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const core = params.filter((p) => !p.advanced)
  const advanced = params.filter((p) => p.advanced)
  const set = (name) => (v) => onChange({ ...values, [name]: v })
  if (!params.length) return <p className="text-body-s text-muted">This algorithm has no hyperparameters.</p>
  return (
    <div className="space-y-5">
      <div className="grid gap-x-5 gap-y-4 sm:grid-cols-2">
        {core.map((p) => (
          <ParamInput key={p.name} p={p} value={values[p.name]} onChange={set(p.name)} />
        ))}
      </div>
      {advanced.length > 0 && (
        <div className="rounded-lg border border-border">
          <button
            type="button"
            aria-expanded={showAdvanced}
            onClick={() => setShowAdvanced((s) => !s)}
            className="flex w-full items-center justify-between px-4 py-3 text-body-s font-medium text-muted hover:text-text"
          >
            <span>
              Advanced <span className="text-subtle">({advanced.length})</span>
            </span>
            <ChevronDown
              size={16}
              aria-hidden
              className={cn('transition-transform duration-fast', showAdvanced && 'rotate-180')}
            />
          </button>
          {showAdvanced && (
            <div className="grid gap-x-5 gap-y-4 border-t border-border p-4 sm:grid-cols-2">
              {advanced.map((p) => (
                <ParamInput key={p.name} p={p} value={values[p.name]} onChange={set(p.name)} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export const defaultsOf = (params = []) => Object.fromEntries(params.map((p) => [p.name, p.default]))
export const paramsValid = (params = [], values = {}) => params.every((p) => !paramError(p, values[p.name]))
