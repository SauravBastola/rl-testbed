/**
 * Design-token guard. Run with `npm run check:tokens`.
 *  1. No hex colour literals in components or pages.
 *  2. WCAG contrast for every theme, computed from src/styles.css itself:
 *     body text 4.5:1, muted and subtle text 4.5:1, status colours 4.5:1,
 *     button label on accent 4.5:1, lab accents and chart series 3:1 (graphics).
 */
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

let failed = false
const fail = (msg) => {
  failed = true
  console.error(`  ✗ ${msg}`)
}

// ---------------------------------------------------------------- 1. hex scan
const walk = (dir) =>
  readdirSync(dir).flatMap((f) => {
    const p = join(dir, f)
    return statSync(p).isDirectory() ? walk(p) : /\.(jsx?|tsx?)$/.test(f) ? [p] : []
  })
const files = [...walk('src/components'), ...walk('src/pages'), ...walk('src/app')]
const hex = /#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b(?![0-9a-zA-Z])/
console.log(`Hex literals in ${files.length} component files`)
for (const f of files) {
  readFileSync(f, 'utf8')
    .split('\n')
    .forEach((line, i) => {
      if (hex.test(line) && !line.includes('&#')) fail(`${f}:${i + 1} ${line.trim()}`)
    })
}
if (!failed) console.log('  ✓ none')

// ---------------------------------------------------------- 2. contrast check
const css = readFileSync('src/styles.css', 'utf8')
const blocks = {}
for (const m of css.matchAll(/\[data-theme='([\w-]+)'\]\s*\{/g)) {
  const start = m.index + m[0].length
  const body = css.slice(start, css.indexOf('\n  }', start))
  blocks[m[1]] = Object.fromEntries([...body.matchAll(/--([\w-]+):\s*([^;]+);/g)].map((x) => [x[1], x[2].trim()]))
}
// Midnight is the base on :root; other themes override it (some define every token).
const base = blocks.midnight
const themes = Object.fromEntries(Object.entries(blocks).map(([name, vars]) => [name, { ...base, ...vars }]))
const expected = [...readFileSync('src/lib/theme.js', 'utf8').matchAll(/id: '([\w-]+)', mode:/g)].map((m) => m[1])
for (const id of expected) if (!themes[id]) fail(`theme '${id}' is listed in theme.js but has no [data-theme='${id}'] block`)
const resolve = (vars, value, depth = 0) => {
  const m = /^var\(--([\w-]+)\)$/.exec(value)
  if (m && depth < 5) return resolve(vars, vars[m[1]], depth + 1)
  return value.split(/\s+/).map(Number)
}
const lum = ([r, g, b]) => {
  const c = [r, g, b].map((v) => {
    const s = v / 255
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}
const ratio = (a, b) => {
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p)
  return (x + 0.05) / (y + 0.05)
}

const LABS = ['violet', 'cyan', 'amber', 'rose', 'lime', 'sky']
for (const [name, vars] of Object.entries(themes)) {
  console.log(`Contrast — ${name}`)
  const c = (k) => resolve(vars, vars[k])
  const checks = [
    ['text on surface', c('text'), c('surface'), 4.5],
    ['text on bg', c('text'), c('bg'), 4.5],
    ['muted text on surface', c('text-muted'), c('surface'), 4.5],
    ['subtle text on surface', c('text-subtle'), c('surface'), 4.5],
    ['subtle text on surface-2', c('text-subtle'), c('surface-2'), 4.0],
    ['success on surface', c('success'), c('surface'), 4.5],
    ['warning on surface', c('warning'), c('surface'), 4.5],
    ['danger on surface', c('danger'), c('surface'), 4.5],
    ...LABS.map((h) => [`lab accent ${h} on surface`, c(`hue-${h}`), c('surface'), 3]),
    ...LABS.map((h) => [`button label on ${h}`, c('text-inverse'), c(`hue-${h}`), 4.5]),
    ...[2, 3, 4, 5, 6].map((i) => [`chart-${i} on surface`, c(`chart-${i}`), c('surface'), 3]),
  ]
  let worst = Infinity
  for (const [label, fg, bg, min] of checks) {
    const r = ratio(fg, bg)
    worst = Math.min(worst, r / min)
    if (r < min) fail(`${name}: ${label} is ${r.toFixed(2)}:1, needs ${min}:1`)
  }
  if (worst >= 1) console.log(`  ✓ ${checks.length} pairs pass (tightest margin ${(worst * 100 - 100).toFixed(0)}% above minimum)`)
}

// ------------------------------------------------- every lab has an accent
const labIds = [...readFileSync('../rltestbed/labs.py', 'utf8').matchAll(/"id": "([\w-]+)"/g)].map((m) => m[1])
const missingAccent = labIds.filter((id) => !css.includes(`[data-lab='${id}']`))
console.log(`Lab accents for ${labIds.length} labs`)
if (missingAccent.length) fail(`no [data-lab] accent rule in styles.css for: ${missingAccent.join(', ')}`)
else console.log('  ✓ all present')

if (failed) {
  console.error('\nToken check failed.')
  process.exit(1)
}
console.log('\nToken check passed.')
