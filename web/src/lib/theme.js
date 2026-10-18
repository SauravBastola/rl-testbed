/**
 * Two modes (night, day), each with several themes. The sidebar switches mode;
 * Settings chooses which theme each mode uses. Every theme is a full peer,
 * verified by `npm run check:tokens`.
 */
export const THEMES = [
  { id: 'midnight', mode: 'night', label: 'Midnight', description: 'Cool, deep slate. The default.' },
  { id: 'graphite', mode: 'night', label: 'Graphite', description: 'Neutral grey, muted accents, for long sessions.' },
  { id: 'abyss', mode: 'night', label: 'Abyss', description: 'Deep navy with luminous accents.' },
  {
    id: 'obsidian',
    mode: 'night',
    label: 'Obsidian',
    description: 'True black and bright text. Highest contrast; ideal on OLED.',
  },
  { id: 'daylight', mode: 'day', label: 'Daylight', description: 'Clean cool white. The default.' },
  { id: 'paper', mode: 'day', label: 'Paper', description: 'Warm off-white, softer on the eyes.' },
  { id: 'contrast', mode: 'day', label: 'High contrast', description: 'Pure white, black text, strong borders.' },
]
export const MODES = ['night', 'day']
export const DEFAULTS = { mode: 'night', night: 'midnight', day: 'daylight' }

const KEYS = { mode: 'rltestbed.mode', night: 'rltestbed.theme.night', day: 'rltestbed.theme.day' }
const LEGACY = 'rltestbed.theme'

export const getTheme = (id) => THEMES.find((t) => t.id === id) || THEMES[0]
export const themesFor = (mode) => THEMES.filter((t) => t.mode === mode)
export const activeTheme = (prefs) => (prefs.mode === 'day' ? prefs.day : prefs.night)

const valid = (id, mode) => THEMES.some((t) => t.id === id && t.mode === mode)

/** Read preferences, migrating the single v3 theme key if that is all there is. */
export function readPrefs() {
  const prefs = { ...DEFAULTS }
  try {
    const ls = window.localStorage
    const legacy = ls.getItem(LEGACY)
    if (legacy && !ls.getItem(KEYS.mode)) {
      const t = getTheme(legacy)
      prefs.mode = legacy === t.id ? t.mode : 'night'
      if (legacy === t.id) prefs[t.mode] = t.id
    }
    const mode = ls.getItem(KEYS.mode)
    if (MODES.includes(mode)) prefs.mode = mode
    const night = ls.getItem(KEYS.night)
    if (valid(night, 'night')) prefs.night = night
    const day = ls.getItem(KEYS.day)
    if (valid(day, 'day')) prefs.day = day
  } catch {
    /* storage disabled */
  }
  return prefs
}

export function applyPrefs(prefs) {
  const id = activeTheme(prefs)
  document.documentElement.dataset.theme = id
  document.documentElement.dataset.mode = prefs.mode
  try {
    const ls = window.localStorage
    ls.setItem(KEYS.mode, prefs.mode)
    ls.setItem(KEYS.night, prefs.night)
    ls.setItem(KEYS.day, prefs.day)
    ls.setItem(LEGACY, id) // read by index.html before first paint
  } catch {
    /* storage disabled */
  }
  return id
}
