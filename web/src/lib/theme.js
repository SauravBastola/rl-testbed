export const THEMES = [
  { id: 'midnight', label: 'Midnight', description: 'Default dark lab', dark: true },
  { id: 'daylight', label: 'Daylight', description: 'Light, for bright rooms and print', dark: false },
  { id: 'graphite', label: 'Graphite', description: 'Neutral, low-chroma for long sessions', dark: true },
]
export const DEFAULT_THEME = 'midnight'
const KEY = 'rltestbed.theme'

export const getTheme = (id) => THEMES.find((t) => t.id === id) || THEMES[0]

export function readStoredTheme() {
  try {
    const stored = window.localStorage.getItem(KEY)
    // aurora/ember were retired in v3; fall back to the default.
    return getTheme(stored).id
  } catch {
    return DEFAULT_THEME
  }
}

export function applyTheme(id) {
  const theme = getTheme(id)
  document.documentElement.dataset.theme = theme.id
  try {
    window.localStorage.setItem(KEY, theme.id)
  } catch {
    /* storage disabled */
  }
  return theme
}
