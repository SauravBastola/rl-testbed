import { afterEach, describe, expect, it } from 'vitest'
import { THEMES, activeTheme, applyPrefs, readPrefs, themesFor } from '../lib/theme.js'

afterEach(() => window.localStorage.clear())

describe('night and day themes', () => {
  it('has at least two themes per mode, all distinct', () => {
    expect(themesFor('night').length).toBeGreaterThanOrEqual(2)
    expect(themesFor('day').length).toBeGreaterThanOrEqual(2)
    expect(new Set(THEMES.map((t) => t.id)).size).toBe(THEMES.length)
  })

  it('defaults to Midnight at night and Daylight by day', () => {
    expect(readPrefs()).toEqual({ mode: 'night', night: 'midnight', day: 'daylight' })
  })

  it('migrates a v3 single-theme choice', () => {
    window.localStorage.setItem('rltestbed.theme', 'graphite')
    expect(readPrefs()).toMatchObject({ mode: 'night', night: 'graphite' })
    window.localStorage.clear()
    window.localStorage.setItem('rltestbed.theme', 'daylight')
    expect(readPrefs()).toMatchObject({ mode: 'day', day: 'daylight' })
  })

  it('remembers each mode separately and applies the active one', () => {
    const prefs = { mode: 'day', night: 'obsidian', day: 'paper' }
    expect(applyPrefs(prefs)).toBe('paper')
    expect(document.documentElement.dataset.theme).toBe('paper')
    expect(readPrefs()).toEqual(prefs)
    expect(activeTheme({ ...prefs, mode: 'night' })).toBe('obsidian')
  })

  it('ignores a stored theme that belongs to the other mode', () => {
    window.localStorage.setItem('rltestbed.mode', 'night')
    window.localStorage.setItem('rltestbed.theme.night', 'paper')
    expect(readPrefs().night).toBe('midnight')
  })
})
