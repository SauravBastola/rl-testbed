import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { applyTheme, readStoredTheme } from '../lib/theme.js'
import { useLabs } from '../lib/queries.js'

const LAB_KEY = 'rltestbed.lab'
export const DEFAULT_LAB = 'foundations'

const Ctx = createContext(null)

function readLab() {
  try {
    return window.localStorage.getItem(LAB_KEY) || DEFAULT_LAB
  } catch {
    return DEFAULT_LAB
  }
}

/** Theme + current lab. The lab drives the accent via [data-lab] on <html>. */
export function AppProvider({ children }) {
  const [theme, setThemeState] = useState(readStoredTheme)
  const [labId, setLabIdState] = useState(readLab)

  useEffect(() => {
    applyTheme(theme)
  }, [theme])
  useEffect(() => {
    document.documentElement.dataset.lab = labId
    try {
      window.localStorage.setItem(LAB_KEY, labId)
    } catch {
      /* storage disabled */
    }
  }, [labId])

  const setTheme = useCallback((id) => setThemeState(id), [])
  const setLabId = useCallback((id) => id && setLabIdState(id), [])
  const value = useMemo(() => ({ theme, setTheme, labId, setLabId }), [theme, setTheme, labId, setLabId])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp() {
  return useContext(Ctx)
}

/** Current lab id, its definition (once loaded) and the setter. */
export function useLab() {
  const { labId, setLabId } = useApp()
  const labs = useLabs()
  const lab = (labs.data || []).find((l) => l.id === labId) || null
  return { labId, lab, labs: labs.data || [], setLabId, isLoading: labs.isLoading, error: labs.error }
}

/** Lab-scoped routes call this so the URL is the source of truth for the lab. */
export function useRouteLab() {
  const { labId: param } = useParams()
  const { labId, setLabId, labs, lab, isLoading, error } = useLab()
  const known = labs.length ? labs.some((l) => l.id === param) : true
  useEffect(() => {
    if (param && known && param !== labId) setLabId(param)
  }, [param, known, labId, setLabId])
  return {
    labId: param || labId,
    lab: labs.find((l) => l.id === (param || labId)) || lab,
    unknown: !known,
    isLoading,
    error,
  }
}

/** Detail pages for an entity that lives in a lab (env, run) adopt that lab's accent. */
export function useAdoptLab(id) {
  const { setLabId, labs } = useLab()
  useEffect(() => {
    if (id && labs.some((l) => l.id === id)) setLabId(id)
  }, [id, labs, setLabId])
}
