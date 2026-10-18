import { useEffect, useMemo, useRef } from 'react'
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './api.js'
import { isActive } from './format.js'

/**
 * Query hooks. Catalogue data is effectively static for a session, so it is
 * cached for a long time; run data refreshes while something is live.
 */
const STATIC = { staleTime: 5 * 60_000, gcTime: 30 * 60_000 }

export const keys = {
  labs: ['labs'],
  envs: ['envs'],
  env: (n) => ['env', n],
  agents: ['agents'],
  agent: (n) => ['agent', n],
  code: (n) => ['code', n],
  caps: ['capabilities'],
  runs: (p = {}) => ['runs', p],
  run: (id) => ['run', id],
  history: (id) => ['history', id],
  models: ['models'],
  diagnostics: ['diagnostics'],
  system: ['system'],
}

export const useLabs = () => useQuery({ queryKey: keys.labs, queryFn: api.labs, ...STATIC })
export const useEnvs = () => useQuery({ queryKey: keys.envs, queryFn: api.envs, ...STATIC })
export const useAgents = () => useQuery({ queryKey: keys.agents, queryFn: api.agents, ...STATIC })
export const useEnv = (name) =>
  useQuery({ queryKey: keys.env(name), queryFn: () => api.env(name), enabled: !!name, ...STATIC })
export const useAgentCode = (name, enabled = true) =>
  useQuery({
    queryKey: keys.code(name),
    queryFn: () => api.agentCode(name),
    enabled: !!name && enabled,
    ...STATIC,
    retry: false,
  })
export const useCapabilities = () => useQuery({ queryKey: keys.caps, queryFn: api.capabilities, ...STATIC })
export const useLearningPath = () => useQuery({ queryKey: ['learning-path'], queryFn: api.learningPath, ...STATIC })
export const useGlossary = () => useQuery({ queryKey: ['glossary'], queryFn: api.glossary, ...STATIC })
export const useSystem = (enabled = true) =>
  useQuery({ queryKey: keys.system, queryFn: api.system, enabled, refetchInterval: enabled ? 5000 : false })

/** Name → spec lookup tables built once per catalogue load. */
export function useCatalogue() {
  const envs = useEnvs()
  const agents = useAgents()
  const labs = useLabs()
  const value = useMemo(() => {
    const envBy = Object.fromEntries((envs.data || []).map((e) => [e.name, e]))
    const agentBy = Object.fromEntries((agents.data || []).map((a) => [a.name, a]))
    const labBy = Object.fromEntries((labs.data || []).map((l) => [l.id, l]))
    return { envBy, agentBy, labBy }
  }, [envs.data, agents.data, labs.data])
  return {
    ...value,
    envs: envs.data || [],
    agents: agents.data || [],
    labs: labs.data || [],
    isLoading: envs.isLoading || agents.isLoading || labs.isLoading,
    error: envs.error || agents.error || labs.error,
    refetch: () => Promise.all([envs.refetch(), agents.refetch(), labs.refetch()]),
  }
}

/** Compatibility verdict for an (agent, env) pair, read from the server matrix. */
export function useCompatMatrix() {
  const caps = useCapabilities()
  const lookup = useMemo(() => {
    const m = caps.data?.matrix || {}
    return (agent, env) => m[agent]?.envs?.[env] || null
  }, [caps.data])
  return { lookup, isLoading: caps.isLoading, error: caps.error }
}

export function useRuns(params = {}) {
  return useQuery({
    queryKey: keys.runs(params),
    queryFn: () => api.runs(params),
    placeholderData: keepPreviousData,
    refetchInterval: (query) => ((query.state.data || []).some((r) => isActive(r.status)) ? 2000 : 15000),
  })
}

export const useModels = () => useQuery({ queryKey: keys.models, queryFn: api.models })
export const useDiagnostics = () =>
  useQuery({
    queryKey: keys.diagnostics,
    queryFn: api.diagnostics,
    refetchInterval: (q) => (JSON.stringify(q.state.data || '').includes('"running"') ? 2500 : false),
  })

/**
 * A run plus its full episode history, kept live while the run is active.
 * Status/progress arrive over the WebSocket; history is fetched incrementally
 * with `?since=` so a long run never re-downloads its whole curve.
 */
export function useLiveRun(runId) {
  const qc = useQueryClient()
  const run = useQuery({
    queryKey: keys.run(runId),
    queryFn: () => api.run(runId),
    enabled: !!runId,
    refetchInterval: (q) => (isActive(q.state.data?.status) ? 4000 : false),
  })
  const live = isActive(run.data?.status)
  const history = useQuery({
    queryKey: keys.history(runId),
    enabled: !!runId,
    queryFn: async () => {
      const prev = qc.getQueryData(keys.history(runId)) || []
      const since = prev.length ? prev[prev.length - 1].episode : 0
      const next = await api.history(runId, since)
      return since ? prev.concat(next) : next
    },
    refetchInterval: live ? 1500 : false,
  })

  const wsRef = useRef(null)
  useEffect(() => {
    if (!runId || !live || typeof WebSocket === 'undefined') return undefined
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/runs/${runId}`)
    wsRef.current = ws
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.error) return
        qc.setQueryData(keys.run(runId), (old) =>
          old ? { ...old, status: msg.status, live: msg.live, progress: { ...old.progress, ...msg } } : old,
        )
        if (!isActive(msg.status)) {
          qc.invalidateQueries({ queryKey: keys.run(runId) })
          qc.invalidateQueries({ queryKey: keys.history(runId) })
          qc.invalidateQueries({ queryKey: ['runs'] })
        }
      } catch {
        /* ignore malformed frames */
      }
    }
    return () => ws.close()
  }, [runId, live, qc])

  return { run, history, live }
}

/** Mutations that touch runs invalidate every run list. */
export function useRunMutation(fn, opts = {}) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: fn,
    ...opts,
    onSuccess: (data, vars, ctx) => {
      qc.invalidateQueries({ queryKey: ['runs'] })
      qc.invalidateQueries({ queryKey: ['run'] })
      qc.invalidateQueries({ queryKey: keys.diagnostics })
      opts.onSuccess?.(data, vars, ctx)
    },
  })
}
