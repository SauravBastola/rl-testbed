/**
 * One function per REST endpoint. Components never call fetch directly; they
 * go through the query hooks in ./queries.js, which wrap these.
 */

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(path, {
      headers: options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' },
      ...options,
    })
  } catch {
    throw new ApiError('The testbed server is not reachable. Is `rltestbed serve` running?', 0)
  }
  if (!res.ok) {
    let detail = null
    let message = res.statusText || `Request failed (${res.status})`
    try {
      const body = await res.json()
      detail = body.detail
      if (typeof detail === 'string') message = detail
      else if (detail?.message) {
        message = detail.reasons?.length ? `${detail.message}: ${detail.reasons.join(' ')}` : detail.message
      } else if (Array.isArray(detail)) {
        message = detail.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ')
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(message, res.status, detail)
  }
  return res.json()
}

const q = (params) => {
  const s = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') s.set(k, Array.isArray(v) ? v.join(',') : v)
  })
  const str = s.toString()
  return str ? `?${str}` : ''
}
const post = (body) => ({ method: 'POST', body: JSON.stringify(body ?? {}) })
const patch = (body) => ({ method: 'PATCH', body: JSON.stringify(body ?? {}) })

export const api = {
  // catalogue
  labs: () => request('/api/labs'),
  envs: () => request('/api/envs'),
  env: (name) => request(`/api/envs/${encodeURIComponent(name)}`),
  agents: () => request('/api/agents'),
  agent: (name) => request(`/api/agents/${encodeURIComponent(name)}`),
  agentCode: (name) => request(`/api/agents/${encodeURIComponent(name)}/code`),
  capabilities: () => request('/api/capabilities'),
  compatibility: (agent, env) => request(`/api/compatibility${q({ agent, env })}`),
  // content
  learningPath: () => request('/api/learning-path'),
  glossary: () => request('/api/glossary'),
  system: () => request('/api/system'),
  diagnostics: () => request('/api/diagnostics'),
  runDiagnostics: (body) => request('/api/diagnostics/run', post(body)),
  // runs
  runs: (params = {}) => request(`/api/runs${q(params)}`),
  run: (id) => request(`/api/runs/${id}`),
  history: (id, since = 0) => request(`/api/runs/${id}/history${q({ since: since || undefined })}`),
  createRun: (body) => request('/api/runs', post(body)),
  updateRun: (id, body) => request(`/api/runs/${id}`, patch(body)),
  startRun: (id) => request(`/api/runs/${id}/start`, post()),
  stopRun: (id) => request(`/api/runs/${id}/stop`, post()),
  cloneRun: (id) => request(`/api/runs/${id}/clone`, post()),
  deleteRun: (id) => request(`/api/runs/${id}`, { method: 'DELETE' }),
  keepModel: (id, body) => request(`/api/runs/${id}/keep-model`, post(body)),
  logs: (id, tail = 400) => request(`/api/runs/${id}/logs${q({ tail })}`),
  artifacts: (id) => request(`/api/runs/${id}/artifacts`),
  artifactUrl: (id, path) => `/api/runs/${id}/artifacts/${path.split('/').map(encodeURIComponent).join('/')}`,
  // analysis
  groups: () => request('/api/groups'),
  group: (id, metric) => request(`/api/groups/${id}${q({ metric })}`),
  aggregate: (runIds, metric = 'reward', groupBy = 'config') =>
    request(`/api/compare/aggregate${q({ run_ids: runIds, metric, group_by: groupBy })}`),
  rollout: (body) => request('/api/rollout', post(body)),
  // models
  models: () => request('/api/models'),
  updateModel: (id, body) => request(`/api/models/${id}`, patch(body)),
  deleteModel: (id) => request(`/api/models/${id}`, { method: 'DELETE' }),
  evaluateModel: (id, episodes = 20) => request(`/api/models/${id}/evaluate`, post({ episodes })),
}
