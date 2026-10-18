import { lazy } from 'react'
import { Navigate, Route, Routes, useParams } from 'react-router-dom'
import { AppShell } from '../components/shell/AppShell.jsx'
import { useApp } from './AppContext.jsx'
import LabOverview from '../pages/LabOverview.jsx'
// Imported directly: several detail pages render it, so it lives in the main chunk.
import NotFound from '../pages/NotFound.jsx'

// Everything except the landing page is split into its own chunk.
const Composer = lazy(() => import('../pages/Composer.jsx'))
const Runs = lazy(() => import('../pages/Runs.jsx'))
const RunDetail = lazy(() => import('../pages/RunDetail.jsx'))
const Compare = lazy(() => import('../pages/Compare.jsx'))
const Experiments = lazy(() => import('../pages/Experiments.jsx'))
const Sweeps = lazy(() => import('../pages/Sweeps.jsx'))
const SweepComposer = lazy(() => import('../pages/SweepComposer.jsx'))
const SweepDetail = lazy(() => import('../pages/SweepDetail.jsx'))
const Models = lazy(() => import('../pages/Models.jsx'))
const Environments = lazy(() => import('../pages/Environments.jsx'))
const EnvironmentDetail = lazy(() => import('../pages/EnvironmentDetail.jsx'))
const Algorithms = lazy(() => import('../pages/Algorithms.jsx'))
const AlgorithmDetail = lazy(() => import('../pages/AlgorithmDetail.jsx'))
const Diagnostics = lazy(() => import('../pages/Diagnostics.jsx'))
const Visualize = lazy(() => import('../pages/Visualize.jsx'))
const Learn = lazy(() => import('../pages/Learn.jsx'))
const Glossary = lazy(() => import('../pages/Glossary.jsx'))

/** Old v2 URLs redirect into the current lab so bookmarks keep working. */
function ToLab({ page = '' }) {
  const { labId } = useApp()
  const suffix = typeof window !== 'undefined' ? window.location.search : ''
  return <Navigate to={`/lab/${labId}${page}${suffix}`} replace />
}

function LegacyLabRedirect() {
  const { labId } = useParams()
  return <Navigate to={`/lab/${labId}`} replace />
}

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<ToLab />} />
        <Route path="lab/:labId" element={<LabOverview />} />
        <Route path="lab/:labId/new" element={<Composer />} />
        <Route path="lab/:labId/runs" element={<Runs />} />
        <Route path="lab/:labId/envs" element={<Environments />} />
        <Route path="lab/:labId/algorithms" element={<Algorithms />} />
        <Route path="lab/:labId/sweeps" element={<Sweeps />} />
        <Route path="lab/:labId/sweeps/new" element={<SweepComposer />} />
        <Route path="lab/:labId/*" element={<LegacyLabRedirect />} />

        <Route path="runs/:runId" element={<RunDetail />} />
        <Route path="sweeps/:sweepId" element={<SweepDetail />} />
        <Route path="envs/:envName" element={<EnvironmentDetail />} />
        <Route path="algorithms/:agentName" element={<AlgorithmDetail />} />
        <Route path="compare" element={<Compare />} />
        <Route path="experiments" element={<Experiments />} />
        <Route path="models" element={<Models />} />
        <Route path="visualize" element={<Visualize />} />
        <Route path="diagnostics" element={<Diagnostics />} />
        <Route path="learn" element={<Learn />} />
        <Route path="glossary" element={<Glossary />} />

        {/* v2 routes */}
        <Route path="train" element={<ToLab page="/new" />} />
        <Route path="new" element={<ToLab page="/new" />} />
        <Route path="runs" element={<ToLab page="/runs" />} />
        <Route path="envs" element={<ToLab page="/envs" />} />
        <Route path="algorithms" element={<ToLab page="/algorithms" />} />
        <Route path="benchmarks" element={<Navigate to="/diagnostics" replace />} />

        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
