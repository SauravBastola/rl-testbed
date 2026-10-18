import { CheckCircle2, CircleSlash } from 'lucide-react'
import { useSystem } from '../../lib/queries.js'
import { fmtBytes } from '../../lib/format.js'
import { Modal } from '../ui/Overlay.jsx'
import { ErrorState, SkeletonRows } from '../ui/Feedback.jsx'
import { KeyValue } from '../ui/Page.jsx'

const EXTRA_LABELS = {
  box2d: 'Box2D (LunarLander, BipedalWalker)',
  mujoco: 'MuJoCo (continuous control)',
  minatar: 'MinAtar (perception)',
  pettingzoo: 'PettingZoo (two-player games)',
  rlcard: 'RLCard (poker)',
  minigrid: 'MiniGrid',
  torch: 'PyTorch',
}

export function SystemDrawer({ open, onClose }) {
  const sys = useSystem(open)
  const d = sys.data
  return (
    <Modal
      open={open}
      onClose={onClose}
      side
      title="Workspace"
      description="This machine, its storage and installed extras."
    >
      {sys.isLoading && <SkeletonRows rows={8} />}
      {sys.error && <ErrorState error={sys.error} onRetry={sys.refetch} title="Server unreachable" />}
      {d && (
        <div className="space-y-6">
          <section>
            <h3 className="mb-2 text-body-s font-semibold">Runtime</h3>
            <KeyValue
              rows={[
                ['Version', d.version],
                ['Python', d.python],
                [
                  'PyTorch',
                  d.torch?.version ? `${d.torch.version}${d.torch.cuda ? ' · CUDA' : ' · CPU'}` : 'not available',
                ],
                ['Platform', d.platform],
                [
                  'Concurrent runs',
                  `${d.jobs.running} running, ${d.jobs.queued} queued, limit ${d.jobs.max_concurrent}`,
                ],
              ]}
            />
          </section>
          <section>
            <h3 className="mb-2 text-body-s font-semibold">Storage</h3>
            <KeyValue
              rows={[
                ['Run artifacts', fmtBytes(d.usage.runs_bytes)],
                ['Kept models', fmtBytes(d.usage.models_bytes)],
                ['Free disk', fmtBytes(d.usage.disk_free_bytes)],
                [
                  'Database',
                  <code key="db" className="break-all text-caption">
                    {d.paths.database}
                  </code>,
                ],
                [
                  'Runs folder',
                  <code key="r" className="break-all text-caption">
                    {d.paths.runs}
                  </code>,
                ],
              ]}
            />
          </section>
          <section>
            <h3 className="mb-2 text-body-s font-semibold">Optional extras</h3>
            <ul className="space-y-1.5">
              {Object.entries(d.extras).map(([k, ok]) => (
                <li key={k} className="flex items-center gap-2 text-body-s">
                  {ok ? (
                    <CheckCircle2 size={14} className="text-success" aria-hidden />
                  ) : (
                    <CircleSlash size={14} className="text-subtle" aria-hidden />
                  )}
                  <span className={ok ? 'text-text' : 'text-muted'}>{EXTRA_LABELS[k] || k}</span>
                  <span className="ml-auto text-caption text-subtle">{ok ? 'Installed' : 'Not installed'}</span>
                </li>
              ))}
            </ul>
          </section>
          <section>
            <h3 className="mb-2 text-body-s font-semibold">Catalogue</h3>
            <KeyValue
              rows={[
                ['Environments', `${d.catalogue.envs_available} of ${d.catalogue.envs} available`],
                ['Algorithms', `${d.catalogue.agents_implemented} of ${d.catalogue.agents} trainable`],
              ]}
            />
          </section>
        </div>
      )}
    </Modal>
  )
}
