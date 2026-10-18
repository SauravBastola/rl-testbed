import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Boxes, Eye, Gauge, Pencil, Star, Trash2 } from 'lucide-react'
import { useLab } from '../app/AppContext.jsx'
import { api } from '../lib/api.js'
import { keys, useCatalogue, useModels } from '../lib/queries.js'
import { fmtBytes, fmtNum, fmtRelative } from '../lib/format.js'
import { cn } from '../lib/utils.js'
import { Button, ButtonLink, IconButton } from '../components/ui/Button.jsx'
import { EmptyState, ErrorState, Notice, SkeletonRows } from '../components/ui/Feedback.jsx'
import { Field, Input, SearchInput, Textarea } from '../components/ui/Field.jsx'
import { Modal } from '../components/ui/Overlay.jsx'
import { PageHeader, Stat } from '../components/ui/Page.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { Table, THead, Th, Tr, Td } from '../components/ui/Table.jsx'
import { useToast } from '../components/ui/Toast.jsx'

function EvaluateModal({ model, onClose }) {
  const [episodes, setEpisodes] = useState(20)
  const ev = useMutation({ mutationFn: () => api.evaluateModel(model.id, Number(episodes)) })
  return (
    <Modal
      open
      onClose={onClose}
      title={`Evaluate ${model.name}`}
      description="Runs the greedy policy for a number of fresh episodes. Nothing is trained."
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Close
          </Button>
          <Button variant="primary" icon={Gauge} loading={ev.isPending} onClick={() => ev.mutate()}>
            Evaluate
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <Field label="Episodes">
          {(id) => (
            <Input
              id={id}
              type="number"
              min={1}
              max={1000}
              className="num"
              value={episodes}
              onChange={(e) => setEpisodes(e.target.value)}
            />
          )}
        </Field>
        {ev.error && <Notice tone="danger">{ev.error.message}</Notice>}
        {ev.data && (
          <div className="grid grid-cols-2 gap-4 rounded-lg border border-border bg-surface-2 p-4 sm:grid-cols-4">
            <Stat label="Mean" value={fmtNum(ev.data.mean)} tone="accent" />
            <Stat label="Std" value={fmtNum(ev.data.std)} />
            <Stat label="Min" value={fmtNum(ev.data.min)} />
            <Stat label="Max" value={fmtNum(ev.data.max)} />
          </div>
        )}
      </div>
    </Modal>
  )
}

function EditModal({ model, onClose, onSave, saving }) {
  const [name, setName] = useState(model.name)
  const [description, setDescription] = useState(model.description || '')
  return (
    <Modal
      open
      onClose={onClose}
      title="Edit model"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" loading={saving} onClick={() => onSave({ name, description })}>
            Save
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <Field label="Name">{(id) => <Input id={id} value={name} onChange={(e) => setName(e.target.value)} />}</Field>
        <Field label="Description">
          {(id) => <Textarea id={id} value={description} onChange={(e) => setDescription(e.target.value)} />}
        </Field>
      </div>
    </Modal>
  )
}

export default function Models() {
  const models = useModels()
  const { envBy, agentBy } = useCatalogue()
  const { labId } = useLab()
  const qc = useQueryClient()
  const { toast } = useToast()
  const [q, setQ] = useState('')
  const [evaluating, setEvaluating] = useState(null)
  const [editing, setEditing] = useState(null)
  const [deleting, setDeleting] = useState(null)

  const invalidate = () => qc.invalidateQueries({ queryKey: keys.models })
  const update = useMutation({
    mutationFn: ({ id, body }) => api.updateModel(id, body),
    onSuccess: () => {
      invalidate()
      setEditing(null)
    },
    onError: (e) => toast({ title: 'Could not save', description: e.message, tone: 'danger' }),
  })
  const del = useMutation({
    mutationFn: (id) => api.deleteModel(id),
    onSuccess: () => {
      invalidate()
      setDeleting(null)
      toast({ title: 'Model deleted' })
    },
  })

  const list = (models.data || []).filter((m) =>
    q
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean)
      .every((t) => `${m.name} ${m.agent} ${m.env} ${m.description}`.toLowerCase().includes(t)),
  )

  return (
    <div>
      <PageHeader
        title="Models"
        description="Checkpoints you chose to keep. Re-evaluate them, watch them play, or compare them with fresh runs."
        actions={<SearchInput value={q} onChange={setQ} placeholder="Search models" className="w-64" />}
      />
      {models.error ? (
        <ErrorState error={models.error} onRetry={models.refetch} />
      ) : models.isLoading ? (
        <SkeletonRows rows={5} />
      ) : !models.data.length ? (
        <EmptyState
          icon={Boxes}
          title="No kept models yet"
          description="When a run finishes, open it and press Keep model. Its best checkpoint is copied here so deleting the run does not lose it."
          action={
            <ButtonLink to={`/lab/${labId}/runs?status=completed`} variant="primary">
              Find a completed run
            </ButtonLink>
          }
        />
      ) : (
        <Table>
          <THead>
            <Th>Model</Th>
            <Th>Trained</Th>
            <Th align="right">Best eval</Th>
            <Th align="right">Size</Th>
            <Th>Source</Th>
            <Th align="right">Actions</Th>
          </THead>
          <tbody>
            {list.map((m) => (
              <Tr key={m.id}>
                <Td className="max-w-80">
                  <span className="flex items-center gap-2">
                    <button
                      type="button"
                      aria-label={m.is_favorite ? 'Unstar' : 'Star'}
                      aria-pressed={m.is_favorite}
                      onClick={() => update.mutate({ id: m.id, body: { is_favorite: !m.is_favorite } })}
                      className={cn('rounded p-0.5', m.is_favorite ? 'text-warning' : 'text-subtle hover:text-muted')}
                    >
                      <Star size={14} aria-hidden fill={m.is_favorite ? 'currentColor' : 'none'} />
                    </button>
                    <span className="truncate font-medium text-text">{m.name}</span>
                    {m.checkpoint_type === 'imported' && <Badge>imported</Badge>}
                  </span>
                  {m.description && (
                    <span className="block truncate pl-6 text-caption text-subtle">{m.description}</span>
                  )}
                </Td>
                <Td className="text-muted">
                  {agentBy[m.agent]?.label || m.agent} on {envBy[m.env]?.label || m.env}
                </Td>
                <Td align="right" mono>
                  {fmtNum(m.best_eval_reward)}
                </Td>
                <Td align="right" mono className="text-muted">
                  {fmtBytes(m.size_bytes)}
                </Td>
                <Td className="text-caption text-subtle">
                  {m.run_id ? (
                    <Link to={`/runs/${m.run_id}`} className="num text-accent hover:underline">
                      {m.run_id}
                    </Link>
                  ) : (
                    'Imported'
                  )}{' '}
                  · {fmtRelative(m.created_at)}
                </Td>
                <Td align="right">
                  <span className="inline-flex gap-1">
                    <IconButton size="sm" icon={Gauge} label="Evaluate" onClick={() => setEvaluating(m)} />
                    <ButtonLink
                      to={`/visualize?env=${m.env}&model=${m.id}`}
                      size="sm"
                      variant="ghost"
                      icon={Eye}
                      aria-label="Watch it play"
                      className="w-8 px-0"
                    />
                    <IconButton size="sm" icon={Pencil} label="Edit" onClick={() => setEditing(m)} />
                    <IconButton size="sm" icon={Trash2} label="Delete" onClick={() => setDeleting(m)} />
                  </span>
                </Td>
              </Tr>
            ))}
          </tbody>
        </Table>
      )}
      {evaluating && <EvaluateModal model={evaluating} onClose={() => setEvaluating(null)} />}
      {editing && (
        <EditModal
          model={editing}
          onClose={() => setEditing(null)}
          saving={update.isPending}
          onSave={(body) => update.mutate({ id: editing.id, body })}
        />
      )}
      <Modal
        open={!!deleting}
        onClose={() => setDeleting(null)}
        title="Delete this model?"
        description="The checkpoint file is removed. The run it came from is not affected."
        size="sm"
        footer={
          <>
            <Button variant="ghost" onClick={() => setDeleting(null)}>
              Cancel
            </Button>
            <Button variant="danger" icon={Trash2} loading={del.isPending} onClick={() => del.mutate(deleting.id)}>
              Delete model
            </Button>
          </>
        }
      >
        <p className="text-body-s text-muted">{deleting?.name}</p>
      </Modal>
    </div>
  )
}
