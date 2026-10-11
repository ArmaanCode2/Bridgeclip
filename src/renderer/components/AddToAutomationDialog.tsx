import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { Plus, X } from 'lucide-react'
import type { Automation } from '../../shared/automations'
import { getApi } from '../lib/ipc'
import { errorMessage } from '../lib/utils'
import { Button } from './ui/Button'
import { Dialog, DialogFooter } from './ui/Dialog'

interface Props {
  outputDir: string
  clipIndices: number[]
  onClose: () => void
  onAdded: (name: string, id: string) => void
}

export function AddToAutomationDialog({ outputDir, clipIndices, onClose, onAdded }: Props): React.JSX.Element {
  const titleId = useId()
  const dialogRef = useRef<HTMLDivElement>(null)
  const [automations, setAutomations] = useState<Automation[] | null>(null)
  const [selectedId, setSelectedId] = useState('')
  const [newName, setNewName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const count = clipIndices.length
  const selected = automations?.find((item) => item.id === selectedId)
  const close = useCallback(() => { if (!busy) onClose() }, [busy, onClose])

  useEffect(() => {
    let active = true
    getApi().automations.list().then((items) => {
      if (!active) return
      setAutomations(items)
      setSelectedId(items[0]?.id ?? '')
    }).catch((cause) => {
      if (active) setError(errorMessage(cause, 'Could not load automations.'))
    })
    return () => { active = false }
  }, [])

  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const dialog = dialogRef.current
    dialog?.focus()
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') { event.preventDefault(); close(); return }
      if (event.key !== 'Tab' || !dialog) return
      const focusable = [...dialog.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled])')]
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) { event.preventDefault(); first.focus() }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => { document.removeEventListener('keydown', onKeyDown); previousFocus?.focus() }
  }, [close])

  const create = async (event: React.FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault()
    if (!newName.trim() || busy) return
    setBusy(true); setError(null)
    try {
      const items = await getApi().automations.create(newName.trim())
      setAutomations(items)
      setSelectedId(items[0].id)
      setNewName('')
    } catch (cause) { setError(errorMessage(cause, 'Could not create the automation.')) }
    finally { setBusy(false) }
  }

  const add = async (): Promise<void> => {
    if (!selected || busy) return
    setBusy(true); setError(null)
    try {
      await getApi().automations.addLibraryClips(selected.id, outputDir, clipIndices)
      try { sessionStorage.setItem('bridgeclip.automations.selectedId', selected.id) } catch { /* Optional selection memory. */ }
      onAdded(selected.name, selected.id)
    } catch (cause) { setError(errorMessage(cause, 'Could not add clips to the content bank.')) }
    finally { setBusy(false) }
  }

  return (
    <Dialog ref={dialogRef} aria-labelledby={titleId} onBackdropMouseDown={close} panelClassName="max-w-[480px]">
      <div className="flex items-start justify-between gap-3 px-5 pt-5">
        <div><h2 id={titleId} className="text-lg font-semibold text-ink">Add to content bank</h2><p className="mt-1 text-sm text-ink-muted">{count} selected clip{count === 1 ? '' : 's'} will be copied into an automation.</p></div>
        <Button variant="ghost" iconOnly aria-label="Close" icon={<X className="h-4 w-4" />} onClick={close} />
      </div>
      <div className="space-y-3 overflow-y-auto px-5 py-5">
        {error && <p role="alert" className="rounded-lg border border-danger/30 bg-danger/[0.06] p-3 text-sm text-danger">{error}</p>}
        {automations === null && !error && <p className="text-sm text-ink-muted">Loading automations…</p>}
        {automations?.length === 0 && <p className="text-sm text-ink-muted">Create an automation to start its content bank.</p>}
        {automations?.map((item) => {
          const queued = item.content.filter((clip) => clip.status === 'queued').length
          const full = item.content.length + count > 500
          return <label key={item.id} className="flex cursor-pointer items-center gap-3 rounded-lg border border-line bg-fill p-3 text-sm text-ink">
            <input type="radio" name="automation" className="h-4 w-4 accent-accent" checked={selectedId === item.id} disabled={busy || full} onChange={() => setSelectedId(item.id)} />
            <span className="min-w-0 flex-1 truncate font-medium">{item.name}</span>
            <span className="shrink-0 text-xs text-ink-muted">{full ? 'Bank full' : `${queued} queued`}</span>
          </label>
        })}
        <form onSubmit={(event) => void create(event)} className="flex gap-2 border-t border-line pt-3">
          <input aria-label="New automation name" className="min-w-0 flex-1 rounded-md border border-line bg-fill px-3 py-1.5 text-sm text-ink outline-none focus:border-line-strong" placeholder="New automation name" maxLength={80} value={newName} disabled={busy} onChange={(event) => setNewName(event.target.value)} />
          <Button type="submit" size="sm" icon={<Plus className="h-3.5 w-3.5" />} disabled={busy || !newName.trim()}>Create</Button>
        </form>
      </div>
      <DialogFooter>
        <Button onClick={close} disabled={busy}>Cancel</Button>
        <Button variant="primary" loading={busy} disabled={!selected || selected.content.length + count > 500} onClick={() => void add()}>Add {count} clip{count === 1 ? '' : 's'}</Button>
      </DialogFooter>
    </Dialog>
  )
}
