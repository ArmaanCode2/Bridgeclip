import { useCallback, useEffect, useId, useReducer, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import { Check, CircleAlert, Download, HardDrive, Loader2, RefreshCw, Trash2, Zap } from 'lucide-react'
import { useSettingsStore } from '../store/use-settings-store'
import { useJobStore, isJobActive } from '../store/use-job-store'
import { getApi } from '../lib/ipc'
import { cn, errorMessage } from '../lib/utils'
import { LOCAL_WHISPER_MODELS } from '../../shared/providers'
import {
  DEVICE_EXPLANATIONS, GPU_LIBRARIES_DISK_MB, INITIAL_DELETE_CONFIRM, WHISPER_SETUP_MESSAGES, deleteBlockedReason,
  deleteConfirmReducer, deviceNote, downloadNote, downloadProgressText, fitWarning, formatBytes, formatMemory, gpuTestModel,
  setupResultMessage,
  type LocalWhisperDevice, type WhisperListing, type WhisperModelEntry, type WhisperModelName, type WhisperSetupProgress, type WhisperSetupResult
} from '../../shared/whisper-models'
import type { CommitSettings } from './ProviderSetup'
import { Badge } from './ui/Badge'
import { Button } from './ui/Button'
import { ProgressBar } from './ui/ProgressBar'
import { Segmented } from './ui/Segmented'

const BACKEND_LABEL = { 'openai-whisper': 'openai-whisper', 'faster-whisper': 'faster-whisper' } as const

/** Arrow keys move between the model radios, which sit in different rows. */
function onModelKeyDown(event: ReactKeyboardEvent<HTMLButtonElement>): void {
  const group = event.currentTarget.closest('[role="radiogroup"]')
  const radios = Array.from(group?.querySelectorAll<HTMLButtonElement>('[role="radio"]:not(:disabled)') ?? [])
  const current = radios.indexOf(event.currentTarget)
  if (current < 0) return
  let next = current
  if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (current + 1) % radios.length
  else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (current - 1 + radios.length) % radios.length
  else if (event.key === 'Home') next = 0
  else if (event.key === 'End') next = radios.length - 1
  else return
  event.preventDefault()
  radios[next].focus()
  radios[next].click()
}

function ModelRow({ entry, selected, tabbable, device, listing, anyJobRunning, confirm, setup, onSelect, onAsk, onCancel, onDelete, onInstall, onStopSetup }: {
  entry: WhisperModelEntry
  selected: boolean
  /** The one radio in the group that Tab reaches. */
  tabbable: boolean
  device: LocalWhisperDevice
  listing: WhisperListing
  anyJobRunning: boolean
  confirm: { pending: string | null; busy: boolean }
  /** The download or GPU setup running now, if any. */
  setup: WhisperSetupProgress | null
  onSelect: () => void
  onAsk: () => void
  onCancel: () => void
  onDelete: () => void
  onInstall: () => void
  onStopSetup: () => void
}): React.JSX.Element {
  const info = LOCAL_WHISPER_MODELS.find((model) => model.id === entry.name)
  const warning = fitWarning({ memory: entry.approxMemoryMB, device, system: listing.system })
  const asking = confirm.pending === entry.name
  const blocked = deleteBlockedReason({ jobRunning: anyJobRunning, busy: confirm.busy })
  const downloading = setup?.task === 'download' && setup.model === entry.name ? setup : null
  const installBlocked = !listing.engine.available ? WHISPER_SETUP_MESSAGES.noEngine : setup ? WHISPER_SETUP_MESSAGES.busy : null

  return (
    <div className={cn('glass-tile flex flex-col gap-2 rounded-xl p-3 sm:flex-row sm:items-start sm:gap-3', selected && 'glass-selected')}>
      <button
        type="button"
        role="radio"
        aria-checked={selected}
        tabIndex={tabbable ? 0 : -1}
        onKeyDown={onModelKeyDown}
        onClick={onSelect}
        className="min-w-0 flex-1 rounded-lg text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
      >
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="text-sm font-semibold text-ink">{info?.label ?? entry.name}</span>
          {selected && <Check aria-label="In use" className="h-3.5 w-3.5 text-success" strokeWidth={3} />}
          {entry.installed
            ? <Badge tone="success">Installed</Badge>
            : <Badge tone="neutral">Not installed</Badge>}
        </span>
        {info && <span className="mt-1 block text-xs leading-relaxed text-ink-muted">{info.hint}</span>}
        <span className="mt-1 block text-2xs leading-relaxed text-ink-subtle">
          {entry.installed
            ? `${entry.sizeOnDisk ? `${formatBytes(entry.sizeOnDisk)} on disk · ` : ''}${entry.backend ? BACKEND_LABEL[entry.backend] : ''}`
            : info ? downloadNote(info.downloadMb) : null}
        </span>
        <span className="mt-0.5 block text-2xs leading-relaxed text-ink-subtle">
          About {formatMemory(entry.approxMemoryMB.cpu)} of memory on the CPU · {formatMemory(entry.approxMemoryMB.gpu)} of video memory on a GPU
        </span>
        {warning && (
          <span className="mt-1 flex items-start gap-1.5 text-xs text-warning">
            <CircleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{warning}</span>
          </span>
        )}
      </button>

      {!entry.installed && (
        <div className="flex shrink-0 flex-col gap-1.5 sm:w-60">
          {downloading ? (
            <>
              <div className="flex items-center justify-between gap-2">
                <span role="status" aria-live="polite" className="text-xs text-ink-muted">
                  {downloadProgressText(downloading.done, downloading.total)}
                </span>
                <Button size="sm" variant="ghost" onClick={onStopSetup} aria-label={`Stop downloading ${entry.name}`}>Cancel</Button>
              </div>
              <ProgressBar value={downloading.total > 0 ? (downloading.done / downloading.total) * 100 : 0} />
            </>
          ) : (
            <Button
              size="sm"
              onClick={onInstall}
              disabled={Boolean(installBlocked)}
              tooltip={installBlocked ?? undefined}
              aria-label={`Install ${entry.name}`}
              icon={<Download className="h-3.5 w-3.5" />}
              className="sm:self-end"
            >
              Install
            </Button>
          )}
        </div>
      )}

      {entry.installed && entry.backend && (
        <div className="flex shrink-0 flex-wrap items-center gap-2 sm:justify-end">
          {asking ? (
            <>
              <span className="text-xs text-ink-muted">Delete {entry.name}? It will download again if you use it.</span>
              <Button size="sm" variant="ghost" onClick={onCancel} disabled={confirm.busy}>Cancel</Button>
              <Button size="sm" variant="danger" onClick={onDelete} loading={confirm.busy} icon={<Trash2 className="h-3.5 w-3.5" />}>
                Delete
              </Button>
            </>
          ) : (
            <Button
              size="sm"
              variant="ghost"
              onClick={onAsk}
              disabled={Boolean(blocked)}
              tooltip={blocked ?? undefined}
              aria-label={`Delete ${entry.name}`}
              icon={<Trash2 className="h-3.5 w-3.5" />}
            >
              Delete
            </Button>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * Settings → Transcription → Local Whisper models. Shown for built-in Whisper only. Nothing is
 * read from disk, and no GPU query is made, until this panel is on screen.
 */
export function WhisperModelsPanel({ commit }: { commit: CommitSettings }): React.JSX.Element {
  const localWhisperModel = useSettingsStore((s) => s.localWhisperModel)
  const device = useSettingsStore((s) => s.localWhisperDevice)
  const anyJobRunning = useJobStore((s) => Object.values(s.jobs).some(isJobActive))
  const headingId = useId()
  const [listing, setListing] = useState<WhisperListing | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [actionMessage, setActionMessage] = useState<{ ok: boolean; text: string } | null>(null)
  const [settingError, setSettingError] = useState<string | null>(null)
  const [confirm, dispatch] = useReducer(deleteConfirmReducer, INITIAL_DELETE_CONFIRM)
  const [setup, setSetup] = useState<WhisperSetupProgress | null>(null)
  /** True while this panel waits on a task it started, which reports its own result. */
  const initiated = useRef(false)
  const latest = useRef(0)

  const refresh = useCallback(async (): Promise<void> => {
    const request = ++latest.current
    setLoading(true)
    try {
      const next = await getApi().whisper.listModels()
      if (request !== latest.current) return
      setListing(next)
      setLoadError(null)
    } catch (err) {
      if (request === latest.current) setLoadError(errorMessage(err, 'Could not read the Whisper models on this computer.'))
    } finally {
      if (request === latest.current) setLoading(false)
    }
  }, [])

  // Load when the panel appears, and again after a settings change. A short delay lets a run of
  // arrow-key presses settle into one refresh, and the list on screen stays until the new one lands.
  useEffect(() => {
    const timer = setTimeout(() => void refresh(), 250)
    return () => clearTimeout(timer)
  }, [refresh, localWhisperModel, device])

  const choose = async (name: string): Promise<void> => {
    if (name === localWhisperModel) return
    setSettingError(await commit({ localWhisperModel: name }))
  }

  const chooseDevice = async (next: LocalWhisperDevice): Promise<void> => setSettingError(await commit({ localWhisperDevice: next }))

  // Downloads and GPU setup run in main and keep going if this panel closes. A panel opened
  // mid-task asks what is running, then follows the events; the last one carries the result.
  useEffect(() => {
    let live = true
    void getApi().whisper.setupState().then((state) => { if (live && state) setSetup(state) }).catch(() => {})
    const stop = getApi().whisper.onSetupProgress((progress) => {
      if (!progress.finished) {
        setSetup(progress)
        return
      }
      setSetup(null)
      void refresh()
      if (!initiated.current && progress.result) setActionMessage(setupResultMessage(progress, progress.result))
    })
    return () => { live = false; stop() }
  }, [refresh])

  const runTask = async (task: 'download' | 'gpu', model: WhisperModelName, start: () => Promise<WhisperSetupResult>): Promise<void> => {
    setActionMessage(null)
    initiated.current = true
    try {
      setActionMessage(setupResultMessage({ task, model }, await start()))
    } catch (err) {
      setActionMessage({ ok: false, text: errorMessage(err, task === 'download' ? 'Could not download the model.' : 'Could not set up the GPU.') })
    } finally {
      initiated.current = false
    }
  }

  const install = (entry: WhisperModelEntry): Promise<void> => runTask('download', entry.name, () => getApi().whisper.downloadModel(entry.name))

  const gpuModel = listing ? gpuTestModel(listing.models, localWhisperModel) : null
  const enableGpu = async (): Promise<void> => {
    if (gpuModel) await runTask('gpu', gpuModel, () => getApi().whisper.enableGpu(gpuModel))
  }

  const stopSetup = (): void => { void getApi().whisper.cancelSetup() }

  const remove = async (entry: WhisperModelEntry): Promise<void> => {
    if (!entry.backend) return
    dispatch({ type: 'start' })
    setActionMessage(null)
    try {
      const result = await getApi().whisper.deleteModel({ name: entry.name, backend: entry.backend })
      setActionMessage(result.ok
        ? { ok: true, text: `Deleted ${entry.name}. Freed ${formatBytes(result.freedBytes)}.` }
        : { ok: false, text: result.message })
    } catch (err) {
      setActionMessage({ ok: false, text: errorMessage(err, 'Could not delete the model.') })
    } finally {
      dispatch({ type: 'finish' })
      void refresh()
    }
  }

  const note = deviceNote(device, listing?.system ?? null, listing?.engine ?? null)
  const system = listing?.system
  const gpuSetup = setup?.task === 'gpu' ? setup : null
  const gpuBlocked = setup ? WHISPER_SETUP_MESSAGES.busy
    : anyJobRunning ? WHISPER_SETUP_MESSAGES.jobRunning
      : !gpuModel ? 'Install a model first. Enable GPU tests your GPU with it.' : null

  return (
    <section aria-labelledby={headingId} className="space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h4 id={headingId} className="text-sm font-semibold text-ink">Local Whisper models</h4>
          <p className="mt-0.5 text-xs text-ink-muted">
            Install a model once and it stays on disk. Pick the one to use; delete the ones you no longer need to free space.
          </p>
        </div>
        <Button size="sm" variant="ghost" onClick={() => void refresh()} disabled={loading} aria-label="Refresh model list"
          icon={loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}>
          Refresh
        </Button>
      </div>

      <div className="space-y-1.5">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <span className="text-xs font-medium text-ink">Device</span>
          <Segmented
            label="Whisper device"
            size="sm"
            value={device}
            options={[{ value: 'auto', label: 'Auto' }, { value: 'cpu', label: 'CPU' }, { value: 'gpu', label: 'GPU' }]}
            onChange={(value) => void chooseDevice(value)}
          />
        </div>
        <p className="text-xs text-ink-muted">{DEVICE_EXPLANATIONS[device]}</p>
        {note && (
          <p role="alert" className="flex items-start gap-1.5 text-xs text-warning">
            <CircleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{note}</span>
          </p>
        )}
        {system && (
          <p className="flex items-start gap-1.5 text-2xs text-ink-subtle">
            <HardDrive className="mt-0.5 h-3 w-3 shrink-0" />
            <span>
              This computer: {formatMemory(system.ramMB)} of memory
              {system.gpu ? ` · ${system.gpu.name} with ${formatMemory(system.gpu.vramMB)} of video memory` : ' · no NVIDIA GPU detected'}
              {listing?.engine.available ? ` · ${listing.engine.detail}` : ''}
            </span>
          </p>
        )}
        {listing && !listing.engine.available && (
          <p className="text-2xs text-warning">{listing.engine.detail}</p>
        )}
        {listing?.gpuSetup && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 pt-1">
            {gpuSetup ? (
              <>
                <span role="status" aria-live="polite" className="flex items-center gap-2 text-xs text-ink-muted">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  {gpuSetup.step === 'install'
                    ? 'Installing the GPU libraries. This can take several minutes…'
                    : 'Testing Whisper on your GPU…'}
                </span>
                <Button size="sm" variant="ghost" onClick={stopSetup}>Cancel</Button>
              </>
            ) : (
              <Button
                size="sm"
                onClick={() => void enableGpu()}
                disabled={Boolean(gpuBlocked)}
                tooltip={gpuBlocked ?? undefined}
                icon={<Zap className="h-3.5 w-3.5" />}
              >
                Enable GPU
              </Button>
            )}
            <span className="text-2xs text-ink-subtle">
              Installs NVIDIA’s CUDA libraries for Whisper once (about {formatMemory(GPU_LIBRARIES_DISK_MB)} on disk), then runs {gpuModel ?? 'an installed model'} on your GPU to check it works.
            </span>
          </div>
        )}
      </div>

      {loading && !listing && (
        <p role="status" className="flex items-center gap-2 text-xs text-ink-muted">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Looking for downloaded models…
        </p>
      )}
      {loadError && (
        <div role="alert" className="flex flex-wrap items-center gap-2 text-xs text-danger">
          <span>{loadError}</span>
          <Button size="sm" onClick={() => void refresh()}>Try again</Button>
        </div>
      )}

      {listing && (
        <div role="radiogroup" aria-label="Whisper model" aria-busy={loading} className="space-y-2">
          {listing.models.map((entry, index) => (
            <ModelRow
              key={entry.name}
              entry={entry}
              selected={entry.name === localWhisperModel}
              tabbable={entry.name === localWhisperModel || (index === 0 && !listing.models.some((m) => m.name === localWhisperModel))}
              device={device}
              listing={listing}
              anyJobRunning={anyJobRunning}
              confirm={confirm}
              setup={setup}
              onSelect={() => void choose(entry.name)}
              onAsk={() => dispatch({ type: 'ask', name: entry.name })}
              onCancel={() => dispatch({ type: 'cancel' })}
              onDelete={() => void remove(entry)}
              onInstall={() => void install(entry)}
              onStopSetup={stopSetup}
            />
          ))}
        </div>
      )}

      {actionMessage && (
        <p role={actionMessage.ok ? 'status' : 'alert'} className={cn('text-xs', actionMessage.ok ? 'text-success' : 'text-danger')}>
          {actionMessage.text}
        </p>
      )}
      {settingError && <p role="alert" className="text-xs text-danger">{settingError}</p>}
    </section>
  )
}
