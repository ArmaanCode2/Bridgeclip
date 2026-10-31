import { useEffect, useId, useState } from 'react'
import { Check, CircleAlert, CircleCheck, Loader2, Minus, Pencil, PlugZap, Plus, Trash2, X } from 'lucide-react'
import { useSettingsStore, whenSettingsSaved, type SettingsPatch } from '../store/use-settings-store'
import type { useApiKeyDrafts } from '../hooks/use-api-key-drafts'
import { getApi } from '../lib/ipc'
import { cn, errorMessage } from '../lib/utils'
import { PROVIDER_LINKS } from '../config/brand'
import {
  AI_PROVIDER_LABELS, DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_COMPATIBLE_TIMEOUT, GEMINI_MODEL_SUGGESTIONS, MAX_MODEL_PRESETS, MAX_MODEL_PRESET_NAME,
  OPENAI_COMPATIBLE_TIMEOUTS, PROVIDER_CHECK_LABELS, aiProviderChange, modelPresetLabel,
  testSubject, visionAdvice,
  checkEndpointUrl, checkLocalWhisperUrl, isLocalEndpoint, isSafeModelName, localWhisperModeChange, looksLikeOllama,
  type AiProvider, type OpenaiCompatibleTimeout, type ProviderCheck, type TranscriptionSource
} from '../../shared/providers'
import type { LocalWhisperStatus, ProviderTestResult, PublicModelPreset } from '../../preload/index'
import { ApiKeyInput } from './ApiKeyInput'
import { ClaudePlanFields } from './ClaudePlanFields'
import { WhisperModelsPanel } from './WhisperModelsPanel'
import { LOCAL_WHISPER_LANGUAGES, type LocalWhisperLanguage } from '../../shared/whisper-models'
import { Select } from './ui/Select'
import { Badge } from './ui/Badge'
import { Button } from './ui/Button'
import { Callout } from './ui/Callout'
import { Field, TextInput } from './ui/Field'
import { Segmented } from './ui/Segmented'
import { SettingRow } from './ui/SettingRow'
import { Switch } from './ui/Switch'

/**
 * Saves part of the settings. Resolves to null on success or to a message the person can
 * read, so the field that asked can show it right where they typed.
 */
export type CommitSettings = (patch: SettingsPatch) => Promise<string | null>

type Keys = ReturnType<typeof useApiKeyDrafts>

const PROVIDER_CHOICES: { id: 'claude-plan' | 'openrouter' | 'gemini'; title: string; description: string }[] = [
  { id: 'claude-plan', title: AI_PROVIDER_LABELS['claude-plan'], description: 'Picks clips with your own plan, through Claude Code on this computer. No key needed.' },
  { id: 'openrouter', title: AI_PROVIDER_LABELS.openrouter, description: 'One key for everything. Includes Jev review and automatic failover.' },
  { id: 'gemini', title: AI_PROVIDER_LABELS.gemini, description: 'Your own Google AI Studio key. Picks clips and frames the shots.' }
]

/**
 * Arrow keys move between the cards of the group. Unlike onRadioKeyDown, the radios are not
 * siblings: each saved model's radio shares its card with Edit and Delete buttons.
 */
function onCardKeyDown(event: React.KeyboardEvent<HTMLButtonElement>): void {
  const radios = Array.from(
    event.currentTarget.closest('[role="radiogroup"]')?.querySelectorAll<HTMLButtonElement>('[role="radio"]:not(:disabled)') ?? []
  )
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

/** The host and port of a saved model's endpoint, for its card. */
function endpointHost(url: string): string {
  try { return new URL(url).host } catch { return url }
}

/**
 * Whether transcription was moved to Local Whisper by choosing a provider, rather than by the
 * person. Kept for the app session, so reopening Settings and going back to OpenRouter still undoes it.
 */
let transcriptionMovedByProvider = false

/** The tick on the card in use. */
function Chosen({ chosen }: { chosen: boolean }): React.JSX.Element | null {
  return chosen
    ? <CircleCheck aria-hidden className="h-4 w-4 shrink-0 text-accent" strokeWidth={2.5} />
    : null
}

/**
 * Who picks the clips: your Claude plan, OpenRouter, Gemini, or one of your saved models. One radio group of cards,
 * so exactly one has the tick, and arrow keys move between them. "+ Add model" saves another model
 * as a card; Edit changes one and lets you test it without choosing it.
 */
export function ProviderPicker({ commit, compact = false }: { commit: CommitSettings; compact?: boolean }): React.JSX.Element {
  const aiProvider = useSettingsStore((s) => s.aiProvider)
  const openrouterConfigured = useSettingsStore((s) => s.openrouterConfigured)
  const geminiConfigured = useSettingsStore((s) => s.geminiConfigured)
  const modelPresets = useSettingsStore((s) => s.modelPresets)
  const activeModelPreset = useSettingsStore((s) => s.activeModelPreset)
  const deleteModelPreset = useSettingsStore((s) => s.deleteModelPreset)
  const [error, setError] = useState<string | null>(null)
  const transcriptionSource = useSettingsStore((s) => s.transcriptionSource)
  const [movedToLocal, setMovedToLocal] = useState(false)
  // The card being added ('new') or edited (its id), and the one waiting for a delete confirmation.
  const [editing, setEditing] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  // The card just added, whose form stays open so it can be tested.
  const [justSaved, setJustSaved] = useState<string | null>(null)
  const choose = async (id: AiProvider, presetId?: string): Promise<void> => {
    let moved: TranscriptionSource | undefined
    const failed = await commit((current) => {
      const patch = aiProviderChange(id, current, transcriptionMovedByProvider)
      moved = patch.transcriptionSource
      return presetId ? { ...patch, activeModelPreset: presetId } : patch
    })
    setError(failed)
    if (failed) return
    if (moved === 'local') transcriptionMovedByProvider = true
    if (moved === 'cloud') transcriptionMovedByProvider = false
    setMovedToLocal(transcriptionMovedByProvider)
  }
  const remove = async (id: string): Promise<void> => {
    try {
      await deleteModelPreset(id)
      setError(null)
      if (editing === id) setEditing(null)
    } catch (err) {
      setError(errorMessage(err, 'Could not delete this model.'))
    }
    setDeleting(null)
  }

  const customChosen = aiProvider === 'openai-compatible' && modelPresets.some((p) => p.id === activeModelPreset)
  const nothingChosen = aiProvider === 'openai-compatible' && !customChosen
  const editingPreset = modelPresets.find((p) => p.id === editing)
  const deletingPreset = modelPresets.find((p) => p.id === deleting)
  const tile = (chosen: boolean): string => cn(
    'glass-tile glass-tile-hover flex w-full flex-col items-start rounded-xl text-left transition-all',
    compact ? 'p-2.5' : 'p-3', chosen && 'glass-selected'
  )

  return (
    <div>
      <div role="radiogroup" aria-label="AI provider" className={cn('grid gap-2', compact ? 'grid-cols-2' : 'grid-cols-1 sm:grid-cols-3')}>
        {PROVIDER_CHOICES.map((choice, index) => {
          const chosen = aiProvider === choice.id
          const ready = choice.id === 'claude-plan' || (choice.id === 'openrouter' ? openrouterConfigured : geminiConfigured)
          return (
            <button
              key={choice.id}
              type="button"
              role="radio"
              aria-checked={chosen}
              tabIndex={chosen || (nothingChosen && index === 0) ? 0 : -1}
              onKeyDown={onCardKeyDown}
              onClick={() => void choose(choice.id)}
              className={tile(chosen)}
            >
              <span className="flex w-full items-center justify-between gap-2">
                <span className="text-sm font-semibold text-ink">{choice.title}</span>
                <Chosen chosen={chosen} />
              </span>
              {!compact && <span className="mt-1 text-2xs leading-relaxed text-ink-muted">{choice.description}</span>}
              {!ready && <span className="mt-1 text-2xs text-ink-subtle">Needs a key</span>}
            </button>
          )
        })}
        {modelPresets.map((preset) => {
          const chosen = aiProvider === 'openai-compatible' && preset.id === activeModelPreset
          const label = modelPresetLabel(preset)
          return (
            <div key={preset.id} className="relative">
              <button
                type="button"
                role="radio"
                aria-checked={chosen}
                tabIndex={chosen ? 0 : -1}
                onKeyDown={onCardKeyDown}
                onClick={() => void choose('openai-compatible', preset.id)}
                className={cn(tile(chosen), 'pr-16')}
              >
                <span className="flex w-full items-center gap-2">
                  <span className="min-w-0 truncate text-sm font-semibold text-ink" title={label}>{label}</span>
                  <Chosen chosen={chosen} />
                </span>
                <span className="mt-1 w-full truncate font-mono text-2xs text-ink-muted" title={`${preset.model} · ${preset.url}`}>
                  {preset.name ? `${preset.model} · ${endpointHost(preset.url)}` : preset.url}
                </span>
                <span className="mt-1.5 flex flex-wrap gap-1">
                  {preset.keySaved && <Badge tone="success">Key saved</Badge>}
                  {isLocalEndpoint(preset.url) && <Badge>This computer or network</Badge>}
                  {preset.vision === 'on' && <Badge>Reads images</Badge>}
                </span>
              </button>
              <span className="absolute right-1.5 top-1.5 flex gap-0.5">
                <Button size="sm" variant="ghost" iconOnly aria-label={`Edit ${label}`} tooltip="Edit or test"
                  icon={<Pencil className="h-3.5 w-3.5" />} onClick={() => { setDeleting(null); setEditing(preset.id) }} />
                <Button size="sm" variant="ghost" iconOnly aria-label={`Delete ${label}`} tooltip="Delete"
                  icon={<Trash2 className="h-3.5 w-3.5" />} onClick={() => { setEditing(null); setDeleting(preset.id) }} />
              </span>
            </div>
          )
        })}
        <button
          type="button"
          onClick={() => { setDeleting(null); setEditing('new') }}
          disabled={modelPresets.length >= MAX_MODEL_PRESETS}
          className={cn('flex w-full items-center justify-center gap-1.5 rounded-xl border border-dashed border-white/15 text-sm text-ink-muted transition-colors hover:border-white/30 hover:text-ink disabled:opacity-50',
            compact ? 'p-2.5' : 'min-h-[4.5rem] p-3')}
        >
          <Plus className="h-4 w-4" aria-hidden /> Add model
        </button>
      </div>
      {deletingPreset && (
        <Callout tone="warning" stacked className="mt-2" action={
          <span className="flex gap-2">
            <Button size="sm" variant="danger" onClick={() => void remove(deletingPreset.id)}>Delete</Button>
            <Button size="sm" variant="ghost" onClick={() => setDeleting(null)}>Cancel</Button>
          </span>
        }>
          Delete “{modelPresetLabel(deletingPreset)}” and its saved key?
          {deletingPreset.id === activeModelPreset && aiProvider === 'openai-compatible' && ' It is the model in use, so clips cannot be made until you choose another.'}
        </Callout>
      )}
      {editing && (editing === 'new' || editingPreset) && (
        <ModelPresetForm
          key={editing}
          preset={editingPreset}
          justSaved={editing === justSaved}
          onClose={() => { setEditing(null); setJustSaved(null) }}
          onSaved={(id) => { setJustSaved(id); setEditing(id) }}
          onMovedToLocal={() => { transcriptionMovedByProvider = true; setMovedToLocal(true) }}
        />
      )}
      {error && <p role="alert" className="mt-2 text-xs text-danger">{error}</p>}
      {movedToLocal && aiProvider !== 'openrouter' && transcriptionSource === 'local' && (
        <p role="status" className="mt-2 text-xs text-ink-muted">
          Transcription now uses Local Whisper on this computer, because cloud transcription needs an OpenRouter key.
        </p>
      )}
    </div>
  )
}

/** A text setting that checks itself, then saves when you leave the field. Errors show under it. */
function TextSetting({ label, hint, value, placeholder, mono = true, validate, commit, suggestions }: {
  label: string
  hint?: string
  value: string
  placeholder: string
  mono?: boolean
  /** A problem with the typed value, or null. Runs before saving. */
  validate?: (value: string) => string | null
  /** Saves the value; resolves to an error message or null. */
  commit: (value: string) => Promise<string | null>
  suggestions?: string[]
}): React.JSX.Element {
  const [draft, setDraft] = useState(value)
  const [error, setError] = useState<string | null>(null)
  const listId = useId()
  const inputId = useId()
  useEffect(() => { setDraft(value) }, [value])

  const save = async (): Promise<void> => {
    const next = draft.trim()
    if (next === value) { setError(null); return }
    const problem = validate?.(next) ?? null
    if (problem) { setError(problem); return }
    setError(await commit(next))
  }

  return (
    <Field label={label} htmlFor={inputId} hint={error ? undefined : hint}>
      <TextInput
        id={inputId}
        mono={mono}
        value={draft}
        placeholder={placeholder}
        list={suggestions && suggestions.length > 0 ? listId : undefined}
        aria-invalid={error ? true : undefined}
        onChange={(event) => { setDraft(event.target.value); if (error) setError(null) }}
        onBlur={() => void save()}
        onKeyDown={(event) => event.key === 'Enter' && event.currentTarget.blur()}
      />
      {suggestions && suggestions.length > 0 && (
        <datalist id={listId}>{suggestions.map((name) => <option key={name} value={name} />)}</datalist>
      )}
      {error && <p role="alert" className="text-xs text-danger">{error}</p>}
    </Field>
  )
}

const validateUrl = (value: string): string | null => {
  const checked = checkEndpointUrl(value)
  return checked.ok ? null : checked.error
}
const validateLocalUrl = (value: string): string | null => {
  const checked = checkLocalWhisperUrl(value)
  return checked.ok ? null : checked.error
}
const validateModel = (value: string): string | null =>
  !value || isSafeModelName(value) ? null : 'Use only letters, numbers and . _ : / @ + - (no spaces).'

function Outcome({ ok, children }: { ok: boolean; children: React.ReactNode }): React.JSX.Element {
  return (
    <p role="status" className={cn('flex items-start gap-1.5 text-xs', ok ? 'text-success' : 'text-warning')}>
      {ok ? <Check className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={3} /> : <CircleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />}
      <span>{children}</span>
    </p>
  )
}

const CHECK_STYLE: Record<ProviderCheck['state'], { Icon: typeof Check; tone: string; label: string }> = {
  ok: { Icon: Check, tone: 'text-success', label: 'Passed' },
  failed: { Icon: X, tone: 'text-danger', label: 'Failed' },
  warning: { Icon: CircleAlert, tone: 'text-warning', label: 'Unsure' },
  skipped: { Icon: Minus, tone: 'text-ink-subtle', label: 'Not needed' }
}

/** One line per check: server, key, model, JSON answers, images. */
export function CheckList({ checks }: { checks: ProviderCheck[] }): React.JSX.Element {
  return (
    <ul aria-label="Test results" className="space-y-1">
      {checks.map((check) => {
        const style = CHECK_STYLE[check.state]
        return (
          <li key={check.id} className="flex items-start gap-2 text-xs">
            <style.Icon role="img" aria-label={style.label} className={cn('mt-0.5 h-3.5 w-3.5 shrink-0', style.tone)} strokeWidth={check.state === 'ok' ? 3 : 2} />
            <span className="w-24 shrink-0 font-medium text-ink">{PROVIDER_CHECK_LABELS[check.id]}</span>
            <span className="min-w-0 text-ink-muted">{check.message}</span>
          </li>
        )
      })}
    </ul>
  )
}

/**
 * Tries the saved provider: its model list, a short JSON request with the saved model and, for a
 * custom endpoint, a small image. Saves anything still being typed first.
 */
function TestConnection({ provider, keys, onModels, preset }: {
  provider: AiProvider
  /** Keys still being typed are saved first; a saved model's key is already saved with it. */
  keys?: Keys
  onModels?: (models: string[]) => void
  /** A saved model to test, chosen or not. Its "can read images" advice changes that card. */
  preset?: PublicModelPreset
}): React.JSX.Element {
  const [testing, setTesting] = useState(false)
  const [result, setResult] = useState<ProviderTestResult | null>(null)
  const saveModelPreset = useSettingsStore((s) => s.saveModelPreset)
  const visionOn = preset?.vision === 'on'
  // A result is about the URL and model it tested. Once either changes, it no longer applies.
  const savedSubject = useSettingsStore((s) => testSubject(provider, s))
  const geminiModel = useSettingsStore((s) => s.geminiModel)
  const subject = preset
    ? testSubject('openai-compatible', { openaiCompatibleUrl: preset.url, openaiCompatibleModel: preset.model, geminiModel })
    : savedSubject
  const current = result && (result.subject === subject || !result.subject) ? result : null
  const [adviceError, setAdviceError] = useState<string | null>(null)
  const setVision = async (on: boolean): Promise<void> => {
    if (!preset) return
    try {
      await saveModelPreset({ id: preset.id, name: preset.name, url: preset.url, model: preset.model, apiKey: null, vision: on ? 'on' : 'off', timeout: preset.timeout })
      setAdviceError(null)
    } catch (err) {
      setAdviceError(errorMessage(err, 'Could not change this setting.'))
    }
  }

  const run = async (): Promise<void> => {
    setTesting(true)
    try {
      await keys?.persist()
      await whenSettingsSaved()
      const outcome = await getApi().settings.testAiProvider(provider, preset?.id)
      setResult(outcome)
      onModels?.(outcome.models)
    } catch (err) {
      setResult({ ok: false, message: errorMessage(err, 'Could not run the test.'), models: [], checks: [], readsImages: null, subject: '' })
    } finally {
      setTesting(false)
    }
  }

  const advice = !testing && preset && current ? visionAdvice(provider, current.readsImages, visionOn) : null
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-3">
        <Button size="sm" onClick={() => void run()} loading={testing} icon={<PlugZap className="h-3.5 w-3.5" />}>
          Test connection
        </Button>
        {testing && provider !== 'openrouter' && (
          <span role="status" className="text-2xs text-ink-subtle">
            {provider === 'claude-plan' ? 'Asking Claude Code a short question, which uses a little of your plan.' : 'Sending short test requests. A local model can take a minute to load.'}
          </span>
        )}
        {!testing && current && <Outcome ok={current.ok}>{current.message}</Outcome>}
      </div>
      {!testing && current && current.checks.length > 1 && <CheckList checks={current.checks} />}
      {advice === 'turn-on' && (
        <Callout tone="success" stacked action={<Button size="sm" onClick={() => void setVision(true)}>Turn on image reading</Button>}>
          This model read the test image. Turn on “This model can read images” so the AI can check facecam framing.
        </Callout>
      )}
      {advice === 'turn-off' && (
        <Callout tone="warning" stacked action={<Button size="sm" onClick={() => void setVision(false)}>Turn off image reading</Button>}>
          This model could not read the test image. With image reading on, framing checks fail and fall back to face tracking.
        </Callout>
      )}
      {adviceError && <p role="alert" className="text-xs text-danger">{adviceError}</p>}
    </div>
  )
}

/** The key, model and endpoint fields of the chosen AI provider. */
export function ProviderFields({ commit, keys, compact = false }: { commit: CommitSettings; keys: Keys; compact?: boolean }): React.JSX.Element {
  const aiProvider = useSettingsStore((s) => s.aiProvider)
  const openrouterConfigured = useSettingsStore((s) => s.openrouterConfigured)
  const geminiConfigured = useSettingsStore((s) => s.geminiConfigured)
  const geminiModel = useSettingsStore((s) => s.geminiModel)
  const modelPresets = useSettingsStore((s) => s.modelPresets)
  const activeModelPreset = useSettingsStore((s) => s.activeModelPreset)
  const sourceContextWebResearch = useSettingsStore((s) => s.sourceContextWebResearch)
  const [models, setModels] = useState<string[]>([])
  useEffect(() => { setModels([]) }, [aiProvider])

  if (aiProvider === 'openrouter') {
    return (
      <div className="space-y-3">
        <div className="glass-tile rounded-2xl px-3 py-2.5">
          <ApiKeyInput
            label="OpenRouter"
            value={keys.drafts.openrouterApiKey}
            configured={openrouterConfigured}
            onChange={(v) => keys.setDraft('openrouterApiKey', v)}
            onRemove={() => void keys.remove('openrouterApiKey')}
            onBlur={() => void keys.persist()}
            placeholder="sk-or-…"
            description="Transcribes with MAI Transcribe 2 and picks the moments worth clipping."
            getKeyUrl={PROVIDER_LINKS.openrouter}
          />
        </div>
        <TestConnection provider="openrouter" keys={keys} />
        {!compact && <label className="flex items-start gap-3 px-3 py-2 text-sm text-ink-muted">
          <input type="checkbox" className="mt-1" checked={sourceContextWebResearch === 'on'}
            onChange={(e) => void commit({ sourceContextWebResearch: e.target.checked ? 'on' : 'off' })} />
          <span>Research the source before clipping <Badge tone="warning" className="ml-1 align-middle">Beta</Badge>
            <span className="mt-1 block text-xs text-ink-subtle">Off by default. When on, the video’s title, description and channel go to OpenRouter web search (up to two searches) and Gemini builds a channel and video overview before transcription. Uses extra OpenRouter credit and adds time. Only YouTube and Twitch sources are researched; local files never are. View the brief and sources in the transcript inspector.</span>
          </span>
        </label>}
      </div>
    )
  }

  const onlyOpenRouter = (
    <p className="px-1 text-xs text-ink-subtle">
      Review &amp; edit and the AI review of automatic clips use this model in place of Jev. Web research of the source needs OpenRouter, so it is off with {AI_PROVIDER_LABELS[aiProvider]}.
    </p>
  )

  if (aiProvider === 'claude-plan') {
    return <ClaudePlanFields commit={commit} test={<TestConnection provider="claude-plan" />}>{onlyOpenRouter}</ClaudePlanFields>
  }

  if (aiProvider === 'gemini') {
    return (
      <div className="space-y-3">
        <div className="glass-tile rounded-2xl px-3 py-2.5">
          <ApiKeyInput
            label="Google Gemini API key"
            value={keys.drafts.geminiApiKey}
            configured={geminiConfigured}
            onChange={(v) => keys.setDraft('geminiApiKey', v)}
            onRemove={() => void keys.remove('geminiApiKey')}
            onBlur={() => void keys.persist()}
            placeholder="AIzaSy…"
            description="Picks the moments worth clipping and looks at video frames for smart framing."
            getKeyUrl={PROVIDER_LINKS.gemini}
          />
        </div>
        <TextSetting
          label="Gemini model"
          hint={`Default: ${DEFAULT_GEMINI_MODEL}. Use Test connection to list the models your key can use.`}
          value={geminiModel}
          placeholder={DEFAULT_GEMINI_MODEL}
          validate={validateModel}
          suggestions={[...new Set([...models, ...GEMINI_MODEL_SUGGESTIONS])]}
          commit={(value) => commit({ geminiModel: value })}
        />
        <TestConnection provider="gemini" keys={keys} onModels={setModels} />
        {onlyOpenRouter}
      </div>
    )
  }

  // OpenAI-compatible: the chosen saved model plans clips. Cards above add, edit and choose them.
  const active = modelPresets.find((preset) => preset.id === activeModelPreset)
  return (
    <div className="space-y-3">
      {active ? (
        <>
          <p className="px-1 text-xs text-ink-muted">
            Clips are planned by <span className="font-medium text-ink">{modelPresetLabel(active)}</span>
            {' '}(<span className="font-mono">{active.model}</span> at {endpointHost(active.url)}). Use its card’s edit button to change it.
          </p>
          {looksLikeOllama(active.url) && <OllamaContextWarning />}
          <TestConnection provider="openai-compatible" preset={active} />
        </>
      ) : (
        <Callout tone="warning">
          {modelPresets.length > 0
            ? 'Choose one of your saved models above to make clips with it.'
            : 'Add a model above: its endpoint URL, model name and, if the server needs one, an API key.'}
        </Callout>
      )}
      {onlyOpenRouter}
    </div>
  )
}

function OllamaContextWarning(): React.JSX.Element {
  return (
    <Callout tone="warning">
      Ollama uses a small context by default (4,096 tokens on most computers) and cuts a long transcript down to fit
      without an error, so clips are picked from part of the video. Set <code>OLLAMA_CONTEXT_LENGTH=32768</code> or
      more for the Ollama server, then restart it.
    </Callout>
  )
}

/**
 * Adds a saved model (no `preset`) or edits one. The key typed here stays in this form until Save
 * sends it to main with the rest; it is never written to the card in use while typing. Editing a
 * card offers Test connection for it, chosen or not, which also lists the server's models.
 */
export function ModelPresetForm({ preset, onClose, onSaved, onMovedToLocal, justSaved = false }: {
  preset?: PublicModelPreset
  onClose: () => void
  /** A new card was saved: the form stays open on it, so it can be tested right away. */
  onSaved?: (id: string) => void
  /** Saving chose the card and moved speech to text to Local Whisper (no OpenRouter key). */
  onMovedToLocal?: () => void
  justSaved?: boolean
}): React.JSX.Element {
  const saveModelPreset = useSettingsStore((s) => s.saveModelPreset)
  const [name, setName] = useState(preset?.name ?? '')
  const [url, setUrl] = useState(preset?.url ?? '')
  const [model, setModel] = useState(preset?.model ?? '')
  const [apiKey, setApiKey] = useState('')
  const [removeKey, setRemoveKey] = useState(false)
  const [vision, setVision] = useState<'on' | 'off'>(preset?.vision ?? 'off')
  const [timeout, setWait] = useState<OpenaiCompatibleTimeout>(preset?.timeout ?? DEFAULT_OPENAI_COMPATIBLE_TIMEOUT)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [models, setModels] = useState<string[]>([])
  const nameId = useId()
  const urlId = useId()
  const modelId = useId()
  const listId = useId()
  const timeoutId = useId()
  // "Turn on image reading" after a test saves the card; the open form follows it.
  const savedVision = preset?.vision
  useEffect(() => { if (savedVision) setVision(savedVision) }, [savedVision])
  // A saved key belongs to its server; main drops it when the card moves to another one.
  const origin = (value: string): string | null => { try { return new URL(value.trim()).origin } catch { return null } }
  const serverChanged = Boolean(preset?.keySaved) && origin(url) !== null && origin(url) !== origin(preset?.url ?? '')

  const save = async (): Promise<void> => {
    const problem = (!url.trim() ? 'Enter the endpoint URL.' : validateUrl(url.trim()))
      ?? (!model.trim() ? 'Enter the model name.' : validateModel(model.trim()))
    if (problem) { setError(problem); return }
    setSaving(true)
    try {
      const before = useSettingsStore.getState()
      await saveModelPreset({
        ...(preset ? { id: preset.id } : {}), name, url: url.trim(), model: model.trim(),
        apiKey: apiKey.trim() ? apiKey.trim() : removeKey ? '' : null, vision, timeout
      })
      const after = useSettingsStore.getState()
      if (before.transcriptionSource === 'cloud' && after.transcriptionSource === 'local') onMovedToLocal?.()
      const added = preset ? undefined : after.modelPresets.find((p) => !before.modelPresets.some((q) => q.id === p.id))
      if (added && onSaved) onSaved(added.id)
      else onClose()
    } catch (err) {
      setError(errorMessage(err, 'Could not save this model.'))
    } finally {
      setSaving(false)
    }
  }

  const title = preset ? `Edit “${modelPresetLabel(preset)}”` : 'Add a model'
  return (
    <div role="group" aria-label={title} className="glass-tile mt-3 space-y-3 rounded-2xl p-3">
      <p className="text-sm font-semibold text-ink">{title}</p>
      <Field label="Name (optional)" htmlFor={nameId} hint="What the card says. Leave empty to show the model and server.">
        <TextInput id={nameId} value={name} maxLength={MAX_MODEL_PRESET_NAME} placeholder="Groq GPT-OSS"
          onChange={(event) => { setName(event.target.value); setError(null) }} />
      </Field>
      <Field label="Endpoint URL" htmlFor={urlId} hint="The base URL of the server, usually ending in /v1. Plain http is fine on this computer or your own network.">
        <TextInput id={urlId} mono value={url} placeholder="https://api.groq.com/openai/v1"
          onChange={(event) => { setUrl(event.target.value); setError(null) }} />
      </Field>
      {looksLikeOllama(url) && <OllamaContextWarning />}
      <Field label="Model name" htmlFor={modelId} hint="The model the server should use, such as gpt-4o-mini or llama3.1.">
        <TextInput id={modelId} mono value={model} placeholder="openai/gpt-oss-120b" list={models.length > 0 ? listId : undefined}
          onChange={(event) => { setModel(event.target.value); setError(null) }} />
        {models.length > 0 && <datalist id={listId}>{models.map((option) => <option key={option} value={option} />)}</datalist>}
      </Field>
      <div className="glass-tile rounded-2xl px-3 py-2.5">
        <ApiKeyInput
          label="API key (optional)"
          value={apiKey}
          configured={Boolean(preset?.keySaved) && !removeKey && !serverChanged}
          onChange={(value) => { setApiKey(value); setRemoveKey(false); setError(null) }}
          onRemove={() => { setApiKey(''); setRemoveKey(true) }}
          placeholder="sk-…"
          description={serverChanged
            ? 'This card now points at another server, so its saved key is removed when you save. Enter the key for the new server.'
            : removeKey ? 'The saved key is removed when you save.' : 'Leave empty for a local server that needs none.'}
        />
      </div>
      <SettingRow
        title="This model can read images"
        description="Lets smart framing look at video frames. Leave off for text-only models, or framing falls back to its built-in face tracking."
        control={<Switch label="This model can read images" checked={vision === 'on'} onChange={(on) => setVision(on ? 'on' : 'off')} />}
      />
      <SettingRow
        title={<label htmlFor={timeoutId}>Wait for a reply</label>}
        description="How long to wait for one answer before giving up. Local models on a CPU can need many minutes for a long video."
        control={
          <Select id={timeoutId} size="sm" className="w-32" value={timeout}
            options={OPENAI_COMPATIBLE_TIMEOUTS.map((minutes) => ({ value: minutes, label: `${minutes} minutes` }))}
            onChange={(value) => setWait(value as OpenaiCompatibleTimeout)} />
        }
      />
      {error && <p role="alert" className="text-xs text-danger">{error}</p>}
      {justSaved && <p role="status" className="text-xs text-success">Saved as a card. Test it below, or close this.</p>}
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" size="sm" loading={saving} onClick={() => void save()}>{preset ? 'Save changes' : 'Save model'}</Button>
        <Button variant="ghost" size="sm" onClick={onClose}>{preset ? 'Close' : 'Cancel'}</Button>
      </div>
      {preset && (
        <div className="space-y-2 border-t border-white/[0.06] pt-3">
          <p className="text-2xs text-ink-subtle">Tests this model as saved, without choosing it. Save your changes first to test them.</p>
          <TestConnection provider="openai-compatible" preset={preset} onModels={setModels} />
        </div>
      )}
    </div>
  )
}

/** Where speech becomes text: OpenRouter's cloud, or Whisper on this computer. */
export function TranscriptionSettings({ commit, keys, compact = false }: { commit: CommitSettings; keys: Keys; compact?: boolean }): React.JSX.Element {
  const aiProvider = useSettingsStore((s) => s.aiProvider)
  const openrouterConfigured = useSettingsStore((s) => s.openrouterConfigured)
  const transcriptionSource = useSettingsStore((s) => s.transcriptionSource)
  const localWhisperMode = useSettingsStore((s) => s.localWhisperMode)
  const localWhisperUrl = useSettingsStore((s) => s.localWhisperUrl)
  const localWhisperModel = useSettingsStore((s) => s.localWhisperModel)
  const localWhisperLanguage = useSettingsStore((s) => s.localWhisperLanguage)
  const languageId = useId()
  const [error, setError] = useState<string | null>(null)
  const [checking, setChecking] = useState(false)
  const [status, setStatus] = useState<LocalWhisperStatus | null>(null)
  useEffect(() => { setStatus(null) }, [localWhisperMode, localWhisperUrl, localWhisperModel])

  const apply = async (patch: SettingsPatch): Promise<void> => setError(await commit(patch))

  // The model that fits the new mode is chosen when the change is applied, from the model saved by then,
  // so two quick changes cannot leave a model that only suits the other mode.
  const switchMode = (mode: 'builtin' | 'server'): void => {
    void apply((current) => localWhisperModeChange(mode, current.localWhisperModel))
  }

  const check = async (): Promise<void> => {
    setChecking(true)
    try {
      await whenSettingsSaved()
      setStatus(await getApi().settings.checkLocalWhisper())
    } catch (err) {
      setStatus({ available: false, backend: null, detail: errorMessage(err, 'Could not run the check.'), device: null, modelCached: null })
    } finally {
      setChecking(false)
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-ink">Speech to text</h3>
          <p className="mt-0.5 text-xs text-ink-muted">Where your video’s audio is turned into words and caption timing.</p>
        </div>
        <Segmented
          label="Transcription source"
          size="sm"
          value={transcriptionSource}
          options={[{ value: 'cloud', label: 'Cloud (OpenRouter)' }, { value: 'local', label: 'Local Whisper' }]}
          onChange={(value) => { transcriptionMovedByProvider = false; void apply({ transcriptionSource: value }) }}
        />
      </div>

      {transcriptionSource === 'cloud' ? (
        <>
          <p className="text-xs text-ink-subtle">Audio is sent to OpenRouter and transcribed with MAI Transcribe 2, with word-level timestamps.</p>
          {aiProvider !== 'openrouter' && (
            <>
              {!openrouterConfigured && (
                <Callout
                  tone="warning"
                  stacked
                  action={<Button size="sm" onClick={() => { transcriptionMovedByProvider = false; void apply({ transcriptionSource: 'local' }) }}>Use Local Whisper instead</Button>}
                >
                  Cloud transcription uses OpenRouter, so it needs an OpenRouter key even though {AI_PROVIDER_LABELS[aiProvider]} picks your clips.
                  Add the key below, or switch to Local Whisper and keep everything on this computer.
                </Callout>
              )}
              <div className="glass-tile rounded-2xl px-3 py-2.5">
                <ApiKeyInput
                  label="OpenRouter key for transcription"
                  value={keys.drafts.openrouterApiKey}
                  configured={openrouterConfigured}
                  onChange={(v) => keys.setDraft('openrouterApiKey', v)}
                  onRemove={() => void keys.remove('openrouterApiKey')}
                  onBlur={() => void keys.persist()}
                  placeholder="sk-or-…"
                  description="Used only to transcribe audio. Clip selection stays on your chosen provider."
                  getKeyUrl={PROVIDER_LINKS.openrouter}
                />
              </div>
            </>
          )}
        </>
      ) : (
        <div className="space-y-3">
          {localWhisperMode === 'builtin' || (localWhisperUrl && isLocalEndpoint(localWhisperUrl))
            ? <p className="text-xs text-success">Audio stays on this computer. No account and no per-minute cost.</p>
            : <p className="text-xs text-ink-subtle">Enter the address of a Whisper server on this computer or your private network.</p>}
          <Segmented
            label="Local Whisper type"
            size="sm"
            value={localWhisperMode}
            options={[{ value: 'builtin', label: 'Built-in' }, { value: 'server', label: 'Server' }]}
            onChange={switchMode}
          />
          {localWhisperMode === 'builtin' ? (
            <>
              {!compact && <WhisperModelsPanel commit={commit} />}
              <p className="text-xs text-ink-subtle">Runs Whisper (faster-whisper) inside BridgeClip’s engine. {compact ? 'Pick and install the model in Settings.' : ''}</p>
            </>
          ) : (
            <>
              <TextSetting
                label="Whisper server URL"
                hint="An OpenAI-compatible server on this computer or your own network, such as faster-whisper-server or LocalAI. Usually ends in /v1."
                value={localWhisperUrl}
                placeholder="http://127.0.0.1:8000/v1"
                validate={validateLocalUrl}
                commit={(value) => commit({ localWhisperUrl: value })}
              />
              {!compact && (
                <TextSetting
                  label="Model name sent to the server"
                  hint="Many servers accept whisper-1; others want a name such as large-v3."
                  value={localWhisperModel}
                  placeholder="whisper-1"
                  validate={(value) => (value && isSafeModelName(value) ? null : 'Use only letters, numbers and . _ : / @ + - (no spaces).')}
                  commit={(value) => commit({ localWhisperModel: value })}
                />
              )}
            </>
          )}
          {!compact && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
              <label htmlFor={languageId} className="text-xs font-medium text-ink">Spoken language</label>
              <Select
                id={languageId}
                size="sm"
                className="w-48"
                value={localWhisperModel.endsWith('.en') && localWhisperMode === 'builtin' ? 'en' : localWhisperLanguage}
                disabled={localWhisperModel.endsWith('.en') && localWhisperMode === 'builtin'}
                options={LOCAL_WHISPER_LANGUAGES.map((language) => ({ value: language.code, label: language.label }))}
                onChange={(value) => void apply({ localWhisperLanguage: value as LocalWhisperLanguage })}
              />
              <span className="text-2xs text-ink-subtle">
                {localWhisperModel.endsWith('.en') && localWhisperMode === 'builtin'
                  ? 'English-only model.'
                  : 'Pick it when you know it: detection can guess wrong when a video opens with music.'}
              </span>
            </div>
          )}
          <div className="flex flex-wrap items-center gap-3">
            <Button size="sm" onClick={() => void check()} disabled={checking} icon={checking ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <PlugZap className="h-3.5 w-3.5" />}>
              {localWhisperMode === 'server' ? 'Check server' : 'Check Whisper'}
            </Button>
            {status && (
              <Outcome ok={status.available}>
                {status.available
                  ? `${status.detail}${status.device ? ` · ${status.device === 'cuda' ? 'NVIDIA GPU found' : 'CPU only'}` : ''}${status.modelCached === false ? ` · install the ${localWhisperModel} model ${compact ? 'in Settings' : 'below'} before starting a job` : ''}`
                  : status.detail}
              </Outcome>
            )}
          </div>
          {status?.available && status.device === 'cpu' && localWhisperMode === 'builtin' && (
            <p className="text-2xs text-ink-subtle">No GPU is in use, so Whisper runs on the CPU. Small is the practical size for long videos; larger models are slower.</p>
          )}
        </div>
      )}
      {error && <p role="alert" className="text-xs text-danger">{error}</p>}
    </div>
  )
}
