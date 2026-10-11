import { useEffect, useId, useState } from 'react'
import { Check, CircleAlert, Loader2, Minus, PlugZap, X } from 'lucide-react'
import { useSettingsStore, whenSettingsSaved, type SettingsPatch } from '../store/use-settings-store'
import type { useApiKeyDrafts } from '../hooks/use-api-key-drafts'
import { getApi } from '../lib/ipc'
import { cn, errorMessage } from '../lib/utils'
import { PROVIDER_LINKS } from '../config/brand'
import {
  AI_PROVIDER_LABELS, DEFAULT_GEMINI_MODEL, GEMINI_MODEL_SUGGESTIONS, OPENAI_COMPATIBLE_TIMEOUTS, PROVIDER_CHECK_LABELS, aiProviderChange,
  testSubject, visionAdvice,
  checkEndpointUrl, checkLocalWhisperUrl, isLocalEndpoint, isSafeModelName, localWhisperModeChange, looksLikeOllama,
  type AiProvider, type OpenaiCompatibleTimeout, type ProviderCheck, type TranscriptionSource
} from '../../shared/providers'
import type { LocalWhisperStatus, ProviderTestResult } from '../../preload/index'
import { ApiKeyInput } from './ApiKeyInput'
import { WhisperModelsPanel } from './WhisperModelsPanel'
import { LOCAL_WHISPER_LANGUAGES, type LocalWhisperLanguage } from '../../shared/whisper-models'
import { Select } from './ui/Select'
import { Badge } from './ui/Badge'
import { Button } from './ui/Button'
import { Callout } from './ui/Callout'
import { Field, TextInput } from './ui/Field'
import { Segmented, onRadioKeyDown } from './ui/Segmented'
import { SettingRow } from './ui/SettingRow'
import { Switch } from './ui/Switch'

/**
 * Saves part of the settings. Resolves to null on success or to a message the person can
 * read, so the field that asked can show it right where they typed.
 */
export type CommitSettings = (patch: SettingsPatch) => Promise<string | null>

type Keys = ReturnType<typeof useApiKeyDrafts>

const PROVIDER_CHOICES: { id: AiProvider; title: string; description: string }[] = [
  { id: 'openrouter', title: AI_PROVIDER_LABELS.openrouter, description: 'One key for everything. Includes Jev review and automatic failover.' },
  { id: 'gemini', title: AI_PROVIDER_LABELS.gemini, description: 'Your own Google AI Studio key. Picks clips and frames the shots.' },
  { id: 'openai-compatible', title: AI_PROVIDER_LABELS['openai-compatible'], description: 'OpenAI, Ollama, LM Studio or any server with an OpenAI-style API.' }
]

function isConfigured(provider: AiProvider, s: { openrouterConfigured: boolean; geminiConfigured: boolean; openaiCompatibleUrl: string; openaiCompatibleModel: string }): boolean {
  if (provider === 'openrouter') return s.openrouterConfigured
  if (provider === 'gemini') return s.geminiConfigured
  return Boolean(s.openaiCompatibleUrl.trim() && s.openaiCompatibleModel.trim())
}

/**
 * Whether transcription was moved to Local Whisper by choosing a provider, rather than by the
 * person. Kept for the app session, so reopening Settings and going back to OpenRouter still undoes it.
 */
let transcriptionMovedByProvider = false

/** Choose where clip selection runs. A radio group, so arrow keys move between the three. */
export function ProviderPicker({ commit, compact = false }: { commit: CommitSettings; compact?: boolean }): React.JSX.Element {
  const aiProvider = useSettingsStore((s) => s.aiProvider)
  const openrouterConfigured = useSettingsStore((s) => s.openrouterConfigured)
  const geminiConfigured = useSettingsStore((s) => s.geminiConfigured)
  const openaiCompatibleUrl = useSettingsStore((s) => s.openaiCompatibleUrl)
  const openaiCompatibleModel = useSettingsStore((s) => s.openaiCompatibleModel)
  const [error, setError] = useState<string | null>(null)
  const transcriptionSource = useSettingsStore((s) => s.transcriptionSource)
  const [movedToLocal, setMovedToLocal] = useState(false)
  const choose = async (id: AiProvider): Promise<void> => {
    let moved: TranscriptionSource | undefined
    const failed = await commit((current) => {
      const patch = aiProviderChange(id, current, transcriptionMovedByProvider)
      moved = patch.transcriptionSource
      return patch
    })
    setError(failed)
    if (failed) return
    if (moved === 'local') transcriptionMovedByProvider = true
    if (moved === 'cloud') transcriptionMovedByProvider = false
    setMovedToLocal(transcriptionMovedByProvider)
  }

  return (
    <div>
      <div
        role="radiogroup"
        aria-label="AI provider"
        className={cn(compact ? 'flex items-center gap-1 rounded-xl bg-black/20 p-1' : 'grid grid-cols-1 gap-2 sm:grid-cols-3')}
      >
        {PROVIDER_CHOICES.map((choice) => {
          const selected = aiProvider === choice.id
          const ready = isConfigured(choice.id, { openrouterConfigured, geminiConfigured, openaiCompatibleUrl, openaiCompatibleModel })
          return (
            <button
              key={choice.id}
              type="button"
              role="radio"
              aria-checked={selected}
              tabIndex={selected ? 0 : -1}
              onKeyDown={onRadioKeyDown}
              onClick={() => void choose(choice.id)}
              className={compact
                ? cn('flex-1 rounded-lg py-1 text-xs font-medium transition-colors', selected ? 'bg-accent text-accent-ink' : 'text-ink-muted hover:text-ink')
                : cn('glass-tile glass-tile-hover flex flex-col items-start rounded-xl p-3 text-left transition-all', selected && 'glass-selected')}
            >
              {compact ? choice.title : (
                <>
                  <span className="flex w-full items-center justify-between gap-2">
                    <span className="text-sm font-semibold text-ink">{choice.title}</span>
                    {ready && <Check aria-label="Set up" className="h-3.5 w-3.5 text-success" />}
                  </span>
                  <span className="mt-1 text-2xs leading-relaxed text-ink-muted">{choice.description}</span>
                </>
              )}
            </button>
          )
        })}
      </div>
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
function TestConnection({ provider, keys, onModels, commit }: {
  provider: AiProvider
  keys: Keys
  onModels?: (models: string[]) => void
  /** Lets the result switch "This model can read images" for a custom endpoint. */
  commit?: CommitSettings
}): React.JSX.Element {
  const [testing, setTesting] = useState(false)
  const [result, setResult] = useState<ProviderTestResult | null>(null)
  const visionOn = useSettingsStore((s) => s.openaiCompatibleVision === 'on')
  // A result is about the URL and model it tested. Once either changes, it no longer applies.
  const subject = useSettingsStore((s) => testSubject(provider, s))
  const current = result && (result.subject === subject || !result.subject) ? result : null

  const run = async (): Promise<void> => {
    setTesting(true)
    try {
      await keys.persist()
      await whenSettingsSaved()
      const outcome = await getApi().settings.testAiProvider(provider)
      setResult(outcome)
      onModels?.(outcome.models)
    } catch (err) {
      setResult({ ok: false, message: errorMessage(err, 'Could not run the test.'), models: [], checks: [], readsImages: null, subject: '' })
    } finally {
      setTesting(false)
    }
  }

  const advice = !testing && commit && current ? visionAdvice(provider, current.readsImages, visionOn) : null
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-3">
        <Button size="sm" onClick={() => void run()} loading={testing} icon={<PlugZap className="h-3.5 w-3.5" />}>
          Test connection
        </Button>
        {testing && provider !== 'openrouter' && (
          <span role="status" className="text-2xs text-ink-subtle">Sending short test requests. A local model can take a minute to load.</span>
        )}
        {!testing && current && <Outcome ok={current.ok}>{current.message}</Outcome>}
      </div>
      {!testing && current && current.checks.length > 1 && <CheckList checks={current.checks} />}
      {advice === 'turn-on' && commit && (
        <Callout tone="success" stacked action={<Button size="sm" onClick={() => void commit({ openaiCompatibleVision: 'on' })}>Turn on image reading</Button>}>
          This model read the test image. Turn on “This model can read images” so the AI can check facecam framing.
        </Callout>
      )}
      {advice === 'turn-off' && commit && (
        <Callout tone="warning" stacked action={<Button size="sm" onClick={() => void commit({ openaiCompatibleVision: 'off' })}>Turn off image reading</Button>}>
          This model could not read the test image. With image reading on, framing checks fail and fall back to face tracking.
        </Callout>
      )}
    </div>
  )
}

/** The key, model and endpoint fields of the chosen AI provider. */
export function ProviderFields({ commit, keys, compact = false }: { commit: CommitSettings; keys: Keys; compact?: boolean }): React.JSX.Element {
  const aiProvider = useSettingsStore((s) => s.aiProvider)
  const openrouterConfigured = useSettingsStore((s) => s.openrouterConfigured)
  const geminiConfigured = useSettingsStore((s) => s.geminiConfigured)
  const openaiCompatibleConfigured = useSettingsStore((s) => s.openaiCompatibleConfigured)
  const geminiModel = useSettingsStore((s) => s.geminiModel)
  const openaiCompatibleUrl = useSettingsStore((s) => s.openaiCompatibleUrl)
  const openaiCompatibleModel = useSettingsStore((s) => s.openaiCompatibleModel)
  const openaiCompatibleVision = useSettingsStore((s) => s.openaiCompatibleVision)
  const openaiCompatibleTimeout = useSettingsStore((s) => s.openaiCompatibleTimeout)
  const timeoutId = useId()
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

  return (
    <div className="space-y-3">
      <TextSetting
        label="Endpoint URL"
        hint="The base URL of the server, usually ending in /v1. Plain http is fine on this computer or your own network."
        value={openaiCompatibleUrl}
        placeholder="http://localhost:11434/v1"
        validate={validateUrl}
        commit={(value) => commit({ openaiCompatibleUrl: value })}
      />
      {looksLikeOllama(openaiCompatibleUrl) && (
        <Callout tone="warning">
          Ollama uses a small context by default (4,096 tokens on most computers) and cuts a long transcript down to fit
          without an error, so clips are picked from part of the video. Set <code>OLLAMA_CONTEXT_LENGTH=32768</code> or
          more for the Ollama server, then restart it.
        </Callout>
      )}
      <div className="glass-tile rounded-2xl px-3 py-2.5">
        <ApiKeyInput
          label="API key (optional)"
          value={keys.drafts.openaiCompatibleApiKey}
          configured={openaiCompatibleConfigured}
          onChange={(v) => keys.setDraft('openaiCompatibleApiKey', v)}
          onRemove={() => void keys.remove('openaiCompatibleApiKey')}
          onBlur={() => void keys.persist()}
          placeholder="sk-…"
          description="Leave empty for a local server that needs none."
        />
      </div>
      <TextSetting
        label="Model name"
        hint="The model the server should use, such as gpt-4o-mini or llama3.1. Use Test connection to list what the server offers."
        value={openaiCompatibleModel}
        placeholder="gpt-4o-mini"
        validate={validateModel}
        suggestions={models}
        commit={(value) => commit({ openaiCompatibleModel: value })}
      />
      <SettingRow
        title="This model can read images"
        description="Lets smart framing look at video frames. Leave off for text-only models, or framing falls back to its built-in face tracking."
        control={<Switch label="This model can read images" checked={openaiCompatibleVision === 'on'} onChange={(on) => void commit({ openaiCompatibleVision: on ? 'on' : 'off' })} />}
      />
      <SettingRow
        title={<label htmlFor={timeoutId}>Wait for a reply</label>}
        description="How long to wait for one answer before giving up. Local models on a CPU can need many minutes for a long video."
        control={
          <Select
            id={timeoutId}
            size="sm"
            className="w-32"
            value={openaiCompatibleTimeout}
            options={OPENAI_COMPATIBLE_TIMEOUTS.map((minutes) => ({ value: minutes, label: `${minutes} minutes` }))}
            onChange={(value) => void commit({ openaiCompatibleTimeout: value as OpenaiCompatibleTimeout })}
          />
        }
      />
      <TestConnection provider="openai-compatible" keys={keys} onModels={setModels} commit={commit} />
      {onlyOpenRouter}
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
