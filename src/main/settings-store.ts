import { JEV_DEFAULTS, JEV_FEATURE_DEFAULTS, type JevThresholdSettings } from '../shared/jev-settings'
import {
  AI_PROVIDERS, DEFAULT_GEMINI_MODEL, DEFAULT_LOCAL_WHISPER_MODEL, DEFAULT_OPENAI_COMPATIBLE_TIMEOUT, LOCAL_WHISPER_MODES,
  OPENAI_COMPATIBLE_TIMEOUTS, TRANSCRIPTION_SOURCES,
  checkEndpointUrl, checkLocalWhisperUrl, isBuiltinWhisperModel, isSafeModelName,
  type AiProvider, type LocalWhisperMode, type OpenaiCompatibleTimeout, type TranscriptionSource
} from '../shared/providers'
import { LOCAL_WHISPER_DEVICES, LOCAL_WHISPER_LANGUAGE_CODES, type LocalWhisperDevice, type LocalWhisperLanguage } from '../shared/whisper-models'
import { DEFAULT_SCREEN_FRAMING, DEFAULT_WEBCAM_LOCATION, DEFAULT_WEBCAM_POSITION, SCREEN_FRAMINGS, WEBCAM_LOCATIONS, WEBCAM_POSITIONS, type ScreenFraming, type WebcamLocation, type WebcamPosition } from '../shared/framing'
import { DEFAULT_VIDEO_ENCODER, VIDEO_ENCODERS, type VideoEncoder } from '../shared/video-encoder'
import { app, safeStorage } from 'electron'
import { closeSync, existsSync, fchmodSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync, unlinkSync, writeFileSync } from 'fs'
import { isAbsolute, join } from 'path'
import { randomUUID } from 'crypto'

/**
 * BridgeClip is bring-your-own-key: every provider call is made from this
 * machine with the user's own keys. Keys are encrypted with the OS keychain
 * (safeStorage) when it is available.
 */
export type { AiProvider, LocalWhisperDevice, LocalWhisperLanguage, LocalWhisperMode, ScreenFraming, TranscriptionSource, VideoEncoder, WebcamLocation, WebcamPosition }

export interface AppSettings extends JevThresholdSettings {
  openrouterApiKey: string
  /** Optional: connects social accounts for posting. Used only by the main process, never sent to the engine. */
  zernioApiKey: string
  /** AI provider selection: OpenRouter, Gemini, or custom OpenAI-compatible endpoint. */
  aiProvider: AiProvider
  geminiApiKey: string
  geminiModel: string
  openaiCompatibleUrl: string
  openaiCompatibleApiKey: string
  openaiCompatibleModel: string
  /** 'on' sends video frames to the OpenAI-compatible model for smart framing. Off unless the model can read images. */
  openaiCompatibleVision: string
  /** Minutes to wait for the OpenAI-compatible endpoint to answer one request. */
  openaiCompatibleTimeout: OpenaiCompatibleTimeout
  /** Speech-to-text source: Cloud (OpenRouter) or Local Whisper. */
  transcriptionSource: TranscriptionSource
  /** Built-in runs Whisper inside the engine's Python; server calls an OpenAI-compatible /audio/transcriptions URL. */
  localWhisperMode: LocalWhisperMode
  /** Only used in server mode. Empty means not configured. */
  localWhisperUrl: string
  localWhisperModel: string
  /** Built-in Whisper only. Auto tries the GPU and falls back to the CPU; CPU never touches CUDA; GPU fails loudly. */
  localWhisperDevice: LocalWhisperDevice
  /** Spoken language for local Whisper: a Whisper language code, or "auto" to detect it. */
  localWhisperLanguage: LocalWhisperLanguage
  /** Smart framing: the webcam panel above or below a shared screen. */
  webcamPosition: WebcamPosition
  /** Smart framing: where the webcam sits in the source videos. Auto finds it; a place means only a face there is the webcam. */
  webcamLocation: WebcamLocation
  /** Smart framing: a screen with no webcam fills the frame (cropped) or shows whole over a blur. */
  screenFraming: ScreenFraming
  /** Exports on Windows and Linux: x264 on the CPU, or NVENC on an NVIDIA GPU (falls back to the CPU). */
  videoEncoder: VideoEncoder
  /** 'on': automatic picks offer more candidates and a second AI request scores each one blind before ranking. */
  clipJudge: string
  /** Opt-in beta: Jev review of automatic clips via the existing OpenRouter key. Review & edit always uses Jev. */
  jevEnabled: string
  /** Additional OpenRouter frame observations, explicitly opt-in. */
  jevVisualContext: string
  /** Opt-in beta: web research of public sources before transcription. */
  sourceContextWebResearch: string
  outputDirectory: string
  pythonPath: string
  /** Names and jargon the speech-to-text should spell correctly, one per line. */
  customVocabulary: string
}

export type ApiKeyName = 'openrouterApiKey' | 'zernioApiKey' | 'geminiApiKey' | 'openaiCompatibleApiKey'
export type PublicSettings = Pick<AppSettings, 'outputDirectory' | 'pythonPath' | 'customVocabulary' | 'jevEnabled' | 'jevVisualContext' | 'sourceContextWebResearch' | 'aiProvider' | 'geminiModel' | 'openaiCompatibleUrl' | 'openaiCompatibleModel' | 'openaiCompatibleVision' | 'openaiCompatibleTimeout' | 'transcriptionSource' | 'localWhisperMode' | 'localWhisperUrl' | 'localWhisperModel' | 'localWhisperDevice' | 'localWhisperLanguage' | 'webcamPosition' | 'webcamLocation' | 'screenFraming' | 'videoEncoder' | 'clipJudge' | keyof JevThresholdSettings> & {
  openrouterConfigured: boolean
  zernioConfigured: boolean
  geminiConfigured: boolean
  openaiCompatibleConfigured: boolean
}

const SECRET_KEYS = ['openrouterApiKey', 'zernioApiKey', 'geminiApiKey', 'openaiCompatibleApiKey'] as const
type SecretKey = (typeof SECRET_KEYS)[number]

const DEFAULT_SETTINGS: AppSettings = {
  ...JEV_DEFAULTS,
  ...JEV_FEATURE_DEFAULTS,
  openrouterApiKey: '',
  zernioApiKey: '',
  aiProvider: 'openrouter',
  geminiApiKey: '',
  geminiModel: DEFAULT_GEMINI_MODEL,
  openaiCompatibleUrl: '',
  openaiCompatibleApiKey: '',
  openaiCompatibleModel: '',
  openaiCompatibleVision: 'off',
  openaiCompatibleTimeout: DEFAULT_OPENAI_COMPATIBLE_TIMEOUT,
  transcriptionSource: 'cloud',
  localWhisperMode: 'builtin',
  localWhisperUrl: '',
  localWhisperModel: DEFAULT_LOCAL_WHISPER_MODEL,
  localWhisperDevice: 'auto',
  localWhisperLanguage: 'auto',
  webcamPosition: DEFAULT_WEBCAM_POSITION,
  webcamLocation: DEFAULT_WEBCAM_LOCATION,
  screenFraming: DEFAULT_SCREEN_FRAMING,
  videoEncoder: DEFAULT_VIDEO_ENCODER,
  clipJudge: 'on',
  outputDirectory: join(app.getPath('home'), 'BridgeClip'),
  pythonPath: process.platform === 'win32' ? 'python' : 'python3',
  customVocabulary: ''
}

const SETTINGS_VERSION = 12
/**
 * Versions 9-11 (pre-release builds of this feature) saved Jev review and web
 * research as on by default, and older versions drop the fields entirely, so a
 * saved "on" before this version is not an opt-in. Those files load both as off.
 */
const OPT_IN_BETA_VERSION = 12
const SWITCHES = ['off', 'on']

type PersistedSecret = { scheme: 'safeStorage' | 'base64'; value: string } | ''

interface PersistedSettings extends JevThresholdSettings {
  version: number
  openrouterApiKey: PersistedSecret
  zernioApiKey: PersistedSecret
  aiProvider?: string
  geminiApiKey?: PersistedSecret
  geminiModel?: string
  openaiCompatibleUrl?: string
  openaiCompatibleApiKey?: PersistedSecret
  openaiCompatibleModel?: string
  openaiCompatibleVision?: string
  openaiCompatibleTimeout?: string
  transcriptionSource?: string
  localWhisperMode?: string
  localWhisperUrl?: string
  localWhisperModel?: string
  localWhisperDevice?: string
  localWhisperLanguage?: string
  webcamPosition?: string
  webcamLocation?: string
  screenFraming?: string
  videoEncoder?: string
  clipJudge?: string
  jevEnabled: string
  jevVisualContext: string
  sourceContextWebResearch: string
  outputDirectory: string
  pythonPath: string
  customVocabulary?: string
}

function ensureDir(dir: string): string {
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true, mode: 0o700 })
  }
  return dir
}

function getSettingsPath(): string {
  return join(ensureDir(app.getPath('userData')), 'settings.json')
}

/** A provider choice or switch that must be one of `allowed`. A missing value takes the default. */
function oneOf<T extends string>(value: string | undefined, allowed: readonly T[], fallback: T, label: string): T {
  if (value === undefined) return fallback
  if (!allowed.includes(value as T)) throw new Error(`Invalid ${label}`)
  return value as T
}

/** An endpoint URL from the renderer. Empty stays empty, meaning not configured. */
function endpointUrl(value: string | undefined, label: string): string {
  const checked = checkEndpointUrl(value ?? '')
  if (!checked.ok) throw new Error(`${label}: ${checked.error}`)
  return checked.url
}

/** The URL of a Whisper server, which must be on this computer or a private network. */
function whisperServerUrl(value: string | undefined): string {
  const checked = checkLocalWhisperUrl(value ?? '')
  if (!checked.ok) throw new Error(`Whisper server URL: ${checked.error}`)
  return checked.url
}

/** A model name. `fallback` applies only to a missing or blank value. */
function modelName(value: string | undefined, fallback: string, label: string): string {
  const trimmed = (value ?? '').trim()
  if (!trimmed) return fallback
  if (!isSafeModelName(trimmed)) throw new Error(`${label} can only contain letters, numbers and . _ : / @ + -`)
  return trimmed
}

function whisperModel(value: string | undefined, mode: LocalWhisperMode): string {
  const trimmed = (value ?? '').trim()
  if (!trimmed) return DEFAULT_LOCAL_WHISPER_MODEL
  if (mode === 'builtin') {
    if (!isBuiltinWhisperModel(trimmed)) throw new Error('Choose one of the built-in Whisper models.')
    return trimmed
  }
  if (!isSafeModelName(trimmed)) throw new Error('Whisper model can only contain letters, numbers and . _ : / @ + -')
  return trimmed
}

function normalizeSettings(settings: Partial<AppSettings>): AppSettings {
  if (!settings || typeof settings !== 'object') throw new Error('Invalid settings')
  for (const key of Object.keys(DEFAULT_SETTINGS) as (keyof AppSettings)[]) {
    if (settings[key] !== undefined && (typeof settings[key] !== 'string' || settings[key]!.length > 8192 || settings[key]!.includes('\0'))) throw new Error(`Invalid ${key}`)
  }
  const localWhisperMode = oneOf(settings.localWhisperMode, LOCAL_WHISPER_MODES, DEFAULT_SETTINGS.localWhisperMode, 'Whisper mode')
  const normalized: AppSettings = {
    jevThreshold: settings.jevThreshold ?? JEV_DEFAULTS.jevThreshold,
    jevSelfContainedThreshold: settings.jevSelfContainedThreshold ?? JEV_DEFAULTS.jevSelfContainedThreshold,
    jevFaithfulToSourceThreshold: settings.jevFaithfulToSourceThreshold ?? JEV_DEFAULTS.jevFaithfulToSourceThreshold,
    jevTitleSupportedThreshold: settings.jevTitleSupportedThreshold ?? JEV_DEFAULTS.jevTitleSupportedThreshold,
    jevSponsorThreshold: settings.jevSponsorThreshold ?? JEV_DEFAULTS.jevSponsorThreshold,
    jevEvidenceThreshold: settings.jevEvidenceThreshold ?? JEV_DEFAULTS.jevEvidenceThreshold,
    jevCutThreshold: settings.jevCutThreshold ?? JEV_DEFAULTS.jevCutThreshold,

    openrouterApiKey: (settings.openrouterApiKey ?? DEFAULT_SETTINGS.openrouterApiKey).trim(),
    zernioApiKey: (settings.zernioApiKey ?? DEFAULT_SETTINGS.zernioApiKey).trim(),
    aiProvider: oneOf(settings.aiProvider, AI_PROVIDERS, DEFAULT_SETTINGS.aiProvider, 'AI provider'),
    geminiApiKey: (settings.geminiApiKey ?? DEFAULT_SETTINGS.geminiApiKey).trim(),
    geminiModel: modelName(settings.geminiModel, DEFAULT_GEMINI_MODEL, 'Gemini model'),
    openaiCompatibleUrl: endpointUrl(settings.openaiCompatibleUrl, 'Endpoint URL'),
    openaiCompatibleApiKey: (settings.openaiCompatibleApiKey ?? DEFAULT_SETTINGS.openaiCompatibleApiKey).trim(),
    openaiCompatibleModel: modelName(settings.openaiCompatibleModel, '', 'Model name'),
    openaiCompatibleVision: settings.openaiCompatibleVision ?? DEFAULT_SETTINGS.openaiCompatibleVision,
    openaiCompatibleTimeout: oneOf(settings.openaiCompatibleTimeout, OPENAI_COMPATIBLE_TIMEOUTS, DEFAULT_SETTINGS.openaiCompatibleTimeout, 'endpoint wait time'),
    transcriptionSource: oneOf(settings.transcriptionSource, TRANSCRIPTION_SOURCES, DEFAULT_SETTINGS.transcriptionSource, 'transcription source'),
    localWhisperMode,
    localWhisperUrl: whisperServerUrl(settings.localWhisperUrl),
    localWhisperModel: whisperModel(settings.localWhisperModel, localWhisperMode),
    localWhisperDevice: oneOf(settings.localWhisperDevice, LOCAL_WHISPER_DEVICES, DEFAULT_SETTINGS.localWhisperDevice, 'Whisper device'),
    localWhisperLanguage: oneOf(settings.localWhisperLanguage, LOCAL_WHISPER_LANGUAGE_CODES, DEFAULT_SETTINGS.localWhisperLanguage, 'Whisper language'),
    webcamPosition: oneOf(settings.webcamPosition, WEBCAM_POSITIONS, DEFAULT_SETTINGS.webcamPosition, 'webcam position'),
    webcamLocation: oneOf(settings.webcamLocation, WEBCAM_LOCATIONS, DEFAULT_SETTINGS.webcamLocation, 'webcam location'),
    screenFraming: oneOf(settings.screenFraming, SCREEN_FRAMINGS, DEFAULT_SETTINGS.screenFraming, 'screen framing'),
    videoEncoder: oneOf(settings.videoEncoder, VIDEO_ENCODERS, DEFAULT_SETTINGS.videoEncoder, 'video encoder'),
    clipJudge: settings.clipJudge ?? DEFAULT_SETTINGS.clipJudge,
    jevEnabled: settings.jevEnabled ?? DEFAULT_SETTINGS.jevEnabled,
    jevVisualContext: settings.jevVisualContext ?? DEFAULT_SETTINGS.jevVisualContext,
    sourceContextWebResearch: settings.sourceContextWebResearch ?? DEFAULT_SETTINGS.sourceContextWebResearch,
    outputDirectory: (settings.outputDirectory || DEFAULT_SETTINGS.outputDirectory).trim(),
    pythonPath: (settings.pythonPath || DEFAULT_SETTINGS.pythonPath).trim(),
    customVocabulary: vocabularyTerms(settings.customVocabulary ?? DEFAULT_SETTINGS.customVocabulary).join('\n')
  }
  for (const key of Object.keys(JEV_DEFAULTS) as (keyof JevThresholdSettings)[]) {
    const value = probability(normalized[key])
    if (value === null) throw new Error(`Invalid Jev threshold: ${key}. Use a probability from 0 to 1.`)
    normalized[key] = value
  }
  normalized.outputDirectory ||= DEFAULT_SETTINGS.outputDirectory
  normalized.pythonPath ||= DEFAULT_SETTINGS.pythonPath
  if (!SWITCHES.includes(normalized.jevEnabled)) throw new Error('Invalid Jev review setting')
  if (!SWITCHES.includes(normalized.jevVisualContext)) throw new Error('Invalid visual context setting')
  if (!SWITCHES.includes(normalized.sourceContextWebResearch)) throw new Error('Invalid source research setting')
  if (!SWITCHES.includes(normalized.openaiCompatibleVision)) throw new Error('Invalid image analysis setting')
  if (!SWITCHES.includes(normalized.clipJudge)) throw new Error('Invalid second opinion setting')
  if (!isAbsolute(normalized.outputDirectory)) throw new Error('Settings folders must be absolute paths')
  return normalized
}

/** A canonical probability string from 0 to 1, or null. */
function probability(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  if (!/^\d+(?:\.\d+)?(?:e[+-]?\d+)?$/i.test(trimmed) || !Number.isFinite(Number(trimmed)) || Number(trimmed) > 1) return null
  return String(Number(trimmed))
}

/**
 * Saved Jev fields are optional preferences. One unreadable value falls back to
 * its own default instead of making the whole file (and the saved keys) unreadable.
 */
function savedJevSettings(raw: Record<string, unknown>): Pick<AppSettings, keyof JevThresholdSettings | 'jevEnabled' | 'jevVisualContext' | 'sourceContextWebResearch'> {
  const thresholds = Object.fromEntries((Object.keys(JEV_DEFAULTS) as (keyof JevThresholdSettings)[])
    .map((key) => [key, probability(raw[key]) ?? JEV_DEFAULTS[key]])) as JevThresholdSettings
  const switchValue = (value: unknown, fallback: string): string => typeof value === 'string' && SWITCHES.includes(value) ? value : fallback
  const optedIn = typeof raw.version === 'number' && raw.version >= OPT_IN_BETA_VERSION
  return {
    ...thresholds,
    jevEnabled: optedIn ? switchValue(raw.jevEnabled, DEFAULT_SETTINGS.jevEnabled) : 'off',
    jevVisualContext: switchValue(raw.jevVisualContext ?? raw.typesafeVisualContext, DEFAULT_SETTINGS.jevVisualContext),
    sourceContextWebResearch: optedIn ? switchValue(raw.sourceContextWebResearch, DEFAULT_SETTINGS.sourceContextWebResearch) : 'off'
  }
}

/**
 * Saved provider choices, read leniently: a value that is missing, malformed or no
 * longer allowed falls back to its default instead of making the whole file (and the
 * saved keys) unreadable.
 *
 * Files written while this feature was in development filled in placeholder values
 * (an Ollama URL, a Whisper server URL, the model names llama3 and whisper-1) that
 * the person never chose. Those files have no `localWhisperMode`, which every file
 * written since then does, so the placeholders are cleared only from such files.
 */
function savedProviderSettings(raw: Record<string, unknown>): Pick<AppSettings,
  'aiProvider' | 'geminiModel' | 'openaiCompatibleUrl' | 'openaiCompatibleModel' | 'openaiCompatibleVision' | 'openaiCompatibleTimeout' |
  'transcriptionSource' | 'localWhisperMode' | 'localWhisperUrl' | 'localWhisperModel' | 'localWhisperDevice' | 'localWhisperLanguage'> {
  const legacy = !Object.hasOwn(raw, 'localWhisperMode')
  const text = (value: unknown): string => typeof value === 'string' ? value.trim() : ''
  const pick = <T extends string>(value: unknown, allowed: readonly T[], fallback: T): T =>
    typeof value === 'string' && allowed.includes(value as T) ? value as T : fallback

  const aiProvider = pick(raw.aiProvider, AI_PROVIDERS, DEFAULT_SETTINGS.aiProvider)
  const transcriptionSource = pick(raw.transcriptionSource, TRANSCRIPTION_SOURCES, DEFAULT_SETTINGS.transcriptionSource)
  const localWhisperMode = pick(raw.localWhisperMode, LOCAL_WHISPER_MODES, DEFAULT_SETTINGS.localWhisperMode)

  let openaiCompatibleUrl = text(raw.openaiCompatibleUrl)
  let openaiCompatibleModel = text(raw.openaiCompatibleModel)
  let localWhisperUrl = text(raw.localWhisperUrl)
  let localWhisperModelValue = text(raw.localWhisperModel)
  let geminiModelValue = text(raw.geminiModel)
  if (legacy) {
    // The model that development builds filled in is no longer open to new Gemini keys.
    if (/^gemini-(?:1\.5|2\.0|2\.5)-/.test(geminiModelValue)) geminiModelValue = ''
    if (openaiCompatibleUrl === 'http://localhost:11434/v1' && aiProvider !== 'openai-compatible') openaiCompatibleUrl = ''
    if (openaiCompatibleModel === 'llama3' && aiProvider !== 'openai-compatible') openaiCompatibleModel = ''
    if (localWhisperUrl === 'http://127.0.0.1:8000/v1') localWhisperUrl = ''
    if (localWhisperModelValue === 'whisper-1') localWhisperModelValue = ''
  }
  const urlOrEmpty = (value: string): string => {
    const checked = checkEndpointUrl(value)
    return checked.ok ? checked.url : ''
  }
  // A saved public Whisper URL breaks the "audio stays local" promise: it loads as not configured.
  const whisperUrlOrEmpty = (value: string): string => {
    const checked = checkLocalWhisperUrl(value)
    return checked.ok ? checked.url : ''
  }
  const modelOrEmpty = (value: string): string => isSafeModelName(value) ? value : ''
  const localModel = localWhisperMode === 'builtin'
    ? (isBuiltinWhisperModel(localWhisperModelValue) ? localWhisperModelValue : DEFAULT_LOCAL_WHISPER_MODEL)
    : (modelOrEmpty(localWhisperModelValue) || DEFAULT_LOCAL_WHISPER_MODEL)

  return {
    aiProvider,
    geminiModel: modelOrEmpty(geminiModelValue) || DEFAULT_GEMINI_MODEL,
    openaiCompatibleUrl: urlOrEmpty(openaiCompatibleUrl),
    openaiCompatibleModel: modelOrEmpty(openaiCompatibleModel),
    openaiCompatibleVision: typeof raw.openaiCompatibleVision === 'string' && SWITCHES.includes(raw.openaiCompatibleVision) ? raw.openaiCompatibleVision : DEFAULT_SETTINGS.openaiCompatibleVision,
    openaiCompatibleTimeout: pick(raw.openaiCompatibleTimeout, OPENAI_COMPATIBLE_TIMEOUTS, DEFAULT_SETTINGS.openaiCompatibleTimeout),
    transcriptionSource,
    localWhisperMode,
    localWhisperUrl: whisperUrlOrEmpty(localWhisperUrl),
    localWhisperModel: localModel,
    // Files saved before this setting existed have none, and so load as Auto.
    localWhisperDevice: pick(raw.localWhisperDevice, LOCAL_WHISPER_DEVICES, DEFAULT_SETTINGS.localWhisperDevice),
    localWhisperLanguage: pick(raw.localWhisperLanguage, LOCAL_WHISPER_LANGUAGE_CODES, DEFAULT_SETTINGS.localWhisperLanguage)
  }
}

/** Saved framing and encoder choices. Files from before these settings existed load the defaults. */
function savedFramingSettings(raw: Record<string, unknown>): Pick<AppSettings, 'webcamPosition' | 'webcamLocation' | 'screenFraming' | 'videoEncoder'> {
  const pick = <T extends string>(value: unknown, allowed: readonly T[], fallback: T): T =>
    typeof value === 'string' && allowed.includes(value as T) ? value as T : fallback
  return {
    webcamPosition: pick(raw.webcamPosition, WEBCAM_POSITIONS, DEFAULT_SETTINGS.webcamPosition),
    webcamLocation: pick(raw.webcamLocation, WEBCAM_LOCATIONS, DEFAULT_SETTINGS.webcamLocation),
    screenFraming: pick(raw.screenFraming, SCREEN_FRAMINGS, DEFAULT_SETTINGS.screenFraming),
    videoEncoder: pick(raw.videoEncoder, VIDEO_ENCODERS, DEFAULT_SETTINGS.videoEncoder)
  }
}

function canEncrypt(): boolean {
  return app.isReady() && safeStorage.isEncryptionAvailable() &&
    (process.platform !== 'linux' || safeStorage.getSelectedStorageBackend() !== 'basic_text')
}

function encodeSecret(value: string): PersistedSecret {
  if (!value) return ''
  if (canEncrypt()) {
    return { scheme: 'safeStorage', value: safeStorage.encryptString(value).toString('base64') }
  }
  throw new Error('Secure key storage is unavailable. Unlock or configure your operating system keychain before saving API keys.')
}

/**
 * Decodes a stored secret. Accepts the current `{scheme, value}` shape and the
 * plain base64 strings written by versions <= 0.1.16. `legacy` reports whether
 * the value should be re-encrypted.
 */
function decodeSecret(value: unknown): { value: string; legacy: boolean } {
  if (!value) return { value: '', legacy: false }
  if (typeof value === 'string') {
    if (!canEncrypt()) throw new Error('Secure key storage is unavailable')
    try {
      return { value: Buffer.from(value, 'base64').toString('utf-8'), legacy: true }
    } catch {
      return { value: '', legacy: false }
    }
  }
  if (typeof value === 'object') {
    const obj = value as { scheme?: string; value?: string }
    if (!obj.value) return { value: '', legacy: false }
    try {
      if (obj.scheme === 'safeStorage') {
        if (!canEncrypt()) throw new Error('Secure key storage is unavailable')
        return { value: safeStorage.decryptString(Buffer.from(obj.value, 'base64')), legacy: false }
      }
      if (obj.scheme === 'base64') {
        if (!canEncrypt()) throw new Error('Secure key storage is unavailable')
        return { value: Buffer.from(obj.value, 'base64').toString('utf-8'), legacy: true }
      }
    } catch {
      throw new Error('Could not decrypt saved API keys. Unlock the system keychain and retry.')
    }
  }
  return { value: '', legacy: false }
}

export function loadSettings(): AppSettings {
  const path = getSettingsPath()
  if (!existsSync(path)) return { ...DEFAULT_SETTINGS }

  try {
    const raw = JSON.parse(readFileSync(path, 'utf-8'))
    let needsMigration = raw.version !== SETTINGS_VERSION || Object.hasOwn(raw, 'elevenLabsApiKey') || Object.hasOwn(raw, 'typesafeApiKey')
    const secrets = {} as Record<SecretKey, string>
    for (const key of SECRET_KEYS) {
      const decoded = decodeSecret(raw[key])
      secrets[key] = decoded.value
      if (decoded.legacy) needsMigration = true
    }

    const settings = normalizeSettings({
      ...secrets,
      ...savedJevSettings(raw),
      ...savedProviderSettings(raw),
      ...savedFramingSettings(raw),
      // Files saved before this setting existed have none, and so load as on.
      clipJudge: typeof raw.clipJudge === 'string' && SWITCHES.includes(raw.clipJudge) ? raw.clipJudge : DEFAULT_SETTINGS.clipJudge,
      outputDirectory: typeof raw.outputDirectory === 'string' ? raw.outputDirectory : DEFAULT_SETTINGS.outputDirectory,
      pythonPath: typeof raw.pythonPath === 'string' ? raw.pythonPath : DEFAULT_SETTINGS.pythonPath,
      customVocabulary: typeof raw.customVocabulary === 'string' ? raw.customVocabulary : DEFAULT_SETTINGS.customVocabulary
    })

    if (needsMigration && canEncrypt()) writeSettings(settings)
    return settings
  } catch (error) {
    throw new Error('Could not read saved settings. The settings file was kept for recovery.', { cause: error })
  }
}

function writeSettings(settings: AppSettings): void {
  const path = getSettingsPath()
  const tempPath = `${path}.${randomUUID()}.tmp`

  const persisted: PersistedSettings = {
    version: SETTINGS_VERSION,
    jevThreshold: settings.jevThreshold,
    jevSelfContainedThreshold: settings.jevSelfContainedThreshold,
    jevFaithfulToSourceThreshold: settings.jevFaithfulToSourceThreshold,
    jevTitleSupportedThreshold: settings.jevTitleSupportedThreshold,
    jevSponsorThreshold: settings.jevSponsorThreshold,
    jevEvidenceThreshold: settings.jevEvidenceThreshold,
    jevCutThreshold: settings.jevCutThreshold,

    openrouterApiKey: encodeSecret(settings.openrouterApiKey),
    zernioApiKey: encodeSecret(settings.zernioApiKey),
    aiProvider: settings.aiProvider,
    geminiApiKey: encodeSecret(settings.geminiApiKey),
    geminiModel: settings.geminiModel,
    openaiCompatibleUrl: settings.openaiCompatibleUrl,
    openaiCompatibleApiKey: encodeSecret(settings.openaiCompatibleApiKey),
    openaiCompatibleModel: settings.openaiCompatibleModel,
    openaiCompatibleVision: settings.openaiCompatibleVision,
    openaiCompatibleTimeout: settings.openaiCompatibleTimeout,
    transcriptionSource: settings.transcriptionSource,
    localWhisperMode: settings.localWhisperMode,
    localWhisperUrl: settings.localWhisperUrl,
    localWhisperModel: settings.localWhisperModel,
    localWhisperDevice: settings.localWhisperDevice,
    localWhisperLanguage: settings.localWhisperLanguage,
    webcamPosition: settings.webcamPosition,
    webcamLocation: settings.webcamLocation,
    screenFraming: settings.screenFraming,
    videoEncoder: settings.videoEncoder,
    clipJudge: settings.clipJudge,
    jevEnabled: settings.jevEnabled,
    jevVisualContext: settings.jevVisualContext,
    sourceContextWebResearch: settings.sourceContextWebResearch,
    outputDirectory: settings.outputDirectory,
    pythonPath: settings.pythonPath,
    customVocabulary: settings.customVocabulary
  }

  let fd: number | undefined
  try {
    fd = openSync(tempPath, 'wx', 0o600)
    writeFileSync(fd, JSON.stringify(persisted, null, 2), { encoding: 'utf-8' })
    fchmodSync(fd, 0o600)
    fsyncSync(fd)
    closeSync(fd)
    fd = undefined
    renameSync(tempPath, path)
  } finally {
    if (fd !== undefined) closeSync(fd)
    if (existsSync(tempPath)) unlinkSync(tempPath)
  }
}

export function saveSettings(settings: AppSettings): AppSettings {
  writeSettings(normalizeSettings(settings))
  return loadSettings()
}

export function publicSettings(settings: AppSettings): PublicSettings {
  return {
    jevThreshold: settings.jevThreshold,
    jevSelfContainedThreshold: settings.jevSelfContainedThreshold,
    jevFaithfulToSourceThreshold: settings.jevFaithfulToSourceThreshold,
    jevTitleSupportedThreshold: settings.jevTitleSupportedThreshold,
    jevSponsorThreshold: settings.jevSponsorThreshold,
    jevEvidenceThreshold: settings.jevEvidenceThreshold,
    jevCutThreshold: settings.jevCutThreshold,

    outputDirectory: settings.outputDirectory,
    pythonPath: settings.pythonPath,
    customVocabulary: settings.customVocabulary,
    openrouterConfigured: Boolean(settings.openrouterApiKey),
    zernioConfigured: Boolean(settings.zernioApiKey),
    geminiConfigured: Boolean(settings.geminiApiKey),
    openaiCompatibleConfigured: Boolean(settings.openaiCompatibleApiKey),
    aiProvider: settings.aiProvider,
    geminiModel: settings.geminiModel,
    openaiCompatibleUrl: settings.openaiCompatibleUrl,
    openaiCompatibleModel: settings.openaiCompatibleModel,
    openaiCompatibleVision: settings.openaiCompatibleVision,
    openaiCompatibleTimeout: settings.openaiCompatibleTimeout,
    transcriptionSource: settings.transcriptionSource,
    localWhisperMode: settings.localWhisperMode,
    localWhisperUrl: settings.localWhisperUrl,
    localWhisperModel: settings.localWhisperModel,
    localWhisperDevice: settings.localWhisperDevice,
    localWhisperLanguage: settings.localWhisperLanguage,
    webcamPosition: settings.webcamPosition,
    webcamLocation: settings.webcamLocation,
    screenFraming: settings.screenFraming,
    videoEncoder: settings.videoEncoder,
    clipJudge: settings.clipJudge,
    jevEnabled: settings.jevEnabled,
    jevVisualContext: settings.jevVisualContext,
    sourceContextWebResearch: settings.sourceContextWebResearch
  }
}

export function savePublicSettings(update: Partial<PublicSettings>): PublicSettings {
  const current = loadSettings()
  return publicSettings(saveSettings({
    ...current,
    jevThreshold: update.jevThreshold ?? current.jevThreshold,
    jevSelfContainedThreshold: update.jevSelfContainedThreshold ?? current.jevSelfContainedThreshold,
    jevFaithfulToSourceThreshold: update.jevFaithfulToSourceThreshold ?? current.jevFaithfulToSourceThreshold,
    jevTitleSupportedThreshold: update.jevTitleSupportedThreshold ?? current.jevTitleSupportedThreshold,
    jevSponsorThreshold: update.jevSponsorThreshold ?? current.jevSponsorThreshold,
    jevEvidenceThreshold: update.jevEvidenceThreshold ?? current.jevEvidenceThreshold,
    jevCutThreshold: update.jevCutThreshold ?? current.jevCutThreshold,
    outputDirectory: update.outputDirectory ?? current.outputDirectory,
    pythonPath: update.pythonPath ?? current.pythonPath,
    customVocabulary: update.customVocabulary ?? current.customVocabulary,
    aiProvider: update.aiProvider ?? current.aiProvider,
    geminiModel: update.geminiModel ?? current.geminiModel,
    openaiCompatibleUrl: update.openaiCompatibleUrl ?? current.openaiCompatibleUrl,
    openaiCompatibleModel: update.openaiCompatibleModel ?? current.openaiCompatibleModel,
    openaiCompatibleVision: update.openaiCompatibleVision ?? current.openaiCompatibleVision,
    openaiCompatibleTimeout: update.openaiCompatibleTimeout ?? current.openaiCompatibleTimeout,
    transcriptionSource: update.transcriptionSource ?? current.transcriptionSource,
    localWhisperMode: update.localWhisperMode ?? current.localWhisperMode,
    localWhisperUrl: update.localWhisperUrl ?? current.localWhisperUrl,
    localWhisperModel: update.localWhisperModel ?? current.localWhisperModel,
    localWhisperDevice: update.localWhisperDevice ?? current.localWhisperDevice,
    localWhisperLanguage: update.localWhisperLanguage ?? current.localWhisperLanguage,
    webcamPosition: update.webcamPosition ?? current.webcamPosition,
    webcamLocation: update.webcamLocation ?? current.webcamLocation,
    screenFraming: update.screenFraming ?? current.screenFraming,
    videoEncoder: update.videoEncoder ?? current.videoEncoder,
    clipJudge: update.clipJudge ?? current.clipJudge,
    jevEnabled: update.jevEnabled ?? current.jevEnabled,
    jevVisualContext: update.jevVisualContext ?? current.jevVisualContext,
    sourceContextWebResearch: update.sourceContextWebResearch ?? current.sourceContextWebResearch
  }))
}

/**
 * Conservative vocabulary hints for MAI Transcribe 2: one term per line or comma,
 * at most five words and 49 characters each, none of <>{}[]\, deduplicated
 * case-insensitively. These application limits keep phrase hints short and bounded.
 */
export function vocabularyTerms(value: string): string[] {
  const terms: string[] = []
  const seen = new Set<string>()
  for (const line of value.split(/[\n,]/)) {
    const term = line.replace(/\s+/g, ' ').trim()
    if (!term || term.length > 49 || term.split(' ').length > 5 || /[<>{}[\]\\]/.test(term)) continue
    const key = term.toLocaleLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    terms.push(term)
    if (terms.length >= 200) break
  }
  return terms
}

export function replaceApiKey(key: ApiKeyName, value: string): PublicSettings {
  if (!SECRET_KEYS.includes(key) || typeof value !== 'string' || value.length > 8192 || value.includes('\0')) throw new Error('Invalid API key update')
  const current = loadSettings()
  return publicSettings(saveSettings({ ...current, [key]: value.trim() }))
}

/**
 * Environment for the Python engine, the only channel provider settings take to it.
 *
 * Jev review, Jev frame context and web research of the source are OpenRouter
 * services. They are forced off for any other provider so a run never spends
 * OpenRouter credit the person did not choose to spend. The OpenRouter key itself
 * is sent only when the run uses OpenRouter, for planning or for cloud transcription.
 */
export function getSettingsForBridge(settings: AppSettings): Record<string, string> {
  const openrouterPlanning = settings.aiProvider === 'openrouter'
  const needsOpenRouterKey = openrouterPlanning || settings.transcriptionSource === 'cloud'
  return {
    AI_PROVIDER: settings.aiProvider,
    OPENROUTER_API_KEY: needsOpenRouterKey ? settings.openrouterApiKey : '',
    GEMINI_API_KEY: settings.aiProvider === 'gemini' ? settings.geminiApiKey : '',
    GEMINI_MODEL: settings.geminiModel,
    OPENAI_COMPATIBLE_URL: settings.aiProvider === 'openai-compatible' ? settings.openaiCompatibleUrl : '',
    OPENAI_COMPATIBLE_API_KEY: settings.aiProvider === 'openai-compatible' ? settings.openaiCompatibleApiKey : '',
    OPENAI_COMPATIBLE_MODEL: settings.openaiCompatibleModel,
    OPENAI_COMPATIBLE_VISION: settings.openaiCompatibleVision === 'on' ? 'true' : 'false',
    OPENAI_COMPATIBLE_TIMEOUT_SECONDS: String(Number(settings.openaiCompatibleTimeout) * 60),
    TRANSCRIPTION_SOURCE: settings.transcriptionSource,
    LOCAL_WHISPER_MODE: settings.localWhisperMode,
    LOCAL_WHISPER_URL: settings.transcriptionSource === 'local' && settings.localWhisperMode === 'server' ? settings.localWhisperUrl : '',
    LOCAL_WHISPER_MODEL: settings.localWhisperModel,
    // Only the built-in engine reads this; it is sent always so a switch back to built-in needs no restart.
    LOCAL_WHISPER_DEVICE: settings.localWhisperDevice,
    LOCAL_WHISPER_LANGUAGE: settings.localWhisperLanguage,
    WEBCAM_POSITION: settings.webcamPosition,
    WEBCAM_LOCATION: settings.webcamLocation,
    SCREEN_FILL: settings.screenFraming === 'fill' ? 'true' : 'false',
    VIDEO_ENCODER: settings.videoEncoder,
    CLIP_JUDGE: settings.clipJudge === 'off' ? 'false' : 'true',
    SOURCE_CONTEXT_WEB_RESEARCH: openrouterPlanning && settings.sourceContextWebResearch !== 'off' ? 'true' : 'false',
    JEV_THRESHOLD: settings.jevThreshold,
    JEV_SELF_CONTAINED_THRESHOLD: settings.jevSelfContainedThreshold,
    JEV_FAITHFUL_TO_SOURCE_THRESHOLD: settings.jevFaithfulToSourceThreshold,
    JEV_TITLE_SUPPORTED_THRESHOLD: settings.jevTitleSupportedThreshold,
    JEV_SPONSOR_THRESHOLD: settings.jevSponsorThreshold,
    JEV_EVIDENCE_THRESHOLD: settings.jevEvidenceThreshold,
    JEV_CUT_THRESHOLD: settings.jevCutThreshold,
    // The review runs on every provider: Jev on OpenRouter, the chosen model otherwise.
    JEV_ENABLED: settings.jevEnabled === 'on' ? 'true' : 'false',
    // Frames go only to a model that can read them: always for OpenRouter and Gemini, and for a
    // custom endpoint only when the person said it can.
    JEV_VISUAL_CONTEXT: settings.jevVisualContext === 'on' && (settings.aiProvider !== 'openai-compatible' || settings.openaiCompatibleVision === 'on') ? 'true' : 'false',
    LOCAL_MODE: 'true',
    LOCAL_OUTPUT_DIR: settings.outputDirectory
  }
}
