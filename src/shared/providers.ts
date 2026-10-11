/**
 * AI provider and transcription choices, shared by the main process (gates,
 * settings validation) and the renderer (setup flow, Settings page) so both
 * always agree on what "configured" means.
 *
 * An empty string always means "not configured". Nothing here supplies a
 * hidden default for a URL or a custom model name.
 */

export const AI_PROVIDERS = ['openrouter', 'gemini', 'openai-compatible'] as const
export type AiProvider = (typeof AI_PROVIDERS)[number]

export const TRANSCRIPTION_SOURCES = ['cloud', 'local'] as const
export type TranscriptionSource = (typeof TRANSCRIPTION_SOURCES)[number]

export const LOCAL_WHISPER_MODES = ['builtin', 'server'] as const
export type LocalWhisperMode = (typeof LOCAL_WHISPER_MODES)[number]

/** Google's OpenAI-compatible endpoint for the Gemini API. */
export const GEMINI_OPENAI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai'
export const DEFAULT_GEMINI_MODEL = 'gemini-3.8-flash'
/** Offered while typing a Gemini model, before Test connection has listed the models a key can use. */
export const GEMINI_MODEL_SUGGESTIONS: readonly string[] = ['gemini-3.8-flash', 'gemini-3.1-pro-preview', 'gemini-3.5-flash-lite']
export const DEFAULT_LOCAL_WHISPER_MODEL = 'small'

/** One line of a Test connection result. */
export type ProviderCheckId = 'server' | 'key' | 'model' | 'json' | 'images'
export type ProviderCheckState = 'ok' | 'failed' | 'warning' | 'skipped'
export interface ProviderCheck {
  id: ProviderCheckId
  state: ProviderCheckState
  /** A fixed sentence, plus at most the model name the person typed. Never provider text. */
  message: string
}

export const PROVIDER_CHECK_LABELS: Record<ProviderCheckId, string> = {
  server: 'Server',
  key: 'Key',
  model: 'Model',
  json: 'JSON answers',
  images: 'Images'
}

/** What "Test connection" tells the Settings page. Never contains a key or a provider reply. */
export interface ProviderTestResult {
  ok: boolean
  message: string
  /** Model ids the provider lists, for the model field's suggestions. At most 200. */
  models: string[]
  checks: ProviderCheck[]
  /** Whether the model read the test image, or null when that was not tested. */
  readsImages: boolean | null
  /** What was tested (testSubject), so a result is not shown for settings changed since. */
  subject: string
}

/** Which provider, endpoint and model a Test connection result is about. */
export function testSubject(provider: AiProvider, settings: { openaiCompatibleUrl: string; openaiCompatibleModel: string; geminiModel: string }): string {
  if (provider === 'gemini') return `gemini|${(settings.geminiModel || DEFAULT_GEMINI_MODEL).trim()}`
  if (provider === 'openai-compatible') {
    const checked = checkEndpointUrl(settings.openaiCompatibleUrl)
    return `openai-compatible|${checked.ok ? checked.url : ''}|${settings.openaiCompatibleModel.trim()}`
  }
  return 'openrouter'
}

/**
 * What the Test connection result suggests for "This model can read images" on a custom endpoint:
 * turn it on when the model read the test image, off when it could not. Gemini always reads images.
 */
export function visionAdvice(provider: AiProvider, readsImages: boolean | null, visionOn: boolean): 'turn-on' | 'turn-off' | null {
  if (provider !== 'openai-compatible' || readsImages === null) return null
  if (readsImages && !visionOn) return 'turn-on'
  if (!readsImages && visionOn) return 'turn-off'
  return null
}

/** How long BridgeClip waits for an OpenAI-compatible endpoint to answer one request, in minutes. */
export const OPENAI_COMPATIBLE_TIMEOUTS = ['5', '10', '30', '60', '120'] as const
export type OpenaiCompatibleTimeout = (typeof OPENAI_COMPATIBLE_TIMEOUTS)[number]
export const DEFAULT_OPENAI_COMPATIBLE_TIMEOUT: OpenaiCompatibleTimeout = '30'

/** An Ollama server: its default port 11434, or "ollama" in the host name. */
export function looksLikeOllama(url: string): boolean {
  try {
    const parsed = new URL(url.trim())
    return parsed.port === '11434' || parsed.hostname.toLowerCase().includes('ollama')
  } catch {
    return false
  }
}

/**
 * The settings change for choosing an AI provider. Cloud transcription runs on OpenRouter, so
 * choosing Gemini or a custom endpoint without an OpenRouter key also moves transcription to
 * Local Whisper; otherwise the next job would be blocked for a key the person never meant to add.
 * `movedEarlier` says that move was made by an earlier choice; going back to OpenRouter undoes it,
 * so looking at the other options (arrow keys select each one) leaves transcription as it was.
 */
export function aiProviderChange(
  provider: AiProvider, current: { openrouterConfigured: boolean; transcriptionSource: TranscriptionSource }, movedEarlier = false
): { aiProvider: AiProvider; transcriptionSource?: TranscriptionSource } {
  if (provider === 'openrouter') {
    return { aiProvider: provider, ...(movedEarlier && current.transcriptionSource === 'local' ? { transcriptionSource: 'cloud' as const } : {}) }
  }
  const moveToLocal = !current.openrouterConfigured && current.transcriptionSource === 'cloud'
  return { aiProvider: provider, ...(moveToLocal ? { transcriptionSource: 'local' as const } : {}) }
}

export const AI_PROVIDER_LABELS: Record<AiProvider, string> = {
  openrouter: 'OpenRouter',
  gemini: 'Google Gemini',
  'openai-compatible': 'OpenAI-compatible'
}

export interface LocalWhisperModelInfo {
  id: string
  label: string
  /** Plain-words speed and quality note for the model picker. */
  hint: string
  /** Approximate one-time download the first time the model is used. */
  downloadMb: number
}

/** Models the built-in Whisper engines understand, smallest first. */
export const LOCAL_WHISPER_MODELS: readonly LocalWhisperModelInfo[] = [
  { id: 'tiny', label: 'Tiny', hint: 'Fastest, roughest captions. Good for a quick test.', downloadMb: 75 },
  { id: 'tiny.en', label: 'Tiny (English)', hint: 'Fastest, roughest captions. English audio only.', downloadMb: 75 },
  { id: 'base', label: 'Base', hint: 'Fast. Fine for clear, single-speaker audio.', downloadMb: 145 },
  { id: 'base.en', label: 'Base (English)', hint: 'Fast. Fine for clear English, single-speaker audio.', downloadMb: 145 },
  { id: 'small', label: 'Small', hint: 'Recommended on most computers. Good accuracy at a reasonable speed.', downloadMb: 480 },
  { id: 'small.en', label: 'Small (English)', hint: 'Good accuracy at a reasonable speed. English audio only.', downloadMb: 480 },
  { id: 'medium', label: 'Medium', hint: 'Better with accents and noise. Several times slower than Small without a GPU.', downloadMb: 1500 },
  { id: 'medium.en', label: 'Medium (English)', hint: 'Better with accents and noise, English audio only. Slow without a GPU.', downloadMb: 1500 },
  { id: 'turbo', label: 'Turbo', hint: 'Near the best accuracy and much faster than Large. Needs a GPU or a lot of patience.', downloadMb: 1600 },
  { id: 'large-v3', label: 'Large v3', hint: 'Best accuracy. Slow without a GPU.', downloadMb: 3000 }
]

const LOCAL_WHISPER_MODEL_IDS: readonly string[] = [
  ...LOCAL_WHISPER_MODELS.map((model) => model.id),
  'large', 'large-v1', 'large-v2', 'large-v3-turbo'
]

/** A model or deployment name sent to a provider: no spaces, no control characters. */
const SAFE_MODEL_NAME = /^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$/

export function isSafeModelName(value: string): boolean {
  return SAFE_MODEL_NAME.test(value)
}

export function isBuiltinWhisperModel(value: string): boolean {
  return LOCAL_WHISPER_MODEL_IDS.includes(value)
}

export type HostClass = 'loopback' | 'private' | 'public' | 'blocked'

const MAX_ENDPOINT_URL_LENGTH = 512

function ipv4Octets(host: string): number[] | null {
  const match = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host)
  if (!match) return null
  const octets = match.slice(1).map(Number)
  return octets.every((part) => part <= 255) ? octets : null
}

function classifyIpv4(octets: number[]): HostClass {
  const [a, b] = octets
  if (a === 0 || a >= 224) return 'blocked'
  if (a === 169 && b === 254) return 'blocked'
  // 100.64.0.0/10 (carrier-grade NAT and Tailscale, which also covers the 100.100.100.200 metadata
  // address). The engine's network guard refuses it, so a URL there could never work: fail closed here too.
  if (a === 100 && b >= 64 && b <= 127) return 'blocked'
  if (a === 127) return 'loopback'
  if (a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168)) return 'private'
  return 'public'
}

/** The IPv4 address inside an IPv4-mapped IPv6 literal such as ::ffff:7f00:1, or null. */
function mappedIpv4(host: string): number[] | null {
  const hex = /^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/.exec(host)
  if (hex) {
    const high = parseInt(hex[1], 16)
    const low = parseInt(hex[2], 16)
    return [high >> 8, high & 255, low >> 8, low & 255]
  }
  const dotted = /^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/.exec(host)
  return dotted ? ipv4Octets(dotted[1]) : null
}

/**
 * Where a host sits: this computer, a private network, the public internet, or
 * somewhere a provider URL must never point (link-local, cloud metadata,
 * unspecified and multicast addresses).
 */
export function classifyHost(rawHost: string): HostClass {
  const host = rawHost.trim().toLowerCase().replace(/^\[|\]$/g, '').replace(/\.$/, '')
  if (!host || host === '::') return 'blocked'
  if (host === 'metadata' || host === 'metadata.google.internal' || host === 'instance-data') return 'blocked'
  const v4 = ipv4Octets(host)
  if (v4) return classifyIpv4(v4)
  if (host.includes(':')) {
    const mapped = mappedIpv4(host)
    if (mapped) return classifyIpv4(mapped)
    if (host === '::1') return 'loopback'
    if (host === 'fd00:ec2::254' || /^fe[89ab]/.test(host) || host.startsWith('ff')) return 'blocked'
    if (/^f[cd]/.test(host)) return 'private'
    return 'public'
  }
  if (host === 'localhost' || host.endsWith('.localhost')) return 'loopback'
  // Local names resolve inside the network: .local, .lan, .internal and bare hostnames
  // such as host.docker.internal or a NAS called "mini".
  if (host.endsWith('.local') || host.endsWith('.lan') || host.endsWith('.internal') || !host.includes('.')) return 'private'
  return 'public'
}

export type EndpointCheck = { ok: true; url: string } | { ok: false; error: string }

/**
 * Validates a provider or Whisper server URL. Only http(s) with no embedded
 * credentials, query or fragment (a key in the URL would be stored unencrypted),
 * plain http only for this computer or a private network, and never an address
 * reserved for link-local or cloud-metadata use. An empty value is valid and means
 * "not configured".
 */
export function checkEndpointUrl(value: string): EndpointCheck {
  const trimmed = value.trim()
  if (!trimmed) return { ok: true, url: '' }
  if (trimmed.length > MAX_ENDPOINT_URL_LENGTH) return { ok: false, error: 'The URL is too long.' }
  let parsed: URL
  try {
    parsed = new URL(trimmed)
  } catch {
    return { ok: false, error: 'Enter a full URL such as http://localhost:11434/v1.' }
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return { ok: false, error: 'Use an http:// or https:// URL.' }
  if (parsed.username || parsed.password) return { ok: false, error: 'Remove the username and password from the URL. Put the key in the API key field.' }
  if (parsed.search || parsed.hash) return { ok: false, error: 'Remove everything after the path (no ? or #).' }
  const hostClass = classifyHost(parsed.hostname)
  if (hostClass === 'blocked') return { ok: false, error: 'That address is not allowed for a provider.' }
  if (parsed.protocol === 'http:' && hostClass === 'public') return { ok: false, error: 'Use https:// for servers outside your own network.' }
  return { ok: true, url: trimmed.replace(/\/+$/, '') }
}

export const LOCAL_WHISPER_URL_MESSAGE =
  'Local Whisper must run on this computer or your private network. For a cloud service use Cloud transcription.'

/**
 * Validates the URL of a Whisper server chosen under "Local Whisper". The privacy promise of that
 * option is that audio stays on this computer or the user's own network, so a public host is
 * refused rather than quietly accepted. An empty value is valid and means "not configured".
 */
export function checkLocalWhisperUrl(value: string): EndpointCheck {
  const checked = checkEndpointUrl(value)
  if (!checked.ok || !checked.url) return checked
  return isLocalEndpoint(checked.url) ? checked : { ok: false, error: LOCAL_WHISPER_URL_MESSAGE }
}

/** True when a configured URL points at this computer or a private network. */
export function isLocalEndpoint(url: string): boolean {
  try {
    const hostClass = classifyHost(new URL(url).hostname)
    return hostClass === 'loopback' || hostClass === 'private'
  } catch {
    return false
  }
}

/**
 * The settings change for switching between built-in Whisper and a Whisper server. A model name
 * that only suits the other mode would be rejected, so the model moves to one that fits. It takes
 * the model as it is when the change is applied, not as it was when the button was pressed.
 */
export function localWhisperModeChange(mode: LocalWhisperMode, currentModel: string): { localWhisperMode: LocalWhisperMode; localWhisperModel?: string } {
  const fits = mode === 'builtin' ? isBuiltinWhisperModel(currentModel) : isSafeModelName(currentModel)
  return { localWhisperMode: mode, ...(fits ? {} : { localWhisperModel: DEFAULT_LOCAL_WHISPER_MODEL }) }
}

/** What the setup gate needs to know. Every field is available to the renderer. */
export interface ProviderSetupInput {
  aiProvider: AiProvider
  openrouterConfigured: boolean
  geminiConfigured: boolean
  openaiCompatibleUrl: string
  openaiCompatibleModel: string
  transcriptionSource: TranscriptionSource
  localWhisperMode: LocalWhisperMode
  localWhisperUrl: string
}

export type SetupGapId = 'openrouter-key' | 'gemini-key' | 'openai-url' | 'openai-model' | 'whisper-url'

export interface SetupGap {
  id: SetupGapId
  /** Short noun phrase for a button or sidebar label: "OpenRouter key". */
  label: string
  /** One sentence for the person, plain words. */
  message: string
  /** What the Create page's disabled button says, where that differs from `message`. */
  blocked?: string
}

/** Whether the chosen AI provider has what it needs to pick clips. */
export function aiReady(input: ProviderSetupInput): boolean {
  if (input.aiProvider === 'openrouter') return input.openrouterConfigured
  if (input.aiProvider === 'gemini') return input.geminiConfigured
  return Boolean(input.openaiCompatibleUrl.trim() && input.openaiCompatibleModel.trim())
}

/** Whether the chosen way of turning speech into text has what it needs. */
export function transcriptionReady(input: ProviderSetupInput): boolean {
  if (input.transcriptionSource === 'cloud') return input.openrouterConfigured
  return input.localWhisperMode === 'builtin' || Boolean(input.localWhisperUrl.trim())
}

/** Everything still missing before a clip job can start. Empty means the choices are complete. */
export function setupGaps(input: ProviderSetupInput): SetupGap[] {
  const gaps: SetupGap[] = []
  if (input.aiProvider === 'openrouter' && !input.openrouterConfigured) {
    gaps.push({
      id: 'openrouter-key',
      label: 'OpenRouter key',
      message: 'Add your OpenRouter API key.',
      // Unchanged for people who only use OpenRouter.
      blocked: 'Add your OpenRouter key above to start.'
    })
  } else if (input.aiProvider === 'gemini' && !input.geminiConfigured) {
    gaps.push({ id: 'gemini-key', label: 'Gemini key', message: 'Add your Google Gemini API key.' })
  } else if (input.aiProvider === 'openai-compatible') {
    if (!input.openaiCompatibleUrl.trim()) gaps.push({ id: 'openai-url', label: 'endpoint URL', message: 'Enter the endpoint URL of your OpenAI-compatible server.' })
    if (!input.openaiCompatibleModel.trim()) gaps.push({ id: 'openai-model', label: 'model name', message: 'Enter the model name your OpenAI-compatible server should use.' })
  }
  if (input.transcriptionSource === 'cloud') {
    // OpenRouter is already reported when it is also the AI provider.
    if (input.aiProvider !== 'openrouter' && !input.openrouterConfigured) {
      gaps.push({
        id: 'openrouter-key',
        label: 'OpenRouter key',
        message: 'Cloud transcription uses OpenRouter. Add an OpenRouter key, or switch transcription to Local Whisper.'
      })
    }
  } else if (input.localWhisperMode === 'server' && !input.localWhisperUrl.trim()) {
    gaps.push({ id: 'whisper-url', label: 'Whisper server URL', message: 'Enter your Whisper server URL, or use the built-in Whisper.' })
  }
  return gaps
}

/**
 * Why titles and captions cannot be written with the saved choices, or null. Writing needs the
 * chosen AI provider, and a way to transcribe the clip: OpenRouter's cloud, or Local Whisper.
 */
export function metadataWriterGap(input: ProviderSetupInput): string | null {
  if (!aiReady(input)) {
    return `Titles and captions are written by your AI provider, ${AI_PROVIDER_LABELS[input.aiProvider]}. Finish setting it up in Settings → AI & transcription.`
  }
  if (input.transcriptionSource === 'cloud' && !input.openrouterConfigured) {
    return 'Clips are transcribed with OpenRouter’s cloud for titles and captions. Add an OpenRouter key, or switch Speech to text to Local Whisper in Settings.'
  }
  if (input.transcriptionSource === 'local' && input.localWhisperMode === 'server' && !input.localWhisperUrl.trim()) {
    return 'Enter your Whisper server URL in Settings, or use the built-in Whisper, to transcribe clips for titles and captions.'
  }
  return null
}

/** Short label for the sidebar and the Create page prompt. */
export function describeGaps(gaps: readonly SetupGap[]): string {
  if (gaps.length === 0) return ''
  if (gaps.length > 1) return 'Finish setup'
  return `Add ${gaps[0].label}`
}
