/**
 * Local Whisper model manager: the shapes main returns, the memory table, and the pure
 * rules the Settings panel uses (does a model fit, what to say, the delete confirm step).
 * Kept free of Electron and React so main and the renderer agree and tests can run it.
 */

export type WhisperBackend = 'openai-whisper' | 'faster-whisper'
export const WHISPER_BACKENDS: readonly WhisperBackend[] = ['openai-whisper', 'faster-whisper']

export const LOCAL_WHISPER_DEVICES = ['auto', 'cpu', 'gpu'] as const
export type LocalWhisperDevice = (typeof LOCAL_WHISPER_DEVICES)[number]

/** What the built-in Whisper check found. Shared so both sides describe it the same way. */
export interface LocalWhisperStatus {
  available: boolean
  backend: 'faster-whisper' | 'openai-whisper' | 'server' | null
  detail: string
  /** 'cuda' means a CUDA device is visible. Its libraries may still be missing, which Auto handles. */
  device: 'cpu' | 'cuda' | null
  /** Built-in only: whether the chosen model is already downloaded. Null when unknown. */
  modelCached: boolean | null
}

/** The models the manager lists and can delete, smallest first. */
export const WHISPER_MODEL_NAMES = [
  'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v3', 'turbo'
] as const
export type WhisperModelName = (typeof WHISPER_MODEL_NAMES)[number]

export function isWhisperModelName(value: unknown): value is WhisperModelName {
  return typeof value === 'string' && (WHISPER_MODEL_NAMES as readonly string[]).includes(value)
}

export function isWhisperBackend(value: unknown): value is WhisperBackend {
  return typeof value === 'string' && (WHISPER_BACKENDS as readonly string[]).includes(value)
}

export interface WhisperMemory {
  cpu: number
  gpu: number
}

/**
 * Approximate memory a model needs while transcribing, in MB. These are rounded planning
 * figures, not measurements: `cpu` is system RAM with a CPU run (faster-whisper int8 or
 * openai-whisper fp32), `gpu` is video memory with a GPU run (fp16). Real use varies with the
 * backend, the audio length and other programs. The `.en` variants match their multilingual
 * counterparts.
 */
export const WHISPER_APPROX_MEMORY_MB: Record<WhisperModelName, WhisperMemory> = {
  tiny: { cpu: 400, gpu: 1000 },
  'tiny.en': { cpu: 400, gpu: 1000 },
  base: { cpu: 500, gpu: 1000 },
  'base.en': { cpu: 500, gpu: 1000 },
  small: { cpu: 1000, gpu: 2000 },
  'small.en': { cpu: 1000, gpu: 2000 },
  medium: { cpu: 2500, gpu: 5000 },
  'medium.en': { cpu: 2500, gpu: 5000 },
  'large-v3': { cpu: 4500, gpu: 10000 },
  turbo: { cpu: 3000, gpu: 6000 }
}

export interface WhisperModelEntry {
  name: WhisperModelName
  installed: boolean
  /** Bytes on disk, or null when not installed or unreadable. */
  sizeOnDisk: number | null
  approxMemoryMB: WhisperMemory
  /** Which Whisper has the files, or null when the model is not installed. */
  backend: WhisperBackend | null
}

export interface WhisperSystemInfo {
  ramMB: number
  /** The largest NVIDIA GPU found, or null when none could be detected. */
  gpu: { name: string; vramMB: number } | null
}

export interface WhisperListing {
  models: WhisperModelEntry[]
  engine: LocalWhisperStatus
  system: WhisperSystemInfo
  /** Whether Enable GPU can install NVIDIA's libraries here: Windows, an NVIDIA GPU and faster-whisper. */
  gpuSetup: boolean
}

export interface WhisperDeleteRequest {
  name: string
  backend: string
}

export type WhisperDeleteResult = { ok: true; freedBytes: number } | { ok: false; message: string }

/** Fixed messages: nothing a path, a file name or an OS error can leak into. */
export const WHISPER_MESSAGES = {
  unknownModel: 'Choose a model from the list.',
  notInstalled: 'That model is not installed.',
  busy: 'A clip job is running. Try again when it finishes.',
  failed: 'Could not delete the model.'
} as const

// ---- Panel rules -------------------------------------------------------------------

/** Share of system RAM a CPU run may take before the model is flagged as too big. */
const RAM_SHARE = 0.75

/** "400 MB", "1.5 GB", "6 GB". GB are rounded to the nearest half, in binary units like OS tools. */
export function formatMemory(mb: number): string {
  if (!Number.isFinite(mb) || mb <= 0) return '0 MB'
  if (mb < 1000) return `${Math.round(mb)} MB`
  const gb = Math.round((mb / 1024) * 2) / 2
  return `${Number.isInteger(gb) ? gb : gb.toFixed(1)} GB`
}

/** File sizes from the disk scan: "142 MB", "1.5 GB". */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 MB'
  const mb = bytes / (1024 * 1024)
  if (mb < 1000) return `${Math.max(1, Math.round(mb))} MB`
  const gb = mb / 1024
  return `${gb >= 10 ? Math.round(gb) : gb.toFixed(1)} GB`
}

export interface FitInput {
  memory: WhisperMemory
  device: LocalWhisperDevice
  system: WhisperSystemInfo | null
}

/**
 * Plain words when a model will not fit, or null. A GPU run is judged against video memory
 * (Auto does that whenever a GPU is present), a CPU run against system RAM.
 */
export function fitWarning({ memory, device, system }: FitInput): string | null {
  if (!system) return null
  const gpu = system.gpu
  if ((device === 'gpu' || device === 'auto') && gpu && gpu.vramMB > 0 && memory.gpu > gpu.vramMB) {
    return `Needs about ${formatMemory(memory.gpu)} of video memory; you have ${formatMemory(gpu.vramMB)}. Use CPU.`
  }
  if ((device === 'cpu' || (device === 'auto' && !gpu)) && system.ramMB > 0 && memory.cpu > system.ramMB * RAM_SHARE) {
    return `Needs about ${formatMemory(memory.cpu)} of memory; you have ${formatMemory(system.ramMB)}. Pick a smaller model.`
  }
  return null
}

/** The line under the Device control, or null when there is nothing to add. */
export function deviceNote(device: LocalWhisperDevice, system: WhisperSystemInfo | null, engine: LocalWhisperStatus | null): string | null {
  if (device !== 'gpu') return null
  const cudaSeen = engine?.device === 'cuda' || Boolean(system?.gpu)
  return cudaSeen ? null : 'No CUDA GPU was detected on this computer. Transcription will stop with an error. Choose Auto or CPU.'
}

export const DEVICE_EXPLANATIONS: Record<LocalWhisperDevice, string> = {
  auto: 'Auto uses the GPU when it works and falls back to the CPU.',
  cpu: 'CPU uses system RAM and is slower.',
  gpu: 'GPU is faster but needs enough video memory and the CUDA libraries.'
}

/** Short speed and accuracy notes for the .en variants. The others come from LOCAL_WHISPER_MODELS. */
export function englishOnlyHint(name: string, baseHint: string): string {
  return name.endsWith('.en') ? `English only, a little more accurate for English. ${baseHint}` : baseHint
}

/** "About 1.5 GB to download". */
export function downloadNote(downloadMb: number): string {
  return `About ${formatMemory(downloadMb)} to download`
}

// ---- One-click setup: model downloads and GPU libraries ---------------------------

/** The spoken-language choices for local Whisper. "auto" detects it. */
export const LOCAL_WHISPER_LANGUAGES = [
  { code: 'auto', label: 'Detect automatically' },
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'Hindi' },
  { code: 'es', label: 'Spanish' },
  { code: 'pt', label: 'Portuguese' },
  { code: 'fr', label: 'French' },
  { code: 'de', label: 'German' },
  { code: 'it', label: 'Italian' },
  { code: 'nl', label: 'Dutch' },
  { code: 'pl', label: 'Polish' },
  { code: 'tr', label: 'Turkish' },
  { code: 'ru', label: 'Russian' },
  { code: 'ja', label: 'Japanese' },
  { code: 'ko', label: 'Korean' },
  { code: 'zh', label: 'Chinese' },
  { code: 'ar', label: 'Arabic' }
] as const
export type LocalWhisperLanguage = (typeof LOCAL_WHISPER_LANGUAGES)[number]['code']
export const LOCAL_WHISPER_LANGUAGE_CODES: readonly LocalWhisperLanguage[] = LOCAL_WHISPER_LANGUAGES.map((language) => language.code)

/** One-click setup work: downloading a model, or installing and testing the GPU libraries. */
export type WhisperSetupTask = 'download' | 'gpu'

/** Progress main sends while a setup task runs. */
export interface WhisperSetupProgress {
  task: WhisperSetupTask
  model: WhisperModelName
  /** gpu only: installing the NVIDIA libraries, or testing the model on the GPU. */
  step: 'download' | 'install' | 'test'
  /** download only: bytes so far and the expected total (0 when unknown). */
  done: number
  total: number
  /** Sent once when the task ends, so a panel opened mid-task knows to stop showing it. */
  finished?: boolean
  /** With `finished`: how the task ended. */
  result?: WhisperSetupResult
}

export type WhisperSetupResult =
  | { ok: true }
  | { ok: false; cancelled: boolean; message: string }

/** Approximate disk space Enable GPU uses: NVIDIA's cuBLAS and cuDNN for CUDA 12, measured on Windows. */
export const GPU_LIBRARIES_DISK_MB = 2048

/** Fixed messages: nothing a path, a file name or a tool's output can leak into. */
export const WHISPER_SETUP_MESSAGES = {
  busy: 'Another download or GPU setup is running. Wait for it to finish.',
  jobRunning: 'A clip job is running. Try again when it finishes.',
  noEngine: 'Whisper is not installed for BridgeClip’s Python, so models cannot be downloaded yet.',
  gpuUnsupported: 'GPU setup is available on Windows with an NVIDIA GPU, when BridgeClip runs from source.',
  cancelled: 'Stopped.',
  pipFailed: 'Could not install the GPU libraries. Check your internet connection and that about 3 GB of disk space is free (download plus install).',
  failed: 'Something went wrong. Try again.',
  gpuRunning: 'Enable GPU is changing Whisper’s GPU libraries. Start the job when it finishes.'
} as const

/** What the panel says when a download or GPU setup ends. */
export function setupResultMessage(progress: Pick<WhisperSetupProgress, 'task' | 'model'>, result: WhisperSetupResult): { ok: boolean; text: string } {
  if (result.ok) {
    return progress.task === 'download'
      ? { ok: true, text: `${progress.model} is installed.` }
      : { ok: true, text: `The GPU works with ${progress.model}. Whisper will use it in Auto and GPU modes.` }
  }
  if (result.cancelled) return { ok: true, text: progress.task === 'download' ? 'Download stopped.' : 'GPU setup stopped.' }
  return { ok: false, text: result.message }
}

/** "Downloading… 120 MB of 480 MB (25%)", or without a total when it is unknown. */
export function downloadProgressText(done: number, total: number): string {
  if (total > 0) {
    const percent = Math.min(100, Math.floor((done / total) * 100))
    return `Downloading… ${formatBytes(done)} of ${formatBytes(total)} (${percent}%)`
  }
  return done > 0 ? `Downloading… ${formatBytes(done)}` : 'Starting download…'
}

/** The model a GPU test loads: the selected one when installed, else the smallest installed one. */
export function gpuTestModel(models: readonly WhisperModelEntry[], selected: string): WhisperModelName | null {
  const installed = models.filter((model) => model.installed)
  return installed.find((model) => model.name === selected)?.name ?? installed[0]?.name ?? null
}

/** The delete button's two-step state: nothing pending, or one model awaiting confirmation. */
export interface DeleteConfirmState {
  pending: string | null
  busy: boolean
}

export type DeleteConfirmAction =
  | { type: 'ask'; name: string }
  | { type: 'cancel' }
  | { type: 'start' }
  | { type: 'finish' }

export const INITIAL_DELETE_CONFIRM: DeleteConfirmState = { pending: null, busy: false }

export function deleteConfirmReducer(state: DeleteConfirmState, action: DeleteConfirmAction): DeleteConfirmState {
  switch (action.type) {
    case 'ask': return state.busy ? state : { pending: action.name, busy: false }
    case 'cancel': return state.busy ? state : INITIAL_DELETE_CONFIRM
    case 'start': return state.pending ? { ...state, busy: true } : state
    case 'finish': return INITIAL_DELETE_CONFIRM
  }
}

/** Why Delete is unavailable right now, or null. */
export function deleteBlockedReason(opts: { jobRunning: boolean; busy: boolean }): string | null {
  if (opts.jobRunning) return 'A clip job is running. Try again when it finishes.'
  if (opts.busy) return 'Deleting…'
  return null
}
