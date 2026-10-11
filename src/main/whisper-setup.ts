import { spawn as nodeSpawn, type ChildProcess } from 'child_process'
import { join } from 'path'
import {
  WHISPER_MESSAGES, WHISPER_SETUP_MESSAGES, isWhisperModelName,
  type WhisperModelName, type WhisperSetupProgress, type WhisperSetupResult
} from '../shared/whisper-models'

/**
 * One-click setup for built-in Whisper: downloading a model, and installing NVIDIA's GPU
 * libraries. Both run the engine's own Python, so they install exactly where a clip job looks:
 *
 *  - Download runs engine/clip_engine/services/whisper_setup.py, which reports progress as
 *    JSON lines and writes into the same caches the model list scans.
 *  - Enable GPU runs pip for NVIDIA's cuBLAS and cuDNN wheels, then the same script loads a
 *    model on the GPU, because a visible GPU says nothing about whether its libraries load.
 *
 * One task runs at a time. Only a model name from the fixed list reaches the command line, and
 * every message shown is one of the fixed strings below, never the tool's own output.
 */

/**
 * cuBLAS and cuDNN for CUDA 12, which the CTranslate2 build behind faster-whisper needs. Pinned to
 * the versions tested with faster-whisper 1.2.1 / CTranslate2 4.8.2 on an RTX 3050 under Windows 11.
 */
export const GPU_PACKAGES = ['nvidia-cublas-cu12==12.9.2.10', 'nvidia-cudnn-cu12==9.27.0.42']

/** What whisper_setup.py reports for each failure, mirrored here so its text is never shown. */
const SCRIPT_MESSAGES: Record<string, string> = {
  unknown_model: WHISPER_MESSAGES.unknownModel,
  missing_backend: WHISPER_SETUP_MESSAGES.noEngine,
  network: 'Could not download the model. Check your internet connection and try again.',
  disk: 'There is not enough free disk space for this model.',
  checksum: 'The download was damaged. Try again.',
  failed: 'Could not download the model.',
  not_installed: 'Install the model before testing the GPU.',
  gpu_unavailable: 'No NVIDIA GPU with CUDA was found.',
  gpu_cublas: 'The GPU libraries did not load (cuBLAS). Try Enable GPU again, or use CPU.',
  gpu_cudnn: 'The GPU libraries did not load (cuDNN). Try Enable GPU again, or use CPU.',
  gpu_memory: 'The model does not fit in your video memory. Pick a smaller model or use CPU.',
  gpu_failed: 'Whisper could not run on the GPU. Use CPU for now.'
}

/** Proxy settings a download needs behind a company network. The engine's job runs do not get these. */
const PROXY_VARIABLES = ['HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy']

const MAX_STDOUT_LINE = 4096

export interface SetupDeps {
  /** The engine's Python, from resolvePythonPath. */
  python: string
  /** The engine folder (getEnginePath): whisper_setup.py is read from inside it. */
  enginePath: string
  /** The same stripped environment engine jobs run with. */
  env: Record<string, string | undefined>
  /** True while a clip job or editor export runs: the GPU install changes the engine's packages. */
  isBusy: () => boolean
  platform?: NodeJS.Platform
  /** An installed app's runtime is replaced by every update, so GPU libraries are not added to it. */
  packaged?: boolean
  spawn?: typeof nodeSpawn
  /** Defaults to this process's environment; tests pass their own. */
  processEnv?: Record<string, string | undefined>
}

export function whisperSetupScript(enginePath: string): string {
  return join(enginePath, 'clip_engine', 'services', 'whisper_setup.py')
}

interface ScriptEvent {
  event?: unknown
  done?: unknown
  total?: unknown
  reason?: unknown
  index?: unknown
  text?: unknown
}

let running: { child: ChildProcess | null; cancelled: boolean; progress: WhisperSetupProgress } | null = null

export function whisperSetupRunning(): boolean {
  return running !== null
}

/** The task running now, as last reported, so a panel opened mid-task can show it. */
export function whisperSetupState(): WhisperSetupProgress | null {
  return running ? { ...running.progress } : null
}

/** Stops the running download or GPU setup. A stopped Hugging Face download resumes next time. */
export function cancelWhisperSetup(): boolean {
  if (!running) return false
  running.cancelled = true
  running.child?.kill()
  return true
}

function setupEnvironment(deps: SetupDeps): NodeJS.ProcessEnv {
  const env: Record<string, string | undefined> = { ...deps.env, PYTHONUNBUFFERED: '1', PYTHONDONTWRITEBYTECODE: '1' }
  const source = deps.processEnv ?? process.env
  for (const key of PROXY_VARIABLES) if (source[key] && !env[key]) env[key] = source[key]
  return env as NodeJS.ProcessEnv
}

/** Runs the engine's Python and resolves with its exit code and the last JSON event it printed. */
function runPython(
  args: string[], deps: SetupDeps, onEvent?: (event: ScriptEvent) => void,
  options: { tracked?: boolean; maxLine?: number; onChild?: (child: ChildProcess) => void } = {}
): Promise<{ code: number | null; last: ScriptEvent | null }> {
  const tracked = options.tracked ?? true
  const maxLine = options.maxLine ?? MAX_STDOUT_LINE
  return new Promise((resolve) => {
    let child: ChildProcess
    try {
      child = (deps.spawn ?? nodeSpawn)(deps.python, args, { env: setupEnvironment(deps), windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] })
    } catch {
      resolve({ code: null, last: null })
      return
    }
    // Only a setup task can be stopped with Cancel; other runs (transcription) are not tracked.
    if (tracked && running) running.child = child
    options.onChild?.(child)
    let buffer = ''
    let last: ScriptEvent | null = null
    const readLine = (line: string): void => {
      const text = line.trim()
      if (!text.startsWith('{') || text.length > maxLine) return
      try {
        const parsed = JSON.parse(text) as ScriptEvent
        if (!parsed || typeof parsed !== 'object') return
        last = parsed
        onEvent?.(parsed)
      } catch {
        // pip and libraries print plain text; only the script's JSON lines matter.
      }
    }
    child.stdout?.setEncoding('utf8')
    child.stdout?.on('data', (chunk: string) => {
      buffer += chunk
      const lines = buffer.split(/\r?\n/)
      buffer = lines.pop() ?? ''
      if (buffer.length > maxLine) buffer = ''
      lines.forEach(readLine)
    })
    // Drained so a chatty tool cannot fill the pipe and stall; never shown.
    child.stderr?.on('data', () => {})
    child.on('error', () => resolve({ code: null, last }))
    child.on('close', (code) => {
      if (buffer) readLine(buffer)
      resolve({ code, last })
    })
  })
}

const fail = (message: string, cancelled = false): WhisperSetupResult => ({ ok: false, cancelled, message })

function scriptFailure(last: ScriptEvent | null, fallback: string): string {
  const reason = last?.event === 'error' && typeof last.reason === 'string' ? last.reason : ''
  return SCRIPT_MESSAGES[reason] ?? fallback
}

function finished(result: { code: number | null; last: ScriptEvent | null }): boolean {
  return result.code === 0 && result.last?.event === 'done'
}

function bytes(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0
}

async function exclusively(
  progress: WhisperSetupProgress, onProgress: (progress: WhisperSetupProgress) => void,
  work: (report: (progress: WhisperSetupProgress) => void) => Promise<WhisperSetupResult>
): Promise<WhisperSetupResult> {
  if (running) return fail(WHISPER_SETUP_MESSAGES.busy)
  const state = { child: null as ChildProcess | null, cancelled: false, progress }
  running = state
  const report = (next: WhisperSetupProgress): void => {
    state.progress = next
    onProgress(next)
  }
  let result: WhisperSetupResult = fail(WHISPER_SETUP_MESSAGES.failed)
  try {
    report(progress)
    result = await work(report)
    return result
  } catch {
    return result
  } finally {
    running = null
    onProgress({ ...state.progress, finished: true, result })
  }
}

/** Downloads one model for the Whisper the engine uses, reporting bytes as they arrive. */
export function downloadWhisperModel(
  name: unknown, deps: SetupDeps, onProgress: (progress: WhisperSetupProgress) => void
): Promise<WhisperSetupResult> {
  if (!isWhisperModelName(name)) return Promise.resolve(fail(WHISPER_MESSAGES.unknownModel))
  const model: WhisperModelName = name
  return exclusively({ task: 'download', model, step: 'download', done: 0, total: 0 }, onProgress, async (report: typeof onProgress) => {
    const result = await runPython([whisperSetupScript(deps.enginePath), 'download', model], deps, (event) => {
      if (event.event === 'progress') report({ task: 'download', model, step: 'download', done: bytes(event.done), total: bytes(event.total) })
    })
    if (running?.cancelled) return fail(WHISPER_SETUP_MESSAGES.cancelled, true)
    return finished(result) ? { ok: true } : fail(scriptFailure(result.last, SCRIPT_MESSAGES.failed))
  })
}

/**
 * Installs NVIDIA's cuBLAS and cuDNN into the engine's Python (pip skips them when already there),
 * then proves the GPU works by running `model` on it.
 */
export function enableWhisperGpu(
  name: unknown, deps: SetupDeps, onProgress: (progress: WhisperSetupProgress) => void
): Promise<WhisperSetupResult> {
  if (!isWhisperModelName(name)) return Promise.resolve(fail(SCRIPT_MESSAGES.not_installed))
  if ((deps.platform ?? process.platform) !== 'win32' || deps.packaged) return Promise.resolve(fail(WHISPER_SETUP_MESSAGES.gpuUnsupported))
  if (deps.isBusy()) return Promise.resolve(fail(WHISPER_SETUP_MESSAGES.jobRunning))
  const model: WhisperModelName = name
  return exclusively({ task: 'gpu', model, step: 'install', done: 0, total: 0 }, onProgress, async (report: typeof onProgress) => {
    const pip = await runPython(
      ['-m', 'pip', 'install', '--disable-pip-version-check', '--no-input', '--no-warn-script-location', ...GPU_PACKAGES], deps
    )
    if (running?.cancelled) return fail(WHISPER_SETUP_MESSAGES.cancelled, true)
    if (pip.code !== 0) return fail(WHISPER_SETUP_MESSAGES.pipFailed)
    report({ task: 'gpu', model, step: 'test', done: 0, total: 0 })
    const test = await runPython([whisperSetupScript(deps.enginePath), 'gpu-test', model], deps)
    if (running?.cancelled) return fail(WHISPER_SETUP_MESSAGES.cancelled, true)
    return finished(test) ? { ok: true } : fail(scriptFailure(test.last, SCRIPT_MESSAGES.gpu_failed))
  })
}

/** Titles and captions wait this long for Whisper on one clip. */
const TRANSCRIBE_TIMEOUT_MS = 30 * 60_000
const transcribing = new Set<ChildProcess>()

/** Stops Whisper runs for titles and captions, so none outlives the app. */
export function stopLocalTranscriptions(): void {
  for (const child of transcribing) child.kill()
  transcribing.clear()
}

const TRANSCRIBE_MESSAGES: Record<string, string> = {
  missing_backend: 'Local Whisper is not installed for BridgeClip’s Python. Install it from Settings → AI & transcription.',
  unknown_model: 'Choose a Whisper model in Settings → AI & transcription.',
  transcribe_model: 'The Whisper model could not be loaded. Install it in Settings → AI & transcription.',
  transcribe_not_installed: 'The Whisper model is not installed yet. Install it in Settings → AI & transcription.',
  gpu_failed: 'Local Whisper could not use the GPU. Choose Auto or CPU in Settings → AI & transcription.'
}

/**
 * Transcribes short 16 kHz mono WAV files with the engine's built-in Whisper, for writing titles
 * and captions. The model loads once for all files. Resolves with one text per file; failures
 * reject with a fixed message that starts with "Local Whisper".
 */
export async function transcribeWithLocalWhisper(
  files: string[], options: { model: string; device: string; language: string },
  deps: Pick<SetupDeps, 'python' | 'enginePath' | 'env' | 'spawn' | 'processEnv'> & { timeoutMs?: number }
): Promise<string[]> {
  const texts: string[] = new Array(files.length).fill('')
  let child: ChildProcess | null = null
  let timedOut = false
  const timer = setTimeout(() => { timedOut = true; child?.kill() }, deps.timeoutMs ?? TRANSCRIBE_TIMEOUT_MS)
  const result = await runPython(
    [whisperSetupScript(deps.enginePath), 'transcribe', options.model, options.device, options.language || 'auto', ...files],
    { ...deps, isBusy: () => false },
    (event) => {
      if (event.event === 'text' && typeof event.index === 'number' && event.index >= 0 && event.index < files.length && typeof event.text === 'string') {
        texts[event.index] = event.text
      }
    },
    // A five-minute transcript fits easily; anything longer is cut off by the caller anyway.
    { tracked: false, maxLine: 1024 * 1024, onChild: (spawned) => { child = spawned; transcribing.add(spawned) } }
  ).finally(() => {
    clearTimeout(timer)
    if (child) transcribing.delete(child)
  })
  if (timedOut) throw new Error('Local Whisper took longer than 30 minutes for this clip. Try a smaller model or the GPU.')
  if (finished(result)) return texts
  const reason = result.last?.event === 'error' && typeof result.last.reason === 'string' ? result.last.reason : ''
  throw new Error(TRANSCRIBE_MESSAGES[reason] ?? 'Local Whisper could not transcribe the clip audio. Try again.')
}
