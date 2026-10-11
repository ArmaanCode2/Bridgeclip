import { execFile as nodeExecFile } from 'child_process'
import { lstat, readdir, realpath, rm } from 'fs/promises'
import { homedir, totalmem } from 'os'
import { dirname, isAbsolute, join } from 'path'
import {
  WHISPER_APPROX_MEMORY_MB, WHISPER_MESSAGES, WHISPER_MODEL_NAMES, isWhisperBackend, isWhisperModelName,
  type LocalWhisperStatus, type WhisperBackend, type WhisperDeleteResult, type WhisperListing, type WhisperModelEntry,
  type WhisperModelName, type WhisperSystemInfo
} from '../shared/whisper-models'

/**
 * The Local Whisper model manager's main-process half: which downloaded models exist, and
 * deleting one.
 *
 * Two caches hold Whisper models, and neither is BridgeClip's:
 *  - openai-whisper: <cache>/whisper/<model>.pt
 *  - faster-whisper (Hugging Face hub): <hub>/models--<org>--faster-whisper-<model>/
 *
 * The page only ever names a model from a fixed list and a backend from a fixed pair. Paths
 * are built here from the cache roots and nothing the page sends reaches a path. Reading and
 * deleting use lstat and never follow a symlink: a symlinked model, or a symlinked `whisper`
 * folder, is skipped when listing and refused when deleting, so a link cannot make BridgeClip
 * remove something outside the cache. The two cache roots themselves are resolved once with
 * realpath, because people legitimately move a whole cache to another drive.
 */

export interface StatLike {
  size: number
  isFile(): boolean
  isDirectory(): boolean
  isSymbolicLink(): boolean
}

export interface WhisperFs {
  lstat(path: string): Promise<StatLike>
  readdir(path: string): Promise<string[]>
  realpath(path: string): Promise<string>
  rm(path: string, options: { recursive: boolean; force: boolean }): Promise<void>
}

export const nodeWhisperFs: WhisperFs = {
  lstat: (path) => lstat(path),
  readdir: (path) => readdir(path),
  realpath: (path) => realpath(path),
  rm: (path, options) => rm(path, options)
}

export interface CacheRoots {
  /** Parent of the `whisper` folder: XDG_CACHE_HOME, else ~/.cache. */
  cacheBase: string
  /** The Hugging Face hub cache. */
  hub: string
}

type Env = Record<string, string | undefined>

function absoluteEnv(env: Env, key: string): string | null {
  const value = env[key]?.trim()
  return value && isAbsolute(value) ? value : null
}

/**
 * Where Whisper keeps its models for this environment. Relative values are ignored, as they
 * would depend on the working directory. The same variables must reach the engine
 * (runtimeEnvironment), or it would download somewhere this scan does not look.
 */
export function resolveCacheRoots(env: Env, home: string): CacheRoots {
  const cacheBase = absoluteEnv(env, 'XDG_CACHE_HOME') ?? join(home, '.cache')
  const hfHome = absoluteEnv(env, 'HF_HOME')
  const hub = absoluteEnv(env, 'HF_HUB_CACHE') ?? absoluteEnv(env, 'HUGGINGFACE_HUB_CACHE')
    ?? (hfHome ? join(hfHome, 'hub') : join(cacheBase, 'huggingface', 'hub'))
  return { cacheBase, hub }
}

/** openai-whisper's file for each model. Turbo is stored under its full name. */
export function openaiWhisperFile(name: WhisperModelName): string {
  return name === 'turbo' ? 'large-v3-turbo.pt' : `${name}.pt`
}

const HUB_FOLDER = /^models--([A-Za-z0-9_.-]+)--(faster-whisper-[A-Za-z0-9_.-]+)$/

/** Whether a Hugging Face cache folder holds this faster-whisper model. */
export function hubFolderHoldsModel(folder: string, name: WhisperModelName): boolean {
  const match = HUB_FOLDER.exec(folder)
  if (!match) return false
  const [, org, repo] = match
  // faster-whisper loads turbo from this one repo (faster_whisper/utils.py); other accounts'
  // conversions would sit unused while a job downloads this one.
  if (name === 'turbo') return org === 'mobiuslabsgmbh' && repo === 'faster-whisper-large-v3-turbo'
  return org === 'Systran' && repo === `faster-whisper-${name}`
}

const MAX_SCANNED_ENTRIES = 5000

async function exists(fs: WhisperFs, path: string): Promise<StatLike | null> {
  try {
    return await fs.lstat(path)
  } catch {
    return null
  }
}

async function safeRealpath(fs: WhisperFs, path: string): Promise<string | null> {
  try {
    return await fs.realpath(path)
  } catch {
    return null
  }
}

/** A faster-whisper download is usable once a snapshot has its model.bin (a file, or a link into blobs/). */
async function hubFolderComplete(fs: WhisperFs, folder: string): Promise<boolean> {
  let revisions: string[]
  try {
    revisions = await fs.readdir(join(folder, 'snapshots'))
  } catch {
    return false
  }
  for (const revision of revisions.slice(0, 50)) {
    const snapshot = await exists(fs, join(folder, 'snapshots', revision))
    if (!snapshot?.isDirectory()) continue
    if (await exists(fs, join(folder, 'snapshots', revision, 'model.bin'))) return true
  }
  return false
}

/**
 * Bytes a Hugging Face model folder occupies. Snapshots are links into blobs/, so blobs/ is
 * counted. Where links are unavailable (Windows without developer mode) the snapshot holds
 * real files instead, which are counted. Links are never followed or counted.
 */
async function hubFolderSize(fs: WhisperFs, folder: string): Promise<number> {
  const sumFiles = async (directory: string): Promise<number> => {
    let names: string[]
    try {
      names = await fs.readdir(directory)
    } catch {
      return 0
    }
    let total = 0
    for (const entry of names.slice(0, MAX_SCANNED_ENTRIES)) {
      const info = await exists(fs, join(directory, entry))
      if (info?.isFile()) total += info.size
    }
    return total
  }
  const blobs = await sumFiles(join(folder, 'blobs'))
  if (blobs > 0) return blobs
  let total = 0
  let revisions: string[] = []
  try {
    revisions = await fs.readdir(join(folder, 'snapshots'))
  } catch {
    return 0
  }
  for (const revision of revisions.slice(0, 50)) total += await sumFiles(join(folder, 'snapshots', revision))
  return total
}

interface Found {
  backend: WhisperBackend
  bytes: number
  /** Absolute paths that belong to this model, for deleting. */
  paths: { path: string; directory: boolean }[]
}

/** The openai-whisper files on disk. A `whisper` folder that is a symlink is not read. */
async function scanOpenai(roots: CacheRoots, fs: WhisperFs): Promise<Map<WhisperModelName, Found>> {
  const found = new Map<WhisperModelName, Found>()
  const base = await safeRealpath(fs, roots.cacheBase)
  if (!base) return found
  const directory = join(base, 'whisper')
  const info = await exists(fs, directory)
  if (!info || info.isSymbolicLink() || !info.isDirectory()) return found
  let names: string[]
  try {
    names = await fs.readdir(directory)
  } catch {
    return found
  }
  for (const name of WHISPER_MODEL_NAMES) {
    const file = openaiWhisperFile(name)
    if (!names.includes(file)) continue
    const stat = await exists(fs, join(directory, file))
    if (stat?.isFile()) found.set(name, { backend: 'openai-whisper', bytes: stat.size, paths: [{ path: join(directory, file), directory: false }] })
  }
  return found
}

/** The faster-whisper folders in the Hugging Face cache that have a usable model. */
async function scanHub(roots: CacheRoots, fs: WhisperFs): Promise<Map<WhisperModelName, Found>> {
  const found = new Map<WhisperModelName, Found>()
  const hub = await safeRealpath(fs, roots.hub)
  if (!hub) return found
  let entries: string[]
  try {
    entries = (await fs.readdir(hub)).filter((entry) => HUB_FOLDER.test(entry)).slice(0, MAX_SCANNED_ENTRIES)
  } catch {
    return found
  }
  for (const entry of entries) {
    const path = join(hub, entry)
    const stat = await exists(fs, path)
    if (!stat || stat.isSymbolicLink() || !stat.isDirectory()) continue
    if (!(await hubFolderComplete(fs, path))) continue
    const bytes = await hubFolderSize(fs, path)
    for (const name of WHISPER_MODEL_NAMES) {
      if (!hubFolderHoldsModel(entry, name)) continue
      const previous = found.get(name)
      found.set(name, {
        backend: 'faster-whisper',
        bytes: (previous?.bytes ?? 0) + bytes,
        paths: [...(previous?.paths ?? []), { path, directory: true }]
      })
    }
  }
  return found
}

/**
 * Whether a model's files are on disk for a backend, by the same scan the model list uses.
 * Null for a name the manager does not know (an older model, or a server's own model name).
 */
export async function isModelDownloaded(
  name: string, backend: WhisperBackend, env: Env, options: { home?: string; fs?: WhisperFs } = {}
): Promise<boolean | null> {
  if (!isWhisperModelName(name)) return null
  const fs = options.fs ?? nodeWhisperFs
  const roots = resolveCacheRoots(env, options.home ?? homedir())
  const found = backend === 'openai-whisper' ? await scanOpenai(roots, fs) : await scanHub(roots, fs)
  return found.has(name)
}

export interface GpuProbe {
  execFile?: typeof nodeExecFile
  env?: Env
}

/**
 * Windows variables nvidia-smi needs to start. Without ProgramFiles it exits with code 255,
 * so a trimmed-down environment (like the one the engine gets) must still carry them.
 */
const NVIDIA_SMI_WINDOWS_VARS = ['ProgramFiles', 'ProgramFiles(x86)', 'ProgramW6432', 'ProgramData', 'ALLUSERSPROFILE']

/** Where older NVIDIA drivers put nvidia-smi when it is not on PATH. */
function nvidiaSmiFallback(): string | null {
  const programFiles = process.env.ProgramFiles
  return process.platform === 'win32' && programFiles ? join(programFiles, 'NVIDIA Corporation', 'NVSMI', 'nvidia-smi.exe') : null
}

/** The largest NVIDIA GPU, from nvidia-smi, or null. Four seconds at most, no shell. */
export function detectGpu(deps: GpuProbe = {}): Promise<{ name: string; vramMB: number } | null> {
  const run = deps.execFile ?? nodeExecFile
  let env: NodeJS.ProcessEnv | undefined
  if (deps.env) {
    env = { ...(deps.env as NodeJS.ProcessEnv) }
    if (process.platform === 'win32') {
      for (const key of NVIDIA_SMI_WINDOWS_VARS) if (process.env[key] && !env[key]) env[key] = process.env[key]
    }
  }
  const attempt = (file: string): Promise<{ name: string; vramMB: number } | null> => new Promise((resolve) => {
    try {
      run(
        file,
        ['--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
        { timeout: 4000, windowsHide: true, maxBuffer: 16 * 1024, ...(env ? { env } : {}) },
        (error, stdout) => resolve(error ? null : parseNvidiaSmi(String(stdout)))
      )
    } catch {
      resolve(null)
    }
  })
  return attempt('nvidia-smi').then(async (found) => {
    const fallback = found ? null : nvidiaSmiFallback()
    return found ?? (fallback ? attempt(fallback) : null)
  })
}

export function parseNvidiaSmi(output: string): { name: string; vramMB: number } | null {
  let best: { name: string; vramMB: number } | null = null
  for (const line of output.split(/\r?\n/)) {
    const cut = line.lastIndexOf(',')
    if (cut < 0) continue
    const name = line.slice(0, cut).trim().replace(/[^\x20-\x7e]/g, '').slice(0, 80)
    const vramMB = Number.parseInt(line.slice(cut + 1).trim(), 10)
    if (!name || !Number.isFinite(vramMB) || vramMB <= 0) continue
    if (!best || vramMB > best.vramMB) best = { name, vramMB }
  }
  return best
}

export interface ListDeps {
  env: Env
  home?: string
  fs?: WhisperFs
  /** The existing local-Whisper check, so the panel and the Check button agree. */
  checkEngine: () => Promise<LocalWhisperStatus>
  gpu?: () => Promise<WhisperSystemInfo['gpu']>
  totalMemoryBytes?: () => number
  platform?: NodeJS.Platform
  packaged?: boolean
}

const ENGINE_UNKNOWN: LocalWhisperStatus = { available: false, backend: null, detail: 'Could not check Whisper.', device: null, modelCached: null }

export async function listWhisperModels(deps: ListDeps): Promise<WhisperListing> {
  const fs = deps.fs ?? nodeWhisperFs
  const roots = resolveCacheRoots(deps.env, deps.home ?? homedir())
  const [openai, hub, engine, gpu] = await Promise.all([
    scanOpenai(roots, fs),
    scanHub(roots, fs),
    deps.checkEngine().catch(() => ENGINE_UNKNOWN),
    (deps.gpu ?? (() => detectGpu({ env: deps.env })))().catch(() => null)
  ])
  const engineBackend = engine.backend === 'faster-whisper' || engine.backend === 'openai-whisper' ? engine.backend : null
  const models: WhisperModelEntry[] = WHISPER_MODEL_NAMES.map((name) => {
    const candidates = [openai.get(name), hub.get(name)].filter((found): found is Found => Boolean(found))
    // When the engine's Whisper is known, only its own files count: the other Whisper's copy of a
    // model is useless to it, and a job would download the model again. Otherwise whichever exists.
    const chosen = engineBackend
      ? candidates.find((found) => found.backend === engineBackend) ?? null
      : candidates[0] ?? null
    return {
      name,
      installed: Boolean(chosen),
      sizeOnDisk: chosen ? chosen.bytes : null,
      approxMemoryMB: { ...WHISPER_APPROX_MEMORY_MB[name] },
      backend: chosen ? chosen.backend : null
    }
  })
  return {
    models,
    engine,
    system: { ramMB: Math.round((deps.totalMemoryBytes ?? totalmem)() / (1024 * 1024)), gpu },
    // NVIDIA publishes the Windows libraries as pip packages; elsewhere they come from the system.
    gpuSetup: (deps.platform ?? process.platform) === 'win32' && !deps.packaged && Boolean(gpu) && engineBackend === 'faster-whisper'
  }
}

export interface DeleteDeps {
  env: Env
  home?: string
  fs?: WhisperFs
  /** True while any clip job or editor bake is running; the models may be in use. */
  isBusy: () => boolean
}

/** A parent directory and a child it must sit directly inside. */
function directlyInside(parent: string, child: string): boolean {
  return dirname(child) === parent
}

const failed = (message: string): WhisperDeleteResult => ({ ok: false, message })

/**
 * Deletes one downloaded model. Never throws: every failure is one of the fixed messages, so
 * no path or operating-system error text can reach the screen or a log.
 */
export async function deleteWhisperModel(request: unknown, deps: DeleteDeps): Promise<WhisperDeleteResult> {
  try {
    if (!request || typeof request !== 'object') return failed(WHISPER_MESSAGES.unknownModel)
    const { name, backend } = request as { name?: unknown; backend?: unknown }
    if (!isWhisperModelName(name) || !isWhisperBackend(backend)) return failed(WHISPER_MESSAGES.unknownModel)
    if (deps.isBusy()) return failed(WHISPER_MESSAGES.busy)

    const fs = deps.fs ?? nodeWhisperFs
    const roots = resolveCacheRoots(deps.env, deps.home ?? homedir())
    const targets = backend === 'openai-whisper'
      ? await openaiTargets(roots, name, fs)
      : await hubTargets(roots, name, fs)
    if (targets === 'missing') return failed(WHISPER_MESSAGES.notInstalled)
    if (targets === 'refused') return failed(WHISPER_MESSAGES.failed)

    // Check again right before removing: the earlier look was a moment ago.
    for (const target of targets.paths) {
      const again = await exists(fs, target.path)
      if (!again || again.isSymbolicLink() || (target.directory ? !again.isDirectory() : !again.isFile())) {
        return failed(WHISPER_MESSAGES.failed)
      }
    }
    for (const target of targets.paths) await fs.rm(target.path, { recursive: target.directory, force: false })
    return { ok: true, freedBytes: targets.bytes }
  } catch {
    return failed(WHISPER_MESSAGES.failed)
  }
}

async function openaiTargets(roots: CacheRoots, name: WhisperModelName, fs: WhisperFs): Promise<Found | 'missing' | 'refused'> {
  const base = await safeRealpath(fs, roots.cacheBase)
  if (!base) return 'missing'
  const directory = join(base, 'whisper')
  const folder = await exists(fs, directory)
  if (!folder) return 'missing'
  if (folder.isSymbolicLink() || !folder.isDirectory()) return 'refused'
  const file = join(directory, openaiWhisperFile(name))
  if (!directlyInside(directory, file)) return 'refused'
  const stat = await exists(fs, file)
  if (!stat) return 'missing'
  if (stat.isSymbolicLink() || !stat.isFile()) return 'refused'
  return { backend: 'openai-whisper', bytes: stat.size, paths: [{ path: file, directory: false }] }
}

async function hubTargets(roots: CacheRoots, name: WhisperModelName, fs: WhisperFs): Promise<Found | 'missing' | 'refused'> {
  const hub = await safeRealpath(fs, roots.hub)
  if (!hub) return 'missing'
  let entries: string[]
  try {
    entries = (await fs.readdir(hub)).filter((entry) => hubFolderHoldsModel(entry, name))
  } catch {
    return 'missing'
  }
  const found: Found = { backend: 'faster-whisper', bytes: 0, paths: [] }
  for (const entry of entries) {
    const path = join(hub, entry)
    if (!directlyInside(hub, path)) return 'refused'
    const stat = await exists(fs, path)
    if (!stat) continue
    // A linked model folder could lead anywhere. Refuse the whole request rather than skip it.
    if (stat.isSymbolicLink() || !stat.isDirectory()) return 'refused'
    if (!(await hubFolderComplete(fs, path))) continue
    found.bytes += await hubFolderSize(fs, path)
    found.paths.push({ path, directory: true })
  }
  return found.paths.length > 0 ? found : 'missing'
}
