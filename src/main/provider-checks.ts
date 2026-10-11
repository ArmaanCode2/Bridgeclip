import { execFile } from 'child_process'
import { join } from 'path'
import { deflateSync } from 'zlib'
import {
  PLANNER_MAX_TOKENS, PLANNER_TEMPERATURE, adjustFor, errorDetail, extractJson, readAnswer, responseFormat, schemaInstruction, send,
  startingSettings, tokenField, type Reply, type RequestSettings
} from './ai-chat'
import {
  AI_PROVIDERS, DEFAULT_GEMINI_MODEL, GEMINI_OPENAI_BASE_URL, checkEndpointUrl, checkLocalWhisperUrl, isLocalEndpoint, testSubject,
  type AiProvider, type ProviderCheck, type ProviderTestResult
} from '../shared/providers'
import type { AppSettings } from './settings-store'
import type { LocalWhisperStatus } from '../shared/whisper-models'
import { isModelDownloaded, type WhisperFs } from './whisper-models'

/**
 * The "Test connection" buttons in Settings. Both read the SAVED settings in the main
 * process: the renderer never sends a key, and never receives one or a provider reply.
 * Messages are fixed strings (plus the model name and address the person typed), so nothing
 * a server sends can reach the screen or a log.
 *
 * Gemini and OpenAI-compatible endpoints get a request shaped like a clip job's, as the
 * engine's ai_client.py sends it: a system message, the user prompt as a list of parts with
 * the JSON schema added, temperature 0.2 unless the model refuses one, the planner's output
 * allowance, and the same schema, JSON mode, plain-text ladder. A custom endpoint also gets a
 * two-colour image whose colours change every run, so a model that cannot see it cannot pass
 * by guessing. Each check is reported on its own line.
 */

export type { LocalWhisperStatus, ProviderCheck, ProviderTestResult }
export { extractJson }

type FetchLike = typeof fetch

export interface ProviderCheckDeps {
  fetch?: FetchLike
  /** The model list request. */
  timeoutMs?: number
  /** Each chat request. Defaults to the endpoint's wait setting, at most five minutes. */
  chatTimeoutMs?: number
  /** Picks the test image's colours. */
  random?: () => number
}

const TEST_TIMEOUT_MS = 10_000
const GEMINI_CHAT_TIMEOUT_MS = 120_000
const MAX_CHAT_TIMEOUT_MS = 300_000
const MAX_MODELS = 200

const JSON_SCHEMA = {
  type: 'object',
  properties: { ok: { type: 'boolean' }, word: { type: 'string' } },
  required: ['ok', 'word'],
  additionalProperties: false
}
const IMAGE_SCHEMA = {
  type: 'object',
  properties: { left: { type: 'string' }, right: { type: 'string' } },
  required: ['left', 'right'],
  additionalProperties: false
}

/** Colours for the image check, each with the words that count as naming it. */
const TEST_COLOURS: readonly { rgb: [number, number, number]; words: RegExp }[] = [
  { rgb: [220, 30, 30], words: /\b(red|crimson|scarlet)\b/i },
  { rgb: [30, 170, 60], words: /\b(green|lime)\b/i },
  { rgb: [30, 70, 230], words: /\b(blue|navy|azure|cobalt|royal blue)\b/i },
  { rgb: [245, 215, 20], words: /\b(yellow|gold|golden)\b/i },
  { rgb: [140, 40, 200], words: /\b(purple|violet|magenta)\b/i }
]

/** Model ids from an OpenAI-style `{data:[{id}]}` list, Gemini's `models/…` ids, or an Ollama-style `{models:[…]}`. */
export function parseModelIds(body: unknown): string[] {
  if (!body || typeof body !== 'object') return []
  const record = body as { data?: unknown; models?: unknown }
  const list = Array.isArray(record.data) ? record.data : Array.isArray(record.models) ? record.models : []
  const ids: string[] = []
  for (const item of list) {
    const raw = typeof item === 'string' ? item
      : item && typeof item === 'object' ? ((item as { id?: unknown; name?: unknown }).id ?? (item as { name?: unknown }).name) : undefined
    if (typeof raw !== 'string') continue
    const id = raw.replace(/^models\//, '').trim()
    if (id && id.length <= 200 && !ids.includes(id)) ids.push(id)
    if (ids.length >= MAX_MODELS) break
  }
  return ids
}

/**
 * Whether the model the person typed is among the ids a server listed. Servers disagree on
 * spelling: Google lists "models/gemini-x", Ollama lists "llama3.1:latest" for "llama3.1". So a
 * leading "models/" is ignored, and "x" matches "x:latest" and any "x:tag".
 */
export function modelListed(wanted: string, listed: readonly string[]): boolean {
  const name = wanted.trim().replace(/^models\//, '')
  if (!name) return true
  return listed.some((id) => {
    const bare = id.replace(/^models\//, '')
    return bare === name || bare === `${name}:latest` || bare.split(':')[0] === name
  })
}

/** "localhost:3001" for http://localhost:3001/v1, with the scheme's port when none is given. */
function hostAndPort(url: string): string {
  try {
    const parsed = new URL(url)
    return `${parsed.hostname}:${parsed.port || (parsed.protocol === 'https:' ? '443' : '80')}`
  } catch {
    return 'that address'
  }
}

// ---- The test image ---------------------------------------------------------------

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  return c >>> 0
})

function crc32(data: Buffer): number {
  let c = 0xffffffff
  for (const byte of data) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

function pngChunk(type: string, data: Buffer): Buffer {
  const head = Buffer.alloc(4)
  head.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const tail = Buffer.alloc(4)
  tail.writeUInt32BE(crc32(body))
  return Buffer.concat([head, body, tail])
}

/** A PNG data URL whose left half is one colour and right half another. */
export function twoColourPng(left: readonly number[], right: readonly number[], size = 64): string {
  const row = Buffer.alloc(1 + size * 3)
  for (let x = 0; x < size; x++) row.set(x < size / 2 ? left : right, 1 + x * 3)
  const header = Buffer.alloc(13)
  header.writeUInt32BE(size, 0)
  header.writeUInt32BE(size, 4)
  header.set([8, 2, 0, 0, 0], 8)
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk('IHDR', header),
    pngChunk('IDAT', deflateSync(Buffer.concat(Array.from({ length: size }, () => row)))),
    pngChunk('IEND', Buffer.alloc(0))
  ])
  return `data:image/png;base64,${png.toString('base64')}`
}

// ---- Requests -------------------------------------------------------------------

function chatBody(model: string, settings: RequestSettings, parts: Record<string, unknown>[], schema: object, name: string): Record<string, unknown> {
  return {
    model,
    messages: [
      { role: 'system', content: 'You are BridgeClip. This is a short connection test before clip planning.' },
      // The planner sends the prompt as a list of parts; the engine adds the schema as one more part.
      { role: 'user', content: [...parts, { type: 'text', text: schemaInstruction(schema) }] }
    ],
    ...tokenField(settings, PLANNER_MAX_TOKENS),
    ...(settings.temperature ? { temperature: PLANNER_TEMPERATURE } : {}),
    ...responseFormat(settings.level, name, schema)
  }
}

interface Target {
  base: string
  key: string
  model: string
  /** How to name the server in a message: "Google's Gemini service", "localhost:3001". */
  where: string
  local: boolean
  /** Whether the image check runs. Gemini models always read images, so it is skipped there. */
  testImages: boolean
  chatTimeoutMs: number
}

const RESULT_MESSAGES = {
  noGeminiKey: 'Add your Gemini API key first.',
  badUrl: 'Enter a valid endpoint URL first.',
  noModel: 'Enter the model name first.'
} as const

const CHECK_ORDER: ProviderCheck['id'][] = ['server', 'key', 'model', 'json', 'images']

function finish(unordered: ProviderCheck[], models: string[], readsImages: boolean | null, subject: string): ProviderTestResult {
  const serverFailed = unordered.some((check) => check.id === 'server' && check.state === 'failed')
  // "No key needed" says nothing useful when nothing answered.
  const checks = unordered.filter((check) => !(serverFailed && check.state === 'skipped'))
    .sort((a, b) => CHECK_ORDER.indexOf(a.id) - CHECK_ORDER.indexOf(b.id))
  const state = (id: ProviderCheck['id']): ProviderCheck['state'] | undefined => checks.find((check) => check.id === id)?.state
  const ok = state('server') !== 'failed' && state('key') !== 'failed' && state('model') === 'ok' && state('json') === 'ok'
  const problem = checks.find((check) => check.state === 'failed') ?? checks.find((check) => check.state === 'warning' && check.id !== 'images' && check.id !== 'server')
  const message = ok
    ? `Ready. Clip planning works with this model${readsImages ? ', and it can read images' : ''}.`
    : problem?.message ?? 'The test did not finish.'
  return { ok, message, models, checks, readsImages, subject }
}

/** Server, key, model, JSON and image checks against Gemini or an OpenAI-compatible endpoint. */
async function testChatProvider(target: Target, subject: string, deps: ProviderCheckDeps): Promise<ProviderTestResult> {
  const doFetch = deps.fetch ?? fetch
  const headers: Record<string, string> = {
    Accept: 'application/json', 'Accept-Encoding': 'identity', ...(target.key ? { Authorization: `Bearer ${target.key}` } : {})
  }
  const checks: ProviderCheck[] = []
  const put = (check: ProviderCheck): void => {
    const index = checks.findIndex((existing) => existing.id === check.id)
    if (index >= 0) checks[index] = check
    else checks.push(check)
  }
  const done = (readsImages: boolean | null = null): ProviderTestResult => finish(checks, models, readsImages, subject)
  const cannotReach = (reply: Reply): string | null => {
    if (reply.kind === 'refused') return `Nothing is answering at ${target.where}. Start the server, or check the address and port.`
    if (reply.kind === 'no-host') return target.local ? `Could not find ${target.where}. Check the address.` : `Could not reach ${target.where}. Check your internet connection.`
    return null
  }

  // 1. The model list. A clip job never asks for it, so it only stops the test when nothing is there.
  let models: string[] = []
  const list = await send(doFetch, `${target.base}/models`, { method: 'GET', headers }, deps.timeoutMs ?? TEST_TIMEOUT_MS)
  const fatal = cannotReach(list)
  if (fatal) {
    put({ id: 'server', state: 'failed', message: fatal })
    return done()
  }
  if (list.kind === 'reply' && list.status === 200) {
    try { models = parseModelIds(JSON.parse(list.text)) } catch { models = [] }
    put({ id: 'server', state: 'ok', message: models.length ? `The server answered and lists ${models.length} model${models.length === 1 ? '' : 's'}.` : 'The server answered.' })
  } else if (list.kind === 'reply' && list.status < 500) {
    // No list, or a key that may not read it (an OpenAI key without the model scope): the chat request decides.
    put({ id: 'server', state: 'ok', message: 'The server answered, without a model list.' })
  } else {
    put({ id: 'server', state: 'warning', message: list.kind === 'timeout'
      ? `The server did not list its models within ${Math.round((deps.timeoutMs ?? TEST_TIMEOUT_MS) / 1000)} seconds.`
      : 'The server’s model list did not work; trying the model directly.' })
  }
  const listed = list.kind === 'reply' && list.status === 200
  if (!target.model) {
    put({ id: 'model', state: 'failed', message: RESULT_MESSAGES.noModel })
    return done()
  }

  // 2. A request shaped like a clip job's, stepping through the same ladder.
  const chatTimeout = deps.chatTimeoutMs ?? target.chatTimeoutMs
  const ask = async (settings: RequestSettings, parts: Record<string, unknown>[], schema: object, name: string): Promise<{ reply: Reply; settings: RequestSettings }> => {
    let current = settings
    for (let attempt = 0; attempt < 8; attempt++) {
      const reply = await send(doFetch, `${target.base}/chat/completions`, {
        method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' }, body: JSON.stringify(chatBody(target.model, current, parts, schema, name))
      }, chatTimeout)
      if (reply.kind !== 'reply' || (reply.status !== 400 && reply.status !== 422)) return { reply, settings: current }
      const adjusted = adjustFor(errorDetail(reply.text), current, PLANNER_MAX_TOKENS)
      if (adjusted) { current = adjusted; continue }
      if (current.level === 2) return { reply, settings: current }
      current = { ...current, level: current.level + 1 }
    }
    return { reply: { kind: 'unreachable' }, settings: current }
  }
  const start = startingSettings(target.model)
  const { reply: answer, settings: worked } = await ask(start,
    [{ type: 'text', text: 'Set ok to true and word to "clip".' }], JSON_SCHEMA, 'connection_test')
  const failedKey = (message: string): ProviderTestResult => {
    put({ id: 'key', state: 'failed', message })
    return done()
  }
  if (answer.kind !== 'reply') {
    const unreachable = cannotReach(answer)
    if (unreachable) put({ id: 'server', state: 'failed', message: unreachable })
    else if (answer.kind === 'timeout') {
      put({ id: 'model', state: 'warning', message: `The model did not answer within ${Math.round(chatTimeout / 1000)} seconds. A local model may still be loading; try again.` })
    } else if (answer.kind === 'redirect') put({ id: 'server', state: 'failed', message: 'The server tried to redirect the request, which is not allowed.' })
    else if (answer.kind === 'too-large') put({ id: 'model', state: 'failed', message: 'The server sent an unexpectedly large reply.' })
    else put({ id: 'server', state: 'failed', message: target.local ? `Could not reach ${target.where}. Check that the server is running.` : `Could not reach ${target.where}. Check your internet connection.` })
    return done()
  }
  const status = answer.status
  const detail = errorDetail(answer.text)
  if (status === 401 || status === 403) {
    return failedKey(target.key ? 'The provider rejected the API key.' : 'The server asks for an API key. Enter it above.')
  }
  if (status === 402) return failedKey('The account is out of credit.')
  if (status === 404) {
    if (detail.includes('model') || listed) {
      put({ id: 'model', state: 'failed', message: `Model “${target.model}” was not found${models.length ? '. Pick one from the list' : ''}.` })
    } else {
      put({ id: 'server', state: 'failed', message: 'Nothing at this URL answers chat requests. Check that it ends in /v1.' })
    }
    return done()
  }
  if (status === 429) {
    put({ id: 'model', state: 'warning', message: /perday/i.test(answer.text)
      ? 'This key’s daily quota is used up. It resets tomorrow; a paid key raises it.'
      : 'The provider is limiting requests right now. Try again in a minute.' })
    return done()
  }
  if (status === 503) {
    put({ id: 'model', state: 'warning', message: 'The server is busy or still loading the model. Try again in a minute.' })
    return done()
  }
  if (status >= 500) {
    put({ id: 'model', state: 'failed', message: 'The server failed on the test request. Check the model name, or the server’s own log.' })
    return done()
  }
  if (status === 400 || status === 422) {
    // Still refused after the ladder, so JSON is not the cause.
    if (/api.?key|invalid.?key|unauthori[sz]ed|authenticat/.test(detail)) return failedKey(target.key ? 'The provider rejected the API key.' : 'The server asks for an API key. Enter it above.')
    if (/context|too long|too many tokens/.test(detail)) {
      put({ id: 'model', state: 'failed', message: 'The server’s context window is too small for clip planning. Use a model or server setting with 32,000 tokens or more.' })
    } else if (detail.includes('model')) {
      put({ id: 'model', state: 'failed', message: `The server did not accept the model “${target.model}”.` })
    } else {
      put({ id: 'model', state: 'failed', message: 'The server rejected the test request.' })
    }
    return done()
  }
  if (status !== 200) {
    put({ id: 'model', state: 'failed', message: 'The server rejected the test request.' })
    return done()
  }
  put({ id: 'key', ...(target.key ? { state: 'ok', message: 'The key was accepted.' } : { state: 'skipped', message: 'No key was sent, and the server did not ask for one.' }) })
  const reading = readAnswer(answer.text)
  if (reading.error) {
    put({ id: 'model', state: 'failed', message: 'The server answered with an error instead of a reply.' })
    return done()
  }
  put({ id: 'model', state: 'ok', message: models.length && !modelListed(target.model, models)
    ? `“${target.model}” answered, though the server does not list it.`
    : `“${target.model}” answered.` })
  const parsed = extractJson(reading.content)
  if (parsed === undefined || parsed === null || typeof parsed !== 'object') {
    put(reading.cutOff && !reading.content.trim()
      ? { id: 'json', state: 'warning', message: 'The model used its whole output allowance before answering. Raise the server’s output limit, or use a model that reasons less.' }
      : { id: 'json', state: 'failed', message: 'The model answered, but not with JSON. Clip planning needs a model that follows JSON instructions.' })
    return done()
  }
  put({ id: 'json', state: 'ok', message: worked.level === 0 ? 'Structured JSON works.' : worked.level === 1 ? 'JSON works (JSON mode).' : 'JSON works (from plain text).' })

  // 3. Two colours that change every run, at the settings that worked. Gemini models read images.
  if (!target.testImages) {
    put({ id: 'images', state: 'skipped', message: 'Gemini models read images.' })
    return done()
  }
  const random = deps.random ?? Math.random
  const first = Math.floor(random() * TEST_COLOURS.length) % TEST_COLOURS.length
  const second = (first + 1 + Math.floor(random() * (TEST_COLOURS.length - 1))) % TEST_COLOURS.length
  const [left, right] = [TEST_COLOURS[first], TEST_COLOURS[second]]
  const { reply: seen } = await ask(worked, [
    { type: 'text', text: 'The image is split into a left half and a right half of different colours. Name each colour in one lowercase word.' },
    { type: 'image_url', image_url: { url: twoColourPng(left.rgb, right.rgb) } }
  ], IMAGE_SCHEMA, 'image_test')
  if (seen.kind === 'reply' && seen.status === 200) {
    const named = extractJson(readAnswer(seen.text).content) as { left?: unknown; right?: unknown } | undefined
    const readsImages = typeof named?.left === 'string' && typeof named?.right === 'string'
      && left.words.test(named.left) && right.words.test(named.right)
    put(readsImages
      ? { id: 'images', state: 'ok', message: 'The model read the test image.' }
      : { id: 'images', state: 'failed', message: 'The model could not read the test image.' })
    return done(readsImages)
  }
  if (seen.kind === 'reply' && (seen.status === 400 || seen.status === 422)) {
    put({ id: 'images', state: 'failed', message: 'This model does not take images.' })
    return done(false)
  }
  put({ id: 'images', state: 'warning', message: 'The image test did not finish, so image support is unknown.' })
  return done()
}

/** Whether the OpenRouter key works. Its models come from the Quality, Economy or Advanced choice, so they are not tried here. */
async function testOpenRouter(key: string, deps: ProviderCheckDeps): Promise<ProviderTestResult> {
  const reply = await send(deps.fetch ?? fetch, 'https://openrouter.ai/api/v1/auth/key', {
    method: 'GET', headers: { Accept: 'application/json', Authorization: `Bearer ${key}` }
  }, deps.timeoutMs ?? TEST_TIMEOUT_MS)
  const checks: ProviderCheck[] = []
  const subject = testSubject('openrouter', { openaiCompatibleUrl: '', openaiCompatibleModel: '', geminiModel: '' })
  if (reply.kind !== 'reply') {
    checks.push({ id: 'server', state: 'failed', message: reply.kind === 'timeout' ? 'OpenRouter did not answer within 10 seconds.' : 'Could not reach OpenRouter. Check your internet connection.' })
  } else if (reply.status === 401 || reply.status === 403) {
    checks.push({ id: 'server', state: 'ok', message: 'OpenRouter answered.' }, { id: 'key', state: 'failed', message: 'The provider rejected the API key.' })
  } else if (reply.status !== 200) {
    checks.push({ id: 'server', state: 'warning', message: 'OpenRouter did not accept the request. Try again later.' })
  } else {
    checks.push({ id: 'server', state: 'ok', message: 'OpenRouter answered.' }, { id: 'key', state: 'ok', message: 'The key works.' })
    return { ok: true, message: 'Connected. The OpenRouter key works.', models: [], checks, readsImages: null, subject }
  }
  const problem = checks.find((check) => check.state !== 'ok')
  return { ok: false, message: problem?.message ?? 'The test did not finish.', models: [], checks, readsImages: null, subject }
}

/** Runs Test connection for one provider, with its saved settings. */
export async function testAiProvider(provider: unknown, settings: AppSettings, deps: ProviderCheckDeps = {}): Promise<ProviderTestResult> {
  if (typeof provider !== 'string' || !AI_PROVIDERS.includes(provider as AiProvider)) throw new Error('Unknown AI provider')
  const subject = testSubject(provider as AiProvider, settings)
  const missing = (message: string, id: ProviderCheck['id']): ProviderTestResult => (
    { ok: false, message, models: [], checks: [{ id, state: 'failed', message }], readsImages: null, subject }
  )
  if (provider === 'openrouter') {
    return settings.openrouterApiKey ? testOpenRouter(settings.openrouterApiKey, deps) : missing('Add your OpenRouter API key first.', 'key')
  }
  if (provider === 'gemini') {
    if (!settings.geminiApiKey) return missing(RESULT_MESSAGES.noGeminiKey, 'key')
    return testChatProvider({
      base: GEMINI_OPENAI_BASE_URL, key: settings.geminiApiKey, model: (settings.geminiModel || DEFAULT_GEMINI_MODEL).trim(),
      where: 'Google’s Gemini service', local: false, testImages: false, chatTimeoutMs: GEMINI_CHAT_TIMEOUT_MS
    }, subject, deps)
  }
  const checked = checkEndpointUrl(settings.openaiCompatibleUrl)
  if (!checked.ok || !checked.url) return missing(RESULT_MESSAGES.badUrl, 'server')
  const waitMs = Number(settings.openaiCompatibleTimeout) * 60_000
  return testChatProvider({
    base: checked.url, key: settings.openaiCompatibleApiKey, model: settings.openaiCompatibleModel.trim(),
    where: hostAndPort(checked.url), local: isLocalEndpoint(checked.url), testImages: true,
    chatTimeoutMs: Number.isFinite(waitMs) && waitMs > 0 ? Math.min(waitMs, MAX_CHAT_TIMEOUT_MS) : MAX_CHAT_TIMEOUT_MS
  }, subject, deps)
}

/**
 * One short script, run with the engine's Python. It loads the engine's own local_whisper.py by
 * file path (that file imports nothing from the engine) and asks it what is available, so this
 * check can never disagree with what a job will do. The path arrives as an argument, never
 * spliced into the code.
 */
const WHISPER_PROBE = String.raw`
import importlib.util, json, sys
out = {"backend": None, "cuda": False, "version": ""}
try:
    spec = importlib.util.spec_from_file_location("bridgeclip_local_whisper", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    info = module.describe_environment()
    out["backend"] = info.get("backend")
    out["cuda"] = bool(info.get("cuda"))
    if out["backend"]:
        from importlib import metadata
        try:
            out["version"] = metadata.version("faster-whisper" if out["backend"] == "faster-whisper" else "openai-whisper")
        except Exception:
            pass
except Exception:
    pass
print(json.dumps(out))
`

export interface WhisperProbeDeps {
  /** The engine's Python, from resolvePythonPath. */
  python: string
  /** The engine folder (getEnginePath): local_whisper.py is read from inside it. */
  enginePath: string
  /** Where model caches are looked up. Defaults to the user's home folder. */
  home?: string
  fs?: WhisperFs
  /** The same stripped environment engine jobs run with. */
  env: Record<string, string | undefined>
  execFile?: typeof execFile
  fetch?: FetchLike
  /** An installed app's bundled Python cannot have packages added, so the advice differs. */
  packaged?: boolean
  /** Importing torch takes several seconds on a cold start. */
  timeoutMs?: number
}

/** The engine file the probe loads. */
export function localWhisperScript(enginePath: string): string {
  return join(enginePath, 'clip_engine', 'services', 'local_whisper.py')
}

function runProbe(deps: WhisperProbeDeps): Promise<string> {
  const run = deps.execFile ?? execFile
  const encoded = Buffer.from(WHISPER_PROBE, 'utf-8').toString('base64')
  return new Promise((resolve, reject) => {
    run(
      deps.python,
      ['-c', `import base64;exec(base64.b64decode('${encoded}').decode())`, localWhisperScript(deps.enginePath)],
      { env: deps.env as NodeJS.ProcessEnv, timeout: deps.timeoutMs ?? 20_000, windowsHide: true, maxBuffer: 64 * 1024 },
      (error, stdout) => (error ? reject(error) : resolve(String(stdout)))
    )
  })
}

/** Parses the probe's one JSON line; anything unexpected counts as "not available". */
export function parseWhisperProbe(output: string): LocalWhisperStatus {
  const none: LocalWhisperStatus = { available: false, backend: null, detail: '', device: null, modelCached: null }
  const line = output.split(/\r?\n/).map((part) => part.trim()).filter(Boolean).pop()
  if (!line) return { ...none }
  try {
    const parsed = JSON.parse(line) as Record<string, unknown>
    const backend = parsed.backend === 'faster-whisper' || parsed.backend === 'openai-whisper' ? parsed.backend : null
    if (!backend) return { ...none }
    const version = typeof parsed.version === 'string' ? parsed.version.replace(/[^\w.+-]/g, '').slice(0, 30) : ''
    return {
      available: true,
      backend,
      detail: version ? `${backend} ${version}` : backend,
      device: parsed.cuda === true ? 'cuda' : 'cpu',
      modelCached: null
    }
  } catch {
    return { ...none }
  }
}

/**
 * Is the chosen local Whisper usable? Built-in asks the engine's own Python which Whisper
 * it can import; server mode asks whether anything answers at the saved URL.
 */
export async function checkLocalWhisper(settings: AppSettings, deps: WhisperProbeDeps): Promise<LocalWhisperStatus> {
  if (settings.localWhisperMode === 'server') {
    const checked = checkLocalWhisperUrl(settings.localWhisperUrl)
    // A saved public host cannot normally exist (settings refuse it), but say why if one is ever seen.
    if (!checked.ok) return { available: false, backend: null, detail: checked.error, device: null, modelCached: null }
    if (!checked.url) {
      return { available: false, backend: null, detail: 'Enter your Whisper server URL first.', device: null, modelCached: null }
    }
    try {
      const response = await (deps.fetch ?? fetch)(checked.url, {
        method: 'GET',
        redirect: 'manual',
        signal: AbortSignal.timeout(5000)
      })
      await response.body?.cancel().catch(() => {})
      // Any HTTP answer means a server is listening. Transcription itself is checked when a job runs.
      return { available: true, backend: 'server', detail: 'The server answered.', device: null, modelCached: null }
    } catch {
      return { available: false, backend: null, detail: 'Could not reach the server.', device: null, modelCached: null }
    }
  }
  try {
    const output = await runProbe(deps)
    // Python answering with no probe result at all (such as the Microsoft Store stub for
    // "python3") is a Python problem, not a missing package.
    if (!/^\s*\{/m.test(output)) throw new Error('no probe output')
    const result = parseWhisperProbe(output)
    if (result.available && result.backend && result.backend !== 'server') {
      // Whether the chosen model is already on disk comes from the same scan the model list uses.
      result.modelCached = await isModelDownloaded(settings.localWhisperModel, result.backend, deps.env, { home: deps.home, fs: deps.fs })
        .catch(() => null)
    }
    if (!result.available) {
      return {
        available: false, backend: null, device: null, modelCached: null,
        detail: deps.packaged
          ? 'This build of BridgeClip was made without built-in Whisper. Install a newer build, or run a Whisper server and choose Server.'
          : 'Whisper is not installed for BridgeClip’s Python. Install it with: python -m pip install faster-whisper'
      }
    }
    return result
  } catch {
    return { available: false, backend: null, detail: 'Could not run Python to check for Whisper. See System check.', device: null, modelCached: null }
  }
}
