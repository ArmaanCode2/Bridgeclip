import {
  AI_PROVIDER_LABELS, DEFAULT_GEMINI_MODEL, GEMINI_OPENAI_BASE_URL, checkEndpointUrl, isLocalEndpoint, type AiProvider
} from '../shared/providers'

/**
 * Chat requests to Gemini and OpenAI-compatible endpoints from the main process, shaped the way
 * the engine's ai_client.py sends them: the JSON schema written into the last user message,
 * strict schema, then JSON mode, then plain text when a server refuses, temperature 0.2 unless
 * the model refuses one, and the output allowance lowered or renamed when the server says so.
 * Replies are read the way the engine reads them. Test connection and metadata writing use this;
 * OpenRouter keeps its own request paths.
 */

type FetchLike = typeof fetch

export const MAX_REPLY_BYTES = 1024 * 1024
/** The engine's planner_max_output_tokens: what a clip job asks for. */
export const PLANNER_MAX_TOKENS = 32_000
/** The engine's default temperature for these providers (ai_client.prepare_payload). */
export const PLANNER_TEMPERATURE = 0.2
/** Models the engine sends no temperature to (ai_client._NO_TEMPERATURE_MODEL). */
export const NO_TEMPERATURE_MODEL = /(^|[/:._-])(o\d|gpt-5)|(^|\/)(models\/)?gemini-3/i
/** Wording of a 400 that refuses the temperature value (ai_client._TEMPERATURE_REFUSED). */
const TEMPERATURE_REFUSED = ['unsupported', 'not supported', 'does not support', 'only the default', 'only default', 'not allowed', 'may only']
/** An output limit stated in a 400 (ai_client._LIMIT_NUMBER). */
const LIMIT_NUMBER = /(?:at most|maximum(?: allowed| value)?(?: is| of)?|limit(?: is| of)?|less than or equal to|up to|supports? (?:at most|up to))\D{0,20}(\d[\d,]{2,8})/i

class ReplyTooLarge extends Error {}

/** Reads at most `maxBytes` of a response body, or fails. */
async function readCapped(response: Response, maxBytes: number): Promise<string> {
  const reader = response.body?.getReader()
  if (!reader) return ''
  const chunks: Uint8Array[] = []
  let size = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    size += value.byteLength
    if (size > maxBytes) {
      await reader.cancel().catch(() => {})
      throw new ReplyTooLarge()
    }
    chunks.push(value)
  }
  return Buffer.concat(chunks).toString('utf-8')
}

/**
 * The JSON a model answered with, read the way the engine reads it (ai_client.tolerant_json_parse):
 * reasoning before the answer (<think>…</think>, or only the closing tag when the chat template
 * opened it), code fences, prose around the JSON and trailing commas are all handled.
 * Undefined when there is no JSON.
 */
export function extractJson(content: string): unknown {
  let text = content
  for (;;) {
    const block = /^\s*<(think|thinking|reasoning)>[\s\S]*?<\/\1>/i.exec(text)
    if (!block) break
    text = text.slice(block[0].length)
  }
  if (/^\s*<(think|thinking|reasoning)>/i.test(text)) return undefined
  const closing = /<\/(think|thinking|reasoning)>/i.exec(text)
  if (closing && !/^\s*[{[`]/.test(text)) text = text.slice(closing.index + closing[0].length)
  const fence = /```(?:json)?\s*([\s\S]*?)\s*```/i.exec(text)
  if (fence) text = fence[1]
  const attempt = (raw: string): unknown => {
    try { return JSON.parse(raw) } catch { return undefined }
  }
  const whole = attempt(text.trim())
  if (whole !== undefined) return whole
  const start = text.search(/[[{]/)
  const end = Math.max(text.lastIndexOf('}'), text.lastIndexOf(']'))
  if (start < 0 || end <= start) return undefined
  const candidate = text.slice(start, end + 1)
  return attempt(candidate) ?? attempt(candidate.replace(/,\s*([\]}])/g, '$1'))
}

/** The provider's own error text, lowercased, for sorting failures only. It is never shown. */
export function errorDetail(text: string): string {
  try {
    let body = JSON.parse(text) as unknown
    if (Array.isArray(body)) body = body[0]
    const record = (body ?? {}) as { error?: unknown; message?: unknown; detail?: unknown }
    const error = record.error
    const message = error && typeof error === 'object' ? (error as { message?: unknown }).message : error ?? record.message ?? record.detail
    return String(message ?? '').toLowerCase()
  } catch {
    return text.slice(0, 300).toLowerCase()
  }
}

/** What a 200 chat reply holds: the answer text, an error the server put in the body, or a cut-off answer. */
export function readAnswer(text: string): { content: string; error: boolean; cutOff: boolean } {
  try {
    let body = JSON.parse(text) as unknown
    // Some servers wrap the reply in a one-element list, as the engine allows.
    if (Array.isArray(body) && body.length === 1) body = body[0]
    const record = (body ?? {}) as { error?: unknown; choices?: { message?: { content?: unknown }; finish_reason?: unknown }[] }
    if (record.error) return { content: '', error: true, cutOff: false }
    const choice = record.choices?.[0]
    const raw = choice?.message?.content
    const content = typeof raw === 'string' ? raw
      : Array.isArray(raw) ? raw.map((part) => (part && typeof part === 'object' && typeof (part as { text?: unknown }).text === 'string' ? (part as { text: string }).text : '')).join('')
        : ''
    return { content, error: !choice, cutOff: choice?.finish_reason === 'length' }
  } catch {
    return { content: '', error: true, cutOff: false }
  }
}

export type Reply =
  | { kind: 'reply'; status: number; text: string }
  | { kind: 'timeout' }
  | { kind: 'redirect' }
  | { kind: 'too-large' }
  | { kind: 'refused' }
  | { kind: 'no-host' }
  | { kind: 'unreachable' }

/** Every error code in a failed fetch, including the attempts inside an AggregateError (IPv6 then IPv4). */
function errorCodes(error: unknown): string[] {
  const codes: string[] = []
  const visit = (value: unknown, depth: number): void => {
    if (!value || typeof value !== 'object' || depth > 4) return
    const record = value as { code?: unknown; cause?: unknown; errors?: unknown }
    if (typeof record.code === 'string') codes.push(record.code)
    if (Array.isArray(record.errors)) record.errors.forEach((inner) => visit(inner, depth + 1))
    visit(record.cause, depth + 1)
  }
  visit(error, 0)
  return codes
}

/** One request. Redirects are refused, so a key is never forwarded somewhere the person did not configure. */
export async function send(doFetch: FetchLike, url: string, init: RequestInit, timeoutMs: number, maxBytes = MAX_REPLY_BYTES): Promise<Reply> {
  try {
    const response = await doFetch(url, { ...init, redirect: 'error', signal: AbortSignal.timeout(timeoutMs) })
    try {
      return { kind: 'reply', status: response.status, text: await readCapped(response, maxBytes) }
    } catch (error) {
      if (error instanceof ReplyTooLarge) return { kind: 'too-large' }
      throw error
    }
  } catch (error) {
    const name = error instanceof Error ? error.name : ''
    if (name === 'TimeoutError' || name === 'AbortError') return { kind: 'timeout' }
    const cause = (error as { cause?: { message?: unknown } })?.cause
    const text = `${error instanceof Error ? error.message : ''} ${cause?.message ?? ''}`.toLowerCase()
    if (text.includes('redirect')) return { kind: 'redirect' }
    const codes = errorCodes(error)
    if (codes.includes('ECONNREFUSED')) return { kind: 'refused' }
    if (codes.includes('ENOTFOUND') || codes.includes('EAI_AGAIN')) return { kind: 'no-host' }
    return { kind: 'unreachable' }
  }
}

/** The engine's _with_schema_in_prompt instruction. */
export function schemaInstruction(schema: object): string {
  return `Reply with one JSON value and nothing else: no prose and no code fences. It must validate against this JSON Schema:\n${JSON.stringify(schema)}`
}

export interface RequestSettings {
  level: number
  tokens: { param: 'max_tokens' | 'max_completion_tokens'; limit: number | null }
  temperature: boolean
}

export function startingSettings(model: string): RequestSettings {
  return { level: 0, tokens: { param: 'max_tokens', limit: null }, temperature: !NO_TEMPERATURE_MODEL.test(model) }
}

/**
 * What a 400 asks to change, the way the engine learns it: drop a refused temperature, switch
 * to max_completion_tokens, or lower the output allowance to a stated limit. Null when it asks
 * for nothing the request can change.
 */
export function adjustFor(detail: string, settings: RequestSettings, requested: number): RequestSettings | null {
  if (settings.temperature && detail.includes('temperature') && TEMPERATURE_REFUSED.some((word) => detail.includes(word))) {
    return { ...settings, temperature: false }
  }
  if (!detail.includes('max_tokens') && !detail.includes('max_completion_tokens')) return null
  if (settings.tokens.param === 'max_tokens' && detail.includes('max_completion_tokens')
    && (detail.includes('unsupported') || detail.includes('not supported') || detail.includes('instead'))) {
    return { ...settings, tokens: { ...settings.tokens, param: 'max_completion_tokens' } }
  }
  const found = LIMIT_NUMBER.exec(detail)
  const limit = found ? Number.parseInt(found[1].replace(/,/g, ''), 10) : 0
  const current = settings.tokens.limit ?? requested
  return limit > 0 && limit < current ? { ...settings, tokens: { ...settings.tokens, limit } } : null
}

/** The output allowance a request asks for. */
export function tokenField(settings: RequestSettings, requested: number): Record<string, number> {
  return { [settings.tokens.param]: settings.tokens.limit ? Math.min(settings.tokens.limit, requested) : requested }
}

export function responseFormat(level: number, name: string, schema: object): Record<string, unknown> {
  if (level === 0) return { response_format: { type: 'json_schema', json_schema: { name, strict: true, schema } } }
  if (level === 1) return { response_format: { type: 'json_object' } }
  return {}
}

// ---- Metadata and other main-process features --------------------------------------

/** A failure with a fixed message that starts with the provider's name. */
export class ProviderError extends Error {}

/** Messages from ProviderError start with one of these. */
export const PROVIDER_MESSAGE_PREFIX = /^(OpenRouter|Google Gemini|Your AI endpoint)/

export interface ChatTarget {
  provider: Exclude<AiProvider, 'openrouter'>
  base: string
  key: string
  model: string
  /** How messages name the provider: "Google Gemini", "Your AI endpoint". */
  label: string
  local: boolean
  timeoutMs: number
}

/** The saved Gemini or OpenAI-compatible model to ask, or null when it is OpenRouter or not set up. */
export function ownModelTarget(settings: {
  aiProvider: AiProvider; geminiApiKey: string; geminiModel: string; openaiCompatibleUrl: string; openaiCompatibleApiKey: string
  openaiCompatibleModel: string; openaiCompatibleTimeout: string
}, maxTimeoutMs = 600_000): ChatTarget | null {
  if (settings.aiProvider === 'gemini') {
    if (!settings.geminiApiKey) return null
    return {
      provider: 'gemini', base: GEMINI_OPENAI_BASE_URL, key: settings.geminiApiKey, model: (settings.geminiModel || DEFAULT_GEMINI_MODEL).trim(),
      label: AI_PROVIDER_LABELS.gemini, local: false, timeoutMs: Math.min(120_000, maxTimeoutMs)
    }
  }
  if (settings.aiProvider !== 'openai-compatible') return null
  const checked = checkEndpointUrl(settings.openaiCompatibleUrl)
  const model = settings.openaiCompatibleModel.trim()
  if (!checked.ok || !checked.url || !model) return null
  const waitMs = Number(settings.openaiCompatibleTimeout) * 60_000
  return {
    provider: 'openai-compatible', base: checked.url, key: settings.openaiCompatibleApiKey, model, label: 'Your AI endpoint',
    local: isLocalEndpoint(checked.url), timeoutMs: Number.isFinite(waitMs) && waitMs > 0 ? Math.min(waitMs, maxTimeoutMs) : maxTimeoutMs
  }
}

export interface ChatMessage { role: 'system' | 'user' | 'assistant'; content: string }

/**
 * Asks the target model for JSON matching `schema`. Resolves with what the reply parses to
 * (undefined when it holds no JSON); a refused or failed request throws a ProviderError.
 */
export async function chatJson(target: ChatTarget, request: {
  messages: ChatMessage[]; schema: object; schemaName: string; maxTokens: number; maxBytes?: number
}, deps: { fetch?: FetchLike } = {}): Promise<unknown> {
  const doFetch = deps.fetch ?? fetch
  const instruction = schemaInstruction(request.schema)
  const lastUser = request.messages.map((message) => message.role).lastIndexOf('user')
  const messages = request.messages.map((message, index) => (index === lastUser ? { ...message, content: `${message.content}\n\n${instruction}` } : message))
  const headers: Record<string, string> = {
    Accept: 'application/json', 'Accept-Encoding': 'identity', 'Content-Type': 'application/json',
    ...(target.key ? { Authorization: `Bearer ${target.key}` } : {})
  }
  let settings = startingSettings(target.model)
  let reply: Reply = { kind: 'unreachable' }
  for (let attempt = 0; attempt < 8; attempt++) {
    reply = await send(doFetch, `${target.base}/chat/completions`, {
      method: 'POST', headers,
      body: JSON.stringify({
        model: target.model, messages, ...tokenField(settings, request.maxTokens),
        ...(settings.temperature ? { temperature: PLANNER_TEMPERATURE } : {}),
        ...responseFormat(settings.level, request.schemaName, request.schema)
      })
    }, target.timeoutMs, request.maxBytes ?? MAX_REPLY_BYTES)
    if (reply.kind !== 'reply' || (reply.status !== 400 && reply.status !== 422)) break
    const adjusted = adjustFor(errorDetail(reply.text), settings, request.maxTokens)
    if (adjusted) { settings = adjusted; continue }
    if (settings.level === 2) break
    settings = { ...settings, level: settings.level + 1 }
  }
  const { label } = target
  if (reply.kind === 'timeout') throw new ProviderError(`${label} took too long to answer. Raise Wait for a reply in Settings, or use a faster model.`)
  if (reply.kind === 'refused') throw new ProviderError(`${label} is not answering. Start the server, or check the address in Settings.`)
  if (reply.kind === 'too-large') throw new ProviderError(`${label} returned too much text. Try again.`)
  if (reply.kind !== 'reply') throw new ProviderError(`${label} could not be reached. ${target.local ? 'Check that the server is running.' : 'Check your internet connection.'}`)
  const status = reply.status
  if (status === 401 || status === 403) throw new ProviderError(`${label} rejected the API key. Check it in Settings.`)
  if (status === 402) throw new ProviderError(`${label} reports insufficient credit.`)
  if (status === 404) throw new ProviderError(`${label} could not find the model “${target.model}”. Check the model name in Settings.`)
  if (status === 429) throw new ProviderError(`${label} is rate limiting requests. Try again shortly.`)
  if (status === 400 || status === 422) throw new ProviderError(`${label} rejected the request. Try again, or use a model that follows JSON instructions.`)
  if (status !== 200) throw new ProviderError(`${label} failed (${status}). Try again later.`)
  const answer = readAnswer(reply.text)
  if (answer.error) throw new ProviderError(`${label} returned an error instead of an answer. Try again.`)
  return extractJson(answer.content)
}
