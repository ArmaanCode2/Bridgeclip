// The assistant on an OpenRouter model. There is no CLI to drive: BridgeClip
// sends the chat to OpenRouter's chat completions API with the user's key,
// streams the reply, runs the tool calls the model asks for (through the same
// path as the CLIs' MCP calls, approvals included) and sends the results back
// until the model answers without calling a tool.

export const OPENROUTER_CHAT_URL = 'https://openrouter.ai/api/v1/chat/completions'
/** Model round trips per user message; each tool batch is one. */
const MAX_STEPS = 24
const MAX_TOOL_CALLS_PER_STEP = 16
const MAX_ARGUMENT_CHARS = 100_000
const MAX_REPLY_CHARS = 200_000
const ERROR_BODY_BYTES = 16 * 1024

export interface OpenRouterTool {
  name: string
  description: string
  parameters: Record<string, unknown>
}

export type OpenRouterMessage =
  | { role: 'system' | 'user'; content: string }
  | { role: 'assistant'; content: string | null; tool_calls?: { id: string; type: 'function'; function: { name: string; arguments: string } }[] }
  | { role: 'tool'; tool_call_id: string; content: string }

export interface OpenRouterTurnOptions {
  apiKey: string
  model: string
  systemPrompt: string
  /** Earlier turns of the chat, oldest first. */
  history: OpenRouterMessage[]
  prompt: string
  tools: OpenRouterTool[]
  signal: AbortSignal
  url?: string
  fetch?: typeof fetch
  /** Streamed reply text; `newParagraph` is set on the first text after a tool step. */
  onText: (text: string, newParagraph: boolean) => void
  /** Anything arrived from OpenRouter (keeps the stall watchdog quiet). */
  onActivity: () => void
  callTool: (name: string, args: Record<string, unknown>) => Promise<{ text: string; isError?: boolean }>
}

/** A failure to show the user as is. */
export class OpenRouterError extends Error {}

interface PendingCall {
  id: string
  name: string
  arguments: string
}

/** One line of plain text: control characters become spaces (as the model catalog does for names). */
function clean(text: string, max = 300): string {
  const printable = text.split('').map((char) => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127 ? ' ' : char).join('')
  return printable.replace(/\s+/g, ' ').trim().slice(0, max)
}

function providerMessage(value: unknown): string | null {
  const error = value && typeof value === 'object' ? (value as { error?: unknown }).error : null
  const message = error && typeof error === 'object' ? (error as { message?: unknown }).message : null
  return typeof message === 'string' && message.trim() ? clean(message) : null
}

/** What went wrong, in words the user can act on. OpenRouter's own message is kept when it explains more. */
export function describeOpenRouterFailure(status: number | null, message: string | null, model: string): string {
  if (status === 401 || status === 403) return 'OpenRouter rejected your API key. Check it in Settings → API keys.'
  if (status === 402) return 'Your OpenRouter account is out of credits. Add credits on openrouter.ai, then try again.'
  if (status === 429) return 'OpenRouter is rate limiting requests. Wait a moment, then try again.'
  if (message && /tool/i.test(message) && /support|endpoint/i.test(message)) {
    return `${model} can’t use tools through OpenRouter right now, so it can’t run BridgeClip. Choose another model.`
  }
  if (status === null) return message ? `OpenRouter stopped the reply: ${message}` : 'OpenRouter stopped the reply. Try again.'
  if (status === 404) return message ? `OpenRouter couldn’t use ${model}: ${message}` : `OpenRouter doesn’t list ${model} anymore. Choose another model.`
  return message ? `OpenRouter returned an error (${status}): ${message}` : `OpenRouter returned an error (${status}). Try again.`
}

async function errorBody(response: Response): Promise<string | null> {
  try {
    const reader = response.body?.getReader()
    if (!reader) return null
    let text = ''
    const decoder = new TextDecoder()
    while (text.length < ERROR_BODY_BYTES) {
      const { done, value } = await reader.read()
      if (done) break
      text += decoder.decode(value, { stream: true })
    }
    await reader.cancel().catch(() => {})
    return providerMessage(JSON.parse(text))
  } catch {
    return null
  }
}

/** One streamed completion: its text and the tool calls it asked for. */
async function complete(options: OpenRouterTurnOptions, messages: OpenRouterMessage[], newParagraph: boolean): Promise<{ text: string; calls: PendingCall[] }> {
  let response: Response
  try {
    response = await (options.fetch ?? fetch)(options.url ?? OPENROUTER_CHAT_URL, {
      method: 'POST',
      redirect: 'error',
      signal: options.signal,
      headers: {
        Authorization: `Bearer ${options.apiKey}`,
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        'HTTP-Referer': 'https://github.com/bridge-mind/bridgeclip',
        'X-Title': 'BridgeClip'
      },
      body: JSON.stringify({
        model: options.model,
        messages,
        tools: options.tools.map((tool) => ({ type: 'function', function: tool })),
        tool_choice: 'auto',
        stream: true,
        // Only route to providers that support tool calling for this model.
        provider: { require_parameters: true }
      })
    })
  } catch (error) {
    if (options.signal.aborted) throw error
    throw new OpenRouterError('OpenRouter couldn’t be reached. Check your connection, then try again.')
  }
  options.onActivity()
  if (!response.ok) throw new OpenRouterError(describeOpenRouterFailure(response.status, await errorBody(response), options.model))
  if (!response.body) throw new OpenRouterError('OpenRouter returned an empty reply. Try again.')

  let text = ''
  const calls: PendingCall[] = []
  let first = true
  const handle = (payload: string): boolean => {
    if (payload === '[DONE]') return true
    let chunk: Record<string, unknown>
    try { chunk = JSON.parse(payload) as Record<string, unknown> } catch { return false }
    const failure = providerMessage(chunk)
    if (failure) throw new OpenRouterError(describeOpenRouterFailure(null, failure, options.model))
    const choice = Array.isArray(chunk.choices) ? chunk.choices[0] as Record<string, unknown> | undefined : undefined
    const delta = choice?.delta as Record<string, unknown> | undefined
    if (!delta) return false
    if (typeof delta.content === 'string' && delta.content) {
      if (text.length + delta.content.length > MAX_REPLY_CHARS) throw new OpenRouterError('The reply was too long and was stopped.')
      text += delta.content
      options.onText(delta.content, newParagraph && first)
      first = false
    }
    if (Array.isArray(delta.tool_calls)) {
      for (const raw of delta.tool_calls as Record<string, unknown>[]) {
        const index = typeof raw.index === 'number' ? raw.index : calls.length
        if (index < 0 || index >= MAX_TOOL_CALLS_PER_STEP) throw new OpenRouterError('The model asked for too many actions at once.')
        const call = calls[index] ??= { id: '', name: '', arguments: '' }
        const fn = raw.function as Record<string, unknown> | undefined
        if (typeof raw.id === 'string' && raw.id) call.id = raw.id
        if (typeof fn?.name === 'string') call.name += fn.name
        if (typeof fn?.arguments === 'string') {
          call.arguments += fn.arguments
          if (call.arguments.length > MAX_ARGUMENT_CHARS) throw new OpenRouterError('The model sent tool input that was too large.')
        }
      }
    }
    return false
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      options.onActivity()
      buffer += decoder.decode(value, { stream: true })
      let newline: number
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline).trim()
        buffer = buffer.slice(newline + 1)
        // Blank lines end events; ':' lines are keep-alive comments.
        if (line.startsWith('data:') && handle(line.slice(5).trim())) return { text, calls: calls.filter(Boolean) }
      }
      if (buffer.length > MAX_ARGUMENT_CHARS * 2) throw new OpenRouterError('OpenRouter sent an unreadable reply. Try again.')
    }
    const rest = buffer.trim()
    if (rest.startsWith('data:')) handle(rest.slice(5).trim())
    return { text, calls: calls.filter(Boolean) }
  } finally {
    await reader.cancel().catch(() => {})
  }
}

/** Run one user message to completion. Resolves with an error message, or null when the model finished. */
export async function runOpenRouterTurn(options: OpenRouterTurnOptions): Promise<string | null> {
  const messages: OpenRouterMessage[] = [
    { role: 'system', content: options.systemPrompt },
    ...options.history,
    { role: 'user', content: options.prompt }
  ]
  let wroteText = false
  try {
    for (let step = 0; step < MAX_STEPS; step++) {
      const { text, calls } = await complete(options, messages, wroteText)
      if (text) wroteText = true
      if (calls.length === 0) {
        return wroteText ? null : 'The model ended without replying. Try again or choose another model.'
      }
      const toolCalls = calls.map((call, index) => ({
        id: call.id || `call_${step}_${index}`,
        type: 'function' as const,
        function: { name: call.name, arguments: call.arguments || '{}' }
      }))
      messages.push({ role: 'assistant', content: text || null, tool_calls: toolCalls })
      for (const call of toolCalls) {
        let args: unknown
        try { args = JSON.parse(call.function.arguments) } catch { args = null }
        const result = args && typeof args === 'object' && !Array.isArray(args)
          ? await options.callTool(call.function.name, args as Record<string, unknown>)
          : { text: 'The tool input was not a JSON object. Send the arguments again as a JSON object.', isError: true }
        messages.push({ role: 'tool', tool_call_id: call.id, content: result.isError ? `Error: ${result.text}` : result.text })
      }
    }
    return `The model took more than ${MAX_STEPS} steps without finishing, so it was stopped. Ask again with a narrower request.`
  } catch (error) {
    if (options.signal.aborted) return null
    if (error instanceof OpenRouterError) return error.message
    return 'The OpenRouter reply failed. Try again.'
  }
}
