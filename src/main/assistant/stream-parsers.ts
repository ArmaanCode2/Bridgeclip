// Normalize the two CLIs' JSONL output into the few things the chat needs:
// assistant text, the id to resume the conversation, and errors. BridgeClip
// tool calls are reported by the MCP server itself, so the CLIs' own tool
// events are only used for activity BridgeClip doesn't run (web searches and
// page reads, Codex shell reads).

export type CliStreamEvent =
  | { kind: 'session'; id: string }
  | { kind: 'text-delta'; text: string }
  /** A new block of assistant text begins; separate it from earlier text. */
  | { kind: 'text-break' }
  | { kind: 'activity'; id: string; title: string; status: 'running' | 'done' | 'error'; summary: string | null }
  | { kind: 'error'; message: string }
  | { kind: 'done'; error: string | null }

export interface CliStreamParser {
  push: (line: string) => CliStreamEvent[]
  /** Called when the process exits; returns an error to show if the turn never finished. */
  finish: () => CliStreamEvent[]
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : null
}

function parseLine(line: string): Record<string, unknown> | null {
  const trimmed = line.trim()
  if (!trimmed.startsWith('{')) return null
  try {
    return record(JSON.parse(trimmed))
  } catch {
    return null
  }
}

function clip(text: string, max = 280): string {
  const single = text.replace(/\s+/g, ' ').trim()
  return single.length > max ? `${single.slice(0, max - 1)}…` : single
}

/** How a use of Claude Code's own web tools reads in the chat. */
function claudeWebActivity(name: unknown, input: unknown): string | null {
  const fields = record(input)
  if (name === 'WebSearch' || name === 'web_search') {
    return typeof fields?.query === 'string' ? `Searched the web for “${clip(fields.query, 80)}”` : 'Searched the web'
  }
  if (name === 'WebFetch' || name === 'web_fetch') {
    try { return `Read ${new URL(String(fields?.url)).hostname}` } catch { return 'Read a web page' }
  }
  return null
}

/** Claude Code `-p --output-format stream-json --verbose --include-partial-messages`. */
export function createClaudeStreamParser(): CliStreamParser {
  let finished = false
  /** Web tool uses waiting for their result, by tool use id. */
  const webTools = new Map<string, string>()
  let sawTextThisTurn = false
  const streamedMessages = new Set<string>()
  let currentMessageId: string | null = null
  let blockIsText = false

  const startText = (events: CliStreamEvent[]): void => {
    if (sawTextThisTurn) events.push({ kind: 'text-break' })
    sawTextThisTurn = true
  }

  return {
    push(line) {
      const event = parseLine(line)
      if (!event) return []
      // Frames from a subagent carry their parent's tool id; only the main thread is shown.
      if (typeof event.parent_tool_use_id === 'string') return []
      const events: CliStreamEvent[] = []
      const type = event.type
      if (type === 'system' && event.subtype === 'init' && typeof event.session_id === 'string') {
        events.push({ kind: 'session', id: event.session_id })
      } else if (type === 'stream_event') {
        const inner = record(event.event)
        if (!inner) return []
        if (inner.type === 'message_start') {
          const message = record(inner.message)
          currentMessageId = typeof message?.id === 'string' ? message.id : null
        } else if (inner.type === 'content_block_start') {
          const block = record(inner.content_block)
          blockIsText = block?.type === 'text'
          if (blockIsText) {
            startText(events)
            if (currentMessageId) streamedMessages.add(currentMessageId)
            if (typeof block?.text === 'string' && block.text) events.push({ kind: 'text-delta', text: block.text })
          }
        } else if (inner.type === 'content_block_delta') {
          const delta = record(inner.delta)
          if (blockIsText && delta?.type === 'text_delta' && typeof delta.text === 'string') {
            events.push({ kind: 'text-delta', text: delta.text })
          }
        } else if (inner.type === 'content_block_stop') {
          blockIsText = false
        }
      } else if (type === 'assistant') {
        // Without partial messages (or if deltas were missed) the final message carries the text.
        const message = record(event.message)
        const id = typeof message?.id === 'string' ? message.id : null
        const streamed = Boolean(id && streamedMessages.has(id))
        const content = Array.isArray(message?.content) ? message.content : []
        for (const part of content) {
          const block = record(part)
          if (block?.type === 'text' && typeof block.text === 'string' && block.text && !streamed) {
            startText(events)
            events.push({ kind: 'text-delta', text: block.text })
          } else if (block?.type === 'web_search_tool_result' && typeof block.tool_use_id === 'string' && webTools.has(block.tool_use_id)) {
            // A server-side search answers inside the same message.
            events.push({ kind: 'activity', id: `web:${block.tool_use_id}`, title: webTools.get(block.tool_use_id)!, status: 'done', summary: null })
            webTools.delete(block.tool_use_id)
          } else if ((block?.type === 'tool_use' || block?.type === 'server_tool_use') && typeof block.id === 'string') {
            const title = claudeWebActivity(block.name, block.input)
            if (title) {
              webTools.set(block.id, title)
              events.push({ kind: 'activity', id: `web:${block.id}`, title, status: 'running', summary: null })
            }
          }
        }
      } else if (type === 'user') {
        // Tool results come back as user messages; close the web tool rows.
        const message = record(event.message)
        const content = Array.isArray(message?.content) ? message.content : []
        for (const part of content) {
          const block = record(part)
          const toolId = typeof block?.tool_use_id === 'string' ? block.tool_use_id : null
          const title = toolId ? webTools.get(toolId) : undefined
          if (!toolId || !title) continue
          webTools.delete(toolId)
          events.push({ kind: 'activity', id: `web:${toolId}`, title, status: block?.is_error === true ? 'error' : 'done', summary: null })
        }
      } else if (type === 'result') {
        finished = true
        if (typeof event.session_id === 'string') events.push({ kind: 'session', id: event.session_id })
        const isError = event.is_error === true || (typeof event.subtype === 'string' && event.subtype !== 'success')
        const message = typeof event.result === 'string' && event.result.trim()
          ? event.result
          : typeof event.subtype === 'string' ? event.subtype.replace(/_/g, ' ') : 'Claude Code stopped'
        events.push({ kind: 'done', error: isError ? clip(message, 600) : null })
      }
      return events
    },
    finish() {
      return finished ? [] : [{ kind: 'done', error: 'Claude Code stopped before finishing its reply.' }]
    }
  }
}

/** Codex `exec --json`. Items arrive whole (or as growing updates), not as token deltas. */
export function createCodexStreamParser(): CliStreamParser {
  let finished = false
  let sawText = false
  let pendingError: string | null = null
  const textSeen = new Map<string, string>()

  const emitText = (id: string, text: string, events: CliStreamEvent[]): void => {
    const previous = textSeen.get(id)
    if (previous === undefined) {
      if (!text) return
      if (sawText) events.push({ kind: 'text-break' })
      sawText = true
      textSeen.set(id, text)
      events.push({ kind: 'text-delta', text })
      return
    }
    if (text.length > previous.length && text.startsWith(previous)) {
      textSeen.set(id, text)
      events.push({ kind: 'text-delta', text: text.slice(previous.length) })
    }
  }

  return {
    push(line) {
      const event = parseLine(line)
      if (!event) return []
      const events: CliStreamEvent[] = []
      const type = event.type
      if (type === 'thread.started' && typeof event.thread_id === 'string') {
        events.push({ kind: 'session', id: event.thread_id })
      } else if (type === 'item.started' || type === 'item.updated' || type === 'item.completed') {
        const item = record(event.item)
        if (!item) return []
        const id = typeof item.id === 'string' ? item.id : ''
        const completed = type === 'item.completed'
        if (item.type === 'agent_message' && typeof item.text === 'string') {
          emitText(id, item.text, events)
        } else if (item.type === 'command_execution' && typeof item.command === 'string') {
          const failed = completed && typeof item.exit_code === 'number' && item.exit_code !== 0
          events.push({
            kind: 'activity',
            id: `command:${id}`,
            title: `Ran ${clip(item.command, 80)}`,
            status: !completed ? 'running' : failed ? 'error' : 'done',
            summary: completed && typeof item.aggregated_output === 'string' ? clip(item.aggregated_output) || null : null
          })
        } else if (item.type === 'web_search') {
          // The query only arrives with the finished item.
          const query = typeof item.query === 'string' ? item.query.trim() : ''
          events.push({ kind: 'activity', id: `search:${id}`, title: query ? `Searched the web for “${clip(query, 80)}”` : 'Searching the web…', status: completed ? 'done' : 'running', summary: null })
        } else if (item.type === 'error' && typeof item.message === 'string') {
          pendingError = item.message
        }
      } else if (type === 'turn.completed') {
        finished = true
        events.push({ kind: 'done', error: null })
      } else if (type === 'turn.failed') {
        finished = true
        const error = record(event.error)
        events.push({ kind: 'done', error: clip(typeof error?.message === 'string' ? error.message : pendingError ?? 'Codex could not finish this reply.', 600) })
      } else if (type === 'error' && typeof event.message === 'string') {
        // Codex retries some errors; only surface it if the turn never completes.
        pendingError = event.message
      }
      return events
    },
    finish() {
      if (finished) return []
      return [{ kind: 'done', error: clip(pendingError ?? 'Codex stopped before finishing its reply.', 600) }]
    }
  }
}

/**
 * Map a failed CLI's stderr to something the user can act on. Matching is on
 * phrases both CLIs have used for years; anything else shows a trimmed tail.
 */
export function describeCliFailure(provider: 'claude' | 'codex', stderr: string, exitCode: number | null): string {
  const text = stderr.toLowerCase()
  if (provider === 'claude') {
    if (/not logged in|please run \/login|invalid api key|authentication_error|oauth token has expired|not authenticated/.test(text)) {
      return 'Claude Code isn’t signed in on this computer. Open a terminal, run `claude`, sign in with your Claude subscription, then try again.'
    }
    if (/rate limit|usage limit|session limit|you've hit your limit|you’ve hit your limit/.test(text)) {
      return 'Your Claude subscription has reached its usage limit for now. Try again after it resets.'
    }
    if (/unknown option|unknown argument/.test(text)) {
      return 'This version of Claude Code is too old for BridgeClip. Run `claude update`, then try again.'
    }
  } else {
    if (/not logged in|codex login|login required|not authenticated|401 unauthorized/.test(text)) {
      return 'Codex isn’t signed in on this computer. Open a terminal, run `codex login`, sign in with ChatGPT, then try again.'
    }
    if (/rate limit|usage limit|you've hit your usage limit|you’ve hit your usage limit/.test(text)) {
      return 'Your ChatGPT plan has reached its Codex usage limit for now. Try again after it resets.'
    }
    if (/unexpected argument|unrecognized subcommand/.test(text)) {
      return 'This version of Codex is too old for BridgeClip. Run `codex update`, then try again.'
    }
  }
  const tail = stderr.trim().split('\n').filter(Boolean).slice(-3).join(' ')
  const name = provider === 'claude' ? 'Claude Code' : 'Codex'
  return tail ? `${name} stopped: ${clip(tail, 400)}` : `${name} stopped unexpectedly${exitCode === null ? '' : ` (exit code ${exitCode})`}.`
}
