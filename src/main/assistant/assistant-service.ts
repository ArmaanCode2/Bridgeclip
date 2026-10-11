import { randomUUID } from 'crypto'
import { mkdirSync } from 'fs'
import { join } from 'path'
import { createInterface } from 'readline'
import {
  ASSISTANT_PROMPT_MAX_CHARS,
  isAssistantModel,
  type AssistantApprovalRequest,
  type AssistantConversation,
  type AssistantConversationSummary,
  type AssistantEvent,
  type AssistantMessage,
  type AssistantCliProviderId,
  type AssistantPreferences,
  type AssistantProviderId,
  type AssistantToolCall
} from '../../shared/assistant'
import { resolveCli, spawnCli, stopAllCliRuns, type ResolvedCli } from './cli-process'
import { ConversationStore, isConversationId } from './conversation-store'
import { LoopbackMcpServer, type McpToolDefinition, type McpToolResult } from './mcp-server'
import { claudeTurnArgs, claudeTurnEnv, codexTurnArgs, codexTurnEnv } from './providers'
import { runOpenRouterTurn, type OpenRouterMessage } from './openrouter-agent'
import { createClaudeStreamParser, createCodexStreamParser, describeCliFailure, type CliStreamEvent } from './stream-parsers'
import { validateToolInput } from './tool-input'
import { AssistantToolError, type AssistantToolSpec } from './tool-types'

const FIRST_OUTPUT_TIMEOUT_MS = 3 * 60 * 1000
const SILENCE_TIMEOUT_MS = 30 * 60 * 1000
const APPROVAL_TIMEOUT_MS = 15 * 60 * 1000
const MAX_ACTIVE_TURNS = 3
const STDERR_TAIL_BYTES = 16 * 1024
const TOOL_RESULT_MAX_CHARS = 60000
const HISTORY_CONTEXT_MAX_CHARS = 24000
/** OpenRouter keeps no session, so each message resends this much of the chat. */
const OPENROUTER_HISTORY_MAX_CHARS = 60000

export interface AssistantServiceOptions {
  /** <userData>/assistant */
  root: string
  appVersion: string
  tools: () => AssistantToolSpec[]
  systemPrompt: () => string
  emit: (event: AssistantEvent) => void
  /** Finds the CLI to run; tests substitute a fake. */
  resolveCli?: (provider: AssistantCliProviderId) => Promise<ResolvedCli | null>
  /** Chat on OpenRouter models with the user's key from Settings. */
  openRouter?: {
    apiKey: () => string
    url?: string
    fetch?: typeof fetch
  }
}

export interface SendMessageInput {
  conversationId: string | null
  provider: AssistantProviderId
  model: string
  text: string
}

interface ActiveTurn {
  conversation: AssistantConversation
  message: AssistantMessage
  stop: (reason: string) => void
  stopReason: string | null
  busy: number
  lastActivity: number
  sawOutput: boolean
}

interface PendingApproval {
  conversationId: string
  resolve: (allowed: boolean) => void
}

function now(): string {
  return new Date().toISOString()
}

function titleFrom(text: string): string {
  const single = text.replace(/\s+/g, ' ').trim()
  return single.length > 60 ? `${single.slice(0, 59)}…` : single || 'New chat'
}

function messageText(message: AssistantMessage): string {
  return message.parts.map((part) => part.kind === 'text' ? part.text ?? '' : part.tool ? `[used ${part.tool.title}]` : '').join('').trim()
}

function stringifyResult(value: unknown): string {
  const text = typeof value === 'string' ? value : JSON.stringify(value ?? { ok: true }, null, 1)
  return text.length > TOOL_RESULT_MAX_CHARS
    ? `${text.slice(0, TOOL_RESULT_MAX_CHARS)}\n…(truncated; ask for fewer items or a narrower query)`
    : text
}

function summarize(value: unknown): string | null {
  if (value && typeof value === 'object' && typeof (value as { summary?: unknown }).summary === 'string') {
    return (value as { summary: string }).summary
  }
  return null
}

const STALE_RESUME = /no conversation found|session .*not found|thread .*not found|no rollout found|could not find (?:session|thread|conversation)/i

export class AssistantService {
  private readonly store: ConversationStore
  private readonly mcp: LoopbackMcpServer
  private readonly turns = new Map<string, ActiveTurn>()
  private readonly approvals = new Map<string, PendingApproval>()

  constructor(private readonly options: AssistantServiceOptions) {
    mkdirSync(options.root, { recursive: true, mode: 0o700 })
    mkdirSync(this.workspace(), { recursive: true, mode: 0o700 })
    this.store = new ConversationStore(options.root)
    this.mcp = new LoopbackMcpServer(options.appVersion)
  }

  /** An empty folder the CLIs run in, so no project's CLAUDE.md, AGENTS.md or settings apply. */
  private workspace(): string {
    return join(this.options.root, 'workspace')
  }

  listConversations(): AssistantConversationSummary[] {
    return this.store.list()
  }

  getConversation(id: string): AssistantConversation | null {
    if (!isConversationId(id)) return null
    return this.turns.get(id)?.conversation ?? this.store.get(id)
  }

  deleteConversation(id: string): void {
    if (!isConversationId(id)) throw new Error('Invalid conversation id')
    if (this.turns.has(id)) throw new Error('Stop the reply before deleting this chat.')
    this.store.delete(id)
  }

  preferences(): AssistantPreferences {
    return this.store.preferences()
  }

  savePreferences(preferences: AssistantPreferences): AssistantPreferences {
    return this.store.savePreferences(preferences)
  }

  isRunning(id: string): boolean {
    return this.turns.has(id)
  }

  runningConversationIds(): string[] {
    return [...this.turns.keys()]
  }

  stop(conversationId: string): void {
    this.turns.get(conversationId)?.stop('You stopped this reply.')
  }

  respondToApproval(requestId: string, allowed: boolean): void {
    const pending = this.approvals.get(requestId)
    if (!pending) return
    this.approvals.delete(requestId)
    pending.resolve(allowed)
    this.options.emit({ type: 'approval-resolved', conversationId: pending.conversationId, requestId })
  }

  async shutdown(): Promise<void> {
    for (const turn of this.turns.values()) turn.stop('BridgeClip is closing.')
    for (const [id, pending] of this.approvals) {
      this.approvals.delete(id)
      pending.resolve(false)
    }
    stopAllCliRuns()
    await this.mcp.stop()
  }

  /** Start a turn and return immediately; progress arrives as events. */
  sendMessage(input: SendMessageInput): { conversationId: string; messageId: string } {
    const text = input.text.trim()
    if (!text) throw new Error('Type a message first.')
    if (text.length > ASSISTANT_PROMPT_MAX_CHARS) throw new Error(`Messages are limited to ${ASSISTANT_PROMPT_MAX_CHARS.toLocaleString()} characters.`)
    if (!isAssistantModel(input.provider, input.model)) throw new Error('Choose a model for this assistant.')
    if (input.provider === 'openrouter' && !input.model) throw new Error('Choose an OpenRouter model first.')
    if (this.turns.size >= MAX_ACTIVE_TURNS) throw new Error('Too many chats are replying at once. Wait for one to finish.')

    let conversation: AssistantConversation | null = null
    if (input.conversationId) {
      if (!isConversationId(input.conversationId)) throw new Error('Invalid conversation id')
      if (this.turns.has(input.conversationId)) throw new Error('This chat is still replying.')
      conversation = this.store.get(input.conversationId)
      if (!conversation) throw new Error('This chat no longer exists.')
    }
    const timestamp = now()
    let historyContext: string | null = null
    if (!conversation) {
      conversation = {
        id: randomUUID(),
        title: titleFrom(text),
        provider: input.provider,
        model: input.model,
        resumeId: null,
        createdAt: timestamp,
        updatedAt: timestamp,
        messages: []
      }
    } else if (conversation.provider !== input.provider) {
      // A different CLI can't resume the other's session; carry the chat over as context.
      // OpenRouter is sent the whole chat on every message anyway.
      if (input.provider !== 'openrouter') historyContext = this.historyContext(conversation)
      conversation.provider = input.provider
      conversation.resumeId = null
    }
    conversation.model = input.model
    const userMessage: AssistantMessage = { id: randomUUID(), role: 'user', createdAt: timestamp, parts: [{ kind: 'text', text }] }
    const reply: AssistantMessage = { id: randomUUID(), role: 'assistant', createdAt: timestamp, parts: [], error: null, provider: input.provider, model: input.model }
    conversation.messages.push(userMessage, reply)
    conversation.updatedAt = timestamp
    this.store.save(conversation)
    this.options.emit({ type: 'conversation', conversation: { id: conversation.id, title: conversation.title, provider: conversation.provider, updatedAt: conversation.updatedAt } })

    const turn: ActiveTurn = {
      conversation,
      message: reply,
      stop: () => {},
      stopReason: null,
      busy: 0,
      lastActivity: Date.now(),
      sawOutput: false
    }
    this.turns.set(conversation.id, turn)
    this.options.emit({ type: 'turn-start', conversationId: conversation.id, messageId: reply.id })
    void this.runTurn(turn, text, historyContext)
    return { conversationId: conversation.id, messageId: reply.id }
  }

  private historyContext(conversation: AssistantConversation): string | null {
    const lines: string[] = []
    let size = 0
    for (const message of [...conversation.messages].reverse()) {
      const text = messageText(message)
      if (!text) continue
      const line = `${message.role === 'user' ? 'User' : 'Assistant'}: ${text}`
      if (size + line.length > HISTORY_CONTEXT_MAX_CHARS) break
      lines.unshift(line)
      size += line.length
    }
    return lines.length ? lines.join('\n\n') : null
  }

  private async runTurn(turn: ActiveTurn, text: string, historyContext: string | null): Promise<void> {
    const { conversation } = turn
    let error: string | null = null
    try {
      const provider = conversation.provider
      if (provider === 'openrouter') {
        error = await this.runOpenRouter(turn, text)
      } else {
        let attempt = await this.runAttempt(turn, provider, historyContext ? this.withHistory(historyContext, text) : text)
        if (attempt.staleResume && !turn.stopReason) {
          // The CLI no longer has this session (cleared, or another computer). Start fresh with the chat as context.
          conversation.resumeId = null
          const context = this.historyContext({ ...conversation, messages: conversation.messages.slice(0, -2) })
          attempt = await this.runAttempt(turn, provider, context ? this.withHistory(context, text) : text)
        }
        error = attempt.error
      }
    } catch (failure) {
      error = failure instanceof Error ? failure.message : 'The assistant could not start.'
    } finally {
      if (turn.stopReason) error = turn.stopReason
      for (const [id, pending] of this.approvals) {
        if (pending.conversationId === conversation.id) {
          this.approvals.delete(id)
          pending.resolve(false)
          this.options.emit({ type: 'approval-resolved', conversationId: conversation.id, requestId: id })
        }
      }
      for (const part of turn.message.parts) {
        if (part.tool && (part.tool.status === 'running' || part.tool.status === 'awaiting-approval')) {
          part.tool.status = 'error'
          part.tool.summary = part.tool.summary ?? 'Interrupted'
        }
      }
      turn.message.error = error
      conversation.updatedAt = now()
      this.turns.delete(conversation.id)
      try { this.store.save(conversation) } catch { /* the event still reaches the chat */ }
      this.options.emit({ type: 'turn-end', conversationId: conversation.id, messageId: turn.message.id, error })
    }
  }

  private withHistory(history: string, text: string): string {
    return `Earlier in this BridgeClip chat (for context; it happened before this session):\n\n${history}\n\n---\n\n${text}`
  }

  /** The chat before this message, as OpenRouter messages, newest kept when it's long. */
  private openRouterHistory(conversation: AssistantConversation): OpenRouterMessage[] {
    const history: OpenRouterMessage[] = []
    let size = 0
    for (const message of conversation.messages.slice(0, -2).reverse()) {
      const text = messageText(message)
      if (!text) continue
      if (size + text.length > OPENROUTER_HISTORY_MAX_CHARS) break
      history.unshift({ role: message.role, content: text })
      size += text.length
    }
    return history
  }

  private appendText(turn: ActiveTurn, text: string, newParagraph: boolean): void {
    const last = turn.message.parts[turn.message.parts.length - 1]
    const textToAdd = newParagraph && last?.kind === 'text' && last.text ? `\n\n${text}` : text
    if (last?.kind === 'text') last.text = (last.text ?? '') + textToAdd
    else turn.message.parts.push({ kind: 'text', text: textToAdd })
    turn.lastActivity = Date.now()
    this.options.emit({ type: 'text-delta', conversationId: turn.conversation.id, messageId: turn.message.id, text: textToAdd })
  }

  private async runOpenRouter(turn: ActiveTurn, text: string): Promise<string | null> {
    const { conversation } = turn
    const apiKey = this.options.openRouter?.apiKey().trim() ?? ''
    if (!apiKey) return 'Add your OpenRouter API key in Settings → API keys to chat with OpenRouter models.'
    if (!conversation.model) return 'Choose an OpenRouter model for this chat.'
    const controller = new AbortController()
    turn.stop = (reason) => {
      turn.stopReason ??= reason
      controller.abort()
    }
    if (turn.stopReason) return turn.stopReason
    turn.lastActivity = Date.now()
    turn.sawOutput = false
    const watchdog = setInterval(() => {
      if (turn.busy > 0) return
      const idle = Date.now() - turn.lastActivity
      if (!turn.sawOutput && idle > FIRST_OUTPUT_TIMEOUT_MS) turn.stop('OpenRouter didn’t respond. Check your connection, or choose another model.')
      else if (idle > SILENCE_TIMEOUT_MS) turn.stop('The reply stalled and was stopped.')
    }, 5000)
    try {
      const error = await runOpenRouterTurn({
        apiKey,
        model: conversation.model,
        systemPrompt: this.options.systemPrompt(),
        history: this.openRouterHistory(conversation),
        prompt: text,
        tools: this.toolsFor('openrouter').map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.inputSchema })),
        signal: controller.signal,
        url: this.options.openRouter?.url,
        fetch: this.options.openRouter?.fetch,
        onText: (delta, newParagraph) => this.appendText(turn, delta, newParagraph),
        onActivity: () => {
          turn.lastActivity = Date.now()
          turn.sawOutput = true
        },
        callTool: (name, args) => this.callTool(turn, name, args, controller.signal)
      })
      return turn.stopReason ?? error
    } finally {
      clearInterval(watchdog)
    }
  }

  private async runAttempt(turn: ActiveTurn, provider: AssistantCliProviderId, prompt: string): Promise<{ error: string | null; staleResume: boolean }> {
    const { conversation } = turn
    const cli = await (this.options.resolveCli ?? resolveCli)(provider)
    if (!cli) {
      return {
        error: provider === 'claude'
          ? 'Claude Code isn’t installed. Connect it in Settings → Assistant.'
          : 'Codex isn’t installed. Connect it in Settings → Assistant.',
        staleResume: false
      }
    }
    const session = await this.mcp.openSession({
      instructions: 'Tools for controlling the BridgeClip desktop app: clipping jobs, the clip library, automations, social posting and settings.',
      listTools: () => this.toolDefinitions(provider),
      callTool: (name, args, signal) => this.callTool(turn, name, args, signal)
    })
    const resumeId = conversation.resumeId
    try {
      const commandOptions = {
        cli,
        model: conversation.model,
        resumeId,
        systemPrompt: this.options.systemPrompt(),
        mcpUrl: session.url,
        cwd: this.workspace()
      }
      const args = provider === 'claude' ? claudeTurnArgs(commandOptions) : codexTurnArgs(commandOptions)
      const env = provider === 'claude' ? claudeTurnEnv(cli, session.token) : codexTurnEnv(cli, session.token)
      if (turn.stopReason) return { error: turn.stopReason, staleResume: false }
      const run = spawnCli(cli, args, { cwd: this.workspace(), env })
      turn.stop = (reason) => {
        turn.stopReason ??= reason
        run.stop()
      }
      turn.lastActivity = Date.now()
      turn.sawOutput = false
      const watchdog = setInterval(() => {
        if (turn.busy > 0) return
        const idle = Date.now() - turn.lastActivity
        if (!turn.sawOutput && idle > FIRST_OUTPUT_TIMEOUT_MS) {
          turn.stop(`${provider === 'claude' ? 'Claude Code' : 'Codex'} didn’t respond. Check that it’s signed in and online, then try again.`)
        } else if (idle > SILENCE_TIMEOUT_MS) {
          turn.stop('The reply stalled and was stopped.')
        }
      }, 5000)

      let stderr = ''
      run.child.stderr?.on('data', (chunk: Buffer) => {
        stderr = (stderr + chunk.toString('utf8')).slice(-STDERR_TAIL_BYTES)
      })
      run.child.stdin?.on('error', () => {})
      run.child.stdin?.end(prompt)

      const parser = provider === 'claude' ? createClaudeStreamParser() : createCodexStreamParser()
      let doneError: string | null | undefined
      let pendingBreak = false
      const apply = (events: CliStreamEvent[]): void => {
        for (const event of events) {
          if (event.kind === 'session') {
            conversation.resumeId = event.id
          } else if (event.kind === 'text-break') {
            pendingBreak = true
          } else if (event.kind === 'text-delta') {
            this.appendText(turn, event.text, pendingBreak)
            pendingBreak = false
          } else if (event.kind === 'activity') {
            this.upsertTool(turn, { id: event.id, name: 'activity', title: event.title, input: null, status: event.status, summary: event.summary })
          } else if (event.kind === 'done') {
            doneError = event.error
          }
        }
      }

      const lines = createInterface({ input: run.child.stdout!, crlfDelay: Infinity })
      lines.on('line', (line) => {
        turn.lastActivity = Date.now()
        turn.sawOutput = true
        apply(parser.push(line))
      })
      const exit = await run.exited
      await new Promise<void>((resolve) => {
        if ((run.child.stdout as NodeJS.ReadableStream & { readableEnded?: boolean })?.readableEnded) resolve()
        else lines.once('close', () => resolve())
        setTimeout(resolve, 1000)
      })
      clearInterval(watchdog)
      apply(parser.finish())

      if (turn.stopReason) return { error: turn.stopReason, staleResume: false }
      const staleResume = Boolean(resumeId) && STALE_RESUME.test(`${stderr}\n${doneError ?? ''}`) && !messageText(turn.message)
      if (doneError === null && (exit.code === 0 || exit.code === null)) return { error: null, staleResume: false }
      if (doneError === null) return { error: null, staleResume: false }
      // Prefer the CLI's own explanation when it printed one; the parser's fallback is generic.
      const stderrExplanation = stderr.trim() ? describeCliFailure(provider, stderr, exit.code) : null
      const parserError = typeof doneError === 'string' ? doneError : null
      const authOrLimit = stderrExplanation && !stderrExplanation.includes('stopped')
      return { error: (authOrLimit ? stderrExplanation : parserError ?? stderrExplanation) ?? describeCliFailure(provider, stderr, exit.code), staleResume }
    } finally {
      session.close()
    }
  }

  private upsertTool(turn: ActiveTurn, tool: AssistantToolCall): void {
    const existing = turn.message.parts.find((part) => part.tool?.id === tool.id)
    if (existing?.tool) Object.assign(existing.tool, tool)
    else turn.message.parts.push({ kind: 'tool', tool: { ...tool } })
    turn.lastActivity = Date.now()
    this.options.emit({ type: 'tool', conversationId: turn.conversation.id, messageId: turn.message.id, tool: { ...tool } })
  }

  /** The tools offered on this provider: the web tools only go to OpenRouter models, which have no web access of their own. */
  private toolsFor(provider: AssistantProviderId): AssistantToolSpec[] {
    return this.options.tools().filter((tool) => !tool.providers || tool.providers.includes(provider))
  }

  private toolDefinitions(provider: AssistantProviderId): McpToolDefinition[] {
    return this.toolsFor(provider).map((tool) => ({
      name: tool.name,
      title: tool.title,
      description: tool.description,
      inputSchema: tool.inputSchema,
      annotations: { readOnlyHint: Boolean(tool.readOnly), destructiveHint: Boolean(tool.destructive), openWorldHint: false }
    }))
  }

  private requestApproval(turn: ActiveTurn, call: AssistantToolCall, details: string[], signal: AbortSignal): Promise<boolean> {
    const request: AssistantApprovalRequest = {
      id: randomUUID(),
      conversationId: turn.conversation.id,
      toolCallId: call.id,
      tool: call.name,
      title: call.title,
      details
    }
    return new Promise((resolve) => {
      const finish = (allowed: boolean): void => {
        clearTimeout(timer)
        signal.removeEventListener('abort', onAbort)
        resolve(allowed)
      }
      const timer = setTimeout(() => this.respondToApproval(request.id, false), APPROVAL_TIMEOUT_MS)
      const onAbort = (): void => this.respondToApproval(request.id, false)
      signal.addEventListener('abort', onAbort, { once: true })
      this.approvals.set(request.id, { conversationId: turn.conversation.id, resolve: finish })
      this.options.emit({ type: 'approval', conversationId: turn.conversation.id, request })
    })
  }

  private async callTool(turn: ActiveTurn, name: string, rawArgs: Record<string, unknown>, signal: AbortSignal): Promise<McpToolResult> {
    const spec = this.toolsFor(turn.conversation.provider).find((tool) => tool.name === name)
    if (!spec) return { text: `Unknown tool: ${name}`, isError: true }
    const call: AssistantToolCall = { id: randomUUID(), name, title: spec.title, input: rawArgs, status: 'running', summary: null }
    turn.busy += 1
    try {
      let input: Record<string, unknown>
      try {
        input = validateToolInput(rawArgs, spec.inputSchema)
      } catch (error) {
        return { text: error instanceof Error ? error.message : 'Invalid input', isError: true }
      }
      call.title = spec.describe?.(input) ?? spec.title
      const details = spec.confirm ? await spec.confirm(input) : null
      if (details) {
        call.status = 'awaiting-approval'
        this.upsertTool(turn, call)
        const allowed = await this.requestApproval(turn, call, details, signal)
        if (!allowed) {
          call.status = 'denied'
          call.summary = 'Not approved'
          this.upsertTool(turn, call)
          return { text: 'The user did not approve this action. Do not retry it unless they ask you to.', isError: true }
        }
        call.status = 'running'
      }
      this.upsertTool(turn, call)
      const result = await spec.run(input, { conversationId: turn.conversation.id, signal })
      call.status = 'done'
      call.summary = summarize(result)
      this.upsertTool(turn, call)
      return { text: stringifyResult(result) }
    } catch (error) {
      const message = error instanceof AssistantToolError || error instanceof Error ? error.message : 'The tool failed.'
      call.status = 'error'
      call.summary = message
      this.upsertTool(turn, call)
      return { text: message, isError: true }
    } finally {
      turn.busy -= 1
      turn.lastActivity = Date.now()
      try { this.store.save(turn.conversation) } catch { /* saved again when the turn ends */ }
    }
  }
}
