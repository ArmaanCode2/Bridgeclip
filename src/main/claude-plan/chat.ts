import { cliEnvironment, resolveCli, spawnCli, type ResolvedCli } from '../assistant/cli-process'
import { claudePlanWorkspace } from './runner'

/**
 * One question to the person's Claude plan from the main process, for metadata writing and Test
 * connection. A clip job asks from the engine instead (engine/clip_engine/services/claude_plan.py);
 * both run Claude Code the same way: no tools, no saved session, an empty folder and an allowlisted
 * environment. Messages are fixed sentences, because Claude Code's own text can echo the request.
 */

export interface ClaudePlanQuestion {
  system: string
  prompt: string
  /** A picture to read with the prompt, as a data: URL. */
  image?: string
  schema: object
  model: string
  effort: string
  timeoutMs: number
}

export type ClaudePlanAnswer = { ok: true; content: string } | { ok: false; message: string }

interface RunEvent {
  type?: unknown
  error?: unknown
  is_error?: unknown
  result?: unknown
  structured_output?: unknown
  apiKeySource?: unknown
  rate_limit_info?: { status?: unknown; isUsingOverage?: unknown }
}

const MAX_OUTPUT_BYTES = 8 * 1024 * 1024
/** Claude Code's own system prompt describes a coding agent, so a question without one still replaces it. */
const DEFAULT_SYSTEM_PROMPT = 'Do what the user asks. Reply with the answer only.'

const NOT_INSTALLED = 'Your Claude plan needs Claude Code, which is not installed on this computer.'
const TOO_LONG = 'Your Claude plan took too long to answer. Try again, or use a faster model.'
const NO_ANSWER = 'Your Claude plan could not answer. Try again.'
const LIMIT = 'Your Claude plan reached its usage limit. Try again after it resets.'
const NOT_ON_PLAN = 'Your Claude plan is not being used: Claude Code is signed in another way. Open a terminal, run claude and sign in with your plan.'

/** Claude Code's error code or words -> the sentence shown. The engine's _FAILURES has the same rows. */
const FAILURES: readonly { codes: readonly string[]; words: RegExp; message: string }[] = [
  {
    codes: ['authentication_failed'], words: /not logged in|\/login|invalid api key|authentication_error|oauth token|not authenticated/i,
    message: 'Your Claude plan is not available: Claude Code is not signed in. Open a terminal, run claude and sign in.'
  },
  {
    codes: ['rate_limit', 'billing_error'], words: /rate limit|usage limit|session limit|weekly limit|hit your \S* ?limit/i, message: LIMIT
  },
  { codes: ['model_not_found'], words: /selected model/i, message: 'Your Claude plan cannot use this model. Pick another in Settings.' },
  { codes: [], words: /prompt is too long/i, message: 'Your Claude plan cannot take a request this long. Pick a larger model in Settings.' },
  { codes: [], words: /unknown option|unknown argument/i, message: 'Your Claude plan needs a newer Claude Code. Run claude update, then try again.' }
]

/** The one user message a run takes, as stdin. */
export function claudePlanMessage(question: ClaudePlanQuestion): string {
  const picture = /^data:(image\/[a-z]+);base64,(.+)$/.exec(question.image ?? '')
  const content = [
    ...(picture ? [{ type: 'image', source: { type: 'base64', media_type: picture[1], data: picture[2] } }] : []),
    { type: 'text', text: question.prompt }
  ]
  return `${JSON.stringify({ type: 'user', message: { role: 'user', content } })}\n`
}

export function claudePlanArgs(question: ClaudePlanQuestion): string[] {
  return [
    '-p',
    '--model', question.model,
    '--effort', question.effort,
    '--tools', '',
    '--strict-mcp-config',
    // The person's settings file could name a key or remap a model. Signing in still works.
    '--setting-sources', '',
    '--settings', JSON.stringify({ disableAllHooks: true }),
    '--disable-slash-commands',
    '--no-session-persistence',
    '--permission-mode', 'dontAsk',
    '--system-prompt', question.system || DEFAULT_SYSTEM_PROMPT,
    '--input-format', 'stream-json',
    '--output-format', 'stream-json',
    '--verbose',
    '--json-schema', JSON.stringify(question.schema)
  ]
}

function runEvents(stdout: string): RunEvent[] {
  const events: RunEvent[] = []
  for (const line of stdout.split('\n')) {
    try {
      const event = JSON.parse(line) as unknown
      if (event && typeof event === 'object') events.push(event as RunEvent)
    } catch { /* Not every line Claude Code prints is an event. */ }
  }
  return events
}

/** A finished run's answer, or why there is none. */
export function readClaudeRun(stdout: string, stderr: string): ClaudePlanAnswer {
  const events = runEvents(stdout)
  const limits = events.filter((event) => event.type === 'rate_limit_event').map((event) => event.rate_limit_info ?? {})
  if (events.some((event) => event.type === 'system' && event.apiKeySource !== undefined && event.apiKeySource !== 'none')) {
    return { ok: false, message: NOT_ON_PLAN }
  }
  // Paid overflow past the plan's limit, or a run the limit refused.
  if (limits.some((limit) => limit.isUsingOverage || limit.status === 'rejected')) return { ok: false, message: LIMIT }
  const result = events.filter((event) => event.type === 'result').pop()
  if (result && !result.is_error) {
    const structured = result.structured_output
    const content = structured !== undefined && structured !== null ? JSON.stringify(structured) : typeof result.result === 'string' ? result.result : ''
    return content ? { ok: true, content } : { ok: false, message: NO_ANSWER }
  }
  const code = events.find((event) => event.type === 'assistant' && event.error)?.error
  const text = `${typeof result?.result === 'string' ? result.result : ''} ${stderr}`
  const failure = FAILURES.find(({ codes, words }) => (typeof code === 'string' && codes.includes(code)) || words.test(text))
  return { ok: false, message: failure?.message ?? NO_ANSWER }
}

export async function askClaudePlan(question: ClaudePlanQuestion, deps: { cli?: ResolvedCli | null; cwd?: string } = {}): Promise<ClaudePlanAnswer> {
  const cli = deps.cli === undefined ? await resolveCli('claude') : deps.cli
  if (!cli) return { ok: false, message: NOT_INSTALLED }
  const run = spawnCli(cli, claudePlanArgs(question), { cwd: deps.cwd ?? claudePlanWorkspace(), env: cliEnvironment(cli.pathValue) })
  const collect = (stream: NodeJS.ReadableStream | null): (() => string) => {
    const chunks: Buffer[] = []
    let size = 0
    stream?.on('data', (chunk: Buffer) => {
      size += chunk.byteLength
      if (size <= MAX_OUTPUT_BYTES) chunks.push(chunk)
    })
    return () => Buffer.concat(chunks).toString('utf-8')
  }
  const stdout = collect(run.child.stdout)
  const stderr = collect(run.child.stderr)
  // Writing to a program that failed to start reports the same failure again.
  run.child.stdin?.on('error', () => {})
  run.child.stdin?.end(claudePlanMessage(question))
  let timedOut = false
  const timer = setTimeout(() => { timedOut = true; run.stop() }, question.timeoutMs)
  const { code, signal } = await run.exited
  clearTimeout(timer)
  if (timedOut) return { ok: false, message: TOO_LONG }
  // No exit code and no signal: the program never started.
  if (code === null && signal === null) return { ok: false, message: NOT_INSTALLED }
  return readClaudeRun(stdout(), stderr())
}
