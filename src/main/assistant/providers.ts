import type { AssistantCliProviderId, AssistantProviderId, AssistantProviderStatus } from '../../shared/assistant'
import { cliEnvironment, resolveCli, runCliCommand, type ResolvedCli } from './cli-process'

export const MCP_SERVER_NAME = 'bridgeclip'
export const MCP_TOKEN_ENV = 'BRIDGECLIP_MCP_TOKEN'
/** Approval prompts can wait on the user; neither CLI may give up on a tool call first. */
const MCP_TOOL_TIMEOUT_SECONDS = 900

function firstVersion(text: string): string | null {
  return text.match(/\d+\.\d+\.\d+(?:[-.][0-9A-Za-z.]+)?/)?.[0] ?? null
}

function status(id: AssistantProviderId, partial: Partial<AssistantProviderStatus> & Pick<AssistantProviderStatus, 'state'>): AssistantProviderStatus {
  return { id, version: null, account: null, plan: null, detail: null, checkedAt: new Date().toISOString(), ...partial }
}

/** Read `claude auth status` JSON. Exit code is 1 when signed out, so parse stdout either way. */
export function parseClaudeAuthStatus(stdout: string): { loggedIn: boolean; account: string | null; plan: string | null; apiKey: boolean } | null {
  const start = stdout.indexOf('{')
  if (start < 0) return null
  try {
    const parsed = JSON.parse(stdout.slice(start)) as Record<string, unknown>
    if (typeof parsed.loggedIn !== 'boolean') return null
    const method = typeof parsed.authMethod === 'string' ? parsed.authMethod : ''
    return {
      loggedIn: parsed.loggedIn && method !== 'none',
      account: typeof parsed.email === 'string' ? parsed.email : null,
      plan: typeof parsed.subscriptionType === 'string' ? parsed.subscriptionType : null,
      apiKey: /api.?key|console/i.test(method)
    }
  } catch {
    return null
  }
}

/** `codex login status` prints to stderr: "Logged in using ChatGPT" / "Not logged in". */
export function parseCodexLoginStatus(output: string, exitCode: number | null): { loggedIn: boolean; apiKey: boolean } | null {
  if (/not logged in/i.test(output)) return { loggedIn: false, apiKey: false }
  if (/logged in/i.test(output)) return { loggedIn: true, apiKey: /api key/i.test(output) }
  if (exitCode === 1) return { loggedIn: false, apiKey: false }
  return null
}

/** OpenRouter has no CLI or sign-in: it's ready when Settings has an OpenRouter key. */
export function openRouterStatus(hasKey: boolean): AssistantProviderStatus {
  return hasKey
    ? status('openrouter', { state: 'connected' })
    : status('openrouter', { state: 'signed-out', detail: 'Add your OpenRouter API key in Settings → API keys.' })
}

export async function checkProvider(id: AssistantCliProviderId): Promise<AssistantProviderStatus> {
  const cli = await resolveCli(id)
  if (!cli) return status(id, { state: 'not-installed' })
  const env = cliEnvironment(cli.pathValue)
  const versionRun = await runCliCommand(cli, ['--version'], env, 15000)
  const version = firstVersion(versionRun.stdout || versionRun.stderr)
  if (id === 'claude') {
    const run = await runCliCommand(cli, ['auth', 'status'], env, 20000)
    const parsed = parseClaudeAuthStatus(run.stdout)
    if (!parsed) return status(id, { state: 'unknown', version, detail: 'Claude Code didn’t report its sign-in status. Update it with `claude update`.' })
    if (!parsed.loggedIn) return status(id, { state: 'signed-out', version })
    return status(id, {
      state: 'connected',
      version,
      account: parsed.account,
      plan: parsed.plan,
      detail: parsed.apiKey ? 'Signed in with an API key, so usage is billed to that key rather than a subscription.' : null
    })
  }
  const run = await runCliCommand(cli, ['login', 'status'], env, 20000)
  const parsed = parseCodexLoginStatus(`${run.stdout}\n${run.stderr}`, run.code)
  if (!parsed) return status(id, { state: 'unknown', version, detail: 'Codex didn’t report its sign-in status. Update it with `codex update`.' })
  if (!parsed.loggedIn) return status(id, { state: 'signed-out', version })
  return status(id, {
    state: 'connected',
    version,
    plan: parsed.apiKey ? null : 'ChatGPT',
    detail: parsed.apiKey ? 'Signed in with an API key, so usage is billed to that key rather than a ChatGPT plan.' : null
  })
}

export interface TurnCommandOptions {
  cli: ResolvedCli
  model: string
  resumeId: string | null
  systemPrompt: string
  mcpUrl: string
  cwd: string
}

/** TOML basic string for `codex -c key=value`. */
function tomlString(value: string): string {
  return JSON.stringify(value)
}

/** Claude Code's own tools the assistant may use: reading the web, nothing that runs code or touches files. */
export const CLAUDE_WEB_TOOLS = ['WebSearch', 'WebFetch'] as const

/**
 * Claude Code runs as a BridgeClip assistant, not a coding agent: of its
 * built-in tools only web search and fetch, plus BridgeClip's MCP server; no
 * project settings, hooks or other MCP servers, and anything that would prompt
 * is denied. The bearer token reaches the MCP config through the child's
 * environment, never argv.
 */
export function claudeTurnArgs(options: TurnCommandOptions): string[] {
  const mcpConfig = JSON.stringify({
    mcpServers: {
      [MCP_SERVER_NAME]: { type: 'http', url: options.mcpUrl, headers: { Authorization: `Bearer \${${MCP_TOKEN_ENV}}` } }
    }
  })
  const args = [
    '-p',
    '--output-format', 'stream-json',
    '--verbose',
    '--include-partial-messages',
    '--tools', CLAUDE_WEB_TOOLS.join(','),
    '--mcp-config', mcpConfig,
    '--strict-mcp-config',
    '--allowedTools', [`mcp__${MCP_SERVER_NAME}`, ...CLAUDE_WEB_TOOLS].join(','),
    '--permission-mode', 'dontAsk',
    '--permission-prompts', 'none',
    '--setting-sources', 'user',
    '--settings', JSON.stringify({ disableAllHooks: true }),
    '--disable-slash-commands',
    '--system-prompt', options.systemPrompt
  ]
  if (options.model) args.push('--model', options.model)
  if (options.resumeId) args.push('--resume', options.resumeId)
  return args
}

export function claudeTurnEnv(cli: ResolvedCli, token: string): Record<string, string> {
  return cliEnvironment(cli.pathValue, {
    [MCP_TOKEN_ENV]: token,
    MCP_TOOL_TIMEOUT: String(MCP_TOOL_TIMEOUT_SECONDS * 1000),
    CLAUDE_CODE_DISABLE_TERMINAL_TITLE: '1'
  })
}

/**
 * Codex runs read-only with approvals off, ignoring the user's config.toml so
 * their own MCP servers and profiles don't join a BridgeClip conversation.
 * Live web search is on: it's a hosted tool, outside the read-only sandbox.
 * The system prompt goes in developer_instructions on every turn.
 */
export function codexTurnArgs(options: TurnCommandOptions): string[] {
  const server = `mcp_servers.${MCP_SERVER_NAME}={url=${tomlString(options.mcpUrl)},bearer_token_env_var=${tomlString(MCP_TOKEN_ENV)},tool_timeout_sec=${MCP_TOOL_TIMEOUT_SECONDS},startup_timeout_sec=30}`
  const args = [
    'exec',
    '--ignore-user-config',
    '--json',
    '--skip-git-repo-check',
    '--color', 'never',
    '--cd', options.cwd,
    '-s', 'read-only',
    '-c', 'approval_policy="never"',
    '-c', 'web_search="live"',
    '-c', server,
    '-c', `developer_instructions=${tomlString(options.systemPrompt)}`
  ]
  if (options.model) args.push('-m', options.model)
  if (options.resumeId) args.push('resume', options.resumeId)
  args.push('--', '-')
  return args
}

export function codexTurnEnv(cli: ResolvedCli, token: string): Record<string, string> {
  return cliEnvironment(cli.pathValue, { [MCP_TOKEN_ENV]: token })
}
