import { mkdirSync } from 'fs'
import { join } from 'path'
import { app } from 'electron'
import { cliEnvironment, resolveCliNow, type ResolvedCli } from '../assistant/cli-process'
import type { AiProvider } from '../../shared/providers'

/** The file in a job's folder listing the Claude Code runs it started, one id per line. The engine adds to it. */
export const SESSIONS_FILE = 'claude-sessions.txt'

/** What the engine needs to start Claude Code (engine/clip_engine/services/claude_plan.py reads this). */
export interface ClaudePlanRunner {
  command: string[]
  env: Record<string, string>
  cwd: string
  /** Where the engine lists each run, so deleting the job can clean up after it. */
  sessions?: string
}

/** The empty folder every Claude plan run starts in, so no project's files are loaded into it. */
export function claudePlanWorkspacePath(root = app.getPath('userData')): string {
  return join(root, 'claude-plan', 'workspace')
}

export function claudePlanWorkspace(root?: string): string {
  const folder = claudePlanWorkspacePath(root)
  mkdirSync(folder, { recursive: true, mode: 0o700 })
  return folder
}

/** The environment comes from an allowlist, so an API key on this computer cannot move a run off the plan. */
export function claudePlanRunner(cli: ResolvedCli, cwd: string): ClaudePlanRunner {
  return { command: [cli.command, ...cli.prefixArgs], env: cliEnvironment(cli.pathValue), cwd }
}

/** What a job or editor run adds to the engine's environment. Nothing unless it runs on the Claude plan. */
export function claudePlanEnvironment(
  aiProvider: AiProvider, deps: { find?: () => ResolvedCli | null; root?: string; runFolder?: string } = {}
): Record<string, string> {
  if (aiProvider !== 'claude-plan') return {}
  const cli = (deps.find ?? (() => resolveCliNow('claude')))()
  if (!cli) return {}
  const runner = claudePlanRunner(cli, claudePlanWorkspace(deps.root))
  return { CLAUDE_PLAN_RUNNER: JSON.stringify(deps.runFolder ? { ...runner, sessions: join(deps.runFolder, SESSIONS_FILE) } : runner) }
}
