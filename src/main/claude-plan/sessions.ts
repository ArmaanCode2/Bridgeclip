import { lstatSync, readFileSync, readdirSync, realpathSync, rmSync } from 'fs'
import { homedir } from 'os'
import { basename, join } from 'path'
import { logger } from '../logger'
import { SESSIONS_FILE, claudePlanWorkspacePath } from './runner'

/**
 * Removes what a deleted job left in Claude Code's own folder. Runs are started with
 * --no-session-persistence, so normally nothing is there: this is the safety net for a Claude Code
 * update that changes that, as far as its layout stays the same. It deletes outside the app's
 * folders, so it stays narrow: only ids from the job's own list, only from the one real folder
 * Claude Code keeps for BridgeClip's workspace. A damaged list cannot reach anything else there
 * but sessions started in that workspace, and nothing outside it.
 */

const SESSION_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
/** Far more than a job's runs, which fit in about 40 KB. */
const MAX_LIST_BYTES = 256 * 1024

const configFolder = (): string => process.env.CLAUDE_CONFIG_DIR || join(homedir(), '.claude')

/** Where Claude Code keeps sessions started in `workspace`: the path with everything else turned into dashes. */
export function claudeProjectFolder(workspace: string, config = configFolder()): string {
  return join(config, 'projects', workspace.replace(/[^A-Za-z0-9]/g, '-'))
}

/** The well-formed ids a job listed, once each. A list that is not a plain small file gives none. */
export function readSessionIds(runFolder: string): string[] {
  const file = join(runFolder, SESSIONS_FILE)
  try {
    const info = lstatSync(file)
    if (!info.isFile() || info.size > MAX_LIST_BYTES) return []
    return [...new Set(readFileSync(file, 'utf8').split('\n').map((line) => line.trim()).filter((line) => SESSION_ID.test(line)))]
  } catch {
    return []
  }
}

/** One session's transcript or folder. Links are left alone. */
function removePath(path: string): boolean {
  try {
    const info = lstatSync(path)
    if (info.isSymbolicLink()) return false
    rmSync(path, { recursive: info.isDirectory() })
    return true
  } catch (error) {
    logger.warn('claude_plan.session.not_removed', { code: (error as NodeJS.ErrnoException).code ?? 'unknown' })
    return false
  }
}

/**
 * Removes these sessions from BridgeClip's folder, and returns how many were removed. One look at
 * the folder decides everything, so a folder that is missing or out of reach costs one call however
 * long the list is. A link in its place could lead to another project's sessions, so it counts as missing.
 */
export function removeClaudeSessions(ids: readonly string[], workspace: string, config = configFolder()): number {
  const folder = claudeProjectFolder(workspace, config)
  let names: Set<string>
  try {
    if (!lstatSync(folder).isDirectory() || realpathSync(folder) !== join(realpathSync(join(config, 'projects')), basename(folder))) return 0
    names = new Set(readdirSync(folder))
  } catch {
    return 0
  }
  let removed = 0
  for (const id of ids) {
    if (!SESSION_ID.test(id)) continue
    const found = [`${id}.jsonl`, id].filter((name) => names.has(name)).map((name) => removePath(join(folder, name)))
    if (found.some(Boolean)) removed++
  }
  return removed
}

/** The cleanup for a run being deleted. Nothing here may stop the deletion. */
export function removeRunSessions(runFolder: string): void {
  try {
    const ids = readSessionIds(runFolder)
    if (ids.length === 0) return
    // Claude Code names its folder after the real path of the workspace.
    const removed = removeClaudeSessions(ids, realpathSync(claudePlanWorkspacePath()))
    if (removed > 0) logger.info('claude_plan.sessions.removed', { removed })
  } catch {
    logger.warn('claude_plan.sessions.cleanup_failed', {})
  }
}
