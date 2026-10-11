import { constants, realpathSync, statSync } from 'fs'
import { open } from 'fs/promises'
import { join } from 'path'
import { assertAbsolutePath, isWithinDirectory } from './security'
import { parseEditAudit, type EditAudit } from '../shared/editorial'

const missing = (error: unknown): boolean => (error as NodeJS.ErrnoException | null)?.code === 'ENOENT'

async function readRecordedJson(run: string, name: string): Promise<unknown> {
  const path = join(run, name)
  const handle = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0) | (constants.O_NONBLOCK ?? 0))
  try {
    const stat = await handle.stat(), current = statSync(path)
    if (!stat.isFile() || stat.size > 32 * 1024 * 1024 || stat.dev !== current.dev || stat.ino !== current.ino || !isWithinDirectory(path, run)) throw new Error('Invalid edit trace')
    const bytes = Buffer.alloc(stat.size + 1)
    let used = 0
    while (used < bytes.length) {
      const { bytesRead } = await handle.read(bytes, used, bytes.length - used, null)
      if (!bytesRead) break
      used += bytesRead
    }
    if (used !== stat.size) throw new Error('Edit trace changed while reading')
    return JSON.parse(bytes.subarray(0, used).toString('utf8'))
  } finally { await handle.close() }
}

/** Every error here reaches the renderer, so none may carry a file system path. */
export async function inspectEdits(outputDir: string, libraryDir: string): Promise<EditAudit> {
  assertAbsolutePath(outputDir)
  if (!isWithinDirectory(outputDir, libraryDir)) throw new Error('Run is outside the library')
  let run: string, library: string
  try { run = realpathSync(outputDir); library = realpathSync(libraryDir) }
  catch { throw new Error('This run is no longer available in the Library.') }
  if (!isWithinDirectory(run, library) || run === library) throw new Error('Invalid run folder')
  try { return parseEditAudit(await readRecordedJson(run, 'edit_audit.json')) }
  catch (error) {
    if (!missing(error)) throw new Error('The saved edit trace is invalid or unsupported.')
  }
  let transcript: { segments?: { start_time_ms: number; end_time_ms: number; text: string }[] }
  try { transcript = await readRecordedJson(run, 'transcript.json') as typeof transcript }
  catch (error) {
    if (missing(error)) throw new Error('No transcript saved for this run. Only runs that saved a transcript or edit trace have details.')
    throw new Error('The saved transcript could not be read.')
  }
  if (!Array.isArray(transcript?.segments) || transcript.segments.length > 100000) throw new Error('Invalid saved transcript')
  try {
    return parseEditAudit({ version: 1, title: 'Recorded transcript', duration_ms: transcript.segments.reduce((end, s) => Math.max(end, s.end_time_ms), 0),
      preferred_range: [null, null], outcome: 'legacy_transcript_only', planner: { requests: [] }, candidates: [],
      transcript: transcript.segments.map(s => ({ start_ms: s.start_time_ms, end_ms: s.end_time_ms, text: s.text })) })
  } catch { throw new Error('Invalid saved transcript') }
}
