import { createHash } from 'crypto'
import { chmodSync, existsSync, lstatSync, readdirSync, renameSync, rmSync } from 'fs'
import { basename, dirname, join } from 'path'

const QUARANTINE_MS = 30 * 24 * 60 * 60_000

/** A non-reversible cache label. Never persist the provider key itself. */
export function workspaceId(key: string): string {
  return createHash('sha256').update(key).digest('hex')
}

export function readableCache(path: string, maxBytes: number): boolean {
  try {
    const file = lstatSync(path)
    return file.isFile() && file.size <= maxBytes
  } catch { return false }
}

/** Preserve unbound legacy data privately for 30 days, without showing it in another workspace. */
export function quarantineUnbound(path: string): void {
  const prefix = `${basename(path)}.quarantine-`
  try {
    for (const name of readdirSync(dirname(path))) {
      if (!name.startsWith(prefix)) continue
      const at = Number(name.slice(prefix.length))
      if (Number.isFinite(at) && Date.now() - at > QUARANTINE_MS) {
        // The cache is deliberately retained only for the stated migration window.
        rmSync(join(dirname(path), name), { force: true })
      }
    }
    if (existsSync(path)) {
      if (lstatSync(path).isFile()) chmodSync(path, 0o600)
      renameSync(path, `${path}.quarantine-${Date.now()}`)
    }
  } catch { /* A failed quarantine must never make unbound data visible. */ }
}
