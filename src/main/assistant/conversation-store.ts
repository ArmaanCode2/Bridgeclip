import { randomUUID } from 'crypto'
import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, readdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from 'fs'
import { join } from 'path'
import {
  ASSISTANT_PROVIDERS,
  DEFAULT_ASSISTANT_PREFERENCES,
  isAssistantModel,
  isAssistantProvider,
  type AssistantConversation,
  type AssistantConversationSummary,
  type AssistantPreferences
} from '../../shared/assistant'

// Conversations are plain JSON files under <userData>/assistant. They hold the
// chat as shown in BridgeClip plus the CLI's resume id; the CLI keeps its own
// transcript in the user's ~/.claude or ~/.codex as it does for any session.

const MAX_CONVERSATIONS = 50
const ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/

export function isConversationId(value: unknown): value is string {
  return typeof value === 'string' && ID_PATTERN.test(value)
}

function writeJsonAtomic(path: string, value: unknown): void {
  const tempPath = `${path}.${randomUUID()}.tmp`
  let fd: number | undefined
  try {
    fd = openSync(tempPath, 'wx', 0o600)
    writeFileSync(fd, JSON.stringify(value), { encoding: 'utf-8' })
    fsyncSync(fd)
    closeSync(fd)
    fd = undefined
    renameSync(tempPath, path)
  } finally {
    if (fd !== undefined) closeSync(fd)
    if (existsSync(tempPath)) unlinkSync(tempPath)
  }
}

export class ConversationStore {
  private readonly directory: string
  private readonly preferencesPath: string

  constructor(root: string) {
    this.directory = join(root, 'conversations')
    this.preferencesPath = join(root, 'preferences.json')
    mkdirSync(this.directory, { recursive: true, mode: 0o700 })
  }

  private pathFor(id: string): string {
    if (!isConversationId(id)) throw new Error('Invalid conversation id')
    return join(this.directory, `${id}.json`)
  }

  get(id: string): AssistantConversation | null {
    try {
      const parsed = JSON.parse(readFileSync(this.pathFor(id), 'utf8')) as AssistantConversation
      if (parsed?.id !== id || !isAssistantProvider(parsed.provider) || !Array.isArray(parsed.messages)) return null
      return parsed
    } catch {
      return null
    }
  }

  save(conversation: AssistantConversation): void {
    writeJsonAtomic(this.pathFor(conversation.id), conversation)
  }

  delete(id: string): void {
    const path = this.pathFor(id)
    if (existsSync(path)) unlinkSync(path)
  }

  list(): AssistantConversationSummary[] {
    let files: string[] = []
    try {
      files = readdirSync(this.directory).filter((name) => name.endsWith('.json'))
    } catch {
      return []
    }
    const summaries: AssistantConversationSummary[] = []
    for (const file of files) {
      const conversation = this.get(file.slice(0, -'.json'.length))
      if (conversation) {
        summaries.push({ id: conversation.id, title: conversation.title, provider: conversation.provider, updatedAt: conversation.updatedAt })
      }
    }
    summaries.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    for (const stale of summaries.splice(MAX_CONVERSATIONS)) {
      try { this.delete(stale.id) } catch { /* keep going */ }
    }
    return summaries
  }

  preferences(): AssistantPreferences {
    try {
      const raw = JSON.parse(readFileSync(this.preferencesPath, 'utf8')) as Record<string, unknown>
      const models = { ...DEFAULT_ASSISTANT_PREFERENCES.models }
      const savedModels = typeof raw.models === 'object' && raw.models !== null ? raw.models as Record<string, unknown> : {}
      for (const provider of ASSISTANT_PROVIDERS) {
        if (isAssistantModel(provider, savedModels[provider])) models[provider] = savedModels[provider] as string
      }
      return { provider: isAssistantProvider(raw.provider) ? raw.provider : null, models }
    } catch {
      return { ...DEFAULT_ASSISTANT_PREFERENCES, models: { ...DEFAULT_ASSISTANT_PREFERENCES.models } }
    }
  }

  savePreferences(preferences: AssistantPreferences): AssistantPreferences {
    writeJsonAtomic(this.preferencesPath, preferences)
    return this.preferences()
  }
}
