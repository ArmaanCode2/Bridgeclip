import { useMemo, useState } from 'react'
import { Loader2, Search, SquarePen, Trash2 } from 'lucide-react'
import type { AssistantConversationSummary } from '../../../shared/assistant'
import { ProviderLogo } from '../brand/ProviderLogo'
import { Button } from '../ui/Button'
import { TextInput } from '../ui/Field'
import { cn } from '../../lib/utils'

const DAY_MS = 24 * 60 * 60 * 1000

function dayGroup(iso: string, today: number): string {
  const at = new Date(iso).getTime()
  if (at >= today) return 'Today'
  if (at >= today - DAY_MS) return 'Yesterday'
  if (at >= today - 7 * DAY_MS) return 'Previous 7 days'
  if (at >= today - 30 * DAY_MS) return 'Previous 30 days'
  return 'Older'
}

/** Recent chats beside the conversation, newest first, grouped by day. */
export function ChatList({ conversations, activeId, running, waiting, onOpen, onNew, onDelete, className }: {
  conversations: AssistantConversationSummary[]
  activeId: string | null
  running: string[]
  /** Chats with an approval waiting. */
  waiting: Set<string>
  onOpen: (id: string) => void
  onNew: () => void
  onDelete: (chat: AssistantConversationSummary) => void
  className?: string
}): React.JSX.Element {
  const [query, setQuery] = useState('')
  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const now = new Date()
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
    const result: { label: string; chats: AssistantConversationSummary[] }[] = []
    for (const chat of conversations) {
      if (needle && !chat.title.toLowerCase().includes(needle)) continue
      const label = dayGroup(chat.updatedAt, today)
      const group = result[result.length - 1]
      if (group?.label === label) group.chats.push(chat)
      else result.push({ label, chats: [chat] })
    }
    return result
  }, [conversations, query])

  return (
    <aside aria-label="Chats" className={cn('w-[248px] shrink-0 flex-col border-r border-white/[0.06]', className)}>
      <div className="flex items-center justify-between gap-2 pb-2 pl-4 pr-2.5 pt-2.5">
        <h2 className="text-sm font-semibold text-ink">Chats</h2>
        <Button variant="ghost" size="sm" iconOnly aria-label="New chat" tooltip="New chat" icon={<SquarePen className="h-3.5 w-3.5" />} onClick={onNew} disabled={activeId === null} />
      </div>
      {conversations.length > 0 && (
        <div className="px-3 pb-1">
          <TextInput inputSize="sm" type="search" aria-label="Search chats" placeholder="Search chats" leading={<Search className="h-3.5 w-3.5" />} value={query} onChange={(event) => setQuery(event.target.value)} />
        </div>
      )}
      <nav aria-label="Recent chats" className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {conversations.length === 0 && <p className="px-2 pt-3 text-xs text-ink-subtle">Your chats show up here.</p>}
        {conversations.length > 0 && groups.length === 0 && <p className="px-2 pt-3 text-xs text-ink-subtle">No chats match “{query.trim()}”.</p>}
        {groups.map((group) => (
          <section key={group.label} aria-label={group.label}>
            <p className="px-2 pb-1 pt-3 text-2xs font-medium text-ink-subtle">{group.label}</p>
            <ul className="space-y-px">
              {group.chats.map((chat) => {
                const active = chat.id === activeId
                const replying = running.includes(chat.id)
                return (
                  <li key={chat.id} className="group/chat relative">
                    <button
                      type="button"
                      aria-current={active ? 'true' : undefined}
                      onClick={() => onOpen(chat.id)}
                      className={cn(
                        'flex h-8 w-full items-center gap-2.5 rounded-lg pl-2.5 pr-8 text-left text-xs transition-colors duration-150',
                        active ? 'bg-white/[0.09] text-ink' : 'text-ink-muted hover:bg-white/[0.05] hover:text-ink'
                      )}
                    >
                      <ProviderLogo provider={chat.provider} className="h-3.5 w-3.5 shrink-0" />
                      <span className="min-w-0 flex-1 truncate">{chat.title}</span>
                    </button>
                    {/* A chat that's replying can't be deleted, so its status takes the delete button's place. */}
                    {waiting.has(chat.id) ? (
                      <span role="img" aria-label="Waiting for your approval" className="pointer-events-none absolute right-3 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-warning" />
                    ) : replying ? (
                      // The wrapper centres the spinner and the icon spins. animate-spin's keyframe
                      // sets `transform`, so on one element it would replace the centring translate.
                      <span role="img" aria-label="Replying" className="pointer-events-none absolute right-2.5 top-1/2 flex -translate-y-1/2">
                        <Loader2 aria-hidden className="h-3.5 w-3.5 animate-spin text-accent" />
                      </span>
                    ) : (
                      <button
                        type="button"
                        aria-label={`Delete “${chat.title}”`}
                        title="Delete chat"
                        onClick={() => onDelete(chat)}
                        className="absolute right-1 top-1/2 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-md text-ink-subtle opacity-0 transition-opacity duration-150 hover:bg-white/[0.08] hover:text-danger focus-visible:opacity-100 group-hover/chat:opacity-100"
                      >
                        <Trash2 aria-hidden className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </li>
                )
              })}
            </ul>
          </section>
        ))}
      </nav>
    </aside>
  )
}
