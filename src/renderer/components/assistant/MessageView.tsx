import { useMemo, useState } from 'react'
import { Ban, Check, ChevronDown, Loader2, ShieldQuestion, Wrench, X } from 'lucide-react'
import {
  ASSISTANT_PROVIDER_INFO,
  assistantModelLabel,
  type AssistantApprovalRequest,
  type AssistantMessage,
  type AssistantProviderId,
  type AssistantToolCall
} from '../../../shared/assistant'
import { AssistantMarkdown } from './AssistantMarkdown'
import { useModelStore } from '../../store/use-model-store'
import { ProviderLogo } from '../brand/ProviderLogo'
import { Button } from '../ui/Button'
import { Callout } from '../ui/Callout'
import { IconTile } from '../ui/IconTile'
import { cn } from '../../lib/utils'

/** Assistant replies indent their body past the logo: a 28px tile plus a 14px gap. */
export const REPLY_INDENT = 'pl-[42px]'

export function UserMessage({ message }: { message: AssistantMessage }): React.JSX.Element {
  const text = message.parts.map((part) => part.text ?? '').join('')
  return (
    <div className="flex justify-end">
      <div className="max-w-[min(82%,560px)] whitespace-pre-wrap rounded-3xl rounded-br-md bg-white/[0.08] px-4 py-2.5 text-base leading-relaxed text-ink shadow-[inset_0_0_0_1px_rgb(255_255_255/0.06)] [overflow-wrap:anywhere]">
        {text}
      </div>
    </div>
  )
}

type Block = { kind: 'text'; text: string; key: string } | { kind: 'tools'; tools: AssistantToolCall[]; key: string }

/** Runs of tool calls become one group, so a reply reads as text with compact activity between. */
function blocksOf(message: AssistantMessage): Block[] {
  const blocks: Block[] = []
  message.parts.forEach((part, index) => {
    if (part.kind === 'tool' && part.tool) {
      const last = blocks[blocks.length - 1]
      if (last?.kind === 'tools') last.tools.push(part.tool)
      else blocks.push({ kind: 'tools', tools: [part.tool], key: part.tool.id })
    } else if (part.text) blocks.push({ kind: 'text', text: part.text, key: `t${index}` })
  })
  return blocks
}

export function AssistantReply({ message, provider, model, working }: {
  message: AssistantMessage
  /** The chat's provider and model, for replies saved before messages recorded their own. */
  provider: AssistantProviderId
  model: string
  working: boolean
}): React.JSX.Element {
  const by = message.provider ?? provider
  const modelId = message.provider ? message.model ?? '' : model
  const catalogName = useModelStore((state) => by === 'openrouter' ? state.catalog?.assistant.find((model) => model.id === modelId)?.name ?? null : null)
  const modelLabel = assistantModelLabel(by, modelId, catalogName)
  const blocks = blocksOf(message)
  const last = message.parts[message.parts.length - 1]
  const toolBusy = last?.kind === 'tool' && (last.tool?.status === 'running' || last.tool?.status === 'awaiting-approval')
  const thinking = working && !toolBusy && (blocks.length === 0 || last?.kind === 'tool')

  return (
    <article aria-label={`Reply from ${ASSISTANT_PROVIDER_INFO[by].brand}`}>
      <header className="flex items-center gap-3.5">
        <ProviderLogo provider={by} variant="tile" size="sm" />
        <p className="flex min-w-0 items-baseline gap-2 text-xs">
          <span className="font-semibold text-ink">{ASSISTANT_PROVIDER_INFO[by].brand}</span>
          {modelLabel && <span className="truncate text-ink-subtle">{modelLabel}</span>}
        </p>
      </header>
      <div className={cn('mt-1.5 space-y-3', REPLY_INDENT)}>
        {blocks.map((block) => block.kind === 'tools'
          ? <ToolGroup key={block.key} tools={block.tools} />
          : <AssistantMarkdown key={block.key} text={block.text} />)}
        {thinking && (
          <p role="status" className="flex h-6 items-center gap-2.5 text-xs text-ink-subtle">
            <span aria-hidden className="flex gap-1">
              {[0, 160, 320].map((delay) => <span key={delay} className="h-1.5 w-1.5 animate-pulse rounded-full bg-ink-muted" style={{ animationDelay: `${delay}ms` }} />)}
            </span>
            {blocks.length === 0 ? 'Thinking…' : 'Working…'}
          </p>
        )}
        {message.error && <Callout tone={/stopped this reply/.test(message.error) ? 'info' : 'danger'}>{message.error}</Callout>}
      </div>
    </article>
  )
}

const TOOL_STATUS: Record<AssistantToolCall['status'], { icon: React.JSX.Element; tone: string; label: string }> = {
  running: { icon: <Loader2 className="h-3.5 w-3.5 animate-spin" />, tone: 'text-accent', label: 'Running' },
  'awaiting-approval': { icon: <ShieldQuestion className="h-3.5 w-3.5" />, tone: 'text-warning', label: 'Waiting for you' },
  done: { icon: <Check className="h-3.5 w-3.5" strokeWidth={2.6} />, tone: 'text-success', label: 'Done' },
  error: { icon: <X className="h-3.5 w-3.5" strokeWidth={2.6} />, tone: 'text-danger', label: 'Failed' },
  denied: { icon: <Ban className="h-3.5 w-3.5" />, tone: 'text-ink-subtle', label: 'Not approved' }
}

/** The actions a reply took. Three or more fold into one line once they've all finished. */
function ToolGroup({ tools }: { tools: AssistantToolCall[] }): React.JSX.Element {
  const active = tools.some((tool) => tool.status === 'running' || tool.status === 'awaiting-approval')
  const failed = tools.filter((tool) => tool.status === 'error').length
  const foldable = tools.length > 2
  const [open, setOpen] = useState(false)
  const expanded = !foldable || open || active

  return (
    <div className="glass-tile overflow-hidden rounded-2xl">
      {foldable && (
        <button
          type="button"
          aria-expanded={expanded}
          disabled={active}
          onClick={() => setOpen(!open)}
          className="flex h-9 w-full items-center gap-2.5 px-3 text-left text-xs text-ink-muted transition-colors duration-150 enabled:hover:bg-white/[0.04] enabled:hover:text-ink disabled:cursor-default"
        >
          <Wrench aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-subtle" />
          <span className="font-medium">{active ? `Working through ${tools.length} actions` : `Took ${tools.length} actions`}</span>
          {failed > 0 && <span className="text-danger">· {failed} failed</span>}
          {!active && <ChevronDown aria-hidden className={cn('ml-auto h-3.5 w-3.5 text-ink-subtle transition-transform duration-150', expanded && 'rotate-180')} />}
        </button>
      )}
      {expanded && (
        <ul className={cn('divide-y divide-white/[0.05]', foldable && 'border-t border-white/[0.06]')}>
          {tools.map((tool) => {
            const status = TOOL_STATUS[tool.status]
            return (
              <li key={tool.id} className="flex items-start gap-2.5 px-3 py-2">
                <span role="img" aria-label={status.label} className={cn('mt-px shrink-0', status.tone)}>{status.icon}</span>
                <span className="min-w-0 flex-1 text-xs leading-[18px]">
                  <span className="text-ink">{tool.title}</span>
                  {tool.summary && <span className={cn('block [overflow-wrap:anywhere]', tool.status === 'error' ? 'text-danger' : 'text-ink-subtle')}>{tool.summary}</span>}
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

export function ApprovalCard({ request, onAnswer }: { request: AssistantApprovalRequest; onAnswer: (allowed: boolean) => void }): React.JSX.Element {
  const [answered, setAnswered] = useState(false)
  const details = useMemo(() => request.details.filter(Boolean), [request.details])
  const answer = (allowed: boolean): void => {
    if (answered) return
    setAnswered(true)
    onAnswer(allowed)
  }
  return (
    <div className={REPLY_INDENT}>
      <div role="group" aria-label={`Approve: ${request.title}`} className="glass rounded-2xl p-4 shadow-[inset_0_0_0_1px_rgb(var(--warning)/0.35)]">
        <div className="flex items-start gap-3">
          <IconTile tone="warning" size="sm"><ShieldQuestion /></IconTile>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold text-ink">{request.title}?</p>
            <ul className="mt-1 space-y-0.5 text-xs text-ink-muted [overflow-wrap:anywhere]">
              {details.map((line, index) => <li key={index}>{line}</li>)}
            </ul>
            <div className="mt-3 flex gap-2">
              <Button variant="primary" size="sm" onClick={() => answer(true)} disabled={answered}>Allow</Button>
              <Button variant="ghost" size="sm" onClick={() => answer(false)} disabled={answered}>Don’t allow</Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
