import { useEffect, useState } from 'react'
import { Check, Copy, ExternalLink, KeyRound, LogIn, RefreshCw, Sparkles } from 'lucide-react'
import {
  ASSISTANT_CLI_PROVIDERS,
  ASSISTANT_MODELS,
  ASSISTANT_PROVIDER_INFO,
  type AssistantCliProviderId,
  type AssistantProviderStatus,
  type AssistantSignInState
} from '../../../shared/assistant'
import { chosenProvider, useAssistantStore } from '../../store/use-assistant-store'
import { useModelStore } from '../../store/use-model-store'
import { getApi } from '../../lib/ipc'
import { ProviderLogo } from '../brand/ProviderLogo'
import { PanelHeader } from '../ui/Panel'
import { Button } from '../ui/Button'
import { Badge } from '../ui/Badge'
import { IconTile } from '../ui/IconTile'
import { Select } from '../ui/Select'
import { TextInput } from '../ui/Field'
import { Callout } from '../ui/Callout'
import { cn } from '../../lib/utils'

/** Settings → Assistant: connect Claude Code or Codex with the user's own subscription, or use an OpenRouter key. */
export function AssistantSettings(): React.JSX.Element {
  const store = useAssistantStore()
  const { statuses, checking, preferences, signIn, error } = store
  const preferred = chosenProvider(store)

  useEffect(() => {
    void useAssistantStore.getState().init()
    void useAssistantStore.getState().refreshStatus(true)
  }, [])

  return (
    <>
      <PanelHeader
        icon={<IconTile tone="accent"><Sparkles /></IconTile>}
        title="Assistant"
        description="Chat with BridgeClip using your Claude or ChatGPT subscription, or any OpenRouter model. BridgeClip runs Claude Code or Codex on this computer (your sign-in stays with them), or calls OpenRouter with your API key."
        action={
          <Button size="sm" variant="ghost" icon={<RefreshCw className={cn('h-3.5 w-3.5', checking && 'animate-spin')} />} onClick={() => void store.refreshStatus(true)} disabled={checking}>
            Check again
          </Button>
        }
      />
      <div className="mt-4 space-y-2">
        {ASSISTANT_CLI_PROVIDERS.map((id) => (
          <ProviderRow
            key={id}
            id={id}
            status={statuses[id]}
            signIn={signIn[id]}
            preferred={preferred === id}
            model={preferences.models[id] ?? ''}
            onPrefer={() => void store.setPreferences({ provider: id })}
            onModel={(model) => void store.setPreferences({ models: { ...preferences.models, [id]: model } })}
          />
        ))}
        <OpenRouterRow
          status={statuses.openrouter}
          preferred={preferred === 'openrouter'}
          model={preferences.models.openrouter ?? ''}
          onPrefer={() => void store.setPreferences({ provider: 'openrouter' })}
          onModel={(model) => void store.setPreferences({ models: { ...preferences.models, openrouter: model } })}
        />
      </div>
      {error && <Callout tone="danger" className="mt-3" onDismiss={store.clearError}>{error}</Callout>}
      <p className="mt-3 px-1 text-2xs text-ink-subtle">
        The assistant gets BridgeClip’s own tools plus web search and page reading (to find videos and look things up): no shell or files. Anything that posts, deletes or uses your OpenRouter credit for clipping asks you in the chat first. Chatting on an OpenRouter model, and its web searches, are billed to your key.
      </p>
    </>
  )
}

function statusBadge(status: AssistantProviderStatus | null): React.JSX.Element {
  if (!status) return <Badge>Checking…</Badge>
  switch (status.state) {
    case 'connected': return <Badge tone="success">Connected{status.plan ? ` · ${status.plan[0].toUpperCase()}${status.plan.slice(1)}` : ''}</Badge>
    case 'signed-out': return <Badge tone="warning">Not signed in</Badge>
    case 'not-installed': return <Badge>Not installed</Badge>
    default: return <Badge tone="warning">Unknown</Badge>
  }
}

/** OpenRouter has no sign-in: it uses the OpenRouter key from API keys above, and any tool-calling model. */
function OpenRouterRow({ status, preferred, model, onPrefer, onModel }: {
  status: AssistantProviderStatus | null
  preferred: boolean
  model: string
  onPrefer: () => void
  onModel: (model: string) => void
}): React.JSX.Element {
  const { catalog, loading, error, load } = useModelStore()
  const connected = status?.state === 'connected'
  useEffect(() => { if (connected) void load() }, [connected, load])
  const models = catalog?.assistant ?? []
  const options = models.map((option) => ({ value: option.id, label: option.name }))
  // Keep a saved model choosable while the catalog loads or after it leaves the list.
  if (model && !models.some((option) => option.id === model)) options.unshift({ value: model, label: model })

  return (
    <div className={cn('glass-tile rounded-2xl px-3.5 py-3', preferred && connected && model && 'glass-selected')}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <ProviderLogo provider="openrouter" variant="tile" size="md" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-ink">OpenRouter</span>
            {!status ? <Badge>Checking…</Badge> : connected ? <Badge tone="success">Key added</Badge> : <Badge tone="warning">No API key</Badge>}
          </div>
          <p className="mt-0.5 truncate text-xs text-ink-muted">Any model that can use tools, billed per message to your OpenRouter key</p>
        </div>
        {connected ? (
          <div className="flex items-center gap-2">
            <Select
              size="sm"
              searchable
              searchPlaceholder="Search models"
              aria-label="OpenRouter model"
              className="w-[220px]"
              value={model}
              placeholder={loading && !models.length ? 'Loading models…' : 'Choose a model'}
              emptyText={error ?? 'No models'}
              options={options}
              onChange={onModel}
            />
            {preferred && model
              ? <Badge tone="accent" icon={<Check className="h-3 w-3" />}>Used for chat</Badge>
              : <Button size="sm" onClick={onPrefer} disabled={!model} tooltip={model ? undefined : 'Choose a model first'}>Use for chat</Button>}
          </div>
        ) : status && (
          <Button size="sm" icon={<KeyRound className="h-3.5 w-3.5" />} onClick={() => document.getElementById('settings-keys')?.scrollIntoView({ behavior: 'smooth', block: 'start' })}>
            Add key
          </Button>
        )}
      </div>
    </div>
  )
}

function ProviderRow({ id, status, signIn, preferred, model, onPrefer, onModel }: {
  id: AssistantCliProviderId
  status: AssistantProviderStatus | null
  signIn: AssistantSignInState | null
  preferred: boolean
  model: string
  onPrefer: () => void
  onModel: (model: string) => void
}): React.JSX.Element {
  const info = ASSISTANT_PROVIDER_INFO[id]
  const store = useAssistantStore()
  const [copied, setCopied] = useState<'install' | 'signIn' | null>(null)
  const [code, setCode] = useState('')
  const waiting = signIn?.status === 'waiting'
  const copy = async (which: 'install' | 'signIn'): Promise<void> => {
    if (await getApi().assistant.copyCommand(id, which).catch(() => false)) {
      setCopied(which)
      window.setTimeout(() => setCopied(null), 1600)
    }
  }

  return (
    <div className={cn('glass-tile rounded-2xl px-3.5 py-3', preferred && status?.state === 'connected' && 'glass-selected')}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <ProviderLogo provider={id} variant="tile" size="md" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-ink">{info.name}</span>
            {statusBadge(status)}
            {status?.version && <span className="font-mono text-2xs tabular text-ink-faint">v{status.version}</span>}
          </div>
          <p className="mt-0.5 truncate text-xs text-ink-muted">
            {status?.state === 'connected' && status.account ? status.account : `Uses your ${info.subscription} subscription`}
          </p>
        </div>
        {status?.state === 'connected' && (
          <div className="flex items-center gap-2">
            <Select size="sm" aria-label={`${info.name} model`} className="w-[140px]" value={model}
              options={ASSISTANT_MODELS[id].map((option) => ({ value: option.id, label: option.label }))} onChange={onModel} />
            {preferred
              ? <Badge tone="accent" icon={<Check className="h-3 w-3" />}>Used for chat</Badge>
              : <Button size="sm" onClick={onPrefer}>Use for chat</Button>}
          </div>
        )}
        {status?.state === 'signed-out' && !waiting && (
          <Button size="sm" variant="primary" icon={<LogIn className="h-3.5 w-3.5" />} onClick={() => void store.startSignIn(id)}>Sign in</Button>
        )}
      </div>

      {status?.detail && <p className="mt-2 text-2xs text-warning">{status.detail}</p>}

      {status?.state === 'not-installed' && (
        <div className="mt-2.5 space-y-1.5">
          <p className="text-xs text-ink-muted">Install {info.name} in Terminal, then come back and choose Check again:</p>
          <div className="glass-well flex items-center gap-2 rounded-xl py-1 pl-3 pr-1">
            <code className="min-w-0 flex-1 truncate font-mono text-xs text-ink">{info.installCommand}</code>
            <Button size="sm" variant="ghost" icon={copied === 'install' ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />} onClick={() => void copy('install')}>
              {copied === 'install' ? 'Copied' : 'Copy'}
            </Button>
          </div>
        </div>
      )}

      {waiting && signIn && (
        <div className="mt-2.5 space-y-2">
          <p className="text-xs text-ink-muted">{signIn.message ?? 'Finish signing in in your browser.'}</p>
          <div className="flex flex-wrap gap-2">
            {signIn.url && (
              <Button size="sm" icon={<ExternalLink className="h-3.5 w-3.5" />} onClick={() => void getApi().assistant.signIn.openPage(id)}>Open sign-in page</Button>
            )}
            <Button size="sm" variant="ghost" onClick={() => void store.cancelSignIn(id)}>Cancel</Button>
          </div>
          {signIn.acceptsCode && (
            <form
              className="flex items-center gap-2"
              onSubmit={(event) => { event.preventDefault(); if (code.trim()) { void store.submitSignInCode(id, code.trim()); setCode('') } }}
            >
              <TextInput inputSize="sm" mono aria-label="Code from the browser" placeholder="If the browser shows a code, paste it here" value={code} onChange={(event) => setCode(event.target.value)} className="flex-1" />
              <Button size="sm" type="submit" disabled={!code.trim()}>Submit code</Button>
            </form>
          )}
        </div>
      )}

      {signIn?.status === 'failed' && <p className="mt-2 text-2xs text-danger">{signIn.message}</p>}

      {status?.state === 'signed-out' && !waiting && (
        <p className="mt-2 text-2xs text-ink-subtle">
          Or run <button type="button" className="font-mono text-ink-muted underline decoration-white/20 underline-offset-2 hover:text-ink" onClick={() => void copy('signIn')}>{info.signInCommand}</button> in Terminal{copied === 'signIn' ? ' (copied)' : ''}.
        </p>
      )}
    </div>
  )
}
