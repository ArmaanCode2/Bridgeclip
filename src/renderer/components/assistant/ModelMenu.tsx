import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { Check, ChevronDown, Loader2, RefreshCw, Search, Sparkles } from 'lucide-react'
import {
  ASSISTANT_MODELS,
  ASSISTANT_PROVIDER_INFO,
  ASSISTANT_PROVIDERS,
  OPENROUTER_ASSISTANT_SUGGESTIONS,
  assistantModelName,
  type AssistantProviderId,
  type AssistantProviderStatus
} from '../../../shared/assistant'
import { searchModels, type OpenRouterModel } from '../../../shared/openrouter-models'
import { useModelStore } from '../../store/use-model-store'
import { ProviderLogo } from '../brand/ProviderLogo'
import { MENU_SURFACE } from '../ui/Select'
import { cn } from '../../lib/utils'

const GAP = 6
const EDGE = 8
/** Rows rendered at once from OpenRouter's catalog; searching narrows the rest. */
const MAX_ROUTER_ROWS = 150

function statusText(id: AssistantProviderId, status: AssistantProviderStatus | null): string {
  if (!status) return 'Checking…'
  if (id === 'openrouter') return status.state === 'connected' ? 'Your API key' : 'No API key yet'
  switch (status.state) {
    case 'connected': return status.plan ? `${status.plan[0].toUpperCase()}${status.plan.slice(1)}` : 'Connected'
    case 'signed-out': return 'Not signed in'
    case 'not-installed': return 'Not installed'
    default: return 'Unavailable'
  }
}

/** Per million tokens, input / output, as OpenRouter lists them. */
function price(model: OpenRouterModel): string | null {
  if (model.inputPrice === null || model.outputPrice === null) return null
  if (model.inputPrice === 0 && model.outputPrice === 0) return 'Free'
  const money = (perToken: number): string => {
    const perMillion = perToken * 1e6
    return `$${perMillion >= 10 || Number.isInteger(perMillion) ? perMillion.toFixed(0) : perMillion.toFixed(2)}`
  }
  return `${money(model.inputPrice)} / ${money(model.outputPrice)}`
}

/** The provider a chat should open on: the chosen one, else the first connected. */
function initialTab(provider: AssistantProviderId | null, statuses: Record<AssistantProviderId, AssistantProviderStatus | null>): AssistantProviderId {
  return provider ?? ASSISTANT_PROVIDERS.find((id) => statuses[id]?.state === 'connected') ?? ASSISTANT_PROVIDERS[0]
}

/**
 * The chat's model picker: a tab per provider (Claude, OpenAI, OpenRouter)
 * under its logo, each listing its models. Choosing a model of another
 * provider switches to it. OpenRouter's tab searches every tool-calling model
 * in its catalog. Providers that aren't connected show their models disabled
 * with a way to connect.
 */
export function ModelMenu({ provider, models, statuses, onChoose, onConnect }: {
  provider: AssistantProviderId | null
  models: Record<AssistantProviderId, string>
  statuses: Record<AssistantProviderId, AssistantProviderStatus | null>
  onChoose: (provider: AssistantProviderId, model: string) => void
  onConnect: (provider: AssistantProviderId) => void
}): React.JSX.Element {
  const id = useId()
  const trigger = useRef<HTMLButtonElement>(null)
  const popover = useRef<HTMLDivElement>(null)
  const search = useRef<HTMLInputElement>(null)
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState<AssistantProviderId>(() => initialTab(provider, statuses))
  const [query, setQuery] = useState('')
  const { catalog, loading, error: catalogError, load } = useModelStore()
  const routerModels = useMemo(() => catalog?.assistant ?? [], [catalog])
  const routerName = (modelId: string): string | null => routerModels.find((model) => model.id === modelId)?.name ?? null

  const chosenModel = provider ? models[provider] ?? '' : ''
  const current = !provider
    ? 'Choose a model'
    : provider === 'openrouter' && !chosenModel
      ? 'OpenRouter · choose a model'
      : assistantModelName(provider, chosenModel, provider === 'openrouter' ? routerName(chosenModel) : null)

  const close = useCallback((restoreFocus = false) => {
    setOpen(false)
    if (restoreFocus) trigger.current?.focus({ preventScroll: true })
  }, [])
  const openMenu = (): void => {
    setTab(initialTab(provider, statuses))
    setQuery('')
    setOpen(true)
  }

  // OpenRouter's catalog loads when its tab is shown or its model names the chip (cached for 10 minutes).
  useEffect(() => {
    if ((open && tab === 'openrouter') || provider === 'openrouter') void load()
  }, [open, tab, provider, load])

  const routerRows = useMemo(() => {
    const matches = query.trim() ? searchModels(routerModels, query) : routerModels
    const suggested = query.trim() ? [] : OPENROUTER_ASSISTANT_SUGGESTIONS
      .map((suggestion) => routerModels.find((model) => model.id === suggestion))
      .filter((model): model is OpenRouterModel => Boolean(model))
    return { suggested, matches: matches.slice(0, MAX_ROUTER_ROWS), total: matches.length }
  }, [routerModels, query])

  // Open above the trigger when it sits low in the window, as the docked composer does.
  const place = useCallback(() => {
    const element = popover.current
    const rect = trigger.current?.getBoundingClientRect()
    if (!element || !rect) return
    const height = element.offsetHeight
    const below = window.innerHeight - rect.bottom - GAP - EDGE
    const downward = below >= height || below >= rect.top - GAP - EDGE
    element.style.left = `${Math.max(EDGE, Math.min(rect.left, window.innerWidth - element.offsetWidth - EDGE))}px`
    element.style.top = `${Math.max(EDGE, downward ? rect.bottom + GAP : rect.top - GAP - height)}px`
    element.style.transformOrigin = downward ? 'top left' : 'bottom left'
  }, [])

  useLayoutEffect(() => {
    if (!open) return
    const element = popover.current
    if (!element) return
    try { element.showPopover?.() } catch { /* still a fixed layer without the popover API */ }
    place()
    const chosen = element.querySelector<HTMLButtonElement>('[role="radio"][aria-checked="true"]:not(:disabled)')
    const first = element.querySelector<HTMLButtonElement>('[role="radio"]:not(:disabled)')
    const target = chosen ?? (tab === 'openrouter' ? search.current : first) ?? element.querySelector<HTMLButtonElement>('[role="tab"][aria-selected="true"]')
    target?.focus({ preventScroll: true })
    chosen?.scrollIntoView({ block: 'nearest' })
    window.addEventListener('resize', place)
    document.addEventListener('scroll', place, true)
    return () => {
      window.removeEventListener('resize', place)
      document.removeEventListener('scroll', place, true)
    }
    // Focus once per opening (not on tab changes, which keep focus where the user put it).
  }, [open, place])

  // The panel's height changes with the tab and the search; keep it anchored to the trigger.
  useLayoutEffect(() => { if (open) place() }, [open, tab, routerRows.total, loading, place])

  useEffect(() => {
    if (!open) return
    const outside = (event: PointerEvent): void => {
      if (!popover.current?.contains(event.target as Node) && !trigger.current?.contains(event.target as Node)) close()
    }
    const blur = (): void => close()
    document.addEventListener('pointerdown', outside, true)
    window.addEventListener('blur', blur)
    return () => {
      document.removeEventListener('pointerdown', outside, true)
      window.removeEventListener('blur', blur)
    }
  }, [open, close])

  const options = (): HTMLButtonElement[] => Array.from(popover.current?.querySelectorAll<HTMLButtonElement>('[role="radio"]:not(:disabled)') ?? [])

  const onPopoverKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      close(true)
    }
  }

  const onTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>): void => {
    const index = ASSISTANT_PROVIDERS.indexOf(tab)
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault()
      const next = ASSISTANT_PROVIDERS[(index + (event.key === 'ArrowRight' ? 1 : -1) + ASSISTANT_PROVIDERS.length) % ASSISTANT_PROVIDERS.length]
      setTab(next)
      setQuery('')
      requestAnimationFrame(() => popover.current?.querySelector<HTMLButtonElement>(`[data-tab="${next}"]`)?.focus())
    } else if (event.key === 'ArrowDown') {
      event.preventDefault()
      ;(tab === 'openrouter' ? search.current : options()[0])?.focus()
    }
  }

  const onOptionKeyDown = (event: KeyboardEvent<HTMLButtonElement>): void => {
    const list = options()
    const index = list.indexOf(event.currentTarget)
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      if (event.key === 'ArrowUp' && index === 0) {
        ;(tab === 'openrouter' ? search.current : popover.current?.querySelector<HTMLButtonElement>(`[data-tab="${tab}"]`))?.focus()
        return
      }
      list[Math.max(0, Math.min(list.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1)))]?.focus()
    } else if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault()
      list[event.key === 'Home' ? 0 : list.length - 1]?.focus()
    }
  }

  const choose = (providerId: AssistantProviderId, modelId: string): void => {
    close(true)
    onChoose(providerId, modelId)
  }

  const status = statuses[tab]
  const connected = status?.state === 'connected'
  const info = ASSISTANT_PROVIDER_INFO[tab]
  const tabId = (providerId: AssistantProviderId): string => `${id}-tab-${providerId}`

  const option = (key: string, label: string, detail: string | null, checked: boolean, onSelect: () => void, title?: string, numeric = false): React.JSX.Element => (
    <button
      key={key}
      type="button"
      role="radio"
      aria-checked={checked}
      disabled={!connected}
      title={title}
      onClick={onSelect}
      onKeyDown={onOptionKeyDown}
      className={cn(
        'flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-xs transition-colors duration-100',
        'focus:outline-none focus-visible:outline-none disabled:opacity-40',
        checked ? 'text-ink' : 'text-ink-muted',
        'enabled:hover:bg-white/[0.08] enabled:hover:text-ink focus:bg-white/[0.08] focus:text-ink'
      )}
    >
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {detail && <span className={cn('shrink-0 text-2xs text-ink-subtle', numeric && 'font-mono tabular')}>{detail}</span>}
      <Check aria-hidden className={cn('h-3.5 w-3.5 shrink-0 text-accent', !checked && 'invisible')} strokeWidth={2.5} />
    </button>
  )

  const routerOption = (model: OpenRouterModel, key: string): React.JSX.Element =>
    option(key, model.name, price(model), provider === 'openrouter' && models.openrouter === model.id, () => choose('openrouter', model.id), `${model.id}${price(model) ? ` · ${price(model)} per million tokens in / out` : ''}`, true)

  const chosenRouterModel = models.openrouter && !routerModels.some((model) => model.id === models.openrouter) ? models.openrouter : null

  return (
    <>
      <button
        ref={trigger}
        type="button"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        aria-label={`Model: ${current}`}
        onClick={() => (open ? close() : openMenu())}
        onKeyDown={(event) => {
          if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); openMenu() }
        }}
        className={cn(
          'flex h-8 min-w-0 max-w-[260px] items-center gap-2 rounded-full pl-2 pr-2.5 text-xs font-medium text-ink',
          'bg-white/[0.05] shadow-[inset_0_0_0_1px_rgb(255_255_255/0.08)] transition-colors duration-150',
          'hover:bg-white/[0.09] aria-expanded:bg-white/[0.1]'
        )}
      >
        {provider
          ? <ProviderLogo provider={provider} className="h-4 w-4 shrink-0" />
          : <Sparkles aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-subtle" />}
        <span className="truncate">{current}</span>
        <ChevronDown aria-hidden className={cn('h-3.5 w-3.5 shrink-0 text-ink-subtle transition-transform duration-150', open && 'rotate-180')} />
      </button>

      {open && (
        <div
          ref={popover}
          id={id}
          role="dialog"
          aria-label="Choose a model"
          popover="manual"
          onKeyDown={onPopoverKeyDown}
          onBlur={(event) => {
            const next = event.relatedTarget as Node | null
            if (next && !popover.current?.contains(next) && !trigger.current?.contains(next)) close()
          }}
          className={cn(MENU_SURFACE, 'fixed inset-auto z-[120] m-0 w-[340px] max-w-[calc(100vw-16px)] border-0 p-0 animate-menu-in')}
          style={{ top: -9999, left: -9999 }}
        >
          <div role="tablist" aria-label="Provider" className="flex gap-1 border-b border-white/[0.07] p-1.5">
            {ASSISTANT_PROVIDERS.map((providerId) => {
              const selected = providerId === tab
              const ready = statuses[providerId]?.state === 'connected'
              return (
                <button
                  key={providerId}
                  id={tabId(providerId)}
                  data-tab={providerId}
                  type="button"
                  role="tab"
                  aria-selected={selected}
                  aria-controls={`${id}-panel`}
                  tabIndex={selected ? 0 : -1}
                  onClick={() => { setTab(providerId); setQuery('') }}
                  onKeyDown={onTabKeyDown}
                  className={cn(
                    'relative flex h-8 min-w-0 flex-1 items-center justify-center gap-1.5 rounded-lg px-2 text-xs font-medium transition-colors duration-150',
                    'focus:outline-none focus-visible:shadow-[inset_0_0_0_1.5px_rgb(var(--accent)/0.75)]',
                    selected ? 'bg-white/[0.1] text-ink' : 'text-ink-muted hover:bg-white/[0.05] hover:text-ink'
                  )}
                >
                  <ProviderLogo provider={providerId} className="h-3.5 w-3.5 shrink-0" />
                  <span className="truncate">{ASSISTANT_PROVIDER_INFO[providerId].brand}</span>
                  {!ready && <span className="sr-only"> (not connected)</span>}
                </button>
              )
            })}
          </div>

          <div id={`${id}-panel`} role="tabpanel" aria-labelledby={tabId(tab)} className="p-1.5">
            <div className="flex min-h-7 items-center gap-2 px-2.5 pb-1">
              <p className="min-w-0 flex-1 truncate text-2xs text-ink-subtle">{info.name} · {statusText(tab, status)}</p>
              {!connected && status && (
                <button
                  type="button"
                  onClick={() => { close(); onConnect(tab) }}
                  className="h-6 shrink-0 rounded-full bg-white/[0.07] px-2.5 text-2xs font-medium text-ink hover:bg-white/[0.12] focus:bg-white/[0.12] focus:outline-none"
                >
                  {tab === 'openrouter' ? 'Add key' : 'Connect'}
                </button>
              )}
            </div>

            {tab === 'openrouter' && (
              <div className="mx-1 mb-1.5 flex h-8 items-center gap-2 rounded-lg bg-black/25 px-2.5 shadow-[inset_0_0_0_1px_rgb(255_255_255/0.07)] focus-within:shadow-[inset_0_0_0_1px_rgb(var(--accent)/0.6)]">
                <Search aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-subtle" />
                <input
                  ref={search}
                  type="search"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'ArrowDown') { event.preventDefault(); options()[0]?.focus() }
                    else if (event.key === 'ArrowUp') { event.preventDefault(); popover.current?.querySelector<HTMLButtonElement>('[data-tab="openrouter"]')?.focus() }
                    else if (event.key === 'Enter') { event.preventDefault(); options()[0]?.click() }
                  }}
                  placeholder="Search OpenRouter models"
                  aria-label="Search OpenRouter models"
                  spellCheck={false}
                  autoComplete="off"
                  className="h-full min-w-0 flex-1 bg-transparent text-xs text-ink placeholder:text-ink-faint focus:outline-none focus-visible:outline-none [&::-webkit-search-cancel-button]:hidden"
                />
                {loading && <Loader2 aria-label="Loading models" className="h-3.5 w-3.5 shrink-0 animate-spin text-ink-subtle" />}
              </div>
            )}

            <div role="radiogroup" aria-label={`${info.brand} models`} className="max-h-[300px] overflow-y-auto overscroll-contain">
              {tab !== 'openrouter' && ASSISTANT_MODELS[tab].map((model) =>
                option(model.id || 'default', model.label, model.id ? null : `${info.name}’s choice`, provider === tab && (models[tab] ?? '') === model.id, () => choose(tab, model.id)))}

              {tab === 'openrouter' && (
                <>
                  {catalogError && !routerModels.length && (
                    <div className="flex items-center gap-2 px-2.5 py-2 text-2xs text-ink-muted">
                      <span className="min-w-0 flex-1">{catalogError}</span>
                      <button type="button" onClick={() => void load(true)} className="flex h-6 items-center gap-1 rounded-full bg-white/[0.07] px-2 text-ink hover:bg-white/[0.12]">
                        <RefreshCw aria-hidden className="h-3 w-3" /> Retry
                      </button>
                    </div>
                  )}
                  {chosenRouterModel && option(`current-${chosenRouterModel}`, chosenRouterModel, null, provider === 'openrouter', () => choose('openrouter', chosenRouterModel))}
                  {routerRows.suggested.length > 0 && (
                    <>
                      <p className="px-2.5 pb-1 pt-1.5 text-2xs font-medium text-ink-subtle">Suggested</p>
                      {routerRows.suggested.map((model) => routerOption(model, `s-${model.id}`))}
                      <p className="px-2.5 pb-1 pt-2.5 text-2xs font-medium text-ink-subtle">All models · {routerModels.length}</p>
                    </>
                  )}
                  {routerRows.matches.map((model) => routerOption(model, model.id))}
                  {!loading && routerModels.length > 0 && routerRows.total === 0 && (
                    <p className="px-2.5 py-2 text-2xs text-ink-subtle">No models match “{query.trim()}”.</p>
                  )}
                  {routerRows.total > MAX_ROUTER_ROWS && (
                    <p className="px-2.5 py-2 text-2xs text-ink-subtle">Showing {MAX_ROUTER_ROWS} of {routerRows.total}. Search to find the rest.</p>
                  )}
                  {loading && !routerModels.length && <p className="px-2.5 py-2 text-2xs text-ink-subtle">Loading OpenRouter models…</p>}
                </>
              )}
            </div>
            {tab === 'openrouter' && (
              <p className="border-t border-white/[0.06] px-2.5 pb-1 pt-2 text-2xs text-ink-subtle">
                Models that can use BridgeClip’s tools. Prices per million tokens, in / out, billed by OpenRouter.
              </p>
            )}
          </div>
        </div>
      )}
    </>
  )
}
