import { create } from 'zustand'
import {
  ASSISTANT_CLI_PROVIDERS,
  ASSISTANT_PROVIDERS,
  DEFAULT_ASSISTANT_PREFERENCES,
  type AssistantApprovalRequest,
  type AssistantConversation,
  type AssistantConversationSummary,
  type AssistantEvent,
  type AssistantCliProviderId,
  type AssistantMessage,
  type AssistantPreferences,
  type AssistantProviderId,
  type AssistantProviderStatus,
  type AssistantSignInState
} from '../../shared/assistant'
import { getApi } from '../lib/ipc'
import { errorMessage } from '../lib/utils'

/**
 * The chat with the user's own Claude Code or Codex. The main process owns
 * conversations and runs turns; this store mirrors the open one from events,
 * so leaving the Chat page and coming back keeps a reply streaming.
 */
interface AssistantState {
  statuses: Record<AssistantProviderId, AssistantProviderStatus | null>
  checking: boolean
  preferences: AssistantPreferences
  conversations: AssistantConversationSummary[]
  activeId: string | null
  conversation: AssistantConversation | null
  running: string[]
  approvals: AssistantApprovalRequest[]
  signIn: Record<AssistantCliProviderId, AssistantSignInState | null>
  draft: string
  sending: boolean
  error: string | null
  initialized: boolean

  init: () => Promise<void>
  refreshStatus: (fresh?: boolean) => Promise<void>
  setPreferences: (patch: Partial<AssistantPreferences>) => Promise<void>
  setDraft: (draft: string) => void
  newChat: () => void
  open: (id: string) => Promise<void>
  remove: (id: string) => Promise<void>
  send: (text: string) => Promise<boolean>
  stop: () => Promise<void>
  approve: (requestId: string, allowed: boolean) => Promise<void>
  startSignIn: (provider: AssistantCliProviderId) => Promise<void>
  cancelSignIn: (provider: AssistantCliProviderId) => Promise<void>
  submitSignInCode: (provider: AssistantCliProviderId, code: string) => Promise<void>
  apply: (event: AssistantEvent) => void
  applyStatus: (status: AssistantProviderStatus) => void
  applySignIn: (state: AssistantSignInState) => void
  clearError: () => void
}

function emptyRecord<T>(): Record<AssistantProviderId, T | null> {
  return Object.fromEntries(ASSISTANT_PROVIDERS.map((id) => [id, null])) as Record<AssistantProviderId, T | null>
}

function withMessage(conversation: AssistantConversation, messageId: string, update: (message: AssistantMessage) => AssistantMessage): AssistantConversation {
  return { ...conversation, messages: conversation.messages.map((message) => message.id === messageId ? update(message) : message) }
}

/** The first connected provider, preferring the saved choice. */
export function chosenProvider(state: Pick<AssistantState, 'preferences' | 'statuses'>): AssistantProviderId | null {
  const saved = state.preferences.provider
  if (saved && state.statuses[saved]?.state === 'connected') return saved
  return ASSISTANT_PROVIDERS.find((id) => state.statuses[id]?.state === 'connected') ?? saved ?? null
}

let statusRequest = 0
let openRequest = 0

export const useAssistantStore = create<AssistantState>((set, get) => ({
  statuses: emptyRecord(),
  checking: false,
  preferences: DEFAULT_ASSISTANT_PREFERENCES,
  conversations: [],
  activeId: null,
  conversation: null,
  running: [],
  approvals: [],
  signIn: Object.fromEntries(ASSISTANT_CLI_PROVIDERS.map((id) => [id, null])) as Record<AssistantCliProviderId, AssistantSignInState | null>,
  draft: '',
  sending: false,
  error: null,
  initialized: false,

  init: async () => {
    if (get().initialized) return
    set({ initialized: true })
    const api = getApi().assistant
    const [preferences, conversations, running] = await Promise.all([
      api.preferences().catch(() => DEFAULT_ASSISTANT_PREFERENCES),
      api.conversations().catch(() => []),
      api.running().catch(() => [])
    ])
    set({ preferences, conversations, running })
    void get().refreshStatus()
  },

  refreshStatus: async (fresh = false) => {
    const request = ++statusRequest
    set({ checking: true })
    try {
      const statuses = await getApi().assistant.status(fresh)
      if (request !== statusRequest) return
      const next = emptyRecord<AssistantProviderStatus>()
      for (const status of statuses) next[status.id] = status
      set({ statuses: next })
    } catch (err) {
      if (request === statusRequest) set({ error: errorMessage(err, 'Could not check your assistants.') })
    } finally {
      if (request === statusRequest) set({ checking: false })
    }
  },

  setPreferences: async (patch) => {
    const optimistic = { ...get().preferences, ...patch, models: { ...get().preferences.models, ...patch.models } }
    set({ preferences: optimistic })
    try {
      set({ preferences: await getApi().assistant.savePreferences(patch) })
    } catch (err) {
      set({ error: errorMessage(err, 'Could not save the assistant choice.') })
    }
  },

  setDraft: (draft) => set({ draft }),

  newChat: () => {
    openRequest++
    set({ activeId: null, conversation: null, error: null })
  },

  open: async (id) => {
    const request = ++openRequest
    set({ activeId: id, error: null })
    try {
      const conversation = await getApi().assistant.conversation(id)
      if (request !== openRequest) return
      if (!conversation) {
        set({ activeId: null, conversation: null, conversations: get().conversations.filter((item) => item.id !== id) })
        return
      }
      set({ conversation })
    } catch (err) {
      if (request === openRequest) set({ error: errorMessage(err, 'Could not open this chat.') })
    }
  },

  remove: async (id) => {
    try {
      const conversations = await getApi().assistant.deleteConversation(id)
      set({ conversations })
      if (get().activeId === id) get().newChat()
    } catch (err) {
      set({ error: errorMessage(err, 'Could not delete this chat.') })
    }
  },

  send: async (text) => {
    const state = get()
    const provider = chosenProvider(state)
    if (!provider) {
      set({ error: 'Connect Claude, Codex or OpenRouter in Settings → Assistant first.' })
      return false
    }
    if (provider === 'openrouter' && !state.preferences.models.openrouter) {
      set({ error: 'Choose an OpenRouter model first.' })
      return false
    }
    set({ sending: true, error: null })
    try {
      const { conversationId } = await getApi().assistant.send({
        conversationId: state.activeId,
        provider,
        model: state.preferences.models[provider] ?? '',
        text
      })
      openRequest++
      set({ activeId: conversationId, draft: '' })
      const conversation = await getApi().assistant.conversation(conversationId)
      if (get().activeId === conversationId && conversation) set({ conversation })
      return true
    } catch (err) {
      set({ error: errorMessage(err, 'Could not send the message.') })
      return false
    } finally {
      set({ sending: false })
    }
  },

  stop: async () => {
    const id = get().activeId
    if (id) await getApi().assistant.stop(id).catch(() => {})
  },

  approve: async (requestId, allowed) => {
    set({ approvals: get().approvals.filter((request) => request.id !== requestId) })
    try {
      await getApi().assistant.approve(requestId, allowed)
    } catch (err) {
      set({ error: errorMessage(err, 'Could not send your answer.') })
    }
  },

  startSignIn: async (provider) => {
    try {
      get().applySignIn(await getApi().assistant.signIn.start(provider))
    } catch (err) {
      set({ error: errorMessage(err, 'Could not start sign-in.') })
    }
  },

  cancelSignIn: async (provider) => {
    await getApi().assistant.signIn.cancel(provider).catch(() => {})
    get().applySignIn({ provider, status: 'idle', url: null, acceptsCode: false, message: null })
  },

  submitSignInCode: async (provider, code) => {
    try {
      await getApi().assistant.signIn.submitCode(provider, code)
    } catch (err) {
      set({ error: errorMessage(err, 'Could not send the code.') })
    }
  },

  apply: (event) => {
    const state = get()
    switch (event.type) {
      case 'conversation': {
        const others = state.conversations.filter((item) => item.id !== event.conversation.id)
        set({ conversations: [event.conversation, ...others].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)) })
        return
      }
      case 'turn-start':
        if (!state.running.includes(event.conversationId)) set({ running: [...state.running, event.conversationId] })
        return
      case 'text-delta': {
        const conversation = state.conversation
        if (!conversation || conversation.id !== event.conversationId) return
        set({
          conversation: withMessage(conversation, event.messageId, (message) => {
            const parts = [...message.parts]
            const last = parts[parts.length - 1]
            if (last?.kind === 'text') parts[parts.length - 1] = { ...last, text: (last.text ?? '') + event.text }
            else parts.push({ kind: 'text', text: event.text })
            return { ...message, parts }
          })
        })
        return
      }
      case 'tool': {
        const conversation = state.conversation
        if (!conversation || conversation.id !== event.conversationId) return
        set({
          conversation: withMessage(conversation, event.messageId, (message) => {
            const index = message.parts.findIndex((part) => part.tool?.id === event.tool.id)
            const parts = [...message.parts]
            if (index >= 0) parts[index] = { kind: 'tool', tool: event.tool }
            else parts.push({ kind: 'tool', tool: event.tool })
            return { ...message, parts }
          })
        })
        return
      }
      case 'approval':
        set({ approvals: [...state.approvals.filter((request) => request.id !== event.request.id), event.request] })
        return
      case 'approval-resolved':
        set({ approvals: state.approvals.filter((request) => request.id !== event.requestId) })
        return
      case 'turn-end': {
        set({ running: state.running.filter((id) => id !== event.conversationId) })
        if (state.activeId === event.conversationId) {
          // Main's copy is authoritative once the turn is saved.
          void getApi().assistant.conversation(event.conversationId).then((conversation) => {
            if (conversation && get().activeId === conversation.id) set({ conversation })
          }).catch(() => {})
        }
        return
      }
    }
  },

  applyStatus: (status) => set({ statuses: { ...get().statuses, [status.id]: status } }),
  applySignIn: (signInState) => set({ signIn: { ...get().signIn, [signInState.provider]: signInState } }),
  clearError: () => set({ error: null })
}))
