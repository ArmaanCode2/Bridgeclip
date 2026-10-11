// A BridgeClip capability the assistant can call. Tools run in the main
// process with the same functions the renderer's IPC handlers use.

import type { AssistantProviderId } from '../../shared/assistant'

export interface AssistantToolContext {
  conversationId: string
  signal: AbortSignal
}

export interface AssistantToolSpec {
  name: string
  title: string
  description: string
  inputSchema: Record<string, unknown>
  /** Only reads state; never changes files, settings, queues or accounts. */
  readOnly?: boolean
  /** Deletes or overwrites something the user can't get back. */
  destructive?: boolean
  /** Only offered to these providers (default: all). The web tools are for OpenRouter models; the CLIs use their own web search. */
  providers?: readonly AssistantProviderId[]
  /**
   * Plain-language lines describing exactly what will happen, when the call
   * needs the user's approval in the chat first (publishing, spending API
   * credit, deleting). Return null to run without asking.
   */
  confirm?: (input: Record<string, unknown>) => Promise<string[] | null> | string[] | null
  /** A short label for the chat, e.g. "Started a clipping job". */
  describe?: (input: Record<string, unknown>) => string
  run: (input: Record<string, unknown>, context: AssistantToolContext) => Promise<unknown>
}

/** A tool error the agent should see verbatim and can recover from. */
export class AssistantToolError extends Error {}
