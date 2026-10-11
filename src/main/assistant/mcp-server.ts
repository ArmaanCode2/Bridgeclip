import { createHash, randomBytes } from 'crypto'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'http'
import type { AddressInfo } from 'net'

// A minimal Model Context Protocol server (Streamable HTTP, JSON responses)
// bound to 127.0.0.1. Each assistant turn gets its own bearer token, so a tool
// call always belongs to exactly one conversation and stops working as soon as
// that turn ends. Other local processes and web pages can't call it: requests
// need the token, a loopback Host header and no foreign Origin.

export interface McpToolDefinition {
  name: string
  title: string
  description: string
  inputSchema: Record<string, unknown>
  annotations?: { readOnlyHint?: boolean; destructiveHint?: boolean; idempotentHint?: boolean; openWorldHint?: boolean }
}

export interface McpToolResult {
  text: string
  isError?: boolean
}

export interface McpSessionHandler {
  instructions: string
  listTools: () => McpToolDefinition[]
  callTool: (name: string, args: Record<string, unknown>, signal: AbortSignal) => Promise<McpToolResult>
}

export interface McpSession {
  url: string
  token: string
  close: () => void
}

const MAX_BODY_BYTES = 1024 * 1024
const SERVER_NAME = 'bridgeclip'
const FALLBACK_PROTOCOL_VERSION = '2025-06-18'

interface JsonRpcRequest {
  jsonrpc?: unknown
  id?: unknown
  method?: unknown
  params?: unknown
}

function tokenDigest(token: string): string {
  return createHash('sha256').update(token).digest('hex')
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export class LoopbackMcpServer {
  private server: Server | null = null
  private port = 0
  private starting: Promise<void> | null = null
  private readonly sessions = new Map<string, { handler: McpSessionHandler; calls: Map<string, AbortController> }>()

  constructor(private readonly version: string) {}

  /** Open a session for one turn; the token is revoked by close(). */
  async openSession(handler: McpSessionHandler): Promise<McpSession> {
    await this.ensureListening()
    const token = randomBytes(32).toString('base64url')
    const digest = tokenDigest(token)
    const calls = new Map<string, AbortController>()
    this.sessions.set(digest, { handler, calls })
    return {
      url: `http://127.0.0.1:${this.port}/mcp`,
      token,
      close: () => {
        for (const controller of calls.values()) controller.abort()
        this.sessions.delete(digest)
      }
    }
  }

  async stop(): Promise<void> {
    for (const session of this.sessions.values()) {
      for (const controller of session.calls.values()) controller.abort()
    }
    this.sessions.clear()
    const server = this.server
    this.server = null
    this.starting = null
    if (server) await new Promise<void>((resolve) => server.close(() => resolve()))
  }

  private ensureListening(): Promise<void> {
    if (this.server?.listening) return Promise.resolve()
    this.starting ??= new Promise<void>((resolve, reject) => {
      const server = createServer((request, response) => { void this.handle(request, response) })
      server.requestTimeout = 0
      server.headersTimeout = 30000
      server.keepAliveTimeout = 5000
      server.once('error', (error) => { this.starting = null; reject(error) })
      server.listen(0, '127.0.0.1', () => {
        this.server = server
        this.port = (server.address() as AddressInfo).port
        resolve()
      })
    })
    return this.starting
  }

  private reply(response: ServerResponse, status: number, body?: unknown): void {
    if (response.headersSent) return
    if (body === undefined) {
      response.writeHead(status, { 'Cache-Control': 'no-store' })
      response.end()
      return
    }
    const text = JSON.stringify(body)
    response.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'Content-Length': Buffer.byteLength(text) })
    response.end(text)
  }

  private async handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const host = request.headers.host ?? ''
    if (host !== `127.0.0.1:${this.port}` && host !== `localhost:${this.port}`) return this.reply(response, 403)
    const origin = request.headers.origin
    if (origin !== undefined && origin !== `http://${host}`) return this.reply(response, 403)
    const url = (request.url ?? '').split('?')[0]
    if (url !== '/mcp') return this.reply(response, 404)
    const authorization = request.headers.authorization ?? ''
    const token = /^Bearer\s+(\S+)$/i.exec(authorization)?.[1]
    const session = token ? this.sessions.get(tokenDigest(token)) : undefined
    if (!session) return this.reply(response, 401)
    // No server-initiated stream: clients fall back to plain POST responses.
    if (request.method === 'GET' || request.method === 'DELETE') return this.reply(response, 405)
    if (request.method !== 'POST') return this.reply(response, 405)
    if (!/^application\/json\b/i.test(request.headers['content-type'] ?? '')) return this.reply(response, 415)

    let body: unknown
    try {
      body = JSON.parse(await readBody(request))
    } catch {
      return this.reply(response, 400, { jsonrpc: '2.0', id: null, error: { code: -32700, message: 'Parse error' } })
    }
    const messages = Array.isArray(body) ? body : [body]
    const results: unknown[] = []
    for (const message of messages) {
      const result = await this.dispatch(session, message as JsonRpcRequest)
      if (result !== null) results.push(result)
    }
    if (!results.length) return this.reply(response, 202)
    this.reply(response, 200, Array.isArray(body) ? results : results[0])
  }

  private async dispatch(session: { handler: McpSessionHandler; calls: Map<string, AbortController> }, message: JsonRpcRequest): Promise<unknown | null> {
    if (!isRecord(message) || typeof message.method !== 'string') {
      return { jsonrpc: '2.0', id: null, error: { code: -32600, message: 'Invalid request' } }
    }
    const isNotification = message.id === undefined || message.id === null
    const id = message.id
    const params = isRecord(message.params) ? message.params : {}
    const ok = (result: unknown): unknown => ({ jsonrpc: '2.0', id, result })
    const fail = (code: number, text: string): unknown => ({ jsonrpc: '2.0', id, error: { code, message: text } })

    switch (message.method) {
      case 'initialize': {
        const requested = typeof params.protocolVersion === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(params.protocolVersion)
          ? params.protocolVersion
          : FALLBACK_PROTOCOL_VERSION
        return ok({
          protocolVersion: requested,
          capabilities: { tools: { listChanged: false } },
          serverInfo: { name: SERVER_NAME, title: 'BridgeClip', version: this.version },
          instructions: session.handler.instructions
        })
      }
      case 'notifications/initialized':
        return null
      case 'notifications/cancelled': {
        const requestId = params.requestId
        if (typeof requestId === 'string' || typeof requestId === 'number') session.calls.get(String(requestId))?.abort()
        return null
      }
      case 'ping':
        return isNotification ? null : ok({})
      case 'tools/list':
        return ok({ tools: session.handler.listTools() })
      case 'tools/call': {
        if (isNotification) return null
        if (typeof params.name !== 'string') return fail(-32602, 'Tool name is required')
        const args = isRecord(params.arguments) ? params.arguments : {}
        const controller = new AbortController()
        const key = String(id)
        session.calls.set(key, controller)
        try {
          const result = await session.handler.callTool(params.name, args, controller.signal)
          return ok({ content: [{ type: 'text', text: result.text }], isError: Boolean(result.isError) })
        } catch (error) {
          return ok({ content: [{ type: 'text', text: error instanceof Error ? error.message : 'Tool failed' }], isError: true })
        } finally {
          session.calls.delete(key)
        }
      }
      default:
        return isNotification ? null : fail(-32601, `Method not found: ${message.method}`)
    }
  }
}

function readBody(request: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    let size = 0
    request.on('data', (chunk: Buffer) => {
      size += chunk.length
      if (size > MAX_BODY_BYTES) {
        reject(new Error('Request too large'))
        request.destroy()
        return
      }
      chunks.push(chunk)
    })
    request.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    request.on('error', reject)
  })
}
