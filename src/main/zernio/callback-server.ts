import http from 'node:http'
import type { AddressInfo } from 'node:net'
import { APP_NAME } from '../../shared/brand'

const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000

type Outcome = 'success' | 'failure'

interface CallbackHandlers {
  onCallback: (params: URLSearchParams) => void
  onTimeout: () => void
}

interface ActiveCallback {
  server: http.Server
  timer: ReturnType<typeof setTimeout>
}

let active: ActiveCallback | null = null
let callbackGeneration = 0

// The page shows fixed text only: nothing from the query string is echoed.
const PAGE_HEADERS = {
  'Content-Type': 'text/html; charset=utf-8',
  'Cache-Control': 'no-store',
  'Content-Security-Policy': "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
  'Referrer-Policy': 'no-referrer',
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  Connection: 'close'
}

function page(title: string, message: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="referrer" content="no-referrer"><title>${APP_NAME}</title>
<style>body{background:#0a0a0a;color:#fafafa;font-family:system-ui,-apple-system,sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}main{text-align:center;max-width:360px;padding:0 16px}h1{font-size:15px;font-weight:600;margin:0 0 8px}p{font-size:13px;color:#a1a1a1;margin:0}</style></head>
<body><main><h1>${title}</h1><p>${message}</p></main></body>
</html>`
}

const PAGES = {
  success: page('Account connected', `You can close this tab and return to ${APP_NAME}.`),
  failure: page("The connection didn't finish", `Return to ${APP_NAME} to see what happened and try again.`),
  waiting: page(`Return to ${APP_NAME}`, `${APP_NAME} is still waiting for Zernio to finish the sign-in.`),
  done: page('This sign-in already finished', `You can close this tab and return to ${APP_NAME}.`)
}

/**
 * The query of a redirect to `path`, or null when the request is for another
 * path. Zernio has appended its params with `?` instead of `&` before, so any
 * later `?` is read as a separator. A literal `?` inside a value is always
 * percent-encoded, so this cannot split a real value.
 */
export function readRedirectParams(requestUrl: string, path: string): URLSearchParams | null {
  if (!requestUrl.startsWith(path)) return null
  let rest = requestUrl.slice(path.length)
  if (rest.startsWith('/')) rest = rest.slice(1)
  if (rest && !/^[?&]/.test(rest)) return null
  return new URLSearchParams(rest.replace(/^[?&]+/, '').replace(/\?/g, '&'))
}

/** Zernio always adds `error` on failure and `connected`/`accountId` on success; anything else is not its redirect. */
function outcomeOf(params: URLSearchParams): Outcome | null {
  if (params.has('error')) return 'failure'
  if (params.has('connected') || params.has('accountId')) return 'success'
  return null
}

/**
 * One-shot loopback server (127.0.0.1 only, random port) for Zernio's connect
 * redirect. `path` should carry an unguessable nonce so only the redirect we
 * issued matches. Starting a new one replaces any pending one. Resolves with
 * the full redirect URL to hand to Zernio.
 */
export function startCallbackServer(
  path: string,
  handlers: CallbackHandlers,
  { timeoutMs = DEFAULT_TIMEOUT_MS }: { timeoutMs?: number } = {}
): Promise<string> {
  stopCallbackServer()
  const startedIn = callbackGeneration

  return new Promise((resolve, reject) => {
    let expectedHost = ''
    let consumed = false

    const server = http.createServer((req, res) => {
      const send = (status: number, body?: string, extra: http.OutgoingHttpHeaders = {}): void => {
        res.writeHead(status, body === undefined ? { 'Cache-Control': 'no-store', Connection: 'close', ...extra } : { ...PAGE_HEADERS, ...extra })
        res.end(req.method === 'HEAD' ? undefined : body)
      }

      // Only this machine's browser, addressing the port we handed out.
      const host = (req.headers.host ?? '').toLowerCase()
      if (host !== expectedHost && host !== expectedHost.replace('127.0.0.1', 'localhost')) return send(400)
      if (req.method !== 'GET' && req.method !== 'HEAD') return send(405, undefined, { Allow: 'GET, HEAD' })

      const params = readRedirectParams(req.url ?? '/', path)
      if (!params) return send(404)

      const outcome = outcomeOf(params)
      // A browser revisiting the tab, or a second request on a kept-alive socket.
      if (consumed || active?.server !== server) return send(200, PAGES.done)
      if (req.method === 'HEAD' || !outcome) return send(200, PAGES.waiting)

      consumed = true
      send(200, PAGES[outcome])
      stopCallbackServer()
      handlers.onCallback(params)
    })

    server.on('clientError', (_error, socket) => socket.destroy())
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      if (startedIn !== callbackGeneration) {
        server.close()
        reject(new Error('This sign-in was cancelled.'))
        return
      }
      const { port } = server.address() as AddressInfo
      expectedHost = `127.0.0.1:${port}`
      const timer = setTimeout(() => {
        if (active?.server !== server) return
        stopCallbackServer()
        handlers.onTimeout()
      }, timeoutMs)
      active = { server, timer }
      resolve(`http://${expectedHost}${path}`)
    })
  })
}

export function isCallbackPending(): boolean {
  return active !== null
}

export function stopCallbackServer(): void {
  callbackGeneration += 1
  if (!active) return
  const { server, timer } = active
  active = null
  clearTimeout(timer)
  // Stop accepting connections; an in-flight response still completes, and
  // `Connection: close` ends each socket after its response.
  server.close()
  server.closeIdleConnections()
}
