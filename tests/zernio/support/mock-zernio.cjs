'use strict'
// A local stand-in for the Zernio REST API (https://docs.zernio.com), plus a
// scripted "browser" that plays the user's side of the connect flow. Tests
// and the isolated e2e app talk to this instead of zernio.com, so no real key
// or social account is ever involved.
//
//   const mock = await createMockZernio({ apiKey: 'test-key', extraRoutes })
//   BRIDGECLIP_ZERNIO_API_URL=mock.apiUrl  BRIDGECLIP_E2E_BROWSER_URL=mock.browserUrl
//
// Other suites (posting) add endpoints with `extraRoutes`; they are matched
// before the built-in routes:
//   { method: 'POST', path: '/api/v1/media/presign' | /^\/api\/v1\/posts\/(\w+)$/, auth?: false,
//     handler: (ctx) => ctx.json(200, {...}) }
// `ctx` = { req, res, method, path, query, params (regex groups), body (parsed
// JSON, or a Buffer for non-JSON), state, mock, json(status, data, headers?),
// text(status, body, headers?), empty(status, headers?) }.

const http = require('node:http')

const OAUTH_HOSTS = {
  tiktok: 'https://www.tiktok.com/v2/auth/authorize/',
  youtube: 'https://accounts.google.com/o/oauth2/auth',
  instagram: 'https://www.instagram.com/oauth/authorize',
  facebook: 'https://www.facebook.com/v21.0/dialog/oauth',
  twitter: 'https://x.com/i/oauth2/authorize',
  linkedin: 'https://www.linkedin.com/oauth/v2/authorization',
  threads: 'https://threads.net/oauth/authorize'
}
// Mirrors Zernio's enum for GET /v1/connect/{platform}.
const CONNECT_PLATFORMS = new Set(['facebook', 'instagram', 'linkedin', 'twitter', 'tiktok', 'youtube', 'threads', 'reddit', 'pinterest', 'bluesky', 'googlebusiness', 'telegram', 'snapchat', 'discord', 'slack', 'whatsapp'])

function objectId(prefix, n) {
  return (prefix + n.toString(16)).padStart(24, '0').slice(-24)
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []
    req.on('data', (chunk) => chunks.push(chunk))
    req.on('end', () => resolve(Buffer.concat(chunks)))
    req.on('error', reject)
  })
}

function get(url, headers = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request(url, { method: 'GET', headers }, (res) => {
      const chunks = []
      res.on('data', (c) => chunks.push(c))
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks).toString('utf8') }))
    })
    req.on('error', reject)
    req.end()
  })
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

/**
 * @param {object} [options]
 * @param {string} [options.apiKey] Bearer token the mock accepts (anything else is 401, like Zernio).
 * @param {Array} [options.extraRoutes] Routes matched before the built-ins.
 * @param {number} [options.rateLimit] Requests per minute before 429 (Zernio's free tier is 60).
 * @param {boolean} [options.withDefaultProfile] Seed a "Default" profile (true).
 */
async function createMockZernio(options = {}) {
  const apiKey = options.apiKey ?? 'test-zernio-key'
  const extraRoutes = [...(options.extraRoutes ?? [])]
  let ids = 0

  const state = {
    profiles: [],
    accounts: [],
    /** accountId -> { status, needsReconnect, issues, canPost, tokenValid } */
    health: {},
    /** Every API request: { method, path, query, authorized } (the Authorization value is never stored). */
    requests: [],
    /** Connect sessions started via GET /v1/connect/{platform}. */
    sessions: [],
    /** URLs the scripted browser was asked to open. */
    opened: [],
    /** Loopback responses the scripted browser received: { url, status, body }. */
    visits: [],
    rateLimit: options.rateLimit ?? 1000,
    rateWindow: [],
    /** One-shot failures: [{ method, path (string|RegExp), status, body, headers }] */
    failures: [],
    /** GET /v1/connect/{platform} answers: platform|'*' -> 'authUrl'|'alreadyConnected'|'reenabled'|'untrustedHost'|{status, body} */
    connectResponses: {},
    /** What the "user" does in the browser: platform|'*' -> behaviour (see openInBrowser). */
    browser: {},
    healthFails: false
  }

  const mock = {
    state,
    apiKey,
    url: '',
    apiUrl: '',
    browserUrl: '',

    addProfile(name, extra = {}) {
      const profile = { _id: objectId('a', ++ids), name, isDefault: false, ...extra }
      state.profiles.push(profile)
      return profile
    },

    addAccount(platform, profileId, extra = {}) {
      const username = extra.username ?? `${platform}_creator`
      const account = {
        _id: objectId('b', ++ids),
        platform,
        profileId: { _id: profileId, name: state.profiles.find((p) => p._id === profileId)?.name ?? 'Default' },
        username,
        displayName: extra.displayName ?? `${platform} creator`,
        isActive: true,
        needsReconnection: false,
        enabled: true,
        ...extra
      }
      state.accounts = state.accounts.filter((a) => !(a.platform === platform && a.profileId._id === profileId))
      state.accounts.push(account)
      state.health[account._id] = { status: 'healthy', needsReconnect: false, issues: [], canPost: true, tokenValid: true }
      return account
    },

    setHealth(accountId, health) {
      state.health[accountId] = { ...state.health[accountId], ...health }
    },

    /** How GET /v1/connect/{platform} answers. */
    setConnectResponse(platform, response) {
      state.connectResponses[platform] = response
    },

    /**
     * What happens in the "browser" after the app opens authUrl:
     *   'success' (default) | 'success-question-mark' | 'no-redirect' | 'abandon'
     *   | { error: 'oauth_denied', params?: {...} } | { delayMs, then: behaviour } | { noise: true, then: behaviour }
     */
    setBrowser(platform, behaviour) {
      state.browser[platform] = behaviour
    },

    /** The next matching request fails once with `status`. */
    failNext(method, path, status, body = {}, headers = {}) {
      state.failures.push({ method, path, status, body, headers })
    },

    route(route) {
      extraRoutes.unshift(route)
    },

    requestsTo(method, pathPrefix) {
      return state.requests.filter((r) => r.method === method && r.path.startsWith(pathPrefix))
    },

    async waitFor(predicate, { timeoutMs = 10_000, intervalMs = 25 } = {}) {
      const started = Date.now()
      for (;;) {
        const value = await predicate()
        if (value) return value
        if (Date.now() - started > timeoutMs) throw new Error('mock-zernio: timed out waiting for condition')
        await sleep(intervalMs)
      }
    },

    close() {
      return new Promise((resolve) => {
        server.close(() => resolve())
        server.closeAllConnections()
      })
    }
  }

  if (options.withDefaultProfile !== false) mock.addProfile('Default', { isDefault: true })

  function send(res, status, body, headers = {}) {
    const payload = body === undefined ? '' : typeof body === 'string' || Buffer.isBuffer(body) ? body : JSON.stringify(body)
    res.writeHead(status, { 'Content-Type': typeof body === 'string' ? 'text/plain' : 'application/json', ...headers })
    res.end(payload)
  }

  function rateHeaders() {
    const now = Date.now()
    state.rateWindow = state.rateWindow.filter((t) => now - t < 60_000)
    const remaining = Math.max(0, state.rateLimit - state.rateWindow.length)
    const reset = Math.ceil(((state.rateWindow[0] ?? now) + 60_000) / 1000)
    return { remaining, headers: { 'X-RateLimit-Limit': String(state.rateLimit), 'X-RateLimit-Remaining': String(remaining), 'X-RateLimit-Reset': String(reset) } }
  }

  function connectSession(platform, query) {
    const session = {
      id: `st_${++ids}`,
      platform,
      profileId: query.get('profileId'),
      redirectUrl: query.get('redirect_url'),
      force: query.get('force') === 'true',
      headless: query.get('headless') === 'true'
    }
    state.sessions.push(session)
    return session
  }

  function accountFor(platform, profileId) {
    return state.accounts.find((a) => a.platform === platform && a.profileId._id === profileId)
  }

  // ---- Built-in API routes (subset of Zernio v1 used by BridgeClip) ----------
  const builtins = [
    { method: 'GET', path: '/api/v1/profiles', handler: (ctx) => ctx.json(200, { profiles: state.profiles }) },
    {
      method: 'POST',
      path: '/api/v1/profiles',
      handler: (ctx) => {
        const name = ctx.body && typeof ctx.body.name === 'string' ? ctx.body.name : ''
        if (!name) return ctx.json(400, { error: 'name is required', type: 'invalid_request_error', code: 'missing_required_field', param: 'name' })
        if (state.profiles.some((p) => p.name === name)) return ctx.json(409, { error: 'A profile with this name already exists' })
        const profile = mock.addProfile(name)
        return ctx.json(201, { message: 'Profile created successfully', profile })
      }
    },
    {
      method: 'DELETE',
      path: /^\/api\/v1\/profiles\/([^/]+)$/,
      handler: (ctx) => {
        const id = decodeURIComponent(ctx.params[0])
        if (!state.profiles.some((profile) => profile._id === id)) return ctx.json(404, { error: 'Profile not found' })
        if (state.accounts.some((account) => account.profileId._id === id)) return ctx.json(400, { error: 'Profile still has accounts' })
        state.profiles = state.profiles.filter((profile) => profile._id !== id)
        return ctx.empty(204)
      }
    },
    {
      method: 'GET',
      path: '/api/v1/accounts',
      handler: (ctx) => ctx.json(200, {
        accounts: state.accounts.filter((account) => !ctx.query.get('profileId') || account.profileId._id === ctx.query.get('profileId')),
        hasAnalyticsAccess: false
      })
    },
    {
      method: 'GET',
      path: '/api/v1/accounts/health',
      handler: (ctx) => {
        if (state.healthFails) return ctx.json(500, { error: 'Internal error', type: 'api_error', code: 'internal_error' })
        const accounts = state.accounts.map((a) => ({
          accountId: a._id,
          platform: a.platform,
          username: a.username,
          profileId: a.profileId._id,
          ...(state.health[a._id] ?? { status: 'healthy', needsReconnect: false, issues: [], canPost: true })
        }))
        const count = (s) => accounts.filter((a) => a.status === s).length
        return ctx.json(200, {
          summary: { total: accounts.length, healthy: count('healthy'), warning: count('warning'), error: count('error'), needsReconnect: accounts.filter((a) => a.needsReconnect).length },
          accounts
        })
      }
    },
    {
      method: 'GET',
      path: /^\/api\/v1\/accounts\/([^/]+)\/health$/,
      handler: (ctx) => {
        const id = decodeURIComponent(ctx.params[0])
        const account = state.accounts.find((a) => a._id === id)
        if (!account) return ctx.json(404, { error: 'Account not found' })
        return ctx.json(200, { accountId: id, platform: account.platform, ...(state.health[id] ?? {}) })
      }
    },
    {
      method: 'DELETE',
      path: /^\/api\/v1\/accounts\/([^/]+)$/,
      handler: (ctx) => {
        const id = decodeURIComponent(ctx.params[0])
        const before = state.accounts.length
        state.accounts = state.accounts.filter((a) => a._id !== id)
        if (state.accounts.length === before) return ctx.json(404, { error: 'Account not found', type: 'not_found', code: 'account_not_found' })
        delete state.health[id]
        return ctx.json(200, { message: 'Account disconnected' })
      }
    },
    {
      method: 'GET',
      path: /^\/api\/v1\/connect\/([^/]+)$/,
      handler: (ctx) => {
        const platform = decodeURIComponent(ctx.params[0])
        if (!CONNECT_PLATFORMS.has(platform)) return ctx.json(400, { error: 'Unsupported platform', type: 'invalid_request_error', code: 'invalid_field_value', param: 'platform' })
        const profileId = ctx.query.get('profileId')
        if (!profileId) return ctx.json(400, { error: 'profileId is required', type: 'invalid_request_error', code: 'missing_required_field', param: 'profileId' })
        if (!/^[a-f0-9]{24}$/.test(profileId)) return ctx.json(400, { error: 'Invalid profileId format', type: 'invalid_request_error', code: 'invalid_field_value', param: 'profileId' })
        if (!state.profiles.some((p) => p._id === profileId)) return ctx.json(404, { error: 'Profile not found', type: 'not_found', code: 'profile_not_found' })
        const redirect = ctx.query.get('redirect_url')
        if (redirect !== null && !/^(https?:\/\/|[a-z][a-z0-9+.-]*:\/\/)/i.test(redirect)) {
          return ctx.json(400, { error: 'redirect_url must be an absolute URL', type: 'invalid_request_error', code: 'INVALID_REDIRECT_URL', param: 'redirect_url' })
        }

        const response = state.connectResponses[platform] ?? state.connectResponses['*'] ?? 'authUrl'
        if (typeof response === 'object') return ctx.json(response.status, response.body ?? {}, response.headers)
        const existing = accountFor(platform, profileId)
        const session = connectSession(platform, ctx.query)
        if (response === 'alreadyConnected' && existing && !session.force) {
          return ctx.json(200, { alreadyConnected: true, accountId: existing._id, platform, username: `@${existing.username}`, displayName: existing.displayName })
        }
        if (response === 'reenabled') {
          const account = existing ?? mock.addAccount(platform, profileId)
          account.isActive = true
          return ctx.json(200, { message: 'Account re-enabled', account: { accountId: account._id, platform, username: account.username, isActive: true } })
        }
        const base = response === 'untrustedHost' ? 'https://login.example-phish.test/oauth' : OAUTH_HOSTS[platform] ?? `https://zernio.com/connect/${platform}`
        const authUrl = `${base}?client_id=mock-client&response_type=code&redirect_uri=${encodeURIComponent('https://zernio.com/api/v1/connect/callback')}&state=${session.id}`
        return ctx.json(200, { authUrl, state: session.id })
      }
    }
  ]

  // ---- The scripted browser --------------------------------------------------
  async function followRedirect(url, noise) {
    const target = new URL(url)
    const host = { Host: target.host }
    if (noise) {
      // What a real browser adds around the redirect: a favicon and a HEAD.
      state.visits.push({ url: `${target.origin}/favicon.ico`, ...(await get(`${target.origin}/favicon.ico`, host).catch((e) => ({ status: 0, body: String(e.code) }))) })
    }
    const first = await get(url, host).catch((e) => ({ status: 0, body: String(e.code) }))
    state.visits.push({ url, ...first })
    if (noise) {
      // The user refreshes the tab after it finished.
      const again = await get(url, host).catch((e) => ({ status: 0, body: String(e.code) }))
      state.visits.push({ url, repeat: true, ...again })
    }
    return first
  }

  function withParams(redirectUrl, params) {
    const url = new URL(redirectUrl)
    for (const [key, value] of Object.entries(params)) url.searchParams.set(key, value)
    return url.toString()
  }

  async function openInBrowser(authUrl, behaviour, session) {
    let noise = false
    while (behaviour && typeof behaviour === 'object' && 'then' in behaviour) {
      if (behaviour.delayMs) await sleep(behaviour.delayMs)
      if (behaviour.noise) noise = true
      behaviour = behaviour.then
    }
    if (behaviour === 'abandon') return
    if (behaviour && typeof behaviour === 'object' && behaviour.error) {
      return followRedirect(withParams(session.redirectUrl, { error: behaviour.error, platform: session.platform, ...(behaviour.params ?? {}) }), noise)
    }
    const account = mock.addAccount(session.platform, session.profileId, session.force ? { username: accountFor(session.platform, session.profileId)?.username } : {})
    if (session.force) mock.setHealth(account._id, { status: 'healthy', needsReconnect: false, issues: [] })
    if (behaviour === 'no-redirect') return
    const params = { connected: session.platform, profileId: session.profileId, accountId: account._id, username: account.username, connect_token: 'ct_mock_not_a_secret' }
    if (behaviour === 'success-question-mark') {
      // Zernio has appended with `?` instead of `&` before.
      const [first, ...rest] = Object.entries(params)
      return followRedirect(`${session.redirectUrl}?${first[0]}=${first[1]}?${rest.map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&')}`, noise)
    }
    return followRedirect(withParams(session.redirectUrl, params), noise)
  }

  // ---- Dispatch --------------------------------------------------------------
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, 'http://127.0.0.1')
    const method = req.method
    const raw = await readBody(req)

    if (method === 'POST' && url.pathname === '/__browser/open') {
      let authUrl
      try {
        authUrl = new URL(JSON.parse(raw.toString('utf8')).url)
      } catch {
        return send(res, 400, { error: 'bad url' })
      }
      state.opened.push(authUrl.toString())
      const session = state.sessions.find((s) => s.id === authUrl.searchParams.get('state'))
      if (!session) return send(res, 404, { error: 'unknown connect session' })
      send(res, 204)
      const behaviour = state.browser[session.platform] ?? state.browser['*'] ?? 'success'
      openInBrowser(authUrl.toString(), behaviour, session).catch(() => {})
      return
    }

    let body = raw
    if (raw.length && /json/.test(req.headers['content-type'] ?? '')) {
      try { body = JSON.parse(raw.toString('utf8')) } catch { body = raw }
    }
    const authorized = req.headers.authorization === `Bearer ${apiKey}`
    const entry = { method, path: url.pathname, query: Object.fromEntries(url.searchParams), authorized }
    state.requests.push(entry)

    const ctx = {
      req,
      res,
      method,
      path: url.pathname,
      query: url.searchParams,
      params: [],
      body,
      state,
      mock,
      json: (status, data, headers) => send(res, status, data, headers),
      text: (status, text, headers) => send(res, status, text, headers),
      empty: (status, headers) => send(res, status, undefined, headers)
    }

    const routes = [...extraRoutes, ...builtins]
    const route = routes.find((r) => {
      if (r.method !== method) return false
      if (typeof r.path === 'string') return r.path === url.pathname
      const match = r.path.exec(url.pathname)
      if (match) ctx.params = match.slice(1)
      return Boolean(match)
    })
    if (!route) return send(res, 404, { error: 'Not found', type: 'not_found', code: 'not_found' })

    const requiresAuth = route.auth !== false
    if (requiresAuth) {
      // Zernio answers any bad or missing key with this exact body (observed with a fake key).
      if (!authorized) return send(res, 401, { error: 'Unauthorized' }, { 'X-RateLimit-Limit': String(state.rateLimit) })
      const { remaining, headers } = rateHeaders()
      if (remaining <= 0) {
        const retry = Math.max(1, Number(headers['X-RateLimit-Reset']) - Math.floor(Date.now() / 1000))
        return send(res, 429, { error: `Rate limit exceeded. Please retry after ${retry} seconds.`, details: { currentCount: state.rateWindow.length + 1, limit: state.rateLimit, retryAfterSeconds: retry } }, { ...headers, 'Retry-After': String(retry) })
      }
      state.rateWindow.push(Date.now())
      const after = rateHeaders().headers
      const failure = state.failures.findIndex((f) => f.method === method && (typeof f.path === 'string' ? f.path === url.pathname : f.path.test(url.pathname)))
      if (failure >= 0) {
        const [f] = state.failures.splice(failure, 1)
        return send(res, f.status, f.body, { ...after, ...f.headers })
      }
      const originalWriteHead = res.writeHead.bind(res)
      res.writeHead = (status, hdrs) => originalWriteHead(status, { ...after, ...hdrs })
    }
    try {
      await route.handler(ctx)
    } catch (error) {
      if (!res.headersSent) send(res, 500, { error: String(error && error.message) })
    }
  })

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const { port } = server.address()
  mock.url = `http://127.0.0.1:${port}`
  mock.apiUrl = `${mock.url}/api/v1`
  mock.browserUrl = `${mock.url}/__browser/open`
  return mock
}

module.exports = { createMockZernio, OAUTH_HOSTS }
