'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const http = require('node:http')
const { loadMain } = require('./support/load-main.cjs')

const load = () => loadMain("export * from './src/main/zernio/callback-server'")

function request(url, { method = 'GET', headers = {} } = {}) {
  return new Promise((resolve, reject) => {
    const target = new URL(url)
    const req = http.request(url, { method, headers: { Host: target.host, ...headers }, agent: false }, (res) => {
      const chunks = []
      res.on('data', (c) => chunks.push(c))
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks).toString('utf8') }))
    })
    req.on('error', reject)
    req.end()
  })
}

function deferred() {
  let resolve
  const promise = new Promise((r) => { resolve = r })
  return { promise, resolve }
}

test('only the nonce path completes the flow, exactly once, and nothing from the query is echoed', async () => {
  const server = load()
  const got = deferred()
  let calls = 0
  const url = await server.startCallbackServer('/zernio/connected/n0nce', {
    onCallback: (params) => { calls += 1; got.resolve(params) },
    onTimeout: () => assert.fail('no timeout expected')
  })
  assert.match(url, /^http:\/\/127\.0\.0\.1:\d+\/zernio\/connected\/n0nce$/)
  const origin = new URL(url).origin

  assert.equal((await request(`${origin}/favicon.ico`)).status, 404)
  assert.equal((await request(`${origin}/zernio/connected/wrong?connected=tiktok`)).status, 404)
  assert.equal((await request(`${origin}/zernio/connected/n0nceX?connected=tiktok`)).status, 404, 'a longer nonce is a different path')
  assert.equal((await request(url, { method: 'POST' })).status, 405)
  assert.equal((await request(url, { headers: { Host: 'evil.test' } })).status, 400, 'DNS-rebinding style hosts are refused')
  const head = await request(`${url}?connected=tiktok`, { method: 'HEAD' })
  assert.equal(head.status, 200)
  assert.equal(head.body, '')
  const bare = await request(url)
  assert.equal(bare.status, 200)
  assert.match(bare.body, /still waiting/, 'a visit without Zernio params does not end the flow')
  assert.equal(calls, 0)
  assert.equal(server.isCallbackPending(), true)

  const evil = '<script>alert(1)</script>'
  const done = await request(`${url}?connected=tiktok&profileId=p1&accountId=a1&username=${encodeURIComponent(evil)}`)
  assert.equal(done.status, 200)
  assert.match(done.body, /Account connected/)
  assert.equal(done.body.includes('script'), false)
  assert.equal(done.body.includes('tiktok'), false)
  assert.match(done.headers['content-security-policy'], /default-src 'none'/)
  assert.equal(done.headers['referrer-policy'], 'no-referrer')
  assert.equal(done.headers['cache-control'], 'no-store')

  const params = await got.promise
  assert.equal(params.get('username'), evil)
  assert.equal(server.isCallbackPending(), false)
  await assert.rejects(request(`${url}?connected=tiktok`), (e) => e.code === 'ECONNREFUSED', 'one-shot: the server is gone')
  assert.equal(calls, 1)
})

test('params appended with ? instead of & are still read', () => {
  const { readRedirectParams } = load()
  const params = readRedirectParams('/cb/n?connected=linkedin?profileId=p1&accountId=a1&username=Jane%20Doe', '/cb/n')
  assert.deepEqual(Object.fromEntries(params), { connected: 'linkedin', profileId: 'p1', accountId: 'a1', username: 'Jane Doe' })
  assert.deepEqual(Object.fromEntries(readRedirectParams('/cb/n/&error=oauth_denied&platform=x', '/cb/n')), { error: 'oauth_denied', platform: 'x' })
  assert.equal(readRedirectParams('/cb/nX?connected=1', '/cb/n'), null)
  assert.equal(readRedirectParams('/other', '/cb/n'), null)
  // An encoded ? inside a value is data, not a separator.
  assert.equal(readRedirectParams('/cb/n?error_message=why%3F&error=x', '/cb/n').get('error_message'), 'why?')
})

test('an error redirect shows the failure page', async () => {
  const server = load()
  const got = deferred()
  const url = await server.startCallbackServer('/cb/err', { onCallback: got.resolve, onTimeout: () => {} })
  const res = await request(`${url}?error=oauth_denied&platform=linkedin`)
  assert.match(res.body, /didn.t finish/)
  assert.equal((await got.promise).get('error'), 'oauth_denied')
})

test('a revisit on a kept-alive socket gets the finished page without a second callback', async () => {
  const server = load()
  let calls = 0
  const url = await server.startCallbackServer('/cb/keep', { onCallback: () => { calls += 1 }, onTimeout: () => {} })
  const agent = new http.Agent({ keepAlive: true, maxSockets: 1 })
  const get = (target) => new Promise((resolve, reject) => {
    http.get(target, { agent }, (res) => {
      let body = ''
      res.on('data', (c) => { body += c })
      res.on('end', () => resolve({ status: res.statusCode, body, connection: res.headers.connection }))
    }).on('error', reject)
  })
  const first = await get(`${url}?connected=x`)
  assert.equal(first.connection, 'close', 'responses close the socket so the server can stop')
  assert.equal(calls, 1)
  await assert.rejects(get(`${url}?connected=x`))
  agent.destroy()
  assert.equal(calls, 1)
})

test('times out, and cancel or a new start stops a pending server without callbacks', async () => {
  const server = load()
  const timedOut = deferred()
  const url = await server.startCallbackServer('/cb/slow', { onCallback: () => assert.fail('no callback'), onTimeout: timedOut.resolve }, { timeoutMs: 50 })
  await timedOut.promise
  assert.equal(server.isCallbackPending(), false)
  await assert.rejects(request(`${url}?connected=x`), (e) => e.code === 'ECONNREFUSED')

  let fired = false
  const cancelled = await server.startCallbackServer('/cb/cancel', { onCallback: () => { fired = true }, onTimeout: () => { fired = true } }, { timeoutMs: 50 })
  server.stopCallbackServer()
  await new Promise((resolve) => setTimeout(resolve, 80))
  assert.equal(fired, false, 'a cancelled flow neither times out nor calls back')
  await assert.rejects(request(`${cancelled}?connected=x`), (e) => e.code === 'ECONNREFUSED')

  const first = await server.startCallbackServer('/cb/one', { onCallback: () => { fired = true }, onTimeout: () => { fired = true } })
  const got = deferred()
  const second = await server.startCallbackServer('/cb/two', { onCallback: got.resolve, onTimeout: () => {} })
  await assert.rejects(request(`${first}?connected=x`), (e) => e.code === 'ECONNREFUSED', 'the older sign-in was replaced')
  await request(`${second}?connected=x`)
  assert.equal((await got.promise).get('connected'), 'x')
  assert.equal(fired, false)
})

test('a replacement during loopback bind leaves only the newest server active', async () => {
  const server = load()
  const first = server.startCallbackServer('/cb/first', { onCallback: () => assert.fail('old callback'), onTimeout: () => assert.fail('old timeout') })
  const second = server.startCallbackServer('/cb/second', { onCallback: () => {}, onTimeout: () => {} })
  await assert.rejects(first, /cancelled/)
  const url = await second
  assert.equal(server.isCallbackPending(), true)
  assert.equal((await request(url)).status, 200)
  server.stopCallbackServer()
})

test('binds to 127.0.0.1 only', async () => {
  const server = load()
  const url = await server.startCallbackServer('/cb/bind', { onCallback: () => {}, onTimeout: () => {} })
  const { port } = new URL(url)
  const os = require('node:os')
  const external = Object.values(os.networkInterfaces()).flat().find((i) => i && i.family === 'IPv4' && !i.internal)
  if (external) {
    await assert.rejects(request(`http://${external.address}:${port}/cb/bind?connected=x`, { headers: { Host: `127.0.0.1:${port}` } }), (e) => e.code === 'ECONNREFUSED')
  }
  server.stopCallbackServer()
})
