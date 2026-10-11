'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const { loadMain } = require('../zernio/support/load-main.cjs')

const chat = loadMain("export * from './src/main/ai-chat'")
const { chatJson, ownModelTarget, ProviderError, PROVIDER_MESSAGE_PREFIX } = chat

const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
const answer = (content) => json({ choices: [{ message: { content } }] })
const SCHEMA = { type: 'object', properties: { title: { type: 'string' } }, required: ['title'], additionalProperties: false }
const TARGET = { provider: 'openai-compatible', base: 'http://localhost:1234/v1', key: '', model: 'qwen3', label: 'Your AI endpoint', local: true, timeoutMs: 5000 }
const MESSAGES = [{ role: 'system', content: 'Write metadata.' }, { role: 'user', content: '{"transcript":"x"}' }]

function server(handler) {
  const calls = []
  const fn = async (url, init) => {
    const body = JSON.parse(init.body)
    calls.push({ url, init, body })
    const out = handler(body, calls.length)
    if (out instanceof Error) throw out
    return out
  }
  fn.calls = calls
  return fn
}
const ask = (fetch, target = TARGET) => chatJson(target, { messages: MESSAGES, schema: SCHEMA, schemaName: 'meta', maxTokens: 4000 }, { fetch })

test('a request is shaped like the engine\'s: schema in the last user message, strict schema, temperature 0.2', async () => {
  const fetch = server(() => answer('{"title": "A"}'))
  assert.deepEqual(await ask(fetch), { title: 'A' })
  const { url, init, body } = fetch.calls[0]
  assert.equal(url, 'http://localhost:1234/v1/chat/completions')
  assert.equal(init.redirect, 'error')
  assert.equal('Authorization' in init.headers, false, 'no key, no header')
  assert.equal(body.messages[0].content, 'Write metadata.', 'the system message is left alone')
  assert.match(body.messages[1].content, /^\{"transcript":"x"\}\n\nReply with one JSON value[\s\S]*JSON Schema/)
  assert.equal(body.response_format.type, 'json_schema')
  assert.equal(body.temperature, 0.2)
  assert.equal(body.max_tokens, 4000)
})

test('it steps down and learns what the server asks for, and reads replies the way the engine does', async () => {
  const fetch = server((body) => {
    if ('temperature' in body) return json({ error: { message: 'temperature is not supported' } }, 400)
    if (body.response_format?.type === 'json_schema') return json({ error: { message: 'json_schema unsupported' } }, 400)
    return answer('<think>which {title}?</think>```json\n{"title": "B",}\n```')
  })
  assert.deepEqual(await ask(fetch), { title: 'B' })
  assert.deepEqual(fetch.calls.map((c) => [c.body.response_format?.type ?? 'none', 'temperature' in c.body]),
    [['json_schema', true], ['json_schema', false], ['json_object', false]])
  const gemini3 = server(() => answer('{"title": "C"}'))
  await ask(gemini3, { ...TARGET, provider: 'gemini', model: 'gemini-3.8-flash', key: 'AIza', label: 'Google Gemini' })
  assert.equal('temperature' in gemini3.calls[0].body, false, 'Gemini 3 keeps its default temperature')
  assert.equal(gemini3.calls[0].init.headers.Authorization, 'Bearer AIza')
})

test('a reply without JSON resolves to undefined so the caller can say so', async () => {
  assert.equal(await ask(server(() => answer('Here is a great title!'))), undefined)
})

test('failures are fixed messages that start with the provider\'s name', async () => {
  const cases = [
    [() => json({}, 401), /^Your AI endpoint rejected the API key/],
    [() => json({}, 402), /^Your AI endpoint reports insufficient credit/],
    [() => json({}, 404), /^Your AI endpoint could not find the model “qwen3”/],
    [() => json({}, 429), /^Your AI endpoint is rate limiting/],
    [() => json({}, 500), /^Your AI endpoint failed \(500\)/],
    [() => json({ error: { message: 'secret detail' } }), /^Your AI endpoint returned an error instead of an answer/],
    [() => json({ error: { message: 'nope' } }, 400), /^Your AI endpoint rejected the request/],
    [() => Object.assign(new Error('slow'), { name: 'TimeoutError' }), /^Your AI endpoint took too long to answer/],
    [() => new TypeError('fetch failed', { cause: Object.assign(new Error('x'), { code: 'ECONNREFUSED' }) }), /^Your AI endpoint is not answering/],
    [() => new TypeError('fetch failed'), /^Your AI endpoint could not be reached\. Check that the server is running/]
  ]
  for (const [reply, message] of cases) {
    await assert.rejects(ask(server(reply)), (error) => {
      assert.ok(error instanceof ProviderError)
      assert.match(error.message, message)
      assert.match(error.message, PROVIDER_MESSAGE_PREFIX)
      assert.doesNotMatch(error.message, /secret/)
      return true
    })
  }
})

test('the target is the saved Gemini or custom model, or nothing for OpenRouter or an unfinished setup', () => {
  const base = { aiProvider: 'gemini', geminiApiKey: 'AIza', geminiModel: '', openaiCompatibleUrl: 'http://localhost:1234/v1', openaiCompatibleApiKey: '', openaiCompatibleModel: 'qwen3', openaiCompatibleTimeout: '30' }
  const gemini = ownModelTarget(base)
  assert.deepEqual([gemini.provider, gemini.model, gemini.label, gemini.base], ['gemini', 'gemini-3.8-flash', 'Google Gemini', 'https://generativelanguage.googleapis.com/v1beta/openai'])
  assert.equal(ownModelTarget({ ...base, geminiApiKey: '' }), null)
  const custom = ownModelTarget({ ...base, aiProvider: 'openai-compatible' })
  assert.deepEqual([custom.base, custom.model, custom.local, custom.timeoutMs], ['http://localhost:1234/v1', 'qwen3', true, 600_000], 'the wait setting, capped at ten minutes')
  assert.equal(ownModelTarget({ ...base, aiProvider: 'openai-compatible', openaiCompatibleModel: '' }), null)
  assert.equal(ownModelTarget({ ...base, aiProvider: 'openai-compatible', openaiCompatibleUrl: 'http://169.254.169.254/x' }), null)
  assert.equal(ownModelTarget({ ...base, aiProvider: 'openrouter' }), null)
})
