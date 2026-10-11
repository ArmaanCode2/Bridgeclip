'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const { spawnSync } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')
const { loadMain, tempDir } = require('../zernio/support/load-main.cjs')

const checks = loadMain("export * from './src/main/provider-checks'")
const { testAiProvider, checkLocalWhisper, parseModelIds, parseWhisperProbe, modelListed, localWhisperScript, extractJson, twoColourPng } = checks

const SETTINGS = {
  openrouterApiKey: 'sk-or-secret', geminiApiKey: 'AIza-secret', geminiModel: 'gemini-3.8-flash',
  openaiCompatibleUrl: 'http://localhost:11434/v1', openaiCompatibleApiKey: '', openaiCompatibleModel: 'llama3.1', openaiCompatibleTimeout: '30',
  localWhisperMode: 'builtin', localWhisperUrl: '', localWhisperModel: 'small'
}
const settings = (patch) => ({ ...SETTINGS, ...patch })

/** A fetch that records its calls and answers with the next canned response. */
function fakeFetch(...responses) {
  const calls = []
  const fn = async (url, init) => {
    calls.push({ url, init })
    const next = responses.shift()
    if (next instanceof Error) throw next
    return typeof next === 'function' ? next() : next
  }
  fn.calls = calls
  return fn
}
const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
const answer = (content, extra = {}) => json({ choices: [{ message: { content }, ...extra }] })
const hasImage = (body) => (body?.messages ?? []).some((m) => Array.isArray(m.content) && m.content.some((part) => part.type === 'image_url'))

/**
 * A fake server: `list` answers GET /models, `chat(body)` each text request to /chat/completions,
 * `image(body)` each image request. Records every call with its parsed body.
 */
function fakeServer({ list = () => json({ data: [{ id: 'llama3.1' }] }), chat = () => answer('{"ok": true, "word": "clip"}'), image } = {}) {
  const calls = []
  const fn = async (url, init) => {
    const body = init.body ? JSON.parse(init.body) : null
    calls.push({ url, init, body })
    const run = (handler) => {
      const out = handler(body)
      if (out instanceof Error) throw out
      return out
    }
    if (url.endsWith('/models')) return run(list)
    if (hasImage(body)) return run(image ?? (() => answer('{"left": "red", "right": "green"}')))
    return run(chat)
  }
  fn.calls = calls
  return fn
}
// random() = 0 always picks red for the left half and green for the right.
const RED_GREEN = { random: () => 0 }
const states = (result) => Object.fromEntries(result.checks.map((check) => [check.id, check.state]))
const refused = () => new TypeError('fetch failed', { cause: Object.assign(new Error('connect ECONNREFUSED 127.0.0.1:11434'), { code: 'ECONNREFUSED' }) })
const run = (provider, patch, server, deps = {}) => testAiProvider(provider, settings(patch), { fetch: server, ...RED_GREEN, ...deps })

test('Gemini: lists models, then sends one request shaped like a clip job with the saved model', async () => {
  const fetch = fakeServer({ list: () => json({ object: 'list', data: [{ id: 'models/gemini-3.8-flash' }, { id: 'models/gemini-3.1-pro-preview' }] }) })
  const result = await run('gemini', {}, fetch)
  assert.equal(result.ok, true)
  assert.equal(result.message, 'Ready. Clip planning works with this model.')
  assert.deepEqual(result.models, ['gemini-3.8-flash', 'gemini-3.1-pro-preview'])
  assert.deepEqual(states(result), { server: 'ok', key: 'ok', model: 'ok', json: 'ok', images: 'skipped' })
  assert.equal(result.readsImages, null, 'Gemini models read images, so no quota is spent finding out')
  assert.equal(result.subject, 'gemini|gemini-3.8-flash')
  assert.deepEqual(fetch.calls.map((c) => c.url), [
    'https://generativelanguage.googleapis.com/v1beta/openai/models',
    'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'
  ])
  for (const call of fetch.calls) {
    assert.equal(call.init.headers.Authorization, 'Bearer AIza-secret')
    assert.equal(call.init.redirect, 'error')
    assert.ok(call.init.signal, 'a timeout is set')
  }
  const body = fetch.calls[1].body
  assert.equal(body.model, 'gemini-3.8-flash')
  assert.equal(body.messages[0].role, 'system')
  assert.ok(Array.isArray(body.messages[1].content), 'the prompt is a list of parts, as the planner sends it')
  assert.match(body.messages[1].content.at(-1).text, /JSON Schema/, 'the schema is the last part, as in a clip job')
  assert.equal(body.response_format.type, 'json_schema')
  assert.equal(body.max_tokens, 32000, 'the planner\'s output allowance')
  assert.equal('temperature' in body, false, 'Gemini 3 keeps its default temperature')
})

test('a custom endpoint gets temperature 0.2 and a two-colour image that changes every run', async () => {
  const fetch = fakeServer()
  const result = await run('openai-compatible', {}, fetch)
  assert.deepEqual([result.ok, result.readsImages], [true, true])
  assert.equal(result.message, 'Ready. Clip planning works with this model, and it can read images.')
  assert.equal(result.subject, 'openai-compatible|http://localhost:11434/v1|llama3.1')
  assert.equal(fetch.calls[1].body.temperature, 0.2)
  const image = fetch.calls[2].body.messages[1].content.find((part) => part.type === 'image_url').image_url.url
  assert.match(image, /^data:image\/png;base64,/)
  const other = fakeServer()
  await run('openai-compatible', {}, other, { random: () => 0.5 })
  assert.notEqual(other.calls[2].body.messages[1].content.find((part) => part.type === 'image_url').image_url.url, image)
})

test('a blind model or a proxy that drops images cannot pass by guessing', async () => {
  for (const guess of ['{"left": "red", "right": "red"}', '{"left": "blue", "right": "green"}', '{"left": "green", "right": "red"}', 'I cannot see images.']) {
    const result = await run('openai-compatible', {}, fakeServer({ image: () => answer(guess) }))
    assert.deepEqual([result.ok, result.readsImages, states(result).images], [true, false, 'failed'], guess)
  }
  const synonyms = await run('openai-compatible', {}, fakeServer({ image: () => answer('{"left": "Crimson", "right": "lime"}') }))
  assert.equal(synonyms.readsImages, true)
})

test('the image request steps down like a job\'s frame check, and a refusal after that means no images', async () => {
  const schemaOnlyWithImages = await run('openai-compatible', {}, fakeServer({
    image: (body) => body.response_format?.type === 'json_schema' ? json({ error: { message: 'response_format not supported with images' } }, 400) : answer('{"left": "red", "right": "green"}')
  }))
  assert.equal(schemaOnlyWithImages.readsImages, true)
  const textOnly = await run('openai-compatible', {}, fakeServer({ image: () => json({ error: { message: 'image input is not supported' } }, 400) }))
  assert.deepEqual([textOnly.ok, textOnly.readsImages, states(textOnly).images], [true, false, 'failed'])
  assert.equal(textOnly.message, 'Ready. Clip planning works with this model.')
})

test('OpenAI-compatible on localhost: no key is sent, and the key line says none was needed', async () => {
  const fetch = fakeServer()
  const result = await run('openai-compatible', {}, fetch)
  assert.equal(fetch.calls[0].url, 'http://localhost:11434/v1/models')
  assert.ok(fetch.calls.every((c) => !('Authorization' in c.init.headers)))
  assert.deepEqual(result.checks.find((c) => c.id === 'key'), { id: 'key', state: 'skipped', message: 'No key was sent, and the server did not ask for one.' })
  const withKey = fakeServer()
  await run('openai-compatible', { openaiCompatibleApiKey: 'sk-custom' }, withKey)
  assert.ok(withKey.calls.every((c) => c.init.headers.Authorization === 'Bearer sk-custom'))
})

test('answers are judged the way a clip job reads them', async () => {
  for (const content of [
    '<think>weigh {a} and [b]</think>\nSure! ```json\n{"ok": true, "word": "clip"}\n```',
    'weigh {a}\n</think>\n{"ok": true, "word": "clip"}',
    '{"ok": "true", "word": "clip",}',
    '[{"ok": true}]'
  ]) {
    const result = await run('openai-compatible', {}, fakeServer({ chat: () => answer(content) }))
    assert.equal(states(result).json, 'ok', content)
  }
  const prose = await run('openai-compatible', {}, fakeServer({ chat: () => answer('Here are some great clips!') }))
  assert.deepEqual([prose.ok, states(prose).json], [false, 'failed'])
  assert.match(prose.message, /not with JSON/)
  const wrapped = await run('openai-compatible', {}, fakeServer({ chat: () => json([{ choices: [{ message: { content: '{"ok": true}' } }] }]) }))
  assert.equal(states(wrapped).json, 'ok', 'a list-wrapped reply is unwrapped, as the engine does')
})

test('a model that spends its whole allowance thinking is a warning, not a JSON failure', async () => {
  const result = await run('openai-compatible', {}, fakeServer({ chat: () => answer(null, { finish_reason: 'length' }) }))
  assert.deepEqual([result.ok, states(result).json], [false, 'warning'])
  assert.match(result.message, /whole output allowance/)
})

test('an error in a 200 reply is a server error, not an answer', async () => {
  const result = await run('openai-compatible', {}, fakeServer({ chat: () => json({ error: { message: 'model crashed' } }) }))
  assert.deepEqual([result.ok, states(result).model], [false, 'failed'])
  assert.equal(result.message, 'The server answered with an error instead of a reply.')
})

test('the request learns what the server asks for, as a clip job does', async () => {
  const strict = fakeServer({ chat: (body) => body.response_format?.type === 'json_schema' ? json({ error: { message: 'response_format json_schema unsupported' } }, 400) : answer('{"ok": true, "word": "clip"}') })
  assert.equal((await run('openai-compatible', {}, strict)).checks.find((c) => c.id === 'json').message, 'JSON works (JSON mode).')

  const temperature = fakeServer({ chat: (body) => 'temperature' in body ? json({ error: { message: 'temperature may only be set to 1' } }, 400) : answer('{"ok": true}') })
  const warm = await run('openai-compatible', {}, temperature)
  assert.equal(states(warm).json, 'ok')
  assert.equal(warm.checks.find((c) => c.id === 'json').message, 'Structured JSON works.', 'dropping temperature does not weaken the format')

  const smallContext = fakeServer({ chat: (body) => body.max_tokens > 8192 ? json({ error: { message: 'max_tokens must be at most 8192' } }, 400) : answer('{"ok": true}') })
  assert.equal((await run('openai-compatible', {}, smallContext)).ok, true)
  assert.ok(smallContext.calls.some((c) => c.body?.max_tokens === 8192))

  const newer = fakeServer({ chat: (body) => 'max_tokens' in body ? json({ error: { message: "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead." } }, 400) : answer('{"ok": true}') })
  assert.equal((await run('openai-compatible', {}, newer)).checks.find((c) => c.id === 'json').message, 'Structured JSON works.')
})

test('a 400 that stays after every step down is sorted by what it says, never blamed on JSON', async () => {
  const cases = [
    [[{ error: { status: 'INVALID_ARGUMENT', message: 'API key not valid. Please pass a valid API key.' } }], 'key', 'The provider rejected the API key.'],
    [{ error: { message: 'Invalid model name passed in model=llama9' } }, 'model', 'The server did not accept the model “llama3.1”.'],
    [{ error: { message: "This model's maximum context length is 4096 tokens" } }, 'model', 'The server’s context window is too small for clip planning. Use a model or server setting with 32,000 tokens or more.'],
    [{ error: { message: 'something else' } }, 'model', 'The server rejected the test request.']
  ]
  for (const [body, id, message] of cases) {
    const result = await run('openai-compatible', { openaiCompatibleApiKey: 'sk-x' }, fakeServer({ chat: () => json(body, 400) }))
    assert.deepEqual([result.ok, states(result)[id], result.message], [false, 'failed', message], message)
    assert.equal(states(result).json, undefined, 'no JSON line when JSON was not the problem')
    if (id !== 'key') assert.notEqual(states(result).key, 'ok', 'a refused request does not prove the key')
  }
})

test('the model list only stops the test when nothing is there; a job never asks for it', async () => {
  for (const list of [
    () => json({ error: { message: 'Missing scopes: api.model.read' } }, 403),
    () => json({}, 404),
    () => json({}, 500),
    () => Object.assign(new Error('timed out'), { name: 'TimeoutError' }),
    () => new TypeError('fetch failed', { cause: new Error('unexpected redirect') }),
    () => new Response('x'.repeat(1024 * 1024 + 10), { status: 200 })
  ]) {
    const result = await run('openai-compatible', {}, fakeServer({ list }))
    assert.equal(result.ok, true, String(list))
  }
})

test('a stopped local server is named by address and port, on IPv4, IPv6 or both', async () => {
  const flat = await run('openai-compatible', { openaiCompatibleUrl: 'http://localhost:3001/v1' }, fakeServer({ list: refused }))
  assert.deepEqual([flat.ok, flat.message, states(flat)], [false, 'Nothing is answering at localhost:3001. Start the server, or check the address and port.', { server: 'failed' }])
  const both = () => new TypeError('fetch failed', { cause: new AggregateError([
    Object.assign(new Error('connect ETIMEDOUT ::1:3001'), { code: 'ETIMEDOUT' }),
    Object.assign(new Error('connect ECONNREFUSED 127.0.0.1:3001'), { code: 'ECONNREFUSED' })
  ]) })
  const happyEyeballs = await run('openai-compatible', { openaiCompatibleUrl: 'http://localhost:3001/v1' }, fakeServer({ list: both }))
  assert.equal(happyEyeballs.message, 'Nothing is answering at localhost:3001. Start the server, or check the address and port.')
  const defaultPort = await run('openai-compatible', { openaiCompatibleUrl: 'http://localhost/v1' }, fakeServer({ list: refused }))
  assert.match(defaultPort.message, /localhost:80\./)
})

test('other connection failures become plain messages', async () => {
  const noHost = () => new TypeError('fetch failed', { cause: Object.assign(new Error('getaddrinfo ENOTFOUND'), { code: 'ENOTFOUND' }) })
  assert.equal((await run('openai-compatible', {}, fakeServer({ list: noHost }))).message, 'Could not find localhost:11434. Check the address.')
  assert.equal((await run('gemini', {}, fakeServer({ list: noHost }))).message, 'Could not reach Google’s Gemini service. Check your internet connection.', 'offline, not a typo')
  const down = () => new TypeError('fetch failed')
  assert.equal((await run('gemini', {}, fakeServer({ list: down, chat: down }))).message, 'Could not reach Google’s Gemini service. Check your internet connection.')
  assert.equal((await run('openai-compatible', {}, fakeServer({ list: down, chat: down }))).message, 'Could not reach localhost:11434. Check that the server is running.')
  const redirect = () => new TypeError('fetch failed', { cause: new Error('unexpected redirect') })
  assert.equal((await run('openai-compatible', {}, fakeServer({ chat: redirect }))).message, 'The server tried to redirect the request, which is not allowed.')
})

test('a missing model and a wrong URL are told apart', async () => {
  const missing = await run('openai-compatible', { openaiCompatibleModel: 'llama9' }, fakeServer({ chat: () => json({ error: { message: 'model "llama9" not found, try pulling it first' } }, 404) }))
  assert.equal(missing.message, 'Model “llama9” was not found. Pick one from the list.')
  assert.deepEqual(missing.models, ['llama3.1'])
  const wrongPath = await run('openai-compatible', { openaiCompatibleUrl: 'http://localhost:11434' }, fakeServer({
    list: () => new Response('404 page not found', { status: 404 }), chat: () => new Response('404 page not found', { status: 404 })
  }))
  assert.equal(wrongPath.message, 'Nothing at this URL answers chat requests. Check that it ends in /v1.')
  assert.deepEqual(states(wrongPath), { server: 'failed' })
})

test('a listed name spelled differently answers; an unlisted one is noted, not failed', async () => {
  const tagged = await run('openai-compatible', { openaiCompatibleModel: 'qwen2.5' }, fakeServer({ list: () => json({ models: [{ name: 'qwen2.5:7b' }] }) }))
  assert.equal(tagged.checks.find((c) => c.id === 'model').message, '“qwen2.5” answered.')
  const unlisted = await run('gemini', { geminiModel: 'gemini-9' }, fakeServer({ list: () => json({ data: [{ id: 'other-model' }] }) }))
  assert.equal(unlisted.ok, true)
  assert.equal(unlisted.checks.find((c) => c.id === 'model').message, '“gemini-9” answered, though the server does not list it.')
})

test('model names match leniently: models/ prefix, :latest and :tag are all the same model', () => {
  const listed = ['llama3.1:latest', 'models/gemini-x', 'qwen2.5:7b', 'phi3']
  for (const wanted of ['llama3.1', 'llama3.1:latest', 'gemini-x', 'models/gemini-x', 'qwen2.5', 'qwen2.5:7b', 'phi3', ' phi3 ', '']) {
    assert.equal(modelListed(wanted, listed), true, JSON.stringify(wanted))
  }
  for (const wanted of ['llama3', 'llama3.1:8b', 'gemini', 'gemini-xy', 'qwen2.5:14b', 'phi']) {
    assert.equal(modelListed(wanted, listed), false, JSON.stringify(wanted))
  }
  assert.equal(modelListed('x', []), false)
  assert.deepEqual(parseModelIds({ models: [{ name: 'qwen2.5:7b' }, { name: 'models/x' }, 5, null] }), ['qwen2.5:7b', 'x'])
  assert.deepEqual(parseModelIds({ data: ['a', 'a', { id: 'b' }] }), ['a', 'b'])
  assert.deepEqual(parseModelIds(null), [])
})

test('JSON is extracted the way the engine extracts it', () => {
  assert.deepEqual(extractJson('{"ok": true}'), { ok: true })
  assert.deepEqual(extractJson('<think>a {b}</think>{"ok": true}'), { ok: true })
  assert.deepEqual(extractJson('reasoning {x} </think>\n{"ok": true}'), { ok: true }, 'template-opened reasoning')
  assert.deepEqual(extractJson('Sure:\n```json\n[1, 2]\n```'), [1, 2])
  assert.deepEqual(extractJson('Answer: {"ok": true,} done'), { ok: true }, 'a trailing comma is repaired')
  assert.deepEqual(extractJson('{"title": "Why models emit <think> then </think> tags"}'), { title: 'Why models emit <think> then </think> tags' })
  assert.equal(extractJson('<think>still going {"ok": true}'), undefined, 'cut off inside the reasoning')
  assert.equal(extractJson('no json here'), undefined)
})

test('the test image is a real PNG with the two colours side by side', () => {
  const url = twoColourPng([220, 30, 30], [30, 170, 60], 4)
  const png = Buffer.from(url.replace(/^data:image\/png;base64,/, ''), 'base64')
  assert.deepEqual([...png.subarray(0, 8)], [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  const idat = png.indexOf('IDAT')
  const pixels = require('node:zlib').inflateSync(png.subarray(idat + 4, idat + 4 + png.readUInt32BE(idat - 4)))
  assert.deepEqual([...pixels.subarray(0, 13)], [0, 220, 30, 30, 220, 30, 30, 30, 170, 60, 30, 170, 60])
})

test('rate limits, quotas, busy servers, timeouts and credit are plain messages, and do not claim the key works', async () => {
  const cases = [
    [() => json({}, 429), 'The provider is limiting requests right now. Try again in a minute.', 'warning'],
    [() => json([{ error: { code: 429, details: [{ violations: [{ quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' }] }] } }], 429), 'This key’s daily quota is used up. It resets tomorrow; a paid key raises it.', 'warning'],
    [() => json({}, 503), 'The server is busy or still loading the model. Try again in a minute.', 'warning'],
    [() => Object.assign(new Error('timed out'), { name: 'TimeoutError' }), 'The model did not answer within 300 seconds. A local model may still be loading; try again.', 'warning'],
    [() => json({}, 500), 'The server failed on the test request. Check the model name, or the server’s own log.', 'failed']
  ]
  for (const [chat, message, state] of cases) {
    const result = await run('openai-compatible', { openaiCompatibleApiKey: 'sk-x' }, fakeServer({ chat }))
    assert.deepEqual([result.ok, result.message, states(result).model], [false, message, state])
    assert.equal(states(result).key, undefined, message)
  }
  const broke = await run('gemini', {}, fakeServer({ chat: () => json({}, 402) }))
  assert.deepEqual([broke.ok, broke.message, states(broke).key], [false, 'The account is out of credit.', 'failed'])
  const locked = await run('openai-compatible', {}, fakeServer({ list: () => json({}, 401), chat: () => json({}, 401) }))
  assert.equal(locked.message, 'The server asks for an API key. Enter it above.')
  const rejected = await run('openai-compatible', { openaiCompatibleApiKey: 'sk-bad' }, fakeServer({ chat: () => json({}, 401) }))
  assert.equal(rejected.message, 'The provider rejected the API key.')
})

test('a slow local model gets the endpoint\'s wait setting, up to five minutes', async () => {
  const slow = () => Object.assign(new Error('timed out'), { name: 'TimeoutError' })
  const short = await run('openai-compatible', { openaiCompatibleTimeout: '5' }, fakeServer({ chat: slow }))
  assert.match(short.message, /within 300 seconds/)
  const fetch = fakeServer({ chat: slow })
  await run('openai-compatible', { openaiCompatibleTimeout: '120' }, fetch)
  assert.match((await run('openai-compatible', { openaiCompatibleTimeout: '120' }, fakeServer({ chat: slow }))).message, /within 300 seconds/, 'capped at five minutes')
})

test('without a model name the server is still checked, and the model line asks for one', async () => {
  const fetch = fakeServer()
  const result = await run('openai-compatible', { openaiCompatibleModel: '' }, fetch)
  assert.deepEqual(states(result), { server: 'ok', model: 'failed' })
  assert.equal(result.message, 'Enter the model name first.')
  assert.equal(fetch.calls.length, 1, 'no chat request without a model')
})

test('OpenRouter: checks the key without trying a model', async () => {
  const fetch = fakeFetch(json({ data: { label: 'x' } }))
  const result = await testAiProvider('openrouter', settings({}), { fetch })
  assert.deepEqual([result.ok, result.message, result.models, result.readsImages, result.subject], [true, 'Connected. The OpenRouter key works.', [], null, 'openrouter'])
  assert.deepEqual(states(result), { server: 'ok', key: 'ok' })
  assert.equal(fetch.calls[0].url, 'https://openrouter.ai/api/v1/auth/key')
  assert.equal(fetch.calls.length, 1)
  const rejected = await testAiProvider('openrouter', settings({}), { fetch: fakeFetch(json({}, 401)) })
  assert.deepEqual([rejected.ok, rejected.message], [false, 'The provider rejected the API key.'])
})

test('nothing is sent until the provider is configured, and an unknown provider is refused', async () => {
  const fetch = fakeFetch()
  assert.equal((await testAiProvider('gemini', settings({ geminiApiKey: '' }), { fetch })).message, 'Add your Gemini API key first.')
  assert.equal((await testAiProvider('openrouter', settings({ openrouterApiKey: '' }), { fetch })).message, 'Add your OpenRouter API key first.')
  assert.equal((await testAiProvider('openai-compatible', settings({ openaiCompatibleUrl: '' }), { fetch })).message, 'Enter a valid endpoint URL first.')
  // A URL that settings would refuse is refused here too.
  assert.equal((await testAiProvider('openai-compatible', settings({ openaiCompatibleUrl: 'http://169.254.169.254/latest' }), { fetch })).ok, false)
  assert.equal(fetch.calls.length, 0)
  await assert.rejects(testAiProvider('bogus', settings({}), { fetch }), /Unknown AI provider/)
  await assert.rejects(testAiProvider(undefined, settings({}), { fetch }), /Unknown AI provider/)
})

test('neither the key nor anything the server says reaches the result', async () => {
  const hostile = 'Bearer AIza-secret <script>alert(1)</script> /home/user/.ssh'
  const servers = [
    { list: () => json({ error: { message: hostile } }, 400) },
    { list: () => json({ data: [{ id: 'ok' }], note: hostile }), chat: () => answer(hostile) },
    { chat: () => json({ error: { message: hostile } }, 400) },
    { chat: () => json({ error: hostile }, 500) },
    { chat: () => json({ error: { message: hostile } }) },
    { list: () => new Error(`connect failed for ${hostile}`), chat: () => new Error(`connect failed for ${hostile}`) }
  ]
  for (const server of servers) {
    for (const provider of ['gemini', 'openai-compatible']) {
      const result = await run(provider, { geminiModel: 'ok', openaiCompatibleModel: 'ok' }, fakeServer({ ...server, image: () => answer(`{"left": "${hostile}"}`) }))
      const text = JSON.stringify(result)
      assert.equal(text.includes('AIza-secret'), false)
      assert.equal(text.includes('script'), false)
      assert.equal(text.includes('.ssh'), false)
    }
  }
})

test('at most 200 model ids come back', async () => {
  const data = Array.from({ length: 500 }, (_, i) => ({ id: `model-${i}` }))
  const result = await run('openai-compatible', { openaiCompatibleModel: 'model-0' }, fakeServer({ list: () => json({ data }) }))
  assert.equal(result.models.length, 200)
})

// ── Local Whisper ────────────────────────────────────────────────────────────

/** An execFile that answers like the probe script, or fails. */
function fakeExec(outcome) {
  const calls = []
  const fn = (file, args, options, callback) => {
    calls.push({ file, args, options })
    if (outcome instanceof Error) callback(outcome)
    else callback(null, outcome, '')
  }
  fn.calls = calls
  return fn
}
const ENGINE = path.resolve(__dirname, '../../engine')
const PY = { python: 'python-under-test', enginePath: '/engine-under-test', env: { PATH: '/bin', HOME: '/home/u' } }

/** A home folder with a Whisper cache, so "is the model downloaded" has something to find. */
function withHome(files = []) {
  const { dir, cleanup } = tempDir('bridgeclip-probe-home-')
  for (const file of files) {
    const target = path.join(dir, '.cache', 'whisper', file)
    fs.mkdirSync(path.dirname(target), { recursive: true })
    fs.writeFileSync(target, 'x')
  }
  return { home: dir, cleanup }
}

test('built-in Whisper: reports the backend, version and device, and whether the chosen model is downloaded', async () => {
  const empty = withHome()
  const cached = withHome(['large-v3-turbo.pt'])
  try {
    const out = '{"backend":"openai-whisper","cuda":false,"version":"20250625"}\n'
    const notYet = await checkLocalWhisper(settings({ localWhisperModel: 'turbo' }), { ...PY, home: empty.home, execFile: fakeExec(out) })
    assert.deepEqual(notYet, { available: true, backend: 'openai-whisper', detail: 'openai-whisper 20250625', device: 'cpu', modelCached: false })
    const there = await checkLocalWhisper(settings({ localWhisperModel: 'turbo' }), { ...PY, home: cached.home, execFile: fakeExec(out) })
    assert.equal(there.modelCached, true, 'turbo is stored as large-v3-turbo.pt')
    // A model name the manager does not know cannot be looked up, so it is not claimed either way.
    const unknown = await checkLocalWhisper(settings({ localWhisperModel: 'large-v2' }), { ...PY, home: empty.home, execFile: fakeExec(out) })
    assert.equal(unknown.modelCached, null)
  } finally { empty.cleanup(); cached.cleanup() }
})

test('built-in Whisper: the probe runs the engine file by path, with the path as an argument and the stripped job environment', async () => {
  const exec = fakeExec('{"backend":"faster-whisper","cuda":true,"version":"1.2.1"}')
  const home = withHome()
  try {
    await checkLocalWhisper(settings({}), { ...PY, home: home.home, execFile: exec })
    const call = exec.calls[0]
    assert.equal(call.file, 'python-under-test')
    assert.equal(call.args[0], '-c')
    assert.equal(call.args[2], localWhisperScript('/engine-under-test'))
    assert.match(call.args[2].replace(/\\/g, '/'), /\/engine-under-test\/clip_engine\/services\/local_whisper\.py$/)
    const code = Buffer.from(/b64decode\('([^']+)'\)/.exec(call.args[1])[1], 'base64').toString('utf-8')
    assert.match(code, /spec_from_file_location/)
    assert.match(code, /describe_environment\(\)/)
    assert.equal(code.includes('engine-under-test'), false, 'the path is an argument, never spliced into the code')
    // The probe no longer decides for itself what is importable: it asks the engine.
    assert.doesNotMatch(code, /import whisper|import torch|import faster_whisper/)
    assert.deepEqual(call.options.env, PY.env)
    assert.ok(call.options.timeout >= 15_000 && call.options.timeout <= 30_000, 'a short, bounded timeout')
    assert.equal(call.options.shell, undefined)
  } finally { home.cleanup() }
})

test('built-in Whisper: a GPU, faster-whisper, noisy output and bad output', async () => {
  const home = withHome()
  try {
    const gpu = await checkLocalWhisper(settings({}), { ...PY, home: home.home, execFile: fakeExec('warning: something\n{"backend":"faster-whisper","cuda":true,"version":"1.2.1"}\n') })
    assert.deepEqual([gpu.available, gpu.backend, gpu.device, gpu.detail], [true, 'faster-whisper', 'cuda', 'faster-whisper 1.2.1'])
    const noVersion = await checkLocalWhisper(settings({}), { ...PY, home: home.home, execFile: fakeExec('{"backend":"openai-whisper","cuda":false}') })
    assert.equal(noVersion.detail, 'openai-whisper')
    for (const output of ['{"backend":null,"cuda":false}', '{"backend":"evil","cuda":true}']) {
      const status = await checkLocalWhisper(settings({}), { ...PY, home: home.home, execFile: fakeExec(output) })
      assert.equal(status.available, false, output)
      assert.match(status.detail, /not installed/)
      assert.match(status.detail, /pip install faster-whisper/)
    }
    // No probe answer at all means Python itself is not usable (for example the Microsoft Store
    // stub for "python3"), which is a different problem from Whisper not being installed.
    for (const output of ['', 'not json', 'Python was not found; run without arguments to install from the Microsoft Store.', '[]']) {
      const status = await checkLocalWhisper(settings({}), { ...PY, home: home.home, execFile: fakeExec(output) })
      assert.equal(status.available, false, output)
      assert.match(status.detail, /Could not run Python/, output)
    }
  } finally { home.cleanup() }
})

test('built-in Whisper: an installed app cannot be told to pip install', async () => {
  const status = await checkLocalWhisper(settings({}), { ...PY, packaged: true, execFile: fakeExec('{"backend":null}') })
  assert.equal(status.available, false)
  assert.doesNotMatch(status.detail, /pip install/)
  assert.match(status.detail, /Whisper server/)
})

test('built-in Whisper: Python failing to run is a message, not an exception', async () => {
  const status = await checkLocalWhisper(settings({}), { ...PY, execFile: fakeExec(new Error('spawn ENOENT')) })
  assert.equal(status.available, false)
  assert.match(status.detail, /Could not run Python/)
  assert.equal(JSON.stringify(status).includes('ENOENT'), false)
})

test('Whisper server: any HTTP answer means something is listening; no answer does not', async () => {
  const server = settings({ localWhisperMode: 'server', localWhisperUrl: 'http://localhost:8000/v1' })
  for (const status of [200, 404, 405, 500]) {
    const fetch = fakeFetch(new Response('', { status }))
    const result = await checkLocalWhisper(server, { ...PY, fetch })
    assert.equal(result.available, true, String(status))
    assert.equal(result.backend, 'server')
    assert.equal(fetch.calls[0].url, 'http://localhost:8000/v1')
    assert.equal(fetch.calls[0].init.redirect, 'manual')
  }
  const down = await checkLocalWhisper(server, { ...PY, fetch: fakeFetch(new TypeError('fetch failed')) })
  assert.deepEqual([down.available, down.detail], [false, 'Could not reach the server.'])
  const unset = await checkLocalWhisper(settings({ localWhisperMode: 'server', localWhisperUrl: '' }), { ...PY, fetch: fakeFetch() })
  assert.deepEqual([unset.available, unset.detail], [false, 'Enter your Whisper server URL first.'])
  const blocked = await checkLocalWhisper(settings({ localWhisperMode: 'server', localWhisperUrl: 'http://169.254.169.254/' }), { ...PY, fetch: fakeFetch() })
  assert.equal(blocked.available, false)
})

test('Whisper server: a public host is refused with the plain reason, and nothing is sent to it', async () => {
  const fetch = fakeFetch()
  for (const url of ['https://api.example.com/v1', 'https://api.openai.com/v1', 'http://8.8.8.8/v1', 'http://100.64.0.5:8000/v1']) {
    const result = await checkLocalWhisper(settings({ localWhisperMode: 'server', localWhisperUrl: url }), { ...PY, fetch })
    assert.equal(result.available, false, url)
    assert.equal(result.backend, null)
  }
  const publicHost = await checkLocalWhisper(settings({ localWhisperMode: 'server', localWhisperUrl: 'https://api.example.com/v1' }), { ...PY, fetch })
  assert.match(publicHost.detail, /Local Whisper must run on this computer or your private network/)
  assert.equal(fetch.calls.length, 0)
  // Private-network servers are fine.
  for (const url of ['http://192.168.1.20:8000/v1', 'http://localhost:8000/v1', 'http://my-nas.local:8000/v1']) {
    const ok = await checkLocalWhisper(settings({ localWhisperMode: 'server', localWhisperUrl: url }), { ...PY, fetch: fakeFetch(new Response('', { status: 200 })) })
    assert.equal(ok.available, true, url)
  }
})

test('the probe parser accepts only the shapes the probe prints', () => {
  assert.equal(parseWhisperProbe('{"backend":"openai-whisper","cuda":true,"version":"1"}').device, 'cuda')
  assert.equal(parseWhisperProbe('{"backend":"openai-whisper","cuda":false}').device, 'cpu')
  assert.equal(parseWhisperProbe('{"backend":"openai-whisper","cuda":"yes"}').device, 'cpu', 'only a real true means a GPU')
  assert.equal(parseWhisperProbe('{"backend":"openai-whisper","device":"cuda"}').device, 'cpu', 'the old device field is not trusted')
  const long = parseWhisperProbe(JSON.stringify({ backend: 'openai-whisper', version: 'x'.repeat(500) }))
  assert.ok(long.detail.length <= 'openai-whisper '.length + 30)
  assert.equal(parseWhisperProbe(JSON.stringify({ backend: 'openai-whisper', version: '1.0 <script>' })).detail.includes('<'), false)
})

// The one test that runs the real probe script with a real Python: proves the command line,
// the encoding and the output parsing work together on this machine.
const python = ['python', 'python3'].find((name) => spawnSync(name, ['--version']).status === 0)
test('the real probe script runs under a real Python and reports an answer', { skip: python ? false : 'no Python on PATH' }, async () => {
  const status = await checkLocalWhisper(settings({ localWhisperModel: 'small' }), { python, enginePath: ENGINE, env: { ...process.env } })
  assert.equal(typeof status.available, 'boolean')
  assert.ok(status.detail.length > 0)
  if (status.available) {
    assert.ok(['openai-whisper', 'faster-whisper'].includes(status.backend))
    assert.ok(['cpu', 'cuda'].includes(status.device))
  } else {
    assert.match(status.detail, /not installed|Could not run Python/)
  }
})

/** An engine folder holding only a stand-in local_whisper.py, so the real probe can be run on any machine. */
function fakeEngine(source) {
  const { dir, cleanup } = tempDir('bridgeclip-fake-engine-')
  const file = localWhisperScript(dir)
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, source)
  return { dir, cleanup }
}

test('the real probe loads the engine file by path and reports exactly what its describe_environment says', { skip: python ? false : 'no Python on PATH' }, async () => {
  const home = withHome()
  try {
    const gpu = fakeEngine('def describe_environment():\n    return {"backend": "faster-whisper", "cuda": True}\n')
    const cpu = fakeEngine('def describe_environment():\n    return {"backend": "openai-whisper", "cuda": False}\n')
    const none = fakeEngine('def describe_environment():\n    return {"backend": None, "cuda": False}\n')
    try {
      const a = await checkLocalWhisper(settings({}), { python, enginePath: gpu.dir, home: home.home, env: { ...process.env } })
      assert.deepEqual([a.available, a.backend, a.device], [true, 'faster-whisper', 'cuda'])
      const b = await checkLocalWhisper(settings({}), { python, enginePath: cpu.dir, home: home.home, env: { ...process.env } })
      assert.deepEqual([b.available, b.backend, b.device], [true, 'openai-whisper', 'cpu'])
      const c = await checkLocalWhisper(settings({}), { python, enginePath: none.dir, home: home.home, env: { ...process.env } })
      assert.equal(c.available, false)
    } finally { gpu.cleanup(); cpu.cleanup(); none.cleanup() }
  } finally { home.cleanup() }
})

test('the real probe treats a broken, missing or hostile engine file as "not available", never as a crash', { skip: python ? false : 'no Python on PATH' }, async () => {
  const home = withHome()
  const sources = [
    'def describe_environment(:\n',                                  // does not even parse
    'raise RuntimeError("boom at import")\n',                        // fails on load
    'def describe_environment():\n    raise RuntimeError("boom")\n', // fails when asked
    'def something_else():\n    pass\n',                            // wrong contract
    'import sys\nprint("noise before the answer")\ndef describe_environment():\n    return {"backend": "openai-whisper", "cuda": False}\n'
  ]
  try {
    const results = []
    for (const source of sources) {
      const engine = fakeEngine(source)
      try { results.push(await checkLocalWhisper(settings({}), { python, enginePath: engine.dir, home: home.home, env: { ...process.env } })) } finally { engine.cleanup() }
    }
    assert.deepEqual(results.slice(0, 4).map((r) => r.available), [false, false, false, false])
    for (const result of results.slice(0, 4)) assert.match(result.detail, /not installed/)
    // Noise printed before the answer does not matter: the last line is the answer.
    assert.equal(results[4].available, true)
    // An engine folder with no such file at all.
    const missing = tempDir('bridgeclip-no-engine-')
    try {
      assert.equal((await checkLocalWhisper(settings({}), { python, enginePath: missing.dir, home: home.home, env: { ...process.env } })).available, false)
    } finally { missing.cleanup() }
  } finally { home.cleanup() }
})

test('with Whisper installed, the real engine file must report it as available (a broken probe cannot hide as "not installed")', { skip: python ? false : 'no Python on PATH' }, async () => {
  const installed = spawnSync(python, ['-c', 'import importlib.util as u,sys; sys.exit(0 if (u.find_spec("faster_whisper") or u.find_spec("whisper")) else 1)']).status === 0
  if (!installed) return // nothing to prove on a machine without Whisper
  const status = await checkLocalWhisper(settings({ localWhisperModel: 'small' }), { python, enginePath: ENGINE, env: { ...process.env } })
  assert.equal(status.available, true, status.detail)
  assert.ok(['openai-whisper', 'faster-whisper'].includes(status.backend))
})
