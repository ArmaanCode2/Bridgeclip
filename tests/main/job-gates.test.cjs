'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const { loadMain } = require('../zernio/support/load-main.cjs')

const { gateJobStart } = loadMain("export * from './src/main/job-gates'")
const { setupGaps, aiReady, transcriptionReady, describeGaps } = loadMain("export * from './src/shared/providers'")

const READY = {
  aiProvider: 'openrouter', openrouterApiKey: 'sk-or', geminiApiKey: 'AIza',
  openaiCompatibleUrl: 'http://localhost:11434/v1', openaiCompatibleModel: 'llama3.1', openaiCompatibleVision: 'off',
  transcriptionSource: 'cloud', localWhisperMode: 'builtin', localWhisperUrl: ''
}
const settings = (patch) => ({ ...READY, ...patch })

/** The rules, written out separately from the implementation. */
function oracle(s, request) {
  const aiOk = s.aiProvider === 'openrouter' ? Boolean(s.openrouterApiKey)
    : s.aiProvider === 'gemini' ? Boolean(s.geminiApiKey)
      : Boolean(s.openaiCompatibleUrl && s.openaiCompatibleModel)
  const speechOk = s.transcriptionSource === 'cloud' ? Boolean(s.openrouterApiKey) : s.localWhisperMode === 'builtin' || Boolean(s.localWhisperUrl)
  if (!aiOk || !speechOk) return 'blocked'
  if (request.clippingMode === 'advanced' && s.aiProvider !== 'openrouter') return 'blocked'
  const mode = request.clippingMode ?? 'quality'
  return `ok:${s.aiProvider !== 'openrouter' && mode === 'economy' ? 'quality' : mode}`
}

test('every provider, transcription, key state, workflow and mode combination is gated exactly as the rules say', () => {
  let checked = 0
  for (const aiProvider of ['openrouter', 'gemini', 'openai-compatible'])
    for (const [transcriptionSource, localWhisperMode] of [['cloud', 'builtin'], ['local', 'builtin'], ['local', 'server']])
      for (const openrouterApiKey of ['', 'sk-or'])
        for (const geminiApiKey of ['', 'AIza'])
          for (const openaiCompatibleUrl of ['', 'http://localhost:11434/v1'])
            for (const openaiCompatibleModel of ['', 'llama3.1'])
              for (const localWhisperUrl of ['', 'http://localhost:8000/v1'])
                for (const workflow of [undefined, 'automatic', 'review'])
                  for (const clippingMode of [undefined, 'quality', 'economy', 'advanced']) {
                    const s = settings({ aiProvider, transcriptionSource, localWhisperMode, openrouterApiKey, geminiApiKey, openaiCompatibleUrl, openaiCompatibleModel, localWhisperUrl })
                    const result = gateJobStart(s, { workflow, clippingMode })
                    const got = result.ok ? `ok:${result.clippingMode}` : 'blocked'
                    assert.equal(got, oracle(s, { workflow, clippingMode }), JSON.stringify({ aiProvider, transcriptionSource, localWhisperMode, openrouterApiKey, geminiApiKey, openaiCompatibleUrl, openaiCompatibleModel, localWhisperUrl, workflow, clippingMode }))
                    if (!result.ok) assert.ok(result.error.length > 10 && result.missing, 'a blocked start explains itself')
                    checked += 1
                  }
  assert.ok(checked > 2000)
})

test('the existing OpenRouter experience is unchanged', () => {
  for (const clippingMode of ['quality', 'economy', 'advanced'])
    for (const workflow of ['automatic', 'review']) {
      const result = gateJobStart(settings({}), { workflow, clippingMode, layoutVision: true })
      assert.deepEqual(result, { ok: true, clippingMode, layoutVision: true })
    }
  const noKey = gateJobStart(settings({ openrouterApiKey: '' }), { workflow: 'automatic' })
  assert.equal(noKey.ok, false)
  assert.equal(noKey.error, 'OpenRouter API key is required for AI clip planning. Go to Settings to add it.')
})

test('a Gemini user with cloud transcription is told why OpenRouter is needed and what to do', () => {
  const result = gateJobStart(settings({ aiProvider: 'gemini', openrouterApiKey: '', transcriptionSource: 'cloud' }), { workflow: 'automatic' })
  assert.equal(result.ok, false)
  assert.match(result.error, /Cloud transcription uses OpenRouter/)
  assert.match(result.error, /Local Whisper/)
  assert.match(result.error, /Settings/)
  // The same user can start once transcription is local, with no OpenRouter key at all.
  const local = gateJobStart(settings({ aiProvider: 'gemini', openrouterApiKey: '', transcriptionSource: 'local' }), { workflow: 'automatic', clippingMode: 'quality' })
  assert.deepEqual(local, { ok: true, clippingMode: 'quality', layoutVision: undefined })
})

test('Review & edit runs on every provider; Advanced explains that it needs OpenRouter', () => {
  for (const aiProvider of ['gemini', 'openai-compatible']) {
    assert.equal(gateJobStart(settings({ aiProvider }), { workflow: 'review' }).ok, true, aiProvider)
  }
  const advanced = gateJobStart(settings({ aiProvider: 'openai-compatible' }), { workflow: 'automatic', clippingMode: 'advanced' })
  assert.equal(advanced.ok, false)
  assert.match(advanced.error, /Advanced mode/)
})

test('Economy is an OpenRouter preset, so other providers run it as Quality', () => {
  for (const aiProvider of ['gemini', 'openai-compatible']) {
    const result = gateJobStart(settings({ aiProvider }), { workflow: 'automatic', clippingMode: 'economy' })
    assert.equal(result.ok && result.clippingMode, 'quality')
  }
})

test('frame analysis follows what the chosen model can do', () => {
  const ask = (patch) => gateJobStart(settings(patch), { workflow: 'automatic', layoutVision: true })
  assert.equal(ask({ aiProvider: 'openrouter' }).layoutVision, true)
  assert.equal(ask({ aiProvider: 'gemini' }).layoutVision, true)
  assert.equal(ask({ aiProvider: 'openai-compatible', openaiCompatibleVision: 'on' }).layoutVision, true)
  assert.equal(ask({ aiProvider: 'openai-compatible', openaiCompatibleVision: 'off' }).layoutVision, false)
  // The person's own choice to turn it off is never overridden.
  assert.equal(gateJobStart(settings({ aiProvider: 'gemini' }), { workflow: 'automatic', layoutVision: false }).layoutVision, false)
})

test('setup gaps, readiness and labels agree for each provider and transcription choice', () => {
  const input = (patch) => ({
    aiProvider: 'openrouter', openrouterConfigured: false, geminiConfigured: false, openaiCompatibleUrl: '', openaiCompatibleModel: '',
    transcriptionSource: 'cloud', localWhisperMode: 'builtin', localWhisperUrl: '', ...patch
  })
  // OpenRouter with cloud transcription needs one key, reported once.
  assert.deepEqual(setupGaps(input({})).map((g) => g.id), ['openrouter-key'])
  // Gemini and cloud transcription: the Gemini key and the OpenRouter key.
  assert.deepEqual(setupGaps(input({ aiProvider: 'gemini' })).map((g) => g.id), ['gemini-key', 'openrouter-key'])
  assert.deepEqual(setupGaps(input({ aiProvider: 'gemini', transcriptionSource: 'local' })).map((g) => g.id), ['gemini-key'])
  assert.deepEqual(setupGaps(input({ aiProvider: 'gemini', geminiConfigured: true, transcriptionSource: 'local' })), [])
  // OpenAI-compatible needs a URL and a model, not a key.
  assert.deepEqual(setupGaps(input({ aiProvider: 'openai-compatible', transcriptionSource: 'local' })).map((g) => g.id), ['openai-url', 'openai-model'])
  assert.deepEqual(setupGaps(input({ aiProvider: 'openai-compatible', openaiCompatibleUrl: 'http://x/v1', transcriptionSource: 'local' })).map((g) => g.id), ['openai-model'])
  assert.deepEqual(setupGaps(input({ aiProvider: 'openai-compatible', openaiCompatibleUrl: 'http://x/v1', openaiCompatibleModel: 'm', transcriptionSource: 'local' })), [])
  // Whitespace is not a value.
  assert.deepEqual(setupGaps(input({ aiProvider: 'openai-compatible', openaiCompatibleUrl: '  ', openaiCompatibleModel: ' ', transcriptionSource: 'local' })).map((g) => g.id), ['openai-url', 'openai-model'])
  // Local Whisper: built-in needs nothing, server needs a URL.
  assert.deepEqual(setupGaps(input({ openrouterConfigured: true, transcriptionSource: 'local', localWhisperMode: 'builtin' })), [])
  assert.deepEqual(setupGaps(input({ openrouterConfigured: true, transcriptionSource: 'local', localWhisperMode: 'server' })).map((g) => g.id), ['whisper-url'])
  // Readiness helpers and the label.
  assert.equal(aiReady(input({ aiProvider: 'gemini', geminiConfigured: true })), true)
  assert.equal(transcriptionReady(input({ transcriptionSource: 'local' })), true)
  assert.equal(transcriptionReady(input({ transcriptionSource: 'cloud' })), false)
  assert.equal(describeGaps([]), '')
  assert.equal(describeGaps(setupGaps(input({}))), 'Add OpenRouter key')
  assert.equal(describeGaps(setupGaps(input({ aiProvider: 'gemini' }))), 'Finish setup')
  for (const gap of setupGaps(input({ aiProvider: 'gemini' }))) assert.doesNotMatch(gap.message, /in Settings/, 'messages must suit the setup card as well as the job error')
})
