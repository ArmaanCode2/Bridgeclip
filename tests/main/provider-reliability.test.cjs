'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { loadMain, tempDir, fakeElectron } = require('../zernio/support/load-main.cjs')

const providers = loadMain("export * from './src/shared/providers'")

function openStore() {
  const { dir, cleanup } = tempDir('bridgeclip-provider-reliability-')
  const { electron } = fakeElectron(dir)
  const store = loadMain("export * from './src/main/settings-store'", { electron })
  const file = path.join(dir, 'userData', 'settings.json')
  return {
    store, dir, cleanup,
    raw: () => JSON.parse(fs.readFileSync(file, 'utf8')),
    writeRaw: (data) => { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, JSON.stringify(data)) }
  }
}

test('choosing Gemini or a custom endpoint without an OpenRouter key moves transcription to Local Whisper', () => {
  const { aiProviderChange } = providers
  const noKeyCloud = { openrouterConfigured: false, transcriptionSource: 'cloud' }
  assert.deepEqual(aiProviderChange('gemini', noKeyCloud), { aiProvider: 'gemini', transcriptionSource: 'local' })
  assert.deepEqual(aiProviderChange('openai-compatible', noKeyCloud), { aiProvider: 'openai-compatible', transcriptionSource: 'local' })
  // With an OpenRouter key, cloud transcription still works, so it is left alone.
  assert.deepEqual(aiProviderChange('gemini', { openrouterConfigured: true, transcriptionSource: 'cloud' }), { aiProvider: 'gemini' })
  // Already local, or going back to OpenRouter: only the provider changes.
  assert.deepEqual(aiProviderChange('gemini', { openrouterConfigured: false, transcriptionSource: 'local' }), { aiProvider: 'gemini' })
  assert.deepEqual(aiProviderChange('openrouter', noKeyCloud), { aiProvider: 'openrouter' })
})

test('going back to OpenRouter undoes an automatic move, so looking at the options changes nothing', () => {
  const { aiProviderChange } = providers
  const start = { openrouterConfigured: false, transcriptionSource: 'cloud' }
  const toGemini = aiProviderChange('gemini', start, false)
  const afterGemini = { ...start, ...toGemini }
  // Arrow keys pass over the custom endpoint too: transcription is already local.
  assert.deepEqual(aiProviderChange('openai-compatible', afterGemini, true), { aiProvider: 'openai-compatible' })
  assert.deepEqual(aiProviderChange('openrouter', afterGemini, true), { aiProvider: 'openrouter', transcriptionSource: 'cloud' })
  // A Local Whisper choice the person made themselves is never undone.
  assert.deepEqual(aiProviderChange('openrouter', { openrouterConfigured: false, transcriptionSource: 'local' }, false), { aiProvider: 'openrouter' })
})

test('after that move, the setup check no longer asks for an OpenRouter key', () => {
  const { aiProviderChange, setupGaps } = providers
  const before = {
    aiProvider: 'openrouter', openrouterConfigured: false, geminiConfigured: true, openaiCompatibleUrl: '', openaiCompatibleModel: '',
    transcriptionSource: 'cloud', localWhisperMode: 'builtin', localWhisperUrl: ''
  }
  const after = { ...before, ...aiProviderChange('gemini', before) }
  assert.deepEqual(setupGaps(after), [])
})

test('Ollama is recognised by its port or host name, and nothing else is', () => {
  const { looksLikeOllama } = providers
  for (const url of ['http://localhost:11434/v1', 'http://127.0.0.1:11434', 'http://ollama.lan:8080/v1', 'http://192.168.1.5:11434/v1']) {
    assert.equal(looksLikeOllama(url), true, url)
  }
  for (const url of ['http://localhost:1234/v1', 'http://localhost:3001/v1', 'https://api.openai.com/v1', '', 'not a url']) {
    assert.equal(looksLikeOllama(url), false, url)
  }
})

test('the endpoint wait time defaults to 30 minutes, saves a listed value and reaches the engine in seconds', () => {
  const t = openStore()
  try {
    assert.equal(t.store.loadSettings().openaiCompatibleTimeout, '30')
    assert.equal(t.store.publicSettings(t.store.loadSettings()).openaiCompatibleTimeout, '30')
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).OPENAI_COMPATIBLE_TIMEOUT_SECONDS, '1800')
    const saved = t.store.savePublicSettings({ openaiCompatibleTimeout: '120' })
    assert.equal(saved.openaiCompatibleTimeout, '120')
    assert.equal(t.raw().openaiCompatibleTimeout, '120')
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).OPENAI_COMPATIBLE_TIMEOUT_SECONDS, '7200')
    for (const value of ['0', '15', '999', 'forever', '']) {
      assert.throws(() => t.store.savePublicSettings({ openaiCompatibleTimeout: value }), /wait time|Invalid/, JSON.stringify(value))
    }
    assert.equal(t.store.loadSettings().openaiCompatibleTimeout, '120')
  } finally { t.cleanup() }
})

test('a saved wait time that is no longer offered loads as the default, keeping the rest of the file', () => {
  const t = openStore()
  try {
    for (const bad of ['7', 30, null, { minutes: 30 }]) {
      t.writeRaw({
        version: 12, openrouterApiKey: '', zernioApiKey: '', jevEnabled: 'off', jevVisualContext: 'off', sourceContextWebResearch: 'off',
        outputDirectory: path.join(t.dir, 'clips'), pythonPath: 'python3', customVocabulary: 'kept', openaiCompatibleTimeout: bad
      })
      const loaded = t.store.loadSettings()
      assert.equal(loaded.openaiCompatibleTimeout, '30', JSON.stringify(bad))
      assert.equal(loaded.customVocabulary, 'kept')
    }
  } finally { t.cleanup() }
})

test('the test result suggests switching image reading only for a custom endpoint, and only when it disagrees', () => {
  const { visionAdvice } = providers
  assert.equal(visionAdvice('openai-compatible', true, false), 'turn-on')
  assert.equal(visionAdvice('openai-compatible', false, true), 'turn-off')
  assert.equal(visionAdvice('openai-compatible', true, true), null)
  assert.equal(visionAdvice('openai-compatible', false, false), null)
  assert.equal(visionAdvice('openai-compatible', null, true), null, 'not tested')
  assert.equal(visionAdvice('gemini', false, false), null, 'Gemini has no switch')
  assert.equal(visionAdvice('openrouter', true, false), null)
})

test('titles and captions need the chosen AI provider and a way to transcribe the clip', () => {
  const { metadataWriterGap } = providers
  const ready = {
    aiProvider: 'gemini', openrouterConfigured: false, geminiConfigured: true, openaiCompatibleUrl: '', openaiCompatibleModel: '',
    transcriptionSource: 'local', localWhisperMode: 'builtin', localWhisperUrl: ''
  }
  assert.equal(metadataWriterGap(ready), null, 'Gemini with Local Whisper needs no OpenRouter key')
  assert.match(metadataWriterGap({ ...ready, geminiConfigured: false }), /written by your AI provider, Google Gemini/)
  assert.match(metadataWriterGap({ ...ready, transcriptionSource: 'cloud' }), /Add an OpenRouter key, or switch Speech to text to Local Whisper/)
  assert.equal(metadataWriterGap({ ...ready, transcriptionSource: 'cloud', openrouterConfigured: true }), null)
  assert.match(metadataWriterGap({ ...ready, localWhisperMode: 'server' }), /Whisper server URL/)
  assert.equal(metadataWriterGap({ ...ready, aiProvider: 'openai-compatible', openaiCompatibleUrl: 'http://localhost:1234/v1', openaiCompatibleModel: 'qwen3' }), null)
  assert.equal(metadataWriterGap({ ...ready, aiProvider: 'openrouter', openrouterConfigured: true, transcriptionSource: 'cloud' }), null)
})
