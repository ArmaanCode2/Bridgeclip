'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { loadMain, tempDir, fakeElectron } = require('../zernio/support/load-main.cjs')

/** The settings store on a temp user-data folder, with a safeStorage whose output is visible ("enc:" prefix). */
function openStore() {
  const { dir, cleanup } = tempDir('bridgeclip-providers-')
  const { electron } = fakeElectron(dir)
  const store = loadMain("export * from './src/main/settings-store'", { electron })
  const file = path.join(dir, 'userData', 'settings.json')
  return {
    store, file, dir, cleanup,
    raw: () => JSON.parse(fs.readFileSync(file, 'utf8')),
    writeRaw: (data) => { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, JSON.stringify(data)) },
    /** What an encrypted secret looks like on disk under the fake keychain. */
    sealed: (value) => ({ scheme: 'safeStorage', value: Buffer.from(`enc:${value}`).toString('base64') })
  }
}

/** Saves a custom endpoint as a card and chooses it, as clicking the card in Settings does. */
function chooseModel(store, fields = {}) {
  const saved = store.saveModelPreset({ name: '', url: 'http://localhost:11434/v1', model: 'llama3.1', apiKey: null, vision: 'off', timeout: '30', ...fields })
  const id = saved.modelPresets.at(-1).id
  return { id, view: store.savePublicSettings({ aiProvider: 'openai-compatible', activeModelPreset: id }) }
}

/** A saved file in the shape written before provider choice existed. */
function preFeatureFile(dir, sealed) {
  return {
    version: 12,
    openrouterApiKey: sealed('sk-or-existing'), zernioApiKey: '',
    jevEnabled: 'off', jevVisualContext: 'off', sourceContextWebResearch: 'off',
    outputDirectory: path.join(dir, 'clips'), pythonPath: 'python3', customVocabulary: 'BridgeMind'
  }
}

test('a fresh install and a pre-feature settings file both stay on OpenRouter with cloud transcription', () => {
  const t = openStore()
  try {
    const fresh = t.store.loadSettings()
    assert.equal(fresh.aiProvider, 'openrouter')
    assert.equal(fresh.transcriptionSource, 'cloud')
    assert.equal(fresh.localWhisperMode, 'builtin')
    assert.equal(fresh.localWhisperModel, 'small')
    assert.equal(fresh.geminiModel, 'gemini-3.8-flash')
    // Nothing is pre-filled: an empty URL or model means "not set".
    assert.equal(fresh.openaiCompatibleUrl, '')
    assert.equal(fresh.openaiCompatibleModel, '')
    assert.equal(fresh.localWhisperUrl, '')
    assert.equal(fresh.openaiCompatibleVision, 'off')

    t.writeRaw(preFeatureFile(t.dir, t.sealed))
    const migrated = t.store.loadSettings()
    assert.equal(migrated.openrouterApiKey, 'sk-or-existing')
    assert.equal(migrated.aiProvider, 'openrouter')
    assert.equal(migrated.transcriptionSource, 'cloud')
    assert.equal(migrated.customVocabulary, 'BridgeMind')
    const env = t.store.getSettingsForBridge(migrated)
    assert.equal(env.OPENROUTER_API_KEY, 'sk-or-existing')
    assert.equal(env.AI_PROVIDER, 'openrouter')
  } finally { t.cleanup() }
})

test('provider keys are encrypted on disk and never appear in the public view', () => {
  const t = openStore()
  try {
    t.store.replaceApiKey('geminiApiKey', 'AIzaGeminiSecret')
    assert.throws(() => t.store.replaceApiKey('openaiCompatibleApiKey', 'sk-custom-secret'), /Choose one of your saved models first/)
    chooseModel(t.store)
    t.store.replaceApiKey('openaiCompatibleApiKey', 'sk-custom-secret')
    t.store.replaceApiKey('openrouterApiKey', 'sk-or-secret')
    const text = fs.readFileSync(t.file, 'utf8')
    for (const secret of ['AIzaGeminiSecret', 'sk-custom-secret', 'sk-or-secret']) {
      assert.equal(text.includes(secret), false, `${secret} is stored in plain text`)
      assert.equal(text.includes(Buffer.from(secret).toString('base64')), false, `${secret} is stored as plain base64`)
    }
    const stored = t.raw()
    for (const secret of ['geminiApiKey', 'openaiCompatibleApiKey', 'openrouterApiKey'].map((key) => stored[key]).concat(stored.modelPresets[0].apiKey)) {
      assert.equal(secret.scheme, 'safeStorage')
      assert.match(Buffer.from(secret.value, 'base64').toString(), /^enc:/)
    }
    const loaded = t.store.loadSettings()
    assert.equal(loaded.geminiApiKey, 'AIzaGeminiSecret')
    assert.equal(loaded.openaiCompatibleApiKey, 'sk-custom-secret')
    const publicView = JSON.stringify(t.store.publicSettings(loaded))
    for (const secret of ['AIzaGeminiSecret', 'sk-custom-secret', 'sk-or-secret']) assert.equal(publicView.includes(secret), false)
    const view = t.store.publicSettings(loaded)
    assert.equal(view.geminiConfigured, true)
    assert.equal(view.openaiCompatibleConfigured, true)
    assert.equal('geminiApiKey' in view, false)
    // Removing a key clears it.
    t.store.replaceApiKey('geminiApiKey', '')
    assert.equal(t.store.publicSettings(t.store.loadSettings()).geminiConfigured, false)
    assert.throws(() => t.store.replaceApiKey('notAKey', 'x'), /Invalid API key update/)
  } finally { t.cleanup() }
})

test('placeholder values from development builds are cleared; real choices are kept', () => {
  const t = openStore()
  try {
    // A file written by the unfinished first attempt: placeholders the person never typed, no localWhisperMode.
    t.writeRaw({
      ...preFeatureFile(t.dir, t.sealed),
      aiProvider: 'openrouter', geminiModel: 'gemini-2.5-flash',
      openaiCompatibleUrl: 'http://localhost:11434/v1', openaiCompatibleModel: 'llama3',
      transcriptionSource: 'local', localWhisperUrl: 'http://127.0.0.1:8000/v1', localWhisperModel: 'whisper-1'
    })
    const cleared = t.store.loadSettings()
    assert.equal(cleared.openaiCompatibleUrl, '')
    assert.equal(cleared.openaiCompatibleModel, '')
    assert.equal(cleared.localWhisperUrl, '')
    assert.equal(cleared.localWhisperModel, 'small')
    assert.equal(cleared.geminiModel, 'gemini-3.8-flash')
    assert.equal(cleared.transcriptionSource, 'local')
    assert.equal(cleared.localWhisperMode, 'builtin')

    // If the section was the active one, the values were a real choice.
    t.writeRaw({ ...preFeatureFile(t.dir, t.sealed), aiProvider: 'openai-compatible', openaiCompatibleUrl: 'http://localhost:11434/v1', openaiCompatibleModel: 'llama3' })
    const kept = t.store.loadSettings()
    assert.equal(kept.openaiCompatibleUrl, 'http://localhost:11434/v1')
    assert.equal(kept.openaiCompatibleModel, 'llama3')

    // That choice became a saved model, which a file written by this version keeps whatever is chosen.
    t.store.savePublicSettings({ aiProvider: 'openrouter' })
    assert.deepEqual(t.store.loadSettings().modelPresets.map((p) => [p.url, p.model]), [['http://localhost:11434/v1', 'llama3']])
    assert.equal(t.store.loadSettings().openaiCompatibleUrl, 'http://localhost:11434/v1')
    assert.equal(t.raw().localWhisperMode, 'builtin')
  } finally { t.cleanup() }
})

test('a damaged provider value in the saved file falls back instead of locking the app', () => {
  const t = openStore()
  try {
    t.writeRaw({
      ...preFeatureFile(t.dir, t.sealed),
      aiProvider: 'bogus', openaiCompatibleUrl: 'ftp://nope', openaiCompatibleModel: 'has spaces', openaiCompatibleVision: 'maybe',
      transcriptionSource: 7, localWhisperMode: 'sideways', localWhisperUrl: 'http://u:p@host/', localWhisperModel: '../etc', geminiModel: ''
    })
    const loaded = t.store.loadSettings()
    assert.equal(loaded.aiProvider, 'openrouter')
    assert.equal(loaded.openaiCompatibleUrl, '')
    assert.equal(loaded.openaiCompatibleModel, '')
    assert.equal(loaded.openaiCompatibleVision, 'off')
    assert.equal(loaded.transcriptionSource, 'cloud')
    assert.equal(loaded.localWhisperMode, 'builtin')
    assert.equal(loaded.localWhisperUrl, '')
    assert.equal(loaded.localWhisperModel, 'small')
    assert.equal(loaded.geminiModel, 'gemini-3.8-flash')
    assert.equal(loaded.openrouterApiKey, 'sk-or-existing', 'the saved key must survive')
  } finally { t.cleanup() }
})

const ACCEPTED_URLS = [
  ['http://localhost:11434/v1', 'http://localhost:11434/v1'],
  ['http://127.0.0.1:8000/v1/', 'http://127.0.0.1:8000/v1'],
  ['http://192.168.1.50:11434/v1', 'http://192.168.1.50:11434/v1'],
  ['http://10.0.0.5/v1', 'http://10.0.0.5/v1'],
  ['http://172.20.0.2:8080/v1', 'http://172.20.0.2:8080/v1'],
  ['http://[::1]:8080/v1', 'http://[::1]:8080/v1'],
  ['http://my-nas.local:9000/v1', 'http://my-nas.local:9000/v1'],
  ['http://mini:8000/v1', 'http://mini:8000/v1'],
  ['https://api.openai.com/v1', 'https://api.openai.com/v1'],
  ['  https://example.com/v1  ', 'https://example.com/v1']
]
const REJECTED_URLS = [
  'ftp://host/v1', 'file:///etc/passwd', 'javascript:alert(1)', 'not a url', '//host/v1',
  'http://user:pass@localhost:11434/v1', 'http://localhost:11434/v1?key=abc', 'http://localhost:11434/v1#frag',
  'http://example.com/v1', // public hosts need https
  'http://8.8.8.8/v1',
  'http://169.254.169.254/latest/meta-data', 'http://0.0.0.0:8000/v1', 'http://metadata.google.internal/',
  'http://[fe80::1]/v1', 'http://[::ffff:169.254.169.254]/', 'http://[::]/v1', 'http://224.0.0.1/',
  `https://example.com/${'a'.repeat(520)}`
]

const WHISPER_PRIVACY_MESSAGE = /Local Whisper must run on this computer or your private network\. For a cloud service use Cloud transcription\./
const isPublicOnly = (input) => /api\.openai\.com|example\.com/.test(input)

test('endpoint URLs: accepted forms are normalised, unsafe ones are refused and change nothing', () => {
  const t = openStore()
  try {
    // The AI endpoint is saved with its model card.
    const { id } = chooseModel(t.store)
    const card = (url) => ({ id, name: '', url, model: 'llama3.1', apiKey: null, vision: 'off', timeout: '30' })
    for (const [input, expected] of ACCEPTED_URLS) {
      assert.equal(t.store.saveModelPreset(card(input)).openaiCompatibleUrl, expected, `card: ${input}`)
    }
    t.store.saveModelPreset(card('http://localhost:9000/v1'))
    for (const input of REJECTED_URLS) {
      assert.throws(() => t.store.saveModelPreset(card(input)), Error, `card: ${input} should be refused`)
      assert.equal(t.store.loadSettings().openaiCompatibleUrl, 'http://localhost:9000/v1', `card changed after refusing ${input}`)
    }
    // A Whisper server must be local, so the public https entries only apply to the AI endpoint.
    for (const [input, expected] of ACCEPTED_URLS.filter(([input]) => !isPublicOnly(input))) {
      assert.equal(t.store.savePublicSettings({ localWhisperUrl: input }).localWhisperUrl, expected, `localWhisperUrl: ${input}`)
    }
    t.store.savePublicSettings({ localWhisperUrl: 'http://localhost:9000/v1' })
    for (const input of REJECTED_URLS) {
      assert.throws(() => t.store.savePublicSettings({ localWhisperUrl: input }), Error, `localWhisperUrl: ${input} should be refused`)
      assert.equal(t.store.loadSettings().localWhisperUrl, 'http://localhost:9000/v1', `localWhisperUrl changed after refusing ${input}`)
    }
  } finally { t.cleanup() }
})

test('a local Whisper server must be on this computer or a private network, with a plain reason when it is not', () => {
  const t = openStore()
  try {
    t.store.savePublicSettings({ localWhisperUrl: 'http://localhost:9000/v1' })
    const publicHosts = [
      'https://api.openai.com/v1', 'https://whisper.example.com/v1', 'https://8.8.8.8/v1', 'http://example.com/v1',
      'https://[2606:4700::1111]/v1', 'https://100.63.255.255/v1', 'https://100.128.0.1/v1'
    ]
    for (const url of publicHosts) {
      assert.throws(() => t.store.savePublicSettings({ localWhisperUrl: url }), (error) => {
        // http to a public host is refused for its own reason first; every https one gets the local-only reason.
        return url.startsWith('https') ? WHISPER_PRIVACY_MESSAGE.test(error.message) : /https|Local Whisper/.test(error.message)
      }, url)
      assert.equal(t.store.loadSettings().localWhisperUrl, 'http://localhost:9000/v1', `unchanged after ${url}`)
    }
    for (const url of ['http://localhost:8000/v1', 'http://127.0.0.1:8000/v1', 'http://[::1]:8000/v1', 'http://192.168.1.20:8000/v1',
      'http://10.1.2.3/v1', 'http://172.16.0.9/v1', 'http://my-nas.local:8000/v1', 'http://mini:8000/v1', 'https://localhost:8443/v1', 'https://192.168.0.5/v1']) {
      assert.equal(t.store.savePublicSettings({ localWhisperUrl: url }).localWhisperUrl, url.replace(/\/+$/, ''), url)
    }
    // The AI endpoint is not subject to this rule: a hosted service is exactly what it is for.
    assert.equal(chooseModel(t.store, { url: 'https://api.openai.com/v1' }).view.openaiCompatibleUrl, 'https://api.openai.com/v1')
  } finally { t.cleanup() }
})

test('a Whisper URL saved before that rule, which points at a public host, loads as not configured and breaks nothing else', () => {
  const t = openStore()
  try {
    for (const url of ['https://api.openai.com/v1', 'https://whisper.example.com/v1', 'http://8.8.8.8/v1']) {
      t.writeRaw({
        ...preFeatureFile(t.dir, t.sealed), version: 12, aiProvider: 'gemini', transcriptionSource: 'local', localWhisperMode: 'server',
        localWhisperUrl: url, localWhisperModel: 'large-v3', geminiModel: 'gemini-3.8-flash', customVocabulary: 'kept'
      })
      const loaded = t.store.loadSettings()
      assert.equal(loaded.localWhisperUrl, '', url)
      assert.equal(loaded.localWhisperMode, 'server')
      assert.equal(loaded.localWhisperModel, 'large-v3')
      assert.equal(loaded.aiProvider, 'gemini')
      assert.equal(loaded.customVocabulary, 'kept')
      assert.equal(t.store.getSettingsForBridge(loaded).LOCAL_WHISPER_URL, '', 'and the engine is never told about it')
    }
    // A local one saved the same way survives.
    t.writeRaw({ ...preFeatureFile(t.dir, t.sealed), version: 12, transcriptionSource: 'local', localWhisperMode: 'server', localWhisperUrl: 'http://192.168.1.20:8000/v1' })
    assert.equal(t.store.loadSettings().localWhisperUrl, 'http://192.168.1.20:8000/v1')
  } finally { t.cleanup() }
})

test('carrier-grade NAT and Tailscale addresses (100.64.0.0/10) are refused for both URLs, matching the engine network guard', () => {
  const t = openStore()
  try {
    const blocked = ['http://100.64.0.1/v1', 'http://100.100.100.200/v1', 'http://100.127.255.254:8000/v1', 'https://100.64.0.1/v1',
      'http://100.96.0.7/v1', 'http://[::ffff:6440:1]/v1']
    const { id } = chooseModel(t.store, { url: 'http://localhost:9000/v1' })
    const card = (url) => ({ id, name: '', url, model: 'llama3.1', apiKey: null, vision: 'off', timeout: '30' })
    t.store.savePublicSettings({ localWhisperUrl: 'http://localhost:9000/v1' })
    for (const url of blocked) {
      assert.throws(() => t.store.saveModelPreset(card(url)), Error, `card: ${url}`)
      assert.throws(() => t.store.savePublicSettings({ localWhisperUrl: url }), Error, `localWhisperUrl: ${url}`)
      assert.equal(t.store.loadSettings().openaiCompatibleUrl, 'http://localhost:9000/v1')
      assert.equal(t.store.loadSettings().localWhisperUrl, 'http://localhost:9000/v1')
    }
    // Just outside the range is an ordinary public address: fine over https for the AI endpoint, never for local Whisper.
    for (const url of ['https://100.63.255.255/v1', 'https://100.128.0.1/v1']) {
      assert.equal(t.store.saveModelPreset(card(url)).openaiCompatibleUrl, url)
      assert.throws(() => t.store.savePublicSettings({ localWhisperUrl: url }), WHISPER_PRIVACY_MESSAGE)
    }
    // Saved files with such an address load as not configured instead of failing to load.
    t.writeRaw({ ...preFeatureFile(t.dir, t.sealed), version: 12, aiProvider: 'openai-compatible', openaiCompatibleUrl: 'http://100.64.0.1/v1', localWhisperMode: 'server', localWhisperUrl: 'http://100.64.0.1/v1' })
    const loaded = t.store.loadSettings()
    assert.equal(loaded.openaiCompatibleUrl, '')
    assert.equal(loaded.localWhisperUrl, '')
  } finally { t.cleanup() }
})

test('a saved model needs a URL and a model; with none chosen the endpoint reads as not configured', () => {
  const t = openStore()
  try {
    const card = { name: '', url: 'http://localhost:11434/v1', model: 'llama3.1', apiKey: null, vision: 'off', timeout: '30' }
    assert.throws(() => t.store.saveModelPreset({ ...card, url: '' }), /Enter the endpoint URL/)
    assert.throws(() => t.store.saveModelPreset({ ...card, model: '  ' }), /Enter the model name/)
    const { id } = chooseModel(t.store)
    t.store.savePublicSettings({ localWhisperMode: 'server', localWhisperUrl: 'http://localhost:8000/v1' })
    const cleared = t.store.savePublicSettings({ activeModelPreset: '', localWhisperUrl: '' })
    assert.equal(cleared.openaiCompatibleUrl, '')
    assert.equal(cleared.openaiCompatibleModel, '')
    assert.equal(cleared.localWhisperUrl, '')
    assert.equal(cleared.modelPresets[0].id, id, 'the card itself stays')
    const reloaded = t.store.publicSettings(t.store.loadSettings())
    assert.equal(reloaded.openaiCompatibleUrl, '')
    assert.equal(reloaded.localWhisperUrl, '')
    // A blank Gemini or Whisper model returns to the default, because those always need a model.
    const blanks = t.store.savePublicSettings({ geminiModel: '', localWhisperMode: 'builtin', localWhisperModel: '' })
    assert.equal(blanks.geminiModel, 'gemini-3.8-flash')
    assert.equal(blanks.localWhisperModel, 'small')
  } finally { t.cleanup() }
})

test('model names: safe names are accepted; spaces, separators and traversal are refused', () => {
  const t = openStore()
  try {
    const { id } = chooseModel(t.store)
    const card = (model) => ({ id, name: '', url: 'http://localhost:11434/v1', model, apiKey: null, vision: 'off', timeout: '30' })
    for (const name of ['gpt-4o-mini', 'llama3.1:8b', 'models/gemini-3.8-flash', 'org/model@v2', 'Qwen2.5-7B-Instruct+q4']) {
      assert.equal(t.store.saveModelPreset(card(name)).openaiCompatibleModel, name)
      assert.equal(t.store.savePublicSettings({ geminiModel: name }).geminiModel, name)
    }
    for (const name of ['my model', 'a;b', 'a\nb', '-leading-dash', '$(whoami)', 'x'.repeat(201), 'a"b']) {
      assert.throws(() => t.store.saveModelPreset(card(name)), /Model name/, JSON.stringify(name))
      assert.throws(() => t.store.savePublicSettings({ geminiModel: name }), /Gemini model/, JSON.stringify(name))
    }
    assert.equal(t.store.loadSettings().openaiCompatibleModel, 'Qwen2.5-7B-Instruct+q4', 'a refused name changes nothing')
  } finally { t.cleanup() }
})

test('Whisper model: built-in mode takes only known models, server mode takes any safe name', () => {
  const t = openStore()
  try {
    for (const name of ['tiny', 'base', 'small', 'medium', 'turbo', 'large-v3', 'small.en', 'large-v3-turbo']) {
      assert.equal(t.store.savePublicSettings({ localWhisperMode: 'builtin', localWhisperModel: name }).localWhisperModel, name)
    }
    for (const name of ['whisper-1', '../x', 'my model', 'Systran/faster-whisper-small']) {
      assert.throws(() => t.store.savePublicSettings({ localWhisperMode: 'builtin', localWhisperModel: name }), /built-in Whisper models/, name)
    }
    for (const name of ['whisper-1', 'Systran/faster-whisper-small', 'large-v3']) {
      assert.equal(t.store.savePublicSettings({ localWhisperMode: 'server', localWhisperModel: name }).localWhisperModel, name)
    }
    for (const name of ['my model', '../x\n']) {
      assert.throws(() => t.store.savePublicSettings({ localWhisperMode: 'server', localWhisperModel: name }), /Whisper model/, JSON.stringify(name))
    }
  } finally { t.cleanup() }
})

test('unknown enum values are refused, not coerced', () => {
  const t = openStore()
  try {
    assert.throws(() => t.store.savePublicSettings({ aiProvider: 'bogus' }), /AI provider/)
    assert.throws(() => t.store.savePublicSettings({ transcriptionSource: 'cloudish' }), /transcription source/)
    assert.throws(() => t.store.savePublicSettings({ localWhisperMode: 'sideways' }), /Whisper mode/)
    const card = { name: '', url: 'http://localhost:11434/v1', model: 'llama3.1', apiKey: null, vision: 'off', timeout: '30' }
    assert.throws(() => t.store.saveModelPreset({ ...card, vision: 'maybe' }), /image reading/)
    assert.throws(() => t.store.saveModelPreset({ ...card, timeout: '7' }), /wait time/)
    assert.throws(() => t.store.savePublicSettings({ activeModelPreset: 'not-a-card' }), /no longer exists/)
    const loaded = t.store.loadSettings()
    assert.equal(loaded.aiProvider, 'openrouter')
    assert.equal(loaded.transcriptionSource, 'cloud')
  } finally { t.cleanup() }
})

// The engine reads exactly these names; if one is missing the setting silently does nothing.
const ENGINE_ENV = ['AI_PROVIDER', 'GEMINI_API_KEY', 'GEMINI_MODEL', 'OPENAI_COMPATIBLE_URL', 'OPENAI_COMPATIBLE_API_KEY',
  'OPENAI_COMPATIBLE_MODEL', 'OPENAI_COMPATIBLE_VISION', 'TRANSCRIPTION_SOURCE', 'LOCAL_WHISPER_MODE', 'LOCAL_WHISPER_URL', 'LOCAL_WHISPER_MODEL', 'OPENROUTER_API_KEY']

test('the engine environment carries every provider setting under the names the engine reads', () => {
  const t = openStore()
  try {
    t.store.replaceApiKey('geminiApiKey', 'AIzaGemini')
    t.store.replaceApiKey('openrouterApiKey', 'sk-or')
    chooseModel(t.store, { apiKey: 'sk-custom', vision: 'on' })
    t.store.savePublicSettings({ transcriptionSource: 'local', localWhisperMode: 'server', localWhisperUrl: 'http://localhost:8000/v1', localWhisperModel: 'large-v3' })
    const env = t.store.getSettingsForBridge(t.store.loadSettings())
    for (const name of ENGINE_ENV) assert.ok(name in env, `${name} is missing`)
    assert.deepEqual(
      { p: env.AI_PROVIDER, u: env.OPENAI_COMPATIBLE_URL, k: env.OPENAI_COMPATIBLE_API_KEY, m: env.OPENAI_COMPATIBLE_MODEL, v: env.OPENAI_COMPATIBLE_VISION },
      { p: 'openai-compatible', u: 'http://localhost:11434/v1', k: 'sk-custom', m: 'llama3.1', v: 'true' })
    assert.deepEqual(
      { s: env.TRANSCRIPTION_SOURCE, mode: env.LOCAL_WHISPER_MODE, url: env.LOCAL_WHISPER_URL, model: env.LOCAL_WHISPER_MODEL },
      { s: 'local', mode: 'server', url: 'http://localhost:8000/v1', model: 'large-v3' })
    // Keys of providers that are not in use are not handed to the engine.
    assert.equal(env.GEMINI_API_KEY, '')
    assert.equal(env.OPENROUTER_API_KEY, '', 'no OpenRouter key when OpenRouter does neither planning nor transcription')

    t.store.savePublicSettings({ aiProvider: 'gemini', transcriptionSource: 'cloud' })
    const gemini = t.store.getSettingsForBridge(t.store.loadSettings())
    assert.equal(gemini.AI_PROVIDER, 'gemini')
    assert.equal(gemini.GEMINI_API_KEY, 'AIzaGemini')
    assert.equal(gemini.GEMINI_MODEL, 'gemini-3.8-flash')
    assert.equal(gemini.OPENAI_COMPATIBLE_API_KEY, '')
    assert.equal(gemini.OPENROUTER_API_KEY, 'sk-or', 'cloud transcription needs the OpenRouter key')
    assert.equal(gemini.LOCAL_WHISPER_URL, '', 'the server URL is only sent when it is used')
  } finally { t.cleanup() }
})

test('web research is forced off with another provider; the review and visual context follow the provider', () => {
  const t = openStore()
  try {
    t.store.replaceApiKey('openrouterApiKey', 'sk-or')
    t.store.replaceApiKey('geminiApiKey', 'AIzaGemini')
    t.store.savePublicSettings({ jevEnabled: 'on', jevVisualContext: 'on', sourceContextWebResearch: 'on' })
    const onOpenRouter = t.store.getSettingsForBridge(t.store.loadSettings())
    assert.equal(onOpenRouter.JEV_ENABLED, 'true')
    assert.equal(onOpenRouter.JEV_VISUAL_CONTEXT, 'true')
    assert.equal(onOpenRouter.SOURCE_CONTEXT_WEB_RESEARCH, 'true')

    for (const provider of ['gemini', 'openai-compatible']) {
      t.store.savePublicSettings({ aiProvider: provider })
      const env = t.store.getSettingsForBridge(t.store.loadSettings())
      // The review runs on the chosen model, so the person's choices reach the engine.
      assert.equal(env.JEV_ENABLED, 'true', provider)
      // Frames go only to a model that reads images: Gemini does; a custom endpoint only when marked.
      assert.equal(env.JEV_VISUAL_CONTEXT, provider === 'gemini' ? 'true' : 'false', provider)
      // Web research is an OpenRouter tool.
      assert.equal(env.SOURCE_CONTEXT_WEB_RESEARCH, 'false', provider)
    }
    // The saved preferences are kept for when the person switches back.
    assert.equal(t.store.loadSettings().jevEnabled, 'on')
  } finally { t.cleanup() }
})

test('saving unrelated settings through the public API keeps every provider choice and key', () => {
  const t = openStore()
  try {
    t.store.replaceApiKey('geminiApiKey', 'AIzaGemini')
    t.store.savePublicSettings({ aiProvider: 'gemini', transcriptionSource: 'local', localWhisperModel: 'medium' })
    // The assistant's update_settings and the vocabulary field both save a partial object like this.
    t.store.savePublicSettings({ customVocabulary: 'BridgeMind' })
    const loaded = t.store.loadSettings()
    assert.equal(loaded.aiProvider, 'gemini')
    assert.equal(loaded.geminiApiKey, 'AIzaGemini')
    assert.equal(loaded.transcriptionSource, 'local')
    assert.equal(loaded.localWhisperModel, 'medium')
    assert.equal(loaded.customVocabulary, 'BridgeMind')
  } finally { t.cleanup() }
})

test('visual context reaches a custom endpoint only when it is marked as able to read images', () => {
  const t = openStore()
  try {
    t.store.savePublicSettings({ jevVisualContext: 'on' })
    const { id } = chooseModel(t.store, { url: 'http://localhost:1234/v1', model: 'qwen3' })
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).JEV_VISUAL_CONTEXT, 'false')
    t.store.saveModelPreset({ id, name: '', url: 'http://localhost:1234/v1', model: 'qwen3', apiKey: null, vision: 'on', timeout: '30' })
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).JEV_VISUAL_CONTEXT, 'true')
  } finally { t.cleanup() }
})

test('an older file turns its one custom endpoint into a chosen card, keeping the key', () => {
  const t = openStore()
  try {
    t.writeRaw({
      ...preFeatureFile(t.dir, t.sealed), aiProvider: 'openai-compatible', localWhisperMode: 'builtin',
      openaiCompatibleUrl: 'https://api.groq.com/openai/v1', openaiCompatibleModel: 'openai/gpt-oss-120b',
      openaiCompatibleApiKey: t.sealed('gsk-groq'), openaiCompatibleVision: 'off', openaiCompatibleTimeout: '5'
    })
    const loaded = t.store.loadSettings()
    assert.equal(loaded.aiProvider, 'openai-compatible')
    assert.equal(loaded.modelPresets.length, 1)
    const [card] = loaded.modelPresets
    assert.deepEqual({ ...card }, { id: 'saved-endpoint', name: '', url: 'https://api.groq.com/openai/v1', model: 'openai/gpt-oss-120b', apiKey: 'gsk-groq', vision: 'off', timeout: '5' })
    assert.equal(loaded.activeModelPreset, 'saved-endpoint')
    const env = t.store.getSettingsForBridge(loaded)
    assert.deepEqual([env.OPENAI_COMPATIBLE_URL, env.OPENAI_COMPATIBLE_MODEL, env.OPENAI_COMPATIBLE_API_KEY, env.OPENAI_COMPATIBLE_TIMEOUT_SECONDS],
      ['https://api.groq.com/openai/v1', 'openai/gpt-oss-120b', 'gsk-groq', '300'])
    const disk = t.raw()
    assert.equal(disk.version, 14)
    assert.equal(disk.modelPresets[0].apiKey.scheme, 'safeStorage')
    assert.equal(disk.openaiCompatibleUrl, 'https://api.groq.com/openai/v1', 'an older BridgeClip still finds the chosen model')
    // Loading again keeps the same card rather than adding another.
    assert.equal(t.store.loadSettings().modelPresets.length, 1)

    // A URL without a model was never usable, so it does not become a card.
    t.writeRaw({ ...preFeatureFile(t.dir, t.sealed), aiProvider: 'openai-compatible', localWhisperMode: 'builtin', openaiCompatibleUrl: 'http://localhost:11434/v1' })
    const half = t.store.loadSettings()
    assert.deepEqual(half.modelPresets, [])
    assert.equal(half.openaiCompatibleUrl, '')
  } finally { t.cleanup() }
})

test('a stale save from the page never overwrites a card, and choosing a card changes only the choice', () => {
  const t = openStore()
  try {
    const a = chooseModel(t.store, { url: 'http://localhost:11434/v1', model: 'llama3.1', apiKey: 'key-a' })
    const b = t.store.saveModelPreset({ name: 'Groq', url: 'https://api.groq.com/openai/v1', model: 'openai/gpt-oss-120b', apiKey: 'key-b', vision: 'on', timeout: '5' })
    const idB = b.modelPresets.find((p) => p.name === 'Groq').id
    assert.equal(b.activeModelPreset, a.id, 'saving a second card does not choose it')
    // The page sends its whole copy, still holding A's endpoint, when the person clicks card B.
    const stale = { ...a.view, aiProvider: 'openai-compatible', activeModelPreset: idB }
    const chosen = t.store.savePublicSettings(stale)
    assert.equal(chosen.activeModelPreset, idB)
    assert.deepEqual([chosen.openaiCompatibleUrl, chosen.openaiCompatibleModel, chosen.openaiCompatibleVision, chosen.openaiCompatibleTimeout],
      ['https://api.groq.com/openai/v1', 'openai/gpt-oss-120b', 'on', '5'])
    const loaded = t.store.loadSettings()
    assert.deepEqual(loaded.modelPresets.map((p) => [p.url, p.model, p.apiKey]),
      [['http://localhost:11434/v1', 'llama3.1', 'key-a'], ['https://api.groq.com/openai/v1', 'openai/gpt-oss-120b', 'key-b']])
    assert.equal(t.store.getSettingsForBridge(loaded).OPENAI_COMPATIBLE_API_KEY, 'key-b')
    // An unrelated save carrying an old list of cards changes no card either.
    t.store.savePublicSettings({ ...chosen, modelPresets: [], customVocabulary: 'Melvin' })
    assert.equal(t.store.loadSettings().modelPresets.length, 2)
  } finally { t.cleanup() }
})

test('editing, deleting and choosing cards, with keys kept out of the page', () => {
  const t = openStore()
  try {
    const a = chooseModel(t.store, { name: '  My   local\tmodel ', apiKey: 'key-a' })
    let view = t.store.publicSettings(t.store.loadSettings())
    assert.deepEqual({ ...view.modelPresets[0] }, { id: a.id, name: 'My local model', url: 'http://localhost:11434/v1', model: 'llama3.1', vision: 'off', timeout: '30', keySaved: true })
    assert.equal(JSON.stringify(view).includes('key-a'), false)
    // null keeps the key; '' removes it; editing the chosen card reaches the engine at once.
    const card = { id: a.id, name: 'Local', url: 'http://localhost:11434/v1', model: 'qwen3', vision: 'on', timeout: '60' }
    t.store.saveModelPreset({ ...card, apiKey: null })
    assert.equal(t.store.loadSettings().modelPresets[0].apiKey, 'key-a')
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).OPENAI_COMPATIBLE_MODEL, 'qwen3')
    // Moving a card to another server drops its key, so the key is never sent to a host it was not for.
    t.store.saveModelPreset({ ...card, url: 'https://proxy.example.com/v1', apiKey: null })
    assert.equal(t.store.loadSettings().modelPresets[0].apiKey, '')
    t.store.saveModelPreset({ ...card, url: 'https://proxy.example.com/v1', apiKey: 'key-proxy' })
    t.store.saveModelPreset({ ...card, url: 'https://proxy.example.com/openai/v1', apiKey: null })
    assert.equal(t.store.loadSettings().modelPresets[0].apiKey, 'key-proxy', 'a new path on the same server keeps it')
    view = t.store.saveModelPreset({ ...card, url: 'https://proxy.example.com/v1', apiKey: '' })
    assert.equal(view.modelPresets[0].keySaved, false)
    assert.throws(() => t.store.saveModelPreset({ ...card, id: 'gone' }), /no longer exists/)

    const b = t.store.saveModelPreset({ name: '', url: 'https://api.openai.com/v1', model: 'gpt-4o-mini', apiKey: 'key-b', vision: 'off', timeout: '30' })
    const idB = b.modelPresets[1].id
    // Deleting a card that is not chosen keeps the choice; deleting the chosen one leaves none.
    view = t.store.deleteModelPreset(idB)
    assert.equal(view.activeModelPreset, a.id)
    view = t.store.deleteModelPreset(a.id)
    assert.equal(view.activeModelPreset, '')
    assert.equal(view.aiProvider, 'openai-compatible')
    assert.equal(view.openaiCompatibleUrl, '')
    assert.throws(() => t.store.deleteModelPreset(a.id), /no longer exists/)
  } finally { t.cleanup() }
})

test('the first card is chosen when nothing else can plan clips; later cards wait to be clicked', () => {
  const t = openStore()
  try {
    // A fresh install: OpenRouter without a key cannot plan, so the new card is chosen, and speech
    // moves to Local Whisper because cloud transcription needs the missing OpenRouter key.
    const first = t.store.saveModelPreset({ name: '', url: 'http://localhost:11434/v1', model: 'llama3.1', apiKey: null, vision: 'off', timeout: '30' })
    assert.equal(first.activeModelPreset, first.modelPresets[0].id)
    assert.equal(first.aiProvider, 'openai-compatible')
    assert.equal(first.transcriptionSource, 'local')
    const second = t.store.saveModelPreset({ name: '', url: 'http://localhost:1234/v1', model: 'qwen3', apiKey: null, vision: 'off', timeout: '30' })
    assert.equal(second.activeModelPreset, first.modelPresets[0].id)
  } finally { t.cleanup() }
})

test('at most 20 cards, and damaged saved cards are dropped without losing the rest of the file', () => {
  const t = openStore()
  try {
    for (let i = 0; i < 20; i++) t.store.saveModelPreset({ name: `M${i}`, url: 'http://localhost:11434/v1', model: `m${i}`, apiKey: null, vision: 'off', timeout: '30' })
    assert.throws(() => t.store.saveModelPreset({ name: 'One more', url: 'http://localhost:11434/v1', model: 'x', apiKey: null, vision: 'off', timeout: '30' }), /up to 20 models/)

    t.writeRaw({
      ...preFeatureFile(t.dir, t.sealed), version: 13, aiProvider: 'openai-compatible', localWhisperMode: 'builtin',
      modelPresets: [
        { id: 'good', name: 'Good', url: 'http://localhost:11434/v1', model: 'llama3.1', apiKey: t.sealed('k'), vision: 'on', timeout: '10' },
        { id: 'bad-url', name: 'x', url: 'ftp://nope', model: 'm', apiKey: '', vision: 'off', timeout: '30' },
        { id: 'bad id!', name: 'x', url: 'http://localhost:1/v1', model: 'm', apiKey: '', vision: 'off', timeout: '30' },
        { id: 'good', name: 'Duplicate', url: 'http://localhost:2/v1', model: 'm', apiKey: '', vision: 'off', timeout: '30' },
        { id: 'odd-values', name: 'Odd', url: 'http://localhost:3/v1', model: 'm', apiKey: '', vision: 'maybe', timeout: '7' },
        'junk'
      ],
      activeModelPreset: 'good'
    })
    const loaded = t.store.loadSettings()
    assert.deepEqual(loaded.modelPresets.map((p) => [p.id, p.vision, p.timeout]), [['good', 'on', '10'], ['odd-values', 'off', '30']])
    assert.equal(loaded.activeModelPreset, 'good')
    assert.equal(loaded.openaiCompatibleApiKey, 'k')
    assert.equal(loaded.openrouterApiKey, 'sk-or-existing', 'the other keys still load')

    // A chosen card that is gone loads as none chosen.
    t.writeRaw({ ...preFeatureFile(t.dir, t.sealed), version: 13, aiProvider: 'openai-compatible', modelPresets: [], activeModelPreset: 'missing' })
    assert.equal(t.store.loadSettings().activeModelPreset, '')
  } finally { t.cleanup() }
})

test('a card can be tested without choosing it', () => {
  const t = openStore()
  try {
    const a = chooseModel(t.store)
    const b = t.store.saveModelPreset({ name: '', url: 'https://api.groq.com/openai/v1', model: 'openai/gpt-oss-120b', apiKey: 'key-b', vision: 'off', timeout: '5' })
    const idB = b.modelPresets[1].id
    const settings = t.store.loadSettings()
    const tested = t.store.settingsWithModelPreset(settings, idB)
    assert.deepEqual([tested.openaiCompatibleUrl, tested.openaiCompatibleModel, tested.openaiCompatibleApiKey, tested.openaiCompatibleTimeout],
      ['https://api.groq.com/openai/v1', 'openai/gpt-oss-120b', 'key-b', '5'])
    assert.equal(t.store.loadSettings().activeModelPreset, a.id)
    assert.throws(() => t.store.settingsWithModelPreset(settings, 'gone'), /no longer exists/)
  } finally { t.cleanup() }
})
