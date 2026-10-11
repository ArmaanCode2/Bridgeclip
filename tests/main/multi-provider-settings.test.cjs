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
    t.store.replaceApiKey('openaiCompatibleApiKey', 'sk-custom-secret')
    t.store.replaceApiKey('openrouterApiKey', 'sk-or-secret')
    const text = fs.readFileSync(t.file, 'utf8')
    for (const secret of ['AIzaGeminiSecret', 'sk-custom-secret', 'sk-or-secret']) {
      assert.equal(text.includes(secret), false, `${secret} is stored in plain text`)
      assert.equal(text.includes(Buffer.from(secret).toString('base64')), false, `${secret} is stored as plain base64`)
    }
    const stored = t.raw()
    for (const key of ['geminiApiKey', 'openaiCompatibleApiKey', 'openrouterApiKey']) {
      assert.equal(stored[key].scheme, 'safeStorage')
      assert.match(Buffer.from(stored[key].value, 'base64').toString(), /^enc:/)
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

    // A file written by this version is never treated as legacy: a typed Ollama URL survives.
    t.store.savePublicSettings({ aiProvider: 'openrouter', openaiCompatibleUrl: 'http://localhost:11434/v1' })
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
    for (const key of ['openaiCompatibleUrl', 'localWhisperUrl']) {
      // A Whisper server must be local, so the public https entries only apply to the AI endpoint.
      for (const [input, expected] of ACCEPTED_URLS.filter(([input]) => key === 'openaiCompatibleUrl' || !isPublicOnly(input))) {
        const saved = t.store.savePublicSettings({ [key]: input })
        assert.equal(saved[key], expected, `${key}: ${input}`)
      }
      t.store.savePublicSettings({ [key]: 'http://localhost:9000/v1' })
      for (const input of REJECTED_URLS) {
        assert.throws(() => t.store.savePublicSettings({ [key]: input }), Error, `${key}: ${input} should be refused`)
        assert.equal(t.store.loadSettings()[key], 'http://localhost:9000/v1', `${key} changed after refusing ${input}`)
      }
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
    assert.equal(t.store.savePublicSettings({ openaiCompatibleUrl: 'https://api.openai.com/v1' }).openaiCompatibleUrl, 'https://api.openai.com/v1')
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
    for (const key of ['openaiCompatibleUrl', 'localWhisperUrl']) {
      t.store.savePublicSettings({ [key]: 'http://localhost:9000/v1' })
      for (const url of blocked) {
        assert.throws(() => t.store.savePublicSettings({ [key]: url }), Error, `${key}: ${url}`)
        assert.equal(t.store.loadSettings()[key], 'http://localhost:9000/v1')
      }
    }
    // Just outside the range is an ordinary public address: fine over https for the AI endpoint, never for local Whisper.
    for (const url of ['https://100.63.255.255/v1', 'https://100.128.0.1/v1']) {
      assert.equal(t.store.savePublicSettings({ openaiCompatibleUrl: url }).openaiCompatibleUrl, url)
      assert.throws(() => t.store.savePublicSettings({ localWhisperUrl: url }), WHISPER_PRIVACY_MESSAGE)
    }
    // Saved files with such an address load as not configured instead of failing to load.
    t.writeRaw({ ...preFeatureFile(t.dir, t.sealed), version: 12, aiProvider: 'openai-compatible', openaiCompatibleUrl: 'http://100.64.0.1/v1', localWhisperMode: 'server', localWhisperUrl: 'http://100.64.0.1/v1' })
    const loaded = t.store.loadSettings()
    assert.equal(loaded.openaiCompatibleUrl, '')
    assert.equal(loaded.localWhisperUrl, '')
  } finally { t.cleanup() }
})

test('an empty URL or model stays empty and reads as not configured', () => {
  const t = openStore()
  try {
    t.store.savePublicSettings({ openaiCompatibleUrl: 'http://localhost:11434/v1', openaiCompatibleModel: 'llama3.1', localWhisperMode: 'server', localWhisperUrl: 'http://localhost:8000/v1' })
    const cleared = t.store.savePublicSettings({ openaiCompatibleUrl: '', openaiCompatibleModel: '', localWhisperUrl: '' })
    assert.equal(cleared.openaiCompatibleUrl, '')
    assert.equal(cleared.openaiCompatibleModel, '')
    assert.equal(cleared.localWhisperUrl, '')
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
    for (const name of ['gpt-4o-mini', 'llama3.1:8b', 'models/gemini-3.8-flash', 'org/model@v2', 'Qwen2.5-7B-Instruct+q4']) {
      assert.equal(t.store.savePublicSettings({ openaiCompatibleModel: name }).openaiCompatibleModel, name)
      assert.equal(t.store.savePublicSettings({ geminiModel: name }).geminiModel, name)
    }
    for (const name of ['my model', 'a;b', 'a\nb', '-leading-dash', '$(whoami)', 'x'.repeat(201), 'a"b']) {
      assert.throws(() => t.store.savePublicSettings({ openaiCompatibleModel: name }), /Model name/, JSON.stringify(name))
      assert.throws(() => t.store.savePublicSettings({ geminiModel: name }), /Gemini model/, JSON.stringify(name))
    }
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
    assert.throws(() => t.store.savePublicSettings({ openaiCompatibleVision: 'maybe' }), /image analysis/)
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
    t.store.replaceApiKey('openaiCompatibleApiKey', 'sk-custom')
    t.store.replaceApiKey('openrouterApiKey', 'sk-or')
    t.store.savePublicSettings({
      aiProvider: 'openai-compatible', openaiCompatibleUrl: 'http://localhost:11434/v1', openaiCompatibleModel: 'llama3.1', openaiCompatibleVision: 'on',
      transcriptionSource: 'local', localWhisperMode: 'server', localWhisperUrl: 'http://localhost:8000/v1', localWhisperModel: 'large-v3'
    })
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
    t.store.savePublicSettings({ jevVisualContext: 'on', aiProvider: 'openai-compatible', openaiCompatibleUrl: 'http://localhost:1234/v1', openaiCompatibleModel: 'qwen3' })
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).JEV_VISUAL_CONTEXT, 'false')
    t.store.savePublicSettings({ openaiCompatibleVision: 'on' })
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).JEV_VISUAL_CONTEXT, 'true')
  } finally { t.cleanup() }
})
