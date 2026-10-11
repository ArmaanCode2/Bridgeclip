'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { loadMain, tempDir, fakeElectron } = require('../zernio/support/load-main.cjs')

const providers = loadMain("export * from './src/shared/providers'")
const { setupGaps, localWhisperModeChange, checkLocalWhisperUrl, classifyHost, isLocalEndpoint, LOCAL_WHISPER_URL_MESSAGE } = providers

const NONE = {
  aiProvider: 'openrouter', openrouterConfigured: false, geminiConfigured: false, openaiCompatibleUrl: '', openaiCompatibleModel: '',
  transcriptionSource: 'cloud', localWhisperMode: 'builtin', localWhisperUrl: ''
}

// ---- the Create page's disabled-button text ----------------------------------------------

test('an OpenRouter-only user sees the original blocked text, word for word', () => {
  const gaps = setupGaps(NONE)
  assert.equal(gaps.length, 1)
  assert.equal(gaps[0].id, 'openrouter-key')
  assert.equal(gaps[0].blocked, 'Add your OpenRouter key above to start.')
  // The same sentence the app showed before providers were configurable.
  assert.equal(gaps[0].blocked ?? gaps[0].message, 'Add your OpenRouter key above to start.')
})

test('other gaps fall back to their own message, so nothing says "above" about a field that is not there', () => {
  const gemini = setupGaps({ ...NONE, aiProvider: 'gemini', openrouterConfigured: true })
  assert.equal(gemini[0].blocked, undefined)
  assert.equal(gemini[0].message, 'Add your Google Gemini API key.')
  const cloud = setupGaps({ ...NONE, aiProvider: 'gemini', geminiConfigured: true })
  assert.equal(cloud[0].id, 'openrouter-key')
  assert.equal(cloud[0].blocked, undefined, 'the cloud-transcription gap explains itself instead')
  assert.match(cloud[0].message, /Cloud transcription uses OpenRouter/)
  const server = setupGaps({ ...NONE, aiProvider: 'gemini', geminiConfigured: true, transcriptionSource: 'local', localWhisperMode: 'server' })
  assert.equal(server[0].id, 'whisper-url')
  assert.equal(server[0].blocked, undefined)
  assert.deepEqual(setupGaps({ ...NONE, openrouterConfigured: true }), [])
})

// ---- the Built-in / Server switch ----------------------------------------------------------

test('switching mode keeps a model that fits and replaces one that does not', () => {
  assert.deepEqual(localWhisperModeChange('builtin', 'medium'), { localWhisperMode: 'builtin' })
  assert.deepEqual(localWhisperModeChange('builtin', 'whisper-1'), { localWhisperMode: 'builtin', localWhisperModel: 'small' })
  assert.deepEqual(localWhisperModeChange('builtin', 'large-v2'), { localWhisperMode: 'builtin' }, 'large-v2 is a built-in model')
  assert.deepEqual(localWhisperModeChange('server', 'whisper-1'), { localWhisperMode: 'server' })
  assert.deepEqual(localWhisperModeChange('server', 'large-v3'), { localWhisperMode: 'server' })
  assert.deepEqual(localWhisperModeChange('server', ''), { localWhisperMode: 'server', localWhisperModel: 'small' })
  assert.deepEqual(localWhisperModeChange('server', 'two words'), { localWhisperMode: 'server', localWhisperModel: 'small' })
})

test('the model for a mode switch is decided from the model at the moment it is applied', () => {
  // Press "Server", set a server-only model name, then press "Built-in" before the first save lands.
  // Deciding from the stale value would keep "whisper-1" for built-in mode, which settings reject.
  let saved = { localWhisperMode: 'builtin', localWhisperModel: 'small' }
  const toServer = () => { saved = { ...saved, ...localWhisperModeChange('server', saved.localWhisperModel), localWhisperModel: 'whisper-1' } }
  const toBuiltin = (current) => { saved = { ...saved, ...localWhisperModeChange('builtin', current.localWhisperModel) } }
  toServer()
  toBuiltin(saved) // the current value, as the queued save sees it
  assert.deepEqual(saved, { localWhisperMode: 'builtin', localWhisperModel: 'small' })
})

// ---- CGNAT / Tailscale -------------------------------------------------------------------

test('100.64.0.0/10 is blocked, and only that range', () => {
  for (const host of ['100.64.0.0', '100.64.0.1', '100.100.100.200', '100.96.5.5', '100.127.255.255', '[::ffff:6440:1]', '::ffff:100.64.0.1']) {
    assert.equal(classifyHost(host), 'blocked', host)
  }
  for (const host of ['100.0.0.1', '100.63.255.255', '100.128.0.0', '100.200.1.1']) {
    assert.equal(classifyHost(host), 'public', host)
  }
  // Other special ranges are unchanged.
  assert.equal(classifyHost('127.0.0.1'), 'loopback')
  assert.equal(classifyHost('192.168.1.5'), 'private')
  assert.equal(classifyHost('169.254.169.254'), 'blocked')
})

// ---- local Whisper must be local ---------------------------------------------------------

test('a Whisper server URL passes only if it is this computer or a private network, with the plain reason otherwise', () => {
  assert.equal(LOCAL_WHISPER_URL_MESSAGE, 'Local Whisper must run on this computer or your private network. For a cloud service use Cloud transcription.')
  assert.deepEqual(checkLocalWhisperUrl(''), { ok: true, url: '' })
  assert.deepEqual(checkLocalWhisperUrl('  '), { ok: true, url: '' })
  for (const url of ['http://localhost:8000/v1', 'http://127.0.0.1:8000/v1/', 'http://192.168.1.2:9000', 'https://10.0.0.7/v1', 'http://my-nas.local:8000/v1']) {
    assert.equal(checkLocalWhisperUrl(url).ok, true, url)
  }
  for (const url of ['https://api.openai.com/v1', 'https://whisper.example.com/v1', 'https://8.8.8.8/v1', 'https://100.63.255.255/v1']) {
    assert.deepEqual(checkLocalWhisperUrl(url), { ok: false, error: LOCAL_WHISPER_URL_MESSAGE }, url)
  }
  // Other refusals keep their own, more specific reasons.
  assert.equal(checkLocalWhisperUrl('http://169.254.169.254/').ok, false)
  assert.notEqual(checkLocalWhisperUrl('http://169.254.169.254/').error, LOCAL_WHISPER_URL_MESSAGE)
  assert.match(checkLocalWhisperUrl('ftp://localhost/').error, /http/)
  assert.equal(checkLocalWhisperUrl('http://u:p@localhost/').ok, false)
})

test('isLocalEndpoint is the single definition of "stays on my network"', () => {
  assert.equal(isLocalEndpoint('http://localhost:1/v1'), true)
  assert.equal(isLocalEndpoint('http://172.20.1.1/v1'), true)
  assert.equal(isLocalEndpoint('https://api.openai.com/v1'), false)
  assert.equal(isLocalEndpoint('http://100.64.0.1/v1'), false)
  assert.equal(isLocalEndpoint('not a url'), false)
})

// ---- job validation: Advanced with local Whisper ---------------------------------------------

function validation() {
  const { dir, cleanup } = tempDir('bridgeclip-validate-')
  const { electron } = fakeElectron(dir)
  const api = loadMain("export * from './src/main/validation'", { electron })
  return { api, cleanup }
}

const BASE = () => ({
  videoUrl: path.resolve(path.sep, 'videos', 'talk.mp4'), workflow: 'automatic', clippingMode: 'quality',
  maxClips: null, autoClipCount: true, durationRanges: null, aspectRatio: '9:16', layoutStyle: 'auto', layoutVision: true, pacing: 'tight',
  includeCaptions: true, captionPreset: 'pop', startTimeSeconds: null, endTimeSeconds: null, bannerPlatform: null, bannerChannelUrl: null
})

test('an Advanced job needs both models, unless Whisper runs locally, when the planning model is enough', () => {
  const t = validation()
  try {
    const { validateJobConfig } = t.api
    const advanced = { ...BASE(), clippingMode: 'advanced', plannerModel: 'vendor/planner' }
    // Default (cloud): unchanged, both are required.
    assert.throws(() => validateJobConfig(advanced), /Choose both models in Advanced mode/)
    assert.throws(() => validateJobConfig({ ...advanced, transcriptionModel: 'not a model' }), /Choose both models/)
    assert.doesNotThrow(() => validateJobConfig({ ...advanced, transcriptionModel: 'vendor/speech' }))
    // Local Whisper: no transcription model is asked for.
    const local = { transcriptionOptional: true }
    const accepted = validateJobConfig(advanced, local)
    assert.equal(accepted.plannerModel, 'vendor/planner')
    assert.equal(accepted.transcriptionModel, undefined)
    // The planning model is still required, and a transcription model that is given must still be a real id.
    assert.throws(() => validateJobConfig({ ...advanced, plannerModel: undefined }, local), /Choose both models/)
    assert.throws(() => validateJobConfig({ ...advanced, plannerModel: 'bad model' }, local), /Choose both models/)
    assert.throws(() => validateJobConfig({ ...advanced, transcriptionModel: 'bad model' }, local), /Choose both models/)
    assert.doesNotThrow(() => validateJobConfig({ ...advanced, transcriptionModel: 'vendor/speech' }, local))
    // The option never loosens anything for the other modes.
    assert.throws(() => validateJobConfig({ ...BASE(), plannerModel: 'vendor/planner' }, local), /Custom models require Advanced mode/)
    assert.throws(() => validateJobConfig({ ...BASE(), transcriptionModel: 'vendor/speech' }, local), /Custom models require Advanced mode/)
  } finally { t.cleanup() }
})
