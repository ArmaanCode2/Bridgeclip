'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { loadMain, tempDir, fakeElectron } = require('../zernio/support/load-main.cjs')

/** The settings store on a temp user-data folder, with a visible fake keychain. */
function openStore() {
  const { dir, cleanup } = tempDir('bridgeclip-whisper-device-')
  const { electron } = fakeElectron(dir)
  const store = loadMain("export * from './src/main/settings-store'", { electron })
  const file = path.join(dir, 'userData', 'settings.json')
  return {
    store, file, dir, cleanup,
    raw: () => JSON.parse(fs.readFileSync(file, 'utf8')),
    writeRaw: (data) => { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, JSON.stringify(data)) }
  }
}

const savedFile = (dir, extra = {}) => ({
  version: 12, openrouterApiKey: '', zernioApiKey: '',
  jevEnabled: 'off', jevVisualContext: 'off', sourceContextWebResearch: 'off',
  outputDirectory: path.join(dir, 'clips'), pythonPath: 'python3', customVocabulary: '',
  ...extra
})

test('the device defaults to auto for a new install', () => {
  const t = openStore()
  try {
    assert.equal(t.store.loadSettings().localWhisperDevice, 'auto')
    assert.equal(t.store.publicSettings(t.store.loadSettings()).localWhisperDevice, 'auto')
  } finally { t.cleanup() }
})

test('a settings file from before the device existed loads as auto, with every other choice kept', () => {
  const t = openStore()
  try {
    t.writeRaw(savedFile(t.dir, { aiProvider: 'gemini', transcriptionSource: 'local', localWhisperMode: 'builtin', localWhisperModel: 'medium' }))
    const loaded = t.store.loadSettings()
    assert.equal(loaded.localWhisperDevice, 'auto')
    assert.equal(loaded.localWhisperModel, 'medium')
    assert.equal(loaded.transcriptionSource, 'local')
    // Even a file older than local Whisper itself.
    t.writeRaw(savedFile(t.dir))
    assert.equal(t.store.loadSettings().localWhisperDevice, 'auto')
  } finally { t.cleanup() }
})

test('save accepts auto, cpu and gpu, persists them, and shows them in the public view', () => {
  const t = openStore()
  try {
    for (const device of ['cpu', 'gpu', 'auto']) {
      const saved = t.store.savePublicSettings({ localWhisperDevice: device })
      assert.equal(saved.localWhisperDevice, device)
      assert.equal(t.raw().localWhisperDevice, device)
      assert.equal(t.store.loadSettings().localWhisperDevice, device)
    }
  } finally { t.cleanup() }
})

test('save rejects anything else, and a rejected save leaves the stored device alone', () => {
  const t = openStore()
  try {
    t.store.savePublicSettings({ localWhisperDevice: 'cpu' })
    const rejected = ['tpu', '', 'GPU', 'Auto', ' cpu', 'cpu ', 'cuda', 'gpu\0', 'auto;rm', '../cpu', 'true', '0']
    for (const value of rejected) {
      assert.throws(() => t.store.savePublicSettings({ localWhisperDevice: value }), /Whisper device|Invalid/, JSON.stringify(value))
    }
    for (const value of [1, true, ['cpu'], {}]) {
      assert.throws(() => t.store.savePublicSettings({ localWhisperDevice: value }), /Invalid/, JSON.stringify(value))
    }
    assert.equal(t.store.loadSettings().localWhisperDevice, 'cpu')
    assert.equal(t.raw().localWhisperDevice, 'cpu')
  } finally { t.cleanup() }
})

test('a damaged saved device falls back to auto instead of making the whole file unreadable', () => {
  const t = openStore()
  try {
    for (const bad of ['warp', '', 7, null, ['gpu'], { a: 1 }, 'GPU']) {
      t.writeRaw(savedFile(t.dir, { localWhisperDevice: bad, customVocabulary: 'kept' }))
      const loaded = t.store.loadSettings()
      assert.equal(loaded.localWhisperDevice, 'auto', JSON.stringify(bad))
      assert.equal(loaded.customVocabulary, 'kept')
    }
  } finally { t.cleanup() }
})

test('changing another setting does not change the device', () => {
  const t = openStore()
  try {
    t.store.savePublicSettings({ localWhisperDevice: 'gpu' })
    t.store.savePublicSettings({ localWhisperModel: 'turbo', transcriptionSource: 'local' })
    assert.equal(t.store.loadSettings().localWhisperDevice, 'gpu')
  } finally { t.cleanup() }
})

test('the device reaches the engine as LOCAL_WHISPER_DEVICE for every combination of source and mode', () => {
  const t = openStore()
  try {
    for (const device of ['auto', 'cpu', 'gpu']) {
      for (const source of ['cloud', 'local']) {
        for (const mode of ['builtin', 'server']) {
          const saved = { ...t.store.loadSettings(), localWhisperDevice: device, transcriptionSource: source, localWhisperMode: mode }
          const env = t.store.getSettingsForBridge(saved)
          assert.equal(env.LOCAL_WHISPER_DEVICE, device, `${device}/${source}/${mode}`)
        }
      }
    }
    const env = t.store.getSettingsForBridge(t.store.loadSettings())
    assert.equal(env.LOCAL_WHISPER_MODE, 'builtin')
    assert.equal(env.LOCAL_WHISPER_MODEL, 'small')
  } finally { t.cleanup() }
})

test('the spoken language defaults to automatic detection, saves a listed code, and reaches the engine', () => {
  const t = openStore()
  try {
    assert.equal(t.store.loadSettings().localWhisperLanguage, 'auto')
    assert.equal(t.store.publicSettings(t.store.loadSettings()).localWhisperLanguage, 'auto')
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).LOCAL_WHISPER_LANGUAGE, 'auto')
    const saved = t.store.savePublicSettings({ localWhisperLanguage: 'hi' })
    assert.equal(saved.localWhisperLanguage, 'hi')
    assert.equal(t.raw().localWhisperLanguage, 'hi')
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).LOCAL_WHISPER_LANGUAGE, 'hi')
    for (const value of ['klingon', 'EN', 'en-US', '', 'en; rm -rf']) {
      assert.throws(() => t.store.savePublicSettings({ localWhisperLanguage: value }), /Whisper language|Invalid/, JSON.stringify(value))
    }
    assert.equal(t.store.loadSettings().localWhisperLanguage, 'hi')
  } finally { t.cleanup() }
})

test('a saved language that is no longer offered loads as automatic detection, keeping the rest of the file', () => {
  const t = openStore()
  try {
    for (const bad of ['xx', 7, null, { code: 'en' }]) {
      t.writeRaw(savedFile(t.dir, { localWhisperLanguage: bad, customVocabulary: 'kept' }))
      const loaded = t.store.loadSettings()
      assert.equal(loaded.localWhisperLanguage, 'auto', JSON.stringify(bad))
      assert.equal(loaded.customVocabulary, 'kept')
    }
  } finally { t.cleanup() }
})
