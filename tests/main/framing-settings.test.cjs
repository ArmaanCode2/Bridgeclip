'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { loadMain, tempDir, fakeElectron } = require('../zernio/support/load-main.cjs')

/** The settings store on a temp user-data folder, with a visible fake keychain. */
function openStore() {
  const { dir, cleanup } = tempDir('bridgeclip-framing-')
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

test('a new install puts the webcam on top and fills screens', () => {
  const t = openStore()
  try {
    const settings = t.store.loadSettings()
    assert.equal(settings.webcamPosition, 'top')
    assert.equal(settings.screenFraming, 'fill')
    const env = t.store.getSettingsForBridge(settings)
    assert.equal(env.WEBCAM_POSITION, 'top')
    assert.equal(env.SCREEN_FILL, 'true')
  } finally { t.cleanup() }
})

test('a settings file from before these choices existed loads the defaults and keeps everything else', () => {
  const t = openStore()
  try {
    t.writeRaw(savedFile(t.dir, { aiProvider: 'gemini', localWhisperDevice: 'cpu' }))
    const loaded = t.store.loadSettings()
    assert.equal(loaded.webcamPosition, 'top')
    assert.equal(loaded.screenFraming, 'fill')
    assert.equal(loaded.aiProvider, 'gemini')
    assert.equal(loaded.localWhisperDevice, 'cpu')
    // An unreadable value falls back to its default instead of losing the file.
    t.writeRaw(savedFile(t.dir, { webcamPosition: 'left', screenFraming: 7, aiProvider: 'gemini' }))
    const fallback = t.store.loadSettings()
    assert.equal(fallback.webcamPosition, 'top')
    assert.equal(fallback.screenFraming, 'fill')
    assert.equal(fallback.aiProvider, 'gemini')
  } finally { t.cleanup() }
})

test('both choices save, persist, show in the public view and reach the engine', () => {
  const t = openStore()
  try {
    const saved = t.store.savePublicSettings({ webcamPosition: 'bottom', screenFraming: 'fit' })
    assert.equal(saved.webcamPosition, 'bottom')
    assert.equal(saved.screenFraming, 'fit')
    assert.equal(t.raw().webcamPosition, 'bottom')
    assert.equal(t.raw().screenFraming, 'fit')
    const env = t.store.getSettingsForBridge(t.store.loadSettings())
    assert.equal(env.WEBCAM_POSITION, 'bottom')
    assert.equal(env.SCREEN_FILL, 'false')
    // A change to one setting keeps the other.
    assert.equal(t.store.savePublicSettings({ webcamPosition: 'top' }).screenFraming, 'fit')
  } finally { t.cleanup() }
})

test('values outside the choices are refused', () => {
  const t = openStore()
  try {
    assert.throws(() => t.store.savePublicSettings({ webcamPosition: 'left' }), /webcam position/)
    assert.throws(() => t.store.savePublicSettings({ screenFraming: 'stretch' }), /screen framing/)
    assert.equal(t.store.loadSettings().webcamPosition, 'top')
  } finally { t.cleanup() }
})

test('the video encoder defaults to the CPU, saves NVIDIA, refuses others and reaches the engine', () => {
  const t = openStore()
  try {
    assert.equal(t.store.loadSettings().videoEncoder, 'cpu')
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).VIDEO_ENCODER, 'cpu')
    t.writeRaw(savedFile(t.dir, { videoEncoder: 'quantum', aiProvider: 'gemini' }))
    assert.equal(t.store.loadSettings().videoEncoder, 'cpu')
    assert.equal(t.store.loadSettings().aiProvider, 'gemini')
    assert.equal(t.store.savePublicSettings({ videoEncoder: 'nvidia' }).videoEncoder, 'nvidia')
    assert.equal(t.raw().videoEncoder, 'nvidia')
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).VIDEO_ENCODER, 'nvidia')
    assert.throws(() => t.store.savePublicSettings({ videoEncoder: 'amd' }), /video encoder/)
  } finally { t.cleanup() }
})

test('the clip judge is on by default, saves off, refuses others and reaches the engine', () => {
  const t = openStore()
  try {
    assert.equal(t.store.loadSettings().clipJudge, 'on')
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).CLIP_JUDGE, 'true')
    // A file from before this setting, or with a value no longer allowed, loads as on.
    t.writeRaw(savedFile(t.dir, { clipJudge: 'maybe', aiProvider: 'gemini' }))
    assert.equal(t.store.loadSettings().clipJudge, 'on')
    assert.equal(t.store.loadSettings().aiProvider, 'gemini')
    assert.equal(t.store.savePublicSettings({ clipJudge: 'off' }).clipJudge, 'off')
    assert.equal(t.raw().clipJudge, 'off')
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).CLIP_JUDGE, 'false')
    assert.throws(() => t.store.savePublicSettings({ clipJudge: 'sometimes' }), /second opinion/)
  } finally { t.cleanup() }
})

test('the webcam location is Auto by default, saves a corner, refuses others and reaches the engine', () => {
  const t = openStore()
  try {
    assert.equal(t.store.loadSettings().webcamLocation, 'auto')
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).WEBCAM_LOCATION, 'auto')
    t.writeRaw(savedFile(t.dir, { webcamLocation: 'middle', webcamPosition: 'bottom' }))
    assert.equal(t.store.loadSettings().webcamLocation, 'auto')
    assert.equal(t.store.loadSettings().webcamPosition, 'bottom')
    assert.equal(t.store.savePublicSettings({ webcamLocation: 'bottom-right' }).webcamLocation, 'bottom-right')
    assert.equal(t.raw().webcamLocation, 'bottom-right')
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).WEBCAM_LOCATION, 'bottom-right')
    assert.throws(() => t.store.savePublicSettings({ webcamLocation: 'left' }), /webcam location/)
  } finally { t.cleanup() }
})
