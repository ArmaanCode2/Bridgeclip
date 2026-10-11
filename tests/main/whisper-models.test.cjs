'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { loadMain, tempDir } = require('../zernio/support/load-main.cjs')

const mod = loadMain("export * from './src/main/whisper-models'")
const { listWhisperModels, deleteWhisperModel, resolveCacheRoots, hubFolderHoldsModel, openaiWhisperFile, parseNvidiaSmi, detectGpu } = mod
const shared = loadMain("export * from './src/shared/whisper-models'")
const { WHISPER_MESSAGES, WHISPER_MODEL_NAMES } = shared

// ---- fixtures ------------------------------------------------------------------------

/** A throwaway home directory with empty Whisper and Hugging Face caches. */
function layout() {
  const { dir, cleanup } = tempDir('bridgeclip-whisper-')
  const home = path.join(dir, 'home')
  const cache = path.join(home, '.cache')
  const whisper = path.join(cache, 'whisper')
  const hub = path.join(cache, 'huggingface', 'hub')
  fs.mkdirSync(whisper, { recursive: true })
  fs.mkdirSync(hub, { recursive: true })
  return { dir, cleanup, home, cache, whisper, hub, env: {} }
}

function writeBytes(file, size) {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, Buffer.alloc(size, 7))
}

/** Tries to make a symlink. False when this machine forbids it (Windows without developer mode). */
function trySymlink(target, link, type) {
  try {
    fs.symlinkSync(target, link, type)
    return true
  } catch (error) {
    if (error.code === 'EPERM' || error.code === 'EACCES') return false
    throw error
  }
}
const SYMLINK_SKIP = 'symlinks need developer mode or administrator rights on Windows (EPERM); the same logic is covered with a fake file system below'

/**
 * A faster-whisper download as the Hugging Face cache lays it out: blobs/ holds the files and
 * snapshots/<rev>/ links to them. `model.bin` is a link when links work here and a copy otherwise;
 * either way blobs/ is what is counted, so the expected size is the same.
 */
function hubModel(hub, folder, { bytes = 1000, withModel = true } = {}) {
  const root = path.join(hub, folder)
  writeBytes(path.join(root, 'blobs', 'sha-model'), bytes)
  writeBytes(path.join(root, 'blobs', 'sha-config'), 10)
  const snapshot = path.join(root, 'snapshots', 'rev1')
  fs.mkdirSync(snapshot, { recursive: true })
  fs.writeFileSync(path.join(snapshot, 'config.json'), '{}')
  if (withModel) {
    const link = path.join(snapshot, 'model.bin')
    if (!trySymlink(path.join('..', '..', 'blobs', 'sha-model'), link, 'file')) fs.copyFileSync(path.join(root, 'blobs', 'sha-model'), link)
  }
  return bytes + 10
}

const engine = (backend) => async () => ({ available: Boolean(backend), backend, detail: 'test', device: 'cpu', modelCached: null })
const GIB = 1024 ** 3

function list(l, extra = {}) {
  return listWhisperModels({
    env: l.env, home: l.home, checkEngine: engine('faster-whisper'), gpu: async () => null, totalMemoryBytes: () => 8 * GIB, ...extra
  })
}

function del(l, request, extra = {}) {
  return deleteWhisperModel(request, { env: l.env, home: l.home, isBusy: () => false, ...extra })
}

const byName = (listing) => Object.fromEntries(listing.models.map((entry) => [entry.name, entry]))

// ---- cache locations -----------------------------------------------------------------

test('cache roots: defaults, XDG_CACHE_HOME, and the Hugging Face variables in their order of precedence', () => {
  const home = path.resolve(path.sep, 'home', 'me')
  const abs = (...parts) => path.resolve(path.sep, ...parts)
  assert.deepEqual(resolveCacheRoots({}, home), { cacheBase: path.join(home, '.cache'), hub: path.join(home, '.cache', 'huggingface', 'hub') })
  const xdg = abs('xdg')
  assert.deepEqual(resolveCacheRoots({ XDG_CACHE_HOME: xdg }, home), { cacheBase: xdg, hub: path.join(xdg, 'huggingface', 'hub') })
  assert.equal(resolveCacheRoots({ HF_HOME: abs('hf') }, home).hub, path.join(abs('hf'), 'hub'))
  assert.equal(resolveCacheRoots({ HF_HOME: abs('hf'), HUGGINGFACE_HUB_CACHE: abs('legacy') }, home).hub, abs('legacy'))
  assert.equal(resolveCacheRoots({ HF_HOME: abs('hf'), HUGGINGFACE_HUB_CACHE: abs('legacy'), HF_HUB_CACHE: abs('direct') }, home).hub, abs('direct'))
})

test('cache roots: relative or blank environment values are ignored, not resolved against the working directory', () => {
  const home = path.resolve(path.sep, 'home', 'me')
  const roots = resolveCacheRoots({ XDG_CACHE_HOME: 'relative/cache', HF_HOME: '  ', HF_HUB_CACHE: '../hub' }, home)
  assert.equal(roots.cacheBase, path.join(home, '.cache'))
  assert.equal(roots.hub, path.join(home, '.cache', 'huggingface', 'hub'))
})

test('model file names: turbo is stored as large-v3-turbo; the rest use their own name', () => {
  assert.equal(openaiWhisperFile('turbo'), 'large-v3-turbo.pt')
  assert.equal(openaiWhisperFile('large-v3'), 'large-v3.pt')
  assert.equal(openaiWhisperFile('small.en'), 'small.en.pt')
})

test('Hugging Face folder matching is exact: base is not base.en, large-v3 is not turbo, and only the repos faster-whisper loads count', () => {
  const cases = [
    ['models--Systran--faster-whisper-base', 'base', true],
    ['models--Systran--faster-whisper-base', 'base.en', false],
    ['models--Systran--faster-whisper-base.en', 'base', false],
    ['models--Systran--faster-whisper-base.en', 'base.en', true],
    ['models--Systran--faster-whisper-large-v3', 'large-v3', true],
    ['models--Systran--faster-whisper-large-v3', 'turbo', false],
    ['models--Systran--faster-whisper-large-v3-turbo', 'large-v3', false],
    ['models--mobiuslabsgmbh--faster-whisper-large-v3-turbo', 'turbo', true],
    // Other conversions would sit unused while a job downloads the repo above.
    ['models--mobiuslabs--faster-whisper-large-v3-turbo', 'turbo', false],
    ['models--deepdml--faster-whisper-large-v3-turbo-ct2', 'turbo', false],
    ['models--someone--faster-whisper-small', 'small', false],
    ['models--Systran--faster-whisper-small/../../x', 'small', false],
    ['models--Systran--faster-whisper-..', 'small', false],
    ['models--Systran--whisper-small', 'small', false],
    ['not-a-model-folder', 'small', false]
  ]
  for (const [folder, name, expected] of cases) assert.equal(hubFolderHoldsModel(folder, name), expected, `${folder} / ${name}`)
})

// ---- listing -------------------------------------------------------------------------

test('listing returns every known model in order, with memory figures, even when nothing is downloaded', async () => {
  const l = layout()
  try {
    const listing = await list(l)
    assert.deepEqual(listing.models.map((m) => m.name), [...WHISPER_MODEL_NAMES])
    assert.deepEqual(listing.models.map((m) => m.name), ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v3', 'turbo'])
    for (const entry of listing.models) {
      assert.equal(entry.installed, false)
      assert.equal(entry.sizeOnDisk, null)
      assert.equal(entry.backend, null)
    }
    const models = byName(listing)
    assert.deepEqual(models.tiny.approxMemoryMB, { cpu: 400, gpu: 1000 })
    assert.deepEqual(models.small.approxMemoryMB, { cpu: 1000, gpu: 2000 })
    assert.deepEqual(models['large-v3'].approxMemoryMB, { cpu: 4500, gpu: 10000 })
    assert.deepEqual(models.turbo.approxMemoryMB, { cpu: 3000, gpu: 6000 })
    assert.deepEqual(models['small.en'].approxMemoryMB, models.small.approxMemoryMB)
    assert.equal(listing.system.ramMB, 8192)
    assert.equal(listing.system.gpu, null)
    assert.equal(listing.engine.backend, 'faster-whisper')
  } finally { l.cleanup() }
})

test('listing finds openai-whisper .pt files with their sizes, and ignores other files', async () => {
  const l = layout()
  try {
    writeBytes(path.join(l.whisper, 'tiny.pt'), 1111)
    writeBytes(path.join(l.whisper, 'small.en.pt'), 2222)
    writeBytes(path.join(l.whisper, 'large-v3-turbo.pt'), 3333)
    writeBytes(path.join(l.whisper, 'notes.txt'), 5)
    writeBytes(path.join(l.whisper, 'large-v2.pt'), 5) // not in the manager's list
    const models = byName(await list(l, { checkEngine: engine('openai-whisper') }))
    assert.deepEqual([models.tiny.installed, models.tiny.sizeOnDisk, models.tiny.backend], [true, 1111, 'openai-whisper'])
    assert.deepEqual([models['small.en'].installed, models['small.en'].sizeOnDisk], [true, 2222])
    assert.deepEqual([models.turbo.installed, models.turbo.sizeOnDisk, models.turbo.backend], [true, 3333, 'openai-whisper'])
    assert.equal(models.small.installed, false)
    assert.equal(models['large-v3'].installed, false)
  } finally { l.cleanup() }
})

test('listing finds faster-whisper folders, sizes them from blobs, and treats a folder with no model.bin as not installed', async () => {
  const l = layout()
  try {
    const baseSize = hubModel(l.hub, 'models--Systran--faster-whisper-base', { bytes: 4000 })
    hubModel(l.hub, 'models--Systran--faster-whisper-medium', { bytes: 9000, withModel: false })
    hubModel(l.hub, 'models--Systran--faster-whisper-large-v3', { bytes: 7000 })
    const models = byName(await list(l))
    assert.deepEqual([models.base.installed, models.base.sizeOnDisk, models.base.backend], [true, baseSize, 'faster-whisper'])
    assert.equal(models['base.en'].installed, false, 'base.en is a different model from base')
    assert.equal(models.medium.installed, false, 'an interrupted download is not installed')
    assert.equal(models.medium.sizeOnDisk, null)
    assert.equal(models['large-v3'].installed, true)
    assert.equal(models.turbo.installed, false, 'large-v3 is not the turbo model')
  } finally { l.cleanup() }
})

test('turbo counts only from the repo faster-whisper loads it from', async () => {
  const l = layout()
  try {
    const size = hubModel(l.hub, 'models--mobiuslabsgmbh--faster-whisper-large-v3-turbo', { bytes: 500 })
    assert.equal(byName(await list(l)).turbo.sizeOnDisk, size)
    const l2 = layout()
    try {
      hubModel(l2.hub, 'models--deepdml--faster-whisper-large-v3-turbo-ct2', { bytes: 600 })
      assert.equal(byName(await list(l2)).turbo.installed, false)
    } finally { l2.cleanup() }
  } finally { l.cleanup() }
})

test('without symlinks (Windows default) the snapshot holds real files and those are counted', async () => {
  const l = layout()
  try {
    const root = path.join(l.hub, 'models--Systran--faster-whisper-small')
    writeBytes(path.join(root, 'snapshots', 'rev1', 'model.bin'), 800)
    writeBytes(path.join(root, 'snapshots', 'rev1', 'tokenizer.json'), 50)
    const small = byName(await list(l)).small
    assert.deepEqual([small.installed, small.sizeOnDisk], [true, 850])
  } finally { l.cleanup() }
})

test('a model present for both Whisper backends is reported for the one the engine uses', async () => {
  const l = layout()
  try {
    writeBytes(path.join(l.whisper, 'base.pt'), 100)
    const hubSize = hubModel(l.hub, 'models--Systran--faster-whisper-base', { bytes: 300 })
    assert.deepEqual(pick(byName(await list(l, { checkEngine: engine('faster-whisper') })).base), ['faster-whisper', hubSize])
    assert.deepEqual(pick(byName(await list(l, { checkEngine: engine('openai-whisper') })).base), ['openai-whisper', 100])
    // No usable engine: the first one found, which is openai-whisper's file.
    assert.equal(byName(await list(l, { checkEngine: engine(null) })).base.backend, 'openai-whisper')
  } finally { l.cleanup() }
  function pick(entry) { return [entry.backend, entry.sizeOnDisk] }
})

test('a model the other Whisper downloaded does not count as installed, since a job would download it again', async () => {
  const l = layout()
  try {
    writeBytes(path.join(l.whisper, 'medium.en.pt'), 100)
    hubModel(l.hub, 'models--Systran--faster-whisper-small', { bytes: 300 })
    const faster = byName(await list(l, { checkEngine: engine('faster-whisper') }))
    assert.deepEqual([faster['medium.en'].installed, faster['medium.en'].backend], [false, null])
    assert.equal(faster.small.installed, true)
    const openai = byName(await list(l, { checkEngine: engine('openai-whisper') }))
    assert.deepEqual([openai['medium.en'].installed, openai.small.installed], [true, false])
  } finally { l.cleanup() }
})

test('Enable GPU is offered only on Windows with an NVIDIA GPU and faster-whisper', async () => {
  const l = layout()
  try {
    const gpu = async () => ({ name: 'NVIDIA GeForce RTX 3050 Laptop GPU', vramMB: 4096 })
    assert.equal((await list(l, { gpu, platform: 'win32' })).gpuSetup, true)
    assert.equal((await list(l, { gpu, platform: 'darwin' })).gpuSetup, false)
    assert.equal((await list(l, { gpu, platform: 'linux' })).gpuSetup, false)
    assert.equal((await list(l, { platform: 'win32' })).gpuSetup, false, 'no GPU')
    assert.equal((await list(l, { gpu, platform: 'win32', checkEngine: engine('openai-whisper') })).gpuSetup, false)
    assert.equal((await list(l, { gpu, platform: 'win32', checkEngine: engine(null) })).gpuSetup, false)
  } finally { l.cleanup() }
})

test('a failing engine check does not stop the listing', async () => {
  const l = layout()
  try {
    writeBytes(path.join(l.whisper, 'tiny.pt'), 10)
    const listing = await list(l, { checkEngine: async () => { throw new Error('python exploded') } })
    assert.equal(listing.engine.available, false)
    assert.equal(byName(listing).tiny.installed, true)
  } finally { l.cleanup() }
})

test('GPU details pass through, and a failing GPU probe just means none', async () => {
  const l = layout()
  try {
    const gpu = { name: 'NVIDIA GeForce RTX 3050 Laptop GPU', vramMB: 4096 }
    assert.deepEqual((await list(l, { gpu: async () => gpu })).system.gpu, gpu)
    assert.equal((await list(l, { gpu: async () => { throw new Error('no') } })).system.gpu, null)
  } finally { l.cleanup() }
})

test('caches in non-default places are found through the environment variables', async () => {
  const l = layout()
  try {
    const elsewhere = path.join(l.dir, 'elsewhere')
    writeBytes(path.join(elsewhere, 'whisper', 'tiny.pt'), 42)
    const hubSize = hubModel(path.join(elsewhere, 'hf', 'hub'), 'models--Systran--faster-whisper-small', { bytes: 100 })
    const env = { XDG_CACHE_HOME: elsewhere, HF_HOME: path.join(elsewhere, 'hf') }
    assert.equal(byName(await list(l, { env, checkEngine: engine('openai-whisper') })).tiny.sizeOnDisk, 42)
    assert.equal(byName(await list(l, { env })).small.sizeOnDisk, hubSize)
  } finally { l.cleanup() }
})

// ---- GPU detection -------------------------------------------------------------------

test('nvidia-smi output: the largest GPU wins, junk lines are skipped, names are cleaned and bounded', () => {
  assert.deepEqual(parseNvidiaSmi('NVIDIA GeForce RTX 3050 Laptop GPU, 4096\n'), { name: 'NVIDIA GeForce RTX 3050 Laptop GPU', vramMB: 4096 })
  assert.deepEqual(parseNvidiaSmi('Small GPU, 2048\r\nBig GPU, 24576\r\n'), { name: 'Big GPU', vramMB: 24576 })
  assert.equal(parseNvidiaSmi(''), null)
  assert.equal(parseNvidiaSmi('No devices were found'), null)
  assert.equal(parseNvidiaSmi('GPU, notanumber'), null)
  assert.equal(parseNvidiaSmi('GPU, -5'), null)
  assert.equal(parseNvidiaSmi(`${'x'.repeat(300)}, 1000`).name.length, 80)
})

test('GPU detection runs nvidia-smi without a shell and with a 4 second limit, and failure means null', async () => {
  let call
  const ok = (file, args, options, callback) => { call = { file, args, options }; callback(null, 'Card, 8192\n') }
  assert.deepEqual(await detectGpu({ execFile: ok }), { name: 'Card', vramMB: 8192 })
  assert.equal(call.file, 'nvidia-smi')
  assert.equal(call.options.timeout, 4000)
  assert.equal(call.options.shell, undefined)
  assert.deepEqual(call.args, ['--query-gpu=name,memory.total', '--format=csv,noheader,nounits'])
  assert.equal(await detectGpu({ execFile: (f, a, o, cb) => cb(new Error('ENOENT')) }), null)
  assert.equal(await detectGpu({ execFile: () => { throw new Error('boom') } }), null)
})

test('GPU detection keeps the Windows variables nvidia-smi needs even in a trimmed environment', async () => {
  const saved = { ProgramFiles: process.env.ProgramFiles, ProgramData: process.env.ProgramData }
  process.env.ProgramFiles = 'C:\\Program Files'
  process.env.ProgramData = 'C:\\ProgramData'
  try {
    let seen
    const ok = (file, args, options, callback) => { seen = options.env; callback(null, 'Card, 4096\n') }
    await detectGpu({ execFile: ok, env: { PATH: 'x', SystemRoot: 'C:\\Windows' } })
    if (process.platform === 'win32') {
      assert.equal(seen.ProgramFiles, 'C:\\Program Files', 'nvidia-smi exits 255 without ProgramFiles')
      assert.equal(seen.ProgramData, 'C:\\ProgramData')
    }
    assert.equal(seen.PATH, 'x')
    assert.equal(seen.OPENROUTER_API_KEY, undefined, 'nothing else is passed along')
  } finally {
    for (const [key, value] of Object.entries(saved)) { if (value === undefined) delete process.env[key]; else process.env[key] = value }
  }
})

test('GPU detection tries the older driver location when nvidia-smi is not on PATH (Windows only)', async () => {
  const files = []
  const exec = (file, args, options, callback) => { files.push(file); callback(file === 'nvidia-smi' ? new Error('ENOENT') : null, 'Card, 6144\n') }
  const found = await detectGpu({ execFile: exec })
  if (process.platform === 'win32' && process.env.ProgramFiles) {
    assert.deepEqual(found, { name: 'Card', vramMB: 6144 })
    assert.match(files[1], /NVSMI[\\/]nvidia-smi\.exe$/)
  } else {
    assert.equal(found, null)
    assert.deepEqual(files, ['nvidia-smi'])
  }
})

// ---- deleting ------------------------------------------------------------------------

test('deleting an openai-whisper model removes only that file and reports the bytes freed', async () => {
  const l = layout()
  try {
    writeBytes(path.join(l.whisper, 'tiny.pt'), 1234)
    writeBytes(path.join(l.whisper, 'base.pt'), 99)
    writeBytes(path.join(l.cache, 'unrelated.txt'), 5)
    assert.deepEqual(await del(l, { name: 'tiny', backend: 'openai-whisper' }), { ok: true, freedBytes: 1234 })
    assert.equal(fs.existsSync(path.join(l.whisper, 'tiny.pt')), false)
    assert.equal(fs.existsSync(path.join(l.whisper, 'base.pt')), true)
    assert.equal(fs.existsSync(path.join(l.cache, 'unrelated.txt')), true)
    assert.equal(fs.existsSync(l.whisper), true)
  } finally { l.cleanup() }
})

test('deleting turbo removes large-v3-turbo.pt and leaves large-v3.pt alone', async () => {
  const l = layout()
  try {
    writeBytes(path.join(l.whisper, 'large-v3-turbo.pt'), 50)
    writeBytes(path.join(l.whisper, 'large-v3.pt'), 60)
    assert.deepEqual(await del(l, { name: 'turbo', backend: 'openai-whisper' }), { ok: true, freedBytes: 50 })
    assert.equal(fs.existsSync(path.join(l.whisper, 'large-v3.pt')), true)
  } finally { l.cleanup() }
})

test('deleting a faster-whisper model removes its whole folder, frees the blob bytes, and keeps the others', async () => {
  const l = layout()
  try {
    const size = hubModel(l.hub, 'models--Systran--faster-whisper-small', { bytes: 5000 })
    hubModel(l.hub, 'models--Systran--faster-whisper-small.en', { bytes: 100 })
    hubModel(l.hub, 'models--Systran--faster-whisper-base', { bytes: 100 })
    assert.deepEqual(await del(l, { name: 'small', backend: 'faster-whisper' }), { ok: true, freedBytes: size })
    assert.equal(fs.existsSync(path.join(l.hub, 'models--Systran--faster-whisper-small')), false)
    assert.equal(fs.existsSync(path.join(l.hub, 'models--Systran--faster-whisper-small.en')), true)
    assert.equal(fs.existsSync(path.join(l.hub, 'models--Systran--faster-whisper-base')), true)
    const listed = byName(await list(l))
    assert.equal(listed.small.installed, false)
    assert.equal(listed['small.en'].installed, true)
  } finally { l.cleanup() }
})

test('deleting turbo removes the repo faster-whisper uses and leaves other folders alone', async () => {
  const l = layout()
  try {
    const a = hubModel(l.hub, 'models--mobiuslabsgmbh--faster-whisper-large-v3-turbo', { bytes: 700 })
    hubModel(l.hub, 'models--deepdml--faster-whisper-large-v3-turbo-ct2', { bytes: 900 })
    hubModel(l.hub, 'models--Systran--faster-whisper-large-v3', { bytes: 1 })
    assert.deepEqual(await del(l, { name: 'turbo', backend: 'faster-whisper' }), { ok: true, freedBytes: a })
    assert.equal(fs.existsSync(path.join(l.hub, 'models--Systran--faster-whisper-large-v3')), true)
    assert.equal(fs.existsSync(path.join(l.hub, 'models--deepdml--faster-whisper-large-v3-turbo-ct2')), true)
  } finally { l.cleanup() }
})

test('a model that is not installed is reported as such, for either backend', async () => {
  const l = layout()
  try {
    assert.deepEqual(await del(l, { name: 'tiny', backend: 'openai-whisper' }), { ok: false, message: WHISPER_MESSAGES.notInstalled })
    assert.deepEqual(await del(l, { name: 'tiny', backend: 'faster-whisper' }), { ok: false, message: WHISPER_MESSAGES.notInstalled })
    // Installed for the other backend only.
    writeBytes(path.join(l.whisper, 'tiny.pt'), 5)
    assert.deepEqual(await del(l, { name: 'tiny', backend: 'faster-whisper' }), { ok: false, message: WHISPER_MESSAGES.notInstalled })
    assert.equal(fs.existsSync(path.join(l.whisper, 'tiny.pt')), true)
  } finally { l.cleanup() }
})

test('missing cache folders are "not installed", not an error', async () => {
  const l = layout()
  try {
    fs.rmSync(l.cache, { recursive: true, force: true })
    assert.equal((await del(l, { name: 'small', backend: 'openai-whisper' })).message, WHISPER_MESSAGES.notInstalled)
    assert.equal((await del(l, { name: 'small', backend: 'faster-whisper' })).message, WHISPER_MESSAGES.notInstalled)
    assert.equal((await list(l)).models.every((m) => !m.installed), true)
  } finally { l.cleanup() }
})

// ---- deleting: input and path-traversal safety -----------------------------------------

test('only names from the fixed list and the two backends are accepted; nothing else reaches a path', async () => {
  const l = layout()
  try {
    writeBytes(path.join(l.whisper, 'tiny.pt'), 5)
    writeBytes(path.join(l.cache, 'secret.txt'), 5)
    writeBytes(path.join(l.dir, 'outside.pt'), 5)
    const badNames = [
      '', ' tiny', 'TINY', 'tiny.pt', 'tiny.pt\0', 'large', 'large-v2', 'large-v3-turbo', 'whisper-1', 'evil',
      '..', '../tiny', '../../secret', '..\\..\\secret', 'tiny/../tiny', 'tiny/', '/tiny', 'C:\\Windows\\system32',
      '../outside', 'models--Systran--faster-whisper-tiny', `..${path.sep}..${path.sep}outside`,
      7, null, undefined, {}, ['tiny'], true
    ]
    for (const name of badNames) {
      for (const backend of ['openai-whisper', 'faster-whisper']) {
        assert.deepEqual(await del(l, { name, backend }), { ok: false, message: WHISPER_MESSAGES.unknownModel }, `${JSON.stringify(name)} / ${backend}`)
      }
    }
    for (const backend of ['', 'whisper.cpp', 'openai-whisper/..', 'FASTER-WHISPER', '../whisper', null, undefined, 3, {}]) {
      assert.deepEqual(await del(l, { name: 'tiny', backend }), { ok: false, message: WHISPER_MESSAGES.unknownModel }, JSON.stringify(backend))
    }
    for (const request of [null, undefined, 'tiny', 5, [], { path: path.join(l.whisper, 'tiny.pt') }]) {
      assert.deepEqual(await del(l, request), { ok: false, message: WHISPER_MESSAGES.unknownModel }, JSON.stringify(request))
    }
    // A path field is never read, even next to valid values.
    assert.deepEqual(await del(l, { name: 'base', backend: 'openai-whisper', path: path.join(l.dir, 'outside.pt') }), { ok: false, message: WHISPER_MESSAGES.notInstalled })
    assert.equal(fs.existsSync(path.join(l.whisper, 'tiny.pt')), true)
    assert.equal(fs.existsSync(path.join(l.cache, 'secret.txt')), true)
    assert.equal(fs.existsSync(path.join(l.dir, 'outside.pt')), true)
  } finally { l.cleanup() }
})

test('deleting is refused while any clip job or editor bake is running, and nothing is removed', async () => {
  const l = layout()
  try {
    writeBytes(path.join(l.whisper, 'tiny.pt'), 5)
    hubModel(l.hub, 'models--Systran--faster-whisper-small')
    assert.deepEqual(await del(l, { name: 'tiny', backend: 'openai-whisper' }, { isBusy: () => true }), { ok: false, message: WHISPER_MESSAGES.busy })
    assert.deepEqual(await del(l, { name: 'small', backend: 'faster-whisper' }, { isBusy: () => true }), { ok: false, message: WHISPER_MESSAGES.busy })
    assert.equal(fs.existsSync(path.join(l.whisper, 'tiny.pt')), true)
    assert.equal(fs.existsSync(path.join(l.hub, 'models--Systran--faster-whisper-small')), true)
    // Once the job ends the same request works.
    assert.equal((await del(l, { name: 'tiny', backend: 'openai-whisper' })).ok, true)
  } finally { l.cleanup() }
})

test('the busy check runs before anything is looked up, so a busy refusal reveals nothing about the disk', async () => {
  const l = layout()
  try {
    assert.equal((await del(l, { name: 'tiny', backend: 'openai-whisper' }, { isBusy: () => true })).message, WHISPER_MESSAGES.busy)
  } finally { l.cleanup() }
})

test('a directory where a model file is expected, and a file where a model folder is expected, are refused untouched', async () => {
  const l = layout()
  try {
    fs.mkdirSync(path.join(l.whisper, 'tiny.pt'))
    writeBytes(path.join(l.whisper, 'tiny.pt', 'inner.txt'), 5)
    assert.deepEqual(await del(l, { name: 'tiny', backend: 'openai-whisper' }), { ok: false, message: WHISPER_MESSAGES.failed })
    assert.equal(fs.existsSync(path.join(l.whisper, 'tiny.pt', 'inner.txt')), true)

    writeBytes(path.join(l.hub, 'models--Systran--faster-whisper-small'), 5)
    assert.deepEqual(await del(l, { name: 'small', backend: 'faster-whisper' }), { ok: false, message: WHISPER_MESSAGES.failed })
    assert.equal(fs.existsSync(path.join(l.hub, 'models--Systran--faster-whisper-small')), true)

    const listed = byName(await list(l))
    assert.equal(listed.tiny.installed, false)
    assert.equal(listed.small.installed, false)
  } finally { l.cleanup() }
})

test('failure messages are fixed text: no paths, file names or operating-system errors', async () => {
  const l = layout()
  try {
    writeBytes(path.join(l.whisper, 'tiny.pt'), 5)
    const throwing = {
      lstat: async (p) => fs.lstatSync(p),
      readdir: async (p) => fs.readdirSync(p),
      realpath: async (p) => fs.realpathSync(p),
      rm: async (p) => { throw new Error(`EBUSY: resource busy or locked, unlink '${p}'`) }
    }
    const result = await del(l, { name: 'tiny', backend: 'openai-whisper' }, { fs: throwing })
    assert.deepEqual(result, { ok: false, message: WHISPER_MESSAGES.failed })
    assert.equal(JSON.stringify(result).includes(l.dir), false)
    const all = Object.values(WHISPER_MESSAGES).join(' ')
    assert.equal(all.includes(path.sep) && /[A-Za-z]:\\|\/home\//.test(all), false)
  } finally { l.cleanup() }
})

// ---- deleting: symlinks (real, where this machine allows them) ---------------------------

test('a symlinked model file is neither listed nor deleted, and its target survives', (t) => {
  const l = layout()
  try {
    const outside = path.join(l.dir, 'outside', 'precious.pt')
    writeBytes(outside, 321)
    if (!trySymlink(outside, path.join(l.whisper, 'tiny.pt'), 'file')) return t.skip(SYMLINK_SKIP)
    return (async () => {
      assert.equal(byName(await list(l)).tiny.installed, false)
      assert.deepEqual(await del(l, { name: 'tiny', backend: 'openai-whisper' }), { ok: false, message: WHISPER_MESSAGES.failed })
      assert.equal(fs.existsSync(outside), true)
      assert.equal(fs.lstatSync(path.join(l.whisper, 'tiny.pt')).isSymbolicLink(), true)
    })().finally(l.cleanup)
  } catch (error) { l.cleanup(); throw error }
})

test('a symlinked whisper folder (a parent escape) is neither read nor deleted from', (t) => {
  const l = layout()
  try {
    const outsideDir = path.join(l.dir, 'outside-dir')
    writeBytes(path.join(outsideDir, 'tiny.pt'), 77)
    fs.rmSync(l.whisper, { recursive: true, force: true })
    if (!trySymlink(outsideDir, l.whisper, 'junction')) return t.skip(SYMLINK_SKIP)
    return (async () => {
      assert.equal(byName(await list(l)).tiny.installed, false)
      assert.deepEqual(await del(l, { name: 'tiny', backend: 'openai-whisper' }), { ok: false, message: WHISPER_MESSAGES.failed })
      assert.equal(fs.existsSync(path.join(outsideDir, 'tiny.pt')), true)
    })().finally(l.cleanup)
  } catch (error) { l.cleanup(); throw error }
})

test('a symlinked Hugging Face model folder is neither listed nor deleted, and its target survives', (t) => {
  const l = layout()
  try {
    const outsideDir = path.join(l.dir, 'outside-model')
    hubModel(path.join(l.dir, 'outside-hub'), 'models--Systran--faster-whisper-small', { bytes: 500 })
    fs.cpSync(path.join(l.dir, 'outside-hub', 'models--Systran--faster-whisper-small'), outsideDir, { recursive: true })
    if (!trySymlink(outsideDir, path.join(l.hub, 'models--Systran--faster-whisper-small'), 'junction')) return t.skip(SYMLINK_SKIP)
    return (async () => {
      assert.equal(byName(await list(l)).small.installed, false)
      assert.deepEqual(await del(l, { name: 'small', backend: 'faster-whisper' }), { ok: false, message: WHISPER_MESSAGES.failed })
      assert.equal(fs.existsSync(path.join(outsideDir, 'blobs', 'sha-model')), true)
    })().finally(l.cleanup)
  } catch (error) { l.cleanup(); throw error }
})

test('links inside a real model folder are removed as links; what they point at is never touched', (t) => {
  const l = layout()
  try {
    const outside = path.join(l.dir, 'outside', 'keep.txt')
    writeBytes(outside, 9)
    hubModel(l.hub, 'models--Systran--faster-whisper-small', { bytes: 100 })
    if (!trySymlink(outside, path.join(l.hub, 'models--Systran--faster-whisper-small', 'snapshots', 'rev1', 'extra.txt'), 'file')) return t.skip(SYMLINK_SKIP)
    return (async () => {
      assert.equal((await del(l, { name: 'small', backend: 'faster-whisper' })).ok, true)
      assert.equal(fs.existsSync(outside), true)
    })().finally(l.cleanup)
  } catch (error) { l.cleanup(); throw error }
})

// ---- deleting: symlink logic through a fake file system (runs everywhere) --------------

const ROOT = path.resolve(path.sep, 'fake-home')
const P = (...parts) => path.join(ROOT, ...parts)

/** A tiny in-memory file system: nodes by absolute path, with kind 'file' | 'dir' | 'link'. */
function memoryFs(nodes, { aliases = {}, swapAfterFirstLstat = {} } = {}) {
  const rmCalls = []
  const lstatCalls = new Map()
  const norm = (p) => path.normalize(p)
  const missing = () => Object.assign(new Error('ENOENT'), { code: 'ENOENT' })
  const memory = {
    rmCalls,
    lstat: async (p) => {
      const key = norm(p)
      const count = (lstatCalls.get(key) ?? 0) + 1
      lstatCalls.set(key, count)
      const node = count > 1 && swapAfterFirstLstat[key] ? swapAfterFirstLstat[key] : nodes[key]
      if (!node) throw missing()
      return { size: node.size ?? 0, isFile: () => node.kind === 'file', isDirectory: () => node.kind === 'dir', isSymbolicLink: () => node.kind === 'link' }
    },
    readdir: async (p) => {
      const key = norm(p)
      if (!nodes[key]) throw missing()
      const prefix = key + path.sep
      return [...new Set(Object.keys(nodes).filter((k) => k.startsWith(prefix)).map((k) => k.slice(prefix.length).split(path.sep)[0]))]
    },
    realpath: async (p) => aliases[norm(p)] ?? norm(p),
    rm: async (p, options) => { rmCalls.push({ path: norm(p), ...options }) }
  }
  return memory
}

const fakeDeps = (memory) => ({ env: {}, home: ROOT, fs: memory, isBusy: () => false })
const cache = P('.cache')
const whisperDir = P('.cache', 'whisper')
const hubDir = P('.cache', 'huggingface', 'hub')

test('fake fs: a normal model file is removed with exactly that path, non-recursively and without force', async () => {
  const memory = memoryFs({ [cache]: { kind: 'dir' }, [whisperDir]: { kind: 'dir' }, [path.join(whisperDir, 'tiny.pt')]: { kind: 'file', size: 42 } })
  assert.deepEqual(await deleteWhisperModel({ name: 'tiny', backend: 'openai-whisper' }, fakeDeps(memory)), { ok: true, freedBytes: 42 })
  assert.deepEqual(memory.rmCalls, [{ path: path.join(whisperDir, 'tiny.pt'), recursive: false, force: false }])
})

test('fake fs: a normal Hugging Face folder is removed recursively, only after it is checked to be a real directory', async () => {
  const folder = path.join(hubDir, 'models--Systran--faster-whisper-small')
  const memory = memoryFs({
    [hubDir]: { kind: 'dir' }, [folder]: { kind: 'dir' },
    [path.join(folder, 'blobs')]: { kind: 'dir' }, [path.join(folder, 'blobs', 'sha')]: { kind: 'file', size: 900 },
    [path.join(folder, 'snapshots')]: { kind: 'dir' }, [path.join(folder, 'snapshots', 'r1')]: { kind: 'dir' },
    [path.join(folder, 'snapshots', 'r1', 'model.bin')]: { kind: 'link' }
  })
  assert.deepEqual(await deleteWhisperModel({ name: 'small', backend: 'faster-whisper' }, fakeDeps(memory)), { ok: true, freedBytes: 900 })
  assert.deepEqual(memory.rmCalls, [{ path: folder, recursive: true, force: false }])
})

test('fake fs: a symlinked model file is refused and nothing is removed', async () => {
  const memory = memoryFs({ [cache]: { kind: 'dir' }, [whisperDir]: { kind: 'dir' }, [path.join(whisperDir, 'tiny.pt')]: { kind: 'link', size: 42 } })
  assert.deepEqual(await deleteWhisperModel({ name: 'tiny', backend: 'openai-whisper' }, fakeDeps(memory)), { ok: false, message: WHISPER_MESSAGES.failed })
  assert.deepEqual(memory.rmCalls, [])
  const listed = byName(await listWhisperModels({ env: {}, home: ROOT, fs: memory, checkEngine: engine(null), gpu: async () => null }))
  assert.equal(listed.tiny.installed, false)
})

test('fake fs: a symlinked whisper folder is refused and nothing is removed', async () => {
  const memory = memoryFs({ [cache]: { kind: 'dir' }, [whisperDir]: { kind: 'link' }, [path.join(whisperDir, 'tiny.pt')]: { kind: 'file', size: 42 } })
  assert.deepEqual(await deleteWhisperModel({ name: 'tiny', backend: 'openai-whisper' }, fakeDeps(memory)), { ok: false, message: WHISPER_MESSAGES.failed })
  assert.deepEqual(memory.rmCalls, [])
  const listed = byName(await listWhisperModels({ env: {}, home: ROOT, fs: memory, checkEngine: engine(null), gpu: async () => null }))
  assert.equal(listed.tiny.installed, false)
})

test('fake fs: a symlinked Hugging Face folder is refused and nothing else is touched', async () => {
  const linked = path.join(hubDir, 'models--mobiuslabsgmbh--faster-whisper-large-v3-turbo')
  const good = path.join(hubDir, 'models--Systran--faster-whisper-large-v3')
  const nodes = { [hubDir]: { kind: 'dir' }, [linked]: { kind: 'link' }, [good]: { kind: 'dir' } }
  for (const folder of [linked, good]) {
    nodes[path.join(folder, 'snapshots')] = { kind: 'dir' }
    nodes[path.join(folder, 'snapshots', 'r1')] = { kind: 'dir' }
    nodes[path.join(folder, 'snapshots', 'r1', 'model.bin')] = { kind: 'file', size: 5 }
  }
  const memory = memoryFs(nodes)
  assert.deepEqual(await deleteWhisperModel({ name: 'turbo', backend: 'faster-whisper' }, fakeDeps(memory)), { ok: false, message: WHISPER_MESSAGES.failed })
  assert.deepEqual(memory.rmCalls, [], 'the linked folder is refused, and large-v3 is a different model')
})

test('fake fs: a file swapped for a symlink between the check and the removal is caught', async () => {
  const target = path.join(whisperDir, 'tiny.pt')
  const memory = memoryFs(
    { [cache]: { kind: 'dir' }, [whisperDir]: { kind: 'dir' }, [target]: { kind: 'file', size: 42 } },
    { swapAfterFirstLstat: { [target]: { kind: 'link', size: 42 } } }
  )
  assert.deepEqual(await deleteWhisperModel({ name: 'tiny', backend: 'openai-whisper' }, fakeDeps(memory)), { ok: false, message: WHISPER_MESSAGES.failed })
  assert.deepEqual(memory.rmCalls, [])
})

test('fake fs: a cache root that was relocated is resolved once; entries inside it are still checked', async () => {
  const moved = path.resolve(path.sep, 'big-drive', 'cache')
  const movedWhisper = path.join(moved, 'whisper')
  const ok = memoryFs(
    { [moved]: { kind: 'dir' }, [movedWhisper]: { kind: 'dir' }, [path.join(movedWhisper, 'tiny.pt')]: { kind: 'file', size: 8 } },
    { aliases: { [cache]: moved } }
  )
  assert.deepEqual(await deleteWhisperModel({ name: 'tiny', backend: 'openai-whisper' }, fakeDeps(ok)), { ok: true, freedBytes: 8 })
  assert.deepEqual(ok.rmCalls.map((c) => c.path), [path.join(movedWhisper, 'tiny.pt')])

  const bad = memoryFs(
    { [moved]: { kind: 'dir' }, [movedWhisper]: { kind: 'link' }, [path.join(movedWhisper, 'tiny.pt')]: { kind: 'file', size: 8 } },
    { aliases: { [cache]: moved } }
  )
  assert.equal((await deleteWhisperModel({ name: 'tiny', backend: 'openai-whisper' }, fakeDeps(bad))).ok, false)
  assert.deepEqual(bad.rmCalls, [])
})

test('fake fs: nothing but the requested model is ever passed to rm, whatever else is in the caches', async () => {
  const memory = memoryFs({
    [cache]: { kind: 'dir' }, [whisperDir]: { kind: 'dir' },
    [path.join(whisperDir, 'tiny.pt')]: { kind: 'file', size: 1 }, [path.join(whisperDir, 'base.pt')]: { kind: 'file', size: 2 },
    [P('.cache', 'other')]: { kind: 'dir' }
  })
  await deleteWhisperModel({ name: 'tiny', backend: 'openai-whisper' }, fakeDeps(memory))
  assert.deepEqual(memory.rmCalls.map((c) => c.path), [path.join(whisperDir, 'tiny.pt')])
})
