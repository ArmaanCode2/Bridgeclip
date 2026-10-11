'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const { loadMain } = require('../zernio/support/load-main.cjs')

const w = loadMain("export * from './src/shared/whisper-models'")
const providers = loadMain("export * from './src/shared/providers'")
const {
  fitWarning, formatMemory, formatBytes, deviceNote, downloadNote, deleteConfirmReducer, INITIAL_DELETE_CONFIRM,
  deleteBlockedReason, WHISPER_APPROX_MEMORY_MB, WHISPER_MODEL_NAMES, DEVICE_EXPLANATIONS, englishOnlyHint
} = w

const memory = (name) => WHISPER_APPROX_MEMORY_MB[name]
const system = (ramMB, gpu) => ({ ramMB, gpu })
const GPU_4GB = { name: 'NVIDIA GeForce RTX 3050 Laptop GPU', vramMB: 4096 }

test('formatting: megabytes under a gigabyte, half-gigabyte steps above', () => {
  assert.equal(formatMemory(400), '400 MB')
  assert.equal(formatMemory(999), '999 MB')
  assert.equal(formatMemory(1000), '1 GB')
  assert.equal(formatMemory(1500), '1.5 GB')
  assert.equal(formatMemory(4096), '4 GB')
  assert.equal(formatMemory(6000), '6 GB')
  assert.equal(formatMemory(4500), '4.5 GB')
  assert.equal(formatMemory(10000), '10 GB')
  assert.equal(formatMemory(0), '0 MB')
  assert.equal(formatMemory(NaN), '0 MB')
  assert.equal(formatBytes(142 * 1024 * 1024), '142 MB')
  assert.equal(formatBytes(1.5 * 1024 ** 3), '1.5 GB')
  assert.equal(formatBytes(1), '1 MB')
  assert.equal(formatBytes(-4), '0 MB')
})

test('a model that needs more video memory than the GPU has is flagged in the exact words the panel shows', () => {
  const warning = fitWarning({ memory: memory('turbo'), device: 'gpu', system: system(23 * 1024, GPU_4GB) })
  assert.equal(warning, 'Needs about 6 GB of video memory; you have 4 GB. Use CPU.')
  // Auto uses the GPU when there is one, so it is judged the same way.
  assert.equal(fitWarning({ memory: memory('turbo'), device: 'auto', system: system(23 * 1024, GPU_4GB) }), warning)
})

test('models that fit are not flagged', () => {
  for (const name of ['tiny', 'base', 'small.en']) {
    assert.equal(fitWarning({ memory: memory(name), device: 'gpu', system: system(8192, GPU_4GB) }), null, name)
  }
  assert.equal(fitWarning({ memory: memory('medium'), device: 'gpu', system: system(8192, { name: 'x', vramMB: 8192 }) }), null)
})

test('CPU runs are judged against system memory with headroom, and never against the GPU', () => {
  // medium needs about 2.5 GB: fine on 8 GB, not on 3 GB.
  assert.equal(fitWarning({ memory: memory('medium'), device: 'cpu', system: system(8192, GPU_4GB) }), null)
  assert.equal(fitWarning({ memory: memory('medium'), device: 'cpu', system: system(3072, null) }),
    'Needs about 2.5 GB of memory; you have 3 GB. Pick a smaller model.')
  // turbo would not fit a 4 GB GPU, but on the CPU only RAM matters.
  assert.equal(fitWarning({ memory: memory('turbo'), device: 'cpu', system: system(23 * 1024, GPU_4GB) }), null)
  // Exactly at the 75% line is allowed; just over is not.
  assert.equal(fitWarning({ memory: { cpu: 750, gpu: 0 }, device: 'cpu', system: system(1000, null) }), null)
  assert.notEqual(fitWarning({ memory: { cpu: 751, gpu: 0 }, device: 'cpu', system: system(1000, null) }), null)
})

test('Auto without a GPU is judged on system memory; GPU without a GPU gets no per-model warning (the device note covers it)', () => {
  assert.equal(fitWarning({ memory: memory('large-v3'), device: 'auto', system: system(4096, null) }),
    'Needs about 4.5 GB of memory; you have 4 GB. Pick a smaller model.')
  assert.equal(fitWarning({ memory: memory('large-v3'), device: 'gpu', system: system(23 * 1024, null) }), null)
})

test('no system information means no warning rather than a guess', () => {
  assert.equal(fitWarning({ memory: memory('large-v3'), device: 'gpu', system: null }), null)
  assert.equal(fitWarning({ memory: memory('large-v3'), device: 'cpu', system: system(0, null) }), null)
  assert.equal(fitWarning({ memory: memory('large-v3'), device: 'gpu', system: system(8192, { name: 'x', vramMB: 0 }) }), null)
})

test('choosing GPU with no CUDA device says so; other devices and a found GPU say nothing', () => {
  const none = deviceNote('gpu', system(8192, null), { available: true, backend: 'faster-whisper', detail: '', device: 'cpu', modelCached: null })
  assert.match(none, /No CUDA GPU was detected/)
  assert.equal(deviceNote('gpu', system(8192, GPU_4GB), null), null)
  assert.equal(deviceNote('gpu', system(8192, null), { available: true, backend: 'faster-whisper', detail: '', device: 'cuda', modelCached: null }), null)
  assert.equal(deviceNote('auto', system(8192, null), null), null)
  assert.equal(deviceNote('cpu', system(8192, null), null), null)
  // Before the first answer arrives there is nothing to claim yet only if both sources are silent.
  assert.match(deviceNote('gpu', null, null), /No CUDA GPU was detected/)
})

test('every device has a one-line explanation in the words the panel promises', () => {
  assert.equal(DEVICE_EXPLANATIONS.cpu, 'CPU uses system RAM and is slower.')
  assert.equal(DEVICE_EXPLANATIONS.gpu, 'GPU is faster but needs enough video memory and the CUDA libraries.')
  assert.equal(DEVICE_EXPLANATIONS.auto, 'Auto uses the GPU when it works and falls back to the CPU.')
  for (const text of Object.values(DEVICE_EXPLANATIONS)) assert.equal(text.includes('\n'), false)
})

test('the download note uses the sizes the picker already knows', () => {
  assert.equal(downloadNote(75), 'About 75 MB to download')
  assert.equal(downloadNote(1500), 'About 1.5 GB to download')
  const sizes = Object.fromEntries(providers.LOCAL_WHISPER_MODELS.map((m) => [m.id, m.downloadMb]))
  for (const name of WHISPER_MODEL_NAMES) assert.ok(sizes[name] > 0, `${name} has a download size`)
})

test('every listed model has a picker entry, a hint, and a memory figure; the list and the table agree', () => {
  assert.deepEqual(Object.keys(WHISPER_APPROX_MEMORY_MB).sort(), [...WHISPER_MODEL_NAMES].sort())
  for (const name of WHISPER_MODEL_NAMES) {
    const info = providers.LOCAL_WHISPER_MODELS.find((m) => m.id === name)
    assert.ok(info && info.hint && info.label, name)
    assert.equal(providers.isBuiltinWhisperModel(name), true, `${name} can be selected in built-in mode`)
    assert.ok(WHISPER_APPROX_MEMORY_MB[name].gpu > WHISPER_APPROX_MEMORY_MB[name].cpu / 2)
  }
  assert.match(englishOnlyHint('tiny.en', 'Fast.'), /^English only/)
  assert.equal(englishOnlyHint('tiny', 'Fast.'), 'Fast.')
})

test('delete confirmation: ask, cancel, then ask again, and only one model at a time', () => {
  let state = INITIAL_DELETE_CONFIRM
  state = deleteConfirmReducer(state, { type: 'ask', name: 'tiny' })
  assert.deepEqual(state, { pending: 'tiny', busy: false })
  state = deleteConfirmReducer(state, { type: 'ask', name: 'base' })
  assert.equal(state.pending, 'base', 'asking about another model replaces the first question')
  state = deleteConfirmReducer(state, { type: 'cancel' })
  assert.deepEqual(state, INITIAL_DELETE_CONFIRM)
})

test('delete confirmation: confirming starts work that cannot be cancelled or re-asked until it finishes', () => {
  let state = deleteConfirmReducer(INITIAL_DELETE_CONFIRM, { type: 'ask', name: 'small' })
  state = deleteConfirmReducer(state, { type: 'start' })
  assert.deepEqual(state, { pending: 'small', busy: true })
  assert.deepEqual(deleteConfirmReducer(state, { type: 'cancel' }), state, 'cancel is ignored mid-delete')
  assert.deepEqual(deleteConfirmReducer(state, { type: 'ask', name: 'base' }), state, 'a second question is ignored mid-delete')
  assert.deepEqual(deleteConfirmReducer(state, { type: 'finish' }), INITIAL_DELETE_CONFIRM)
})

test('delete confirmation: starting without a question does nothing, and finishing always resets', () => {
  assert.deepEqual(deleteConfirmReducer(INITIAL_DELETE_CONFIRM, { type: 'start' }), INITIAL_DELETE_CONFIRM)
  assert.deepEqual(deleteConfirmReducer({ pending: 'x', busy: true }, { type: 'finish' }), INITIAL_DELETE_CONFIRM)
})

test('Delete is blocked with a reason while a job runs or a delete is in progress', () => {
  assert.equal(deleteBlockedReason({ jobRunning: false, busy: false }), null)
  assert.equal(deleteBlockedReason({ jobRunning: true, busy: false }), 'A clip job is running. Try again when it finishes.')
  assert.equal(deleteBlockedReason({ jobRunning: true, busy: true }), 'A clip job is running. Try again when it finishes.')
  assert.equal(deleteBlockedReason({ jobRunning: false, busy: true }), 'Deleting…')
})

test('selecting a model only ever uses names from the list, in the built-in validation too', () => {
  for (const name of WHISPER_MODEL_NAMES) assert.equal(providers.isBuiltinWhisperModel(name), true)
  for (const name of ['', 'whisper-1', '../tiny', 'tiny.pt', 'gpt-4']) assert.equal(providers.isBuiltinWhisperModel(name), false)
})
