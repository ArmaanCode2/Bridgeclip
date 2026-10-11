'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const { EventEmitter } = require('node:events')
const { PassThrough } = require('node:stream')
const { loadMain } = require('../zernio/support/load-main.cjs')

const w = loadMain("export * from './src/shared/whisper-models'")
const { localWhisperBlock, whisperSetupBlock } = loadMain("export * from './src/main/job-gates'")

/** A fresh copy of the module, so the one-task-at-a-time state never leaks between tests. */
const setup = () => loadMain("export * from './src/main/whisper-setup'")

/**
 * A stand-in for spawn. Each call takes the next script: lines for stdout, then an exit code.
 * `hold: true` keeps the process running until killed or released.
 */
function fakeSpawn(scripts) {
  const calls = []
  const spawn = (file, args, options) => {
    const script = scripts[calls.length] ?? { lines: [], code: 0 }
    const child = new EventEmitter()
    child.stdout = new PassThrough()
    child.stderr = new PassThrough()
    child.killed = false
    const finish = (code) => {
      child.stdout.end()
      child.stderr.end()
      setImmediate(() => child.emit('close', code))
    }
    child.kill = () => { child.killed = true; finish(null); return true }
    child.release = () => finish(script.code ?? 0)
    calls.push({ file, args, options, child })
    setImmediate(() => {
      for (const line of script.lines ?? []) child.stdout.write(`${line}\n`)
      if (script.stderr) child.stderr.write(script.stderr)
      if (!script.hold) finish(script.code ?? 0)
    })
    return child
  }
  return { spawn, calls }
}

const deps = (spawn, extra = {}) => ({
  python: 'C:\\BridgeClip\\python.exe', enginePath: 'C:\\BridgeClip\\engine', env: { PATH: 'x' },
  isBusy: () => false, platform: 'win32', spawn, processEnv: {}, ...extra
})
const json = (value) => JSON.stringify(value)

test('download runs the engine script with the model name and reports byte progress, then a finished event', async () => {
  const s = setup()
  const { spawn, calls } = fakeSpawn([{ lines: [
    'some library chatter',
    json({ event: 'progress', done: 100, total: 1000 }),
    json({ event: 'progress', done: 1000, total: 1000 }),
    json({ event: 'done', backend: 'faster-whisper' })
  ] }])
  const events = []
  const result = await s.downloadWhisperModel('small', deps(spawn), (p) => events.push(p))
  assert.deepEqual(result, { ok: true })
  assert.equal(calls.length, 1)
  assert.equal(calls[0].file, 'C:\\BridgeClip\\python.exe')
  assert.match(calls[0].args[0], /whisper_setup\.py$/)
  assert.deepEqual(calls[0].args.slice(1), ['download', 'small'])
  assert.equal(calls[0].options.windowsHide, true)
  assert.equal(calls[0].options.shell, undefined)
  assert.deepEqual(events.map((e) => [e.done, e.total, Boolean(e.finished)]), [[0, 0, false], [100, 1000, false], [1000, 1000, false], [1000, 1000, true]])
  assert.ok(events.every((e) => e.task === 'download' && e.model === 'small'))
  assert.deepEqual(events.at(-1).result, { ok: true }, 'the finished event carries the result for a panel that did not start the task')
  assert.equal(s.whisperSetupRunning(), false)
  assert.equal(s.whisperSetupState(), null)
})

test('a panel opened mid-task can ask what is running', async () => {
  const s = setup()
  const { spawn, calls } = fakeSpawn([{ lines: [json({ event: 'progress', done: 40, total: 80 })], hold: true }])
  const running = s.downloadWhisperModel('base', deps(spawn), () => {})
  await new Promise((resolve) => setTimeout(resolve, 20))
  assert.deepEqual(s.whisperSetupState(), { task: 'download', model: 'base', step: 'download', done: 40, total: 80 })
  calls[0].child.release()
  await running
  assert.equal(s.whisperSetupState(), null)
})

test('only a model from the fixed list ever reaches the command line', async () => {
  const s = setup()
  const { spawn, calls } = fakeSpawn([])
  for (const bad of ['small; rm -rf /', '../../evil', '', null, 42, { name: 'small' }, 'large-v2']) {
    const result = await s.downloadWhisperModel(bad, deps(spawn), () => {})
    assert.equal(result.ok, false, JSON.stringify(bad))
    assert.equal(result.message, w.WHISPER_MESSAGES.unknownModel)
  }
  assert.equal(calls.length, 0)
})

test('a failed download shows a fixed message for the reason, never the script or library text', async () => {
  for (const [reason, pattern] of [['network', /internet connection/], ['disk', /disk space/], ['missing_backend', /not installed/], ['weird', /Could not download/]]) {
    const s = setup()
    const { spawn } = fakeSpawn([{ lines: [json({ event: 'error', reason, message: 'C:\\Users\\secret\\path exploded' })], code: 1 }])
    const result = await s.downloadWhisperModel('tiny', deps(spawn), () => {})
    assert.equal(result.ok, false)
    assert.equal(result.cancelled, false)
    assert.match(result.message, pattern, reason)
    assert.doesNotMatch(result.message, /secret/)
  }
  // An exit without any JSON (Python crashed) is a plain failure too.
  const s = setup()
  const { spawn } = fakeSpawn([{ lines: ['Traceback (most recent call last):'], code: 1 }])
  assert.match((await s.downloadWhisperModel('tiny', deps(spawn), () => {})).message, /Could not download/)
})

test('one task at a time: a second request while one runs is refused without starting anything', async () => {
  const s = setup()
  const { spawn, calls } = fakeSpawn([{ lines: [json({ event: 'done' })], hold: true }])
  const first = s.downloadWhisperModel('small', deps(spawn), () => {})
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(s.whisperSetupRunning(), true)
  const second = await s.downloadWhisperModel('base', deps(spawn), () => {})
  assert.deepEqual(second, { ok: false, cancelled: false, message: w.WHISPER_SETUP_MESSAGES.busy })
  const gpu = await s.enableWhisperGpu('small', deps(spawn), () => {})
  assert.equal(gpu.message, w.WHISPER_SETUP_MESSAGES.busy)
  assert.equal(calls.length, 1)
  calls[0].child.release()
  assert.deepEqual(await first, { ok: true })
  assert.equal(s.whisperSetupRunning(), false)
})

test('cancel stops the running process and reports it as stopped, not failed', async () => {
  const s = setup()
  const { spawn, calls } = fakeSpawn([{ lines: [json({ event: 'progress', done: 5, total: 10 })], hold: true }])
  const events = []
  const running = s.downloadWhisperModel('medium', deps(spawn), (p) => events.push(p))
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(s.cancelWhisperSetup(), true)
  const result = await running
  assert.equal(calls[0].child.killed, true)
  assert.deepEqual(result, { ok: false, cancelled: true, message: w.WHISPER_SETUP_MESSAGES.cancelled })
  assert.equal(events.at(-1).finished, true)
  assert.equal(s.cancelWhisperSetup(), false, 'nothing left to cancel')
})

test('a Python that cannot start is a plain failure, and the next task can run', async () => {
  const s = setup()
  const throwing = () => { throw new Error('ENOENT') }
  const result = await s.downloadWhisperModel('tiny', deps(throwing), () => {})
  assert.equal(result.ok, false)
  assert.equal(s.whisperSetupRunning(), false)
  const { spawn } = fakeSpawn([{ lines: [json({ event: 'done' })] }])
  assert.deepEqual(await s.downloadWhisperModel('tiny', deps(spawn), () => {}), { ok: true })
})

test('proxy settings reach the download; nothing else from this process does', async () => {
  const s = setup()
  const { spawn, calls } = fakeSpawn([{ lines: [json({ event: 'done' })] }])
  const processEnv = { HTTPS_PROXY: 'http://proxy:8080', NO_PROXY: 'localhost', OPENROUTER_API_KEY: 'sk-or-secret' }
  await s.downloadWhisperModel('tiny', deps(spawn, { processEnv }), () => {})
  const env = calls[0].options.env
  assert.equal(env.HTTPS_PROXY, 'http://proxy:8080')
  assert.equal(env.NO_PROXY, 'localhost')
  assert.equal(env.PATH, 'x')
  assert.equal(env.OPENROUTER_API_KEY, undefined)
})

test('Enable GPU installs NVIDIA\'s CUDA 12 libraries with pip, then tests the chosen model on the GPU', async () => {
  const s = setup()
  const { spawn, calls } = fakeSpawn([
    { lines: ['Successfully installed nvidia-cublas-cu12'], code: 0 },
    { lines: [json({ event: 'done', backend: 'faster-whisper' })] }
  ])
  const steps = []
  const result = await s.enableWhisperGpu('small', deps(spawn), (p) => steps.push(p.finished ? 'finished' : p.step))
  assert.deepEqual(result, { ok: true })
  assert.deepEqual(steps, ['install', 'test', 'finished'])
  assert.deepEqual(calls[0].args.slice(0, 3), ['-m', 'pip', 'install'])
  assert.ok(calls[0].args.some((arg) => arg.startsWith('nvidia-cublas-cu12')))
  assert.ok(calls[0].args.some((arg) => arg.startsWith('nvidia-cudnn-cu12')))
  assert.deepEqual(calls[1].args.slice(1), ['gpu-test', 'small'])
})

test('Enable GPU: a failed pip install stops before the test; a failed test explains what is missing', async () => {
  let s = setup()
  let fake = fakeSpawn([{ lines: ['ERROR: No space left on device'], code: 1 }])
  let result = await s.enableWhisperGpu('small', deps(fake.spawn), () => {})
  assert.deepEqual(result, { ok: false, cancelled: false, message: w.WHISPER_SETUP_MESSAGES.pipFailed })
  assert.equal(fake.calls.length, 1)

  for (const [reason, pattern] of [['gpu_cublas', /cuBLAS/], ['gpu_cudnn', /cuDNN/], ['gpu_memory', /video memory/], ['not_installed', /Install the model/], ['gpu_unavailable', /No NVIDIA GPU/]]) {
    s = setup()
    fake = fakeSpawn([{ lines: [], code: 0 }, { lines: [json({ event: 'error', reason })], code: 1 }])
    result = await s.enableWhisperGpu('small', deps(fake.spawn), () => {})
    assert.equal(result.ok, false)
    assert.match(result.message, pattern, reason)
  }
})

test('Enable GPU is refused off Windows, in an installed app, during a clip job, and for an unknown model, without running anything', async () => {
  const s = setup()
  const { spawn, calls } = fakeSpawn([])
  assert.equal((await s.enableWhisperGpu('small', deps(spawn, { platform: 'darwin' }), () => {})).message, w.WHISPER_SETUP_MESSAGES.gpuUnsupported)
  assert.equal((await s.enableWhisperGpu('small', deps(spawn, { packaged: true }), () => {})).message, w.WHISPER_SETUP_MESSAGES.gpuUnsupported)
  assert.equal((await s.enableWhisperGpu('small', deps(spawn, { isBusy: () => true }), () => {})).message, w.WHISPER_SETUP_MESSAGES.jobRunning)
  assert.equal((await s.enableWhisperGpu('../x', deps(spawn), () => {})).ok, false)
  assert.equal(calls.length, 0)
})

test('panel helpers: download progress text and the model a GPU test uses', () => {
  assert.equal(w.downloadProgressText(0, 0), 'Starting download…')
  assert.equal(w.downloadProgressText(5 * 1024 * 1024, 0), 'Downloading… 5 MB')
  assert.equal(w.downloadProgressText(120 * 1024 * 1024, 480 * 1024 * 1024), 'Downloading… 120 MB of 480 MB (25%)')
  assert.equal(w.downloadProgressText(600, 500), 'Downloading… 1 MB of 1 MB (100%)', 'never above 100%')
  const models = [{ name: 'tiny', installed: true }, { name: 'small', installed: false }, { name: 'medium', installed: true }]
  assert.equal(w.gpuTestModel(models, 'medium'), 'medium')
  assert.equal(w.gpuTestModel(models, 'small'), 'tiny', 'the selected model is not installed: the smallest installed one')
  assert.equal(w.gpuTestModel([{ name: 'tiny', installed: false }], 'tiny'), null)
})

test('a job using built-in Whisper is held back until Whisper and the chosen model are installed', () => {
  const ready = { available: true, backend: 'faster-whisper', detail: 'faster-whisper 1.2.1', device: 'cpu', modelCached: true }
  assert.equal(localWhisperBlock(ready, 'small'), null)
  assert.equal(localWhisperBlock({ ...ready, modelCached: null }, 'small'), null, 'unknown is let through')
  assert.match(localWhisperBlock({ ...ready, modelCached: false }, 'medium.en'), /“medium\.en” is not downloaded yet\. Install it in Settings/)
  const missing = { available: false, backend: null, detail: 'Whisper is not installed for BridgeClip’s Python.', device: null, modelCached: null }
  assert.match(localWhisperBlock(missing, 'small'), /^Whisper is not installed for BridgeClip’s Python\. Open Settings/)
})

test('the language list is offered by code, with automatic detection first', () => {
  assert.equal(w.LOCAL_WHISPER_LANGUAGES[0].code, 'auto')
  assert.ok(w.LOCAL_WHISPER_LANGUAGE_CODES.includes('en'))
  assert.ok(w.LOCAL_WHISPER_LANGUAGE_CODES.every((code) => code === 'auto' || /^[a-z]{2,3}$/.test(code)))
  assert.equal(new Set(w.LOCAL_WHISPER_LANGUAGE_CODES).size, w.LOCAL_WHISPER_LANGUAGE_CODES.length)
})

test('no job starts while Enable GPU rewrites the libraries a job loads; a download does not block', () => {
  assert.equal(whisperSetupBlock('gpu'), w.WHISPER_SETUP_MESSAGES.gpuRunning)
  assert.equal(whisperSetupBlock('download'), null)
  assert.equal(whisperSetupBlock(null), null)
})

test('older model names that Settings cannot install are held back instead of downloading silently', () => {
  const ready = { available: true, backend: 'faster-whisper', detail: 'faster-whisper 1.2.1', device: 'cpu', modelCached: null }
  for (const name of ['large', 'large-v1', 'large-v2', 'large-v3-turbo']) {
    assert.match(localWhisperBlock(ready, name), /Choose a Whisper model from the list/, name)
  }
})

test('the end-of-task message names the model, and a stop is not shown as a failure', () => {
  assert.deepEqual(w.setupResultMessage({ task: 'download', model: 'small' }, { ok: true }), { ok: true, text: 'small is installed.' })
  assert.match(w.setupResultMessage({ task: 'gpu', model: 'medium' }, { ok: true }).text, /^The GPU works with medium\./)
  assert.deepEqual(w.setupResultMessage({ task: 'download', model: 'small' }, { ok: false, cancelled: true, message: 'x' }), { ok: true, text: 'Download stopped.' })
  assert.deepEqual(w.setupResultMessage({ task: 'gpu', model: 'small' }, { ok: false, cancelled: false, message: 'No NVIDIA GPU with CUDA was found.' }),
    { ok: false, text: 'No NVIDIA GPU with CUDA was found.' })
})

test('the GPU libraries are pinned to the versions that were tested', async () => {
  const { GPU_PACKAGES } = setup()
  assert.ok(GPU_PACKAGES.every((spec) => /^nvidia-[a-z0-9-]+==[0-9.]+$/.test(spec)), GPU_PACKAGES.join(' '))
})

test('transcribing for titles and captions runs the engine script once for every file and is not a cancellable setup task', async () => {
  const s = setup()
  const { spawn, calls } = fakeSpawn([{ lines: [
    json({ event: 'text', index: 1, text: 'second part' }),
    json({ event: 'text', index: 0, text: 'x'.repeat(10_000) }),
    json({ event: 'done' })
  ] }])
  const texts = await s.transcribeWithLocalWhisper(['a.wav', 'b.wav'], { model: 'small', device: 'gpu', language: 'auto' }, deps(spawn))
  assert.deepEqual(texts, ['x'.repeat(10_000), 'second part'], 'long lines and out-of-order events are fine')
  assert.deepEqual(calls[0].args.slice(1), ['transcribe', 'small', 'gpu', 'auto', 'a.wav', 'b.wav'])
  assert.equal(s.whisperSetupRunning(), false)
  assert.equal(s.cancelWhisperSetup(), false, 'Cancel in the models panel cannot stop it')
})

test('transcription failures are plain messages that name Local Whisper', async () => {
  for (const [reason, pattern] of [['missing_backend', /not installed/], ['transcribe_model', /could not be loaded/], ['gpu_failed', /could not use the GPU/], ['weird', /could not transcribe/]]) {
    const s = setup()
    const { spawn } = fakeSpawn([{ lines: [json({ event: 'error', reason, message: 'C:\\secret' })], code: 1 }])
    await assert.rejects(s.transcribeWithLocalWhisper(['a.wav'], { model: 'small', device: 'cpu', language: '' }, deps(spawn)), (error) => {
      assert.match(error.message, /^(Local Whisper|The Whisper model|Choose a Whisper)/)
      assert.match(error.message, pattern, reason)
      assert.doesNotMatch(error.message, /secret/)
      return true
    })
  }
})

test('Whisper for captions has a time limit, and the app can stop it on quit', async () => {
  let s = setup()
  let fake = fakeSpawn([{ lines: [], hold: true }])
  await assert.rejects(
    s.transcribeWithLocalWhisper(['a.wav'], { model: 'small', device: 'cpu', language: 'auto' }, { ...deps(fake.spawn), timeoutMs: 20 }),
    /took longer than 30 minutes/
  )
  assert.equal(fake.calls[0].child.killed, true)
  s = setup()
  fake = fakeSpawn([{ lines: [], hold: true }])
  const running = s.transcribeWithLocalWhisper(['a.wav'], { model: 'small', device: 'cpu', language: 'auto' }, deps(fake.spawn))
  await new Promise((resolve) => setImmediate(resolve))
  s.stopLocalTranscriptions()
  await assert.rejects(running, /Local Whisper could not transcribe/)
  assert.equal(fake.calls[0].child.killed, true)
})

test('a model that is not downloaded is reported, never fetched silently for captions', async () => {
  const s = setup()
  const { spawn } = fakeSpawn([{ lines: [json({ event: 'error', reason: 'transcribe_not_installed' })], code: 1 }])
  await assert.rejects(s.transcribeWithLocalWhisper(['a.wav'], { model: 'turbo', device: 'cpu', language: 'auto' }, deps(spawn)), /not installed yet\. Install it in Settings/)
})
