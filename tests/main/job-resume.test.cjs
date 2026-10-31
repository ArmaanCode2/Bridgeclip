'use strict'
// Continuing a stopped job from its saved transcript: what main checks, skips and passes on.
const assert = require('node:assert/strict')
const { test } = require('node:test')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')
const { loadMain } = require('../zernio/support/load-main.cjs')

const gates = loadMain("export * from './src/main/job-gates'")

function loadJobStart(mocks) {
  const source = fs.readFileSync(path.join(__dirname, '../../src/main/job-start.ts'), 'utf8')
  const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
  const module = { exports: {} }
  vm.runInNewContext(code, { module, exports: module.exports, require: (id) => mocks[id] ?? require(id), Date, JSON, Error })
  return module.exports
}

const READY = {
  aiProvider: 'gemini', openrouterApiKey: '', geminiApiKey: 'AIza', openaiCompatibleUrl: '', openaiCompatibleModel: '',
  openaiCompatibleVision: 'off', transcriptionSource: 'cloud', localWhisperMode: 'builtin', localWhisperUrl: '',
  localWhisperModel: 'medium.en', outputDirectory: '/clips', pythonPath: 'python'
}
const REQUEST = {
  videoUrl: '/videos/talk.mp4', maxClips: null, autoClipCount: true, durationRanges: null, aspectRatio: '9:16',
  layoutStyle: 'auto', layoutVision: true, pacing: 'tight', includeCaptions: true, captionPreset: 'pop',
  startTimeSeconds: null, endTimeSeconds: null, bannerPlatform: null, bannerChannelUrl: null
}

/** job-start with every outside effect recorded; `resumable` stands in for the job list. */
function setup(settingsPatch = {}, resumable = {}, claude = { state: 'connected', plan: 'max' }) {
  const calls = { enqueued: [], validated: [], whisperChecks: 0, advanced: [] }
  const settings = { ...READY, ...settingsPatch }
  const mod = loadJobStart({
    electron: { app: { isPackaged: false } },
    './settings-store': { loadSettings: () => settings },
    './file-manager': { ensureOutputDir: () => {} },
    './pipeline-runner': {
      getBridgeRunnerPath: () => '/bridge.py', getEnginePath: () => '/engine', resolvePythonPath: () => '/python',
      runtimeEnvironment: () => ({}), preflightCheck: () => ({ ok: true }), validatePython: async () => ({ ok: true })
    },
    './provider-checks': { checkLocalWhisper: async () => { calls.whisperChecks++; return { available: true, modelCached: false } } },
    './assistant/providers': { checkProvider: async () => { calls.claudeChecks = (calls.claudeChecks ?? 0) + 1; return claude } },
    './run-history': { createRunRecord: () => {}, finishRunRecord: () => {} },
    './job-manager': {
      enqueueJob: (jobId, config, outputDirectory, request) => {
        calls.enqueued.push({ config, request })
        return { id: jobId, status: 'pending' }
      },
      resumableJob: (jobId) => resumable[jobId] ?? null
    },
    './logger': { logger: { info() {}, warn() {}, error() {} } },
    './security': { assertMediaPath: () => {}, isWebUrl: () => false },
    './network-policy': { assertPublicWebUrl: async () => {} },
    // Only known fields continue, as in the real validator.
    './validation': { validateJobConfig: (value, options) => { calls.validated.push(options); return { ...value } } },
    './job-gates': gates,
    './whisper-setup': { whisperSetupState: () => ({ task: 'gpu' }) },
    './openrouter-models': { resolveAdvancedModels: async (planner, transcription) => { calls.advanced.push([planner, transcription]); return { maxOutputTokens: 32000 } } },
    './tools': { supportsCaptionFilter: async () => true }
  })
  return { mod, calls }
}

test('a fresh start still needs everything transcription needs', async () => {
  const { mod } = setup()
  assert.match((await mod.startClipJobRequest(REQUEST)).error, /Cloud transcription uses OpenRouter/)
  const local = setup({ transcriptionSource: 'local' }, {})
  // Enable GPU is running, so no job may start; then the model is missing.
  assert.ok((await local.mod.startClipJobRequest(REQUEST)).error)
})

test('continuing skips the transcription checks but keeps the submitted options for later', async () => {
  for (const patch of [{}, { transcriptionSource: 'local' }, { transcriptionSource: 'local', localWhisperMode: 'server' }]) {
    const { mod, calls } = setup(patch)
    const result = await mod.startClipJobRequest(REQUEST, { resumeTranscript: '/clips/earlier/transcript.json' })
    assert.equal(result.error, undefined, JSON.stringify(patch))
    assert.equal(calls.whisperChecks, 0)
    assert.equal(calls.validated[0].transcriptionOptional, true)
    const [{ config, request }] = calls.enqueued
    assert.equal(config.resumeTranscript, '/clips/earlier/transcript.json')
    assert.equal(request.resumeTranscript, undefined, 'Run again on the new job transcribes again')
  }
})

test('the planner still has to be ready when continuing', async () => {
  const { mod, calls } = setup({ geminiApiKey: '' })
  assert.match((await mod.startClipJobRequest(REQUEST, { resumeTranscript: '/t/transcript.json' })).error, /Gemini/)
  assert.equal(calls.enqueued.length, 0)
})

test('an Advanced job continues as Quality after a switch away from OpenRouter, and Run again still sees Advanced', async () => {
  const advanced = { ...REQUEST, clippingMode: 'advanced', plannerModel: 'vendor/planner', transcriptionModel: 'openai/whisper-large-v3' }
  const { mod, calls } = setup({ aiProvider: 'openai-compatible', openaiCompatibleUrl: 'http://localhost:11434/v1', openaiCompatibleModel: 'qwen3' })
  assert.equal((await mod.startClipJobRequest(advanced, { resumeTranscript: '/t/transcript.json' })).error, undefined)
  const [{ config, request }] = calls.enqueued
  assert.equal(config.clippingMode, 'quality')
  assert.equal(config.plannerModel, undefined)
  assert.equal(request.clippingMode, 'advanced')
  assert.equal(request.transcriptionModel, 'openai/whisper-large-v3')
  assert.deepEqual(calls.advanced, [])
})

test('an Advanced job on OpenRouter continues with its planner and no transcription model', async () => {
  const advanced = { ...REQUEST, clippingMode: 'advanced', plannerModel: 'vendor/planner', transcriptionModel: 'openai/whisper-large-v3' }
  const { mod, calls } = setup({ aiProvider: 'openrouter', openrouterApiKey: 'sk-or' })
  assert.equal((await mod.startClipJobRequest(advanced, { resumeTranscript: '/t/transcript.json' })).error, undefined)
  assert.deepEqual(calls.advanced.map((call) => [...call]), [['vendor/planner', undefined]])
  const [{ config, request }] = calls.enqueued
  assert.equal(config.transcriptionModel, undefined)
  assert.equal(request.transcriptionModel, 'openai/whisper-large-v3', 'the submitted request is not changed')
})

test('only a job with a saved transcript can continue, and it continues from that file', async () => {
  const { mod, calls } = setup({}, { stopped: { request: REQUEST, transcript: '/clips/stopped/transcript.json' } })
  for (const id of ['unknown', 42, undefined]) {
    assert.match((await mod.resumeClipJobRequest(id)).error, /no saved transcript/)
  }
  assert.equal((await mod.resumeClipJobRequest('stopped')).error, undefined)
  assert.equal(calls.enqueued[0].config.resumeTranscript, '/clips/stopped/transcript.json')
})

test('a transcript path from the renderer never reaches a job', () => {
  const { validateJobConfig } = loadMain("export { validateJobConfig } from './src/main/validation'", { electron: {} })
  const config = validateJobConfig({ ...REQUEST, resumeTranscript: '/home/someone/secret.json' })
  assert.equal(config.resumeTranscript, undefined)
})

test('a job on the Claude plan stops before it starts when Claude Code cannot use the plan, and nothing is queued', async () => {
  const claude = { aiProvider: 'claude-plan', claudePlanVision: 'off', transcriptionSource: 'local' }
  for (const [status, words] of [
    [{ state: 'not-installed', plan: null }, /not installed/],
    [{ state: 'signed-out', plan: null }, /not signed in/],
    [{ state: 'connected', plan: null }, /API key/],
    [{ state: 'unknown', plan: null }, /did not report/]
  ]) {
    const { mod, calls } = setup({ ...claude, transcriptionSource: 'cloud', openrouterApiKey: 'sk-or' }, {}, status)
    assert.match((await mod.startClipJobRequest(REQUEST)).error, words)
    assert.deepEqual(calls.enqueued, [])
  }
})

test('only a job on the Claude plan checks Claude Code', async () => {
  const { mod, calls } = setup({ openrouterApiKey: 'sk-or' })
  await mod.startClipJobRequest(REQUEST)
  assert.equal(calls.claudeChecks, undefined)
})

test('a failed job continues on the Claude plan from its saved transcript, after a provider switch', async () => {
  const { mod, calls } = setup({ aiProvider: 'claude-plan', claudePlanVision: 'off' }, { failed: { request: { ...REQUEST, clippingMode: 'advanced', plannerModel: 'x/y' }, transcript: '/runs/transcript.json' } })
  const started = await mod.resumeClipJobRequest('failed')
  assert.equal(started.error, undefined)
  assert.equal(calls.claudeChecks, 1)
  assert.equal(calls.whisperChecks, 0)
  const { config } = calls.enqueued[0]
  assert.equal(config.resumeTranscript, '/runs/transcript.json')
  assert.equal(config.clippingMode, 'quality')
  assert.equal(config.plannerModel, undefined)
  // Frames stay off until the image switch is on.
  assert.equal(config.layoutVision, false)
})
