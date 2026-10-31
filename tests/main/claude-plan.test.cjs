'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { loadMain, tempDir, fakeElectron } = require('../zernio/support/load-main.cjs')
const { directoryLinkType } = require('../support/symlinks.cjs')

const { gateJobStart, claudePlanBlock } = loadMain("export * from './src/main/job-gates'")
const { AI_PROVIDERS, aiProviderChange, aiReady, setupGaps, providerReadsImages, testSubject } = loadMain("export * from './src/shared/providers'")
const { claudePlanArgs, claudePlanMessage, readClaudeRun, askClaudePlan } = loadMain("export * from './src/main/claude-plan/chat'")
const { SESSIONS_FILE, claudePlanRunner, claudePlanEnvironment, claudePlanWorkspace } = loadMain("export * from './src/main/claude-plan/runner'")
const { claudeProjectFolder, readSessionIds, removeClaudeSessions } = loadMain("export * from './src/main/claude-plan/sessions'")
const { PROVIDER_MESSAGE_PREFIX, ProviderError, chatJson, ownModelTarget } = loadMain("export * from './src/main/ai-chat'")
const { testAiProvider } = loadMain("export * from './src/main/provider-checks'")

function openStore() {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  const { electron } = fakeElectron(dir)
  const store = loadMain("export * from './src/main/settings-store'", { electron })
  const file = path.join(dir, 'userData', 'settings.json')
  return {
    store, file, dir, cleanup,
    raw: () => JSON.parse(fs.readFileSync(file, 'utf8')),
    writeRaw: (data) => { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, JSON.stringify(data)) }
  }
}

const SCHEMA = { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'], additionalProperties: false }

/** What the real program prints, captured from version 2.1.308. */
const init = { type: 'system', subtype: 'init', model: 'claude-sonnet-5-5', apiKeySource: 'none' }
const limit = (info = {}) => ({ type: 'rate_limit_event', rate_limit_info: { status: 'allowed', overageStatus: 'rejected', isUsingOverage: false, ...info } })
const result = (fields) => ({ type: 'result', subtype: 'success', is_error: false, stop_reason: 'tool_use', ...fields })
const answered = (fields) => [init, { type: 'assistant', message: {} }, limit(), result(fields)]
const failed = (text, code) => [init, { type: 'assistant', error: code, message: {} },
  { type: 'result', subtype: 'success', is_error: true, result: text, stop_reason: 'stop_sequence', terminal_reason: 'api_error' }]
const lines = (events) => events.map((event) => JSON.stringify(event)).join('\n')

// ---- Settings ----------------------------------------------------------------------

test('the Claude plan is a provider, and a new install still starts on OpenRouter with Sonnet at medium effort and images off', () => {
  assert.ok(AI_PROVIDERS.includes('claude-plan'))
  const t = openStore()
  try {
    const view = t.store.publicSettings(t.store.loadSettings())
    assert.equal(view.aiProvider, 'openrouter')
    assert.equal(view.claudePlanModel, 'sonnet')
    assert.equal(view.claudePlanEffort, 'medium')
    assert.equal(view.claudePlanVision, 'off')
  } finally { t.cleanup() }
})

test('the Claude plan choice, model, effort and image switch survive a restart', () => {
  const t = openStore()
  try {
    t.store.savePublicSettings({ aiProvider: 'claude-plan', claudePlanModel: 'opus', claudePlanEffort: 'max', claudePlanVision: 'on' })
    const again = loadMain("export * from './src/main/settings-store'", { electron: fakeElectron(t.dir).electron })
    const saved = again.loadSettings()
    assert.equal(saved.aiProvider, 'claude-plan')
    assert.equal(saved.claudePlanModel, 'opus')
    assert.equal(saved.claudePlanEffort, 'max')
    assert.equal(saved.claudePlanVision, 'on')
    assert.equal(t.raw().claudePlanEffort, 'max')
  } finally { t.cleanup() }
})

test('a model, effort or switch outside the lists is refused and nothing changes', () => {
  const t = openStore()
  try {
    assert.throws(() => t.store.savePublicSettings({ claudePlanModel: 'fable' }), /Claude plan model/)
    assert.throws(() => t.store.savePublicSettings({ claudePlanEffort: '--dangerously-skip-permissions' }), /Claude plan effort/)
    assert.throws(() => t.store.savePublicSettings({ claudePlanVision: 'maybe' }), /image reading/i)
    assert.equal(t.store.loadSettings().claudePlanModel, 'sonnet')
  } finally { t.cleanup() }
})

test('damaged saved values load as the defaults rather than stopping the app', () => {
  const t = openStore()
  try {
    t.writeRaw({ version: 14, aiProvider: 'claude-plan', claudePlanModel: 'gpt-5', claudePlanEffort: 7, claudePlanVision: 'sideways', localWhisperMode: 'builtin' })
    const saved = t.store.loadSettings()
    assert.equal(saved.aiProvider, 'claude-plan')
    assert.equal(saved.claudePlanModel, 'sonnet')
    assert.equal(saved.claudePlanEffort, 'medium')
    assert.equal(saved.claudePlanVision, 'off')
  } finally { t.cleanup() }
})

test('a version 13 file upgrades, keeping the saved provider and cards and adding the defaults', () => {
  const t = openStore()
  try {
    t.writeRaw({
      version: 13, aiProvider: 'openai-compatible', activeModelPreset: 'one', localWhisperMode: 'builtin', outputDirectory: path.join(t.dir, 'out'),
      modelPresets: [{ id: 'one', name: 'Groq', url: 'https://api.groq.com/openai/v1', model: 'llama', apiKey: '', vision: 'off', timeout: '30' }]
    })
    const saved = t.store.loadSettings()
    assert.equal(saved.aiProvider, 'openai-compatible')
    assert.equal(saved.activeModelPreset, 'one')
    assert.equal(saved.modelPresets.length, 1)
    assert.equal(saved.claudePlanModel, 'sonnet')
    assert.equal(t.raw().version, 14)
  } finally { t.cleanup() }
})

test('a job on the Claude plan gets the model, effort and image switch, and no key it does not need', () => {
  const t = openStore()
  try {
    t.store.replaceApiKey('openrouterApiKey', 'sk-or-secret')
    t.store.replaceApiKey('geminiApiKey', 'AIza-secret')
    t.store.savePublicSettings({ aiProvider: 'claude-plan', claudePlanModel: 'haiku', claudePlanEffort: 'xhigh', claudePlanVision: 'on', transcriptionSource: 'local' })
    const env = t.store.getSettingsForBridge(t.store.loadSettings())
    assert.equal(env.AI_PROVIDER, 'claude-plan')
    assert.equal(env.CLAUDE_PLAN_MODEL, 'haiku')
    assert.equal(env.CLAUDE_PLAN_EFFORT, 'xhigh')
    assert.equal(env.CLAUDE_PLAN_VISION, 'true')
    assert.equal(env.OPENROUTER_API_KEY, '')
    assert.equal(env.GEMINI_API_KEY, '')
    assert.equal(env.OPENAI_COMPATIBLE_API_KEY, '')
  } finally { t.cleanup() }
})

test('cloud transcription on the Claude plan still passes the OpenRouter key', () => {
  const t = openStore()
  try {
    t.store.replaceApiKey('openrouterApiKey', 'sk-or-secret')
    t.store.savePublicSettings({ aiProvider: 'claude-plan', transcriptionSource: 'cloud' })
    assert.equal(t.store.getSettingsForBridge(t.store.loadSettings()).OPENROUTER_API_KEY, 'sk-or-secret')
  } finally { t.cleanup() }
})

test('Read images alone decides whether frames reach the review on the Claude plan', () => {
  const t = openStore()
  try {
    const env = () => t.store.getSettingsForBridge(t.store.loadSettings()).JEV_VISUAL_CONTEXT
    t.store.savePublicSettings({ aiProvider: 'claude-plan', jevVisualContext: 'on', claudePlanVision: 'off' })
    assert.equal(env(), 'false')
    t.store.savePublicSettings({ jevVisualContext: 'off', claudePlanVision: 'on' })
    assert.equal(env(), 'true')
    // Every other provider keeps its own checkbox.
    t.store.savePublicSettings({ aiProvider: 'openrouter' })
    assert.equal(env(), 'false')
    t.store.savePublicSettings({ jevVisualContext: 'on' })
    assert.equal(env(), 'true')
  } finally { t.cleanup() }
})

test('a run wholly on the Claude plan shows no cost in its row; a real charge still shows', () => {
  const { shownRunCost } = loadMain("export * from './src/shared/job-output'")
  const planning = { provider: 'claude-plan' }
  assert.equal(shownRunCost({ total_estimated_cost_usd: 0, planning }), null)
  assert.equal(shownRunCost({ total_estimated_cost_usd: 0.12, planning }), 0.12)
  assert.equal(shownRunCost({ total_estimated_cost_usd: 0, planning: { provider: 'gemini' } }), 0)
  assert.equal(shownRunCost({ total_estimated_cost_usd: 0 }), 0)
  assert.equal(shownRunCost(undefined), null)
  assert.equal(shownRunCost({ total_estimated_cost_usd: 'free' }), null)
})

test('choosing the Claude plan with no OpenRouter key moves transcription to Local Whisper, as Gemini does', () => {
  assert.deepEqual(aiProviderChange('claude-plan', { openrouterConfigured: false, transcriptionSource: 'cloud' }),
    { aiProvider: 'claude-plan', transcriptionSource: 'local' })
  assert.deepEqual(aiProviderChange('claude-plan', { openrouterConfigured: true, transcriptionSource: 'cloud' }), { aiProvider: 'claude-plan' })
})

test('which providers read images', () => {
  const reads = (aiProvider, openaiCompatibleVision, claudePlanVision) => providerReadsImages({ aiProvider, openaiCompatibleVision, claudePlanVision })
  assert.equal(reads('openrouter', 'off', 'off'), true)
  assert.equal(reads('gemini', 'off', 'off'), true)
  assert.equal(reads('openai-compatible', 'off', 'on'), false)
  assert.equal(reads('openai-compatible', 'on', 'off'), true)
  assert.equal(reads('claude-plan', 'on', 'off'), false)
  assert.equal(reads('claude-plan', 'off', 'on'), true)
})

test('a Test connection result belongs to one model, effort and image setting', () => {
  const base = { openaiCompatibleUrl: '', openaiCompatibleModel: '', geminiModel: '', claudePlanVision: 'off' }
  const subject = testSubject('claude-plan', { ...base, claudePlanModel: 'opus', claudePlanEffort: 'low' })
  assert.notEqual(subject, testSubject('claude-plan', { ...base, claudePlanModel: 'opus', claudePlanEffort: 'low', claudePlanVision: 'on' }))
  assert.notEqual(subject, testSubject('claude-plan', { ...base, claudePlanModel: 'sonnet', claudePlanEffort: 'low' }))
  assert.notEqual(subject, testSubject('claude-plan', { ...base, claudePlanModel: 'opus', claudePlanEffort: 'max' }))
})

// ---- Starting a job ----------------------------------------------------------------

const CLAUDE = {
  aiProvider: 'claude-plan', openrouterApiKey: '', geminiApiKey: '', openaiCompatibleUrl: '', openaiCompatibleModel: '',
  openaiCompatibleVision: 'off', claudePlanVision: 'off', transcriptionSource: 'local', localWhisperMode: 'builtin', localWhisperUrl: ''
}
const input = { aiProvider: 'claude-plan', openrouterConfigured: false, geminiConfigured: false, openaiCompatibleUrl: '', openaiCompatibleModel: '',
  transcriptionSource: 'local', localWhisperMode: 'builtin', localWhisperUrl: '' }

test('the Claude plan needs no key to start a job', () => {
  assert.equal(aiReady(input), true)
  assert.deepEqual(setupGaps(input), [])
  assert.equal(gateJobStart(CLAUDE, {}).ok, true)
})

test('cloud transcription on the Claude plan still needs an OpenRouter key', () => {
  const gaps = setupGaps({ ...input, transcriptionSource: 'cloud' })
  assert.deepEqual(gaps.map((gap) => gap.id), ['openrouter-key'])
  assert.equal(gateJobStart({ ...CLAUDE, transcriptionSource: 'cloud' }, {}).ok, false)
})

test('continuing from a saved transcript needs nothing for transcription', () => {
  assert.equal(gateJobStart({ ...CLAUDE, transcriptionSource: 'cloud' }, {}, { transcribes: false }).ok, true)
})

test('a job reads frames on the Claude plan only when the image switch is on', () => {
  assert.equal(gateJobStart(CLAUDE, { layoutVision: true }).layoutVision, false)
  assert.equal(gateJobStart({ ...CLAUDE, claudePlanVision: 'on' }, { layoutVision: true }).layoutVision, true)
})

test('Advanced needs OpenRouter, and Economy runs as Quality, as on every other own model', () => {
  assert.equal(gateJobStart(CLAUDE, { clippingMode: 'advanced' }).missing, 'advanced-needs-openrouter')
  assert.equal(gateJobStart(CLAUDE, { clippingMode: 'economy' }).clippingMode, 'quality')
})

test('a job stops before it starts when Claude Code cannot use the plan', () => {
  const status = (fields) => ({ id: 'claude', state: 'connected', version: '2.1.308', account: null, plan: 'max', detail: null, ...fields })
  assert.equal(claudePlanBlock(status({})), null)
  assert.equal(claudePlanBlock(status({ plan: 'pro' })), null)
  assert.match(claudePlanBlock(status({ state: 'not-installed' })), /not installed/)
  assert.match(claudePlanBlock(status({ state: 'signed-out' })), /not signed in/)
  // Only a sign-in that reports a plan passes: nothing billed some other way gets through.
  assert.match(claudePlanBlock(status({ plan: null })), /would not use your Claude plan/)
  assert.match(claudePlanBlock(status({ state: 'unknown', plan: null })), /did not report/)
})

// ---- How a run starts --------------------------------------------------------------

test('the environment a run gets holds no API key, whatever this process has', () => {
  const secrets = { ANTHROPIC_API_KEY: 'sk-ant', ANTHROPIC_AUTH_TOKEN: 'token', OPENROUTER_API_KEY: 'sk-or', CLAUDECODE: '1' }
  const saved = Object.fromEntries(Object.keys(secrets).map((key) => [key, process.env[key]]))
  Object.assign(process.env, secrets)
  try {
    const runner = claudePlanRunner({ command: 'claude.exe', prefixArgs: ['cli.js'], pathValue: 'C:\\bin' }, 'C:\\work')
    assert.deepEqual(runner.command, ['claude.exe', 'cli.js'])
    assert.equal(runner.cwd, 'C:\\work')
    assert.equal(runner.env.PATH, 'C:\\bin')
    for (const key of Object.keys(secrets)) assert.equal(key in runner.env, false, key)
  } finally {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key]
      else process.env[key] = value
    }
  }
})

test('only a job on the Claude plan is told how to start it', () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    const find = () => ({ command: 'claude.exe', prefixArgs: [], pathValue: '' })
    assert.deepEqual(claudePlanEnvironment('gemini', { find, root: dir }), {})
    const runner = JSON.parse(claudePlanEnvironment('claude-plan', { find, root: dir }).CLAUDE_PLAN_RUNNER)
    assert.deepEqual(runner.command, ['claude.exe'])
    assert.equal(runner.cwd, claudePlanWorkspace(dir))
    assert.deepEqual(claudePlanEnvironment('claude-plan', { find: () => null, root: dir }), {})
  } finally { cleanup() }
})

test('runs start in an empty folder of BridgeClip’s own', () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    const folder = claudePlanWorkspace(dir)
    assert.ok(folder.startsWith(dir))
    assert.deepEqual(fs.readdirSync(folder), [])
  } finally { cleanup() }
})

test('a job tells the engine where to list its runs', () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    const find = () => ({ command: 'claude.exe', prefixArgs: [], pathValue: '' })
    const run = path.join(dir, 'library', 'job-1')
    const runner = JSON.parse(claudePlanEnvironment('claude-plan', { find, root: dir, runFolder: run }).CLAUDE_PLAN_RUNNER)
    assert.equal(runner.sessions, path.join(run, SESSIONS_FILE))
    assert.equal('sessions' in JSON.parse(claudePlanEnvironment('claude-plan', { find, root: dir }).CLAUDE_PLAN_RUNNER), false)
  } finally { cleanup() }
})

// ---- Cleaning up after a job -------------------------------------------------------

const ONE = '11111111-2222-4333-8444-555555555555'
const TWO = '66666666-7777-4888-9999-000000000000'
const OTHER = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee'
const WORKSPACE = 'C:\\Users\\me\\AppData\\Roaming\\BridgeClip\\claude-plan\\workspace'

test('the folder Claude Code uses is the working folder with everything else turned into dashes', () => {
  // The name of this repository's own folder under ~/.claude/projects.
  assert.equal(path.basename(claudeProjectFolder('E:\\antigravity\\bridgeclip\\code', 'x')), 'E--antigravity-bridgeclip-code')
  assert.equal(claudeProjectFolder(WORKSPACE, 'config'), path.join('config', 'projects', 'C--Users-me-AppData-Roaming-BridgeClip-claude-plan-workspace'))
})

test('a job’s list gives only well-formed ids, once each', () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    assert.deepEqual(readSessionIds(dir), [], 'no list means no sessions')
    fs.writeFileSync(path.join(dir, SESSIONS_FILE), [ONE, ONE, '', '  ' + TWO + '  ', '../../secrets', '*', 'C:\\Windows', `${ONE}/..`, ONE.slice(1)].join('\r\n'))
    assert.deepEqual(readSessionIds(dir), [ONE, TWO])
    fs.writeFileSync(path.join(dir, SESSIONS_FILE), `${ONE}\n`.repeat(40000))
    assert.deepEqual(readSessionIds(dir), [], 'an oversized list is not read')
  } finally { cleanup() }
})

/** A fake ~/.claude with sessions of BridgeClip and of a terminal project. */
function claudeFolder(dir) {
  const config = path.join(dir, 'claude')
  const ours = claudeProjectFolder(WORKSPACE, config)
  const theirs = claudeProjectFolder('E:\\work\\project', config)
  fs.mkdirSync(path.join(ours, ONE, 'subagents'), { recursive: true })
  fs.mkdirSync(theirs, { recursive: true })
  for (const file of [`${ONE}.jsonl`, `${TWO}.jsonl`, `${OTHER}.jsonl`, path.join(ONE, 'subagents', 'a.jsonl')]) fs.writeFileSync(path.join(ours, file), 'x')
  fs.writeFileSync(path.join(theirs, `${ONE}.jsonl`), 'a terminal session')
  return { config, ours, theirs }
}

test('removing a job’s sessions takes exactly its ids, from BridgeClip’s folder alone', () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    const { config, ours, theirs } = claudeFolder(dir)
    assert.equal(removeClaudeSessions([ONE, TWO], WORKSPACE, config), 2)
    assert.deepEqual(fs.readdirSync(ours), [`${OTHER}.jsonl`], 'another job’s session stays')
    assert.equal(fs.readFileSync(path.join(theirs, `${ONE}.jsonl`), 'utf8'), 'a terminal session')
  } finally { cleanup() }
})

test('an id with nothing saved changes nothing, and Claude Code’s own folder stays even when emptied', () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    const { config, ours } = claudeFolder(dir)
    assert.equal(removeClaudeSessions([TWO.replace('6', '7')], WORKSPACE, config), 0)
    assert.equal(fs.readdirSync(ours).length, 4)
    assert.equal(removeClaudeSessions([ONE, TWO, OTHER], WORKSPACE, config), 3)
    assert.deepEqual(fs.readdirSync(ours), [])
  } finally { cleanup() }
})

test('a link where BridgeClip’s folder should be is left alone, so a tampered list cannot reach a terminal session', () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    const { config, ours, theirs } = claudeFolder(dir)
    fs.rmSync(ours, { recursive: true })
    fs.symlinkSync(theirs, ours, 'junction')
    assert.equal(removeClaudeSessions([ONE], WORKSPACE, config), 0)
    assert.equal(fs.readFileSync(path.join(theirs, `${ONE}.jsonl`), 'utf8'), 'a terminal session')
  } finally { cleanup() }
})

test('a link in place of a session is left alone, and what it points at is kept', { skip: !directoryLinkType && 'links are not available here' }, () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    const { config, ours } = claudeFolder(dir)
    const elsewhere = path.join(dir, 'elsewhere')
    fs.mkdirSync(elsewhere)
    fs.writeFileSync(path.join(elsewhere, 'keep.txt'), 'keep')
    fs.rmSync(path.join(ours, ONE), { recursive: true })
    fs.symlinkSync(elsewhere, path.join(ours, ONE), directoryLinkType)
    fs.rmSync(path.join(ours, `${ONE}.jsonl`))
    assert.equal(removeClaudeSessions([ONE], WORKSPACE, config), 0)
    assert.equal(fs.readFileSync(path.join(elsewhere, 'keep.txt'), 'utf8'), 'keep')
  } finally { cleanup() }
})

test('a Claude folder out of reach costs one look, however long the list is', () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    const config = path.join(dir, 'a-file')
    fs.writeFileSync(config, 'x')
    const ids = Array.from({ length: 5000 }, (_, index) => `${String(index).padStart(8, '0')}-2222-4333-8444-555555555555`)
    const started = process.hrtime.bigint()
    assert.equal(removeClaudeSessions(ids, WORKSPACE, config), 0)
    assert.ok(Number(process.hrtime.bigint() - started) / 1e6 < 50)
  } finally { cleanup() }
})

test('a missing Claude folder, or an id that is not one, removes nothing and does not throw', () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    assert.equal(removeClaudeSessions([ONE], WORKSPACE, path.join(dir, 'nothing')), 0)
    const { config, ours } = claudeFolder(dir)
    assert.equal(removeClaudeSessions(['..', '../..', '*', ''], WORKSPACE, config), 0)
    assert.equal(fs.readdirSync(ours).length, 4)
  } finally { cleanup() }
})

// ---- Asking from the main process --------------------------------------------------

const question = { system: 'Write titles.', prompt: 'A transcript.', schema: SCHEMA, model: 'sonnet', effort: 'high', timeoutMs: 20000 }
const after = (args, flag) => args[args.indexOf(flag) + 1]

test('a question runs Claude Code locked down, with the chosen model and effort', () => {
  const args = claudePlanArgs(question)
  assert.equal(args[0], '-p')
  assert.equal(after(args, '--model'), 'sonnet')
  assert.equal(after(args, '--effort'), 'high')
  assert.equal(after(args, '--tools'), '')
  assert.equal(after(args, '--system-prompt'), 'Write titles.')
  // Not even the person's own Claude Code settings: one could name a key or remap a model.
  assert.equal(after(args, '--setting-sources'), '')
  assert.equal(after(args, '--permission-mode'), 'dontAsk')
  assert.deepEqual(JSON.parse(after(args, '--settings')), { disableAllHooks: true })
  assert.deepEqual(JSON.parse(after(args, '--json-schema')), SCHEMA)
  for (const flag of ['--strict-mcp-config', '--disable-slash-commands', '--no-session-persistence', '--verbose']) assert.ok(args.includes(flag), flag)
  assert.notEqual(after(claudePlanArgs({ ...question, system: '' }), '--system-prompt'), '')
})

test('a run’s answer is read from its structured output, or its text', () => {
  assert.deepEqual(readClaudeRun(lines(answered({ structured_output: { answer: 'yes' }, result: 'ignored' })), ''),
    { ok: true, content: '{"answer":"yes"}' })
  assert.deepEqual(readClaudeRun(`Update available\n${lines(answered({ result: '{"answer":"text"}' }))}`, ''),
    { ok: true, content: '{"answer":"text"}' })
})

test('every failure becomes one fixed sentence starting with the provider, never Claude Code’s own words', () => {
  const cases = [
    [lines(failed('Not logged in · Please run /login', 'authentication_failed')), '', /not signed in/],
    [lines(failed('You’ve hit your session limit · resets 3:45pm', 'rate_limit')), '', /usage limit/],
    [lines(failed('You have reached the weekly limit', undefined)), '', /usage limit/],
    [lines(failed('There’s an issue with the selected model (x).', 'model_not_found')), '[claude-code:unrecognized_model] {}', /cannot use this model/],
    ['', 'error: unknown option \'--no-session-persistence\'', /newer Claude Code/],
    ['', 'Something else went wrong with secret-value', /could not answer/],
    [lines(answered({})), '', /could not answer/],
    [lines(failed('Prompt is too long', 'invalid_request')), '', /larger model/],
    [lines([init, limit({ status: 'rejected' }), result({ is_error: true, result: 'Something new' })]), '', /usage limit/],
    [lines([init, limit({ isUsingOverage: true }), result({ structured_output: { answer: 'yes' } })]), '', /usage limit/],
    [lines([{ ...init, apiKeySource: 'ANTHROPIC_API_KEY' }, result({ structured_output: { answer: 'yes' } })]), '', /not being used/]
  ]
  for (const [stdout, stderr, expected] of cases) {
    const answer = readClaudeRun(stdout, stderr)
    assert.equal(answer.ok, false)
    assert.match(answer.message, expected)
    assert.match(answer.message, PROVIDER_MESSAGE_PREFIX)
    assert.doesNotMatch(answer.message, /secret-value|login|3:45pm/)
  }
})

/** A small program standing in for `claude`: records what it was given, then prints `stdout`. */
function fakeProgram(dir, { stdout, exit = 0, wait = 0 }) {
  const record = path.join(dir, 'record.json')
  const script = path.join(dir, 'fake-claude.js')
  fs.writeFileSync(script, `
    const fs = require('fs')
    let stdin = ''
    process.stdin.on('data', (chunk) => { stdin += chunk })
    process.stdin.on('end', () => setTimeout(() => {
      fs.writeFileSync(${JSON.stringify(record)}, JSON.stringify({ argv: process.argv.slice(2), cwd: process.cwd(), env: process.env, stdin }))
      process.stdout.write(${JSON.stringify(stdout)} + '\\n')
      process.exit(${exit})
    }, ${wait}))
  `)
  return { cli: { command: process.execPath, prefixArgs: [script], pathValue: process.env.PATH }, record: () => JSON.parse(fs.readFileSync(record, 'utf8')) }
}

test('a question goes in on stdin from the empty folder, and the answer comes back', async () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    const fake = fakeProgram(dir, { stdout: lines(answered({ structured_output: { answer: 'yes' } })) })
    const workspace = claudePlanWorkspace(dir)
    const answer = await askClaudePlan(question, { cli: fake.cli, cwd: workspace })
    assert.deepEqual(answer, { ok: true, content: '{"answer":"yes"}' })
    const record = fake.record()
    assert.equal(fs.realpathSync(record.cwd), fs.realpathSync(workspace))
    const message = JSON.parse(record.stdin)
    assert.equal(message.type, 'user')
    assert.deepEqual(message.message, { role: 'user', content: [{ type: 'text', text: 'A transcript.' }] })
  } finally { cleanup() }
})

test('a run that fails says why', async () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    const fake = fakeProgram(dir, { stdout: lines(failed('Not logged in · Please run /login', 'authentication_failed')), exit: 1 })
    const answer = await askClaudePlan(question, { cli: fake.cli, cwd: dir })
    assert.equal(answer.ok, false)
    assert.match(answer.message, /not signed in/)
  } finally { cleanup() }
})

test('a run that takes too long is stopped', async () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    const fake = fakeProgram(dir, { stdout: lines(answered({ structured_output: { answer: 'late' } })), wait: 20000 })
    const started = Date.now()
    const answer = await askClaudePlan({ ...question, timeoutMs: 400 }, { cli: fake.cli, cwd: dir })
    assert.equal(answer.ok, false)
    assert.match(answer.message, /took too long/)
    assert.ok(Date.now() - started < 10000)
  } finally { cleanup() }
})

test('no Claude Code on this computer, or one that will not start, says so', async () => {
  const { dir, cleanup } = tempDir('bridgeclip-claude-plan-')
  try {
    const missing = await askClaudePlan(question, { cli: null, cwd: dir })
    assert.match(missing.message, /not installed/)
    const broken = await askClaudePlan(question, { cli: { command: path.join(dir, 'gone.exe'), prefixArgs: [], pathValue: '' }, cwd: dir })
    assert.match(broken.message, /not installed/)
  } finally { cleanup() }
})

// ---- Metadata writing --------------------------------------------------------------

const own = { aiProvider: 'claude-plan', claudePlanModel: 'opus', geminiApiKey: '', geminiModel: '', openaiCompatibleUrl: '', openaiCompatibleApiKey: '', openaiCompatibleModel: '', openaiCompatibleTimeout: '30' }

test('metadata is written by the Claude plan when it is the provider', async () => {
  const target = ownModelTarget(own)
  assert.equal(target.provider, 'claude-plan')
  assert.equal(target.model, 'opus')
  assert.ok(target.timeoutMs > 0 && target.timeoutMs <= 600000)
  let asked = null
  const reply = await chatJson(target, {
    messages: [{ role: 'system', content: 'Write a title.' }, { role: 'user', content: 'The transcript.' }], schema: SCHEMA, schemaName: 'metadata', maxTokens: 500
  }, { askClaudePlan: async (request) => { asked = request; return { ok: true, content: '```json\n{"answer":"A title"}\n```' } } })
  assert.deepEqual(reply, { answer: 'A title' })
  assert.equal(asked.system, 'Write a title.')
  assert.equal(asked.prompt, 'The transcript.')
  assert.equal(asked.model, 'opus')
  // Titles are quick work: the effort chosen for picking clips could outlast the wait.
  assert.equal(asked.effort, 'medium')
})

test('a failed metadata request is reported like any other provider’s', async () => {
  const request = { messages: [{ role: 'user', content: 'x' }], schema: SCHEMA, schemaName: 'metadata', maxTokens: 500 }
  await assert.rejects(chatJson(ownModelTarget(own), request, { askClaudePlan: async () => ({ ok: false, message: 'Your Claude plan reached its usage limit. Try again after it resets.' }) }),
    (error) => error instanceof ProviderError && PROVIDER_MESSAGE_PREFIX.test(error.message))
})

// ---- Test connection ---------------------------------------------------------------

const connected = { id: 'claude', state: 'connected', version: '2.1.308', account: 'me@example.com', plan: 'max', detail: null, checkedAt: '' }
const saved = { claudePlanModel: 'sonnet', claudePlanEffort: 'max', openaiCompatibleUrl: '', openaiCompatibleModel: '', geminiModel: '' }
const states = (result) => Object.fromEntries(result.checks.map((check) => [check.id, check.state]))

test('Test connection checks Claude Code, the sign-in and the model, at low effort to spare the plan', async () => {
  let asked = null
  const result = await testAiProvider('claude-plan', saved, {
    claudeStatus: async () => connected,
    askClaudePlan: async (request) => { asked = request; return { ok: true, content: '{"ok":true,"word":"clip"}' } }
  })
  assert.equal(result.ok, true)
  assert.deepEqual(states(result), { server: 'ok', key: 'ok', model: 'ok', json: 'ok', images: 'skipped' })
  assert.equal(asked.model, 'sonnet')
  assert.equal(asked.effort, 'low')
  assert.equal(result.subject, testSubject('claude-plan', saved))
  assert.equal(JSON.stringify(result).includes('me@example.com'), false)
})

test('Test connection stops at the first thing that is missing, before using the plan', async () => {
  const never = async () => { throw new Error('the plan must not be used') }
  const missing = await testAiProvider('claude-plan', saved, { claudeStatus: async () => ({ ...connected, state: 'not-installed' }), askClaudePlan: never })
  assert.equal(missing.ok, false)
  assert.deepEqual(states(missing), { server: 'failed' })
  const out = await testAiProvider('claude-plan', saved, { claudeStatus: async () => ({ ...connected, state: 'signed-out' }), askClaudePlan: never })
  assert.deepEqual(states(out), { server: 'ok', key: 'failed' })
  assert.match(out.message, /not signed in/)
  const billed = await testAiProvider('claude-plan', saved, { claudeStatus: async () => ({ ...connected, plan: null, detail: 'Signed in with an API key, so usage is billed to that key rather than a subscription.' }), askClaudePlan: never })
  assert.deepEqual(states(billed), { server: 'ok', key: 'failed' })
  assert.match(billed.message, /API key/)
  const unknown = await testAiProvider('claude-plan', saved, { claudeStatus: async () => ({ ...connected, state: 'unknown', plan: null }), askClaudePlan: never })
  assert.deepEqual(states(unknown), { server: 'ok', key: 'failed' })
})

test('Test connection shows why the model did not answer', async () => {
  const result = await testAiProvider('claude-plan', saved, {
    claudeStatus: async () => connected,
    askClaudePlan: async () => ({ ok: false, message: 'Your Claude plan reached its usage limit. Try again after it resets.' })
  })
  assert.equal(result.ok, false)
  assert.deepEqual(states(result), { server: 'ok', key: 'ok', model: 'failed' })
  assert.match(result.message, /usage limit/)
})

test('a question with a picture hands it over as an image block', () => {
  const message = JSON.parse(claudePlanMessage({ ...question, image: 'data:image/png;base64,QUJD' }))
  assert.deepEqual(message.message.content, [
    { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'QUJD' } },
    { type: 'text', text: 'A transcript.' }
  ])
})

/** Answers the JSON question, then names the two halves of whatever picture it was shown. */
const seeing = (names) => async (request) => request.image
  ? { ok: true, content: JSON.stringify(names(request.image)) }
  : { ok: true, content: '{"ok":true,"word":"clip"}' }

test('with Read images on, Test connection shows the plan a two-colour picture and checks the answer', async () => {
  const asked = []
  const random = () => 0   // Red on the left, green on the right.
  const reads = await testAiProvider('claude-plan', { ...saved, claudePlanVision: 'on' }, {
    claudeStatus: async () => connected, random,
    askClaudePlan: async (request) => { asked.push(request); return seeing(() => ({ left: 'red', right: 'green' }))(request) }
  })
  assert.equal(reads.ok, true)
  assert.equal(reads.readsImages, true)
  assert.equal(states(reads).images, 'ok')
  assert.match(reads.message, /can read images/)
  assert.equal(asked.length, 2)
  assert.match(asked[1].image, /^data:image\/png;base64,/)
  assert.equal(asked[1].effort, 'low')

  const blind = await testAiProvider('claude-plan', { ...saved, claudePlanVision: 'on' }, {
    claudeStatus: async () => connected, random, askClaudePlan: seeing(() => ({ left: 'blue', right: 'blue' }))
  })
  assert.equal(blind.readsImages, false)
  assert.equal(states(blind).images, 'failed')
  assert.equal(blind.ok, false, 'a job would send frames this model cannot read')
  assert.match(blind.message, /could not read the test image/)

  const cutOff = await testAiProvider('claude-plan', { ...saved, claudePlanVision: 'on' }, {
    claudeStatus: async () => connected,
    askClaudePlan: async (request) => request.image
      ? { ok: false, message: 'Your Claude plan reached its usage limit. Try again after it resets.' }
      : { ok: true, content: '{"ok":true,"word":"clip"}' }
  })
  assert.equal(cutOff.ok, false)
  assert.equal(cutOff.readsImages, null)
  assert.match(cutOff.message, /usage limit/)
})

test('with Read images off, Test connection spends nothing on a picture', async () => {
  const asked = []
  const result = await testAiProvider('claude-plan', { ...saved, claudePlanVision: 'off' }, {
    claudeStatus: async () => connected, askClaudePlan: async (request) => { asked.push(request); return { ok: true, content: '{"ok":true,"word":"clip"}' } }
  })
  assert.equal(asked.length, 1)
  assert.equal(result.ok, true)
  assert.equal(states(result).images, 'skipped')
})

test('Test connection fails a model that answers without JSON', async () => {
  const result = await testAiProvider('claude-plan', saved, { claudeStatus: async () => connected, askClaudePlan: async () => ({ ok: true, content: 'Sure!' }) })
  assert.equal(result.ok, false)
  assert.deepEqual(states(result), { server: 'ok', key: 'ok', model: 'ok', json: 'failed' })
})
