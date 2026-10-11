'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { loadMain, tempDir, fakeElectron } = require('../zernio/support/load-main.cjs')

const ROOT = path.join(__dirname, '../..')

function loadTools(t) {
  const { dir, cleanup } = tempDir('bridgeclip-assistant-tools-')
  t.after(cleanup)
  const { electron } = fakeElectron(dir)
  const host = { changed: [], shown: [] }
  const mod = loadMain("export * from './src/main/assistant/bridgeclip-tools'; export { validateJobConfig } from './src/main/validation'", { electron: { ...electron, dialog: {} } })
  const tools = mod.createBridgeClipTools({
    getMainWindow: () => null,
    dataChanged: (scope) => host.changed.push(scope),
    navigate: (page, runDir) => host.shown.push([page, runDir])
  })
  return { mod, tools, host, dir }
}

test('every BridgeClip tool has a strict schema, and anything that publishes, deletes or spends asks first', (t) => {
  const { tools } = loadTools(t)
  const names = tools.map((tool) => tool.name)
  assert.equal(new Set(names).size, names.length, 'names are unique')
  for (const tool of tools) {
    assert.match(tool.name, /^[a-z_]{3,64}$/, tool.name)
    assert.equal(tool.inputSchema.type, 'object', tool.name)
    assert.equal(tool.inputSchema.additionalProperties, false, tool.name)
    for (const key of tool.inputSchema.required ?? []) assert.ok(tool.inputSchema.properties[key], `${tool.name}.${key}`)
    assert.ok(tool.description.length > 20, tool.name)
    if (tool.destructive) assert.equal(typeof tool.confirm, 'function', `${tool.name} is destructive and must ask`)
    if (tool.readOnly) assert.equal(tool.confirm, undefined, `${tool.name} is read-only`)
  }
  for (const name of ['start_clip_job', 'post_clip', 'run_automation_now', 'delete_library_run', 'delete_clips', 'delete_automation', 'cancel_job', 'cancel_scheduled_post', 'retry_post', 'reschedule_post', 'update_settings', 'update_automation', 'remove_automation_clip']) {
    assert.equal(typeof tools.find((tool) => tool.name === name)?.confirm, 'function', `${name} asks first`)
  }
  // Never exposed: keys, updates, TikTok consent, account sign-in.
  for (const forbidden of ['replace_api_key', 'install_update', 'approve_tiktok_review', 'connect_account']) assert.ok(!names.includes(forbidden))
})

test('start_clip_job builds a request the Create page’s validator accepts, with its defaults', (t) => {
  const { mod } = loadTools(t)
  const minimal = mod.clipJobRequestFromInput({ source: ' https://www.youtube.com/watch?v=hqP9fivmBqI ' })
  assert.deepEqual(minimal, {
    videoUrl: 'https://www.youtube.com/watch?v=hqP9fivmBqI',
    workflow: 'automatic',
    clippingMode: 'quality',
    maxClips: null,
    autoClipCount: true,
    durationRanges: ['short'],
    aspectRatio: '9:16',
    layoutStyle: 'auto',
    layoutVision: true,
    pacing: 'tight',
    videoSpeed: 1,
    includeCaptions: true,
    captionPreset: 'pop',
    includeTitle: true,
    startTimeSeconds: null,
    endTimeSeconds: null,
    bannerPlatform: null,
    bannerChannelUrl: null
  })
  assert.doesNotThrow(() => mod.validateJobConfig(minimal))
  const custom = mod.clipJobRequestFromInput({
    source: 'https://www.twitch.tv/videos/123', workflow: 'review', mode: 'economy', aspectRatio: '16:9', captions: false, captionStyle: 'neon',
    titleCard: false, durations: ['xshort', 'medium'], maxClips: 4, speed: 1.25, pacing: 'natural', layout: 'fit', startSeconds: 30, endSeconds: 900, clipRequest: '  the pricing debate '
  })
  assert.equal(custom.layoutVision, false, 'economy never pays for vision')
  assert.equal(custom.autoClipCount, false)
  assert.equal(custom.clipRequest, 'the pricing debate')
  const validated = mod.validateJobConfig(custom)
  assert.equal(validated.maxClips, 4)
  assert.equal(validated.captionPreset, 'neon')
})

test('job validation drops fields it doesn’t know, so a typo can’t ride along in the job record', (t) => {
  const { mod } = loadTools(t)
  const request = { ...mod.clipJobRequestFromInput({ source: 'https://example.com/video.mp4' }), clip_request: 'typo', speed: 2, plannerCapabilities: { maxOutputTokens: 1 } }
  const validated = mod.validateJobConfig(request)
  assert.equal('clip_request' in validated, false)
  assert.equal('speed' in validated, false)
  assert.equal(validated.plannerCapabilities, undefined)
  assert.equal(validated.videoSpeed, 1)
})

test('tools refuse run ids that could leave the Library', async (t) => {
  const { tools } = loadTools(t)
  const getRun = tools.find((tool) => tool.name === 'get_library_run')
  for (const runId of ['../secrets', '/etc', 'C:\\Windows', '11111111-1111-4111-8111-11111111111']) {
    await assert.rejects(getRun.run({ runId }, { conversationId: 'c', signal: new AbortController().signal }), /run id/)
  }
})

test('get_clip_options lists exactly the caption styles the engine and the picker know', (t) => {
  const { tools } = loadTools(t)
  return tools.find((tool) => tool.name === 'get_clip_options').run({}, {}).then((options) => {
    const ids = options.captionStyles.map((style) => style.id)
    const engine = fs.readFileSync(path.join(ROOT, 'engine/clip_engine/config.py'), 'utf8')
    const engineIds = [...engine.slice(engine.indexOf('class CaptionPreset:'), engine.indexOf('DEFAULT_CAPTION_PRESET')).matchAll(/^\s+[A-Z_]+ = "([a-z_-]+)"/gm)].map((match) => match[1])
    const picker = fs.readFileSync(path.join(ROOT, 'src/renderer/components/CaptionPresetPicker.tsx'), 'utf8')
    const pickerIds = [...picker.matchAll(/^\s{4}id: '([a-z_-]+)',$/gm)].map((match) => match[1])
    assert.deepEqual(ids, engineIds)
    assert.deepEqual(ids, pickerIds)
  })
})

test('show_in_bridgeclip asks the window to open a page or a Library run', async (t) => {
  const { tools, host, dir } = loadTools(t)
  const show = tools.find((tool) => tool.name === 'show_in_bridgeclip')
  await show.run({ page: 'automations' }, {})
  await show.run({ page: 'library', runId: '11111111-1111-4111-8111-111111111111' }, {})
  assert.deepEqual(host.shown[0], ['automations', undefined])
  assert.equal(host.shown[1][0], 'library')
  assert.equal(path.basename(host.shown[1][1]), '11111111-1111-4111-8111-111111111111')
  assert.ok(host.shown[1][1].startsWith(dir), 'inside the Library folder')
})
