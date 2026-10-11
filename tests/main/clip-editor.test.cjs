const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { loadMain, tempDir, fakeElectron } = require('../zernio/support/load-main.cjs')
const fixture = require('../fixtures/editor/project.json')
const schema = loadMain("export * from './src/shared/clip-editor'")
const clone = () => structuredClone(fixture)

test('editor project validates cuts, geometry and candidate identity; strips extra authority', () => {
  const p = clone(); p.apiKey = 'secret'; p.candidates[0].sourcePath = '/private.mp4'
  const parsed = schema.parseEditorProject(p)
  assert.equal(parsed.candidates[0].review.questions.length, 8)
  assert.ok(!JSON.stringify(parsed).includes('secret'))
  assert.ok(!JSON.stringify(parsed).includes('/private'))
  for (const modify of [
    p => { p.version = 2 }, p => { p.width = Infinity }, p => { p.candidates[0].ranges[0][1] = NaN },
    p => { p.candidates[0].ranges[1][0] = 4000 }, p => { p.candidates[0].scenes[0].crops[0][0] = .9 },
    p => { p.candidates[0].scenes[0].at_ms = 1 }, p => { p.candidates[1].id = p.candidates[0].id },
    p => { p.candidates[0].captions = 'yes' }, p => { p.candidates[0].review.questions[0].probability = 1.5 }
  ]) { const p = clone(); modify(p); assert.throws(() => schema.parseEditorProject(p)) }
})

test('long display text with emoji is clamped by UTF-16 length instead of rejecting the project', () => {
  const p = clone()
  p.candidates[0].title = 'a'.repeat(199) + '😀'
  p.candidates[0].reason = 'r'.repeat(3999) + '👍🏽'
  p.title = 't'.repeat(1030)
  p.transcript[0].text = 'w'.repeat(19999) + '😀'
  const parsed = schema.parseEditorProject(p)
  assert.equal(parsed.candidates[0].title, 'a'.repeat(199))
  assert.equal(parsed.candidates[0].reason, 'r'.repeat(3999))
  assert.equal(parsed.title.length, 1024)
  assert.equal(parsed.transcript[0].text, 'w'.repeat(19999))
  assert.equal(schema.clampText('ab😀c', 4), 'ab😀')
  for (const bad of [null, 5, ['a']]) assert.throws(() => schema.clampText(bad, 10))
  const blank = clone(); blank.candidates[0].title = ' '.repeat(300)
  assert.throws(() => schema.parseEditorProject(blank))
})

test('editor progress prioritizes unfinished candidates, then baked ones, over discards', () => {
  for (const [statuses, remaining, initialCandidate] of [
    [['discarded', 'refining', 'ready'], 2, 1],
    [['baked', 'discarded', 'ready', 'refining'], 2, 2],
    [['discarded', 'baked'], 0, 1],
    [['discarded', 'discarded'], 0, 0],
    [[undefined, 'discarded'], 1, 0]
  ]) assert.deepEqual(schema.editorProgress(statuses.map(status => ({ status }))), { remaining, initialCandidate })
})

test('edit signature invalidates Jev after trims, titles or framing, and preserves it for caption settings', () => {
  const c = clone().candidates[0]
  assert.equal(JSON.stringify(JSON.parse(c.review.signature)), schema.editSignature(c))
  const key = schema.editSignature(c)
  c.captions = false; c.video_speed = 1.5
  assert.equal(schema.editSignature(c), key)
  c.ranges[0][0] = 1200
  assert.notEqual(schema.editSignature(c), key)
  assert.equal(schema.sceneAt(c, 9000).layout, 'split')
  assert.ok(schema.defaultCrop(1920, 1080, 9 / 16, 0)[0] === 0)
  assert.equal(schema.editDuration(c), 6800 / 1.5)
})

test('smooth movement preserves legacy projects and eases crops continuously through interrupted layouts', () => {
  const c = clone().candidates[0]
  const original = schema.editSignature(c)
  c.scenes[0].transition_ms = 0
  assert.equal(schema.editSignature(c), original)
  c.scenes = [
    { at_ms: 0, layout: 'fill', crops: [[0, 0, .4, 1]] },
    { at_ms: 2000, layout: 'fill', crops: [[.6, .5, .2, .5]], transition_ms: 1000 },
    { at_ms: 2500, layout: 'fill', crops: [[0, 0, .4, 1]], transition_ms: 1000 }
  ]
  assert.deepEqual(schema.framingAt(c, 2000).crops[0], [0, 0, .4, 1])
  assert.deepEqual(schema.framingAt(c, 2500).crops[0], [.3, .25, .30000000000000004, .75])
  assert.ok(Math.abs(schema.framingAt(c, 3000).crops[0][0] - .15) < 1e-9)
  assert.deepEqual(schema.framingAt(c, 3500).crops[0], [0, 0, .4, 1])
  assert.equal(schema.parseCandidateEdit(c, 12000).scenes[1].transition_ms, 1000)
  const animatedKey = schema.editSignature(c)
  c.scenes[1].transition_ms = 600
  assert.notEqual(schema.editSignature(c), animatedKey)
  for (const duration of [-1, 99, 5001, Infinity, NaN, '600', 600.5, null]) {
    c.scenes[1].transition_ms = duration
    assert.throws(() => schema.parseCandidateEdit(c, 12000))
  }
  c.scenes[1].transition_ms = 600; c.scenes[0].layout = 'fit'
  assert.throws(() => schema.parseCandidateEdit(c, 12000))
  c.scenes = schema.normalizeSceneTransitions(c.scenes)
  assert.equal(c.scenes[1].transition_ms, undefined)
  assert.equal(schema.parseCandidateEdit(c, 12000).scenes[2].transition_ms, 1000)
})

test('timeline edges extend to source limits without crossing other cuts or collapsing a cut', () => {
  const cuts = [[10000, 15000], [18000, 20000]]
  assert.deepEqual(schema.trimRange(cuts, 0, 0, -500, 60000), [[0, 15000], [18000, 20000]])
  assert.deepEqual(schema.trimRange(cuts, 1, 1, 70000, 60000), [[10000, 15000], [18000, 60000]])
  assert.deepEqual(schema.trimRange(cuts, 0, 1, 19000, 60000), [[10000, 18000], [18000, 20000]])
  assert.deepEqual(schema.trimRange(cuts, 1, 0, 0, 60000), [[10000, 15000], [15000, 20000]])
  assert.deepEqual(schema.trimRange(cuts, 0, 1, 0, 60000), [[10000, 10100], [18000, 20000]])
  assert.deepEqual(cuts, [[10000, 15000], [18000, 20000]])
})

test('layout start times move between neighbours without changing crops, motion or the initial layout', () => {
  const c = schema.parseEditorProject(clone()).candidates[0]
  const scenes = [c.scenes[0], { ...c.scenes[0], at_ms: 3000, transition_ms: 700 }, { ...c.scenes[1], at_ms: 8000 }]
  const before = structuredClone(scenes)
  for (const [time, expected] of [[4000.4, 4000], [2500, 2500], [-1000, 1], [9000, 7999]]) {
    const moved = schema.retimeScene(scenes, 1, time, 12000)
    assert.equal(moved[1].at_ms, expected)
    assert.deepEqual({ ...moved[1], at_ms: 3000 }, scenes[1])
    assert.equal(moved[0], scenes[0]); assert.equal(moved[2], scenes[2])
    assert.doesNotThrow(() => schema.parseCandidateEdit({ ...c, scenes: moved }, 12000))
  }
  assert.equal(schema.retimeScene(scenes, 2, 20000, 12000)[2].at_ms, 11999)
  assert.equal(schema.retimeScene(scenes, 2, 0, 12000)[2].at_ms, 3001)
  for (const [index, time] of [[0, 5000], [-1, 2000], [3, 2000], [1, NaN], [1, Infinity], [1, 3000]]) {
    assert.equal(schema.retimeScene(scenes, index, time, 12000), scenes)
  }
  const adjacent = [scenes[0], { ...scenes[1], at_ms: 1 }, { ...scenes[2], at_ms: 2 }]
  assert.equal(schema.retimeScene(adjacent, 1, 1000, 12000), adjacent)
  assert.deepEqual(scenes, before)
  const approved = { ...c, status: 'ready', scenes }
  const moved = schema.refineEdit(approved, { scenes: schema.retimeScene(scenes, 1, 4000, 12000) })
  assert.equal(moved.status, 'refining')
  assert.notEqual(schema.editSignature(moved), schema.editSignature(approved))
})

test('each crop corner resizes proportionally while anchoring its opposite corner', () => {
  const original = [.25, .25, .2, .4], width = 640, height = 360
  const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`)
  for (const corner of ['top-left', 'top-right', 'bottom-left', 'bottom-right']) {
    const sx = corner.endsWith('right') ? 1 : -1, sy = corner.startsWith('bottom') ? 1 : -1
    for (const scale of [.75, 1.4]) {
      const next = schema.resizeCrop(original, corner, sx * original[2] * width * (scale - 1), sy * original[3] * height * (scale - 1), width, height)
      close(next[2], original[2] * scale); close(next[3], original[3] * scale)
      close(next[0] + (sx < 0 ? next[2] : 0), original[0] + (sx < 0 ? original[2] : 0))
      close(next[1] + (sy < 0 ? next[3] : 0), original[1] + (sy < 0 ? original[3] : 0))
    }
    // Vertical-only movement also resizes without stretching the output.
    const vertical = schema.resizeCrop(original, corner, 0, sy * -20, width, height)
    assert.ok(vertical[2] < original[2]); close(vertical[2] / vertical[3], original[2] / original[3])
  }
  assert.deepEqual(original, [.25, .25, .2, .4])
})

test('corner drags stop at source edges and zoom limits without flipping or producing invalid crops', () => {
  for (const [width, height] of [[1920, 1080], [1080, 1920], [1280, 720]]) for (const aspect of [9 / 16, 9 / 8, 16 / 9]) {
    for (const corner of ['top-left', 'top-right', 'bottom-left', 'bottom-right']) {
      const original = schema.defaultCrop(width, height, aspect, .4, .6, 2)
      const sx = corner.endsWith('right') ? 1 : -1, sy = corner.startsWith('bottom') ? 1 : -1
      for (const distance of [-100000, -1, 0, 1, 100000]) {
        const next = schema.resizeCrop(original, corner, sx * distance, sy * distance, 640, 360)
        assert.ok(Math.abs(next[2] / next[3] - original[2] / original[3]) < 1e-9)
        assert.ok(next[0] >= 0 && next[1] >= 0 && next[0] + next[2] <= 1 && next[1] + next[3] <= 1)
        const full = schema.defaultCrop(width, height, aspect)
        assert.ok(next[2] >= full[2] / 4 - 1e-10 && next[3] >= full[3] / 4 - 1e-10)
        const c = schema.parseEditorProject(clone()).candidates[0]
        c.scenes = [{ at_ms: 0, layout: 'fill', crops: [next] }]
        assert.doesNotThrow(() => schema.parseCandidateEdit(c, 12000, 4))
      }
    }
  }
  const legacy = [.2, .2, .02, .04]
  assert.deepEqual(schema.resizeCrop(legacy, 'bottom-right', -10000, -10000, 640, 360), legacy)
  assert.deepEqual(schema.resizeCrop(legacy, 'top-left', 100, 100, 0, 360), legacy)
})

test('legacy candidates start refining; caption corrections are bounded and refer to unique source lines', () => {
  const parsed = schema.parseEditorProject(clone())
  assert.equal(parsed.candidates[0].status, 'refining')
  assert.deepEqual(parsed.candidates[0].caption_edits, [])
  const c = parsed.candidates[0]
  c.caption_edits = [{ segment: 2, text: '' }, { segment: 0, text: 'A corrected line.\nCafé 👋' }]
  assert.deepEqual(schema.parseCandidateEdit(c, 12000, 4).caption_edits.map(e => e.segment), [0, 2])
  for (const patch of [
    { status: 'published' }, { status: null }, { caption_edits: null },
    { caption_edits: [{ segment: 4, text: 'outside source' }] },
    { caption_edits: [{ segment: .5, text: 'fractional' }] },
    { caption_edits: [{ segment: 0, text: 'a' }, { segment: 0, text: 'b' }] },
    { caption_edits: [{ segment: 0, text: 'a'.repeat(2001) }] },
    { caption_edits: [{ segment: 0, text: 'bad\u0000text' }] },
    { caption_edits: Array.from({ length: 2001 }, (_, segment) => ({ segment, text: '' })) }
  ]) assert.throws(() => schema.parseCandidateEdit({ ...c, ...patch }, 12000, 4))
})

test('content changes return approved candidates to refining; status changes preserve source review', () => {
  const c = schema.parseEditorProject(clone()).candidates[0]
  const patches = [{ title: 'Changed title' }, { ranges: [[1000, 4000]] }, { scenes: [c.scenes[0]] },
    { captions: false }, { caption_preset: 'minimal' }, { video_speed: 1.5 }, { caption_edits: [{ segment: 1, text: 'Correction' }] }]
  for (const status of ['ready', 'baked']) for (const patch of patches) {
    const next = schema.refineEdit({ ...c, status }, patch)
    assert.equal(next.status, 'refining')
    assert.deepEqual(next.review, c.review)
  }
  assert.equal(schema.refineEdit({ ...c, status: 'ready' }, { title: c.title }).status, 'ready')
  assert.equal(schema.refineEdit(c, { status: 'ready' }).status, 'ready')
  assert.equal(schema.refineEdit(c, { status: 'discarded' }).status, 'discarded')
  assert.equal(schema.editSignature(schema.refineEdit(c, patches.at(-1))), schema.editSignature(c))
})

function setup(overrides = {}) {
  const temp = tempDir('bridgeclip-editor-')
  const library = path.join(temp.dir, 'library'), run = path.join(library, 'review-run')
  fs.mkdirSync(run, { recursive: true })
  for (const file of ['editor-source.mp4', 'editor-preview.mp4']) fs.writeFileSync(path.join(run, file), 'video')
  fs.writeFileSync(path.join(run, 'editor-project.json'), JSON.stringify(fixture))
  fs.writeFileSync(path.join(run, 'job_output.json'), JSON.stringify({ job_id: 'review-run', clips: [], editor_project: true }))
  const mocks = { electron: fakeElectron(temp.dir).electron, ...overrides }
  const source = "export * from './src/main/clip-editor'; export * as settings from './src/main/settings-store'; export { authorizeMedia } from './src/main/security'"
  const main = loadMain(source, mocks)
  main.settings.savePublicSettings({ outputDirectory: library, pythonPath: 'python3' })
  return { ...temp, run, library, main, reload: () => loadMain(source, mocks) }
}

test('editor saves only editable fields, survives reload and refuses stale revisions', async () => {
  const f = setup()
  try {
    const opened = await f.main.openEditor(f.run)
    const edits = opened.project.candidates
    edits[0].title = 'A deliberate title'
    edits[0].status = 'ready'
    edits[0].caption_edits = [{ segment: 1, text: 'The correction happened.' }]
    edits[0].scenes.splice(1, 0, { ...structuredClone(edits[0].scenes[0]), at_ms: 3000, transition_ms: 600 })
    edits[0].review = null; edits[0].exports = [999]
    const saved = await f.main.saveEditor(f.run, 0, edits)
    assert.equal(saved.project.revision, 1)
    assert.equal(saved.project.candidates[0].title, 'A deliberate title')
    assert.ok(saved.project.candidates[0].review)
    assert.deepEqual(saved.project.candidates[0].exports, [])
    assert.equal(saved.project.candidates[0].status, 'ready')
    assert.deepEqual(saved.project.candidates[0].caption_edits, edits[0].caption_edits)
    assert.deepEqual(saved.project.transcript, fixture.transcript)
    assert.deepEqual(saved.project.candidates[1].caption_edits, [])
    assert.equal((await f.reload().openEditor(f.run)).project.candidates[0].title, 'A deliberate title')
    assert.equal((await f.reload().openEditor(f.run)).project.candidates[0].scenes[1].transition_ms, 600)
    await assert.rejects(f.main.saveEditor(f.run, 0, edits), /changed/)
    await assert.rejects(f.main.runEditor(f.run, 1, 'unknown', 'export'), /changed/)
  } finally { f.cleanup() }
})

test('only ready candidates render and only a completed render can mark a candidate baked', async () => {
  const f = setup()
  try {
    const project = (await f.main.openEditor(f.run)).project
    await assert.rejects(f.main.runEditor(f.run, 0, 'candidate-1', 'export'), /Mark this clip ready/)
    project.candidates[0].status = 'baked'
    await assert.rejects(f.main.saveEditor(f.run, 0, project.candidates), /completed render/)
    project.candidates[0].status = 'discarded'
    await f.main.saveEditor(f.run, 0, project.candidates)
    await assert.rejects(f.main.runEditor(f.run, 1, 'candidate-1', 'export'), /Mark this clip ready/)
    // A genuine previous render can survive unchanged saves, never edited content.
    project.revision = 2; project.candidates[0].status = 'baked'; project.candidates[0].exports = [0]
    fs.writeFileSync(path.join(f.run, 'editor-project.json'), JSON.stringify(project))
    await f.main.saveEditor(f.run, 2, project.candidates)
    project.candidates[0].caption_edits = [{ segment: 1, text: 'New words' }]
    await assert.rejects(f.main.saveEditor(f.run, 3, project.candidates), /completed render/)
    project.candidates[0].status = 'refining'
    const saved = await f.main.saveEditor(f.run, 3, project.candidates)
    assert.equal(saved.project.candidates[0].status, 'refining')
    assert.deepEqual(saved.project.candidates[0].exports, [0])
  } finally { f.cleanup() }
})

test('editor rejects external roots and symlinked source/project files', async () => {
  const f = setup()
  try {
    await assert.rejects(f.main.openEditor(f.dir))
    await assert.rejects(f.main.openEditor(f.library))
    const target = path.join(f.dir, 'external.json'); fs.writeFileSync(target, JSON.stringify(fixture))
    const project = path.join(f.run, 'editor-project.json'); fs.unlinkSync(project); fs.symlinkSync(target, project)
    await assert.rejects(f.main.openEditor(f.run))
    fs.unlinkSync(project); fs.writeFileSync(project, JSON.stringify(fixture))
    const source = path.join(f.run, 'editor-source.mp4'); fs.unlinkSync(source); fs.symlinkSync(target, source)
    await assert.rejects(f.main.openEditor(f.run))
    assert.equal(fs.readFileSync(target, 'utf8'), JSON.stringify(fixture))
  } finally { f.cleanup() }
})

test('editor review requires Jev even with automatic review off and locks the project', async () => {
  const { EventEmitter } = require('node:events')
  const { PassThrough } = require('node:stream')
  let release, entered, config, workerEnv
  const started = new Promise((r) => { entered = r })
  const f = setup({ child_process: { ...require('node:child_process'), spawn: (_cmd, _args, options) => {
    workerEnv = options.env
    const child = new EventEmitter()
    child.stdout = new PassThrough(); child.stderr = new PassThrough(); child.stdin = new PassThrough()
    child.stdin.end = (data) => {
      config = JSON.parse(data); entered()
      release = () => { child.stdout.write('{"ok":true}'); child.emit('close', 0) }
    }
    return child
  } } })
  try {
    f.main.settings.replaceApiKey('openrouterApiKey', 'fixture-key')
    f.main.settings.savePublicSettings({ outputDirectory: f.library, pythonPath: 'python3', jevEnabled: 'off' })
    const pending = f.main.runEditor(f.run, 0, 'candidate-1', 'review')
    await started
    assert.equal(config.action, 'review')
    assert.equal(workerEnv.JEV_ENABLED, 'true')
    assert.equal(f.main.settings.loadSettings().jevEnabled, 'off')
    assert.equal(config.library, fs.realpathSync(f.library))
    assert.equal((await f.main.openEditor(f.run)).operation, 'review')
    await assert.rejects(f.main.saveEditor(f.run, 0, fixture.candidates), /Wait/)
    await assert.rejects(f.main.runEditor(f.run, 0, 'candidate-1', 'export'), /already running/)
    release(); await pending
    assert.equal((await f.main.openEditor(f.run)).operation, null)
    f.main.settings.savePublicSettings({ outputDirectory: f.library, pythonPath: 'python3', jevEnabled: 'off' })
    f.main.settings.replaceApiKey('openrouterApiKey', '')
    await assert.rejects(f.main.runEditor(f.run, 0, 'candidate-1', 'review'), /Reviewing a clip uses OpenRouter\. Finish setting it up/)
  } finally { f.cleanup() }
})

test('caption suppression validates source intervals and changes render state without changing review evidence', () => {
  const c = schema.parseEditorProject(clone()).candidates[0]
  assert.deepEqual(c.caption_suppression_ranges, [])
  const ranges = [[0, 2000], [4000, 7000], [11000, 12000]]
  const next = schema.refineEdit({ ...c, status: 'ready' }, { caption_suppression_ranges: ranges })
  assert.equal(next.status, 'refining')
  assert.equal(schema.editSignature(next), schema.editSignature(c))
  assert.deepEqual(schema.parseCandidateEdit(next, 12000).caption_suppression_ranges, ranges)
  for (const value of [null, 'bad', [[0]], [[0, 100, 200]], [[-1, 1000]], [[0, 12001]], [[0, 99]],
    [[1000, 1000]], [[2000, 3000], [1000, 2000]], [[0, 2000], [1000, 3000]], [[NaN, 1000]],
    [[0, Infinity]], [[true, 1000]], Array.from({ length: 201 }, (_, i) => [i * 100, i * 100 + 100])]) {
    assert.throws(() => schema.parseCandidateEdit({ ...c, caption_suppression_ranges: value }, 12000))
  }
})

test('caption suppression persists per candidate through save and reopen without modifying text or cuts', async () => {
  const f = setup()
  try {
    const opened = await f.main.openEditor(f.run)
    const c = opened.project.candidates[0]
    c.caption_suppression_ranges = [[2000, 4000], [6000, 9000]]
    await f.main.saveEditor(f.run, 0, opened.project.candidates)
    const saved = (await f.reload().openEditor(f.run)).project
    assert.deepEqual(saved.candidates[0].caption_suppression_ranges, c.caption_suppression_ranges)
    assert.deepEqual(saved.candidates[1].caption_suppression_ranges, [])
    assert.deepEqual(saved.candidates[0].ranges, c.ranges)
    assert.deepEqual(saved.transcript, opened.project.transcript)
    assert.deepEqual(saved.candidates[0].review, c.review)
  } finally { f.cleanup() }
})


test('older main-process sessions and empty caption suppression have the same render identity', () => {
  const c = schema.parseEditorProject(clone()).candidates[0]
  const legacy = { ...c }
  delete legacy.caption_suppression_ranges
  assert.deepEqual(schema.candidateEdit(legacy).caption_suppression_ranges, [])
  assert.equal(schema.renderEditKey(legacy), schema.renderEditKey(c))
  assert.equal(schema.refineEdit({ ...legacy, status: 'ready' }, { caption_suppression_ranges: [] }).status, 'ready')
  assert.equal(schema.refineEdit({ ...legacy, status: 'ready' }, { caption_suppression_ranges: [[1000, 2000]] }).status, 'refining')
})


test('additional caption-free sections use available retained footage without replacing previous ranges', () => {
  const { nextCaptionRange } = loadMain("export * from './src/renderer/lib/caption-ranges'")
  const cuts = [[2000, 5000], [7000, 10000]]
  const ranges = [[3000, 4500]]
  assert.deepEqual(nextCaptionRange(ranges, cuts, 3500), [4500, 5000])
  assert.deepEqual(nextCaptionRange(ranges, cuts, 5500), [7000, 9000])
  assert.deepEqual(nextCaptionRange(ranges, cuts, 11000), [2000, 3000])
  assert.deepEqual(nextCaptionRange([[0, 6000]], cuts, 3500), [7000, 9000])
  assert.deepEqual(nextCaptionRange([[3000, 4500], [4500, 5000]], cuts, 4500), [7000, 9000])
  assert.equal(nextCaptionRange([[0, 12000]], cuts, 3500), null)
  assert.equal(nextCaptionRange([[2000, 4950], [7000, 10000]], cuts, 4000), null)
  assert.deepEqual(ranges, [[3000, 4500]])
})

function batchSetup() {
  const { EventEmitter } = require('node:events')
  const { PassThrough } = require('node:stream')
  const workers = []
  const f = setup({ child_process: { ...require('node:child_process'), spawn: () => {
    const child = new EventEmitter()
    child.stdout = new PassThrough(); child.stderr = new PassThrough(); child.stdin = new PassThrough()
    child.stdin.end = (data) => {
      const config = JSON.parse(data)
      workers.push({ config, write: (data) => child.stdout.write(data), close: (code) => child.emit('close', code), finish(ok = true) {
        if (ok) {
          const file = path.join(f.run, 'editor-project.json')
          const project = JSON.parse(fs.readFileSync(file))
          assert.equal(config.revision, project.revision)
          const candidate = project.candidates.find(c => c.id === config.candidate_id)
          assert.equal(candidate.status, 'ready')
          candidate.status = 'baked'; candidate.exports.push(project.revision)
          project.revision++
          fs.writeFileSync(file, JSON.stringify(project))
        }
        child.stdout.write(JSON.stringify({ ok })); child.emit('close', ok ? 0 : 1)
      } })
    }
    return child
  } } })
  const project = schema.parseEditorProject(clone())
  project.candidates = ['refining', 'ready', 'baked', 'discarded', 'ready'].map((status, i) => ({
    ...structuredClone(project.candidates[0]), id: `candidate-${i}`, status,
    captions: i !== 4, caption_suppression_ranges: i === 1 ? [[2000, 3000], [7000, 9000]] : []
  }))
  fs.writeFileSync(path.join(f.run, 'editor-project.json'), JSON.stringify(project))
  return { ...f, workers }
}
const nextWorker = async (workers, count) => {
  const deadline = Date.now() + 5000
  while (workers.length < count && Date.now() < deadline) await new Promise(r => setTimeout(r, 5))
  assert.equal(workers.length, count)
  return workers[count - 1]
}

test('batch bakes only ready clips sequentially, keeps caption settings, revisions and project lock', async () => {
  const f = batchSetup()
  try {
    const pending = f.main.runEditor(f.run, 0, 'candidate-0', 'export-all')
    const first = await nextWorker(f.workers, 1)
    assert.equal(first.config.candidate_id, 'candidate-1')
    assert.equal(first.config.action, 'export')
    assert.deepEqual((await f.main.openEditor(f.run)).batch, { completed: 0, total: 2 })
    await assert.rejects(f.main.saveEditor(f.run, 0, []), /Wait/)
    await assert.rejects(f.main.runEditor(f.run, 0, '', 'export-all'), /already running/)
    first.finish()
    const second = await nextWorker(f.workers, 2)
    assert.equal(second.config.candidate_id, 'candidate-4')
    assert.equal(second.config.revision, 1)
    const reopened = await f.main.openEditor(f.run)
    assert.equal(reopened.operation, 'export-all')
    assert.deepEqual(reopened.batch, { completed: 1, total: 2 })
    assert.equal(reopened.project.candidates[4].captions, false)
    assert.deepEqual(reopened.project.candidates[1].caption_suppression_ranges, [[2000, 3000], [7000, 9000]])
    second.finish()
    const done = await pending
    assert.equal(done.operation, null)
    assert.equal(done.project.revision, 2)
    assert.deepEqual(done.project.candidates.map(c => c.status), ['refining', 'baked', 'baked', 'discarded', 'baked'])
    await assert.rejects(f.main.runEditor(f.run, 2, '', 'export-all'), /at least one clip ready/)
    await assert.rejects(f.main.runEditor(f.run, 0, '', 'export-all'), /changed/)
    assert.equal(f.workers.length, 2)
  } finally { f.cleanup() }
})

for (const cancelled of [false, true]) test(`batch ${cancelled ? 'cancellation' : 'failure'} preserves completed exports and leaves unfinished clips ready`, async () => {
  const f = batchSetup()
  try {
    const pending = f.main.runEditor(f.run, 0, '', 'export-all')
    const rejected = assert.rejects(pending, cancelled ? /Baked 1 of 2.*cancelled/ : /Baked 1 of 2 ready clips\. Could not bake “[^”]+”: Export stopped or failed.*still ready/)
    ;(await nextWorker(f.workers, 1)).finish()
    const second = await nextWorker(f.workers, 2)
    if (cancelled) f.main.cancelEditor(f.run)
    second.finish(false)
    await rejected
    const reopened = await f.main.openEditor(f.run)
    assert.equal(reopened.operation, null)
    assert.equal(reopened.project.candidates[1].status, 'baked')
    assert.equal(reopened.project.candidates[4].status, 'ready')
    const retry = f.main.runEditor(f.run, 1, '', 'export-all')
    const remaining = await nextWorker(f.workers, 3)
    assert.equal(remaining.config.candidate_id, 'candidate-4')
    remaining.finish()
    await retry
  } finally { f.cleanup() }
})

test('cancelling between batch clips never starts the next worker', async () => {
  const f = batchSetup()
  try {
    const pending = f.main.runEditor(f.run, 0, '', 'export-all')
    const rejected = assert.rejects(pending, /Baked 1 of 2.*cancelled/)
    ;(await nextWorker(f.workers, 1)).finish()
    f.main.cancelEditor(f.run)
    await rejected
    assert.equal(f.workers.length, 1)
    assert.equal((await f.main.openEditor(f.run)).project.candidates[4].status, 'ready')
  } finally { f.cleanup() }
})

test('source generations cannot supply arbitrary media paths', () => {
  for (const source_id of ['../private', '/outside', 'a'.repeat(31), 123, null]) {
    assert.throws(() => schema.parseEditorProject({ ...clone(), source_id }))
  }
  assert.equal(schema.parseEditorProject({ ...clone(), source_id: 'a'.repeat(32) }).source_id, 'a'.repeat(32))
})

for (const outcome of ['success', 'failure', 'cancelled', 'committed-before-exit']) test(`source replacement ${outcome} preserves the correct media pair and edits`, async () => {
  const { EventEmitter } = require('node:events')
  const { PassThrough } = require('node:stream')
  let complete, config, started
  const entered = new Promise(r => { started = r })
  const f = setup({ child_process: { ...require('node:child_process'), spawn: () => {
    const child = new EventEmitter()
    child.stdout = new PassThrough(); child.stderr = new PassThrough(); child.stdin = new PassThrough()
    child.stdin.end = data => {
      config = JSON.parse(data)
      complete = () => {
        const preview = path.join(f.run, `editor-preview-${config.source_id}.mp4`)
        fs.writeFileSync(preview, 'new preview')
        if (outcome === 'success' || outcome === 'committed-before-exit') {
          const project = JSON.parse(fs.readFileSync(path.join(f.run, 'editor-project.json')))
          project.source_id = config.source_id; project.revision++; project.width = 3840; project.height = 2160
          fs.writeFileSync(path.join(f.run, 'editor-project.json'), JSON.stringify(project))
        }
        child.stdout.write(JSON.stringify({ ok: outcome === 'success', error: outcome === 'failure' ? 'duration' : undefined }))
        child.emit('close', outcome === 'success' ? 0 : 1)
      }
      started()
    }
    return child
  } } })
  try {
    const replacement = path.join(f.dir, 'higher-quality.mp4')
    fs.writeFileSync(replacement, 'higher resolution source')
    await assert.rejects(f.main.replaceEditorSource(f.run, 0, replacement), /outside/)
    f.main.authorizeMedia(replacement)
    await assert.rejects(f.main.replaceEditorSource(f.run, 999, replacement), /changed/)
    await assert.rejects(f.main.runEditor(f.run, 0, '', 'replace-source'), /Invalid/)
    const pending = f.main.replaceEditorSource(f.run, 0, replacement)
    const result = outcome === 'success' ? pending : assert.rejects(pending, outcome === 'failure' ? /different duration/ : outcome === 'cancelled' ? /replacement cancelled\. The previous source is still in use/ : /replacement stopped/)
    await entered
    assert.equal(config.action, 'replace-source')
    assert.equal(fs.readFileSync(path.join(f.run, `editor-source-${config.source_id}.mp4`), 'utf8'), 'higher resolution source')
    assert.equal((await f.main.openEditor(f.run)).operation, 'replace-source')
    assert.equal((await f.main.openEditor(f.run)).sourcePath, fs.realpathSync(path.join(f.run, 'editor-source.mp4')))
    await assert.rejects(f.main.saveEditor(f.run, 0, fixture.candidates), /Wait/)
    await assert.rejects(f.main.replaceEditorSource(f.run, 0, replacement), /already running/)
    if (outcome === 'cancelled') f.main.cancelEditor(f.run)
    complete(); await result
    const reopened = await f.main.openEditor(f.run)
    assert.equal(reopened.operation, null)
    const committed = ['success', 'committed-before-exit'].includes(outcome)
    assert.equal(reopened.project.source_id, committed ? config.source_id : undefined)
    assert.equal(fs.existsSync(path.join(f.run, 'editor-source.mp4')), !committed)
    assert.equal(fs.existsSync(path.join(f.run, `editor-source-${config.source_id}.mp4`)), committed)
    assert.equal(fs.existsSync(path.join(f.run, `editor-preview-${config.source_id}.mp4`)), committed)
    const saved = await f.main.saveEditor(f.run, reopened.project.revision, reopened.project.candidates)
    assert.equal(saved.project.source_id, reopened.project.source_id)
    assert.deepEqual(saved.project.candidates, reopened.project.candidates)
    assert.equal(fs.readFileSync(replacement, 'utf8'), 'higher resolution source')
  } finally { f.cleanup() }
})

test('camera markers retain exact frame times and dismissals do not invalidate baked clips or reviews', () => {
  const p = clone(), c = p.candidates[0]
  c.status = 'baked'
  c.camera_scan = { start_ms: 1000, end_ms: 12000, frames: [1001, 1042.708, 1084.417, 2002, 2043.708], markers: [{ at_ms: 1042.708, score: .1 }, { at_ms: 2002, score: .04 }] }
  c.scenes[1].at_ms = 1042.708
  p.preview_id = 'a'.repeat(32); p.frame_preview = true
  const parsed = schema.parseEditorProject(p)
  assert.equal(parsed.candidates[0].scenes[1].at_ms, 1042.708)
  assert.deepEqual(parsed.candidates[0].camera_scan, c.camera_scan)
  const before = schema.renderEditKey(c), signature = schema.editSignature(c)
  const dismissed = schema.refineEdit(c, { dismissed_camera_markers: [1042.708] })
  assert.equal(dismissed.status, 'baked')
  assert.equal(schema.renderEditKey(dismissed), before)
  assert.equal(schema.editSignature(dismissed), signature)
  assert.equal(schema.cameraMarkers(c, .08).length, 1)
  assert.equal(schema.cameraMarkers(dismissed, .08).length, 0)
  assert.equal(schema.cameraMarkers(dismissed, .025).length, 1)
  assert.ok(!schema.candidateEdit(c).camera_scan)
  for (const mutate of [p => { p.preview_id = '../outside' }, p => { p.candidates[0].camera_scan.frames = [1001, 1001] },
    p => { p.candidates[0].camera_scan.markers[0].at_ms = 1100 }, p => { p.candidates[0].camera_scan.markers[0].score = NaN },
    p => { p.candidates[0].dismissed_camera_markers = [-1] }]) {
    const bad = structuredClone(p); mutate(bad); assert.throws(() => schema.parseEditorProject(bad))
  }
})

test('frame stepping and layout snapping use presentation timestamps including variable frame durations', () => {
  const frames = [0, 41.708, 83.417, 125.125, 208.542, 250.25]
  assert.equal(schema.stepFrame(frames, 83.417, -1), 41.708)
  assert.equal(schema.stepFrame(frames, 83.416999, 1), 125.125)
  assert.equal(schema.stepFrame(frames, 170, -1), 125.125)
  assert.equal(schema.stepFrame(frames, 170, 1), 208.542)
  assert.equal(schema.snapFrame(frames, 202), 208.542)
  assert.equal(schema.snapFrame(frames, 900), 900)
  const scenes = [{ at_ms: 0, layout: 'fit', crops: [[0, 0, 1, 1]] }, { at_ms: 125.125, layout: 'fit', crops: [[0, 0, 1, 1]] }]
  assert.equal(schema.retimeScene(scenes, 1, 202, 1000, frames)[1].at_ms, 208.542)
  assert.equal(schema.retimeScene(scenes, 1, 0, 1000, frames), scenes)
})

test('frame navigation can cross scan edges and recover in either direction', () => {
  const frames = [1000, 1041.708, 1083.417, 1125.125]
  const step = 1000 / 30
  assert.equal(schema.stepFrame(frames, 1000, -1), 1000 - step)
  assert.equal(schema.stepFrame(frames, 1125.125, 1), 1125.125 + step)
  assert.equal(schema.stepFrame(frames, 900, 1), 900 + step)
  assert.equal(schema.stepFrame(frames, 1200, -1), 1200 - step)
  assert.equal(schema.stepFrame(frames, 990, 1), 1000, 're-enter at the first known frame')
  assert.equal(schema.stepFrame(frames, 1140, -1), 1125.125, 're-enter at the last known frame')
  assert.equal(schema.stepFrame(frames, 1125.126, -1), 1083.417, 'browser timestamp noise must not trap the playhead')
  assert.equal(schema.stepFrame([], 3000, -1), 3000 - step)
})

test('saved camera dismissals persist while scan results and preview paths stay main-owned', async () => {
  const f = setup()
  try {
    const p = clone(), c = p.candidates[0]
    c.status = 'baked'
    c.camera_scan = { start_ms: 0, end_ms: 12000, frames: [0, 1001, 1042.708], markers: [{ at_ms: 1042.708, score: .3 }] }
    p.preview_id = 'b'.repeat(32); p.frame_preview = true
    fs.writeFileSync(path.join(f.run, `editor-preview-${p.preview_id}.mp4`), 'precise preview')
    fs.writeFileSync(path.join(f.run, 'editor-project.json'), JSON.stringify(p))
    const opened = await f.main.openEditor(f.run)
    assert.ok(opened.previewPath.endsWith(`editor-preview-${p.preview_id}.mp4`))
    const edits = structuredClone(opened.project.candidates)
    edits[0].dismissed_camera_markers = [1042.708]
    edits[0].camera_scan = { arbitrary: 'forged' }
    const saved = await f.main.saveEditor(f.run, opened.project.revision, edits)
    assert.deepEqual(saved.project.candidates[0].camera_scan, c.camera_scan)
    assert.equal(saved.project.candidates[0].status, 'baked')
    assert.deepEqual((await f.reload().openEditor(f.run)).project.candidates[0].dismissed_camera_markers, [1042.708])
  } finally { f.cleanup() }
})

test('camera scan locks edits, needs no provider key, and cancellation cleans only the uncommitted preview', async () => {
  const f = batchSetup()
  try {
    const before = fs.readFileSync(path.join(f.run, 'editor-project.json'), 'utf8')
    const pending = f.main.runEditor(f.run, 0, 'candidate-1', 'scan-cameras')
    const rejected = assert.rejects(pending, /Camera scan cancelled\. Your edits and previous markers are saved/)
    const worker = await nextWorker(f.workers, 1)
    assert.equal(worker.config.action, 'scan-cameras')
    assert.match(worker.config.preview_id, /^[a-f0-9]{32}$/)
    assert.equal((await f.main.openEditor(f.run)).operation, 'scan-cameras')
    await assert.rejects(f.main.saveEditor(f.run, 0, []), /Wait/)
    const pendingPreview = path.join(f.run, `editor-preview-${worker.config.preview_id}.mp4.partial.mp4`)
    fs.writeFileSync(pendingPreview, 'partial')
    f.main.cancelEditor(f.run); worker.finish(false); await rejected
    assert.equal(fs.readFileSync(path.join(f.run, 'editor-project.json'), 'utf8'), before)
    assert.equal(fs.existsSync(pendingPreview), false)
    assert.equal(fs.existsSync(path.join(f.run, 'editor-preview.mp4')), true)
    assert.equal((await f.main.openEditor(f.run)).operation, null)
  } finally { f.cleanup() }
})


test('camera progress streams across chunks, reconnects, stays monotonic, and clears on completion', async () => {
  const f = batchSetup()
  try {
    const pending = f.main.runEditor(f.run, 0, 'candidate-1', 'scan-cameras')
    const worker = await nextWorker(f.workers, 1)
    const progress = async () => (await f.main.openEditor(f.run)).progress
    assert.deepEqual(await progress(), { phase: 'scan', percent: 0 })
    worker.write('{"type":"progress","phase":"scan","per')
    assert.equal((await progress()).percent, 0)
    worker.write('cent":43}\n{"type":"progress","phase":"scan","percent":60}\n')
    assert.deepEqual(await progress(), { phase: 'scan', percent: 60 })
    for (const percent of [20, -1, 101, '90', null]) worker.write(JSON.stringify({ type: 'progress', phase: 'scan', percent }) + '\n')
    assert.equal((await progress()).percent, 60)
    worker.write('{"type":"progress","phase":"preview","percent":0}\n')
    worker.write('{"type":"progress","phase":"scan","percent":100}\n')
    worker.write('{"type":"progress","phase":"preview","percent":24}\n')
    assert.deepEqual(await progress(), { phase: 'preview', percent: 24 })
    worker.write('{"ok":true}\n'); worker.close(0)
    await pending
    assert.equal(await progress(), undefined)
  } finally { f.cleanup() }
})

for (const output of ['malformed\n', 'x'.repeat(16385)]) test('invalid editor progress protocol fails safely: ' + output.length, async () => {
  const f = batchSetup()
  try {
    const pending = f.main.runEditor(f.run, 0, 'candidate-1', 'scan-cameras')
    const rejected = assert.rejects(pending, /Camera scan stopped/)
    const worker = await nextWorker(f.workers, 1)
    worker.write(output); worker.write('\n{"ok":true}\n'); worker.close(0)
    await rejected
    assert.equal((await f.main.openEditor(f.run)).progress, undefined)
  } finally { f.cleanup() }
})

test('one failed clip does not stop "Bake all"; the report names it and it stays ready', async () => {
  const f = batchSetup()
  try {
    const pending = f.main.runEditor(f.run, 0, '', 'export-all')
    const rejected = assert.rejects(pending, (error) => {
      assert.match(error.message, /^Baked 1 of 2 ready clips\. Could not bake “The result”: /)
      assert.doesNotMatch(error.message, /System check/)
      assert.match(error.message, /still ready/)
      return true
    })
    ;(await nextWorker(f.workers, 1)).finish(false)
    const second = await nextWorker(f.workers, 2)
    assert.deepEqual((await f.main.openEditor(f.run)).batch, { completed: 0, total: 2, failed: 1 })
    second.finish()
    await rejected
    const reopened = await f.main.openEditor(f.run)
    assert.deepEqual(reopened.project.candidates.map(c => c.status), ['refining', 'ready', 'baked', 'discarded', 'baked'])
  } finally { f.cleanup() }
})

test('worker failures map fixed codes to specific messages and log only a redacted stderr tail', async () => {
  const { EventEmitter } = require('node:events')
  const { PassThrough } = require('node:stream')
  let spawned
  const f = setup({ child_process: { ...require('node:child_process'), spawn: (command, args, options) => {
    const child = new EventEmitter()
    child.stdout = new PassThrough(); child.stderr = new PassThrough(); child.stdin = new PassThrough()
    spawned = { command, args, options, child }
    return child
  } } })
  try {
    const project = JSON.parse(fs.readFileSync(path.join(f.run, 'editor-project.json')))
    project.candidates[0].status = 'ready'
    fs.writeFileSync(path.join(f.run, 'editor-project.json'), JSON.stringify(project))
    for (const [code, expected] of [['render_failed', /^Rendering failed\. Your edits are saved\..*Show Logs/], ['source_missing', /Replace source/],
      ['engine_unavailable', /System check/], ['/private/forged', /^Export stopped or failed\. Your edits are saved/]]) {
      const pending = f.main.runEditor(f.run, 0, 'candidate-1', 'export')
      const deadline = Date.now() + 5000
      while (!spawned && Date.now() < deadline) await new Promise(r => setTimeout(r, 5))
      const { child, options } = spawned; spawned = undefined
      assert.equal(options.env.PYTHONDONTWRITEBYTECODE, '1')
      child.stderr.write('2026-09-28 10:00:00,000 - editor_runner - ERROR - Editor export failed (render_failed)\n')
      child.stderr.write("RenderingError: Error opening /Users/someone/Movies/clip.mp4 api_key=sk-or-v1-abcdef123456 https://example.com/x\nNo such filter: 'perspective'\n") // gitleaks:allow -- synthetic credential exercises log redaction
      child.stdout.write(JSON.stringify({ ok: false, error: code }) + '\n'); child.emit('close', 1, null)
      const error = await pending.then(() => null, (e) => e)
      assert.match(error.message, expected)
      if (code !== 'engine_unavailable') assert.doesNotMatch(error.message, /System check/)
    }
    const log = fs.readFileSync(path.join(f.dir, 'logs', 'bridgeclip.log'), 'utf8')
    assert.match(log, /"code":"render_failed"/)
    assert.match(log, /No such filter: 'perspective'/)
    assert.doesNotMatch(log, /someone|sk-or|example\.com/)
  } finally { f.cleanup() }
})

test('progress summaries report counts without the project and refresh when the file changes', async () => {
  const f = setup()
  try {
    const first = await f.main.readEditorProgress(f.run)
    assert.deepEqual({ ...first, previewPath: path.basename(first.previewPath) }, {
      total: 2, remaining: 2, initialCandidate: 0, counts: { refining: 2, ready: 0, baked: 0, discarded: 0 },
      previewPath: 'editor-preview.mp4', thumbnailMs: fixture.candidates[0].ranges[0][0], mediaFreed: false, operation: null })
    assert.equal(JSON.stringify(first).includes('transcript'), false)
    const project = (await f.main.openEditor(f.run)).project
    project.candidates[0].status = 'discarded'
    await f.main.saveEditor(f.run, 0, project.candidates)
    const second = await f.main.readEditorProgress(f.run)
    assert.deepEqual([second.remaining, second.initialCandidate, second.counts.discarded], [1, 1, 1])
    await assert.rejects(f.main.readEditorProgress(path.join(f.dir, 'elsewhere')), /./)
  } finally { f.cleanup() }
})

test('idle editor runs sweep temporary folders and unreferenced media, never the active pair', async () => {
  const f = setup()
  try {
    const stale = ['.editor-export-abc/clip.mp4', '.editor-review-x/frame.jpg', `.editor-${'1'.repeat(8)}.tmp`,
      `editor-source-${'a'.repeat(32)}.mp4`, `editor-preview-${'b'.repeat(32)}.mp4.partial.mp4`, 'editor-preview.mp4.partial.mp4']
    for (const file of stale) { fs.mkdirSync(path.dirname(path.join(f.run, file)), { recursive: true }); fs.writeFileSync(path.join(f.run, file), 'x') }
    fs.writeFileSync(path.join(f.run, 'clip_00.mp4'), 'export')
    fs.writeFileSync(path.join(f.run, 'editor-notes.mp4'), 'unrelated')
    await f.main.openEditor(f.run)
    const left = fs.readdirSync(f.run).sort()
    assert.deepEqual(left, ['clip_00.mp4', 'editor-notes.mp4', 'editor-preview.mp4', 'editor-project.json', 'editor-source.mp4', 'job_output.json'])
  } finally { f.cleanup() }
})

test('undoing "Refine again" can restore Baked only for the exact baked render', async () => {
  const f = setup()
  try {
    const project = JSON.parse(fs.readFileSync(path.join(f.run, 'editor-project.json')))
    Object.assign(project.candidates[0], { status: 'baked', exports: [0] })
    fs.writeFileSync(path.join(f.run, 'editor-project.json'), JSON.stringify(project))
    const baked = (await f.main.openEditor(f.run)).project.candidates
    const refining = structuredClone(baked); refining[0].status = 'refining'
    let saved = await f.main.saveEditor(f.run, 0, refining)
    assert.match(saved.project.candidates[0].baked_hash, /^[a-f0-9]{64}$/)
    const changed = structuredClone(refining); changed[0].title = 'Different'; changed[0].status = 'baked'
    await assert.rejects(f.main.saveEditor(f.run, 1, changed), /completed render/)
    saved = await f.main.saveEditor(f.run, 1, baked)
    assert.equal(saved.project.candidates[0].status, 'baked')
    assert.equal(saved.project.candidates[0].baked_hash, undefined)
  } finally { f.cleanup() }
})

test('editor media can be freed only when nothing is left to finish, and the project becomes read-only', async () => {
  const f = setup()
  try {
    await assert.rejects(f.main.freeEditorMedia(f.run, 0), /Bake or discard every clip/)
    const project = JSON.parse(fs.readFileSync(path.join(f.run, 'editor-project.json')))
    project.candidates[0].status = 'discarded'; Object.assign(project.candidates[1], { status: 'baked', exports: [0] })
    fs.writeFileSync(path.join(f.run, 'editor-project.json'), JSON.stringify(project))
    assert.equal((await f.main.openEditor(f.run)).mediaBytes, 10)
    await assert.rejects(f.main.freeEditorMedia(f.run, 7), /changed/)
    const freed = await f.main.freeEditorMedia(f.run, 0)
    assert.equal(freed.project.media_freed, true)
    assert.deepEqual([freed.sourcePath, freed.previewPath], ['', ''])
    assert.equal(fs.existsSync(path.join(f.run, 'editor-source.mp4')) || fs.existsSync(path.join(f.run, 'editor-preview.mp4')), false)
    assert.equal((await f.main.readEditorProgress(f.run)).previewPath, null)
    await assert.rejects(f.main.saveEditor(f.run, 1, freed.project.candidates), /read-only/)
    await assert.rejects(f.main.runEditor(f.run, 1, 'candidate-2', 'export'), /read-only/)
  } finally { f.cleanup() }
})

test('the editor review runs on the chosen provider and is never handed a local Whisper server', async () => {
  const { EventEmitter } = require('node:events')
  const { PassThrough } = require('node:stream')
  let release, entered, workerEnv
  const started = new Promise((r) => { entered = r })
  const f = setup({ child_process: { ...require('node:child_process'), spawn: (_cmd, _args, options) => {
    workerEnv = options.env
    const child = new EventEmitter()
    child.stdout = new PassThrough(); child.stderr = new PassThrough(); child.stdin = new PassThrough()
    child.stdin.end = () => { entered(); release = () => { child.stdout.write('{"ok":true}'); child.emit('close', 0) } }
    return child
  } } })
  try {
    f.main.settings.replaceApiKey('openrouterApiKey', 'fixture-key')
    f.main.settings.replaceApiKey('openaiCompatibleApiKey', 'sk-custom-endpoint-key')
    f.main.settings.savePublicSettings({
      outputDirectory: f.library, pythonPath: 'python3',
      aiProvider: 'openai-compatible', openaiCompatibleUrl: 'http://localhost:11434/v1', openaiCompatibleModel: 'llama3.1',
      transcriptionSource: 'local', localWhisperMode: 'server', localWhisperUrl: 'http://192.168.1.20:8000/v1', localWhisperDevice: 'gpu'
    })
    const pending = f.main.runEditor(f.run, 0, 'candidate-1', 'review')
    await started
    try {
      // The review asks the chosen model, so it gets that endpoint and its key.
      assert.equal(workerEnv.AI_PROVIDER, 'openai-compatible')
      assert.equal(workerEnv.OPENAI_COMPATIBLE_URL, 'http://localhost:11434/v1')
      assert.equal(workerEnv.OPENAI_COMPATIBLE_API_KEY, 'sk-custom-endpoint-key')
      assert.equal(workerEnv.LOCAL_WHISPER_URL, '', 'no local Whisper server address')
      assert.equal(workerEnv.OPENROUTER_API_KEY, '', 'and no OpenRouter key the review does not use')
      assert.equal(workerEnv.JEV_ENABLED, 'true')
    } finally {
      // Let the fake worker finish even when an assertion above failed, so a regression reports instead of hanging.
      release(); await pending
    }
    // The saved choices were not touched by running the editor.
    const saved = f.main.settings.loadSettings()
    assert.equal(saved.aiProvider, 'openai-compatible')
    assert.equal(saved.transcriptionSource, 'local')
    assert.equal(saved.localWhisperUrl, 'http://192.168.1.20:8000/v1')
  } finally { f.cleanup() }
})

test('a Split crop can be used in every clip still being edited', () => {
  const split = (at, top, bottom) => ({ at_ms: at, layout: 'split', crops: [top, bottom] })
  const base = { title: 't', ranges: [[0, 10000]], captions: true, caption_preset: 'x', video_speed: 1, caption_edits: [], caption_suppression_ranges: [] }
  const wrong = [0.05, 0.05, 0.2, 0.2], right = [0.7, 0.7, 0.25, 0.25], screen = [0.2, 0.1, 0.5, 0.5]
  const clips = [
    { ...base, id: 'a', status: 'ready', scenes: [split(0, wrong, screen), { at_ms: 5000, layout: 'fill', crops: [screen] }] },
    { ...base, id: 'b', status: 'refining', scenes: [split(0, wrong, screen), split(4000, wrong, screen)] },
    { ...base, id: 'c', status: 'baked', scenes: [split(0, wrong, screen)] },
    { ...base, id: 'd', status: 'discarded', scenes: [split(0, wrong, screen)] },
    { ...base, id: 'e', status: 'ready', scenes: [split(0, right, screen)] },
    { ...base, id: 'f', status: 'ready', scenes: [{ ...split(0, wrong, screen), kind: 'two_shot' }, { ...split(4000, wrong, screen), kind: 'screen_cam' }] }
  ]
  const result = schema.copySplitCrop(clips, 0, right)
  assert.deepEqual([result.sections, result.clips], [4, 3])
  const [a, b, c, d, e, f] = result.candidates
  assert.deepEqual(f.scenes.map((s) => s.crops[0]), [wrong, right])  // two people side by side are left alone
  assert.deepEqual(a.scenes[0].crops, [right, screen])
  assert.deepEqual(a.scenes[1], clips[0].scenes[1])        // Full frame sections are left alone
  assert.equal(a.status, 'refining')                        // an edited clip is refined again
  assert.deepEqual(b.scenes.map((s) => s.crops[0]), [right, right])
  assert.equal(c, clips[2]); assert.equal(d, clips[3])      // baked and discarded clips are untouched
  assert.equal(e, clips[4])                                 // already right: unchanged, still ready
  assert.deepEqual(schema.copySplitCrop(result.candidates, 0, right).sections, 0)
})

test('a suggested Split keeps what kind it is, and only a Split can have one', () => {
  const p = JSON.parse(JSON.stringify(fixture))
  const scene = p.candidates[0].scenes[0]
  Object.assign(scene, { layout: 'split', crops: [[0, 0, 0.3, 0.3], [0.3, 0.3, 0.5, 0.5]], kind: 'two_shot' })
  assert.equal(schema.parseEditorProject(p).candidates[0].scenes[0].kind, 'two_shot')
  scene.kind = 'banana'
  assert.equal(schema.parseEditorProject(p).candidates[0].scenes[0].kind, undefined)
  Object.assign(scene, { layout: 'fill', crops: [[0, 0, 0.5, 0.5]], kind: 'screen_cam' })
  assert.equal(schema.parseEditorProject(p).candidates[0].scenes[0].kind, undefined)
})
