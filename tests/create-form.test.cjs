const assert = require('node:assert/strict')
const { test } = require('node:test')
const path = require('node:path')
const { buildSync } = require('esbuild')
const React = require('react')
const { renderToStaticMarkup } = require('react-dom/server')

const bundled = buildSync({
  stdin: {
    contents: `export { FormatStep, ClipsStep, CaptionsStep, JobForm, buildJobRequest, parseTrimRange } from './src/renderer/components/JobForm';
      export { JobProgress } from './src/renderer/components/JobProgress';
      export { SetupCard } from './src/renderer/components/SetupCard';
      export { WorkflowPicker } from './src/renderer/components/WorkflowPicker';
      export { JevSettings } from './src/renderer/components/JevSettings';
      export { RunStats } from './src/renderer/components/RunStats';
      export { ProviderPicker, ProviderFields, TranscriptionSettings, CheckList } from './src/renderer/components/ProviderSetup';
      export { useApiKeyDrafts } from './src/renderer/hooks/use-api-key-drafts';
      export { useSettingsStore, useSetupState } from './src/renderer/store/use-settings-store';
      export { useDraftStore } from './src/renderer/store/use-draft-store';
      export { SourcePicker, isValidSourceLink } from './src/renderer/components/SourcePicker';
      export { framingProblem, sourceAnalysisNotice } from './src/renderer/components/ClipList';
      export { parseJobOutput } from './src/shared/job-output';
      export { twitchVodId, normalizeVideoSource } from './src/shared/video-source';`,
    resolveDir: path.resolve(__dirname, '..'),
    loader: 'ts'
  },
  bundle: true,
  platform: 'node',
  format: 'cjs',
  packages: 'external',
  loader: { '.css': 'empty' },
  define: { __APP_VERSION__: JSON.stringify(require('../package.json').version) },
  jsx: 'automatic',
  write: false
}).outputFiles[0].text

const form = { exports: {} }
new Function('module', 'exports', 'require', bundled)(form, form.exports, require)
const { FormatStep, JobForm, isValidSourceLink, parseTrimRange, framingProblem, sourceAnalysisNotice, parseJobOutput } = form.exports

test('setup needs only OpenRouter for clipping', () => {
  const { SetupCard, useSettingsStore } = form.exports
  useSettingsStore.setState({ openrouterConfigured: false })
  const setup = renderToStaticMarkup(React.createElement(SetupCard, { onOpenSettings() {} }))
  assert.match(setup, /OpenRouter/)
  assert.doesNotMatch(setup, /ElevenLabs/)
  useSettingsStore.setState({ openrouterConfigured: false, toolStatus: null })
})

test('source picker accepts full HTTP(S) links and rejects malformed or credentialed links', () => {
  assert.equal(isValidSourceLink(' https://www.youtube.com/watch?v=abc '), true)
  assert.equal(isValidSourceLink('http://example.com/video.mp4'), true)
  assert.equal(isValidSourceLink('https://'), false)
  assert.equal(isValidSourceLink('https://user:pass@example.com/video'), false)
  assert.equal(isValidSourceLink('file:///tmp/video.mp4'), false)
})

test('trim validation matches main process bounds, including an end at zero', () => {
  assert.equal(parseTrimRange(true, '', '0').error, 'End must be after the start.')
  assert.equal(parseTrimRange(true, '1:30', '90').error, 'End must be after the start.')
  assert.deepEqual(parseTrimRange(true, '', '1:30'), { start: null, end: 90, error: null })
  assert.deepEqual(parseTrimRange(false, 'oops', '0'), { start: null, end: null, error: null })
})

test('the wizard opens on the video step with the steps listed in order', () => {
  const html = renderToStaticMarkup(React.createElement(JobForm, { onSubmit() {} }))
  const labels = [...html.matchAll(/aria-label="Create steps">(.*?)<\/nav>/gs)][0]?.[1] ?? ''
  const order = ['Video', 'Format', 'Clips', 'Captions', 'Review'].map((label) => labels.indexOf(label))
  assert.ok(order.every((index, i) => index > -1 && (i === 0 || index > order[i - 1])))
  assert.match(html, /aria-current="step"[^>]*>.*?Video/s)
  assert.match(html, /Choose a video/)
  assert.equal(form.exports.useDraftStore.getState().workflow, null)
  const radios = html.match(/<button[^>]*role="radio"[^>]*>/g) ?? []
  assert.equal(radios.length, 2)
  assert.ok(radios.every((radio) => radio.includes('aria-checked="false"')))
  assert.equal(radios.filter((radio) => radio.includes('tabindex="0"')).length, 1)
  assert.match(html, /Beginner friendly/)
  assert.match(html, /For advanced users/)
})

test('a job requires an explicit workflow and each new video resets that choice', () => {
  const { useDraftStore, buildJobRequest } = form.exports
  const original = useDraftStore.getState()
  const trim = { start: null, end: null }
  try {
    original.update({ source: 'https://example.com/video', workflow: null })
    assert.throws(() => buildJobRequest(useDraftStore.getState(), trim), /Choose a workflow/)
    for (const workflow of ['automatic', 'review']) {
      original.update({ workflow })
      original.setStep('format')
      original.setStep('video')
      assert.equal(useDraftStore.getState().workflow, workflow)
      assert.equal(buildJobRequest(useDraftStore.getState(), trim).workflow, workflow)
      original.startAnother()
      assert.equal(useDraftStore.getState().workflow, null)
    }
  } finally { useDraftStore.setState(original) }
})

test('clipping mode is selectable and economy disables paid vision in the submitted request', () => {
  const { ClipsStep, buildJobRequest } = form.exports
  const draft = {
    workflow: 'automatic', source: 'https://example.com/video', clippingMode: 'economy', aspectRatio: '9:16', layoutStyle: 'auto',
    layoutVision: true, pacing: 'tight', durations: ['short'], autoClipCount: true, maxClips: 5,
    includeCaptions: true, captionPreset: 'pop'
  }
  const html = renderToStaticMarkup(React.createElement(ClipsStep, { draft, update() {} }))
  assert.match(html, /aria-label="Clipping mode"/)
  assert.match(html, /Economy/)
  const request = buildJobRequest(draft, { start: null, end: null })
  assert.equal(request.clippingMode, 'economy')
  assert.equal(Object.hasOwn(request, 'debugCapture'), false)
  assert.equal(request.layoutVision, false)
  assert.equal(buildJobRequest({ ...draft, clippingMode: 'quality' }, { start: null, end: null }).layoutVision, true)
})

test('the title card is shown by default and can be turned off for automatic runs', () => {
  const { CaptionsStep, useDraftStore, buildJobRequest } = form.exports
  const original = useDraftStore.getState()
  try {
    assert.equal(original.includeTitle, true)
    original.update({ workflow: 'automatic', source: 'https://example.com/video' })
    const automatic = renderToStaticMarkup(React.createElement(CaptionsStep, { draft: useDraftStore.getState(), update() {} }))
    assert.match(automatic, /Show title at the top/)
    assert.equal(buildJobRequest(useDraftStore.getState(), { start: null, end: null }).includeTitle, true)
    original.update({ includeTitle: false })
    assert.equal(buildJobRequest(useDraftStore.getState(), { start: null, end: null }).includeTitle, false)
    original.startAnother()
    assert.equal(useDraftStore.getState().includeTitle, false)
    // Review exports never draw a title card, so the switch is not offered there.
    const review = renderToStaticMarkup(React.createElement(CaptionsStep, { draft: { ...useDraftStore.getState(), workflow: 'review' }, update() {} }))
    assert.doesNotMatch(review, /Show title at the top/)
  } finally { useDraftStore.setState(original) }
})

test('what to clip is optional, trimmed into the request and cleared for the next video', () => {
  const { ClipsStep, useDraftStore, buildJobRequest } = form.exports
  const original = useDraftStore.getState()
  try {
    assert.equal(original.clipRequest, '')
    original.update({ workflow: 'automatic', source: 'https://example.com/video' })
    const html = renderToStaticMarkup(React.createElement(ClipsStep, { draft: useDraftStore.getState(), update() {} }))
    assert.match(html, /aria-label="What to clip"/)
    assert.match(html, /maxLength="1000"/)
    assert.match(html, /aria-describedby="clip-request-help"/)
    assert.match(html, /id="clip-request-help"[^>]*>Only matching moments are clipped/)
    assert.equal(Object.hasOwn(buildJobRequest(useDraftStore.getState(), { start: null, end: null }), 'clipRequest'), false)
    original.update({ clipRequest: '   ' })
    assert.equal(Object.hasOwn(buildJobRequest(useDraftStore.getState(), { start: null, end: null }), 'clipRequest'), false)
    original.update({ clipRequest: '  every time they talk about pricing \n' })
    assert.equal(buildJobRequest(useDraftStore.getState(), { start: null, end: null }).clipRequest, 'every time they talk about pricing')
    original.startAnother()
    assert.equal(useDraftStore.getState().clipRequest, '')
  } finally { useDraftStore.setState(original) }
})

test('a running job shows what the user asked to clip', () => {
  const { JobProgress } = form.exports
  const job = {
    id: 'job', revision: 1, status: 'planning', percent: 40, step: 'Finding moments', clipsDone: 0, clipsTotal: 0,
    queuedAt: Date.now(), startedAt: Date.now(),
    request: { videoUrl: 'https://www.youtube.com/watch?v=abc123def45', workflow: 'automatic', clipRequest: ' every time they talk about pricing ' }
  }
  const html = renderToStaticMarkup(React.createElement(JobProgress, { job, onCancel() {} }))
  assert.match(html, /title="every time they talk about pricing"><span class="sr-only">What to clip: <\/span>every time they talk about pricing</)
  const plain = renderToStaticMarkup(React.createElement(JobProgress, { job: { ...job, request: { ...job.request, clipRequest: undefined } }, onCancel() {} }))
  assert.doesNotMatch(plain, /What to clip/)
})

test('format and framing radio groups each expose one keyboard tab stop', () => {
  const draft = { aspectRatio: '9:16', layoutStyle: 'auto', layoutVision: true, pacing: 'tight' }
  const html = renderToStaticMarkup(React.createElement(FormatStep, { draft, update() {} }))
  for (const label of ['Format', 'Framing', 'Video speed']) {
    const group = html.match(new RegExp(`role="radiogroup" aria-label="${label}"[^>]*>(.*?)<\\/div>`, 's'))?.[1]
    assert.ok(group)
    assert.equal((group.match(/tabindex="0"/g) ?? []).length, 1)
  }
})

test('speed survives navigation and another job, and appears in the submitted request', () => {
  const { useDraftStore, buildJobRequest, ClipsStep } = form.exports
  const original = useDraftStore.getState()
  try {
    assert.equal(original.videoSpeed, 1)
    original.update({ workflow: 'automatic', source: 'https://example.com/video', videoSpeed: 1.5 })
    original.setStep('review')
    assert.equal(useDraftStore.getState().step, 'review')
    const lengths = renderToStaticMarkup(React.createElement(ClipsStep, { draft: useDraftStore.getState(), update() {} }))
    assert.match(lengths, /60 seconds becomes about 40 seconds/)
    assert.equal(buildJobRequest(useDraftStore.getState(), { start: 10, end: 70 }).videoSpeed, 1.5)
    original.startAnother()
    assert.equal(useDraftStore.getState().videoSpeed, 1.5)
    assert.equal(useDraftStore.getState().step, 'video')
  } finally { useDraftStore.setState(original) }
})

test('saved run speed is retained while invalid speed metadata is discarded', () => {
  assert.equal(parseJobOutput({ clips: [], metrics: { requested_settings: { video_speed: 1.5 } } }).metrics.requested_settings.video_speed, 1.5)
  for (const video_speed of ['2', null, Infinity, 0, 3]) {
    assert.equal(parseJobOutput({ clips: [], metrics: { requested_settings: { video_speed } } }).metrics.requested_settings.video_speed, undefined)
  }
})

test('advanced selections travel with the job while presets ignore retained custom choices', () => {
  const { ClipsStep, buildJobRequest } = form.exports
  const draft = {
    workflow: 'automatic', source: 'https://example.com/video', clippingMode: 'advanced',
    plannerModel: 'provider/planning', transcriptionModel: 'provider/speech',
    aspectRatio: '9:16', layoutStyle: 'auto', layoutVision: true, pacing: 'tight',
    durations: ['short'], autoClipCount: true, maxClips: 5, includeCaptions: true, captionPreset: 'pop'
  }
  const html = renderToStaticMarkup(React.createElement(ClipsStep, { draft, update() {} }))
  assert.match(html, /Clip planning model/)
  assert.match(html, /Transcription model/)
  assert.equal((html.match(/role="combobox"/g) ?? []).length, 2)
  const request = buildJobRequest(draft, { start: null, end: null })
  assert.equal(request.plannerModel, 'provider/planning')
  assert.equal(request.transcriptionModel, 'provider/speech')
  assert.equal(request.layoutVision, true)
  for (const clippingMode of ['quality', 'economy']) {
    const preset = buildJobRequest({ ...draft, clippingMode }, { start: null, end: null })
    assert.equal(preset.plannerModel, undefined)
    assert.equal(preset.transcriptionModel, undefined)
  }
})

test('clip list explains when smart framing intentionally keeps the whole frame', () => {
  const clip = (index) => ({
    clip_index: index, s3_url: `/tmp/clip-${index}.mp4`, duration_ms: 5000,
    start_time_ms: index * 5000, end_time_ms: (index + 1) * 5000, virality_score: 0.5
  })
  const output = parseJobOutput({
    clips: [clip(0), clip(1)],
    metrics: { clip_layouts: [
      { clip_index: 0, framing_status: 'whole_frame_auto' },
      { clip_index: 0, framing_status: 'whole_frame_auto' },
      { clip_index: 9, framing_status: 'whole_frame_auto' }
    ] }
  })
  assert.ok(output)
  assert.match(framingProblem(output, true), /whole frame for clip 1/)
  assert.equal(framingProblem(output, false), null)
  const classic = parseJobOutput({ clips: [clip(0)], metrics: {
    smart_framing_available: false, requested_settings: { aspect_ratio: '9:16', layout_style: 'fit' }
  } })
  assert.ok(classic)
  assert.equal(framingProblem(classic, true), null)
})

test('visual-only runs disclose unavailable captions and preserve analysis status', () => {
  const output = parseJobOutput({ clips: [], metrics: {
    transcription_status: 'no_speech', planning_source: 'visual', visual_frame_count: 12,
    captions_status: 'unavailable_without_transcript'
  } })
  assert.ok(output)
  assert.equal(output.metrics.visual_frame_count, 12)
  assert.equal(output.metrics.captions_status, 'unavailable_without_transcript')
  assert.match(sourceAnalysisNotice(output), /No speech was detected/)
})

test('Twitch VOD links canonicalize while other Twitch pages are rejected', () => {
  const { normalizeVideoSource, twitchVodId, SourcePicker } = form.exports
  for (const host of ['twitch.tv', 'www.twitch.tv', 'm.twitch.tv', 'go.twitch.tv']) {
    const source = `https://${host}/videos/12345/?t=1h&tracking=secret`
    assert.equal(isValidSourceLink(source), true)
    assert.equal(normalizeVideoSource(source), 'https://www.twitch.tv/videos/12345')
    assert.equal(twitchVodId(source), '12345')
  }
  for (const source of ['https://twitch.tv/channel', 'https://clips.twitch.tv/Clip', 'https://player.twitch.tv/?video=123', 'https://twitch.tv/videos/nope', 'https://twitch.tv:8443/videos/123']) assert.equal(isValidSourceLink(source), false)
  assert.equal(twitchVodId('https://twitch.tv.evil.test/videos/123'), null)
  const html = renderToStaticMarkup(React.createElement(SourcePicker, { value: 'https://www.twitch.tv/videos/12345', onChange() {} }))
  assert.match(html, /Twitch VOD/)
  assert.match(html, /Public, completed videos only/)
  assert.doesNotMatch(html, /<img/)
})


test('Review & edit shows required Jev review even when automatic review is off', () => {
  const { ClipsStep, useDraftStore, useSettingsStore } = form.exports
  const original = useSettingsStore.getState()
  const initialState = useSettingsStore.getInitialState()
  const initialJev = initialState.jevEnabled
  try {
    useSettingsStore.setState({ jevEnabled: 'off' })
    initialState.jevEnabled = 'off'
    const draft = useDraftStore.getState()
    const review = renderToStaticMarkup(React.createElement(ClipsStep, { draft: { ...draft, workflow: 'review' }, update() {} }))
    const automatic = renderToStaticMarkup(React.createElement(ClipsStep, { draft: { ...draft, workflow: 'automatic' }, update() {} }))
    assert.match(review, /Jev review required/)
    assert.doesNotMatch(review, /Jev review &amp; repairs off/)
    assert.match(automatic, /Jev review &amp; repairs off/)
    assert.equal(useSettingsStore.getState().jevEnabled, 'off')
  } finally { initialState.jevEnabled = initialJev; useSettingsStore.setState(original) }
})

// ── AI provider and transcription choices ───────────────────────────────────────

const TOOLS_OK = { python: true, pythonDeps: true, ffmpeg: true, ffprobe: true, ytdlp: true, engine: true, bridgeRunner: true, ffmpegCaptions: true }
const NOTHING_SET = {
  aiProvider: 'openrouter', openrouterConfigured: false, geminiConfigured: false, openaiCompatibleConfigured: false,
  openaiCompatibleUrl: '', openaiCompatibleModel: '', openaiCompatibleVision: 'off', geminiModel: 'gemini-3.8-flash',
  transcriptionSource: 'cloud', localWhisperMode: 'builtin', localWhisperUrl: '', localWhisperModel: 'small',
  jevEnabled: 'off', toolStatus: TOOLS_OK, toolError: null, checkingTools: false
}

/**
 * Runs `fn` with the settings store in a known state, then puts it back. React's server renderer
 * reads zustand's initial state rather than the current one, so both are set.
 */
function withSettings(patch, fn) {
  const { useSettingsStore } = form.exports
  const original = useSettingsStore.getState()
  const initial = useSettingsStore.getInitialState()
  const next = { ...NOTHING_SET, ...patch }
  const before = Object.fromEntries(Object.keys(next).map((key) => [key, initial[key]]))
  Object.assign(initial, next)
  useSettingsStore.setState(next)
  try { return fn() } finally { Object.assign(initial, before); useSettingsStore.setState(original) }
}
const render = (type, props = {}) => renderToStaticMarkup(React.createElement(type, props))
/** The disabled attribute itself; a Tailwind class such as disabled:opacity-45 does not count. */
const isDisabled = (tag) => /\sdisabled=""/.test(tag ?? '')

/** What useSetupState reports, read back out of a rendered probe. */
function setupState(patch) {
  return withSettings(patch, () => {
    const Probe = () => React.createElement('i', { 'data-state': JSON.stringify(form.exports.useSetupState()) })
    const attribute = /data-state="([^"]*)"/.exec(render(Probe))[1]
    return JSON.parse(attribute.replaceAll('&quot;', '"').replaceAll('&amp;', '&'))
  })
}

const DRAFT = {
  workflow: 'automatic', source: 'https://example.com/video', clippingMode: 'economy', aspectRatio: '9:16', layoutStyle: 'auto',
  layoutVision: true, pacing: 'tight', durations: ['short'], autoClipCount: true, maxClips: 5, includeCaptions: true, captionPreset: 'pop',
  plannerModel: 'anthropic/claude-opus-5.5', transcriptionModel: 'openai/whisper-large-v3-turbo'
}
const TRIM = { start: null, end: null }

test('a request keeps OpenRouter modes, and sends other providers only what they can run', () => {
  const { buildJobRequest } = form.exports
  // OpenRouter is untouched.
  assert.equal(buildJobRequest(DRAFT, TRIM, 'openrouter').clippingMode, 'economy')
  const advanced = buildJobRequest({ ...DRAFT, clippingMode: 'advanced' }, TRIM, 'openrouter')
  assert.equal(advanced.plannerModel, 'anthropic/claude-opus-5.5')
  assert.equal(advanced.transcriptionModel, 'openai/whisper-large-v3-turbo')
  assert.equal(buildJobRequest(DRAFT, TRIM).clippingMode, 'economy', 'the provider argument is optional and defaults to OpenRouter')
  // Gemini and OpenAI-compatible run Quality with their own model, and never send OpenRouter model ids.
  for (const provider of ['gemini', 'openai-compatible']) {
    for (const clippingMode of ['economy', 'advanced', 'quality']) {
      const request = buildJobRequest({ ...DRAFT, clippingMode }, TRIM, provider)
      assert.equal(request.clippingMode, 'quality', `${provider} ${clippingMode}`)
      assert.equal(Object.hasOwn(request, 'plannerModel'), false)
      assert.equal(Object.hasOwn(request, 'transcriptionModel'), false)
    }
  }
  // buildJobRequest never rewrites the workflow it is given. Moving a draft off Review & edit when the provider
  // cannot run it is JobForm's effect (tested below), and the main process refuses an impossible one regardless.
  assert.equal(buildJobRequest({ ...DRAFT, workflow: 'review' }, TRIM, 'gemini').workflow, 'review')
})

test('setup state: each provider and transcription choice reports exactly what is still missing', () => {
  const ids = (state) => state.gaps.map((gap) => gap.id)
  assert.deepEqual(ids(setupState({})), ['openrouter-key'])
  assert.equal(setupState({}).summary, 'Add OpenRouter key')
  assert.equal(setupState({ openrouterConfigured: true }).ready, true)
  // Gemini with the default (cloud) transcription still needs an OpenRouter key, and says so.
  const gemini = setupState({ aiProvider: 'gemini', geminiConfigured: true })
  assert.deepEqual(ids(gemini), ['openrouter-key'])
  assert.match(gemini.gaps[0].message, /Cloud transcription uses OpenRouter/)
  assert.equal(gemini.ready, false)
  // ...but a Gemini key and local Whisper is complete with no OpenRouter key at all.
  const local = setupState({ aiProvider: 'gemini', geminiConfigured: true, transcriptionSource: 'local' })
  assert.deepEqual([local.gaps.length, local.ready, local.summary], [0, true, ''])
  assert.deepEqual(ids(setupState({ aiProvider: 'gemini', transcriptionSource: 'local' })), ['gemini-key'])
  // OpenAI-compatible needs a URL and a model; the key is optional.
  assert.deepEqual(ids(setupState({ aiProvider: 'openai-compatible', transcriptionSource: 'local' })), ['openai-url', 'openai-model'])
  assert.equal(setupState({ aiProvider: 'openai-compatible', transcriptionSource: 'local', openaiCompatibleUrl: 'http://localhost:11434/v1', openaiCompatibleModel: 'llama3.1' }).ready, true)
  // A Whisper server needs its URL; the built-in one needs nothing.
  assert.deepEqual(ids(setupState({ openrouterConfigured: true, transcriptionSource: 'local', localWhisperMode: 'server' })), ['whisper-url'])
  assert.equal(setupState({ openrouterConfigured: true, transcriptionSource: 'local', localWhisperMode: 'builtin' }).ready, true)
  // Complete choices still wait for the tools check.
  assert.equal(setupState({ openrouterConfigured: true, toolStatus: null }).ready, false)
  assert.equal(setupState({ openrouterConfigured: true, toolStatus: { ...TOOLS_OK, ffmpeg: false } }).toolsOk, false)
})

test('first run: a Gemini user can finish setup from the card, and sees why OpenRouter is asked for', () => {
  const card = (patch) => withSettings(patch, () => render(form.exports.SetupCard, { onOpenSettings() {} }))
  const stuck = card({ aiProvider: 'gemini', geminiConfigured: true })
  assert.match(stuck, /Cloud transcription uses OpenRouter/)
  assert.match(stuck, /OpenRouter key for transcription/, 'the key field is on the card, not only in Settings')
  assert.match(stuck, /Use Local Whisper instead/, 'and so is the way out')
  assert.doesNotMatch(stuck, /Add your OpenRouter \(for cloud transcription\) key/)
  // Choosing local transcription completes setup; the card is gone.
  assert.equal(card({ aiProvider: 'gemini', geminiConfigured: true, transcriptionSource: 'local' }), '')
  // The same for a custom endpoint: its fields are on the card until URL and model are set.
  const custom = card({ aiProvider: 'openai-compatible', transcriptionSource: 'local' })
  assert.match(custom, /Endpoint URL/)
  assert.match(custom, /Model name/)
  assert.match(custom, /API key \(optional\)/)
  assert.equal(card({ aiProvider: 'openai-compatible', transcriptionSource: 'local', openaiCompatibleUrl: 'http://localhost:11434/v1', openaiCompatibleModel: 'llama3.1' }), '')
  // A Whisper server asks for its URL on the card.
  assert.match(card({ openrouterConfigured: true, transcriptionSource: 'local', localWhisperMode: 'server' }), /Whisper server URL/)
  // An OpenRouter user with a key sees nothing.
  assert.equal(card({ openrouterConfigured: true }), '')
})

test('the provider choice is a radio group with one tab stop and the chosen provider checked', () => {
  for (const [aiProvider, label] of [['openrouter', 'OpenRouter'], ['gemini', 'Google Gemini'], ['openai-compatible', 'OpenAI-compatible']]) {
    for (const compact of [false, true]) {
      const html = withSettings({ aiProvider }, () => render(form.exports.ProviderPicker, { commit: async () => null, compact }))
      assert.match(html, /role="radiogroup"[^>]*aria-label="AI provider"|aria-label="AI provider"[^>]*role="radiogroup"/)
      const radios = html.match(/<button[^>]*role="radio"[^>]*>/g) ?? []
      assert.equal(radios.length, 3)
      assert.equal(radios.filter((radio) => radio.includes('aria-checked="true"')).length, 1)
      assert.equal(radios.filter((radio) => radio.includes('tabindex="0"')).length, 1)
      const checked = radios.find((radio) => radio.includes('aria-checked="true"'))
      assert.ok(checked.includes('tabindex="0"'), 'the checked radio is the tab stop')
      assert.match(html, new RegExp(label))
    }
  }
})

test('Review & edit is available with every provider', () => {
  const { WorkflowPicker } = form.exports
  const radio = (html, id) => (html.match(/<button[^>]*role="radio"[^>]*>/g) ?? []).find((button) => button.includes(`workflow-${id}-title`))
  for (const aiProvider of ['openrouter', 'gemini', 'openai-compatible']) {
    const html = withSettings({ aiProvider }, () => render(WorkflowPicker, { value: 'automatic', onChange() {} }))
    assert.equal(isDisabled(radio(html, 'review')), false, `${aiProvider}: Review & edit can be chosen`)
    assert.doesNotMatch(html, /OpenRouter only/)
    assert.equal(isDisabled(radio(html, 'automatic')), false)
  }
})

test('the Clips step shows the real provider and model, with no stand-in names', () => {
  const { ClipsStep } = form.exports
  const gemini = withSettings({ aiProvider: 'gemini', geminiConfigured: true, geminiModel: 'gemini-3.8-flash', transcriptionSource: 'local', localWhisperModel: 'medium' }, () => render(ClipsStep, { draft: DRAFT, update() {} }))
  assert.match(gemini, /Google Gemini/)
  assert.match(gemini, /gemini-3\.8-flash/)
  assert.match(gemini, /Local Whisper · medium/)
  assert.doesNotMatch(gemini, /aria-label="Clipping mode"/, 'Economy and Advanced are OpenRouter presets')
  assert.doesNotMatch(gemini, /llama3|whisper-1|gemini-2\.5/)
  const custom = withSettings({ aiProvider: 'openai-compatible', openaiCompatibleModel: 'qwen2.5:7b', transcriptionSource: 'local', localWhisperMode: 'server' }, () => render(ClipsStep, { draft: DRAFT, update() {} }))
  assert.match(custom, /qwen2\.5:7b/)
  assert.match(custom, /Local Whisper server/)
  const router = withSettings({}, () => render(ClipsStep, { draft: DRAFT, update() {} }))
  assert.match(router, /aria-label="Clipping mode"/)
  assert.match(router, /Economy/)
})

test('the vision switch follows what the chosen model can do and names the right charges', () => {
  const { FormatStep } = form.exports
  const draft = { ...DRAFT, clippingMode: 'quality' }
  const open = (patch) => withSettings(patch, () => render(FormatStep, { draft, update() {} }))
  const switchOf = (html) => (html.match(/<button[^>]*aria-label="AI vision for smart framing"[^>]*>/) ?? [''])[0]

  const router = open({})
  assert.match(router, /May add OpenRouter charges/)
  assert.equal(isDisabled(switchOf(router)), false)

  const gemini = open({ aiProvider: 'gemini' })
  assert.doesNotMatch(gemini, /OpenRouter charges/)
  assert.match(gemini, /Gemini quota/)
  assert.equal(isDisabled(switchOf(gemini)), false)

  const textOnly = open({ aiProvider: 'openai-compatible', openaiCompatibleVision: 'off' })
  assert.match(textOnly, /text-only/)
  assert.ok(isDisabled(switchOf(textOnly)), 'a text-only model cannot be asked to read frames')
  assert.match(switchOf(textOnly), /aria-checked="false"/)

  assert.equal(isDisabled(switchOf(open({ aiProvider: 'openai-compatible', openaiCompatibleVision: 'on' }))), false)

  // A leftover Economy draft does not hide the switch from a provider that has no Economy.
  const leftover = withSettings({ aiProvider: 'gemini' }, () => render(FormatStep, { draft: { ...DRAFT, clippingMode: 'economy' }, update() {} }))
  assert.doesNotMatch(leftover, /off in Economy mode/)
  assert.match(leftover, /AI vision for smart framing/)
})

test('with another provider, the review settings say the chosen model answers the checks, and still work', () => {
  const toggle = (html) => html.match(/<button[^>]*role="switch"[^>]*>/)[0]
  const router = withSettings({ loaded: true }, () => render(form.exports.JevSettings))
  assert.doesNotMatch(router, /your own model answers the same checks/)
  assert.equal(isDisabled(toggle(router)), false, 'the switch works with OpenRouter')
  const gemini = withSettings({ loaded: true, aiProvider: 'gemini' }, () => render(form.exports.JevSettings))
  assert.match(gemini, /With Google Gemini, your own model answers the same checks instead/)
  assert.match(gemini, /less calibrated than Jev/)
  assert.equal(isDisabled(toggle(gemini)), false, 'the switch works with Gemini too')
})

test('a run on Gemini or a custom endpoint with no reported price shows "Not tracked", not a small number', () => {
  const { RunStats } = form.exports
  const output = { total_clips: 2, clips: [{ virality_score: 0.8 }], source_video_duration_seconds: 600, processing_time_seconds: 90, created_at: null, metrics: {} }
  const planning = (provider, cost) => ({ provider, model: 'm', prompt_tokens: 1, completion_tokens: 1, total_tokens: 2, estimated_cost_usd: cost, attempts: 1 })
  for (const provider of ['gemini', 'openai-compatible']) {
    const html = render(RunStats, { output, costs: { total_estimated_cost_usd: 0, cost_incomplete: true, planning: planning(provider, 0) }, videoSpeed: null })
    assert.match(html, /Not tracked/)
    assert.match(html, /check its dashboard/)
    assert.doesNotMatch(html, /\$0\.00/)
    assert.doesNotMatch(html, /OpusClip credit rate/)
  }
  // Local transcription at $0 with a priced OpenRouter plan is still an ordinary number.
  const priced = render(RunStats, { output, costs: { total_estimated_cost_usd: 0.42, transcription: { provider: 'local', model: 'small', audio_duration_seconds: 600, estimated_cost_usd: 0 }, planning: planning('openrouter', 0.42) }, videoSpeed: null })
  assert.doesNotMatch(priced, /Not tracked/)
  assert.match(priced, /\$0\.42/)
  // OpenRouter's own incomplete runs keep their existing wording.
  const partial = render(RunStats, { output, costs: { total_estimated_cost_usd: 0, cost_incomplete: true, planning: planning('openrouter', 0) }, videoSpeed: null })
  assert.match(partial, /Partial: a model didn’t report its price/)
})

// ── Review fixes: the effect, the request, the wizard and first-run setup ──────────────────────

/** The draft store in a known state for the server renderer, which reads the initial state. */
function withDraft(patch, fn) {
  const { useDraftStore } = form.exports
  const original = useDraftStore.getState()
  const initial = useDraftStore.getInitialState()
  const before = Object.fromEntries(Object.keys(patch).map((key) => [key, initial[key]]))
  Object.assign(initial, patch)
  useDraftStore.setState(patch)
  try { return fn() } finally { Object.assign(initial, before); useDraftStore.setState(original) }
}

/**
 * Renders JobForm once, captures the effects it registers, runs them, and reports the draft's workflow
 * afterwards. The server renderer never runs effects, so the component's own useEffect callbacks are
 * collected and called here: this exercises the real effect code, not a copy of its rule.
 */
function workflowAfterEffects(aiProvider, workflow) {
  const { useDraftStore, JobForm } = form.exports
  const captured = []
  const realUseEffect = React.useEffect
  const realWindow = globalThis.window
  React.useEffect = (effect) => { captured.push(effect) }
  globalThis.window = { addEventListener() {}, removeEventListener() {} }
  try {
    return withSettings({ aiProvider }, () => withDraft({ workflow, source: 'https://example.com/video' }, () => {
      render(JobForm, { onSubmit() {} })
      assert.ok(captured.length >= 1, 'JobForm registers its effects')
      for (const effect of captured) {
        const cleanup = effect()
        if (typeof cleanup === 'function') cleanup()
      }
      return useDraftStore.getState().workflow
    }))
  } finally {
    React.useEffect = realUseEffect
    if (realWindow === undefined) delete globalThis.window; else globalThis.window = realWindow
  }
}

test('JobForm keeps the chosen workflow with every provider', () => {
  for (const aiProvider of ['openrouter', 'gemini', 'openai-compatible']) {
    assert.equal(workflowAfterEffects(aiProvider, 'review'), 'review', `${aiProvider}: Review & edit stays`)
    assert.equal(workflowAfterEffects(aiProvider, 'automatic'), 'automatic', `${aiProvider}: Automatic stays`)
    assert.equal(workflowAfterEffects(aiProvider, null), null, `${aiProvider}: no choice stays no choice`)
  }
})

test('an Advanced request with local Whisper carries the planner and no cloud transcription model', () => {
  const { buildJobRequest } = form.exports
  const advanced = { ...DRAFT, clippingMode: 'advanced' }
  const cloud = buildJobRequest(advanced, TRIM, 'openrouter', 'cloud')
  assert.equal(cloud.plannerModel, 'anthropic/claude-opus-5.5')
  assert.equal(cloud.transcriptionModel, 'openai/whisper-large-v3-turbo')
  assert.equal(buildJobRequest(advanced, TRIM, 'openrouter').transcriptionModel, 'openai/whisper-large-v3-turbo', 'cloud is the default')
  const local = buildJobRequest(advanced, TRIM, 'openrouter', 'local')
  assert.equal(local.clippingMode, 'advanced')
  assert.equal(local.plannerModel, 'anthropic/claude-opus-5.5')
  assert.equal(Object.hasOwn(local, 'transcriptionModel'), false, 'a stale cloud model is not sent')
  for (const clippingMode of ['quality', 'economy']) {
    const request = buildJobRequest({ ...DRAFT, clippingMode }, TRIM, 'openrouter', 'local')
    assert.equal(Object.hasOwn(request, 'plannerModel'), false)
    assert.equal(Object.hasOwn(request, 'transcriptionModel'), false)
  }
})

test('the wizard does not ask for, show, or demand a cloud transcription model when Whisper is local', () => {
  const { ClipsStep, JobForm } = form.exports
  const advanced = { ...DRAFT, clippingMode: 'advanced', plannerModel: 'vendor/planner', transcriptionModel: '' }
  const clips = (patch) => withSettings({ openrouterConfigured: true, ...patch }, () => render(ClipsStep, { draft: advanced, update() {} }))
  const cloudClips = clips({})
  assert.match(cloudClips, /Transcription model/)
  assert.equal((cloudClips.match(/role="combobox"/g) ?? []).length, 2)
  const localClips = clips({ transcriptionSource: 'local' })
  assert.doesNotMatch(localClips, /Transcription model/, 'no cloud model picker')
  assert.match(localClips, /Clip planning model/, 'the planning model is still chosen')
  assert.equal((localClips.match(/role="combobox"/g) ?? []).length, 1)

  const review = (patch) => withSettings({ openrouterConfigured: true, ...patch }, () => withDraft({ ...advanced, step: 'review', source: 'https://example.com/video' }, () => render(JobForm, { onSubmit() {} })))
  const cloudReview = review({})
  assert.match(cloudReview, /Choose a model/, 'a cloud run with no transcription model is incomplete')
  assert.match(cloudReview, /Choose both models in Advanced mode/)
  const localReview = review({ transcriptionSource: 'local', localWhisperModel: 'medium' })
  assert.match(localReview, /Local Whisper · medium/, 'the Transcribe row names what really transcribes')
  assert.doesNotMatch(localReview, /Choose a model/)
  assert.doesNotMatch(localReview, /Choose both models in Advanced mode/, 'and does not block the run')
  const aiRow = /Transcribe<\/dt><dd[^>]*>([^<]*)</.exec(localReview)
  assert.equal(aiRow?.[1], 'Local Whisper · medium')
})

test('first-run setup for OpenRouter users is unchanged: no beta checkbox in the setup card, which stays in Settings', () => {
  const { SetupCard, ProviderFields, useApiKeyDrafts } = form.exports
  const card = withSettings({}, () => render(SetupCard, { onOpenSettings() {} }))
  assert.doesNotMatch(card, /Research the source before clipping/)
  assert.doesNotMatch(card, /Beta/)
  const Fields = (props) => React.createElement(ProviderFields, { commit: async () => null, keys: useApiKeyDrafts(), ...props })
  assert.match(withSettings({}, () => render(Fields)), /Research the source before clipping/, 'Settings still shows it')
  assert.doesNotMatch(withSettings({}, () => render(Fields, { compact: true })), /Research the source/)
})

test('the "audio stays on this computer" line is shown only when it is true', () => {
  const { TranscriptionSettings, useApiKeyDrafts } = form.exports
  const Settings = () => React.createElement(TranscriptionSettings, { commit: async () => null, keys: useApiKeyDrafts() })
  const claim = /Audio stays on this computer/
  const html = (patch) => withSettings({ transcriptionSource: 'local', ...patch }, () => render(Settings))
  assert.match(html({ localWhisperMode: 'builtin' }), claim)
  assert.match(html({ localWhisperMode: 'server', localWhisperUrl: 'http://127.0.0.1:8000/v1' }), claim)
  assert.match(html({ localWhisperMode: 'server', localWhisperUrl: 'http://192.168.1.20:8000/v1' }), claim)
  assert.doesNotMatch(html({ localWhisperMode: 'server', localWhisperUrl: '' }), claim, 'no server chosen yet')
  assert.doesNotMatch(html({ localWhisperMode: 'server', localWhisperUrl: 'https://api.openai.com/v1' }), claim, 'a public host would break the promise')
  assert.match(html({ localWhisperMode: 'server', localWhisperUrl: '' }), /on this computer or your private network/)
  assert.doesNotMatch(withSettings({ transcriptionSource: 'cloud' }, () => render(Settings)), claim)
})

test('the Local Whisper models panel and device control appear only for built-in Whisper outside the setup card', () => {
  const { TranscriptionSettings, useApiKeyDrafts } = form.exports
  const Settings = (props) => React.createElement(TranscriptionSettings, { commit: async () => null, keys: useApiKeyDrafts(), ...props })
  const html = (patch, props) => withSettings({ transcriptionSource: 'local', localWhisperMode: 'builtin', ...patch }, () => render(Settings, props))

  const panel = html({ localWhisperDevice: 'cpu' })
  assert.match(panel, /Local Whisper models/)
  assert.match(panel, /Install a model once and it stays on disk/)
  assert.match(panel, /CPU uses system RAM and is slower\./)
  const device = /<div[^>]*role="radiogroup"[^>]*aria-label="Whisper device"[^>]*>(.*?)<\/div>/s.exec(panel)
  assert.ok(device, 'the device control is a radiogroup')
  const radios = device[1].match(/<button[^>]*role="radio"[^>]*>/g) ?? []
  assert.equal(radios.length, 3)
  assert.deepEqual(radios.map((radio) => /aria-checked="(\w+)"/.exec(radio)[1]), ['false', 'true', 'false'], 'CPU is the one checked')
  assert.equal(radios.filter((radio) => radio.includes('tabindex="0"')).length, 1)
  assert.match(device[1], />Auto</)
  assert.match(device[1], />GPU</)

  assert.match(html({ localWhisperDevice: 'auto' }), /Auto uses the GPU when it works and falls back to the CPU\./)
  assert.match(html({ localWhisperDevice: 'gpu' }), /GPU is faster but needs enough video memory and the CUDA libraries\./)

  assert.doesNotMatch(html({}, { compact: true }), /Local Whisper models/, 'the setup card stays short')
  assert.doesNotMatch(html({ localWhisperMode: 'server', localWhisperUrl: 'http://127.0.0.1:8000/v1' }), /Local Whisper models/)
  assert.doesNotMatch(html({ transcriptionSource: 'cloud' }), /Local Whisper models/)
})

test('a custom endpoint shows its wait time, and warns about Ollama\'s short default context only for Ollama', () => {
  const { ProviderFields, useApiKeyDrafts } = form.exports
  const Fields = () => React.createElement(ProviderFields, { commit: async () => null, keys: useApiKeyDrafts() })
  const html = (patch) => withSettings({ aiProvider: 'openai-compatible', openaiCompatibleModel: 'llama3.1', ...patch }, () => render(Fields))
  const ollama = html({ openaiCompatibleUrl: 'http://localhost:11434/v1' })
  assert.match(ollama, /OLLAMA_CONTEXT_LENGTH=32768/)
  assert.match(ollama, /Wait for a reply/)
  assert.match(ollama, /30 minutes/, 'the default wait is shown')
  assert.doesNotMatch(html({ openaiCompatibleUrl: 'http://localhost:1234/v1' }), /OLLAMA_CONTEXT_LENGTH/, 'LM Studio')
  assert.doesNotMatch(html({ openaiCompatibleUrl: 'http://localhost:3001/v1' }), /OLLAMA_CONTEXT_LENGTH/)
  assert.match(html({ openaiCompatibleUrl: 'http://localhost:3001/v1', openaiCompatibleTimeout: '60' }), /60 minutes/)
  assert.doesNotMatch(withSettings({ aiProvider: 'gemini' }, () => render(Fields)), /Wait for a reply/, 'Gemini has no wait setting')
})

test('Test connection shows one labelled line per check, with a word for each icon', () => {
  const { CheckList } = form.exports
  const html = render(CheckList, { checks: [
    { id: 'server', state: 'ok', message: 'The server answered.' },
    { id: 'key', state: 'skipped', message: 'No key was sent, and the server did not ask for one.' },
    { id: 'model', state: 'ok', message: '“gemini-3.7-flash” answered.' },
    { id: 'json', state: 'failed', message: 'The model answered, but not with the JSON it was asked for.' },
    { id: 'images', state: 'warning', message: 'The image test did not finish, so image support is unknown.' }
  ] })
  for (const label of ['Server', 'Key', 'Model', 'JSON answers', 'Images']) assert.match(html, new RegExp(`>${label}<`))
  for (const word of ['Passed', 'Not needed', 'Failed', 'Unsure']) assert.match(html, new RegExp(`aria-label="${word}"`))
  assert.match(html, /aria-label="Test results"/)
})

test('with Gemini or a custom endpoint the Clips step says why there are no Economy or Advanced presets', () => {
  const { ClipsStep } = form.exports
  for (const aiProvider of ['gemini', 'openai-compatible']) {
    const html = withSettings({ aiProvider, geminiConfigured: true, openaiCompatibleUrl: 'http://localhost:1234/v1', openaiCompatibleModel: 'qwen3' }, () => render(ClipsStep, { draft: DRAFT, update() {} }))
    assert.match(html, /Quality, Economy and Advanced are OpenRouter model presets/, aiProvider)
  }
  const openrouter = withSettings({ aiProvider: 'openrouter' }, () => render(ClipsStep, { draft: DRAFT, update() {} }))
  assert.doesNotMatch(openrouter, /are OpenRouter model presets/)
})

test('the visual evidence option is off limits for a text-only custom model, and says why', () => {
  const box = (html) => html.match(/<input type="checkbox"[^>]*>/)[0]
  const textOnly = withSettings({ loaded: true, aiProvider: 'openai-compatible', openaiCompatibleVision: 'off' }, () => render(form.exports.JevSettings))
  assert.ok(isDisabled(box(textOnly)))
  assert.match(textOnly, /set as text-only, so no frames are sent/)
  const reads = withSettings({ loaded: true, aiProvider: 'openai-compatible', openaiCompatibleVision: 'on' }, () => render(form.exports.JevSettings))
  assert.equal(isDisabled(box(reads)), false)
  assert.doesNotMatch(withSettings({ loaded: true, aiProvider: 'gemini' }, () => render(form.exports.JevSettings)), /text-only/)
  assert.doesNotMatch(withSettings({ loaded: true, aiProvider: 'gemini' }, () => render(form.exports.JevSettings)), /&amp;amp;/)
})
