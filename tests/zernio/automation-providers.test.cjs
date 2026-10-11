'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { loadMain, tempDir, fakeElectron } = require('./support/load-main.cjs')
const { createMockZernio } = require('./support/mock-zernio.cjs')
const ENTRY = "export * as automations from './src/main/automations'; export * as settings from './src/main/settings-store'"

test('a custom endpoint on localhost writes automation titles and captions with no OpenRouter key; web research is skipped', async () => {
  const { dir, cleanup } = tempDir('bridgeclip-own-model-metadata-')
  const requests = []
  const mock = await createMockZernio({ extraRoutes: [{ method: 'POST', path: '/v1/chat/completions', auth: false, handler: (ctx) => {
    requests.push(ctx.body)
    const prompt = ctx.body.messages.at(-1).content
    const input = JSON.parse(prompt.slice(0, prompt.indexOf('\n\nReply with one JSON value')))
    // A local reasoning model: thinks first, then answers.
    const clips = input.clips.map((clip) => ({ id: clip.id, posts: [{ platform: 'youtube', title: 'Testing catches bugs', caption: clip.transcript, evidence: clip.transcript, tags: [], categoryId: '28', topicTag: null }] }))
    ctx.json(200, { choices: [{ message: { content: `<think>pick {titles}</think>${JSON.stringify({ clips })}` } }] })
  } }] })
  try {
    const { electron } = fakeElectron(dir)
    let main = loadMain(ENTRY, { electron })
    main.settings.replaceApiKey('zernioApiKey', 'test-workspace')
    const library = path.join(dir, 'library'); fs.mkdirSync(library)
    main.settings.savePublicSettings({
      outputDirectory: library, pythonPath: 'python3', aiProvider: 'openai-compatible', openaiCompatibleUrl: `${mock.url}/v1`,
      openaiCompatibleModel: 'qwen3', transcriptionSource: 'local', localWhisperMode: 'builtin'
    })
    const clip = path.join(library, 'clip.mp4'); fs.writeFileSync(clip, 'clip bytes')
    const source = { title: 'Workshop', description: '', channel: '', url: null, videoId: 'c'.repeat(64) }
    const [created] = main.automations.createAutomation('Own model')
    await main.automations.addAutomationContent(created.id, [clip, clip], ['One', 'Two'], source)
    const storeFile = path.join(dir, 'userData', fs.readdirSync(path.join(dir, 'userData')).find((name) => /^automations-.*\.json$/.test(name)))
    const stored = JSON.parse(fs.readFileSync(storeFile, 'utf8'))
    stored.automations[0].content.forEach((item, index) => { item.transcript = `Tests catch bugs early, example ${index}.` })
    fs.writeFileSync(storeFile, JSON.stringify(stored))
    main = loadMain(ENTRY, { electron })
    const [group] = await main.automations.automationEnhancementGroups(created.id)
    const result = await main.automations.enhanceAutomationBatch(created.id, group.contentIds, group.key, 'Focus on testing tips.')
    assert.equal(result.completed, 2)
    assert.ok(requests.length >= 1)
    for (const body of requests) {
      assert.equal(body.model, 'qwen3', 'the saved model writes the metadata')
      assert.equal(body.tools, undefined, 'no OpenRouter web search')
      assert.equal(body.provider, undefined, 'no OpenRouter routing fields')
      assert.match(body.messages.at(-1).content, /JSON Schema/)
    }
    const item = result.automations[0].content[0]
    assert.equal(item.metadataDraft.posts[0].title, 'Testing catches bugs')
    assert.match(item.metadataDraft.research.summary, /Web research uses OpenRouter, so it is off with OpenAI-compatible/)
  } finally {
    await mock.close(); cleanup()
  }
})

test('enabling AI metadata needs the chosen provider and a way to transcribe, not an OpenRouter key', async () => {
  const { dir, cleanup } = tempDir('bridgeclip-own-model-gate-')
  const mock = await createMockZernio({ apiKey: 'test-workspace' })
  const previousUrl = process.env.BRIDGECLIP_ZERNIO_API_URL
  process.env.BRIDGECLIP_ZERNIO_API_URL = mock.apiUrl
  try {
    const { electron } = fakeElectron(dir)
    const main = loadMain(ENTRY, { electron })
    main.settings.replaceApiKey('zernioApiKey', 'test-workspace')
    main.settings.replaceApiKey('geminiApiKey', 'AIza-test')
    main.settings.savePublicSettings({ outputDirectory: path.join(dir, 'library'), pythonPath: 'python3', aiProvider: 'gemini', transcriptionSource: 'cloud' })
    const [profile] = mock.state.profiles
    const youtube = mock.addAccount('youtube', profile._id)
    const [created] = main.automations.createAutomation('Gemini captions')
    const update = {
      name: created.name, enabled: true, profileId: profile._id, metadataMode: 'ai', timezone: 'UTC', times: ['09:00'],
      youtubeVisibility: 'unlisted', youtubeMadeForKids: false, accounts: [{ platform: 'youtube', accountId: youtube._id }]
    }
    // Cloud transcription still needs OpenRouter; the message says how to avoid it.
    await assert.rejects(main.automations.updateAutomation(created.id, update), /switch Speech to text to Local Whisper/)
    main.settings.savePublicSettings({ transcriptionSource: 'local', localWhisperMode: 'builtin' })
    const [saved] = await main.automations.updateAutomation(created.id, update)
    assert.equal(saved.metadataMode, 'ai', 'Gemini with Local Whisper can write captions, with no OpenRouter key')
  } finally {
    if (previousUrl === undefined) delete process.env.BRIDGECLIP_ZERNIO_API_URL; else process.env.BRIDGECLIP_ZERNIO_API_URL = previousUrl
    await mock.close(); cleanup()
  }
})
