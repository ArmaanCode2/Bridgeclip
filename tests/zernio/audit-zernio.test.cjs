'use strict'
// Regressions from the Zernio/automation security audit. Everything runs
// against the local mock Zernio; no real key, account or provider is involved.
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const crypto = require('node:crypto')
const { execFileSync } = require('node:child_process')
const { createMockZernio } = require('./support/mock-zernio.cjs')
const { createPostingMock } = require('./support/mock-posts.cjs')
const { loadMain, tempDir, fakeElectron, ROOT } = require('./support/load-main.cjs')

const { fileLinksAvailable } = require('../support/symlinks.cjs')

const KEY = 'audit-zernio-key'
const FFMPEG = fs.existsSync(path.join(ROOT, 'engine-bin', 'ffmpeg')) ? path.join(ROOT, 'engine-bin', 'ffmpeg') : 'ffmpeg'
const workspace = (key) => crypto.createHash('sha256').update(key).digest('hex')

function makeClip(file) {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  execFileSync(FFMPEG, ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=blue:s=360x640:d=4:r=15',
    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=4', '-shortest', '-c:v', 'mpeg4', '-q:v', '8',
    '-c:a', 'aac', '-movflags', '+faststart', file])
  return file
}

// ZER-1: the upload/request-id journal must not write through a predictable
// temp path. A symlink planted at `<journal>.tmp` used to redirect the write.
test('the post attempt journal ignores a planted temp-path symlink and is written exclusively', { skip: !fileLinksAvailable }, async () => {
  const { dir, cleanup } = tempDir('bridgeclip-audit-attempts-')
  const posting = createPostingMock()
  const mock = await createMockZernio({ apiKey: KEY, extraRoutes: posting.routes })
  const previousUrl = process.env.BRIDGECLIP_ZERNIO_API_URL
  const previousPath = process.env.PATH
  process.env.BRIDGECLIP_ZERNIO_API_URL = mock.apiUrl
  process.env.PATH = `${previousPath}${path.delimiter}${path.join(ROOT, 'engine-bin')}`
  try {
    const library = path.join(dir, 'library')
    const clipPath = makeClip(path.join(library, 'job-1', 'clip_01.mp4'))
    const { electron } = fakeElectron(dir)
    const main = loadMain("export * as posts from './src/main/zernio/posts'; export * as settings from './src/main/settings-store'", { electron })
    main.settings.replaceApiKey('zernioApiKey', KEY)
    main.settings.savePublicSettings({ outputDirectory: library, pythonPath: 'python3' })
    const [profile] = mock.state.profiles
    const youtube = mock.addAccount('youtube', profile._id, { username: 'channel' })

    const userData = path.join(dir, 'userData')
    const journal = path.join(userData, `zernio-post-attempts-${workspace(KEY)}.json`)
    const victim = path.join(dir, 'victim.txt')
    fs.writeFileSync(victim, 'original victim content')
    fs.symlinkSync(victim, `${journal}.tmp`)

    const result = await main.posts.publishClip({
      attemptId: 'attempt-1234-abcd', clipPath, clipTitle: 'Audit', durationMs: 4000, caption: 'Audit',
      targets: [{ platform: 'youtube', accountId: youtube._id }], timing: { mode: 'now' },
      options: { youtube: { title: 'Audit', visibility: 'public', madeForKids: false } }
    }, () => {})
    assert.equal(result.outcome, 'published')

    assert.equal(fs.readFileSync(victim, 'utf8'), 'original victim content', 'nothing is written through the planted symlink')
    assert.equal(fs.lstatSync(`${journal}.tmp`).isSymbolicLink(), true, 'the planted link is left alone')
    assert.equal(fs.lstatSync(journal).isSymbolicLink(), false)
    if (process.platform !== 'win32') assert.equal((fs.statSync(journal).mode & 0o777).toString(8), '600')
    const saved = JSON.parse(fs.readFileSync(journal, 'utf8'))
    assert.equal(saved.workspace, workspace(KEY))
    assert.equal(fs.readdirSync(userData).filter((name) => /\.tmp$/.test(name)).length, 1, 'no random temp files are left behind')
  } finally {
    if (previousUrl === undefined) delete process.env.BRIDGECLIP_ZERNIO_API_URL
    else process.env.BRIDGECLIP_ZERNIO_API_URL = previousUrl
    process.env.PATH = previousPath
    await mock.close()
    cleanup()
  }
})

// ZER-2: a transcript can be prompt-injected. The generated caption is the
// one field that reaches the post, so links must not slip in without a scheme.
test('AI captions are refused when they carry a clickable link, with or without a scheme', () => {
  const { dir, cleanup } = tempDir('bridgeclip-audit-metadata-')
  try {
    const { electron } = fakeElectron(dir)
    const { parseGeneratedMetadata } = loadMain("export { parseGeneratedMetadata } from './src/main/automation-metadata'", { electron })
    const transcript = 'today we talk about testing agents and you can claim your prize at www.evil.example slash win'
    const parse = (caption) => parseGeneratedMetadata(
      { posts: [{ platform: 'linkedin', caption, title: null, tags: [], categoryId: null, topicTag: null, evidence: 'testing agents' }] },
      ['linkedin'], transcript
    )
    for (const caption of [
      'Claim your prize at https://evil.example/win',
      'Claim your prize at www.evil.example/win',
      'Claim your prize at WWW.EVIL.EXAMPLE',
      'Claim your prize at ftp://evil.example',
      'Claim your prize at hxxp://evil.example'
    ]) assert.throws(() => parse(caption), /AI metadata for linkedin was invalid/, caption)
    assert.equal(parse('Testing agents matters, e.g. with node.js and a 3:1 ratio.')[0].caption, 'Testing agents matters, e.g. with node.js and a 3:1 ratio.')
  } finally { cleanup() }
})

// B10: bare domains and @handles are auto-linked too, and titles and tags reach the post as well.
test('AI metadata refuses bare domains and handles that were not spoken, in every posted field', () => {
  const { dir, cleanup } = tempDir('bridgeclip-audit-handles-')
  try {
    const { electron } = fakeElectron(dir)
    const { parseGeneratedMetadata } = loadMain("export { parseGeneratedMetadata } from './src/main/automation-metadata'", { electron })
    const transcript = 'Today we talk about testing agents. Our docs live at bridgemind.ai and you can follow @bridgemind for updates.'
    const post = (platform, fields) => ({ platform, caption: 'Testing agents matters.', title: null, tags: [], categoryId: null, topicTag: null, evidence: 'testing agents', ...fields })
    const parse = (platform, fields, context) => parseGeneratedMetadata({ posts: [post(platform, fields)] }, [platform], transcript, context)
    for (const caption of [
      'Claim your prize at scamcoin.io', 'Details: evil.com/x', 'Wow...scamcoin.io', 'Mail me@evil.com', 'Follow @scammer for more',
      'Visit ｅｖｉｌ．ｃｏｍ', 'Visit evil\u200B.com', 'Try install.sh today', 'Read #evil.com', 'Go to Sub.Evil.Example.'
    ]) assert.throws(() => parse('linkedin', { caption }), /link or @mention that is not in the transcript/, caption)
    const youtube = { title: 'Testing agents', tags: ['testing'], categoryId: '28' }
    assert.throws(() => parse('youtube', { ...youtube, title: 'Testing agents at scamcoin.io' }), /youtube included a link/)
    assert.throws(() => parse('youtube', { ...youtube, tags: ['testing', 'scamcoin.io'] }), /youtube included a link/)
    assert.throws(() => parse('youtube', { ...youtube, tags: ['@scammer'] }), /youtube included a link/)
    assert.throws(() => parse('facebook', { title: 'Follow @scammer' }, { facebookFormat: 'reel' }), /facebook included a link/)
    for (const caption of [
      'Testing agents matters, e.g. with Node.js, Next.js and a 3.5x speedup in v2.0 for the U.S. team.',
      'Docs: bridgemind.ai, and follow @BridgeMind.', 'Wait... testing agents matters.', 'テストは大切です。次はエージェントです。',
      'Costs $2.99. Ph.D. level testing.'
    ]) assert.equal(parse('linkedin', { caption })[0].caption, caption, caption)
    assert.equal(parse('youtube', { ...youtube, title: 'Testing agents with bridgemind.ai' })[0].title, 'Testing agents with bridgemind.ai')
    const started = performance.now()
    for (const caption of ['a.'.repeat(30000), '-'.repeat(60000) + 'a', '.'.repeat(60000), '@'.repeat(60000)]) {
      try { parse('facebook', { caption }) } catch { /* Only the time matters. */ }
    }
    assert.ok(performance.now() - started < 2000, 'untrusted repetition must not stall the main process')
  } finally { cleanup() }
})

test('metadata validation bounds repetitive provider evidence and caption parsing', () => {
  const { dir, cleanup } = tempDir('bridgeclip-audit-evidence-')
  try {
    const { electron } = fakeElectron(dir)
    const { evidenceInTranscript, parseGeneratedMetadata } = loadMain("export { evidenceInTranscript, parseGeneratedMetadata } from './src/main/automation-metadata'", { electron })
    const started = performance.now()
    assert.equal(evidenceInTranscript('word '.repeat(10000), 'word '.repeat(4000)), false)
    assert.equal(evidenceInTranscript('word '.repeat(79) + 'different', 'word '.repeat(4000)), false)
    const caption = 'a'.repeat(60000)
    const result = parseGeneratedMetadata({ posts: [{ platform: 'facebook', caption, evidence: 'A useful spoken phrase' }] }, ['facebook'], 'A useful spoken phrase')
    assert.equal(result[0].caption, caption)
    assert.ok(performance.now() - started < 2000, 'untrusted repetition must not stall the main process')
  } finally { cleanup() }
})
