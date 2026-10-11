const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')

function load(file, mocks = {}) {
  const source = fs.readFileSync(path.join(__dirname, '../../src', file), 'utf8')
  const js = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText
  const module = { exports: {} }
  vm.runInNewContext(js, {
    module, exports: module.exports, require: (id) => mocks[id] ?? require(id), URL, Intl, Date, Set
  })
  return module.exports
}

const zernio = load('shared/zernio.ts')
const sharedPosts = load('shared/zernio-posts.ts')
const posts = load('main/zernio/posts-payload.ts', {
  './client': { sanitizeProviderText: (value) => typeof value === 'string' ? value : undefined },
  '../../shared/zernio': zernio,
  '../../shared/zernio-posts': sharedPosts
})

function request() {
  return {
    attemptId: 'test-attempt-1234',
    clipPath: '/tmp/clip.mp4',
    clipTitle: 'A clip',
    durationMs: 12_000,
    caption: 'A caption',
    targets: [{ platform: 'youtube', accountId: 'abc123' }],
    timing: { mode: 'now' },
    options: { youtube: { title: 'A clip', visibility: 'private', madeForKids: false } }
  }
}

test('posting IPC rejects duplicate accounts, invalid targets and embedded NUL', () => {
  const valid = request()
  assert.equal(posts.parsePostClipRequest(valid).targets.length, 1)
  assert.throws(() => posts.parsePostClipRequest({ ...valid, targets: [...valid.targets, ...valid.targets] }), /duplicate account/)
  assert.throws(() => posts.parsePostClipRequest({ ...valid, targets: [{ platform: 'unknown', accountId: 'abc123' }] }), /account/)
  assert.throws(() => posts.parsePostClipRequest({ ...valid, caption: 'unsafe\0caption' }), /caption/)
  assert.throws(() => posts.parsePostClipRequest({ ...valid, clipPath: '/tmp/clip.mp4\0other' }), /clip/)
})

test('posting IPC preserves privacy and scheduling choices in the provider request', () => {
  const parsed = posts.parsePostClipRequest(request())
  const body = posts.buildCreatePostBody(parsed, { publicUrl: 'https://media.example.test/clip.mp4' })
  assert.equal(body.publishNow, true)
  assert.equal(body.platforms[0].platformSpecificData.visibility, 'private')
  assert.equal(body.platforms[0].platformSpecificData.madeForKids, false)
  assert.equal(body.mediaItems[0].url, 'https://media.example.test/clip.mp4')
})

test('published links are restricted to the selected platform HTTPS hosts', () => {
  assert.equal(posts.isPostUrl('https://www.youtube.com/watch?v=123', 'youtube'), true)
  assert.equal(posts.isPostUrl('https://youtube.com.evil.test/watch?v=123', 'youtube'), false)
  assert.equal(posts.isPostUrl('https://www.linkedin.com/feed/update/123', 'youtube'), false)
  assert.equal(posts.isPostUrl('http://www.youtube.com/watch?v=123', 'youtube'), false)
  assert.equal(posts.isPostUrl('https://user:pass@www.youtube.com/watch?v=123', 'youtube'), false)
})
