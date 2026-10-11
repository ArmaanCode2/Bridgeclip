'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const vm = require('node:vm')
const { buildSync } = require('esbuild')

const bundle = buildSync({
  stdin: {
    contents: "export * from './src/renderer/store/use-posts-store'",
    resolveDir: path.resolve(__dirname, '../..'),
    loader: 'ts'
  },
  bundle: true, platform: 'node', format: 'cjs', packages: 'external', write: false, logLevel: 'silent'
}).outputFiles[0].text

function deferred() {
  let resolve
  const promise = new Promise((done) => { resolve = done })
  return { promise, resolve }
}

function load(postsApi) {
  const module = { exports: {} }
  const window = { bridgeclip: { zernio: { posts: postsApi } } }
  vm.runInNewContext(bundle, { module, exports: module.exports, require, window, Date, Promise, console })
  const store = module.exports.usePostsStore
  return () => store.getState()
}

const post = (id) => ({ id, clipTitle: id, status: 'scheduled', targets: [] })

test('a pending initial list cannot replace a newly created post', async () => {
  const list = deferred()
  const state = load({ list: () => list.promise })
  const loading = state().load()
  state().upsert(post('new'))
  list.resolve([post('old')])
  await loading
  assert.equal(state().posts.map((entry) => entry.id).join(','), 'new')
  assert.equal(state().loaded, true)
})

test('a pending refresh cannot replace a newly created post', async () => {
  const refresh = deferred()
  const state = load({ refresh: () => refresh.promise })
  const refreshing = state().refresh()
  state().upsert(post('new'))
  refresh.resolve({ posts: [post('old')], error: null })
  await refreshing
  assert.equal(state().posts.map((entry) => entry.id).join(','), 'new')
  assert.equal(state().refreshing, false)
})

test('overlapping actions update their own records without undoing another action', async () => {
  const cancelled = deferred()
  const rescheduled = deferred()
  const state = load({ cancel: () => cancelled.promise, reschedule: () => rescheduled.promise })
  state().upsert(post('first'))
  state().upsert(post('second'))
  const cancel = state().cancel('first')
  const reschedule = state().reschedule('second', '2030-01-01T12:00:00Z', 'UTC')
  rescheduled.resolve([{ ...post('first') }, { ...post('second'), scheduledFor: '2030-01-01T12:00:00Z' }])
  await reschedule
  cancelled.resolve([{ ...post('first'), status: 'cancelled' }, post('second')])
  await cancel
  assert.equal(state().posts.find((entry) => entry.id === 'first').status, 'cancelled')
  assert.equal(state().posts.find((entry) => entry.id === 'second').scheduledFor, '2030-01-01T12:00:00Z')
})
