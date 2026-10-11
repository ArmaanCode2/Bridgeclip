'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { buildSync } = require('esbuild')
const React = require('react')
const { renderToStaticMarkup } = require('react-dom/server')
const editFixture = require('../fixtures/editorial/edit-audit.json')
const { fileLinksAvailable } = require('../support/symlinks.cjs')

function load(entry) {
  const code = buildSync({ entryPoints: [path.resolve(__dirname, '../../', entry)], bundle: true, jsx: 'automatic',
    platform: 'node', format: 'cjs', packages: 'external', write: false }).outputFiles[0].text
  const mod = { exports: {} }
  new Function('module', 'exports', 'require', code)(mod, mod.exports, require)
  return mod.exports
}

test('runs without a saved transcript get a friendly message, and no error reveals a path', async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'edit-inspector-'))
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))
  const run = path.join(root, 'private-run-folder')
  fs.mkdirSync(run)
  const { inspectEdits } = load('src/main/edit-inspector.ts')
  const noPath = (error) => !error.message.includes(root) && !error.message.includes('private-run-folder')

  await assert.rejects(inspectEdits(run, root), (error) => /^No transcript saved/.test(error.message) && noPath(error))
  fs.mkdirSync(path.join(run, 'transcript.json'))
  await assert.rejects(inspectEdits(run, root), (error) => /transcript could not be read/.test(error.message) && noPath(error))
  fs.rmdirSync(path.join(run, 'transcript.json'))
  if (fileLinksAvailable) {
    fs.writeFileSync(path.join(root, 'elsewhere.json'), '{"segments":[]}')
    fs.symlinkSync(path.join(root, 'elsewhere.json'), path.join(run, 'transcript.json'), 'file')
    await assert.rejects(inspectEdits(run, root), (error) => /transcript could not be read/.test(error.message) && noPath(error))
    fs.unlinkSync(path.join(run, 'transcript.json'))
  }
  fs.writeFileSync(path.join(run, 'transcript.json'), JSON.stringify({ segments: [{ start_time_ms: 0, end_time_ms: 'soon', text: 'Words' }] }))
  await assert.rejects(inspectEdits(run, root), (error) => /Invalid saved transcript/.test(error.message) && noPath(error))
  await assert.rejects(inspectEdits(path.join(root, 'missing-run'), root), (error) => noPath(error))
})

test('research citations are shown as selectable text, never as buttons that try to open them', () => {
  const { parseEditAudit } = load('src/shared/editorial.ts')
  const { RecordedEditView } = load('src/renderer/components/EditInspector.tsx')
  const audit = parseEditAudit(editFixture)
  audit.source_context = { status: 'completed', research_status: 'completed', created_at: '2026-09-28T00:00:00.000Z',
    source: { title: 'Source title', channel: 'Channel' }, brief: null, cost_usd: 0, cost_incomplete: false, requests: [],
    citations: [{ title: 'A cited article', url: 'https://example.com/article' }] }
  const html = renderToStaticMarkup(React.createElement(RecordedEditView, { audit }))
  const sources = html.slice(html.indexOf('Research sources'))
  assert.ok(sources.includes('A cited article') && sources.includes('https://example.com/article'))
  assert.ok(!/<button/.test(sources.slice(0, sources.indexOf('</ul>'))), 'no clickable citation')
  assert.match(sources, /data-selectable/)
})
