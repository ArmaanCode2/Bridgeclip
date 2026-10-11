const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const http = require('node:http')
const os = require('node:os')
const path = require('node:path')
const { buildApp, launchApp } = require('../zernio/support/electron-app.cjs')

/** A model-list server on this computer, recording what it was asked. */
function modelServer(models) {
  const requests = []
  const server = http.createServer((request, response) => {
    requests.push({ url: request.url, authorization: request.headers.authorization ?? null })
    if (request.url === '/v1/models') {
      response.writeHead(200, { 'content-type': 'application/json' })
      response.end(JSON.stringify({ object: 'list', data: models.map((id) => ({ id })) }))
    } else { response.writeHead(404).end() }
  })
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({ server, requests, port: server.address().port })))
}

test('provider setup: a Gemini user finishes from the Create page, and Settings validates, tests and saves providers', { timeout: 180000 }, async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'bridgeclip-providers-e2e-'))
  const userData = path.join(root, 'user-data')
  const settingsFile = path.join(userData, 'settings.json')
  const mock = await modelServer(['qwen2.5:7b', 'llama3.1'])
  const session = await launchApp({ appDir: buildApp(path.join(root, 'app')), userDataDir: userData })
  t.after(async () => { await session.close(); await new Promise((resolve) => mock.server.close(resolve)); fs.rmSync(root, { recursive: true, force: true }) })
  const { app, page } = session
  const errors = []
  page.on('pageerror', (error) => errors.push(error.message))
  const saved = () => JSON.parse(fs.readFileSync(settingsFile, 'utf8'))
  const shots = process.env.BRIDGECLIP_E2E_SHOTS
  const shot = async (name) => { if (shots) { fs.mkdirSync(shots, { recursive: true }); await page.screenshot({ path: path.join(shots, name) }) } }

  // The tools check depends on this machine, so it is not what is under test here.
  await app.evaluate(({ ipcMain }) => {
    ipcMain.removeHandler('system:checkTools')
    ipcMain.handle('system:checkTools', () => ({ python: true, pythonDeps: true, ffmpeg: true, ffmpegCaptions: true, ffprobe: true, ytdlp: true, engine: true, bridgeRunner: true }))
  })
  await page.reload()

  // ── First run on the Create page ──────────────────────────────────────────────
  const card = page.getByRole('region', { name: 'Finish setup' })
  await card.waitFor()
  const providers = card.getByRole('radiogroup', { name: 'AI provider', exact: true })
  assert.equal(await providers.getByRole('radio', { name: 'OpenRouter' }).getAttribute('aria-checked'), 'true')
  await shot('provider-setup-1-first-run.png')

  await providers.getByRole('radio', { name: 'Google Gemini' }).click()
  await card.getByLabel('Google Gemini API key').fill('AIzaE2eSecretKey')
  // The key is saved as you type, but cloud transcription still needs OpenRouter, and the card says so.
  await card.getByText('Cloud transcription uses OpenRouter', { exact: false }).waitFor()
  await card.getByLabel('OpenRouter key for transcription').waitFor()
  await shot('provider-setup-2-gemini-needs-transcription.png')

  await card.getByRole('button', { name: 'Use Local Whisper instead' }).click()
  await card.waitFor({ state: 'detached' })
  await page.getByRole('button', { name: 'Ready to clip' }).waitFor()

  const afterSetup = saved()
  assert.equal(afterSetup.aiProvider, 'gemini')
  assert.equal(afterSetup.transcriptionSource, 'local')
  assert.equal(afterSetup.localWhisperMode, 'builtin')
  assert.equal(fs.readFileSync(settingsFile, 'utf8').includes('AIzaE2eSecretKey'), false, 'the key must not be stored in plain text')
  assert.ok(afterSetup.geminiApiKey && afterSetup.geminiApiKey.scheme, 'the key is stored sealed')

  // The wizard offers no Review & edit for Gemini, and says why.
  const workflow = page.getByRole('radiogroup', { name: 'Workflow', exact: true })
  assert.equal(await workflow.getByRole('radio', { name: 'Review & edit' }).isDisabled(), true)
  assert.equal(await workflow.getByRole('radio', { name: 'Automatic' }).isDisabled(), false)
  await workflow.getByText('OpenRouter only').waitFor()

  // ── Settings ──────────────────────────────────────────────────────────────────
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+,' : 'Control+,')
  const picker = page.getByRole('radiogroup', { name: 'AI provider', exact: true })
  await picker.waitFor()
  assert.equal(await picker.getByRole('radio', { name: /Google Gemini/ }).getAttribute('aria-checked'), 'true')
  await page.getByText('Jev runs on OpenRouter, so it is not used while Google Gemini picks your clips', { exact: false }).waitFor()

  await picker.getByRole('radio', { name: /OpenAI-compatible/ }).click()
  const url = page.getByLabel('Endpoint URL', { exact: true })
  const model = page.getByLabel('Model name', { exact: true })

  // An unsafe URL is explained under the field and is never saved.
  await url.fill('http://example.com/v1')
  await url.press('Tab')
  await page.getByText('Use https:// for servers outside your own network.').waitFor()
  assert.equal(saved().openaiCompatibleUrl, '')
  await url.fill('http://169.254.169.254/latest')
  await url.press('Tab')
  await page.getByText('That address is not allowed for a provider.').waitFor()
  assert.equal(saved().openaiCompatibleUrl, '')

  // A good URL and model save, and Test connection reaches the server with no key when none is set.
  await url.fill(`http://127.0.0.1:${mock.port}/v1`)
  await url.press('Tab')
  await model.fill('llama3.1')
  await model.press('Tab')
  await page.getByRole('button', { name: 'Test connection' }).click()
  await page.getByText('Connected. 2 models available.').waitFor()
  const call = mock.requests.find((entry) => entry.url === '/v1/models')
  assert.ok(call, 'the server was asked for its models')
  assert.equal(call.authorization, null)
  assert.deepEqual(await page.locator('datalist option').evaluateAll((options) => options.map((option) => option.value)), ['qwen2.5:7b', 'llama3.1'])
  assert.equal(saved().openaiCompatibleUrl, `http://127.0.0.1:${mock.port}/v1`)
  assert.equal(saved().openaiCompatibleModel, 'llama3.1')

  await model.fill('missing-model')
  await model.press('Tab')
  await page.getByRole('button', { name: 'Test connection' }).click()
  // A model missing from the list is a spelling hint, not a failure: /models is not authoritative.
  await page.getByText('Connected. Model "missing-model" is not in the server’s list, so check the spelling.').waitFor()

  // The vision switch is saved, and read back after a reload.
  await page.getByRole('switch', { name: 'This model can read images' }).click()
  await page.waitForFunction(() => document.querySelector('[role="switch"][aria-label="This model can read images"]')?.getAttribute('aria-checked') === 'true')
  assert.equal(saved().openaiCompatibleVision, 'on')
  await shot('provider-setup-3-settings-openai-compatible.png')

  // ── Local Whisper ─────────────────────────────────────────────────────────────
  const source = page.getByRole('radiogroup', { name: 'Transcription source', exact: true })
  assert.equal(await source.getByRole('radio', { name: 'Local Whisper' }).getAttribute('aria-checked'), 'true')
  await page.getByRole('button', { name: 'Check Whisper' }).click()
  // Whatever this machine has, the answer arrives as a readable sentence, never an error dialog.
  await page.getByRole('status').filter({ hasText: /Whisper|Python|whisper/ }).first().waitFor({ timeout: 40000 })

  await page.getByRole('radiogroup', { name: 'Local Whisper type', exact: true }).getByRole('radio', { name: 'Server' }).click()
  await page.getByRole('button', { name: 'Check server' }).click()
  await page.getByText('Enter your Whisper server URL first.').waitFor()
  await page.getByLabel('Whisper server URL', { exact: true }).fill(`http://127.0.0.1:${mock.port}/v1`)
  await page.getByLabel('Whisper server URL', { exact: true }).press('Tab')
  await page.getByRole('button', { name: 'Check server' }).click()
  await page.getByText('The server answered.').waitFor()
  assert.equal(saved().localWhisperMode, 'server')
  assert.equal(saved().localWhisperUrl, `http://127.0.0.1:${mock.port}/v1`)
  // Back to the built-in Whisper: the model moves to one the built-in engine knows.
  await page.getByRole('radiogroup', { name: 'Local Whisper type', exact: true }).getByRole('radio', { name: 'Built-in' }).click()
  await page.getByRole('radiogroup', { name: 'Whisper model', exact: true }).waitFor()
  await shot('provider-setup-4-settings-whisper.png')

  assert.deepEqual(errors, [])
})
