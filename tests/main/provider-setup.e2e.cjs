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
    // Whether Claude Code is here depends on this machine too.
    ipcMain.removeHandler('assistant:status')
    ipcMain.handle('assistant:status', () => [{ id: 'claude', state: 'connected', version: '2.0.0', account: null, plan: 'max', detail: null, checkedAt: '' }])
  })
  await page.reload()

  // ── First run on the Create page ──────────────────────────────────────────────
  const card = page.getByRole('region', { name: 'Finish setup' })
  await card.waitFor()
  const providers = card.getByRole('radiogroup', { name: 'AI provider', exact: true })
  assert.equal(await providers.getByRole('radio', { name: 'OpenRouter' }).getAttribute('aria-checked'), 'true')
  await shot('provider-setup-1-first-run.png')

  // The Claude plan needs no key, so it can finish setup on the spot. The card stays open until Done,
  // and keyboard choosing keeps working: arrows move on from it to the next card.
  await providers.getByRole('radio', { name: 'OpenRouter' }).focus()
  await page.keyboard.press('ArrowLeft')
  await card.getByText('Signed in · Max').waitFor()
  await card.getByRole('button', { name: 'Done' }).waitFor()
  assert.equal(saved().aiProvider, 'claude-plan')
  assert.equal(saved().transcriptionSource, 'local')
  await page.keyboard.press('ArrowRight')
  await page.waitForFunction(() => document.querySelector('[role="radio"][aria-checked="true"]')?.textContent.includes('OpenRouter'))
  assert.equal(saved().transcriptionSource, 'cloud')

  await providers.getByRole('radio', { name: 'Google Gemini' }).click()
  // With no OpenRouter key, choosing Gemini moves speech to text to Local Whisper, and the card says so.
  await card.getByText('Transcription now uses Local Whisper on this computer', { exact: false }).waitFor()
  await shot('provider-setup-2-gemini-local-transcription.png')
  // The key is saved as you type; with it, setup is complete and the card goes away.
  await card.getByLabel('Google Gemini API key').fill('AIzaE2eSecretKey')
  await card.waitFor({ state: 'detached' })
  await page.getByRole('button', { name: 'Ready to clip' }).waitFor()

  const afterSetup = saved()
  assert.equal(afterSetup.aiProvider, 'gemini')
  assert.equal(afterSetup.transcriptionSource, 'local')
  assert.equal(afterSetup.localWhisperMode, 'builtin')
  assert.equal(fs.readFileSync(settingsFile, 'utf8').includes('AIzaE2eSecretKey'), false, 'the key must not be stored in plain text')
  assert.ok(afterSetup.geminiApiKey && afterSetup.geminiApiKey.scheme, 'the key is stored sealed')

  // Review & edit runs on every provider: with Gemini, Gemini answers the review questions.
  const workflow = page.getByRole('radiogroup', { name: 'Workflow', exact: true })
  assert.equal(await workflow.getByRole('radio', { name: 'Review & edit' }).isDisabled(), false)
  assert.equal(await workflow.getByRole('radio', { name: 'Automatic' }).isDisabled(), false)

  // ── Settings ──────────────────────────────────────────────────────────────────
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+,' : 'Control+,')
  const picker = page.getByRole('radiogroup', { name: 'AI provider', exact: true })
  await picker.waitFor()
  assert.equal(await picker.getByRole('radio', { name: /Google Gemini/ }).getAttribute('aria-checked'), 'true')
  await page.getByText('Review & edit and the AI review of automatic clips use this model in place of Jev', { exact: false }).first().waitFor()

  // A custom endpoint is saved as a model card. An unsafe URL is explained in the form and never saved.
  await page.getByRole('button', { name: 'Add model' }).click()
  const adding = page.getByRole('group', { name: 'Add a model' })
  const url = adding.getByLabel('Endpoint URL', { exact: true })
  await url.fill('http://example.com/v1')
  await adding.getByLabel('Model name', { exact: true }).fill('llama3.1')
  await adding.getByRole('button', { name: 'Save model' }).click()
  await adding.getByText('Use https:// for servers outside your own network.').waitFor()
  assert.deepEqual(saved().modelPresets ?? [], [])
  await url.fill('http://169.254.169.254/latest')
  await adding.getByRole('button', { name: 'Save model' }).click()
  await adding.getByText('That address is not allowed for a provider.').waitFor()
  assert.deepEqual(saved().modelPresets ?? [], [])

  // A good URL saves the card. Gemini works, so the new card waits to be clicked.
  await url.fill(`http://127.0.0.1:${mock.port}/v1`)
  await adding.getByRole('button', { name: 'Save model' }).click()
  const custom = picker.getByRole('radio', { name: new RegExp(`llama3\\.1 · 127\\.0\\.0\\.1:${mock.port}`) })
  await custom.waitFor()
  // The form stays open on the new card, ready to test it; close it for now.
  const justAdded = page.getByRole('group', { name: /^Edit / })
  await justAdded.getByText('Saved as a card. Test it below, or close this.').waitFor()
  await justAdded.getByRole('button', { name: 'Close' }).click()
  await justAdded.waitFor({ state: 'detached' })
  assert.equal(await custom.getAttribute('aria-checked'), 'false')
  assert.equal(saved().modelPresets.length, 1)
  assert.equal(saved().aiProvider, 'gemini')

  // Clicking the card ticks it, and that model plans clips from now on.
  await custom.click()
  await page.waitForFunction(() => [...document.querySelectorAll('[role="radio"][aria-checked="true"]')].some((radio) => radio.textContent.includes('llama3.1')))
  assert.equal(saved().aiProvider, 'openai-compatible')
  assert.equal(saved().activeModelPreset, saved().modelPresets[0].id)
  assert.equal(saved().openaiCompatibleUrl, `http://127.0.0.1:${mock.port}/v1`)
  assert.equal(saved().openaiCompatibleModel, 'llama3.1')

  // Test connection tests the chosen card's endpoint, with no key when none is set. This mock lists
  // models but answers no chat request, so the model check fails with a plain reason.
  await page.getByRole('button', { name: 'Test connection' }).click()
  await page.getByText('The server answered and lists 2 models.').waitFor()
  const call = mock.requests.find((entry) => entry.url === '/v1/models')
  assert.ok(call, 'the server was asked for its models')
  assert.equal(call.authorization, null)
  assert.ok(mock.requests.some((entry) => entry.url === '/v1/chat/completions'), 'the saved model was asked to answer')

  // Editing the card: its own Test lists the server's models for the model field.
  await page.getByRole('button', { name: /^Edit llama3\.1/ }).click()
  const editing = page.getByRole('group', { name: /^Edit / })
  await editing.getByRole('button', { name: 'Test connection' }).click()
  await editing.getByText('The server answered and lists 2 models.').waitFor()
  assert.deepEqual(await editing.locator('datalist option').evaluateAll((options) => options.map((option) => option.value)), ['qwen2.5:7b', 'llama3.1'])
  await editing.getByLabel('Model name', { exact: true }).fill('missing-model')
  await editing.getByRole('button', { name: 'Save changes' }).click()
  await editing.waitFor({ state: 'detached' })
  assert.equal(saved().openaiCompatibleModel, 'missing-model')
  await page.getByRole('button', { name: 'Test connection' }).click()
  await page.getByText('Model “missing-model” was not found. Pick one from the list.').first().waitFor()

  // Image reading is part of the card, saved with it.
  await page.getByRole('button', { name: /^Edit missing-model/ }).click()
  await page.getByRole('group', { name: /^Edit / }).getByRole('switch', { name: 'This model can read images' }).click()
  await page.getByRole('group', { name: /^Edit / }).getByRole('button', { name: 'Save changes' }).click()
  await page.getByText('Reads images').waitFor()
  assert.equal(saved().modelPresets[0].vision, 'on')
  assert.equal(saved().openaiCompatibleVision, 'on')
  await shot('provider-setup-3-settings-openai-compatible.png')

  // ── Claude plan ───────────────────────────────────────────────────────────────
  // The first card. It needs no key: choosing it offers the model, effort and image reading.
  const claudePlan = picker.getByRole('radio').first()
  assert.match(await claudePlan.textContent(), /Claude plan/)
  await claudePlan.click()
  await page.getByText('Signed in · Max').waitFor()
  assert.equal(saved().aiProvider, 'claude-plan')
  assert.equal(await page.getByRole('switch', { name: 'Read images with my Claude plan' }).getAttribute('aria-checked'), 'false')
  await page.getByLabel('Model', { exact: true }).click()
  await page.getByRole('option', { name: 'Opus' }).click()
  await page.getByText('The strongest picks. Uses your plan the fastest.').waitFor()
  await page.getByLabel('Effort', { exact: true }).click()
  await page.getByRole('option', { name: 'Extra high' }).click()
  await page.getByRole('switch', { name: 'Read images with my Claude plan' }).click()
  await page.waitForFunction(() => document.querySelector('[role="switch"][aria-label="Read images with my Claude plan"]')?.getAttribute('aria-checked') === 'true')
  await page.evaluate(() => window.bridgeclip.settings.load())
  await shot('provider-setup-5-claude-plan.png')
  assert.equal(saved().claudePlanModel, 'opus')
  assert.equal(saved().claudePlanEffort, 'xhigh')
  assert.equal(saved().claudePlanVision, 'on')

  // The sign-in line follows Claude Code: Check again finds a sign-out, and a failed check says so.
  await app.evaluate(({ ipcMain }) => {
    ipcMain.removeHandler('assistant:status')
    ipcMain.handle('assistant:status', () => [{ id: 'claude', state: 'signed-out', version: '2.0.0', account: null, plan: null, detail: null, checkedAt: '' }])
  })
  await page.locator('div', { hasText: 'Clips are picked by Claude Code' }).last().getByRole('button', { name: 'Check again' }).click()
  await page.getByText('Claude Code is not signed in.', { exact: false }).waitFor()
  await app.evaluate(({ ipcMain }) => {
    ipcMain.removeHandler('assistant:status')
    ipcMain.handle('assistant:status', () => { throw new Error('no answer') })
  })
  await picker.getByRole('radio', { name: /Google Gemini/ }).click()
  await picker.getByRole('radio').first().click()
  await page.getByText('Could not check Claude Code.').waitFor()

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
