'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { loadMain, tempDir } = require('../zernio/support/load-main.cjs')

const FAKE_CLI = path.join(__dirname, 'fixtures/fake-assistant-cli.cjs')

const mcp = loadMain("export * from './src/main/assistant/mcp-server'")
const parsers = loadMain("export * from './src/main/assistant/stream-parsers'")
const providers = loadMain("export * from './src/main/assistant/providers'")
const cliProcess = loadMain("export * from './src/main/assistant/cli-process'")
const toolInput = loadMain("export * from './src/main/assistant/tool-input'")
const store = loadMain("export * from './src/main/assistant/conversation-store'")
const signIn = loadMain("export * from './src/main/assistant/sign-in'", { electron: { shell: {} } })
const { AssistantService } = loadMain("export * from './src/main/assistant/assistant-service'")

async function post(session, body, headers = {}) {
  const url = new URL(session.url)
  const response = await fetch(session.url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${session.token}`, Host: url.host, ...headers },
    body: typeof body === 'string' ? body : JSON.stringify(body)
  })
  const text = await response.text()
  return { status: response.status, body: text ? JSON.parse(text) : null }
}

test('the MCP server only answers its own token, from loopback, without a foreign Origin', async (t) => {
  const server = new mcp.LoopbackMcpServer('1.0.0')
  t.after(() => server.stop())
  const session = await server.openSession({ instructions: 'hi', listTools: () => [], callTool: async () => ({ text: '' }) })
  assert.match(session.url, /^http:\/\/127\.0\.0\.1:\d+\/mcp$/)
  const ping = { jsonrpc: '2.0', id: 1, method: 'ping' }
  assert.equal((await post(session, ping)).status, 200)
  assert.equal((await post(session, ping, { Authorization: 'Bearer wrong' })).status, 401)
  assert.equal((await post(session, ping, { Authorization: '' })).status, 401)
  assert.equal((await post(session, ping, { Origin: 'https://evil.example' })).status, 403)
  assert.equal((await post(session, ping, { 'Content-Type': 'text/plain' })).status, 415)
  const get = await fetch(session.url, { headers: { Authorization: `Bearer ${session.token}` } })
  assert.equal(get.status, 405)
  const wrongPath = await fetch(session.url.replace('/mcp', '/other'), { method: 'POST', headers: { Authorization: `Bearer ${session.token}` } })
  assert.equal(wrongPath.status, 404)
  // A rebinding page reaches the port under another host name.
  const rebinding = await new Promise((resolve, reject) => {
    const request = require('node:http').request(session.url, { method: 'POST', headers: { Host: 'attacker.example', 'Content-Type': 'application/json', Authorization: `Bearer ${session.token}` } }, (response) => resolve(response.statusCode))
    request.on('error', reject)
    request.end(JSON.stringify(ping))
  })
  assert.equal(rebinding, 403)
  session.close()
  assert.equal((await post(session, ping)).status, 401, 'a closed turn revokes its token')
})

test('the MCP server lists and calls tools, reports failures as tool errors and honours cancellation', async (t) => {
  const server = new mcp.LoopbackMcpServer('1.2.3')
  t.after(() => server.stop())
  let cancelled = false
  const session = await server.openSession({
    instructions: 'BridgeClip tools',
    listTools: () => [{ name: 'echo', title: 'Echo', description: 'Echo', inputSchema: { type: 'object' } }],
    callTool: async (name, args, signal) => {
      if (name === 'slow') {
        await new Promise((resolve) => signal.addEventListener('abort', resolve, { once: true }))
        cancelled = true
        return { text: 'stopped', isError: true }
      }
      if (name === 'boom') throw new Error('it broke')
      return { text: JSON.stringify(args) }
    }
  })
  const init = await post(session, { jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-11-25', capabilities: {}, clientInfo: { name: 't', version: '1' } } })
  assert.equal(init.body.result.protocolVersion, '2025-11-25')
  assert.equal(init.body.result.serverInfo.name, 'bridgeclip')
  assert.equal(init.body.result.instructions, 'BridgeClip tools')
  assert.deepEqual(init.body.result.capabilities, { tools: { listChanged: false } })
  assert.equal((await post(session, { jsonrpc: '2.0', method: 'notifications/initialized' })).status, 202)
  const list = await post(session, { jsonrpc: '2.0', id: 2, method: 'tools/list' })
  assert.deepEqual(list.body.result.tools.map((tool) => tool.name), ['echo'])
  const call = await post(session, { jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'echo', arguments: { a: 1 } } })
  assert.deepEqual(call.body.result, { content: [{ type: 'text', text: '{"a":1}' }], isError: false })
  const boom = await post(session, { jsonrpc: '2.0', id: 4, method: 'tools/call', params: { name: 'boom' } })
  assert.equal(boom.body.result.isError, true)
  assert.equal(boom.body.result.content[0].text, 'it broke')
  assert.equal((await post(session, { jsonrpc: '2.0', id: 5, method: 'nope' })).body.error.code, -32601)
  assert.equal((await post(session, '{not json')).body.error.code, -32700)
  const slow = post(session, { jsonrpc: '2.0', id: 'slow-1', method: 'tools/call', params: { name: 'slow' } })
  await new Promise((resolve) => setTimeout(resolve, 50))
  assert.equal((await post(session, { jsonrpc: '2.0', method: 'notifications/cancelled', params: { requestId: 'slow-1' } })).status, 202)
  assert.equal((await slow).body.result.isError, true)
  assert.equal(cancelled, true)
})

test('Claude stream-json becomes text, a resume id and a final status', () => {
  const parser = parsers.createClaudeStreamParser()
  const events = [
    { type: 'system', subtype: 'init', session_id: 's-1' },
    { type: 'stream_event', event: { type: 'message_start', message: { id: 'm1' } } },
    { type: 'stream_event', event: { type: 'content_block_start', content_block: { type: 'text', text: '' } } },
    { type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'text_delta', text: 'Hel' } } },
    { type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'text_delta', text: 'lo' } } },
    { type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'text_delta', text: 'subagent' } }, parent_tool_use_id: 'toolu_1' },
    { type: 'stream_event', event: { type: 'content_block_stop' } },
    { type: 'assistant', message: { id: 'm1', content: [{ type: 'text', text: 'Hello' }] } },
    { type: 'assistant', message: { id: 'm2', content: [{ type: 'text', text: 'Done.' }] } },
    { type: 'result', subtype: 'success', is_error: false, session_id: 's-1', result: 'Done.' }
  ].flatMap((event) => parser.push(JSON.stringify(event)))
  assert.deepEqual(events, [
    { kind: 'session', id: 's-1' },
    { kind: 'text-delta', text: 'Hel' },
    { kind: 'text-delta', text: 'lo' },
    { kind: 'text-break' },
    { kind: 'text-delta', text: 'Done.' },
    { kind: 'session', id: 's-1' },
    { kind: 'done', error: null }
  ])
  assert.deepEqual(parser.finish(), [])
  assert.deepEqual(parser.push('not json'), [])

  const failing = parsers.createClaudeStreamParser()
  assert.deepEqual(failing.push(JSON.stringify({ type: 'result', subtype: 'error_max_turns', is_error: true })), [{ kind: 'done', error: 'error max turns' }])
  assert.match(parsers.createClaudeStreamParser().finish()[0].error, /stopped before finishing/)
})

test('Codex exec JSONL becomes text, activity and errors', () => {
  const parser = parsers.createCodexStreamParser()
  const events = [
    { type: 'thread.started', thread_id: 't-1' },
    { type: 'item.updated', item: { id: 'item_0', type: 'agent_message', text: 'Look' } },
    { type: 'item.completed', item: { id: 'item_0', type: 'agent_message', text: 'Looking now.' } },
    { type: 'item.started', item: { id: 'item_1', type: 'command_execution', command: 'ls', status: 'in_progress' } },
    { type: 'item.completed', item: { id: 'item_1', type: 'command_execution', command: 'ls', aggregated_output: 'a\nb', exit_code: 0 } },
    { type: 'item.completed', item: { id: 'item_2', type: 'mcp_tool_call', tool: 'get_overview', status: 'completed' } },
    { type: 'item.completed', item: { id: 'item_3', type: 'agent_message', text: 'Found 2.' } },
    { type: 'turn.completed', usage: {} }
  ].flatMap((event) => parser.push(JSON.stringify(event)))
  assert.deepEqual(events, [
    { kind: 'session', id: 't-1' },
    { kind: 'text-delta', text: 'Look' },
    { kind: 'text-delta', text: 'ing now.' },
    { kind: 'activity', id: 'command:item_1', title: 'Ran ls', status: 'running', summary: null },
    { kind: 'activity', id: 'command:item_1', title: 'Ran ls', status: 'done', summary: 'a b' },
    { kind: 'text-break' },
    { kind: 'text-delta', text: 'Found 2.' },
    { kind: 'done', error: null }
  ])
  const failing = parsers.createCodexStreamParser()
  failing.push(JSON.stringify({ type: 'error', message: 'Reconnecting… 1/5' }))
  assert.deepEqual(failing.finish(), [{ kind: 'done', error: 'Reconnecting… 1/5' }])
  const failed = parsers.createCodexStreamParser()
  assert.deepEqual(failed.push(JSON.stringify({ type: 'turn.failed', error: { message: 'usage limit' } })), [{ kind: 'done', error: 'usage limit' }])
})

test('CLI failures map to sign-in, limit and update guidance', () => {
  assert.match(parsers.describeCliFailure('claude', 'Error: Not logged in · Please run /login', 1), /isn’t signed in.*run `claude`/)
  assert.match(parsers.describeCliFailure('claude', "You've hit your limit · resets 3pm", 1), /usage limit/)
  assert.match(parsers.describeCliFailure('claude', "error: unknown option '--permission-prompts'", 1), /too old/)
  assert.match(parsers.describeCliFailure('codex', 'Error: Not logged in. Run `codex login`', 1), /run `codex login`/)
  assert.match(parsers.describeCliFailure('codex', "error: unexpected argument '--ignore-user-config'", 2), /too old/)
  assert.equal(parsers.describeCliFailure('codex', '', 9), 'Codex stopped unexpectedly (exit code 9).')
})

test('turn commands keep the CLIs on BridgeClip tools and never put the token in argv', () => {
  const cli = { command: '/bin/claude', prefixArgs: [], pathValue: '/usr/bin' }
  const options = { cli, model: 'sonnet', resumeId: 'sess-9', systemPrompt: 'SYSTEM', mcpUrl: 'http://127.0.0.1:5555/mcp', cwd: '/tmp/ws' }
  const claude = providers.claudeTurnArgs(options)
  const value = (args, flag) => args[args.indexOf(flag) + 1]
  // Only Claude Code's web tools: nothing that runs code or touches files.
  assert.equal(value(claude, '--tools'), 'WebSearch,WebFetch')
  assert.equal(value(claude, '--permission-mode'), 'dontAsk')
  assert.equal(value(claude, '--allowedTools'), 'mcp__bridgeclip,WebSearch,WebFetch')
  assert.ok(!/Bash|Edit|Write|Read\b/.test(value(claude, '--tools')))
  assert.equal(value(claude, '--setting-sources'), 'user')
  assert.equal(value(claude, '--system-prompt'), 'SYSTEM')
  assert.equal(value(claude, '--resume'), 'sess-9')
  assert.equal(value(claude, '--model'), 'sonnet')
  assert.ok(claude.includes('--strict-mcp-config'))
  assert.ok(!claude.includes('--bare'), '--bare disables subscription sign-in')
  assert.ok(!claude.includes('--dangerously-skip-permissions'))
  const config = JSON.parse(value(claude, '--mcp-config'))
  assert.deepEqual(config.mcpServers.bridgeclip, { type: 'http', url: options.mcpUrl, headers: { Authorization: 'Bearer ${BRIDGECLIP_MCP_TOKEN}' } })
  const claudeEnv = providers.claudeTurnEnv(cli, 'secret-token')
  assert.equal(claudeEnv.BRIDGECLIP_MCP_TOKEN, 'secret-token')
  assert.ok(!claude.join(' ').includes('secret-token'))

  const codex = providers.codexTurnArgs({ ...options, model: '', resumeId: 'thread-3' })
  assert.equal(codex[0], 'exec')
  assert.deepEqual(codex.slice(-4), ['resume', 'thread-3', '--', '-'])
  assert.ok(codex.indexOf('-c') < codex.indexOf('resume'), 'config flags precede resume')
  assert.equal(value(codex, '-s'), 'read-only')
  assert.ok(codex.includes('--ignore-user-config'))
  assert.ok(!codex.includes('-m'), 'the default model is left to Codex')
  assert.ok(!codex.some((arg) => arg.includes('dangerously')))
  const server = codex.find((arg) => arg.startsWith('mcp_servers.bridgeclip='))
  assert.match(server, /url="http:\/\/127\.0\.0\.1:5555\/mcp"/)
  assert.match(server, /bearer_token_env_var="BRIDGECLIP_MCP_TOKEN"/)
  assert.match(server, /tool_timeout_sec=900/)
  assert.ok(codex.includes('developer_instructions="SYSTEM"'))
  assert.ok(codex.includes('web_search="live"'), 'Codex can search the web')
  assert.equal(providers.codexTurnEnv(cli, 'tok').BRIDGECLIP_MCP_TOKEN, 'tok')
})

test('sign-in status parsing covers signed in, signed out and API-key logins', () => {
  assert.deepEqual(providers.parseClaudeAuthStatus('{"loggedIn":true,"authMethod":"claude.ai","email":"a@b.c","subscriptionType":"max"}'), { loggedIn: true, account: 'a@b.c', plan: 'max', apiKey: false })
  assert.equal(providers.parseClaudeAuthStatus('{"loggedIn":false,"authMethod":"none"}').loggedIn, false)
  assert.equal(providers.parseClaudeAuthStatus('{"loggedIn":true,"authMethod":"api_key"}').apiKey, true)
  assert.equal(providers.parseClaudeAuthStatus('garbage'), null)
  assert.deepEqual(providers.parseCodexLoginStatus('Logged in using ChatGPT', 0), { loggedIn: true, apiKey: false })
  assert.deepEqual(providers.parseCodexLoginStatus('Logged in using an API key - sk-***', 0), { loggedIn: true, apiKey: true })
  assert.deepEqual(providers.parseCodexLoginStatus('Not logged in', 1), { loggedIn: false, apiKey: false })
  assert.equal(providers.parseCodexLoginStatus('', 0), null)
})

test('the CLI environment drops API keys so the subscription is billed, and keeps the keychain user', () => {
  const env = cliProcess.cliEnvironment('/opt/bin:/usr/bin', { EXTRA: '1' }, {
    HOME: '/Users/u', USER: 'u', LANG: 'en_US.UTF-8', LC_ALL: 'C', PATH: '/usr/bin',
    ANTHROPIC_API_KEY: 'sk-ant', OPENAI_API_KEY: 'sk-oa', CLAUDE_CODE_OAUTH_TOKEN: 'x', AWS_SECRET_ACCESS_KEY: 'y', DYLD_INSERT_LIBRARIES: 'z', ELECTRON_RUN_AS_NODE: '1'
  })
  assert.equal(env.PATH, '/opt/bin:/usr/bin')
  assert.equal(env.HOME, '/Users/u')
  assert.equal(env.USER, 'u')
  assert.equal(env.LC_ALL, 'C')
  assert.equal(env.EXTRA, '1')
  for (const key of ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN', 'AWS_SECRET_ACCESS_KEY', 'DYLD_INSERT_LIBRARIES', 'ELECTRON_RUN_AS_NODE']) assert.equal(env[key], undefined, key)
  assert.ok(cliProcess.cliEnvironment('/x', {}, {}).USER, 'falls back to the account name')
  const dirs = cliProcess.wellKnownCliDirectories('/Users/u', 'darwin')
  for (const dir of ['/Users/u/.local/bin', '/opt/homebrew/bin', '/usr/local/bin']) assert.ok(dirs.includes(dir), dir)
})

test('tool input is checked against its schema before a tool runs', () => {
  const schema = {
    type: 'object',
    properties: { id: { type: 'string', pattern: '^[a-z]+$' }, n: { type: 'integer', minimum: 1, maximum: 3 }, kind: { type: 'string', enum: ['a', 'b'] }, list: { type: 'array', items: { type: 'string' }, maxItems: 2 } },
    required: ['id'],
    additionalProperties: false
  }
  assert.deepEqual(toolInput.validateToolInput({ id: 'abc', n: 2, kind: null }, schema), { id: 'abc', n: 2 })
  assert.throws(() => toolInput.validateToolInput({}, schema), /id is required/)
  assert.throws(() => toolInput.validateToolInput({ id: 'ABC' }, schema), /invalid format/)
  assert.throws(() => toolInput.validateToolInput({ id: 'a', n: 1.5 }, schema), /whole number/)
  assert.throws(() => toolInput.validateToolInput({ id: 'a', n: 9 }, schema), /at most 3/)
  assert.throws(() => toolInput.validateToolInput({ id: 'a', kind: 'c' }, schema), /one of/)
  assert.throws(() => toolInput.validateToolInput({ id: 'a', list: ['x', 'y', 'z'] }, schema), /at most 2/)
  assert.throws(() => toolInput.validateToolInput({ id: 'a', extra: 1 }, schema), /Unknown field extra/)
})

test('conversations persist privately and preferences ignore unknown models', (t) => {
  const { dir, cleanup } = tempDir('bridgeclip-assistant-store-')
  t.after(cleanup)
  const conversations = new store.ConversationStore(dir)
  const make = (id, updatedAt) => ({ id, title: id.slice(0, 4), provider: 'claude', model: '', resumeId: null, createdAt: updatedAt, updatedAt, messages: [] })
  conversations.save(make('11111111-1111-4111-8111-111111111111', '2026-01-01T00:00:00.000Z'))
  conversations.save(make('22222222-2222-4222-8222-222222222222', '2026-02-01T00:00:00.000Z'))
  assert.deepEqual(conversations.list().map((item) => item.id[0]), ['2', '1'])
  assert.equal(conversations.get('../../etc/passwd'), null)
  assert.throws(() => conversations.save(make('../escape', '2026-01-01T00:00:00.000Z')), /Invalid conversation id/)
  if (process.platform !== 'win32') assert.equal(fs.statSync(path.join(dir, 'conversations', '11111111-1111-4111-8111-111111111111.json')).mode & 0o777, 0o600)
  conversations.delete('11111111-1111-4111-8111-111111111111')
  assert.equal(conversations.list().length, 1)
  assert.deepEqual(conversations.preferences(), { provider: null, models: { claude: '', codex: '', openrouter: '' } })
  fs.writeFileSync(path.join(dir, 'preferences.json'), JSON.stringify({ provider: 'codex', models: { claude: 'opus', codex: 'not-a-model', openrouter: 'vendor/model-1' } }))
  assert.deepEqual(conversations.preferences(), { provider: 'codex', models: { claude: 'opus', codex: '', openrouter: 'vendor/model-1' } })
  fs.writeFileSync(path.join(dir, 'preferences.json'), JSON.stringify({ provider: 'openrouter', models: { openrouter: 'https://evil.test/x' } }))
  assert.deepEqual(conversations.preferences(), { provider: 'openrouter', models: { claude: '', codex: '', openrouter: '' } })
})

test('sign-in only relays links to the provider’s own sign-in hosts', () => {
  const claudeOutput = 'Opening browser to sign in…\nIf the browser didn’t open, visit: https://claude.com/cai/oauth/authorize?code=true&state=x\nPaste code here if prompted > '
  assert.equal(signIn.signInUrl('claude', claudeOutput), 'https://claude.com/cai/oauth/authorize?code=true&state=x')
  assert.equal(signIn.signInUrl('codex', 'Starting local login server on http://localhost:1455.\n\nhttps://auth.openai.com/oauth/authorize?client_id=abc\n'), 'https://auth.openai.com/oauth/authorize?client_id=abc')
  assert.equal(signIn.signInUrl('codex', 'visit https://auth.openai.com.evil.test/x or http://auth.openai.com/y'), null)
  assert.equal(signIn.signInUrl('claude', 'https://user:pw@claude.ai/login'), null)
})

function serviceHarness(t, provider = 'claude', extraTools = [], { openRouter, model = '' } = {}) {
  const { dir, cleanup } = tempDir('bridgeclip-assistant-service-')
  const events = []
  const waiters = []
  const calls = []
  const tools = [
    { name: 'echo', title: 'Echo', description: 'Echo text', inputSchema: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'], additionalProperties: false }, readOnly: true,
      run: async (input) => { calls.push(['echo', input]); return { echoed: input.text, summary: `echoed ${input.text}` } } },
    { name: 'publish', title: 'Publish', description: 'Needs approval', inputSchema: { type: 'object', properties: {}, additionalProperties: false },
      confirm: () => ['Publish publicly'], run: async () => { calls.push(['publish']); return { published: true } } },
    ...extraTools
  ]
  const service = new AssistantService({
    root: dir,
    appVersion: '0.0.0-test',
    tools: () => tools,
    systemPrompt: () => 'TEST SYSTEM',
    emit: (event) => {
      events.push(event)
      for (const waiter of [...waiters]) if (waiter.match(event)) { waiters.splice(waiters.indexOf(waiter), 1); waiter.resolve(event) }
    },
    resolveCli: async () => ({ command: process.execPath, prefixArgs: [FAKE_CLI], pathValue: process.env.PATH ?? '' }),
    openRouter
  })
  t.after(async () => { await service.shutdown(); cleanup() })
  const next = (match, timeout = 15000) => {
    const found = events.find(match)
    if (found) return Promise.resolve(found)
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('timed out waiting for event')), timeout)
      waiters.push({ match, resolve: (event) => { clearTimeout(timer); resolve(event) } })
    })
  }
  const turnEnd = (conversationId, after = 0) => next((event) => event.type === 'turn-end' && event.conversationId === conversationId && events.indexOf(event) >= after)
  const send = (text, conversationId = null) => service.sendMessage({ conversationId, provider, model, text })
  return { service, events, calls, next, turnEnd, send }
}

for (const provider of ['claude', 'codex']) {
  test(`${provider}: a turn streams text, calls BridgeClip tools over MCP and resumes the session`, async (t) => {
    const { service, events, calls, turnEnd, send } = serviceHarness(t, provider)
    const first = send('CALL echo {"text":"hi"}')
    const end = await turnEnd(first.conversationId)
    assert.equal(end.error, null)
    assert.deepEqual(calls, [['echo', { text: 'hi' }]])
    const tool = events.filter((event) => event.type === 'tool').at(-1).tool
    assert.equal(tool.status, 'done')
    assert.equal(tool.summary, 'echoed hi')
    const conversation = service.getConversation(first.conversationId)
    assert.equal(conversation.resumeId, provider === 'claude' ? 'session-1' : 'thread-1')
    const reply = conversation.messages[1]
    const text = reply.parts.filter((part) => part.kind === 'text').map((part) => part.text).join('')
    assert.match(text, /ok: \{\n? ?"echoed": "hi"/)
    assert.ok(reply.parts.some((part) => part.kind === 'tool' && part.tool.name === 'echo'))
    assert.deepEqual([reply.provider, reply.model], [provider, ''], 'the reply records who wrote it')
    if (provider === 'claude') {
      assert.ok(events.filter((event) => event.type === 'text-delta').length > 3, 'streams token deltas')
      assert.match(text, /^Working on it\.\n\nok:/, 'separate messages are separated')
    }

    const mark = events.length
    send('hello again', first.conversationId)
    assert.equal((await turnEnd(first.conversationId, mark)).error, null)
    const again = service.getConversation(first.conversationId)
    assert.equal(again.messages.length, 4)
    assert.match(again.messages[3].parts.map((part) => part.text ?? '').join(''), new RegExp(`resumed ${provider === 'claude' ? 'session-1' : 'thread-1'}`))
    assert.deepEqual(service.listConversations().map((item) => item.id), [first.conversationId])
  })
}

test('actions that need approval wait for the user and respect a refusal', async (t) => {
  const { service, events, calls, next, turnEnd, send } = serviceHarness(t)
  const first = send('CALL publish {}')
  const approval = await next((event) => event.type === 'approval')
  assert.deepEqual(approval.request.details, ['Publish publicly'])
  assert.equal(calls.length, 0, 'nothing runs before approval')
  service.respondToApproval(approval.request.id, false)
  await turnEnd(first.conversationId)
  assert.equal(calls.length, 0)
  const text = service.getConversation(first.conversationId).messages[1].parts.map((part) => part.text ?? '').join('')
  assert.match(text, /error: The user did not approve/)
  assert.equal(events.filter((event) => event.type === 'tool').at(-1).tool.status, 'denied')

  const mark = events.length
  send('CALL publish {}', first.conversationId)
  const second = await next((event) => event.type === 'approval' && events.indexOf(event) >= mark)
  service.respondToApproval(second.request.id, true)
  await turnEnd(first.conversationId, mark)
  assert.deepEqual(calls, [['publish']])
})

test('bad tool input is reported to the agent without running the tool', async (t) => {
  const { service, calls, turnEnd, send } = serviceHarness(t)
  const first = send('CALL echo {"text":5}')
  await turnEnd(first.conversationId)
  assert.equal(calls.length, 0)
  assert.match(service.getConversation(first.conversationId).messages[1].parts.map((part) => part.text ?? '').join(''), /error: text must be a string/)
})

test('a signed-out CLI explains how to sign in', async (t) => {
  const { turnEnd, send } = serviceHarness(t)
  const first = send('FAIL_AUTH')
  const end = await turnEnd(first.conversationId)
  assert.match(end.error, /isn’t signed in/)
})

test('stopping a reply ends the CLI and keeps the chat usable', async (t) => {
  const { service, events, next, turnEnd, send } = serviceHarness(t)
  const first = send('HANG')
  await next((event) => event.type === 'turn-start')
  await new Promise((resolve) => setTimeout(resolve, 300))
  assert.throws(() => send('second', first.conversationId), /still replying/)
  service.stop(first.conversationId)
  const end = await turnEnd(first.conversationId)
  assert.equal(end.error, 'You stopped this reply.')
  assert.equal(service.isRunning(first.conversationId), false)
  const mark = events.length
  send('after stop', first.conversationId)
  assert.equal((await turnEnd(first.conversationId, mark)).error, null)
})

test('messages are validated before a turn starts', (t) => {
  const { service } = serviceHarness(t)
  assert.throws(() => service.sendMessage({ conversationId: null, provider: 'claude', model: '', text: '   ' }), /Type a message/)
  assert.throws(() => service.sendMessage({ conversationId: null, provider: 'claude', model: 'gpt-9', text: 'hi' }), /Choose a model/)
  assert.throws(() => service.sendMessage({ conversationId: 'not-an-id', provider: 'claude', model: '', text: 'hi' }), /Invalid conversation id/)
  assert.throws(() => service.sendMessage({ conversationId: null, provider: 'claude', model: '', text: 'x'.repeat(20001) }), /limited/)
})

// ---- OpenRouter: BridgeClip runs the agent loop itself ----

/** A streamed chat completion, as OpenRouter sends it. */
function sse(chunks) {
  const body = chunks.map((chunk) => `: OPENROUTER PROCESSING\n\ndata: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n'
  return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } })
}
const textChunks = (...parts) => parts.map((content) => ({ choices: [{ delta: { content } }] }))
const toolChunks = (id, name, args) => [
  { choices: [{ delta: { tool_calls: [{ index: 0, id, type: 'function', function: { name, arguments: '' } }] } }] },
  // Arguments arrive in pieces.
  { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: args.slice(0, 4) } }] } }] },
  { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: args.slice(4) } }] }, finish_reason: 'tool_calls' }] }
]

const openRouterOnlyTool = { name: 'web_only', title: 'Web only', description: 'Only for OpenRouter models', inputSchema: { type: 'object', properties: {}, additionalProperties: false }, readOnly: true, providers: ['openrouter'], run: async () => ({ ok: true }) }

function openRouterHarness(t, replies, { key = 'sk-or-test', model = 'vendor/model-1' } = {}) {
  const requests = []
  const fetch = async (url, init) => {
    const body = JSON.parse(init.body)
    requests.push({ url, init, body })
    const reply = replies[requests.length - 1]
    if (!reply) throw new Error('unexpected request')
    return typeof reply === 'function' ? reply(init, body) : reply
  }
  const harness = serviceHarness(t, 'openrouter', [openRouterOnlyTool], { model, openRouter: { apiKey: () => key, url: 'https://openrouter.test/chat', fetch } })
  return { ...harness, requests }
}

test('openrouter: streams a reply, runs BridgeClip tools and sends their results back', async (t) => {
  const { service, calls, turnEnd, send, requests } = openRouterHarness(t, [
    () => sse([...textChunks('Let me check.'), ...toolChunks('call_1', 'echo', '{"text":"hi"}')]),
    () => sse(textChunks('Done: ', 'hi'))
  ])
  const first = send('Say hi through echo')
  assert.equal((await turnEnd(first.conversationId)).error, null)
  assert.deepEqual(calls, [['echo', { text: 'hi' }]])

  const [one, two] = requests
  assert.equal(one.url, 'https://openrouter.test/chat')
  assert.equal(one.init.headers.Authorization, 'Bearer sk-or-test')
  assert.equal(one.init.redirect, 'error')
  assert.equal(one.body.model, 'vendor/model-1')
  assert.equal(one.body.stream, true)
  assert.deepEqual(one.body.provider, { require_parameters: true })
  assert.deepEqual(one.body.messages.map((message) => message.role), ['system', 'user'])
  assert.equal(one.body.messages[0].content, 'TEST SYSTEM')
  assert.deepEqual(one.body.tools.map((tool) => tool.function.name), ['echo', 'publish', 'web_only'])
  assert.equal(one.body.tools[0].function.parameters.required[0], 'text')
  const assistantStep = two.body.messages[2]
  assert.equal(assistantStep.role, 'assistant')
  assert.deepEqual(assistantStep.tool_calls, [{ id: 'call_1', type: 'function', function: { name: 'echo', arguments: '{"text":"hi"}' } }])
  assert.equal(two.body.messages[3].role, 'tool')
  assert.equal(two.body.messages[3].tool_call_id, 'call_1')
  assert.match(two.body.messages[3].content, /"echoed": "hi"/)

  const reply = service.getConversation(first.conversationId).messages[1]
  assert.deepEqual([reply.provider, reply.model], ['openrouter', 'vendor/model-1'])
  assert.deepEqual(reply.parts.map((part) => part.kind === 'text' ? part.text : `[${part.tool.name} ${part.tool.status}]`), ['Let me check.', '[echo done]', 'Done: hi'])
})

test('openrouter: later messages carry the chat so far, since OpenRouter keeps no session', async (t) => {
  const { events, turnEnd, send, requests } = openRouterHarness(t, [
    () => sse(textChunks('First answer')),
    () => sse(textChunks('Second answer'))
  ])
  const first = send('first question')
  await turnEnd(first.conversationId)
  const mark = events.length
  send('second question', first.conversationId)
  assert.equal((await turnEnd(first.conversationId, mark)).error, null)
  assert.deepEqual(requests[1].body.messages.map((message) => [message.role, message.content]), [
    ['system', 'TEST SYSTEM'], ['user', 'first question'], ['assistant', 'First answer'], ['user', 'second question']
  ])
})

test('openrouter: approvals work the same, and a refusal goes back to the model', async (t) => {
  const { service, calls, next, turnEnd, send, requests } = openRouterHarness(t, [
    () => sse(toolChunks('call_p', 'publish', '{}')),
    () => sse(textChunks('Okay, not publishing.'))
  ])
  const first = send('publish it')
  const approval = await next((event) => event.type === 'approval')
  assert.deepEqual(approval.request.details, ['Publish publicly'])
  service.respondToApproval(approval.request.id, false)
  assert.equal((await turnEnd(first.conversationId)).error, null)
  assert.deepEqual(calls, [])
  assert.match(requests[1].body.messages.at(-1).content, /^Error: The user did not approve/)
})

test('openrouter: failures explain what to do', async (t) => {
  const failing = (status, message) => () => new Response(JSON.stringify({ error: { message } }), { status })
  for (const [response, pattern] of [
    [failing(401, 'No auth credentials found'), /rejected your API key/],
    [failing(402, 'Insufficient credits'), /out of credits/],
    [failing(404, 'No endpoints found that support tool use'), /can’t use tools/],
    [failing(400, 'Bad request\u0000 detail'), /OpenRouter returned an error \(400\): Bad request detail/],
    [() => sse([{ error: { message: 'Provider overloaded' } }]), /OpenRouter stopped the reply: Provider overloaded/],
    [() => { throw new TypeError('fetch failed') }, /couldn’t be reached/]
  ]) {
    const { turnEnd, send } = openRouterHarness(t, [response])
    const first = send('hi')
    assert.match((await turnEnd(first.conversationId)).error, pattern)
  }
  const noKey = openRouterHarness(t, [], { key: '' })
  const first = noKey.send('hi')
  assert.match((await noKey.turnEnd(first.conversationId)).error, /Add your OpenRouter API key/)
  assert.equal(noKey.requests.length, 0)
  const noModel = openRouterHarness(t, [], { model: '' })
  assert.throws(() => noModel.send('hi'), /Choose an OpenRouter model/)
})

test('openrouter: stopping aborts the request', async (t) => {
  const { service, next, turnEnd, send, requests } = openRouterHarness(t, [
    (init) => new Promise((resolve, reject) => init.signal.addEventListener('abort', () => reject(init.signal.reason)))
  ])
  const first = send('hang')
  await next((event) => event.type === 'turn-start')
  while (requests.length === 0) await new Promise((resolve) => setTimeout(resolve, 10))
  service.stop(first.conversationId)
  assert.equal((await turnEnd(first.conversationId)).error, 'You stopped this reply.')
  assert.equal(service.isRunning(first.conversationId), false)
})

test('tools limited to OpenRouter models are neither listed nor callable from a CLI', async (t) => {
  const { service, calls, turnEnd, send } = serviceHarness(t, 'claude', [openRouterOnlyTool])
  const first = send('CALL web_only {}')
  await turnEnd(first.conversationId)
  assert.equal(calls.length, 0)
  assert.match(service.getConversation(first.conversationId).messages[1].parts.map((part) => part.text ?? '').join(''), /Unknown tool: web_only/)
})
