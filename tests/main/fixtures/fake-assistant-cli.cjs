'use strict'
// Stands in for `claude -p --output-format stream-json` and `codex exec --json`
// in the assistant tests. It talks to BridgeClip's MCP server exactly as the
// real CLIs do (HTTP JSON-RPC with the bearer token from the environment) and
// prints each CLI's event format. Prompts drive it:
//   CALL <tool> <json args>   call a BridgeClip tool, then report its result
//   FAIL_AUTH                 exit like a signed-out CLI
//   HANG                      never answer (for stop and watchdog tests)
//   anything else             reply with a short echo

const argv = process.argv.slice(2)
const codex = argv[0] === 'exec'

function arg(name) {
  const index = argv.indexOf(name)
  return index >= 0 ? argv[index + 1] : undefined
}

function mcpEndpoint() {
  if (!codex) {
    const config = JSON.parse(arg('--mcp-config'))
    const server = config.mcpServers.bridgeclip
    const authorization = server.headers.Authorization.replace(/\$\{(\w+)\}/g, (_, name) => process.env[name] ?? '')
    return { url: server.url, authorization }
  }
  const setting = argv.find((value) => value.startsWith('mcp_servers.bridgeclip='))
  const url = /url="([^"]+)"/.exec(setting)[1]
  const tokenVar = /bearer_token_env_var="([^"]+)"/.exec(setting)[1]
  return { url, authorization: `Bearer ${process.env[tokenVar]}` }
}

let rpcId = 0
async function rpc(endpoint, method, params) {
  const response = await fetch(endpoint.url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream', Authorization: endpoint.authorization },
    body: JSON.stringify({ jsonrpc: '2.0', id: ++rpcId, method, params })
  })
  if (!response.ok) throw new Error(`MCP HTTP ${response.status}`)
  return (await response.json()).result
}

const print = (event) => process.stdout.write(`${JSON.stringify(event)}\n`)

function claudeText(id, text) {
  print({ type: 'stream_event', event: { type: 'message_start', message: { id } }, parent_tool_use_id: null })
  print({ type: 'stream_event', event: { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }, parent_tool_use_id: null })
  for (const piece of text.match(/.{1,6}/gs) ?? []) {
    print({ type: 'stream_event', event: { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }, parent_tool_use_id: null })
  }
  print({ type: 'stream_event', event: { type: 'content_block_stop', index: 0 }, parent_tool_use_id: null })
  print({ type: 'assistant', message: { id, content: [{ type: 'text', text }] }, parent_tool_use_id: null })
}

async function main() {
  let prompt = ''
  for await (const chunk of process.stdin) prompt += chunk
  prompt = prompt.trim()
  if (prompt.endsWith('FAIL_AUTH')) {
    process.stderr.write('Error: Not logged in · Please run /login\n')
    process.exit(1)
  }
  const resumeIndex = argv.indexOf('resume')
  const resumed = codex ? (resumeIndex >= 0 ? argv[resumeIndex + 1] : null) : arg('--resume') ?? null
  const session = resumed ?? (codex ? 'thread-1' : 'session-1')
  if (codex) print({ type: 'thread.started', thread_id: session })
  else print({ type: 'system', subtype: 'init', session_id: session })

  if (prompt.endsWith('HANG')) {
    setInterval(() => {}, 1000)
    return
  }

  const endpoint = mcpEndpoint()
  await rpc(endpoint, 'initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'fake', version: '1' } })
  const { tools } = await rpc(endpoint, 'tools/list', {})
  const reply = []
  if (resumed) reply.push(`resumed ${resumed}`)
  const call = /CALL (\w+) (\{.*\})$/s.exec(prompt)
  if (call) {
    const result = await rpc(endpoint, 'tools/call', { name: call[1], arguments: JSON.parse(call[2]) })
    reply.push(`${result.isError ? 'error' : 'ok'}: ${result.content[0].text}`)
  } else {
    reply.push(`echo: ${prompt.split('\n').pop()} (${tools.length} tools)`)
  }

  if (codex) {
    print({ type: 'item.completed', item: { id: 'item_0', type: 'agent_message', text: reply.join(' | ') } })
    print({ type: 'turn.completed', usage: { input_tokens: 1, output_tokens: 1 } })
  } else {
    claudeText('msg_1', 'Working on it.')
    claudeText('msg_2', reply.join(' | '))
    print({ type: 'result', subtype: 'success', is_error: false, session_id: session, result: reply.join(' | ') })
  }
}

main().catch((error) => {
  process.stderr.write(`fake cli failed: ${error.message}\n`)
  process.exit(2)
})
