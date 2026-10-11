'use strict'
// The assistant's internet access: finding YouTube videos with yt-dlp, and the
// web search and page reader OpenRouter models get.
const test = require('node:test')
const assert = require('node:assert/strict')
const { loadMain } = require('../zernio/support/load-main.cjs')

const youtube = loadMain("export * from './src/main/assistant/youtube-discovery'")
const web = loadMain("export * from './src/main/assistant/web-tools'", { electron: { app: { isPackaged: false } } })
const parsers = loadMain("export * from './src/main/assistant/stream-parsers'")

const entry = (id, title, extra = {}) => JSON.stringify({ ie_key: 'Youtube', id, title, ...extra })
const channelEntry = (name, handle, followers) => JSON.stringify({ ie_key: 'YoutubeTab', title: name, channel_id: `UC${'x'.repeat(21)}${handle.length % 10}`, uploader_id: handle, channel_follower_count: followers })

test('channel input becomes a YouTube channel page, or a name to search for', () => {
  assert.equal(youtube.channelUrl('@bridgemindai'), 'https://www.youtube.com/@bridgemindai')
  assert.equal(youtube.channelUrl('https://www.youtube.com/@bridgemindai/streams'), 'https://www.youtube.com/@bridgemindai')
  assert.equal(youtube.channelUrl('youtube.com/channel/UCwaTGE53GLGC3fDClVl_7TA'), 'https://www.youtube.com/channel/UCwaTGE53GLGC3fDClVl_7TA')
  assert.equal(youtube.channelUrl('https://m.youtube.com/c/SomeName'), 'https://www.youtube.com/c/SomeName')
  for (const name of ['BridgeMind', 'Bridge Mind', 'https://evil.example/@bridgemindai', 'https://www.youtube.com/watch?v=abc']) {
    assert.equal(youtube.channelUrl(name), null, name)
  }
})

test('listing lines become videos; junk lines, bad ids and non-video entries are skipped', () => {
  const stdout = [
    entry('IuV1gMP0-_g', 'OpenAI DevDay 2026', { duration: 19155, view_count: 170000, timestamp: 1790812800, live_status: 'was_live', playlist_channel: 'BridgeMind' }),
    'WARNING: something',
    entry('bad', 'Wrong id'),
    JSON.stringify({ ie_key: 'YoutubeTab', id: 'UCwaTGE53GLGC3fDClVl_7TA', title: 'A channel' }),
    entry('f5ATszIO3ME', 'Day 235\u0007', { live_status: 'is_live' })
  ].join('\n')
  const videos = youtube.parseVideoLines(stdout)
  assert.deepEqual(videos.map((video) => [video.url, video.title, video.live, video.publishedOn, video.channel]), [
    ['https://www.youtube.com/watch?v=IuV1gMP0-_g', 'OpenAI DevDay 2026', 'was live', '2026-10-01', 'BridgeMind'],
    ['https://www.youtube.com/watch?v=f5ATszIO3ME', 'Day 235', 'live now', null, null]
  ])
})

test('a channel name is matched to YouTube’s best channel, and live streams list first', async () => {
  const calls = []
  const run = async (args) => {
    calls.push(args)
    const url = args.at(-1)
    assert.equal(args.at(-2), '--', 'the URL is always after --')
    assert.ok(args.includes('--ignore-config') && args.includes('--use-extractors'))
    if (url.includes('/results?')) return [channelEntry('BridgeMind', '@bridgemindai', 110000), channelEntry('BridgeMind Shorts', '@bridgemindshorts', 27)].join('\n')
    if (url.endsWith('/videos')) return [entry('qY3_rbtMtm4', 'Claude Opus 5.5', { timestamp: 1790208000 })].join('\n')
    if (url.endsWith('/streams')) return [entry('f5ATszIO3ME', 'Day 235', { live_status: 'is_live' }), entry('IuV1gMP0-_g', 'OpenAI DevDay 2026', { timestamp: 1790812800, live_status: 'was_live' })].join('\n')
    throw new Error(`unexpected ${url}`)
  }
  const result = await youtube.findYouTubeVideos({ channel: 'BridgeMind', limit: 3 }, run)
  assert.equal(result.channel.handle, '@bridgemindai')
  assert.deepEqual(result.otherChannels.map((channel) => channel.handle), ['@bridgemindshorts'])
  assert.deepEqual(result.videos.map((video) => video.title), ['Day 235', 'OpenAI DevDay 2026', 'Claude Opus 5.5'])
  assert.match(result.summary, /newest: Day 235 \(live now\)/)
  assert.match(new URL(calls[0].at(-1)).searchParams.get('sp'), /^EgIQAg==$/, 'the name search is limited to channels')
  assert.deepEqual(calls.slice(1).map((args) => args.at(-1)).sort(), ['https://www.youtube.com/@bridgemindai/streams', 'https://www.youtube.com/@bridgemindai/videos'])
})

test('a missing channel, a missing streams tab and an empty request are handled', async () => {
  const missing = async (args) => { if (args.at(-1).endsWith('/videos')) throw new Error('YouTube: HTTP Error 404: Not Found'); return '' }
  await assert.rejects(youtube.findYouTubeVideos({ channel: '@nobodyhere' }, missing), /no YouTube channel at @nobodyhere/)
  const noStreams = async (args) => {
    if (args.at(-1).endsWith('/streams')) throw new Error('This channel does not have a streams tab')
    return entry('qY3_rbtMtm4', 'Only upload', { playlist_channel: 'Someone' })
  }
  const result = await youtube.findYouTubeVideos({ channel: '@someone' }, noStreams)
  assert.deepEqual(result.videos.map((video) => video.title), ['Only upload'])
  assert.equal(result.channel.handle, '@someone')
  await assert.rejects(youtube.findYouTubeVideos({}, noStreams), /channel .* or a search query/)
})

test('a search sorts by upload date only when asked', async () => {
  const urls = []
  const run = async (args) => { urls.push(new URL(args.at(-1))); return entry('Fls_onRviPM', 'Keynote') }
  await youtube.findYouTubeVideos({ query: 'openai devday' }, run)
  await youtube.findYouTubeVideos({ query: 'openai devday', sort: 'newest' }, run)
  assert.equal(urls[0].searchParams.get('search_query'), 'openai devday')
  assert.equal(urls[0].searchParams.get('sp'), null)
  assert.equal(urls[1].searchParams.get('sp'), 'CAI=')
})

test('HTML becomes readable text without scripts, styles or markup', () => {
  const page = web.htmlToText('<html><head><title>Ship &amp; Tell</title><style>p{}</style></head><body><script>alert(1)</script><h1>Hello&nbsp;there</h1><p>One <b>two</b><br>three &#8212; &#x2603;</p><!-- hidden --></body></html>')
  assert.equal(page.title, 'Ship & Tell')
  assert.equal(page.text, 'Hello there\nOne two\nthree — ☃')
})

test('the page reader refuses local and private addresses, including through redirects', async () => {
  const signal = new AbortController().signal
  const never = async () => { throw new Error('must not fetch') }
  for (const url of ['http://127.0.0.1:5555/', 'http://localhost/', 'http://10.0.0.1/', 'http://[::1]/', 'file:///etc/passwd', 'http://192.168.1.1/admin']) {
    await assert.rejects(web.readWebPage(url, signal, never), /Only public http\(s\) web pages/, url)
  }
  const redirecting = async () => new Response(null, { status: 302, headers: { location: 'http://169.254.169.254/latest/meta-data' } })
  await assert.rejects(web.readWebPage('https://93.184.215.14/start', signal, redirecting), /Only public http\(s\) web pages/)
})

test('the page reader returns text, refuses videos and files, and bounds long pages', async () => {
  const signal = new AbortController().signal
  const seen = []
  const html = async (url, init) => {
    seen.push(init)
    return new Response(`<title>News</title><p>${'word '.repeat(6000)}</p>`, { headers: { 'content-type': 'text/html; charset=utf-8' } })
  }
  const page = await web.readWebPage('https://93.184.215.14/news', signal, html)
  assert.equal(page.title, 'News')
  assert.equal(page.truncated, true)
  assert.ok(page.text.length <= 20000 + 20)
  assert.equal(seen[0].redirect, 'manual', 'redirects are checked hop by hop')
  assert.ok(seen[0].dispatcher, 'connections go through the public-only agent')
  const video = async () => new Response('binary', { headers: { 'content-type': 'video/mp4' } })
  await assert.rejects(web.readWebPage('https://93.184.215.14/clip.mp4', signal, video), /video\/mp4, not a web page/)
})

test('web search uses OpenRouter’s server-side search and returns findings with sources', async () => {
  const requests = []
  const fetch = async (url, init) => {
    requests.push({ url, init, body: JSON.parse(init.body) })
    return Response.json({ choices: [{ message: { content: '- BridgeMind streams daily https://youtube.com/@bridgemindai', annotations: [
      { type: 'url_citation', url_citation: { url: 'https://www.youtube.com/@bridgemindai', title: 'BridgeMind' } },
      { type: 'url_citation', url_citation: { url: 'javascript:alert(1)', title: 'bad' } }
    ] } }] })
  }
  const signal = new AbortController().signal
  const result = await web.searchWeb('bridgemind youtube', { apiKey: () => 'sk-or', url: 'https://openrouter.test/chat', fetch }, signal)
  assert.deepEqual(result.sources, [{ title: 'BridgeMind', url: 'https://www.youtube.com/@bridgemindai' }])
  assert.match(result.findings, /streams daily/)
  const { body, init } = requests[0]
  assert.equal(init.headers.Authorization, 'Bearer sk-or')
  assert.equal(body.tools[0].type, 'openrouter:web_search')
  assert.equal(body.tool_choice, 'required')
  assert.match(body.messages[1].content, /bridgemind youtube/)

  const broke = async () => Response.json({ error: { message: 'Insufficient credits' } }, { status: 402 })
  await assert.rejects(web.searchWeb('x', { apiKey: () => 'sk-or', fetch: broke }, signal), /out of credits/)
  await assert.rejects(web.searchWeb('x', { apiKey: () => '', fetch: broke }, signal), /needs an OpenRouter API key/)
})

test('the web tools are offered to OpenRouter models only', () => {
  const tools = web.createWebTools({ apiKey: () => 'k' })
  assert.deepEqual(tools.map((tool) => [tool.name, tool.providers]), [['web_search', ['openrouter']], ['read_web_page', ['openrouter']]])
  assert.ok(tools.every((tool) => tool.readOnly))
})

test('Claude Code’s own web searches and page reads show in the chat', () => {
  const parser = parsers.createClaudeStreamParser()
  const events = [
    { type: 'assistant', message: { id: 'm1', content: [{ type: 'tool_use', id: 't1', name: 'WebSearch', input: { query: 'latest bridgemind video' } }, { type: 'tool_use', id: 't2', name: 'mcp__bridgeclip__get_overview', input: {} }] } },
    { type: 'user', message: { content: [{ type: 'tool_result', tool_use_id: 't1', content: 'results' }, { type: 'tool_result', tool_use_id: 't2', content: '{}' }] } },
    { type: 'assistant', message: { id: 'm2', content: [{ type: 'tool_use', id: 't3', name: 'WebFetch', input: { url: 'https://openai.com/devday', prompt: 'x' } }] } },
    { type: 'user', message: { content: [{ type: 'tool_result', tool_use_id: 't3', is_error: true, content: 'blocked' }] } }
  ].flatMap((event) => parser.push(JSON.stringify(event)))
  assert.deepEqual(events.map((event) => [event.kind, event.title, event.status]), [
    ['activity', 'Searched the web for “latest bridgemind video”', 'running'],
    ['activity', 'Searched the web for “latest bridgemind video”', 'done'],
    ['activity', 'Read openai.com', 'running'],
    ['activity', 'Read openai.com', 'error']
  ])
})

test('Codex’s web searches show in the chat, named once the query arrives', () => {
  const parser = parsers.createCodexStreamParser()
  const events = [
    { type: 'item.started', item: { id: 'ws_1', type: 'web_search', query: '' } },
    { type: 'item.completed', item: { id: 'ws_1', type: 'web_search', query: 'BridgeMind YouTube channel' } }
  ].flatMap((event) => parser.push(JSON.stringify(event)))
  assert.deepEqual(events.map((event) => [event.id, event.title, event.status]), [
    ['search:ws_1', 'Searching the web…', 'running'],
    ['search:ws_1', 'Searched the web for “BridgeMind YouTube channel”', 'done']
  ])
})
