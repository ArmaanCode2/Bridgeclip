import { execFile } from 'child_process'
import { AssistantToolError } from './tool-types'

// Finding videos on YouTube for the assistant: a channel's newest uploads and
// streams, or a search. yt-dlp reads YouTube's own listing pages (metadata
// only, flat, no downloads) with the user's config, cookies and other sites
// switched off, so a tool call can only ever reach YouTube.

export interface YouTubeVideoResult {
  title: string
  url: string
  channel: string | null
  /** YYYY-MM-DD. YouTube shows listings as "3 days ago", so it's approximate. */
  publishedOn: string | null
  durationSeconds: number | null
  views: number | null
  /** Streams that are live, scheduled or still processing can't be clipped yet. */
  live: 'live now' | 'upcoming' | 'processing' | 'was live' | null
}

export interface YouTubeChannelResult {
  name: string
  handle: string | null
  url: string
  subscribers: number | null
}

export type YtDlpRunner = (args: string[]) => Promise<string>

const MAX_RESULTS = 20
const HANDLE = /^@[A-Za-z0-9._-]{3,100}$/
const CHANNEL_ID = /^UC[A-Za-z0-9_-]{22}$/
const YOUTUBE_HOSTS = new Set(['youtube.com', 'www.youtube.com', 'm.youtube.com'])
/** YouTube's search filters: channels only, and videos sorted by upload date. */
const CHANNEL_FILTER = 'EgIQAg=='
const NEWEST_FIRST = 'CAI='

function clean(value: unknown, max = 300): string | null {
  if (typeof value !== 'string') return null
  const text = Array.from(value).map((char) => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127 ? ' ' : char).join('').trim()
  return text ? text.slice(0, max) : null
}

function count(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.round(value) : null
}

/** A channel's page from a handle, channel link or id; null when the input is a name to search for. */
export function channelUrl(input: string): string | null {
  const value = input.trim()
  if (HANDLE.test(value)) return `https://www.youtube.com/${value}`
  if (CHANNEL_ID.test(value)) return `https://www.youtube.com/channel/${value}`
  let url: URL
  try { url = new URL(/^https?:\/\//i.test(value) ? value : `https://${value}`) } catch { return null }
  if (!YOUTUBE_HOSTS.has(url.hostname.toLowerCase())) return null
  const [first, second] = url.pathname.split('/').filter(Boolean)
  if (first && HANDLE.test(decodeURIComponent(first))) return `https://www.youtube.com/${decodeURIComponent(first)}`
  if (first === 'channel' && second && CHANNEL_ID.test(second)) return `https://www.youtube.com/channel/${second}`
  if ((first === 'c' || first === 'user') && second && /^[A-Za-z0-9._-]{1,100}$/.test(second)) return `https://www.youtube.com/${first}/${second}`
  return null
}

function liveState(status: unknown): YouTubeVideoResult['live'] {
  switch (status) {
    case 'is_live': return 'live now'
    case 'is_upcoming': return 'upcoming'
    case 'post_live': return 'processing'
    case 'was_live': return 'was live'
    default: return null
  }
}

/** yt-dlp `--flat-playlist -j` output: one JSON object per line. Unusable lines are skipped. */
export function parseVideoLines(stdout: string, fallbackChannel: string | null = null): (YouTubeVideoResult & { timestamp: number | null })[] {
  const videos: (YouTubeVideoResult & { timestamp: number | null })[] = []
  for (const line of stdout.split('\n')) {
    if (!line.trim().startsWith('{')) continue
    let entry: Record<string, unknown>
    try { entry = JSON.parse(line) as Record<string, unknown> } catch { continue }
    const id = typeof entry.id === 'string' && /^[A-Za-z0-9_-]{11}$/.test(entry.id) ? entry.id : null
    const title = clean(entry.title)
    if (!id || !title || entry.ie_key !== 'Youtube') continue
    const timestamp = count(entry.timestamp)
    videos.push({
      title,
      url: `https://www.youtube.com/watch?v=${id}`,
      channel: clean(entry.channel ?? entry.playlist_channel ?? entry.uploader) ?? fallbackChannel,
      publishedOn: timestamp ? new Date(timestamp * 1000).toISOString().slice(0, 10) : null,
      durationSeconds: count(entry.duration),
      views: count(entry.view_count),
      live: liveState(entry.live_status),
      timestamp
    })
  }
  return videos
}

export function parseChannelLines(stdout: string): YouTubeChannelResult[] {
  const channels: YouTubeChannelResult[] = []
  for (const line of stdout.split('\n')) {
    if (!line.trim().startsWith('{')) continue
    let entry: Record<string, unknown>
    try { entry = JSON.parse(line) as Record<string, unknown> } catch { continue }
    const id = typeof entry.channel_id === 'string' && CHANNEL_ID.test(entry.channel_id) ? entry.channel_id : null
    const name = clean(entry.title ?? entry.channel)
    if (entry.ie_key !== 'YoutubeTab' || !id || !name) continue
    const handle = typeof entry.uploader_id === 'string' && HANDLE.test(entry.uploader_id) ? entry.uploader_id : null
    channels.push({ name, handle, url: handle ? `https://www.youtube.com/${handle}` : `https://www.youtube.com/channel/${id}`, subscribers: count(entry.channel_follower_count) })
  }
  return channels
}

const BASE_ARGS = [
  '--ignore-config', '--no-cache-dir', '--no-warnings', '--flat-playlist', '--dump-json',
  // YouTube only: a URL can never lead yt-dlp to another site or a generic page fetch.
  '--use-extractors', 'youtube,youtube:tab,youtube:search_url,youtube:user',
  // Listings say "3 days ago"; this turns that into a date so streams and uploads can be merged.
  '--extractor-args', 'youtubetab:approximate_date',
  '--socket-timeout', '10', '--retries', '1', '--extractor-retries', '1'
]

function listing(run: YtDlpRunner, url: string, limit: number): Promise<string> {
  return run([...BASE_ARGS, '--playlist-end', String(limit), '--', url])
}

function searchUrl(query: string, filter?: string): string {
  const params = new URLSearchParams({ search_query: query })
  if (filter) params.set('sp', filter)
  return `https://www.youtube.com/results?${params.toString()}`
}

function withoutTimestamp(video: YouTubeVideoResult & { timestamp: number | null }): YouTubeVideoResult {
  const { title, url, channel, publishedOn, durationSeconds, views, live } = video
  return { title, url, channel, publishedOn, durationSeconds, views, live }
}

export interface FindVideosInput {
  channel?: string
  query?: string
  limit?: number
  sort?: 'newest' | 'relevance'
}

export interface FindVideosResult {
  channel: YouTubeChannelResult | null
  /** Other channels that matched a channel name, in case the first was the wrong one. */
  otherChannels: YouTubeChannelResult[]
  videos: YouTubeVideoResult[]
  summary: string
}

/** A channel's newest uploads and streams, newest first, or YouTube search results. */
export async function findYouTubeVideos(input: FindVideosInput, run: YtDlpRunner): Promise<FindVideosResult> {
  const limit = Math.max(1, Math.min(MAX_RESULTS, Math.round(input.limit ?? 8)))
  const channelInput = input.channel?.trim()
  const query = input.query?.trim()
  if (!channelInput && !query) throw new AssistantToolError('Give a channel (handle, link or name) or a search query.')

  if (channelInput) {
    let base = channelUrl(channelInput)
    let channel: YouTubeChannelResult | null = null
    let otherChannels: YouTubeChannelResult[] = []
    if (!base) {
      // A name: take YouTube's best channel match, and report the runners-up.
      const matches = parseChannelLines(await listing(run, searchUrl(channelInput, CHANNEL_FILTER), 4))
      if (!matches.length) throw new AssistantToolError(`No YouTube channel matches “${channelInput}”. Try its @handle or channel link.`)
      ;[channel, ...otherChannels] = matches
      base = channel.url
    }
    // Uploads and live streams are separate tabs; a channel may have no streams tab.
    const [uploads, streams] = await Promise.all([
      listing(run, `${base}/videos`, limit).catch((error: unknown) => {
        if (error instanceof Error && /404|does not exist|not found/i.test(error.message)) {
          throw new AssistantToolError(`There’s no YouTube channel at ${channelInput}. Check the @handle, or look it up by name.`)
        }
        throw error
      }),
      listing(run, `${base}/streams`, limit).catch(() => '')
    ])
    const name = channel?.name ?? null
    const merged = [...parseVideoLines(uploads, name), ...parseVideoLines(streams, name)]
    // Newest first. A stream that's live, scheduled or processing has no date yet but is the newest;
    // other undated entries keep their listing order after dated ones.
    const rank = (video: YouTubeVideoResult & { timestamp: number | null }): number =>
      video.live === 'live now' || video.live === 'upcoming' || video.live === 'processing' ? Number.MAX_SAFE_INTEGER : video.timestamp ?? -1
    const videos = merged
      .map((video, order) => ({ video, order }))
      .sort((a, b) => rank(b.video) - rank(a.video) || a.order - b.order)
      .map(({ video }) => video)
      .filter((video, index, all) => all.findIndex((other) => other.url === video.url) === index)
      .slice(0, limit)
      .map(withoutTimestamp)
    const label = channel?.name ?? videos.find((video) => video.channel)?.channel ?? channelInput
    if (!channel && videos[0]?.channel) channel = { name: videos[0].channel, handle: /\/(@[^/]+)$/.exec(base)?.[1] ?? null, url: base, subscribers: null }
    return {
      channel,
      otherChannels,
      videos,
      summary: videos.length ? `${videos.length} from ${label} · newest: ${videos[0].title}${videos[0].live ? ` (${videos[0].live})` : ''}` : `No videos found on ${label}`
    }
  }

  const sort = input.sort ?? 'relevance'
  const videos = parseVideoLines(await listing(run, searchUrl(query!, sort === 'newest' ? NEWEST_FIRST : undefined), limit))
    .slice(0, limit)
    .map(withoutTimestamp)
  return { channel: null, otherChannels: [], videos, summary: `${videos.length} result${videos.length === 1 ? '' : 's'} for “${query}”` }
}

/** Runs yt-dlp with a time and output limit; its errors are reduced to something the agent can act on. */
export function ytDlpRunner(binary: string): YtDlpRunner {
  return (args) => new Promise((resolve, reject) => {
    execFile(binary, args, { timeout: 30000, maxBuffer: 8 * 1024 * 1024, windowsHide: true }, (error, stdout, stderr) => {
      if (!error) return resolve(stdout)
      const message = String(stderr || '').split('\n').find((line) => line.startsWith('ERROR:'))?.replace(/^ERROR:\s*(\[[^\]]+\]\s*)?/, '')
      reject(new AssistantToolError(message ? `YouTube: ${message.slice(0, 300)}` : 'YouTube couldn’t be reached. Check the connection and try again.'))
    })
  })
}
