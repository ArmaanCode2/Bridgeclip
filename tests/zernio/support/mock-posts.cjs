'use strict'
// Zernio's media and posts endpoints for the local mock (see mock-zernio.cjs).
// Behaviour follows docs.zernio.com (API 1.62): presign + storage PUT,
// POST /v1/posts with x-request-id replays (200 existingPost) and the 24-hour
// content dedup (409), 207 for partial/failed inline publishes, and TikTok's
// required tiktokSettings checked against creator info.
//
//   const posting = createPostingMock()
//   const mock = await createMockZernio({ apiKey, extraRoutes: posting.routes })
//   posting.state.uploads / creates / posts   (what the app sent)

const crypto = require('node:crypto')

const PRESIGN_TYPES = new Set(['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif', 'video/mp4', 'video/mpeg', 'video/quicktime', 'video/avi', 'video/x-msvideo', 'video/webm', 'video/x-m4v'])
const MAX_UPLOAD = 5 * 1024 * 1024 * 1024
const REPLAY_WINDOW_MS = 5 * 60_000
const DEDUP_WINDOW_MS = 24 * 3_600_000

const DEFAULT_CREATOR_INFO = {
  creator: { nickname: 'mock creator', avatarUrl: 'https://example.test/a.jpg', isVerified: false, canPostMore: true },
  privacyLevels: [
    { value: 'PUBLIC_TO_EVERYONE', label: 'Public To Everyone' },
    { value: 'MUTUAL_FOLLOW_FRIENDS', label: 'Mutual Follow Friends' },
    { value: 'SELF_ONLY', label: 'Self Only' }
  ],
  postingLimits: {
    maxVideoDurationSec: 600,
    interactionSettings: {
      allow_comment: { enabled: true, required: true, default: false, label: 'Allow Comment' },
      allow_duet: { enabled: true, required: true, default: false, label: 'Allow Duet' },
      allow_stitch: { enabled: false, required: true, default: false, label: 'Allow Stitch' }
    }
  },
  commercialContentTypes: [
    { value: 'none', label: 'No Commercial Content' },
    { value: 'brand_organic', label: 'Your Brand', requires: ['is_brand_organic_post'] },
    { value: 'brand_content', label: 'Branded Content', requires: ['brand_partner_promote'] }
  ]
}

function hexId() {
  return crypto.randomBytes(12).toString('hex')
}

function error(ctx, status, message, extra = {}) {
  return ctx.json(status, { error: message, type: status === 409 ? 'invalid_request_error' : 'invalid_request_error', ...extra })
}

function postUrl(platform, id) {
  switch (platform) {
    case 'youtube': return `https://www.youtube.com/shorts/${id.slice(0, 11)}`
    case 'instagram': return `https://www.instagram.com/reel/${id.slice(0, 11)}/`
    case 'facebook': return `https://www.facebook.com/reel/${parseInt(id.slice(0, 12), 16)}`
    case 'twitter': return `https://x.com/mock/status/${parseInt(id.slice(0, 12), 16)}`
    case 'linkedin': return `https://www.linkedin.com/feed/update/urn:li:share:${parseInt(id.slice(0, 12), 16)}`
    case 'threads': return `https://www.threads.net/@mock/post/${id.slice(0, 11)}`
    // TikTok resolves the link minutes later; GET /posts fills it in (see resolveTikTokLinks).
    default: return null
  }
}

function createPostingMock(options = {}) {
  const state = {
    /** Presigned uploads: key -> { publicUrl, contentType, size } */
    presigned: new Map(),
    /** Completed storage PUTs: { key, bytes, sha256, contentType, contentLength, transferEncoding, authorization } */
    uploads: [],
    /** Every POST /v1/posts: { body, requestId, status } */
    creates: [],
    /** Posts by id (Zernio's shape, accountId stored as a plain id). */
    posts: new Map(),
    /** x-request-id -> { postId, at } */
    requestIds: new Map(),
    /** accountId -> creator info body (defaults to DEFAULT_CREATOR_INFO). */
    creatorInfo: {},
    /** Per-platform outcome of the next inline publish: platform -> { status: 'failed', errorMessage } */
    nextPublish: {},
    /** Answer the next storage PUT with this status. */
    failNextUpload: null,
    /** Hand out an upload URL on this origin instead of the mock (e.g. 'http://evil.example'). */
    uploadOrigin: options.uploadOrigin ?? null,
    /** Delay (ms) before answering POST /v1/posts, to exercise timeouts. */
    createDelayMs: 0
  }

  const view = (ctx, post) => ({
    ...post,
    platforms: post.platforms.map((p) => {
      const account = ctx.state.accounts.find((a) => a._id === p.accountId)
      return { ...p, accountId: account ? { _id: account._id, platform: account.platform, username: account.username, displayName: account.displayName, isActive: account.isActive } : p.accountId }
    })
  })

  function aggregate(platforms) {
    const published = platforms.filter((p) => p.status === 'published').length
    const failed = platforms.filter((p) => p.status === 'failed').length
    if (failed === 0) return 'published'
    return published > 0 ? 'partial' : 'failed'
  }

  function publishInline(post) {
    post.status = 'publishing'
    for (const target of post.platforms) {
      const scripted = state.nextPublish[target.platform]
      if (scripted) {
        delete state.nextPublish[target.platform]
        Object.assign(target, { status: 'failed', errorMessage: scripted.errorMessage ?? 'Publishing failed', errorCategory: scripted.errorCategory ?? 'platform_error', errorSource: 'platform' })
        continue
      }
      const draft = target.platform === 'tiktok' && post.tiktokSettings?.draft === true
      Object.assign(target, {
        status: 'published',
        publishedAt: new Date().toISOString(),
        platformPostId: hexId(),
        platformPostUrl: draft ? null : postUrl(target.platform, hexId()),
        ...(draft ? { platformSpecificData: { ...(target.platformSpecificData ?? {}), isDraft: true } } : {})
      })
    }
    post.status = aggregate(post.platforms)
    if (post.status === 'published') post.publishedAt = new Date().toISOString()
  }

  function validateCreate(ctx, body) {
    if (!Array.isArray(body.platforms) || body.platforms.length === 0) return 'platforms is required for non-draft posts'
    for (const [i, target] of body.platforms.entries()) {
      const account = ctx.state.accounts.find((a) => a._id === target.accountId)
      if (!account) return { status: 403, message: `Account ${target.accountId} does not belong to you` }
      if (account.platform !== target.platform) return `platforms[${i}].platform does not match the account`
      if (target.platform === 'youtube') {
        const title = target.platformSpecificData?.title
        if (typeof title === 'string' && [...title].length > 100) return 'YouTube title must be 100 characters or fewer'
      }
      if (target.platform === 'tiktok') {
        const s = { ...(body.tiktokSettings ?? {}), ...(target.platformSpecificData?.tiktokSettings ?? {}) }
        const info = state.creatorInfo[target.accountId] ?? DEFAULT_CREATOR_INFO
        const privacy = s.privacy_level ?? s.privacyLevel
        if (!info.privacyLevels.some((l) => l.value === privacy)) return 'tiktokSettings.privacy_level must be one of the creator info privacy levels'
        for (const key of ['allow_comment', 'allow_duet', 'allow_stitch']) {
          if (typeof s[key] !== 'boolean') return `tiktokSettings.${key} is required`
        }
        if (s.content_preview_confirmed !== true || s.express_consent_given !== true) return 'TikTok requires content_preview_confirmed and express_consent_given to be true'
      }
    }
    const media = Array.isArray(body.mediaItems) ? body.mediaItems : []
    if (media.length === 0) return 'Media is required'
    for (const item of media) {
      if (!state.uploads.some((u) => state.presigned.get(u.key)?.publicUrl === item.url)) return `mediaItems url was never uploaded: ${item.url}`
    }
    if (!body.publishNow) {
      if (typeof body.scheduledFor !== 'string' || !Number.isFinite(Date.parse(body.scheduledFor))) return 'scheduledFor is required'
      if (body.timezone !== undefined) {
        try { new Intl.DateTimeFormat('en-US', { timeZone: body.timezone }) } catch { return 'Unknown timezone' }
      }
    }
    return null
  }

  function contentHash(target, body) {
    return crypto.createHash('sha256').update(JSON.stringify([target.platform, target.accountId, body.content ?? '', (body.mediaItems ?? []).map((m) => m.url)])).digest('hex')
  }

  const routes = [
    {
      method: 'POST',
      path: '/api/v1/media/presign',
      handler: (ctx) => {
        const { filename, contentType, size } = ctx.body ?? {}
        if (typeof filename !== 'string' || !filename) return error(ctx, 400, 'filename is required', { code: 'missing_required_field', param: 'filename' })
        if (!PRESIGN_TYPES.has(contentType)) return error(ctx, 400, 'Unsupported contentType', { code: 'INVALID_FIELD_VALUE', param: 'contentType' })
        if (size !== undefined && (!Number.isInteger(size) || size <= 0 || size > MAX_UPLOAD)) return error(ctx, 400, 'size out of range', { code: 'invalid_field_value', param: 'size' })
        const key = `temp/${Date.now()}_${hexId().slice(0, 6)}_${filename.replace(/[^\w.-]/g, '_')}`
        const origin = state.uploadOrigin ?? ctx.mock.url
        const publicUrl = `${ctx.mock.url}/media/${key}`
        state.presigned.set(key, { publicUrl, contentType, size })
        return ctx.json(200, { uploadUrl: `${origin}/upload/${key}?X-Amz-Signature=mock-signature-not-a-secret`, publicUrl, key, expiresIn: 3600 })
      }
    },
    {
      // Storage, not the API: authenticated by the signature in the URL only.
      method: 'PUT',
      path: /^\/upload\/(.+)$/,
      auth: false,
      handler: (ctx) => {
        const key = decodeURIComponent(ctx.params[0])
        const presigned = state.presigned.get(key)
        const bytes = Buffer.isBuffer(ctx.body) ? ctx.body : Buffer.from(JSON.stringify(ctx.body ?? ''))
        const record = {
          key,
          bytes: bytes.length,
          sha256: crypto.createHash('sha256').update(bytes).digest('hex'),
          contentType: ctx.req.headers['content-type'] ?? null,
          contentLength: ctx.req.headers['content-length'] ?? null,
          transferEncoding: ctx.req.headers['transfer-encoding'] ?? null,
          authorization: ctx.req.headers.authorization !== undefined,
          signed: ctx.query.get('X-Amz-Signature') === 'mock-signature-not-a-secret'
        }
        if (state.failNextUpload) {
          const status = state.failNextUpload
          state.failNextUpload = null
          return ctx.text(status, '<Error><Code>SignatureDoesNotMatch</Code></Error>')
        }
        if (!presigned || !record.signed) return ctx.text(403, '<Error><Code>AccessDenied</Code></Error>')
        if (presigned.contentType !== record.contentType) return ctx.text(403, '<Error><Code>SignatureDoesNotMatch</Code></Error>')
        state.uploads.push(record)
        return ctx.empty(200)
      }
    },
    {
      method: 'POST',
      path: '/api/v1/posts',
      handler: async (ctx) => {
        if (state.createDelayMs) await new Promise((resolve) => setTimeout(resolve, state.createDelayMs))
        const body = ctx.body && !Buffer.isBuffer(ctx.body) ? ctx.body : {}
        const requestId = ctx.req.headers['x-request-id'] ?? null
        const create = { body, requestId, status: 0 }
        state.creates.push(create)

        const replay = requestId && state.requestIds.get(requestId)
        if (replay && Date.now() - replay.at < REPLAY_WINDOW_MS) {
          create.status = 200
          return ctx.json(200, { existingPost: view(ctx, state.posts.get(replay.postId)) })
        }
        const invalid = validateCreate(ctx, body)
        if (invalid) {
          create.status = typeof invalid === 'object' ? invalid.status : 400
          return error(ctx, create.status, typeof invalid === 'object' ? invalid.message : invalid, { code: 'invalid_field_value' })
        }
        for (const target of body.platforms) {
          const hash = contentHash(target, body)
          const existing = [...state.posts.values()].find((p) => p.hashes.includes(hash) && p.status !== 'failed' && p.status !== 'cancelled' && Date.now() - Date.parse(p.createdAt) < DEDUP_WINDOW_MS)
          if (existing) {
            create.status = 409
            return ctx.json(409, {
              error: 'This exact content is already scheduled, publishing, or was posted to this account within the last 24 hours.',
              details: { accountId: target.accountId, platform: target.platform, existingPostId: existing._id }
            })
          }
        }

        const post = {
          _id: hexId(),
          content: body.content ?? '',
          mediaItems: body.mediaItems,
          platforms: body.platforms.map((t) => ({ platform: t.platform, accountId: t.accountId, platformSpecificData: t.platformSpecificData, status: 'pending' })),
          scheduledFor: body.publishNow ? undefined : new Date(Date.parse(body.scheduledFor)).toISOString(),
          timezone: body.timezone ?? 'UTC',
          status: 'scheduled',
          tiktokSettings: body.tiktokSettings,
          metadata: body.metadata,
          hashes: body.platforms.map((t) => contentHash(t, body)),
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        }
        state.posts.set(post._id, post)
        if (requestId) state.requestIds.set(requestId, { postId: post._id, at: Date.now() })

        if (body.publishNow || Date.parse(post.scheduledFor) <= Date.now()) {
          publishInline(post)
          const status = post.status === 'published' ? 201 : 207
          create.status = status
          return ctx.json(status, {
            post: view(ctx, post),
            message: status === 201 ? 'Post published successfully' : 'Post created, but publishing did not fully succeed',
            ...(status === 207 ? {
              platformResults: post.platforms.map((p) => ({ platform: p.platform, status: p.status, error: p.errorMessage ?? null })),
              ...(post.status === 'failed' ? { error: 'No platform published' } : {})
            } : {})
          })
        }
        create.status = 201
        return ctx.json(201, { message: 'Post scheduled successfully', post: view(ctx, post) })
      }
    },
    {
      method: 'GET',
      path: /^\/api\/v1\/posts\/([^/]+)$/,
      handler: (ctx) => {
        const post = state.posts.get(ctx.params[0])
        if (!post) return ctx.json(404, { error: 'Post not found', type: 'not_found', code: 'post_not_found' })
        return ctx.json(200, { post: view(ctx, post) })
      }
    },
    {
      method: 'PUT',
      path: /^\/api\/v1\/posts\/([^/]+)$/,
      handler: (ctx) => {
        const post = state.posts.get(ctx.params[0])
        if (!post) return ctx.json(404, { error: 'Post not found', type: 'not_found', code: 'post_not_found' })
        if (!['draft', 'scheduled', 'failed', 'partial', 'cancelled'].includes(post.status)) return error(ctx, 400, 'Published posts can only have their recycling config updated', { code: 'invalid_resource_state' })
        const body = ctx.body ?? {}
        if (body.scheduledFor !== undefined) {
          if (!Number.isFinite(Date.parse(body.scheduledFor))) return error(ctx, 400, 'Invalid scheduledFor', { param: 'scheduledFor' })
          post.scheduledFor = new Date(Date.parse(body.scheduledFor)).toISOString()
        }
        if (body.timezone !== undefined) post.timezone = body.timezone
        post.updatedAt = new Date().toISOString()
        return ctx.json(200, { message: 'Post updated successfully', post: { _id: post._id, content: post.content, status: post.status, scheduledFor: post.scheduledFor } })
      }
    },
    {
      method: 'DELETE',
      path: /^\/api\/v1\/posts\/([^/]+)$/,
      handler: (ctx) => {
        const post = state.posts.get(ctx.params[0])
        if (!post) return ctx.json(404, { error: 'Post not found', type: 'not_found', code: 'post_not_found' })
        if (post.status === 'published') return error(ctx, 400, 'Published posts cannot be deleted', { code: 'invalid_resource_state' })
        state.posts.delete(post._id)
        return ctx.json(200, { message: 'Post deleted successfully' })
      }
    },
    {
      method: 'POST',
      path: /^\/api\/v1\/posts\/([^/]+)\/retry$/,
      handler: (ctx) => {
        const post = state.posts.get(ctx.params[0])
        if (!post) return ctx.json(404, { error: 'Post not found', type: 'not_found', code: 'post_not_found' })
        if (post.status !== 'failed' && post.status !== 'partial') return error(ctx, 400, 'Only failed or partial posts can be retried', { code: 'invalid_resource_state' })
        for (const target of post.platforms) {
          if (target.status !== 'failed') continue
          Object.assign(target, { status: 'published', errorMessage: undefined, errorCategory: undefined, errorSource: undefined, platformPostUrl: postUrl(target.platform, hexId()), publishedAt: new Date().toISOString() })
        }
        post.status = aggregate(post.platforms)
        return ctx.json(200, { message: 'Retry successful', post: view(ctx, post) })
      }
    },
    {
      method: 'GET',
      path: /^\/api\/v1\/accounts\/([^/]+)\/tiktok\/creator-info$/,
      handler: (ctx) => {
        const id = ctx.params[0]
        const account = ctx.state.accounts.find((a) => a._id === id)
        if (!account) return ctx.json(404, { error: 'Account not found', type: 'not_found', code: 'account_not_found' })
        if (account.platform !== 'tiktok') return error(ctx, 400, 'This endpoint only works with TikTok accounts', { code: 'invalid_field_value', param: 'accountId' })
        return ctx.json(200, state.creatorInfo[id] ?? DEFAULT_CREATOR_INFO)
      }
    }
  ]

  return {
    state,
    routes,
    /** Publishes every scheduled post as if its time had come. */
    publishScheduled() {
      for (const post of state.posts.values()) if (post.status === 'scheduled') publishInline(post)
    },
    /** TikTok links resolve minutes after publishing. */
    resolveTikTokLinks() {
      for (const post of state.posts.values()) {
        for (const target of post.platforms) {
          if (target.platform === 'tiktok' && target.status === 'published' && !target.platformPostUrl && !target.platformSpecificData?.isDraft) {
            target.platformPostUrl = `https://www.tiktok.com/@mock/video/${parseInt(hexId().slice(0, 12), 16)}`
          }
        }
      }
    }
  }
}

module.exports = { createPostingMock, DEFAULT_CREATOR_INFO }
