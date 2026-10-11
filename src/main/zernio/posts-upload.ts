import type { FileHandle } from 'fs/promises'
import { readResponseText } from '../http-response'
import { lookup as dnsLookup } from 'dns'
import { Readable, Transform } from 'stream'
import { Agent } from 'undici'
import { isPublicAddress } from '../network-policy'
import { isUploadUrl, ZernioApiError } from './client'

export interface UploadOptions {
  contentType: string
  size: number
  onProgress?: (sent: number, total: number) => void
  signal?: AbortSignal
}

export class ZernioUploadCancelledError extends ZernioApiError {
  constructor() {
    super('Upload cancelled.', 0)
  }
}

const STALL_MS = 60_000
/** Once every byte is sent, storage gets this long to confirm. */
const CONFIRM_MS = 120_000
/** Presigned upload URLs expire after an hour. */
const MAX_MS = 55 * 60_000

/** Container types Zernio's presign accepts for the clip formats BridgeClip writes. */
const VIDEO_TYPES: Record<string, string> = {
  '.mp4': 'video/mp4',
  '.m4v': 'video/x-m4v',
  '.mov': 'video/quicktime',
  '.webm': 'video/webm'
}

export function videoContentType(filePath: string): string | null {
  const match = /\.[^./\\]+$/.exec(filePath)
  return match ? VIDEO_TYPES[match[0].toLowerCase()] ?? null : null
}

/**
 * Stream a file to a presigned storage URL. The request carries no
 * Authorization header (storage authenticates by the signature in the URL),
 * reports progress as bytes leave the file, and aborts when nothing moves
 * for a minute.
 */
export async function putFile(uploadUrl: string, source: FileHandle, options: UploadOptions): Promise<void> {
  if (options.signal?.aborted) throw new ZernioUploadCancelledError()
  const secure = isUploadUrl(uploadUrl)
  if (!secure && !(uploadUrl.startsWith('http://') && isUploadUrl(uploadUrl, true))) {
    throw new ZernioApiError('Zernio did not return a secure upload link.', 502)
  }
  // Resolve at socket creation and pass only the checked address to connect.
  // Undici retains the original hostname for TLS SNI and certificate checks.
  const dispatcher = secure ? new Agent({ connect: {
    lookup(hostname, options, callback) {
      dnsLookup(hostname, { all: true }, (error, addresses) => {
        const publicAddresses = addresses?.filter(({ address }) => isPublicAddress(address)) ?? []
        if (error || publicAddresses.length === 0) {
          callback(new Error('Upload destination is not public'), '', 0)
          return
        }
        // Undici requests all addresses for family selection. Returning one
        // string to that callback fails with ERR_INVALID_IP_ADDRESS.
        // Pass only validated addresses so mixed DNS answers cannot connect
        // to a private address.
        if (options.all) callback(null, publicAddresses)
        else callback(null, publicAddresses[0].address, publicAddresses[0].family)
      })
    }
  } }) : null
  const controller = new AbortController()
  const cancel = (): void => controller.abort()
  options.signal?.addEventListener('abort', cancel, { once: true })

  let sent = 0
  let lastActivity = Date.now()
  let stalled = false
  const watchdog = setInterval(() => {
    if (Date.now() - lastActivity > (sent >= options.size ? CONFIRM_MS : STALL_MS)) {
      stalled = true
      controller.abort()
    }
  }, 2_000)
  const deadline = setTimeout(() => {
    stalled = true
    controller.abort()
  }, MAX_MS)

  const file = source.createReadStream({ highWaterMark: 256 * 1024, autoClose: false })
  const counter = new Transform({
    transform(chunk: Buffer, _encoding, callback) {
      sent += chunk.length
      lastActivity = Date.now()
      options.onProgress?.(sent, options.size)
      callback(null, chunk)
    }
  })
  file.on('error', (error) => counter.destroy(error))

  try {
    const response = await fetch(uploadUrl, {
      method: 'PUT',
      // Storage needs the exact length (no chunked encoding) and the type the presign named.
      headers: { 'Content-Type': options.contentType, 'Content-Length': String(options.size) },
      body: Readable.toWeb(file.pipe(counter)) as unknown as BodyInit,
      duplex: 'half',
      redirect: 'error',
      signal: controller.signal,
      ...(dispatcher ? { dispatcher } : {})
    } as RequestInit & { duplex: 'half'; dispatcher?: Agent })
    await readResponseText(response, 64 * 1024)
    if (!response.ok) throw new ZernioApiError(`Uploading the clip to Zernio's storage failed (HTTP ${response.status}).`, response.status)
  } catch (error) {
    if (error instanceof ZernioApiError) throw error
    if (options.signal?.aborted) throw new ZernioUploadCancelledError()
    throw new ZernioApiError(
      stalled ? 'The upload stalled. Check your connection and try again.' : 'The upload was interrupted. Check your connection and try again.',
      0
    )
  } finally {
    clearInterval(watchdog)
    clearTimeout(deadline)
    options.signal?.removeEventListener('abort', cancel)
    file.destroy()
    await dispatcher?.close()
  }
}
