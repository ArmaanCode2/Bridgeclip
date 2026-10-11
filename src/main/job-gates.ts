import { setupGaps, type AiProvider, type LocalWhisperMode, type ProviderSetupInput, type TranscriptionSource } from '../shared/providers'
import { WHISPER_SETUP_MESSAGES, isWhisperModelName, type LocalWhisperStatus } from '../shared/whisper-models'

/** The settings a job start depends on. A subset of AppSettings, so tests need not build the rest. */
export interface JobGateSettings {
  aiProvider: AiProvider
  openrouterApiKey: string
  geminiApiKey: string
  openaiCompatibleUrl: string
  openaiCompatibleModel: string
  openaiCompatibleVision: string
  transcriptionSource: TranscriptionSource
  localWhisperMode: LocalWhisperMode
  localWhisperUrl: string
}

export type ClippingMode = 'quality' | 'economy' | 'advanced'

export interface JobGateRequest {
  workflow?: string
  clippingMode?: string
  layoutVision?: boolean
}

export type JobGateResult =
  | { ok: true; clippingMode: ClippingMode; layoutVision: boolean | undefined }
  | { ok: false; error: string; missing: string }

export function setupInputFromSettings(settings: JobGateSettings): ProviderSetupInput {
  return {
    aiProvider: settings.aiProvider,
    openrouterConfigured: Boolean(settings.openrouterApiKey),
    geminiConfigured: Boolean(settings.geminiApiKey),
    openaiCompatibleUrl: settings.openaiCompatibleUrl,
    openaiCompatibleModel: settings.openaiCompatibleModel,
    transcriptionSource: settings.transcriptionSource,
    localWhisperMode: settings.localWhisperMode,
    localWhisperUrl: settings.localWhisperUrl
  }
}

/**
 * Decides whether a clip job may start with the saved provider choices, and what
 * the job runs with. Both the Create page and the assistant start jobs through
 * this, so the renderer cannot be the only thing keeping a job off a provider
 * that cannot run it.
 *
 * - Keys and URLs the chosen providers need must be set.
 * - Review & edit runs on any provider: Jev on OpenRouter, the chosen model otherwise.
 * - Advanced mode picks models from OpenRouter's catalog, so it needs OpenRouter.
 * - Economy is an OpenRouter model preset; elsewhere it runs as Quality, with the
 *   saved model, and without the frame analysis the person asked to skip.
 * - A custom OpenAI-compatible model reads video frames only if the person said it can.
 */
export function gateJobStart(settings: JobGateSettings, request: JobGateRequest): JobGateResult {
  const gap = setupGaps(setupInputFromSettings(settings))[0]
  if (gap) {
    const error = gap.id === 'openrouter-key' && settings.aiProvider === 'openrouter'
      ? 'OpenRouter API key is required for AI clip planning. Go to Settings to add it.'
      : `${gap.message} Open Settings to finish setup.`
    return { ok: false, error, missing: gap.id }
  }
  const openrouter = settings.aiProvider === 'openrouter'
  const requested: ClippingMode = request.clippingMode === 'economy' || request.clippingMode === 'advanced' ? request.clippingMode : 'quality'
  if (!openrouter && requested === 'advanced') {
    return {
      ok: false,
      missing: 'advanced-needs-openrouter',
      error: 'Advanced mode picks models from the OpenRouter catalog. Use Quality with your own model, or switch the AI provider to OpenRouter.'
    }
  }
  const clippingMode: ClippingMode = !openrouter && requested === 'economy' ? 'quality' : requested
  const analyzesFrames = openrouter || settings.aiProvider === 'gemini' || settings.openaiCompatibleVision === 'on'
  const layoutVision = !analyzesFrames ? false : request.layoutVision
  return { ok: true, clippingMode, layoutVision }
}

/**
 * Why a job using built-in Whisper cannot start, or null. Without Whisper the job fails once
 * transcription begins; without the chosen model it would stall on a silent download of up to
 * 3 GB. An unknown model state (null) is let through.
 */
export function localWhisperBlock(status: LocalWhisperStatus, model: string): string | null {
  if (!status.available) return `${status.detail} Open Settings → AI & transcription to finish setup.`
  // Older names such as large-v2 still load, but cannot be installed from Settings, so a job
  // would download them silently.
  if (!isWhisperModelName(model)) return 'Choose a Whisper model from the list in Settings → AI & transcription, and install it there.'
  if (status.modelCached === false) {
    return `The Whisper model “${model}” is not downloaded yet. Install it in Settings → AI & transcription, then start again.`
  }
  return null
}

/** Why no job may start while Whisper setup runs, or null. Enable GPU rewrites the libraries a job loads. */
export function whisperSetupBlock(task: 'download' | 'gpu' | null): string | null {
  return task === 'gpu' ? WHISPER_SETUP_MESSAGES.gpuRunning : null
}
