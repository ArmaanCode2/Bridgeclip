import { JEV_DEFAULTS, JEV_FEATURE_DEFAULTS } from '../../shared/jev-settings'
import { DEFAULT_GEMINI_MODEL, DEFAULT_LOCAL_WHISPER_MODEL, DEFAULT_OPENAI_COMPATIBLE_TIMEOUT, describeGaps, metadataWriterGap, setupGaps, type SetupGap } from '../../shared/providers'
import { DEFAULT_SCREEN_FRAMING, DEFAULT_WEBCAM_LOCATION, DEFAULT_WEBCAM_POSITION } from '../../shared/framing'
import { DEFAULT_VIDEO_ENCODER } from '../../shared/video-encoder'
import { create } from 'zustand'
import { errorMessage } from '../lib/utils'
import { getApi } from '../lib/ipc'
import type { ClipSettings, ToolStatus } from '../../preload/index'

interface SettingsState extends ClipSettings {
  loaded: boolean
  saving: boolean
  toolStatus: ToolStatus | null
  checkingTools: boolean
  toolError: string | null
  load: () => Promise<void>
  /**
   * Saves a change. A function receives the settings as they are once earlier saves have landed, so a
   * change that depends on current values (like moving to a model that fits) is never based on a stale read.
   */
  save: (settings: SettingsPatch) => Promise<void>
  replaceApiKey: (key: 'openrouterApiKey' | 'zernioApiKey' | 'geminiApiKey' | 'openaiCompatibleApiKey', value: string) => Promise<void>
  checkTools: () => Promise<void>
}

export type SettingsPatch = Partial<ClipSettings> | ((current: ClipSettings) => Partial<ClipSettings>)

// Queue writes so each partial update merges with the last successful save.
let saveQueue: Promise<void> = Promise.resolve()
let pendingSaves = 0
let latestToolCheck = 0

export const useSettingsStore = create<SettingsState>((set, get) => ({
  openrouterConfigured: false,
  zernioConfigured: false,
  geminiConfigured: false,
  openaiCompatibleConfigured: false,
  aiProvider: 'openrouter',
  geminiModel: DEFAULT_GEMINI_MODEL,
  openaiCompatibleUrl: '',
  openaiCompatibleModel: '',
  openaiCompatibleVision: 'off',
  openaiCompatibleTimeout: DEFAULT_OPENAI_COMPATIBLE_TIMEOUT,
  transcriptionSource: 'cloud',
  localWhisperMode: 'builtin',
  localWhisperUrl: '',
  localWhisperModel: DEFAULT_LOCAL_WHISPER_MODEL,
  localWhisperDevice: 'auto',
  localWhisperLanguage: 'auto',
  webcamPosition: DEFAULT_WEBCAM_POSITION,
  webcamLocation: DEFAULT_WEBCAM_LOCATION,
  screenFraming: DEFAULT_SCREEN_FRAMING,
  videoEncoder: DEFAULT_VIDEO_ENCODER,
  clipJudge: 'on',
  ...JEV_DEFAULTS,
  ...JEV_FEATURE_DEFAULTS,
  outputDirectory: '',
  pythonPath: 'python3',
  customVocabulary: '',
  loaded: false,
  saving: false,
  toolStatus: null,
  checkingTools: false,
  toolError: null,

  load: async () => {
    const settings = await getApi().settings.load()
    set({ ...pickSettings(settings), loaded: true })
  },

  save: (updates) => {
    // An object is copied now; a function is called when its turn in the queue comes.
    const snapshot = typeof updates === 'function' ? null : { ...updates }
    pendingSaves += 1
    set({ saving: true })
    const task = saveQueue.then(async () => {
      const current = pickSettings(get())
      const patch = typeof updates === 'function' ? updates(current) : snapshot
      const merged: ClipSettings = { ...current, ...patch }
      const saved = await getApi().settings.save(merged)
      set({ ...pickSettings(saved), loaded: true })
    })
    saveQueue = task.catch(() => {})
    return task.finally(() => {
      pendingSaves -= 1
      set({ saving: pendingSaves > 0 })
    })
  },

  replaceApiKey: (key, value) => {
    pendingSaves += 1
    set({ saving: true })
    const task = saveQueue.then(async () => {
      const saved = await getApi().settings.replaceApiKey(key, value)
      set({ ...pickSettings(saved), loaded: true })
    })
    saveQueue = task.catch(() => {})
    return task.finally(() => {
      pendingSaves -= 1
      set({ saving: pendingSaves > 0 })
    })
  },

  checkTools: async () => {
    const request = ++latestToolCheck
    set({ checkingTools: true, toolError: null })
    try {
      const status = await getApi().system.checkTools()
      if (request === latestToolCheck) set({ toolStatus: status })
    } catch (err) {
      if (request === latestToolCheck) {
        set({ toolStatus: null, toolError: errorMessage(err, 'Could not check required tools. Try again in Settings.') })
      }
    } finally {
      if (request === latestToolCheck) set({ checkingTools: false })
    }
  }
}))

function pickSettings(s: ClipSettings): ClipSettings {
  return {
    jevThreshold: s.jevThreshold ?? JEV_DEFAULTS.jevThreshold,
    jevSelfContainedThreshold: s.jevSelfContainedThreshold ?? JEV_DEFAULTS.jevSelfContainedThreshold,
    jevFaithfulToSourceThreshold: s.jevFaithfulToSourceThreshold ?? JEV_DEFAULTS.jevFaithfulToSourceThreshold,
    jevTitleSupportedThreshold: s.jevTitleSupportedThreshold ?? JEV_DEFAULTS.jevTitleSupportedThreshold,
    jevSponsorThreshold: s.jevSponsorThreshold ?? JEV_DEFAULTS.jevSponsorThreshold,
    jevEvidenceThreshold: s.jevEvidenceThreshold ?? JEV_DEFAULTS.jevEvidenceThreshold,
    jevCutThreshold: s.jevCutThreshold ?? JEV_DEFAULTS.jevCutThreshold,
    openrouterConfigured: s.openrouterConfigured,
    zernioConfigured: s.zernioConfigured,
    geminiConfigured: s.geminiConfigured,
    openaiCompatibleConfigured: s.openaiCompatibleConfigured,
    aiProvider: s.aiProvider ?? 'openrouter',
    geminiModel: s.geminiModel ?? DEFAULT_GEMINI_MODEL,
    openaiCompatibleUrl: s.openaiCompatibleUrl ?? '',
    openaiCompatibleModel: s.openaiCompatibleModel ?? '',
    openaiCompatibleVision: s.openaiCompatibleVision ?? 'off',
    openaiCompatibleTimeout: s.openaiCompatibleTimeout ?? DEFAULT_OPENAI_COMPATIBLE_TIMEOUT,
    transcriptionSource: s.transcriptionSource ?? 'cloud',
    localWhisperMode: s.localWhisperMode ?? 'builtin',
    localWhisperUrl: s.localWhisperUrl ?? '',
    localWhisperModel: s.localWhisperModel ?? DEFAULT_LOCAL_WHISPER_MODEL,
    localWhisperDevice: s.localWhisperDevice ?? 'auto',
    localWhisperLanguage: s.localWhisperLanguage ?? 'auto',
    webcamPosition: s.webcamPosition ?? DEFAULT_WEBCAM_POSITION,
    webcamLocation: s.webcamLocation ?? DEFAULT_WEBCAM_LOCATION,
    screenFraming: s.screenFraming ?? DEFAULT_SCREEN_FRAMING,
    videoEncoder: s.videoEncoder ?? DEFAULT_VIDEO_ENCODER,
    clipJudge: s.clipJudge ?? 'on',
    jevEnabled: s.jevEnabled ?? JEV_FEATURE_DEFAULTS.jevEnabled,
    jevVisualContext: s.jevVisualContext ?? JEV_FEATURE_DEFAULTS.jevVisualContext,
    sourceContextWebResearch: s.sourceContextWebResearch ?? JEV_FEATURE_DEFAULTS.sourceContextWebResearch,
    outputDirectory: s.outputDirectory,
    pythonPath: s.pythonPath,
    customVocabulary: s.customVocabulary
  }
}

export type SetupState = {
  ready: boolean
  /** Short names of what is still missing ("OpenRouter key"), for labels. */
  missingKeys: string[]
  /** The same gaps with a plain sentence each, for the setup card and the blocked message. */
  gaps: SetupGap[]
  /** "Add OpenRouter key", or "Finish setup" when several things are missing. */
  summary: string
  toolsOk: boolean | null
}

/** Whether a clip job can start: the chosen AI provider and transcription are set up and, once the
 *  system check has run, every required tool found. */
export function useSetupState(): SetupState {
  const aiProvider = useSettingsStore((s) => s.aiProvider)
  const openrouterConfigured = useSettingsStore((s) => s.openrouterConfigured)
  const geminiConfigured = useSettingsStore((s) => s.geminiConfigured)
  const openaiCompatibleUrl = useSettingsStore((s) => s.openaiCompatibleUrl)
  const openaiCompatibleModel = useSettingsStore((s) => s.openaiCompatibleModel)
  const transcriptionSource = useSettingsStore((s) => s.transcriptionSource)
  const localWhisperMode = useSettingsStore((s) => s.localWhisperMode)
  const localWhisperUrl = useSettingsStore((s) => s.localWhisperUrl)
  const tools = useSettingsStore((s) => s.toolStatus)
  const toolError = useSettingsStore((s) => s.toolError)
  const checkingTools = useSettingsStore((s) => s.checkingTools)

  const gaps = setupGaps({
    aiProvider, openrouterConfigured, geminiConfigured, openaiCompatibleUrl, openaiCompatibleModel,
    transcriptionSource, localWhisperMode, localWhisperUrl
  })
  const toolsOk = toolError ? false : tools
    ? tools.python && tools.pythonDeps && tools.ffmpeg && tools.ffprobe && tools.ytdlp && tools.engine && tools.bridgeRunner
    : null
  return {
    ready: gaps.length === 0 && toolsOk === true && !checkingTools,
    missingKeys: gaps.map((gap) => gap.label),
    gaps,
    summary: describeGaps(gaps),
    toolsOk
  }
}

/** Why titles and captions cannot be written right now, or null. Main checks the same rule. */
export function useMetadataWriterGap(): string | null {
  return useSettingsStore((s) => metadataWriterGap({
    aiProvider: s.aiProvider, openrouterConfigured: s.openrouterConfigured, geminiConfigured: s.geminiConfigured,
    openaiCompatibleUrl: s.openaiCompatibleUrl, openaiCompatibleModel: s.openaiCompatibleModel,
    transcriptionSource: s.transcriptionSource, localWhisperMode: s.localWhisperMode, localWhisperUrl: s.localWhisperUrl
  }))
}

/** Resolves once every settings write queued so far has finished, so a check that reads
 *  saved settings (Test connection) sees what the person just typed. */
export function whenSettingsSaved(): Promise<void> {
  return saveQueue
}
