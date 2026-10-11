import { JevSettings } from '../components/JevSettings'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { ArrowUpRight, BookA, Check, ChevronDown, Cpu, Crop, FolderOpen, Gavel, Github, History, Info, KeyRound, Loader2, RefreshCw, ScrollText, SlidersHorizontal, Sparkles } from 'lucide-react'
import { useSettingsStore } from '../store/use-settings-store'
import { useChangelogStore } from '../store/use-changelog-store'
import { useApiKeyDrafts } from '../hooks/use-api-key-drafts'
import { getApi } from '../lib/ipc'
import { cn, errorMessage, isMac } from '../lib/utils'
import { APP_NAME, APP_VERSION, BRIDGEMIND_URL, ISSUES_URL, LICENSE_NAME, PROVIDER_LINKS, REPO_URL } from '../config/brand'
import type { ClipSettings, ToolStatus } from '../../preload/index'
import { ApiKeyInput } from '../components/ApiKeyInput'
import { ProviderFields, ProviderPicker, TranscriptionSettings, type CommitSettings } from '../components/ProviderSetup'
import { AI_PROVIDER_LABELS, aiReady, transcriptionReady } from '../../shared/providers'
import { BridgeClipLogo } from '../components/brand/BridgeClipLogo'
import { Page } from '../components/ui/Page'
import { PageHeader } from '../components/ui/PageHeader'
import { Panel, PanelHeader } from '../components/ui/Panel'
import { Button } from '../components/ui/Button'
import { Field, TextArea, TextInput } from '../components/ui/Field'
import { Badge, StatusDot } from '../components/ui/Badge'
import { IconTile } from '../components/ui/IconTile'
import { Callout } from '../components/ui/Callout'
import { UpdatesRow } from '../components/Updates'
import { OutputStorage } from '../components/OutputStorage'
import { AssistantSettings } from '../components/assistant/AssistantSettings'
import { useAssistantStore } from '../store/use-assistant-store'
import { takeSettingsSection } from '../lib/settings-focus'
import { Segmented } from '../components/ui/Segmented'
import { Select } from '../components/ui/Select'
import { WEBCAM_LOCATIONS, WEBCAM_LOCATION_LABELS, type WebcamLocation } from '../../shared/framing'
import { SettingRow } from '../components/ui/SettingRow'

type SectionId = 'keys' | 'assistant' | 'jev' | 'vocabulary' | 'picking' | 'framing' | 'output' | 'system' | 'about'
type SectionTone = 'success' | 'warning' | 'danger' | 'idle'

/** `showUpdates` changes each time Help → Check for Updates… asks for the Updates row. */
export function SettingsPage({ showUpdates = 0 }: { showUpdates?: number }): React.JSX.Element {
  const {
    outputDirectory, pythonPath, customVocabulary, openrouterConfigured, zernioConfigured,
    geminiConfigured, aiProvider, openaiCompatibleUrl, openaiCompatibleModel,
    transcriptionSource, localWhisperMode, localWhisperUrl, webcamPosition, webcamLocation, screenFraming, videoEncoder, clipJudge,
    saving, save, toolStatus, toolError, checkTools, checkingTools
  } = useSettingsStore()
  const keys = useApiKeyDrafts()
  const [isPackaged, setIsPackaged] = useState(true)
  const [savedAt, setSavedAt] = useState<number | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)

  useEffect(() => {
    getApi().system.isPackaged().then(setIsPackaged).catch(() => {})
  }, [])

  const commit = async (patch: Partial<ClipSettings>, recheck = false): Promise<void> => {
    try {
      await save(patch)
      setSaveError(null)
      setSavedAt(Date.now())
      if (recheck) checkTools()
    } catch (err) {
      setSaveError(errorMessage(err, 'Could not save settings'))
    }
  }

  /** Saves a provider field. The error goes back to the field that asked, not to the header. */
  const commitProvider: CommitSettings = async (patch) => {
    try {
      await save(patch)
      setSaveError(null)
      setSavedAt(Date.now())
      return null
    } catch (err) {
      return errorMessage(err, 'Could not save this setting')
    }
  }

  const lastSaved = Math.max(savedAt ?? 0, keys.savedAt ?? 0)
  const error = saveError ?? keys.error
  const tools = toolRows(toolStatus)
  const toolsChecked = tools.every((row) => row.ok != null)
  const toolsMissing = tools.filter((row) => !row.optional && row.ok === false).length
  const setupInput = { aiProvider, openrouterConfigured, geminiConfigured, openaiCompatibleUrl, openaiCompatibleModel, transcriptionSource, localWhisperMode, localWhisperUrl }
  const activeAiConfigured = aiReady(setupInput)
  const transcriptionConfigured = transcriptionReady(setupInput)
  const keysMissing = Number(!activeAiConfigured || !transcriptionConfigured)
  const vocabularyTerms = customVocabulary.split('\n').filter((line) => line.trim()).length
  const assistantConnected = useAssistantStore((state) => Object.values(state.statuses).some((status) => status?.state === 'connected'))

  const sections: { id: SectionId; label: string; icon: ReactNode; tone: SectionTone }[] = [
    { id: 'keys', label: 'AI & transcription', icon: <KeyRound />, tone: keysMissing ? 'warning' : 'success' },
    { id: 'assistant', label: 'Assistant', icon: <Sparkles />, tone: assistantConnected ? 'success' : 'idle' },
    { id: 'jev', label: 'TypeSafe Jev', icon: <SlidersHorizontal />, tone: 'idle' },
    { id: 'vocabulary', label: 'Vocabulary', icon: <BookA />, tone: 'idle' },
    { id: 'picking', label: 'Clip picking', icon: <Gavel />, tone: 'idle' },
    { id: 'framing', label: 'Framing', icon: <Crop />, tone: 'idle' },
    { id: 'output', label: 'Output', icon: <FolderOpen />, tone: 'idle' },
    { id: 'system', label: 'System check', icon: <Cpu />, tone: !toolsChecked ? 'idle' : toolsMissing ? 'danger' : 'success' },
    { id: 'about', label: 'About', icon: <Info />, tone: 'idle' }
  ]
  const [active, jump] = useActiveSection(sections.map((section) => section.id))
  useEffect(() => {
    if (!showUpdates) return
    // After App's scroll-to-top for a page change, which runs after this effect.
    const frame = requestAnimationFrame(() => jump('about'))
    return () => cancelAnimationFrame(frame)
    // Only a new request scrolls; jump is recreated on every render.
  }, [showUpdates])
  useEffect(() => {
    const section = takeSettingsSection()
    if (!section) return
    const frame = requestAnimationFrame(() => jump(section))
    return () => cancelAnimationFrame(frame)
  }, [])

  const checks: { label: string; ok: boolean | null; detail: string; section: SectionId; optional?: boolean; tone?: 'danger' }[] = [
    {
      label: AI_PROVIDER_LABELS[aiProvider],
      ok: activeAiConfigured,
      detail: activeAiConfigured ? 'Set up' : 'Needed to pick clips',
      section: 'keys'
    },
    {
      label: transcriptionSource === 'local' ? 'Local Whisper' : 'Cloud transcription',
      ok: transcriptionConfigured,
      detail: transcriptionConfigured
        ? transcriptionSource === 'local' ? (localWhisperMode === 'server' ? 'Server set' : 'Built-in') : 'Ready'
        : transcriptionSource === 'local' ? 'Needs a server URL' : 'Needs an OpenRouter key',
      section: 'keys'
    },
    { label: 'Tools', ok: toolsChecked ? toolsMissing === 0 : null, detail: !toolsChecked ? (checkingTools ? 'Checking…' : 'Not checked') : toolsMissing ? `${toolsMissing} missing` : 'All installed', section: 'system', tone: 'danger' },
    { label: 'Zernio', ok: zernioConfigured, detail: zernioConfigured ? 'Posting on' : 'Optional, for posting', section: 'keys', optional: true }
  ]
  const blocking = checks.filter((check) => !check.optional && check.ok === false).length

  return (
    <Page width="default" className="max-w-[1080px]">
      <PageHeader
        eyebrow="Preferences"
        title="Settings"
        description="Changes save automatically. Jev thresholds use Apply."
        actions={<SaveIndicator saving={saving} savedAt={lastSaved || null} error={error} />}
      />

      <div className="mt-5 grid items-start gap-4 lg:grid-cols-[184px_minmax(0,1fr)]">
        <nav aria-label="Settings sections" className="hidden space-y-0.5 lg:sticky lg:top-12 lg:block">
          {sections.map((section) => (
            <button
              key={section.id}
              type="button"
              onClick={() => jump(section.id)}
              aria-current={active === section.id ? 'true' : undefined}
              className={cn(
                'flex h-8 w-full items-center gap-2.5 rounded-full px-3 text-left text-sm transition-colors duration-150 [&_svg]:h-3.5 [&_svg]:w-3.5',
                active === section.id ? 'bg-white/[0.09] text-ink' : 'text-ink-muted hover:bg-white/[0.05] hover:text-ink'
              )}
            >
              {section.icon}
              <span className="flex-1">{section.label}</span>
              {section.tone !== 'idle' && <StatusDot tone={section.tone} />}
            </button>
          ))}
        </nav>

        <div className="min-w-0 space-y-4">
          <Panel>
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <IconTile tone={blocking ? 'warning' : 'success'} size="lg">{blocking ? <KeyRound /> : <Check strokeWidth={3} />}</IconTile>
                <div>
                  <h2 className="text-sm font-semibold text-ink">{blocking ? `${blocking} thing${blocking === 1 ? '' : 's'} to set up before clipping` : 'Ready to clip'}</h2>
                  <p className="mt-0.5 text-xs text-ink-muted">
                    {APP_NAME} runs on this computer. {aiProvider === 'openrouter' && transcriptionSource === 'cloud'
                      ? 'One OpenRouter key covers transcription and clip selection.'
                      : `${AI_PROVIDER_LABELS[aiProvider]} picks the clips. ${transcriptionSource === 'local' ? 'Whisper on this computer does the transcription.' : 'OpenRouter does the transcription.'}`}
                  </p>
                </div>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3">
              {checks.map((check) => (
                <button
                  key={check.label}
                  type="button"
                  onClick={() => jump(check.section)}
                  className="glass-tile glass-tile-hover flex items-start gap-2.5 rounded-xl px-3 py-2 text-left"
                >
                  <StatusDot className="mt-1.5" tone={check.ok == null || check.optional && !check.ok ? 'idle' : check.ok ? 'success' : check.tone ?? 'warning'} />
                  <span className="min-w-0">
                    <span className="block text-sm font-medium text-ink">{check.label}</span>
                    <span className={cn('block truncate text-2xs', check.ok === false && !check.optional ? (check.tone === 'danger' ? 'text-danger' : 'text-warning') : 'text-ink-subtle')}>{check.detail}</span>
                  </span>
                </button>
              ))}
            </div>
          </Panel>

          <Section id="keys">
            <PanelHeader
              icon={<IconTile tone="accent"><KeyRound /></IconTile>}
              title="AI provider & transcription"
              description="Keys are encrypted with your system keychain. BridgeClip has no account and no server of its own."
            />

            <div className="mt-4">
              <p className="eyebrow mb-2 px-1">Who picks your clips</p>
              <ProviderPicker commit={commitProvider} />
            </div>

            <div className="mt-4">
              <ProviderFields commit={commitProvider} keys={keys} />
            </div>

            <div className="mt-6 rounded-2xl border border-white/[0.08] bg-white/[0.02] p-4">
              <TranscriptionSettings commit={commitProvider} keys={keys} />
            </div>

            <p className="eyebrow px-1 pt-6">Optional</p>
            <div className="glass-tile mt-2 rounded-2xl px-3 py-2.5">
              <ApiKeyInput
                label="Zernio (optional)"
                value={keys.drafts.zernioApiKey}
                configured={zernioConfigured}
                onChange={(v) => keys.setDraft('zernioApiKey', v)}
                onRemove={() => void keys.remove('zernioApiKey')}
                onBlur={() => void keys.persist()}
                placeholder="sk_…"
                description="Connects your social accounts so you can post and schedule clips. Manage them under Accounts."
                getKeyUrl={PROVIDER_LINKS.zernio}
              />
            </div>
          </Section>

          <Section id="assistant">
            <AssistantSettings />
          </Section>

          <JevSettings />

          <Section id="vocabulary">
            <PanelHeader
              icon={<IconTile><BookA /></IconTile>}
              title="Custom vocabulary"
              description="Names, products and jargon that transcription should spell exactly, one per line. Used for captions and generated post metadata."
              action={vocabularyTerms > 0 && <Badge className="font-mono tabular">{vocabularyTerms} term{vocabularyTerms === 1 ? '' : 's'}</Badge>}
            />
            <VocabularyField value={customVocabulary} onCommit={(value) => commit({ customVocabulary: value })} />
          </Section>

          <Section id="picking">
            <PanelHeader
              icon={<IconTile><Gavel /></IconTile>}
              title="Clip picking"
              description="How automatic clips are chosen. Jev review, clip requests and longform edits choose their own way."
            />
            <div className="mt-4 space-y-2">
              <SettingRow
                title="Second opinion"
                description="Your AI offers up to three times as many moments as you asked for, then separate requests score each one without seeing its first scores, and the best are kept. Costs about one extra AI request per eight candidates."
                control={
                  <Segmented
                    label="Second opinion"
                    size="sm"
                    value={clipJudge === 'off' ? 'off' : 'on'}
                    options={[{ value: 'on', label: 'On' }, { value: 'off', label: 'Off' }]}
                    onChange={(value) => void commit({ clipJudge: value })}
                  />
                }
              />
            </div>
          </Section>

          <Section id="framing">
            <PanelHeader
              icon={<IconTile><Crop /></IconTile>}
              title="Framing"
              description="How Smart framing lays out screen recordings in vertical clips. Full Frame and Classic framing do not use these."
            />
            <div className="mt-4 space-y-2">
              <SettingRow
                title="Webcam in your videos"
                description="Where your webcam sits in the videos you clip. Auto finds it. Choose a place if faces in your games get picked as the webcam: only a face there is used, and faces elsewhere are left as part of the game. Set it only when every video you clip uses that layout."
                control={
                  <Select
                    aria-label="Webcam in your videos"
                    size="sm"
                    className="w-[150px]"
                    value={webcamLocation}
                    options={WEBCAM_LOCATIONS.map((value) => ({ value, label: WEBCAM_LOCATION_LABELS[value] }))}
                    onChange={(value) => void commit({ webcamLocation: value as WebcamLocation })}
                  />
                }
              />
              <SettingRow
                title="Webcam position"
                description="Where the webcam panel goes when a screen and a webcam are stacked."
                control={
                  <Segmented
                    label="Webcam position"
                    size="sm"
                    value={webcamPosition}
                    options={[{ value: 'top', label: 'Top' }, { value: 'bottom', label: 'Bottom' }]}
                    onChange={(value) => void commit({ webcamPosition: value })}
                  />
                }
              />
              <SettingRow
                title="Screen with no webcam"
                description="Fill crops the screen to the whole frame: around the area that matters when your AI can read images, otherwise the centre. Whole shows all of it over a blurred copy."
                control={
                  <Segmented
                    label="Screen with no webcam"
                    size="sm"
                    value={screenFraming}
                    options={[{ value: 'fill', label: 'Fill' }, { value: 'fit', label: 'Whole' }]}
                    onChange={(value) => void commit({ screenFraming: value })}
                  />
                }
              />
            </div>
          </Section>

          <Section id="output">
            <PanelHeader
              icon={<IconTile><FolderOpen /></IconTile>}
              title="Output"
              description="Every run gets its own folder of clips, transcript and plan."
            />
            <OutputFolder value={outputDirectory} onCommit={(dir) => commit({ outputDirectory: dir })} />
            {!isMac && (
              <SettingRow
                className="mt-3"
                title="Video encoder"
                description="Clips encode on the CPU by default. NVIDIA uses the graphics card's encoder (NVENC) and is several times quicker; without a usable NVIDIA card, clips encode on the CPU."
                control={
                  <Segmented
                    label="Video encoder"
                    size="sm"
                    value={videoEncoder}
                    options={[{ value: 'cpu', label: 'CPU' }, { value: 'nvidia', label: 'NVIDIA (faster)' }]}
                    onChange={(value) => void commit({ videoEncoder: value })}
                  />
                }
              />
            )}
          </Section>

          <Section id="system">
            <PanelHeader
              icon={<IconTile tone={toolsChecked && toolsMissing ? 'danger' : 'neutral'}><Cpu /></IconTile>}
              title="System check"
              description="Tools BridgeClip needs to download, transcribe and cut video."
              action={
                <Button
                  size="sm"
                  onClick={checkTools}
                  disabled={checkingTools}
                  icon={<RefreshCw className={cn('h-3.5 w-3.5', checkingTools && 'animate-spin')} />}
                >
                  Re-check
                </Button>
              }
            />
            <ToolList rows={tools} checking={checkingTools} />
            {toolError && <Callout tone="danger" className="mt-3">{toolError}</Callout>}
            {!isPackaged && (
              <div className="mt-4 grid gap-4 border-t border-white/[0.06] pt-4">
                <DevPathField
                  label="Python path"
                  value={pythonPath}
                  placeholder={toolStatus?.pythonPath || (isPackaged ? "python" : "python3")}
                  onCommit={(v) => commit({ pythonPath: v }, true)}
                />
              </div>
            )}
          </Section>

          <Section id="about" className="overflow-hidden">
            <div className="flex flex-wrap items-center gap-4">
              <BridgeClipLogo variant="icon" className="-m-1 h-12" />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2.5">
                  <h2 className="text-base font-semibold tracking-[-0.01em] text-ink">{APP_NAME}</h2>
                  <Badge className="font-mono tabular">v{APP_VERSION}</Badge>
                </div>
                <p className="mt-1 text-xs leading-relaxed text-ink-muted">
                  Released under the {LICENSE_NAME} license. Built by{' '}
                  <button
                    onClick={() => getApi().shell.openPath(BRIDGEMIND_URL)}
                    className="text-ink underline decoration-white/25 underline-offset-2 transition-colors hover:decoration-ink"
                  >
                    BridgeMind
                  </button>
                  .
                </p>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <Button size="sm" icon={<Github className="h-3.5 w-3.5" />} onClick={() => getApi().shell.openPath(REPO_URL)}>
                  GitHub
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  icon={<History className="h-3.5 w-3.5" />}
                  onClick={() => useChangelogStore.getState().setOpen(true)}
                >
                  Changelog
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  trailingIcon={<ArrowUpRight className="h-3.5 w-3.5" />}
                  onClick={() => getApi().shell.openPath(ISSUES_URL)}
                >
                  Report an issue
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  icon={<ScrollText className="h-3.5 w-3.5" />}
                  onClick={() => getApi().diagnostics.openLogFolder()}
                >
                  Show logs
                </Button>
              </div>
            </div>
            <UpdatesRow />
            <OutputStorage outputDirectory={outputDirectory} />
          </Section>
        </div>
      </div>
    </Page>
  )
}

function Section({ id, className, children }: { id: SectionId; className?: string; children: ReactNode }): React.JSX.Element {
  return (
    <section id={`settings-${id}`} className="scroll-mt-14">
      <Panel className={className}>{children}</Panel>
    </section>
  )
}

/** How far below the top of the scroll area a section counts as reached. */
const ACTIVE_OFFSET = 72

/** The section nearest the top of the scroll area, for the side nav, and a jump that selects it. */
function useActiveSection(ids: SectionId[]): [SectionId, (id: SectionId) => void] {
  const [active, setActive] = useState<SectionId>(ids[0])
  // A jumped-to section stays selected through its smooth scroll, even if it
  // can't reach the top, until the user scrolls on their own.
  const pinned = useRef(false)
  const key = ids.join(',')
  useEffect(() => {
    const root = document.getElementById('page-scroll')
    if (!root) return
    const update = (): void => {
      if (pinned.current) return
      // Sections near the end can't scroll up to the offset line, so over the
      // last stretch the line slides down to the bottom edge. Every section
      // takes its turn, and the last is selected at the bottom.
      const max = root.scrollHeight - root.clientHeight
      const tail = Math.min(max, root.clientHeight - ACTIVE_OFFSET)
      const progress = tail > 0 ? Math.min(1, Math.max(0, root.scrollTop - (max - tail)) / tail) : 0
      const line = root.getBoundingClientRect().top + ACTIVE_OFFSET + progress * (root.clientHeight - ACTIVE_OFFSET)
      let current = ids[0]
      for (const id of ids) {
        const el = document.getElementById(`settings-${id}`)
        if (el && el.getBoundingClientRect().top <= line) current = id
      }
      setActive(current)
    }
    const release = (): void => { pinned.current = false }
    const inputs = ['wheel', 'touchstart', 'pointerdown', 'keydown'] as const
    update()
    root.addEventListener('scroll', update, { passive: true })
    for (const type of inputs) window.addEventListener(type, release, { passive: true })
    return () => {
      root.removeEventListener('scroll', update)
      for (const type of inputs) window.removeEventListener(type, release)
    }
  }, [key])
  const jump = (id: SectionId): void => {
    pinned.current = true
    setActive(id)
    document.getElementById(`settings-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
  return [active, jump]
}

function SaveIndicator({ saving, savedAt, error }: { saving: boolean; savedAt: number | null; error: string | null }): React.JSX.Element | null {
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    if (!savedAt) return
    setVisible(true)
    const t = setTimeout(() => setVisible(false), 2200)
    return () => clearTimeout(t)
  }, [savedAt])

  if (error) return <Badge tone="danger">{error}</Badge>
  if (saving) {
    return (
      <Badge icon={<Loader2 className="h-3 w-3 animate-spin" />}>
        Saving…
      </Badge>
    )
  }
  return (
    <span
      className={cn('transition-opacity duration-300', visible ? 'opacity-100' : 'opacity-0')}
      aria-live="polite"
    >
      <Badge tone="success" icon={<Check className="h-3 w-3" strokeWidth={3} />}>
        Saved
      </Badge>
    </span>
  )
}

function OutputFolder({ value, onCommit }: { value: string; onCommit: (dir: string) => void }): React.JSX.Element {
  const [error, setError] = useState<string | null>(null)

  const choose = async (): Promise<void> => {
    try {
      const dir = await getApi().settings.selectOutputDir()
      if (dir) {
        setError(null)
        onCommit(dir)
      }
    } catch (err) {
      setError(errorMessage(err, 'Could not choose an output folder'))
    }
  }

  const open = async (): Promise<void> => {
    try {
      if (!(await getApi().shell.openPath(value))) setError('Could not open the output folder')
      else setError(null)
    } catch (err) {
      setError(errorMessage(err, 'Could not open the output folder'))
    }
  }

  return (
    <div className="mt-4">
      <div className="flex items-center gap-2">
        <TextInput
          className="flex-1"
          mono
          value={value}
          readOnly
          title="Use Change to select a folder"
          leading={<FolderOpen className="h-3.5 w-3.5" />}
          aria-label="Output folder"
        />
        <Button onClick={() => void choose()}>Change…</Button>
        <Button variant="ghost" onClick={() => void open()} disabled={!value}>
          Open
        </Button>
      </div>
      {error && <p role="alert" className="mt-2 text-xs text-danger">{error}</p>}
    </div>
  )
}

function VocabularyField({ value, onCommit }: { value: string; onCommit: (value: string) => void }): React.JSX.Element {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  return (
    <div className="mt-4">
      <TextArea
        rows={4}
        value={draft}
        placeholder={'GPT 6 Sol\nOpus 5.5\nBridgeMind'}
        aria-label="Custom vocabulary"
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => draft !== value && onCommit(draft)}
      />
      <p className="mt-2 px-1 text-xs text-ink-subtle">Up to five words per term. Applies to new transcriptions; generated metadata uses it right away.</p>
    </div>
  )
}

function DevPathField({
  label,
  value,
  placeholder,
  onCommit
}: {
  label: string
  value: string
  placeholder: string
  onCommit: (value: string) => void
}): React.JSX.Element {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  return (
    <Field label={label} hint="Developer builds only">
      <TextInput
        mono
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => draft.trim() !== value && onCommit(draft.trim())}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
      />
    </Field>
  )
}

interface ToolRow { name: string; ok: boolean | null; detail?: ReactNode; hint?: string | null; repairCommand?: string | null; failureLabel?: string; optional?: boolean }

function toolRows(status: ToolStatus | null): ToolRow[] {
  return [
    { name: 'Python', ok: status?.python ?? null, detail: status?.pythonPath },
    {
      name: 'Clipping dependencies and smart framing',
      ok: status?.pythonDeps ?? null,
      detail: status && !status.pythonDeps ? status.pythonError : 'Includes OpenCV and the smart framing model',
      hint: status && !status.pythonDeps ? status.pythonHint : null,
      repairCommand: status && !status.pythonDeps ? status.pythonRepairCommand : null,
      failureLabel: 'Needs attention'
    },
    { name: 'FFmpeg', ok: status?.ffmpeg ?? null },
    {
      name: 'FFmpeg captions',
      ok: status?.ffmpegCaptions ?? null,
      detail: 'Required when captions are enabled; FFmpeg must include the ass filter',
      optional: true
    },
    { name: 'FFprobe', ok: status?.ffprobe ?? null },
    { name: 'yt-dlp', ok: status?.ytdlp ?? null, detail: 'Downloads YouTube videos and Twitch VODs' },
    { name: 'BridgeClip clipping engine', ok: status?.engine ?? null, detail: status?.enginePath },
    { name: 'Bridge runner', ok: status?.bridgeRunner ?? null, detail: status?.bridgePath }
  ]
}

function ToolList({ rows, checking }: { rows: ToolRow[]; checking: boolean }): React.JSX.Element {
  const [expanded, setExpanded] = useState(false)
  const required = rows.filter((row) => !row.optional)
  const missing = required.filter((row) => row.ok === false).length
  const checked = rows.every((row) => row.ok != null)

  return (
    <div className="mt-4">
      {checked && (
        <div className="mb-3 flex items-center gap-2 px-1 text-xs">
          <StatusDot tone={missing > 0 ? 'danger' : 'success'} />
          <span className={cn('flex-1', missing > 0 ? 'text-danger' : 'text-ink-muted')}>
            {missing > 0 ? `${missing} required check${missing === 1 ? '' : 's'} need${missing === 1 ? 's' : ''} attention` : `Everything BridgeClip needs is installed (${rows.filter((row) => row.ok).length} tools)`}
          </span>
          {missing === 0 && (
            <Button
              size="sm"
              variant="ghost"
              aria-expanded={expanded}
              trailingIcon={<ChevronDown className={cn('h-3.5 w-3.5 transition-transform', expanded && 'rotate-180')} />}
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? 'Hide details' : 'Show details'}
            </Button>
          )}
        </div>
      )}
      {(!checked || missing > 0 || expanded) && <div className="glass-well divide-y divide-white/[0.05] overflow-hidden rounded-2xl animate-fade-in">
        {rows.map((row) => {
          const tone = row.ok == null ? 'idle' : row.ok ? 'success' : row.optional ? 'warning' : 'danger'
          return (
            <div key={row.name} className="flex items-center gap-3 px-3 py-2">
              <StatusDot tone={tone} />
              <div className="min-w-0 flex-1">
                <p className="text-sm text-ink">{row.name}</p>
                {row.detail && (
                  <p className="mt-0.5 break-words font-mono text-2xs text-ink-subtle" data-selectable>
                    {row.detail}
                  </p>
                )}
                {row.hint && <p className="mt-2 text-xs leading-relaxed text-ink-muted">{row.hint}</p>}
                {row.repairCommand && (
                  <pre className="mt-2 whitespace-pre-wrap break-all rounded-lg bg-black/20 p-2 font-mono text-2xs text-ink" data-selectable>
                    <code>{row.repairCommand}</code>
                  </pre>
                )}
              </div>
              {row.ok == null ? (
                <span className="shrink-0 text-xs text-ink-faint">{checking ? 'Checking…' : 'Not checked'}</span>
              ) : (
                <Badge tone={row.ok ? 'neutral' : row.optional ? 'warning' : 'danger'} className="shrink-0">
                  {row.ok ? 'Found' : row.optional ? 'Unavailable' : row.failureLabel ?? 'Missing'}
                </Badge>
              )}
            </div>
          )
        })}
      </div>}
    </div>
  )
}
