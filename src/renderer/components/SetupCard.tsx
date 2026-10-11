import { useState } from 'react'
import { KeyRound, ShieldCheck, Wrench } from 'lucide-react'
import { useSettingsStore, useSetupState } from '../store/use-settings-store'
import { useApiKeyDrafts } from '../hooks/use-api-key-drafts'
import { cn, errorMessage } from '../lib/utils'
import { ProviderFields, ProviderPicker, TranscriptionSettings, type CommitSettings } from './ProviderSetup'
import { Button } from './ui/Button'
import { IconTile } from './ui/IconTile'

/**
 * First-run setup, inline on the Create page: choose an AI provider, add its key or endpoint,
 * and choose where speech becomes text, without being sent to Settings. Hidden once the
 * choices are complete and the tools check out.
 */
export function SetupCard({ onOpenSettings, className }: { onOpenSettings: () => void; className?: string }): React.JSX.Element | null {
  const { gaps, toolsOk } = useSetupState()
  const toolStatus = useSettingsStore((s) => s.toolStatus)
  const toolError = useSettingsStore((s) => s.toolError)
  const save = useSettingsStore((s) => s.save)
  const keys = useApiKeyDrafts()
  const [saveError, setSaveError] = useState<string | null>(null)
  const commit: CommitSettings = async (patch) => {
    try {
      await save(patch)
      setSaveError(null)
      return null
    } catch (err) {
      return errorMessage(err, 'Could not save this setting')
    }
  }
  const missingTools = toolStatus
    ? [
        !toolStatus.python && 'Python',
        !toolStatus.pythonDeps && 'Clipping dependencies or smart framing',
        !toolStatus.ffmpeg && 'FFmpeg',
        !toolStatus.ffprobe && 'FFprobe',
        !toolStatus.ytdlp && 'yt-dlp',
        !toolStatus.engine && 'BridgeClip clipping engine',
        !toolStatus.bridgeRunner && 'Bridge runner'
      ].filter(Boolean).join(', ')
    : ''

  if (gaps.length > 0) {
    return (
      <section
        className={cn(
          'glass relative overflow-hidden rounded-3xl bg-accent/[0.05] animate-fade-in',
          'shadow-[inset_0_1px_0_rgb(255_255_255/0.08),inset_0_0_0_1px_rgb(var(--accent)/0.22)]',
          className
        )}
        aria-label="Finish setup"
      >
        <div className="relative flex items-start gap-3 p-3.5 pb-0">
          <IconTile tone="accent">
            <KeyRound />
          </IconTile>
          <div className="min-w-0">
            <p className="eyebrow text-accent">One-time setup</p>
            <h2 className="mt-0.5 text-base font-semibold text-ink">Connect an AI provider</h2>
            <p className="mt-0.5 max-w-2xl text-xs text-ink-muted">
              BridgeClip runs on your computer. Choose who picks your clips and where speech becomes text.
            </p>
          </div>
        </div>

        <div className="relative mx-3 mt-3">
          <ProviderPicker commit={commit} compact />
        </div>

        <div className="relative m-3 grid gap-4 rounded-xl bg-black/15 p-3 shadow-[inset_0_0_0_1px_rgb(255_255_255/0.06)]">
          <ProviderFields commit={commit} keys={keys} compact />
          <div className="border-t border-white/[0.06] pt-3">
            <TranscriptionSettings commit={commit} keys={keys} compact />
          </div>
        </div>

        <p className="relative flex items-center gap-1.5 px-3.5 pb-3 text-2xs text-ink-subtle">
          <ShieldCheck className="h-3.5 w-3.5 text-success/80" />
          Keys are encrypted on this computer with your system keychain.
          <Button size="sm" variant="ghost" onClick={onOpenSettings} className="ml-auto">More options in Settings</Button>
        </p>
        {(keys.error || saveError) && <p role="alert" className="relative -mt-2 px-4 pb-4 text-xs text-danger">{keys.error ?? saveError}</p>}
      </section>
    )
  }

  if (toolsOk === false) {
    return (
      <section
        className={cn(
          'flex items-center gap-3 rounded-3xl bg-danger/[0.07] p-3 pr-4 animate-fade-in',
          'shadow-[inset_0_0_0_1px_rgb(var(--danger)/0.26),inset_0_1px_0_rgb(255_255_255/0.06)]',
          className
        )}
      >
        <IconTile tone="danger">
          <Wrench />
        </IconTile>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-ink">{toolError ? 'Could not check required tools' : 'Some required tools are missing'}</p>
          <p className="mt-0.5 text-xs text-ink-muted">{toolError ? 'Open Settings to retry the system check.' : `Missing: ${missingTools}. Open Settings for details.`}</p>
        </div>
        <Button onClick={onOpenSettings}>See what’s missing</Button>
      </section>
    )
  }

  return null
}
