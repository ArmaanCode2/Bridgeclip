import { useCallback, useEffect, useId, useState, type ReactNode } from 'react'
import { useSettingsStore } from '../store/use-settings-store'
import { getApi } from '../lib/ipc'
import {
  CLAUDE_PLAN_EFFORTS, CLAUDE_PLAN_EFFORT_LABELS, CLAUDE_PLAN_MODELS, CLAUDE_PLAN_MODEL_HINTS, CLAUDE_PLAN_MODEL_LABELS, claudePlanBlock,
  type ClaudePlanEffort, type ClaudePlanModel
} from '../../shared/providers'
import type { AssistantProviderStatus } from '../../shared/assistant'
import type { CommitSettings } from './ProviderSetup'
import { Badge } from './ui/Badge'
import { Button } from './ui/Button'
import { Callout } from './ui/Callout'
import { Select } from './ui/Select'
import { SettingRow } from './ui/SettingRow'
import { Switch } from './ui/Switch'

/** How Claude Code is signed in on this computer: null until it has answered. */
function useClaudeStatus(): { status: AssistantProviderStatus | null; checking: boolean; failed: boolean; check: (fresh: boolean) => Promise<void> } {
  const [status, setStatus] = useState<AssistantProviderStatus | null>(null)
  const [checking, setChecking] = useState(false)
  const [failed, setFailed] = useState(false)
  const check = useCallback(async (fresh: boolean): Promise<void> => {
    setChecking(true)
    try {
      const statuses = await getApi().assistant.status(fresh)
      setStatus(statuses.find((item) => item.id === 'claude') ?? null)
      setFailed(false)
    } catch {
      setFailed(true)
    } finally {
      setChecking(false)
    }
  }, [])
  useEffect(() => { void check(false) }, [check])
  return { status, checking, failed, check }
}

/**
 * The model, effort and image reading of the Claude plan, shown when its card is chosen. Claude Code
 * signs in on its own, so there is no key here: only whether a job could use the plan right now.
 */
export function ClaudePlanFields({ commit, test, children }: {
  commit: CommitSettings
  /** Test connection for the plan. */
  test: ReactNode
  /** What the provider choice changes elsewhere. */
  children?: ReactNode
}): React.JSX.Element {
  const model = useSettingsStore((s) => s.claudePlanModel)
  const effort = useSettingsStore((s) => s.claudePlanEffort)
  const vision = useSettingsStore((s) => s.claudePlanVision)
  const { status, checking, failed, check } = useClaudeStatus()
  const [error, setError] = useState<string | null>(null)
  const modelId = useId()
  const effortId = useId()
  const problem = status && claudePlanBlock(status)

  return (
    <div className="space-y-3">
      {problem ? (
        <Callout tone="warning" stacked action={<Button size="sm" loading={checking} onClick={() => void check(true)}>Check again</Button>}>
          {problem}
        </Callout>
      ) : (
        <div className="flex flex-wrap items-center gap-2 px-1 text-xs text-ink-muted">
          {status
            ? <Badge tone="success">{status.plan ? `Signed in · ${status.plan.charAt(0).toUpperCase()}${status.plan.slice(1)}` : 'Signed in'}</Badge>
            : <span role="status">{failed ? 'Could not check Claude Code.' : 'Checking Claude Code…'}</span>}
          <span className="min-w-0 flex-1">Clips are picked by Claude Code on this computer, with your plan.</span>
          <Button size="sm" variant="ghost" loading={checking} onClick={() => void check(true)}>Check again</Button>
        </div>
      )}
      <SettingRow
        title={<label htmlFor={modelId}>Model</label>}
        description={CLAUDE_PLAN_MODEL_HINTS[model]}
        control={
          <Select id={modelId} size="sm" className="w-32" value={model}
            options={CLAUDE_PLAN_MODELS.map((id) => ({ value: id, label: CLAUDE_PLAN_MODEL_LABELS[id] }))}
            onChange={(value) => void commit({ claudePlanModel: value as ClaudePlanModel }).then(setError)} />
        }
      />
      <SettingRow
        title={<label htmlFor={effortId}>Effort</label>}
        description="Higher effort thinks longer and uses more of your plan. Medium suits most videos."
        control={
          <Select id={effortId} size="sm" className="w-32" value={effort}
            options={CLAUDE_PLAN_EFFORTS.map((id) => ({ value: id, label: CLAUDE_PLAN_EFFORT_LABELS[id] }))}
            onChange={(value) => void commit({ claudePlanEffort: value as ClaudePlanEffort }).then(setError)} />
        }
      />
      <SettingRow
        title="Read images"
        description="Lets smart framing and the review look at video frames, which uses more of your plan. With this off, framing uses built-in face tracking, and a video with no speech can’t be planned."
        control={<Switch label="Read images with my Claude plan" checked={vision === 'on'} onChange={(on) => void commit({ claudePlanVision: on ? 'on' : 'off' }).then(setError)} />}
      />
      {error && <p role="alert" className="text-xs text-danger">{error}</p>}
      {test}
      {children}
    </div>
  )
}
