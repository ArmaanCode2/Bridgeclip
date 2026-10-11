import { create } from 'zustand'
import type { AppDataScope } from '../../shared/assistant'

/**
 * Bumped when the main process changes data behind a page's back (the
 * assistant deleting clips, posting, editing an automation). Pages that cache
 * what they show reload when their scope's version changes.
 */
interface DataVersionState {
  versions: Record<AppDataScope, number>
  bump: (scope: AppDataScope) => void
}

export const useDataVersionStore = create<DataVersionState>((set) => ({
  versions: { library: 0, automations: 0, posts: 0, settings: 0, accounts: 0 },
  bump: (scope) => set((state) => ({ versions: { ...state.versions, [scope]: state.versions[scope] + 1 } }))
}))

export function useDataVersion(scope: AppDataScope): number {
  return useDataVersionStore((state) => state.versions[scope])
}
