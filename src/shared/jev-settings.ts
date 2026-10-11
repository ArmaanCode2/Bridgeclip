/** BridgeClip approval policy defaults, independent of provider confidence. */
export const JEV_DEFAULTS = {
  jevThreshold: '0.75',
  jevSelfContainedThreshold: '0.70',
  jevFaithfulToSourceThreshold: '0.65',
  jevTitleSupportedThreshold: '0.70',
  jevSponsorThreshold: '0.80',
  jevEvidenceThreshold: '0.50',
  jevCutThreshold: '0.95',
} as const

/**
 * Jev review, its visual context and source web research are opt-in beta
 * features that spend extra OpenRouter credit. They are off unless the user
 * turns them on. Review & edit always runs Jev regardless of `jevEnabled`.
 */
export const JEV_FEATURE_DEFAULTS = {
  jevEnabled: 'off',
  jevVisualContext: 'off',
  sourceContextWebResearch: 'off',
} as const

export type JevThresholdKey = keyof typeof JEV_DEFAULTS
export type JevThresholdSettings = Record<JevThresholdKey, string>
export const JEV_DOCS_URL = 'https://docs.typesafe.ai/introduction'
export const JEV_CONFIDENCE_URL = 'https://docs.typesafe.ai/confidence'
