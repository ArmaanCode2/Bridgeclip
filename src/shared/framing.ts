/**
 * Smart framing preferences. Shared so main (validation, the engine's environment) and the
 * Settings page offer the same choices.
 */

/** Where the webcam panel goes when a screen share and a webcam are stacked. */
export const WEBCAM_POSITIONS = ['top', 'bottom'] as const
export type WebcamPosition = (typeof WEBCAM_POSITIONS)[number]
export const DEFAULT_WEBCAM_POSITION: WebcamPosition = 'top'

/**
 * Where the webcam sits in the person's own videos. Auto finds it; a chosen place means only a face
 * there can be the webcam, so faces in the game elsewhere are never taken for it.
 */
export const WEBCAM_LOCATIONS = ['auto', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'bottom-center'] as const
export type WebcamLocation = (typeof WEBCAM_LOCATIONS)[number]
export const DEFAULT_WEBCAM_LOCATION: WebcamLocation = 'auto'
export const WEBCAM_LOCATION_LABELS: Record<WebcamLocation, string> = {
  auto: 'Auto', 'top-left': 'Top left', 'top-right': 'Top right',
  'bottom-left': 'Bottom left', 'bottom-right': 'Bottom right', 'bottom-center': 'Bottom centre'
}

/** A screen with no webcam: crop it to fill the frame, or show it whole over a blur. */
export const SCREEN_FRAMINGS = ['fill', 'fit'] as const
export type ScreenFraming = (typeof SCREEN_FRAMINGS)[number]
export const DEFAULT_SCREEN_FRAMING: ScreenFraming = 'fill'
