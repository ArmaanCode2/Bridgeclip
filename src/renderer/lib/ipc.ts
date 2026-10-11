import type { BridgeClipAPI } from '../../preload/index'

declare global {
  interface Window {
    bridgeclip: BridgeClipAPI
  }
}

export function getApi(): BridgeClipAPI {
  return window.bridgeclip
}
