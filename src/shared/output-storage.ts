export interface OutputStorageUsage {
  outputDirectory: string
  bytes: number
  fileCount: number
  exists: boolean
  unreadableCount: number
  /** Counting stopped at the scan limits; the total is a lower bound. */
  truncated?: boolean
}
