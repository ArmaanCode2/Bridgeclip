/** H.264 encoder for exports on Windows and Linux. macOS always uses VideoToolbox. */
export const VIDEO_ENCODERS = ['cpu', 'nvidia'] as const
export type VideoEncoder = (typeof VIDEO_ENCODERS)[number]
/** x264 on the CPU: the best quality per byte. NVIDIA (NVENC) is the faster option. */
export const DEFAULT_VIDEO_ENCODER: VideoEncoder = 'cpu'
