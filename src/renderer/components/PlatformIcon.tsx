import type { CSSProperties } from 'react'
import { AtSign, Facebook, Instagram, Linkedin, Music2, Share2, Youtube, type LucideIcon } from 'lucide-react'
import { cn } from '../lib/utils'
import { ZERNIO_PLATFORM_NAMES, type ZernioPlatform } from '../../shared/zernio'

interface PlatformInfo {
  name: string
  icon: LucideIcon | null
  /** Shown under the name; only for requirements that decide whether connecting works. */
  note?: string
}

export const PLATFORM_INFO: Record<ZernioPlatform, PlatformInfo> = {
  tiktok: { name: ZERNIO_PLATFORM_NAMES.tiktok, icon: Music2 },
  youtube: { name: ZERNIO_PLATFORM_NAMES.youtube, icon: Youtube, note: 'Clips under 3 minutes post as Shorts' },
  instagram: { name: ZERNIO_PLATFORM_NAMES.instagram, icon: Instagram, note: 'Needs a Business or Creator account' },
  facebook: { name: ZERNIO_PLATFORM_NAMES.facebook, icon: Facebook, note: 'Posts to a Page you manage' },
  // X has no Lucide glyph; XGlyph draws the mark, monochrome like the rest.
  twitter: { name: ZERNIO_PLATFORM_NAMES.twitter, icon: null, note: 'Zernio needs a card on file for X' },
  linkedin: { name: ZERNIO_PLATFORM_NAMES.linkedin, icon: Linkedin, note: 'Your profile or a company Page' },
  threads: { name: ZERNIO_PLATFORM_NAMES.threads, icon: AtSign }
}

/**
 * A faint glow of each platform's colour at the foot of its lens. The glyph
 * itself stays white: the tint says which platform, it doesn't shout it.
 */
const BRAND_TINT: Record<string, string> = {
  tiktok: '37 244 238',
  youtube: '255 40 70',
  instagram: '225 48 108',
  facebook: '24 119 242',
  twitter: '150 160 180',
  linkedin: '10 102 194',
  threads: '150 160 180'
}

export function platformName(platform: string): string {
  return (PLATFORM_INFO as Record<string, PlatformInfo>)[platform]?.name ?? platform.charAt(0).toUpperCase() + platform.slice(1)
}

function XGlyph({ className }: { className?: string }): React.JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden className={className}>
      <path d="M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z" />
    </svg>
  )
}

interface PlatformIconProps {
  platform: string
  className?: string
  /** `tile` (default): a glass lens. `glyph`: the bare mark, for inline chips. */
  variant?: 'tile' | 'glyph'
}

export function PlatformIcon({ platform, className, variant = 'tile' }: PlatformIconProps): React.JSX.Element {
  const info = (PLATFORM_INFO as Record<string, PlatformInfo | undefined>)[platform]
  const Icon = info ? info.icon : Share2
  const glyph = Icon ? <Icon className="h-4 w-4" strokeWidth={1.9} /> : <XGlyph className="h-[14px] w-[14px]" />

  if (variant === 'glyph') {
    return <span aria-hidden className={cn('inline-flex shrink-0 items-center justify-center text-ink [&_svg]:h-3 [&_svg]:w-3', className)}>{glyph}</span>
  }

  const tint = BRAND_TINT[platform]
  const style: CSSProperties = {
    backgroundColor: tint ? `rgb(${tint} / 0.22)` : 'rgb(255 255 255 / 0.08)'
  }
  return (
    <span
      aria-hidden
      style={style}
      className={cn(
        'flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-ink',
        'shadow-[inset_0_1px_0_rgb(255_255_255/0.2),inset_0_0_0_1px_rgb(255_255_255/0.1),0_6px_16px_-8px_rgb(0_0_0/0.6)]',
        className
      )}
    >
      {glyph}
    </span>
  )
}
