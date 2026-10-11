import { shell } from 'electron'
import type { AssistantCliProviderId, AssistantSignInState } from '../../shared/assistant'
import { cliEnvironment, resolveCli, spawnCli, type CliRun } from './cli-process'

// Sign-in runs the provider's own command, which opens the browser and stores
// the credentials where that CLI always keeps them. BridgeClip only relays the
// sign-in link and, for Claude, the code the browser may show.

const SIGN_IN_TIMEOUT_MS = 10 * 60 * 1000
const OUTPUT_LIMIT = 32 * 1024

const SIGN_IN_HOSTS: Record<AssistantCliProviderId, readonly string[]> = {
  claude: ['claude.com', 'claude.ai', 'platform.claude.com', 'console.anthropic.com'],
  codex: ['auth.openai.com', 'chatgpt.com']
}

/** The first https link on an allowed sign-in host, so a stray URL in CLI output is never opened. */
export function signInUrl(provider: AssistantCliProviderId, output: string): string | null {
  for (const match of output.matchAll(/https:\/\/[^\s"'<>]+/g)) {
    try {
      const url = new URL(match[0])
      if (url.protocol === 'https:' && !url.username && !url.password && SIGN_IN_HOSTS[provider].includes(url.hostname)) return url.toString()
    } catch { /* not a URL */ }
  }
  return null
}

interface ActiveSignIn {
  run: CliRun
  state: AssistantSignInState
  timer: NodeJS.Timeout
}

export class SignInManager {
  private readonly active = new Map<AssistantCliProviderId, ActiveSignIn>()
  private readonly states = new Map<AssistantCliProviderId, AssistantSignInState>()

  constructor(private readonly emit: (state: AssistantSignInState) => void, private readonly onSignedIn: (provider: AssistantCliProviderId) => void) {}

  state(provider: AssistantCliProviderId): AssistantSignInState {
    return this.states.get(provider) ?? { provider, status: 'idle', url: null, acceptsCode: false, message: null }
  }

  private update(provider: AssistantCliProviderId, patch: Partial<AssistantSignInState>): void {
    const next = { ...this.state(provider), ...patch, provider }
    this.states.set(provider, next)
    const active = this.active.get(provider)
    if (active) active.state = next
    this.emit(next)
  }

  async start(provider: AssistantCliProviderId): Promise<AssistantSignInState> {
    if (this.active.has(provider)) return this.state(provider)
    const cli = await resolveCli(provider)
    if (!cli) {
      this.update(provider, { status: 'failed', url: null, acceptsCode: false, message: `${provider === 'claude' ? 'Claude Code' : 'Codex'} isn’t installed.` })
      return this.state(provider)
    }
    const args = provider === 'claude' ? ['auth', 'login', '--claudeai'] : ['login']
    const run = spawnCli(cli, args, { cwd: process.env.HOME || process.cwd(), env: cliEnvironment(cli.pathValue) })
    let output = ''
    const timer = setTimeout(() => {
      this.update(provider, { status: 'failed', message: 'Sign-in timed out. Try again.' })
      run.stop()
    }, SIGN_IN_TIMEOUT_MS)
    this.active.set(provider, { run, state: this.state(provider), timer })
    this.update(provider, { status: 'waiting', url: null, acceptsCode: false, message: 'Finish signing in in your browser.' })
    const onData = (chunk: Buffer): void => {
      output = (output + chunk.toString('utf8')).slice(-OUTPUT_LIMIT)
      const url = signInUrl(provider, output)
      const acceptsCode = provider === 'claude' && /paste code/i.test(output)
      const current = this.state(provider)
      if (current.status === 'waiting' && (url !== current.url || acceptsCode !== current.acceptsCode)) this.update(provider, { url, acceptsCode })
    }
    run.child.stdout?.on('data', onData)
    run.child.stderr?.on('data', onData)
    run.child.stdin?.on('error', () => {})
    void run.exited.then(({ code }) => {
      clearTimeout(timer)
      this.active.delete(provider)
      if (this.state(provider).status !== 'waiting') return
      if (code === 0) {
        this.update(provider, { status: 'succeeded', acceptsCode: false, message: 'Signed in.' })
        this.onSignedIn(provider)
      } else {
        const tail = output.replace(/https:\/\/\S+/g, '').trim().split('\n').filter(Boolean).slice(-2).join(' ').slice(0, 300)
        this.update(provider, { status: 'failed', acceptsCode: false, message: tail ? `Sign-in didn’t finish: ${tail}` : 'Sign-in didn’t finish. Try again.' })
      }
    })
    return this.state(provider)
  }

  submitCode(provider: AssistantCliProviderId, code: unknown): void {
    const active = this.active.get(provider)
    if (!active || provider !== 'claude') throw new Error('No sign-in is waiting for a code.')
    if (typeof code !== 'string' || !/^[A-Za-z0-9#_\-.~]{8,1024}$/.test(code.trim())) throw new Error('Paste the code exactly as the browser shows it.')
    active.run.child.stdin?.write(`${code.trim()}\n`)
    this.update(provider, { message: 'Checking the code…' })
  }

  cancel(provider: AssistantCliProviderId): void {
    const active = this.active.get(provider)
    if (!active) return
    this.update(provider, { status: 'idle', url: null, acceptsCode: false, message: null })
    clearTimeout(active.timer)
    active.run.stop()
  }

  async openUrl(provider: AssistantCliProviderId): Promise<boolean> {
    const url = this.state(provider).url
    if (!url || signInUrl(provider, url) !== url) return false
    await shell.openExternal(url)
    return true
  }

  stopAll(): void {
    for (const provider of [...this.active.keys()]) this.cancel(provider)
  }
}
