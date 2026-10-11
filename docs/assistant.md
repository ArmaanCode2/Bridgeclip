# Assistant

BridgeClip's **Chat** page lets you run the app in plain words: "clip the best moments from this link", "which clips haven't been posted?", "add the top three to my daily automation". It runs on your own Claude or ChatGPT subscription, or on any OpenRouter model with your OpenRouter API key. BridgeClip has no AI account of its own and never sees your Claude or ChatGPT sign-in.

## Connect a subscription

Open **Settings → Assistant**.

| Assistant | Subscription | Install | Sign in |
| --- | --- | --- | --- |
| Claude Code | Claude Pro or Max | `curl -fsSL https://claude.ai/install.sh \| bash` | **Sign in** in Settings, or run `claude` in Terminal |
| Codex | ChatGPT Plus, Pro or Business | `curl -fsSL https://chatgpt.com/codex/install.sh \| sh` | **Sign in** in Settings, or run `codex login` |

**Sign in** runs the CLI's own sign-in (`claude auth login`, `codex login`). It opens your browser, and the CLI stores the credentials where it always does: the system keychain for Claude Code, `~/.codex` for Codex. If Claude's page shows a code, paste it into the field that appears.

BridgeClip finds the CLIs on your login shell's `PATH` and in the usual install folders (`~/.local/bin`, Homebrew, npm, Volta, nvm, fnm, mise, asdf). If one isn't found, install it, then choose **Check again**.

### Or use OpenRouter

Chat can also run on an OpenRouter model, with no CLI to install. It uses the OpenRouter key from **Settings → API keys** (the same key clipping uses), and OpenRouter bills each message to it at that model's price. Choose a model in Chat's model menu (the **OpenRouter** tab) or in **Settings → Assistant**.

## Choose a model

The model chip in Chat's message box shows who will answer, with Claude's, OpenAI's or OpenRouter's logo. It opens a tab per provider: **Claude** (Default, Sonnet, Opus, Fable, Haiku, through Claude Code), **OpenAI** (Default and the GPT models, through Codex) and **OpenRouter** (a searchable list of every OpenRouter model that can call tools, with its price per million tokens in and out, and three suggestions first). **Default** leaves the CLI's own default. Providers that aren't connected are greyed out with a **Connect** button (**Add key** for OpenRouter). Settings → Assistant has the same choice.

You can switch models, or even providers, partway through a chat. Each reply shows the logo and model that wrote it. When you switch to a CLI, it can't resume the other's session, so BridgeClip passes it the recent conversation as context. OpenRouter keeps no session at all: every message sends the chat so far (up to about 60,000 characters, newest first).

## What the assistant can do

It works only through BridgeClip's tools, the same functions the app's pages use:

- **Find videos:** list a channel's newest uploads and live streams (by @handle, link or name, e.g. "BridgeMind") or search YouTube, so "clip the latest BridgeMind video" works without pasting a link. This reads YouTube's listing pages with the bundled yt-dlp (metadata only, nothing downloaded). Streams that are live, upcoming or still processing are marked and can't be clipped yet.
- **The web:** search the web and read public pages for current information. Claude Code uses its own WebSearch and WebFetch tools and Codex its live web search, both on your subscription. OpenRouter models get BridgeClip's `web_search` (OpenRouter's web search, about a cent or two per search on your key) and `read_web_page`.
- **Clip:** start Automatic or Review & edit jobs from a link, or a file you choose in a picker; follow progress; cancel.
- **Library:** browse runs and clips, read transcripts, bookmark, mark clips posted, delete runs or clips, reveal files, open a run in the app.
- **Review & edit:** list candidates, rename them, mark them ready or discarded, change captions, export one or all ready clips.
- **Automations:** create, configure (accounts, daily times, time zone, on/off, captions), add Library clips, edit, reorder, remove, post the next clip now.
- **Posting:** list connected accounts, post or schedule a Library clip, list posts, reschedule, cancel or retry.
- **Settings:** read them, change custom vocabulary and the Jev / web research switches.

It cannot read or change API keys, the Library folder, app updates or social account connections. TikTok posts and TikTok automation reviews stay in the app because TikTok requires you to confirm its settings yourself.

## Approvals

Every action that **publishes**, **deletes** or **spends OpenRouter credit** stops and shows an approval card in the chat, describing exactly what will happen: the video, the accounts, the caption, the time. Nothing happens until you choose **Allow**. **Don't allow**, stopping the reply or 15 minutes without an answer all decline. Chat's sidebar item shows a badge while a card is waiting.

## Privacy and isolation

- Your messages, and the tool results the assistant reads (video titles, transcripts, clip and account names), go to Anthropic or OpenAI under your subscription, the same as any Claude Code or Codex session. On an OpenRouter model they go to OpenRouter and the model's provider under your OpenRouter account.
- On OpenRouter, BridgeClip itself runs the conversation: it calls OpenRouter's chat completions API from the main process with your key (never from the page), offers the model only BridgeClip's tools, runs each tool call through the same approval cards, and asks OpenRouter to route only to providers that support tool calling.
- Each reply starts the CLI in an empty folder in BridgeClip's app data, with an environment built from an allowlist. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and other credentials in your environment are not passed on, so usage counts against your subscription, not an API key.
- Claude Code runs with only its web tools on (`--tools WebSearch,WebFetch`: no shell, file or code tools), only BridgeClip's MCP server (`--strict-mcp-config`), user settings only, hooks disabled and permission prompts auto-denied. Codex runs read-only with approvals off, live web search on, and ignores your `config.toml`.
- `read_web_page` only reaches public addresses: local, private and link-local addresses are refused before the request and again when each connection opens (so DNS can't redirect it), redirects are checked hop by hop, and only text pages are read (2 MB, 20,000 characters). Search results and pages are treated as data; a page can't make the assistant post or delete anything without your approval.
- BridgeClip's tools are served over a local MCP endpoint on `127.0.0.1` with a new random token for every reply, passed to the CLI through its environment (never its command line). Requests from other hosts, browsers (any `Origin`) or without the token are refused.
- Conversations are saved in `assistant/conversations/` inside BridgeClip's app data (the 50 most recent). The CLI also keeps its own session history, as for any session you run.

## Troubleshooting

| Message | Fix |
| --- | --- |
| "Claude Code isn't signed in" / "Codex isn't signed in" | Sign in from Settings → Assistant, or run `claude` / `codex login` in Terminal. |
| "…reached its usage limit" | Your plan's limit was hit; try again after it resets, or switch assistants. |
| "This version of … is too old" | Run `claude update` or `codex update`. |
| "…didn't respond" | The CLI produced nothing for 3 minutes. Check it's signed in and online. |
| A tool says the OpenRouter or Zernio key is missing | Add the key in Settings → API keys. |
| "OpenRouter rejected your API key" / "out of credits" | Check the key in Settings → API keys, or add credits on openrouter.ai. |
| "… can't use tools through OpenRouter right now" | No provider for that model currently supports tool calling. Choose another model. |
