# Changelog

What's new in each version of BridgeClip, newest first. You can also read this in the app under **Settings → About → Changelog** or **Help → Changelog**. Downloads for every version are on [GitHub Releases](https://github.com/bridge-mind/bridgeclip/releases).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Maintainers: see [Changelog](CONTRIBUTING.md#changelog) in the contributing guide before editing.

## [Unreleased]

### Added

- **What to clip** in Create → Clips: describe the moments you want, such as "every time they talk about pricing". Leave it blank to get the strongest moments as before.
- **Show title at the top** in Create → Captions turns off the title card on Automatic clips.
- **Review & edit** workflow: look over clip candidates before exporting. Trim, split and extend cuts, change layouts and camera changes, fix and place captions, then bake one clip or every ready clip with **Bake all**. You can also swap in a higher-quality copy of the source video.
- Jobs show progress for each stage on a timeline, a **Job Breakdown** of where the time went, and **Details** with reviews, the transcript and source research.
- The Library shows how many clips in each run are posted, lets you bookmark runs, and deletes whole runs or single clips.
- Settings → About shows how much disk space your output folder uses.
- YouTube links show a preview in Create.
- Automations have **Submitted** and **Needs attention** sections, drag to reorder, **View in Library**, reviewable title and caption drafts, **Refresh post status** and **Return to queue**.
- **TypeSafe Jev review** and **Research the source before clipping**, both in beta and off by default. Turn them on in Settings. Both use extra OpenRouter credit.
- **Free editor media** removes an editor project's source copy and preview once you're done with it.
- BridgeClip asks whether to save or discard unsaved edits before quitting, and offers **Reload project** when a project was changed elsewhere.
- This changelog, in Settings → About and the Help menu.
- **Chat**: run BridgeClip in plain words. It uses your Claude (Pro or Max, through Claude Code) or ChatGPT (through Codex) subscription, or any OpenRouter model with your OpenRouter key; connect them in **Settings → Assistant**. The assistant can start clipping jobs, manage the Library and automations, and post or schedule clips. Anything that publishes, deletes or spends OpenRouter credit on clipping asks you first.
- Chat's model menu has a tab each for Claude, OpenAI and OpenRouter (search every OpenRouter model that can use tools, with prices), and each reply shows the logo and model that wrote it. Recent chats are listed beside the conversation, with search.
- Chat can find YouTube videos by channel or search, so "clip the latest BridgeMind video" works without a link, and can search the web and read web pages.
- **Gemini and OpenAI-compatible providers** in Settings → AI & transcription: plan clips with your own Google Gemini key, or any OpenAI-compatible endpoint such as OpenAI, Ollama or LM Studio, instead of OpenRouter. **Test connection** checks the key and lists the models you can use. Review & edit, the AI review of automatic clips, the clip editor's Review, visual context, and Library and Automations titles and captions all run on your model too; only web research still needs OpenRouter.
- **Local Whisper transcription** in Settings → AI & transcription: transcribe on your own computer with the built-in Whisper (`faster-whisper` or `openai-whisper`) or your own Whisper server. Audio stays on your machine, there is no per-minute cost, and no OpenRouter key is needed for this step. Captions and cuts use the same word timing as the cloud models.
- **Local Whisper models** panel in Settings → AI & transcription: see which Whisper models are installed and how much disk space and memory each needs, get a warning when a model will not fit your video memory or RAM, delete models you no longer use, and choose **Auto**, **CPU** or **GPU** for transcription. **GPU** reports a clear error instead of quietly using the CPU.
- Install a Whisper model with one click, with a progress bar and **Cancel**, and turn on the GPU with **Enable GPU** on Windows with an NVIDIA card. A job waits for its model to be installed instead of stalling on a silent download. Choose the **Spoken language**, or let Whisper detect it from real speech rather than a music intro.
- Runs that use Gemini or a custom endpoint show **Not tracked** for the cost instead of a guessed price, and failures now say what went wrong (key rejected, model not found, rate limit reached, server unreachable).
- Gemini and custom endpoints are more dependable: servers that ignore JSON mode, models that think out loud first and models that refuse a temperature now work. A Gemini rate limit waits as long as Google asks. A URL missing `/v1` says so. Choose how long to wait for a slow local model, and see a warning when Ollama would cut off a long transcript. One rejected frame no longer turns off framing checks for the whole job, and a Gemini daily quota or a model too slow to answer stops them instead of stalling the job. Gemini 3 models keep their default temperature, as Google recommends. Choosing Gemini or a custom endpoint without an OpenRouter key switches transcription to Local Whisper.
- **Test connection** for Gemini and custom endpoints now tries your model with a request shaped like a clip job's and shows each check on its own line: server, key, model, JSON answers and images. A stopped local server is named by address and port. For a custom endpoint, a two-colour image that changes every run shows whether the model really reads images, and Settings offers to switch **This model can read images** to match.
- **Second opinion** in Settings → Clip picking, on by default: for Automatic clips the AI offers up to three times as many candidates, separate requests score each one without seeing its first scores, and the best are kept. Turn it off to save the extra AI requests.
- **Webcam in your videos** in Settings → Framing: tell BridgeClip where your webcam sits (a corner or the bottom centre) so faces elsewhere in your games are not taken for it. Auto stays the default.
- **Use in all clips** in Review & edit: copy a fixed Split crop, such as the webcam panel, to every clip not yet baked. Splits of two people are left alone.
- **Video encoder** in Settings → Output (Windows and Linux): **NVIDIA (faster)** encodes clips on an NVIDIA graphics card, several times quicker, and falls back to the CPU when the card's encoder is missing or fails.
- **Framing** in Settings: put the webcam above or below the screen in screen + webcam clips (above by default), and choose whether a screen with no webcam fills the whole frame (the default) or shows whole over a blurred copy. A filled screen is cropped around the area your AI marks as important when it can read images, otherwise around the centre. Screens below 720p, small screen areas and shots with people on camera still show whole.

### Changed

- Windows builds not signed by BridgeMind (one you package yourself) no longer update themselves from official releases, which would have replaced them.
- Smart framing finds webcams it used to miss: at the bottom centre, at the middle of a side, and small ones, which get a second look in sharper frames. The webcam's box comes from edges that stay still while the game moves, and a face roaming the game next to your webcam is no longer taken for it.
- Vertical clips keep the source's frame rate up to 60 fps instead of always 30.
- Sharper encoding: x264's `medium` preset at CRF 18 when FFmpeg has x264 and not OpenH264.
- Every clip's volume is levelled with one gain for the whole clip, measured first, so it no longer rides up and down; a limiter catches the peaks.
- The number of clips is now a limit, not a quota: the AI returns fewer clips rather than pad with filler, and clips scoring below 60/100 are left out (the best two are kept if none reach it). Requests in **What to clip** and reviewed runs are not filtered by score.
- Videos longer than 35 minutes are searched in 25-minute parts and every candidate is ranked together, so the middle of a long stream is no longer skimmed and local models with smaller context windows can plan long videos. 16:9 longform episodes and Review & edit with a time range still use one request.
- Gaming streams and livestreams are recognised as their own kind of video, so plays and reactions score as strong moments.
- Clips start at the beginning of a sentence more often, including in live commentary transcribed without punctuation.
- Captions start a new line after a pause, so a line no longer stays on screen through a silence with the first words of the next sentence in it.
- Built-in Whisper uses your custom vocabulary for the whole recording, not only its first 30 seconds.
- Smart framing ignores weak background faces, stays on a speaker who briefly looks away, follows talking heads inside 4:3 video, and analyzes footage faster.
- Screen + webcam clips fill the top panel again.
- Editor previews use much less disk space, and downloaded sources are moved into the project instead of copied.

### Fixed

- Gaming clips no longer take a game character's still portrait for your webcam when your real webcam is in another corner: the webcam is now the face that looks most like one (inside a found border, there through the whole clip, moving like a person), not the steadiest one. Following the webcam through a shot no longer jumps to another face when its face is missed for a moment.
- Smooth camera movement in the editor works in installed apps.
- **Bake all** keeps going when one clip fails, and says what went wrong.
- Titles with emoji at the length limit no longer break editor projects.
- Automations: submitted and held clips can be removed, posted clips no longer count toward the 500-clip limit, a post deleted in Zernio no longer leaves its clip stuck, slots that come due during an enhancement are kept, and one pending draft no longer blocks the rest of the queue.
- Unattended posts no longer use text from the source video's description, and refuse web addresses or @handles that aren't said in the video.
- Deleting a clip also removes its caption and YouTube text files.
- A missing transcript shows a clear message, and research citations can be selected and copied.
- Screen + webcam clips fill the bottom panel with the webcam, instead of showing a small webcam over a blurred copy of itself when the webcam in the source video is small.

## [0.1.19] - 2026-09-27

One release for macOS, Windows and Linux.

### Added

- Linux installers for x64: an AppImage and a DEB package.

### Changed

- YouTube downloads that fail for a temporary reason retry automatically with a fresh link.
- Playlists and live or upcoming streams are turned away before downloading. Paste a link to a single, finished video.

### Fixed

- An automation keeps posting the rest of its queue when one clip's details can't be verified.
- Generated post captions that quote the video are accepted when the speaker stutters, and captions can no longer contain web links.
- Hardened how BridgeClip handles captions, file paths, titles, downloads and its bundled Python, so unusual input can't change video processing commands or load unexpected code.

## [0.1.18] - 2026-09-25

### Added

- A signed installer for Windows x64. This version shipped for Windows only and has no other changes from 0.1.17.

## [0.1.17] - 2026-09-25

The first version with downloadable installers, starting with macOS.

### Added

- macOS apps for Apple silicon and Intel, signed with BridgeMind's Developer ID and notarized by Apple.
- Automatic updates. BridgeClip checks shortly after launch and every four hours, downloads in the background, and installs when you choose **Restart to update** or the next time you quit. Settings → About shows the status and has **Check for updates**.
- Clip public, finished Twitch VODs by pasting their link.
- **Video speed** in Create → Format speeds up every clip in a job, from 1.1× to 2×. Voices keep their pitch and captions stay in sync.
- TikTok accounts in automations. Each clip gets a review of its caption, audience, interactions and disclosures before it can post.
- **Economy** clipping mode, which uses lower-cost models and skips AI vision checks.
- **Advanced** clipping mode, for choosing your own transcription and planning models from OpenRouter.
- A **Posts** page with your posting history.
- Run stats that compare time and cost with OpusClip when there's enough information for a fair comparison.
- New app icons.

### Changed

- Captions are placed more carefully around faces.
- Jobs with a start and end time only transcribe that part of the video, which is faster and cheaper.
- Accounts, Automations and Jobs are more compact, and menus and confirmation dialogs work with the keyboard.
- Clip cost details name both transcription models when a job switches models partway through.
- Setting up a Zernio profile, and recovering from access problems, is clearer in Accounts.

### Fixed

- Automations could post the same clip twice after a slow or failed response.
- A clip you return to the automation queue after checking it by hand can post again.
- A scheduled slot stays free when a changed clip's TikTok approval is withdrawn, so you can review it and still post on time.
- Automation retries no longer stop at transcription with an HTTP 400 error.
- Page shortcuts no longer close a post dialog while it's uploading.
- Starting a job no longer fails on some IPv6 networks.
- Windows: Python is found reliably, edits with many cuts work, and saved results are opened safely.
- Exports whose audio drifts out of sync are caught before they're saved.

## [0.1.16] - 2026-09-24

### Added

- BridgeClip's source code is public under the MIT license. Installers start with 0.1.17.

[Unreleased]: https://github.com/bridge-mind/bridgeclip/compare/v0.1.19...HEAD
[0.1.19]: https://github.com/bridge-mind/bridgeclip/releases/tag/v0.1.19
[0.1.18]: https://github.com/bridge-mind/bridgeclip/releases/tag/v0.1.18
[0.1.17]: https://github.com/bridge-mind/bridgeclip/releases/tag/v0.1.17
[0.1.16]: https://github.com/bridge-mind/bridgeclip/tree/7107574215cc7f1d4059d10d62bd82014a3ba46f
