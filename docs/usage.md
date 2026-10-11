# Using BridgeClip

[Overview](../README.md) · [Editor](editor.md) · [AI and privacy](ai-and-privacy.md)

Install BridgeClip, create clips, follow jobs, and manage your Library and publishing queues. These instructions describe the current source; check the [release notes](https://github.com/bridge-mind/bridgeclip/releases) for the features in your installed version.

## Install and update

Choose your platform on [bridgeclip.ai](https://www.bridgeclip.ai) or [GitHub Releases](https://github.com/bridge-mind/bridgeclip/releases). Packages include Python, FFmpeg and yt-dlp.

- **macOS (Apple silicon or Intel):** open the matching DMG and drag BridgeClip to Applications. Official builds are Developer ID signed and notarized by Apple.
- **Windows x64:** run the signed EXE installer.
- **Linux x64:** install the DEB with your package manager, or make the AppImage executable before opening it. An unlocked desktop secret service is required to save API keys.

See [release verification](RELEASING.md#verify-a-download) for signatures and checksums.

### Automatic updates

BridgeClip keeps itself up to date. It checks [Releases](https://github.com/bridge-mind/bridgeclip/releases) shortly after launch and every four hours, downloads a new version in the background, and installs it when you choose **Restart to update** (in the sidebar or **Settings → About**) or the next time you quit. macOS only installs an update signed by the same developer, and every download is checked against the SHA-512 published with the release. A build not signed by BridgeMind, such as one you package yourself with `npm run dist:win`, never updates itself, so an official release cannot replace it.

Copies run from source, local package builds and apps opened straight from the disk image don't update themselves; **Settings → About** says why. To turn updates off, start BridgeClip with `BRIDGECLIP_DISABLE_AUTO_UPDATE=1`.

## Create your first clips

1. Add an [OpenRouter API key](https://openrouter.ai/keys) with available credit in the setup card. A saved key is never shown again; paste a new one to replace it, or choose **Remove key** in Settings.
2. Open **Create** and use the file picker for a local video, or paste a public YouTube or completed Twitch VOD link. Dropping a local file opens the picker so you can grant access.
3. Choose **Automatic** for finished exports, or **Review & edit** to adjust candidates in the [editor](editor.md).
4. Choose format, framing, clip lengths, models and caption style, then generate.
5. Follow progress in **Jobs**. Finished clips appear in **Library** and your output folder (by default, `~/BridgeClip`).

Keys are encrypted with your operating system's secure storage. If secure storage is unavailable, BridgeClip asks you to configure or unlock it before saving keys.

Only download or clip material you have permission to use. Remote sites may limit downloads or change access rules.

### Choose a workflow

In **Create**, choose a workflow before continuing: **Automatic · Beginner friendly** finds and exports clips; **Review & edit · For advanced users** lets you choose and adjust candidates before export. Each new video starts with neither workflow selected. YouTube links show a video card with its title, channel and thumbnail, plus duration, views and upload date when available. Metadata loads in the background and never blocks clip setup.

**Review & edit** always reviews candidates: with Jev on OpenRouter, or with your own model on Gemini or an OpenAI-compatible endpoint. For Automatic, the review is off by default. On OpenRouter, reviews use OpenRouter credit; see [review behavior and costs](ai-and-privacy.md#jev-editorial-review).

### How clips are picked

The number of clips is a limit, not a quota. In Automatic, the AI scores every candidate on hook, standalone clarity, story arc, quotability and ending, and a clip averaging below 60/100 is left out, so you can get fewer clips than you asked for. If nothing reaches 60, the best two are kept. The run's plan (`plan.json` in its folder) records how many weaker moments were left out, and the job shows it briefly while planning. **What to clip** requests, reviewed runs (Review & edit, or Automatic with the AI review on), 16:9 longform episodes and silent videos planned from frames are not filtered by score.

Automatic picks also get a second opinion (**Settings → Clip picking → Second opinion**, on by default). When the video has room for more moments than you asked for, the AI offers up to three times as many candidates: at most 30, and at most half as many as fit in the range, so it never has to cover the whole video. Separate requests to the same model then score each candidate from the words spoken inside it only, without its title or first scores, on hook, payoff, standalone clarity, energy and shareability. A clip's final score is 70% the second opinion and 30% the first; the 60/100 minimum applies to that score, and all candidates are ranked together. Each request scores up to eight candidates from one stretch of the video, so a run makes about one extra AI request per eight candidates, and the planner writes a longer answer. If any request fails twice, or the model's answer leaves a clip out, the clips are ranked by the first scores alone and the plan says so. `plan.json` records how many candidates were scored, and each clip's `ranking` holds both scores and the reason given. The second opinion reads only words, so a moment that is all action and no speech can score lower than it deserves. **What to clip** requests, reviewed runs, clips that may run longer than 5 minutes, 16:9 longform episodes, and videos with no room for extra candidates do not use it.

Videos longer than 35 minutes are searched in parts of about 25 minutes that overlap by a minute (a long break in the speech ends a part without overlap), and the candidates from every part are ranked together. The middle of a long stream gets the same attention as the start, and a local model with a 32,000-token context window can plan a multi-hour video. Three cases still send the whole transcript in one request: 16:9 longform episodes, Review & edit with a time range, and the reviewed second search for more moments. An OpenAI-compatible endpoint gets one part at a time, since local servers answer one request at once; OpenRouter and Gemini get two at once. If one part fails for a temporary reason, the others still count and the plan names the part that was skipped.

Gaming streams and livestreams are recognised as their own kind of video: clutch plays, fails, comebacks and the streamer's reactions count as strong moments even when nothing quotable is said.

Clips start at the beginning of a sentence. A sentence starting within half a second after the AI's start time is taken as the one it meant. Otherwise a start inside a sentence moves back to where it began (up to 3 seconds), or forward to the next sentence (up to 2 seconds) when that keeps the clip long enough. A pause of 0.7 seconds also counts as a sentence start, since live commentary is often transcribed without punctuation.

### Video and audio quality

Exports keep the source's frame rate up to 60 fps, so 60 fps gameplay stays smooth in vertical clips too (they used to be 30 fps), and 24 or 25 fps footage is not resampled. A variable-rate recording that averages under 23 fps still becomes a 30 fps vertical clip.

When FFmpeg has x264 and not OpenH264 (a full FFmpeg build, as when running from source), clips are encoded with x264's `medium` preset at CRF 18. That takes about three times as long to encode as the old `veryfast` preset and keeps fast motion and on-screen text cleaner. The installed app's FFmpeg uses OpenH264 at a fixed bitrate instead. On Windows and Linux, **Settings → Output → Video encoder → NVIDIA (faster)** encodes on an NVIDIA graphics card (NVENC): several times quicker, about the same quality in our tests, with files 1.1 to 2 times larger. Each job first encodes one test frame on the card and uses the CPU if that fails; if the card's encoder fails mid-job, the clip is encoded again on the CPU and the rest of the job stays on the CPU. Editor previews always use the fast preset.

Every clip's volume is levelled in two passes. BridgeClip measures the loudness of the audio the clip keeps (not stretches the edit cuts out), then applies one gain to the whole clip to reach -14 LUFS, so the volume no longer rides up and down within a clip. When that gain would push peaks too high, or the clip goes from quiet talk to shouting, a limiter catches only the peaks. Very quiet sources are raised by at most 20 dB.

### Models

In **Create → Clips**, choose **Quality**, **Economy**, or **Advanced**. Advanced offers searchable OpenRouter model pickers for transcription and clip planning, with model IDs, planning prices and compatibility notes. Both selections are required and appear in Review. Advanced retries the selected models without automatically switching models. Transcription must provide word timestamps; planning must support structured output. See [model selection and transcription](transcription.md).

### Framing, captions and speed

Smart framing follows faces and arranges screen shares with facecams shot by shot. Optional AI vision checks help with ambiguous layouts and can add OpenRouter cost. Use [Review & edit](editor.md) to adjust crops and camera changes before exporting.

Without an AI that reads images, BridgeClip finds the webcam itself. A small face that stays in one place near an edge of the frame is the webcam: in a corner, or at the bottom centre or the middle of a side when a border is found around it and the face is changing. A face that roams around next to your webcam is treated as part of the game, and a face that never changes while the game moves (a frozen portrait) is never the webcam. When several faces could be the webcam, the one that looks most like one wins. A border found around it counts most; being there through the whole clip and the whole shot, and a head that moves a little around one place, count next; a face that changes and a usual webcam size count a little. A face that holds pixel-still over two seconds or more (a game's character portrait) counts against itself, even when video compression makes the portrait look as if it changes. While following the webcam through a shot, one missed detection never hands it to another face, and faces that sit still elsewhere through the shot are never followed. A face in the game that is there all the time, moves like a person and has a frame around it can still be mistaken for the webcam; choose **Webcam in your videos** for that. The webcam's box comes from the edges that stay put while the game around them moves, so an overlay is cropped to its real edges whether or not it has a drawn border; an AI that reads images keeps its own box unless the two agree. When no face at all turns up at the usual detection size in a shot of at least 3 seconds, five sharper frames (twice the detection size, 1280 by 720 for a 1080p video) from up to 20 seconds of the shot are checked for a small, changing webcam face, for up to six shots per clip. That adds a few seconds per shot, and a webcam found that way must stay in place for the whole shot.

**Settings → Framing** has three choices for Smart framing. **Webcam in your videos** says where the webcam sits in the videos you clip: **Auto** (the default) finds it, or choose **Top left**, **Top right**, **Bottom left**, **Bottom right** or **Bottom centre**. With a place chosen, only a small face there can be the webcam, with no border needed, and small faces near other edges are left as part of the game: never the webcam, never followed as a person, and no reason to show a screen whole instead of filling it. The places are the same corner and bottom-centre bands Auto uses, so a presenter or a larger face is still treated as a person. An AI that reads images is overruled when it puts the webcam elsewhere. Choose a place only when every video you clip has the webcam there: in a video with it elsewhere, or a webcam large enough that its face sits near the middle of a side, it gets no webcam panel. **Webcam position** puts the webcam panel above the screen (the default) or below it when a screen and a webcam are stacked. **Screen with no webcam** either fills the whole vertical frame (**Fill**, the default) or shows the whole screen over a blurred copy of itself (**Whole**). Fill crops around the area the AI marks as important, such as the game view or the active window, when your AI can read images; otherwise it keeps the centre. A screen still shows whole when filling would enlarge it too much (sources below 720p, or a small screen area) or when people are on camera in it, such as a panel of three, so nobody is cut off. With the webcam on top, the title card sits below the captions instead of over the face. Full Frame and Classic framing do not use these settings.

Choose from nine caption styles: Viral, Hormozi, Bold, Clean, Minimal, Fire, Glow, Neon and Karaoke. In **Create → Captions**, the selected style plays an animated sample showing how words appear and highlights advance. Pause or restart it to compare styles.

In **Create → Format → Video speed**, choose **1×** (normal), **1.1×**, **1.25×**, **1.5×**, **1.75×**, or **2×** for every clip in the job. Exports preserve voice pitch and keep captions synchronized. Speed works with **Cut dead air** and appears in Review and the saved results. Clip lengths and source trim times refer to the original footage: a 60-second clip at 1.5× exports in about 40 seconds, before any dead-air cuts. The choice stays selected when you clip another video in the same session.

Existing exports stay as they are; generate a new job to change their speed.

See [video speed](video-speed.md) for timing and export details.

### Clip a Twitch VOD

Paste a public, completed Twitch video link such as `https://www.twitch.tv/videos/1234567890` into Create, then choose your clip settings and generate. BridgeClip downloads the saved video and uses the same transcription, AI moment selection and rendering flow as other sources. Links on `twitch.tv`, `www.twitch.tv`, `m.twitch.tv` and `go.twitch.tv` are accepted and normalized to the canonical video URL.

Live channels, Twitch clips, collections, subscriber-only videos and deleted or expired VODs are not supported. No Twitch login or cookies are used. The original source must be at most six hours and 20 GB. BridgeClip downloads the full source before applying the optional start and end times; a link's timestamp or tracking parameters are ignored. For a longer source, trim a downloaded file before adding it. Downloads also stop after four hours or when less than 1 GB of free space would remain.

## Follow jobs

Every run gets its own folder. The **Library** shows completed clips with virality scores, timecodes and tags. **Jobs** shows what is running or queued right now (up to two clipping runs go at once; more wait in a queue) and every earlier run, including completed, failed, cancelled and interrupted jobs; completed runs open directly in Library, and failed runs from this session can run again.

Each previous job has a three-dot menu for **Open in Library** (or **Open job** for session jobs), **Open folder** and **Details**. Details opens the saved transcript and edit trace, including for failed runs. Older runs without a saved status appear as unfinished. You can optionally connect social accounts through Zernio to publish or schedule a selected clip.

### Progress and cost

During creation, each stage shows its own elapsed time and progress: downloaded bytes, transcription chunks, prepared candidates, rendered clips, saved files and editor-preview encoding where measurable. AI requests without measurable progress show an indeterminate bar. The overall percentage is an estimate; completed runs retain a **Processing time by stage** breakdown in the Library and editor. Each stage keeps its own color in the time breakdown. The progress page reuses the source preview from Create for YouTube and local files.

New runs show live model requests, input/output tokens and provider-reported cost; missing usage stays unknown and partial totals exclude unreported charges. **Inside frame & review** separates face sampling, camera scans, detailed face tracking, shot-layout checks and Jev review, with the current candidate and accumulated timings. These diagnostics are saved with the run and remain available in its processing-time details. See [the framing performance investigation](FRAMING_PERFORMANCE_2026-09-27.md) for the long-video decoding fix.

## Organize your Library

### Find clips and track posting

Inside a Library run, **Search clips** filters by title, tag or clip number across both posted and unposted clips. Selection and bulk actions apply only to visible results. The YouTube icon beside **Inspect transcript & edits** opens the original video in your browser when a YouTube source link is saved. **Not Posted** and **Posted** are independently collapsible sections with clip counts. **Not Posted** opens by default and appears first; **Posted** starts collapsed below it. Selection applies only to expanded sections.

Badges distinguish scheduled, publishing, partially posted and failed posts using this workspace’s saved posting history.

### Clip actions and publishing drafts

Each clip’s **…** menu contains posting, adding to an automation, opening its folder, and **Delete clip**. Delete a single clip directly from this menu without selecting it first; the confirmation names the clip before permanently removing its local export. Choose **Mark as posted** for clips you published yourself; the mark persists without a connected account and can be undone from the same menu. It changes local Library status only.

Select clips to reveal the red trash button for deleting just those exports, with confirmation; source footage, editor edits and other clips are preserved. The post dialog can enhance platform-specific titles, captions and tags using the same transcript, source context and optional research as Automations; review and apply the draft before posting.

### Bookmarks and run deletion

Library overview cards show the total clip count at the top left and separate **Posted** / **Not Posted** counts below the title. Bookmark a run to keep it in the **Bookmarked** section. Cards move smoothly between Bookmarked and Recent runs without reloading their thumbnails; removing the last bookmark returns to one **All runs** grid. Bookmarks persist across restarts, and reduced-motion preferences turn off movement.

The trash action asks for confirmation before permanently deleting the run folder, all clips and other files inside it, and cached clip previews. It does not remove published social posts or copies stored elsewhere. Active runs cannot be deleted.

**Settings → About → Content storage** shows the total file size and file count in your output folder. Refresh to recalculate; inaccessible files are reported as a partial total.

## Publish and automate

Connect social accounts with your own Zernio account and API key to post or schedule clips. Metadata enhancement uses your AI provider. For platform-specific drafts, scheduling consent and required TikTok reviews, see [publishing and metadata](automation-metadata.md). **BridgeClip must be open for daily automations to run.**

### Organize the content bank

In **Automations**, drag a queued clip’s handle to change its order with animated movement and automatic scrolling. For keyboard sorting, focus the handle, press Space to pick up the clip, use the Up/Down arrow keys, then press Space to drop or Escape to cancel. Submitted or uncertain items stay in place. The bank separates **Queued**, **Needs attention** (when needed), and **Submitted** clips. The Queued heading shows the next posting slot and its timezone, or indicates that scheduling is paused or unconfigured. Hover or focus each row’s info icon for its added/submitted timestamps, source, caption and metadata status.

Use the actions menu to edit a bank item, **Show in Finder** (**Show in folder** on other systems) to reveal its saved video, or **Remove from queue**, and **View in Library** to open its source run filtered to that exact clip, including submitted clips and items with enhanced titles. Choose **Show all clips** to return to the full run. Submitted clips have no removal option. Removing a queued clip leaves the original file intact.

Reviewed metadata is marked **Enhanced** in the info tooltip and hides the Enhance button; selecting a platform that still needs metadata makes enhancement available again.

**Enhance by source video** works with both linked videos and attached files. Select a source and optionally describe the video or suggest an emphasis for its titles, captions and tags. Review the generated draft before applying it.

### Warnings and recovery

In **Automations**, the header shows scheduling status once. **More settings** holds caption writing, YouTube options and the automation name; previous failures are expandable historical messages. Use **Acknowledge warnings** for one automation or **Acknowledge all warnings** for every automation to clear existing failure indicators without a successful post. Acknowledgements survive restarts; new failures warn again. Messages remain in the run details or clip info, and acknowledged clips awaiting review appear under **Held clips** without being retried or returned to the queue.

Dismiss a clip’s upload or metadata error with its **×** button; dismissal survives restarts, preserves the message in clip info, and a new failure shows the warning again. **Run now** prioritizes ready clips without previous upload errors. Use **Retry clip** to retry a specific failed upload without changing queue order, even after dismissing its error. Retry is available only for queued clips that have not been submitted; metadata drafts and required TikTok review must be resolved first.

Held clips have a visible **Return to queue** action and, when linked to a Zernio post, **Refresh post status**. A fresh Zernio check moves already published or active posts to **Submitted** and permits requeuing only fully failed posts; partial or uncertain results stay held for review in Posts. Returning a failed linked clip disables Retry on its old Posts entry so only the new queue attempt can post.

Recovery also works when local post history was removed or aged out, provided Zernio returns complete post details. Returning an unlinked clip requires confirming it was not published. Required setup and TikTok approvals still apply.

### Check account health

**Check Zernio status** makes a fresh, read-only account health request and shows connection/permission issues without uploading or posting. Zernio’s documented health API does not expose its protective upload cooldown, so the result explicitly leaves that hold unconfirmed even when the connection is healthy. Failed checks never fall back to cached success.

## Troubleshoot a run

- Run **Settings → System check** and use the in-app error to identify missing dependencies or provider issues. In development, set **Python path** if the virtual environment is not detected.
- If a link fails, check it in a signed-out browser or download the video yourself and select the local file.
- Open a job’s **Details** to inspect its saved transcript and edit trace, including failed runs. Inspecting saved results makes no provider calls. See [saved reviews and diagnostics](ai-and-privacy.md#inspect-saved-reviews).
- Logs omit raw provider responses and private source details, but review logs and run files before sharing them. See [local storage and cleanup](ai-and-privacy.md#local-storage-and-cleanup).

For help, use [GitHub Issues](https://github.com/bridge-mind/bridgeclip/issues) or [Discord](https://www.bridgemind.ai/discord). Report security issues using [SECURITY.md](../SECURITY.md).
