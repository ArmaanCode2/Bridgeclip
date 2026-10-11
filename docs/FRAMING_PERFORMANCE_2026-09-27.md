# Framing performance

Two measurements live here. They compare different things, so read the
baseline of each before quoting a number:

| Section | Compares | Baseline |
| --- | --- | --- |
| [September 28: against main](#september-28-against-main) | The shipped framing analysis on this branch | `main` 8a999a0 (v0.1.19) and the PR #63 head c7fd7a7 |
| [September 27: fork-internal fix](#september-27-fork-internal-fix) | Two versions of the PR's own detailed pass | The fork's earlier code, **not** main |

The September 27 speed-up (about 23×) was measured against the fork's own
earlier, slower detailed pass. It was never a comparison with main. Against
main, that version of the PR was still 2–5× slower, because it decoded every
window three times (face sampling, camera scan, detailed frames) where main
decodes once.

## September 28: against main

### What changed

Framing analysis (`LayoutAnalyzer.analyze`) now decodes each window once:

- One FFmpeg process splits the decoded picture. A 320-px branch runs the
  camera-change scan (the same filter chain as `scan_camera_changes`, so the
  editor gets identical timestamps and markers). An analysis-size branch
  streams frames as I420: every frame up to about 40 fps, every other frame
  at 60 fps.
- Face samples are the streamed frames nearest each 250 ms tick, labelled
  with their exact source time. Around a camera change, only the frames that
  decide it are analyzed. A strong cut uses its first frame. A weak cut also
  uses the frame before it and two frames within the next 350 ms. The
  frame budget caps instead of raising.
- Face detection runs on a small shared thread pool (up to 4 threads) while
  FFmpeg keeps decoding.
- Padded-inset detection runs on a 160-px copy, with the edges then located
  at full resolution: about 0.4 ms per frame, down from 4.8 ms, with
  identical boxes.
- Cut confirmation uses binary search: 5.4 s down to about 0.03 s for a
  60-minute window with 5,000 markers.
- Pacing-only analysis (16:9 output, Classic style) skips the camera scan
  and decodes exactly as main does.
- A window longer than the scan's frame budget (duration × frame rate over
  120,000 frames) is refused before decoding and analyzed as main does.

### Method

- Apple M3 Ultra (28 cores), the shipped LGPL FFmpeg 8.1.3 (`engine-bin`),
  Python 3.12 and OpenCV 4.14.
- Window: 60 s starting at 10 s, 1920×1080 source, `style=auto`, default
  `vision=True` (the automatic path) with the vision model disabled, so no
  network calls. "Pacing only" is `vision=False`.
- Each configuration ran in its own process. Five rounds interleaved the
  configurations, and the table shows medians. The machine was shared (load
  average about 15), so absolute times are noisy. Compare rows within a
  round.
- CPU is user + system time of Python and FFmpeg together.

Sources, generated with the shipped FFmpeg:

```sh
# 30 fps, no cuts
ffmpeg -f lavfi -i "testsrc2=s=1920x1080:r=30:d=75" \
  -c:v h264_videotoolbox -allow_sw 1 -b:v 12M -g 60 -pix_fmt yuv420p steady.mp4
# 30 fps, a hard cut every 2 s (60 fps: r=60, -b:v 16M -g 120)
ffmpeg -f lavfi -i "testsrc2=s=1920x1080:r=30:d=75" -f lavfi -i "mandelbrot=s=1920x1080:r=30" \
  -filter_complex "[1:v]trim=duration=75,setpts=PTS-STARTPTS[m];[0:v][m]overlay=enable='lt(mod(t\,4)\,2)',format=yuv420p" \
  -c:v h264_videotoolbox -allow_sw 1 -b:v 12M -g 60 cuts.mp4
```

### Results

Wall time, median of five runs, with CPU time in parentheses:

| Source | main 8a999a0 | PR head c7fd7a7 | This branch | This branch, pacing only |
| --- | ---: | ---: | ---: | ---: |
| 1080p30, no cuts | 2.01 s (13.2 s) | 3.85 s (23.0 s) | **1.65 s** (17.6 s) | 1.57 s (12.7 s) |
| 1080p30, cut every 2 s | 2.03 s (15.5 s) | 9.61 s (52.8 s) | **1.74 s** (19.8 s) | 1.55 s (14.5 s) |
| 1080p60, cut every 2 s | 2.43 s (22.2 s) | 16.13 s (90.0 s) | **2.71 s** (29.7 s) | 2.42 s (21.9 s) |

- At 30 fps, the automatic path is 14–18% faster than main in wall time.
  It also adds an every-frame camera scan that main does not have.
- At 60 fps, it is about 10% slower than main in wall time (2.62 s vs 2.39 s
  best of five). The scan reads twice as many frames there, and FFmpeg
  decoding is the bottleneck. The pacing-only path matches main.
- CPU time is about 30–35% higher than main on the automatic path, from the
  scan and streaming branch. Detection itself uses about the same CPU as
  main.
- On the PR head, the cut source's detailed pass also failed at the end
  ("Incomplete precise layout frame": the scan kept a frame at exactly the
  window end), so those 9.6 s produced no camera scan. That bug is fixed.

## September 27: fork-internal fix

> Historical record. This compares two versions of the PR's own detailed
> pass, not main. The separate detailed decode it describes no longer exists
> (see above).

The slow review preparation was dominated by local video decoding, rather than Jev. A sparse-frame FFmpeg command placed its duration limit after the input. When the selected frames ended before that limit, FFmpeg could continue decoding the remainder of the source while the caller waited for EOF. Each candidate repeated this work.

### Evidence from the saved run

The inspected run used a 50:15.7 source (3840×2160 VP9, approximately 23.976 fps) and prepared 21 candidates. Its saved stage timings were:

| Stage | Elapsed |
| --- | ---: |
| Download / read video | 2:13 |
| Understand source | 0:21 |
| Transcribe | 1:09 |
| Find moments | 1:13 |
| Frame & review | 65:01 |
| Save files | 0:03 |
| Build preview | 8:26 |

For all 21 candidates, the recorded Jev review completion followed the layout-plan log by approximately 0.45–0.52 seconds. These are observed intervals, not provider-side timings. They point to framing as the primary bottleneck in this run.

### Fix and local comparison

The detailed pass (then `LayoutAnalyzer._precise_frames`) was changed to put `-t` before `-i`, bounding input decoding to the camera-scan interval. FFmpeg documents the distinction between input and output duration limits in its [main options reference](https://ffmpeg.org/ffmpeg.html#Main-options).

Both versions of the fork were profiled locally using candidate 16's 30.679-second excerpt from the saved source. No provider requests were made. Main was not measured.

| Measurement | Fork before | Fork after |
| --- | ---: | ---: |
| Initial face sampling | 4.08 s | 6.34 s |
| Camera scan | 3.55 s | 6.39 s |
| Detailed face tracking | 317.41 s | 7.37 s |
| Combined camera scan + detailed tracking | 320.97 s | 13.76 s |
| Scanned frames | 759 | 759 |
| Analyzed frames | 165 | 165 |

The combined detailed pass was approximately 23.3× faster than the fork's earlier version. Sampling and scan timings varied with machine load. This was a single-candidate comparison, not an end-to-end speed guarantee, and the full 50-minute job was not rerun. Build preview is a separate cost.

### Visibility for subsequent jobs

The progress screen reports the candidate index, source-excerpt duration, current subtask, subtask progress when available, and cumulative time per phase:

- Face sampling
- Camera-change scanning
- Detailed face tracking
- Shot-layout checks
- Jev editorial review

Since September 28 the camera-change scan and cut details share face sampling's single decode, so their time is reported under face sampling. Camera-change scanning stays near zero, and detailed face tracking covers only detection still running after decoding ends.

Usage rows are grouped by requested model and pipeline stage, with calls in flight, failures, input/output tokens, and provider-reported cost. Responses update usage; a one-second heartbeat keeps elapsed diagnostics current during local processing. Missing usage and cost remain explicitly unknown. Cached Jev responses do not count as network requests. Completed jobs persist this breakdown alongside their stage timings; earlier jobs cannot recover telemetry that was never recorded.
