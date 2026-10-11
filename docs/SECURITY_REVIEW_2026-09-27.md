# Security and release review — 2026-09-27

The v0.1.19 candidate reconciles the original local rewrite with public main at `f6c7226`. Already-shipped updater, Windows network/file handling, video-speed, selected-range transcription, UI and packaging changes are preserved. The pending YouTube retry and automation queue/grounding improvements are included. Local social artwork and regression tests are retained. Historical public workflow definitions, old release packaging, the unused callback server, stale model selections and regressions to newer source are not reintroduced. The original checkout remains intact.

## Findings fixed

- Provider transcript words could inject ASS sections/SRT cues through line breaks. Normalize both at the transcription boundary and in caption writers.
- FFmpeg filter paths were escaped for only one parser, allowing apostrophes/colons in paths to alter filter options. Escape both parsing levels and exercise real captioned renders from hostile paths.
- Unbounded model titles could allocate very large raster images; malformed numeric timings could overflow. Bound titles/summaries and reject overflowing timing values.
- Planner JSON and loudness fallbacks used backtracking regular expressions on untrusted text. Use linear JSON/block parsing.
- Desktop object-storage links could select the AWS SDK and ambient credentials. Use anonymous direct downloads and reject bare object keys in local mode.
- Playlist/channel and unknown-duration downloads could bypass the single-video resource model. Reject them before download; flatten playlist metadata extraction.
- Frame sampling captured unbounded FFmpeg diagnostics. Use the shared output/time-limited media runner.
- yt-dlp could load local plugin code; `python -m` launchers could import a planted package from the working directory. Disable plugins in the worker and use `-P` in every bundled launcher, including private macOS workflows.
- Library folder opening could execute additional macOS package types. Reject executable/installable bundle suffixes throughout the path.
- Generated automation captions allowed scheme-less `www.` links. Reject these and URL scheme delimiters with a bounded check. Bound fuzzy evidence matching to prevent repetitive provider responses stalling the main process.
- npm integrity validation accepted a registry tarball for the wrong package/version. Require the exact package's tarball.
- Private CI/release tools were not fully hash locked. Pin their transitive Python dependencies and enforce hashes.
- The private fallback publisher did not enforce required assets for `all`. Require every target's package, updater feed and corresponding source; reject unknown platforms, even for otherwise correctly signed manifests.

## Validation

Local typechecking, lint, application/main/renderer/posting/bridge/release tests and production build passed. The engine suite passed 522 tests with one optional test skipped, using FFmpeg with libass. npm audit reports zero vulnerabilities; pip-audit reports no known vulnerabilities in the locked engine requirements. Gitleaks reports no leaks in the candidate. Private workflow tests cover signed-but-incomplete all-platform publication; actionlint passes.

Native CI, package rehearsals, signed release acceptance and publication are required separately and recorded on the PR and release workflow. The test suite uses local/mock provider responses: it does not prove paid provider availability or live social posting. Dependency audit results describe known advisories at review time, not an absence of all vulnerabilities.
