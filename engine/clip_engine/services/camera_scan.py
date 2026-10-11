"""Local, every-frame cut suggestions. Never edits a user's layouts."""
import json
import math
import re
import subprocess
import sys
from fractions import Fraction
from typing import Optional

from .media_process import MEDIA_INPUT_OPTIONS, PROBE_TIMEOUT_SECONDS, media_process, run_media

MAX_FRAMES = 120000
MAX_MARKERS = 5000
MIN_SCORE = .025
# Decode this much before the window so a cut on its first frame is scored.
CONTEXT_MS = 1000
# Below this many seconds a scan cannot reach MAX_FRAMES at any plausible
# rate, so the frame-rate probe is skipped.
MAX_EXPECTED_FPS = 240
# Scene scores of 320-px-wide frames on a microsecond clock. The layout
# analyzer runs this same chain inside its single decode, so both produce
# identical timestamps and markers.
SCAN_FILTER = "scale=320:-2,settb=1/1000000,select='gte(scene,0)'"
MAX_LINE_BYTES = 1024
_FRAME_LINE = re.compile(r'frame:\d+\s+pts:(-?\d+)\s')


def scan_window(start_ms, end_ms):
    """The decoded interval [start, end): the window plus leading context."""
    return round(max(0, start_ms - CONTEXT_MS), 3), round(end_ms, 3)


def probe_frame_rate(source) -> Optional[float]:
    """The video stream's average frame rate, or None when unknown."""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=avg_frame_rate,r_frame_rate', '-of', 'json',
           *MEDIA_INPUT_OPTIONS, str(source)]
    try:
        streams = json.loads(run_media(cmd, timeout=PROBE_TIMEOUT_SECONDS).stdout or b'{}').get('streams') or [{}]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    for key in ('avg_frame_rate', 'r_frame_rate'):
        try:
            rate = Fraction(str(streams[0].get(key, '0/0')))
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return float(rate)
    return None


def check_frame_budget(source, start_ms, end_ms):
    """Refuse before decoding when the stream's rate says the scan would overflow.

    Otherwise a long 60 fps window decodes MAX_FRAMES frames only to be
    discarded. The in-stream limit still guards variable-rate sources.
    """
    seconds = max(0, end_ms - start_ms) / 1000
    if seconds * MAX_EXPECTED_FPS <= MAX_FRAMES:
        return
    rate = probe_frame_rate(source)
    if rate is not None and seconds * rate > MAX_FRAMES:
        raise ValueError('Camera scan is too long')


def filter_path(path) -> str:
    """Escape a file path as a filter option value inside a filtergraph.

    Same two escaping levels as RenderingService._escape_filter_path: the
    graph parser, then the filter's own option parser.
    """
    path = str(path)
    if sys.platform == 'win32':
        path = path.replace('\\', '/')
    option = re.sub(r"([\\':=\s])", r"\\\1", path)
    return re.sub(r"([\\'\[\],;\s])", r"\\\1", option)


class ScanParser:
    """Frame timestamps and scene scores from FFmpeg's metadata printout.

    `records` holds one entry per decoded frame, in decode output order: its
    time, or None when it is outside [start, end) or not after the previous
    frame; `pts` holds its raw microsecond timestamp. Strict parsing raises on any budget or validity problem (a camera
    scan the user asked for). Lenient parsing only marks the scan unusable
    and keeps timing every frame, for a caller that needs the timestamps.
    """

    def __init__(self, start_ms, end_ms, strict=True):
        self.start_ms, self.end_ms, self.strict = start_ms, end_ms, strict
        self.frames: list[float] = []
        self.markers: list[dict] = []
        self.records: list[Optional[float]] = []
        self.pts: list[Optional[int]] = []
        self.scores: list[Optional[float]] = []
        # Records whose score line was read (or that a later frame closed).
        self.complete = 0
        self.usable = True
        self._pending = b''
        self._last: Optional[float] = None

    def feed(self, data: bytes):
        lines = (self._pending + data).split(b'\n')
        self._pending = lines.pop()
        if len(self._pending) > MAX_LINE_BYTES:
            raise ValueError('Invalid camera scan output')
        for line in lines:
            if len(line) > MAX_LINE_BYTES:
                raise ValueError('Invalid camera scan output')
            self._line(line.decode('ascii', errors='replace').strip())

    def finish(self):
        if self._pending:
            self._line(self._pending.decode('ascii', errors='replace').strip())
            self._pending = b''
        self.complete = len(self.records)

    def result(self):
        return {'start_ms': self.start_ms, 'end_ms': self.end_ms, 'frames': self.frames, 'markers': self.markers}

    def _fail(self, message):
        if self.strict:
            raise ValueError(message)
        self.usable = False

    def _line(self, line):
        if line.startswith('frame:'):
            self.complete = len(self.records)
            match = _FRAME_LINE.match(line + ' ')
            pts = int(match[1]) if match else None
            at = None if pts is None else round(self.start_ms + pts / 1000, 3)
            # [start, end): a frame exactly at the end belongs to the next window.
            if at is not None and (at < self.start_ms or at >= self.end_ms or (self._last is not None and at <= self._last)):
                at = None
            self.records.append(at)
            self.pts.append(pts)
            self.scores.append(None)
            if at is not None:
                self._last = at
            if at is not None and self.usable:
                self.frames.append(at)
                if len(self.frames) > MAX_FRAMES:
                    self._fail('Camera scan is too long')
        elif line.startswith('lavfi.scene_score=') and self.records:
            try:
                score = float(line.split('=', 1)[1])
            except ValueError:
                score = math.nan
            if not math.isfinite(score) or not 0 <= score <= 1:
                self._fail('Invalid camera score')
                return
            self.scores[-1] = score
            self.complete = len(self.records)
            at = self.records[-1]
            if at is not None and score >= MIN_SCORE and self.usable:
                self.markers.append({'at_ms': at, 'score': score})
                if len(self.markers) > MAX_MARKERS:
                    self._fail('Too many camera changes')


def scan_camera_changes(source, start_ms, end_ms, progress=None):
    start_ms, end_ms = scan_window(start_ms, end_ms)
    if progress:
        progress(0)
    check_frame_budget(source, start_ms, end_ms)
    parser = ScanParser(start_ms, end_ms)
    reported = 0
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{start_ms / 1000:.6f}',
           *MEDIA_INPUT_OPTIONS, '-i', str(source), '-t', f'{(end_ms - start_ms) / 1000:.6f}',
           '-map', '0:v:0', '-an', '-sn', '-dn', '-vf',
           f"{SCAN_FILTER},metadata=mode=print:file='pipe\\:1'",
           '-fps_mode', 'passthrough', '-f', 'null', '-']
    with media_process(cmd, timeout=20 * 60) as (process, _):
        while raw := process.stdout.readline(MAX_LINE_BYTES + 1):
            parser.feed(raw)
            if progress and parser.frames:
                percent = min(99, int((parser.frames[-1] - start_ms) / max(1, end_ms - start_ms) * 100))
                if percent > reported:
                    reported = percent
                    progress(percent)
        if process.wait() != 0:
            raise ValueError('Camera scan failed')
    parser.finish()
    if not parser.frames:
        raise ValueError('Camera scan has no video frames')
    if progress:
        progress(100)
    return parser.result()
