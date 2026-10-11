"""
Layout Analyzer - decides how each shot of a clip should be framed for 9:16.

For a clip it:
1. Decodes the window once with FFmpeg: low-res face samples (ANALYSIS_FPS)
   and, when exact cuts matter, an every-frame camera-change scan from the
   same decode plus the few frames that decide each change.
2. Detects faces per frame (OpenCV YuNet), shot cuts (HSV histogram jumps),
   and sustained changes between corner webcams and full-screen speakers.
3. Tracks faces within each shot and classifies the shot's layout:
     talking_head  one on-camera person          -> face-tracked full-frame crop
     two_shot      two people side by side       -> stacked split, one per panel
     screen_cam    screen content + webcam overlay -> screen panel over webcam panel
     screen        no usable person on camera    -> fit with blurred background
4. Optionally asks a vision LLM, once per distinct setup, to confirm the
   layout and return the real webcam / screen overlay rectangles. Face boxes
   alone underestimate a webcam overlay by 3-4x, which is what made the old
   split-screen renders look over-zoomed.

Every step degrades gracefully: without OpenCV or the face model the clip
falls back to the classic letterbox render; without the vision model the
heuristic classification is used.
"""

import asyncio
import base64
import json
import hashlib
import logging
import math
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Optional

import httpx

from clip_engine.config import LayoutStyle, get_output_dimensions, get_settings
from clip_engine.services.media_process import MEDIA_INPUT_OPTIONS, media_process, MediaProcessError, run_media, validate_video_dimensions
from clip_engine.services.ai_client import (
    ai_chat_completion,
    get_ai_client,
    has_credentials,
    prepare_payload,
    supports_images,
    tolerant_json_parse,
    uses_openrouter,
)
from clip_engine.services.openrouter import (
    OpenRouterError,
    apply_reasoning,
    chat_completion,
    json_schema_format,
    message_text,
)

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np

    # OpenCV 5's DNN graph engine warns about unsupported targets on every
    # detector creation; it's harmless noise in job logs.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except ImportError:  # pragma: no cover - exercised only without the CV extras
    cv2 = None
    np = None


# ------------------------------------------------------------------
# Tunables
# ------------------------------------------------------------------

ANALYSIS_FPS = 4.0
ANALYSIS_WIDTH = 640
KEYFRAME_WIDTH = 768
KEYFRAME_FPS = 1.0
MAX_ANALYSIS_HEIGHT = 1280
MAX_ANALYSIS_DURATION_MS = 60 * 60 * 1000
MAX_ANALYSIS_FRAMES = int(MAX_ANALYSIS_DURATION_MS / 1000 * ANALYSIS_FPS) + 1
MAX_RETAINED_KEYFRAMES = 256
MAX_KEYFRAME_BYTES = 32 * 1024 * 1024
MAX_FACES_PER_FRAME = 32
# A shot in which no face turned up at analysis size gets a second look at
# this multiple of the analysis scale (1280x720 for 1080p), from a few frames:
# small webcams have faces too small to detect at analysis size. Only Smart
# framing pays for it, only when the source is that much sharper, and only for
# so many shots.
SHARP_SCAN_FACTOR = 2.0
SHARP_SCAN_MIN_GAIN = 1.5
SHARP_SCAN_SAMPLES = 5
SHARP_SCAN_MIN_HITS = 4
SHARP_MIN_FACE_HEIGHT = 0.02
SHARP_SCAN_MIN_SHOT_MS = 3000
SHARP_SCAN_SPAN_MS = 20_000
MAX_SHARP_SCANS = 6
# Inset (padding) detection runs on a copy this wide.
CONTENT_BOX_WIDTH = 160


# Face detection runs on a few threads while FFmpeg keeps decoding. YuNet
# barely uses a second core by itself; separate detectors scale nearly
# linearly. One pool is shared by concurrent clips to bound total CPU.
DETECTION_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))
MAX_PENDING_DETECTIONS = 4 * DETECTION_WORKERS
# How long the single-pass analysis waits for a frame's scan timestamp.
SCAN_OUTPUT_WAIT_S = 30
# The scan reads every frame, but frames closer together than this are not
# streamed for face analysis (a 60 fps source sends every other frame).
# Integer microseconds, so the reader reproduces FFmpeg's choice exactly.
MIN_STREAMED_GAP_US = 24500
_detection_pool: Optional[ThreadPoolExecutor] = None
_detection_pool_lock = threading.Lock()


def detection_pool() -> ThreadPoolExecutor:
    global _detection_pool
    with _detection_pool_lock:
        if _detection_pool is None:
            _detection_pool = ThreadPoolExecutor(DETECTION_WORKERS, thread_name_prefix="face-detection")
        return _detection_pool


def analysis_dimensions(width: int, height: int) -> tuple[int, int]:
    validate_video_dimensions(width, height)
    scale = min(ANALYSIS_WIDTH / width, MAX_ANALYSIS_HEIGHT / height)
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


FACE_SCORE_THRESHOLD = 0.72
STRONG_FACE_SCORE = 0.90
COMPETING_FACE_SCORE = 0.85
# Bhattacharyya distance between consecutive HSV histograms that counts as a cut.
SHOT_CUT_THRESHOLD = 0.42
MIN_SHOT_MS = 1200
# Require three consecutive observations before changing layout. Missing faces
# need a longer hold: looking away must not be mistaken for a scene change.
LAYOUT_CHANGE_MS = 500
LAYOUT_CHANGE_SAMPLES = 3
# Samples carry their frame's true time, up to half a frame from the 250 ms
# grid, so three of them can span slightly less than 500 ms at 20-30 fps.
SAMPLE_JITTER_MS = 50

# A face track must be visible in this share of a shot's frames to count.
MIN_TRACK_PRESENCE = 0.35
MAX_TRACK_GAP_MS = 1000
# Faces smaller than this (fraction of frame height) sitting in a corner are
# treated as a webcam overlay rather than an on-camera person.
OVERLAY_MAX_FACE_HEIGHT = 0.17
# Faces smaller than this are ignored entirely (crowds, posters, thumbnails).
MIN_FACE_HEIGHT = 0.035
# Share of a webcam overlay's height its face spans (head-and-shoulders framing).
CAM_FACE_SHARE = 0.28
# A face inside a padded landscape inset (4:3 pillarbox, a framed 16:9
# picture) is followed rather than fitted when it is the only face, visible for
# most of the shot, and the inset is tall enough that a crop inside it stays
# sharp. Square and portrait insets already suit a 9:16 frame when fitted.
INSET_SPEAKER_PRESENCE = 0.6
INSET_FILL_MIN_HEIGHT = 0.6
INSET_FILL_MIN_ASPECT = 1.2

# Face-tracked crop camera: ignore motion inside this share of the crop
# width, and cap how fast the virtual camera pans (crop widths per second).
CAMERA_DEADZONE = 0.12
CAMERA_MAX_SPEED = 0.9
MAX_PATH_KEYFRAMES = 40


class LayoutType:
    TALKING_HEAD = "talking_head"
    TWO_SHOT = "two_shot"
    SCREEN_CAM = "screen_cam"
    SCREEN = "screen"

    ALL = (TALKING_HEAD, TWO_SHOT, SCREEN_CAM, SCREEN)


# ------------------------------------------------------------------
# Data structures (all boxes normalized to 0-1 of the source frame)
# ------------------------------------------------------------------


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h

    def clamp(self) -> "Box":
        w = min(max(self.w, 0.0), 1.0)
        h = min(max(self.h, 0.0), 1.0)
        x = min(max(self.x, 0.0), 1.0 - w)
        y = min(max(self.y, 0.0), 1.0 - h)
        return Box(x, y, w, h)

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def to_list(self) -> list[float]:
        return [round(self.x, 4), round(self.y, 4), round(self.w, 4), round(self.h, 4)]


@dataclass
class FaceTrack:
    """One person's face across the frames of a shot."""

    samples: list[tuple[int, Box]] = field(default_factory=list)  # (t_ms, box)

    def median_box(self) -> Box:
        return Box(
            median(b.x for _, b in self.samples),
            median(b.y for _, b in self.samples),
            median(b.w for _, b in self.samples),
            median(b.h for _, b in self.samples),
        )


@dataclass
class FrameInfo:
    t_ms: int
    faces: list[Box]
    hist: Any  # np.ndarray
    scores: list[float] = field(default_factory=list)
    content_box: Optional[Box] = None


@dataclass
class ShotLayout:
    """How to frame one shot. Times are ms from the start of the render window."""

    start_ms: int
    end_ms: int
    layout: str
    source: str = "heuristic"  # heuristic | vision | style
    # What the shot actually shows, kept when a style overrides the framing
    # (pacing rules depend on content, not on how it is framed).
    detected_layout: Optional[str] = None
    # talking_head: virtual-camera focus path of (t_ms from shot start, cx, cy)
    focus_path: list[tuple[int, float, float]] = field(default_factory=list)
    # two_shot: the two people, left to right
    people: list[Box] = field(default_factory=list)
    # screen_cam
    screen_box: Optional[Box] = None
    # Where the action is inside the screen (active pane, chat, game view).
    screen_focus: Optional[Box] = None
    cam_box: Optional[Box] = None
    cam_face: Optional[Box] = None
    # A video inset surrounded by padding. Preserve its composition, rather
    # than following faces inside it or carrying an anchor across the cut.
    content_box: Optional[Box] = None
    # talking_head inside a padded inset (e.g. 4:3 pillarboxed in 16:9): the
    # face-tracked crop must stay within these bounds so no padding shows.
    crop_bounds: Optional[Box] = None
    cam_box_refined: bool = False
    manual_crops: list[tuple[float, float, float, float]] = field(default_factory=list)
    manual_from_crops: list[tuple[float, float, float, float]] = field(default_factory=list)
    manual_transition_start_ms: int = 0
    manual_transition_ms: int = 0
    # screen_cam: which end of the 9:16 frame the webcam panel sits at. Plans
    # saved before this setting existed had it at the bottom.
    cam_position: str = "bottom"
    # screen (no webcam): crop the screen to fill the frame instead of showing
    # it whole over a blurred copy of itself.
    screen_fill: bool = False

    def summary(self) -> dict:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "layout": self.layout,
            "source": self.source,
            "screen_box": self.screen_box.to_list() if self.screen_box else None,
            "screen_focus": self.screen_focus.to_list() if self.screen_focus else None,
            "cam_box": self.cam_box.to_list() if self.cam_box else None,
            "people": [p.to_list() for p in self.people],
            "content_box": self.content_box.to_list() if self.content_box else None,
            "crop_bounds": self.crop_bounds.to_list() if self.crop_bounds else None,
            "cam_box_refined": self.cam_box_refined,
            "cam_position": self.cam_position,
            "screen_fill": self.screen_fill,
        }


@dataclass
class ClipLayoutPlan:
    shots: list[ShotLayout]
    source_width: int
    source_height: int
    vision_cost_usd: float = 0.0
    # Every analyzed frame's faces as (t_ms in window time, boxes), including
    # frames without any. Captions use them to stay off faces.
    face_samples: list[tuple[int, list[Box]]] = field(default_factory=list)
    trace: Optional[dict] = None
    camera_scan: Optional[dict] = None

    @property
    def dominant_layout(self) -> str:
        if not self.shots:
            return LayoutType.SCREEN
        totals: dict[str, int] = {}
        for s in self.shots:
            key = "screen_fill" if s.screen_fill and fills_screen_layout(s) else s.layout
            totals[key] = totals.get(key, 0) + (s.end_ms - s.start_ms)
        return max(totals, key=totals.get)

    @property
    def is_letterbox_only(self) -> bool:
        return all(s.layout == LayoutType.SCREEN and s.content_box is None and not s.screen_fill for s in self.shots)


# ------------------------------------------------------------------
# Pure helpers (unit tested)
# ------------------------------------------------------------------


def framing_faces(frame: FrameInfo) -> list[Box]:
    """Weak competing detections must not displace a confident real face.

    Keep the original observations in diagnostics. Unknown scores (older
    records / fixtures) retain the existing behavior.
    """
    reliable = any(score >= STRONG_FACE_SCORE for score in frame.scores)
    return [box for i, box in enumerate(frame.faces)
            if not reliable or i >= len(frame.scores) or frame.scores[i] >= COMPETING_FACE_SCORE]


def _content_edges(image):
    """(left, top, right, bottom, background) of a padded inset, in pixels."""
    h, w = image.shape[:2]
    if w < 80 or h < 60:
        return None
    small = cv2.GaussianBlur(image, (5, 5), 0).astype(np.float32)
    margin = max(2, round(w * .06))
    sides = np.concatenate((small[:, :margin].reshape(-1, 3), small[:, -margin:].reshape(-1, 3)))
    background = np.median(sides, axis=0)
    if np.quantile(np.max(np.abs(sides - background), axis=1), .9) > 16:
        return None
    mask = np.max(np.abs(small - background), axis=2) > 25
    columns = np.flatnonzero(mask.mean(axis=0) > .45)
    if not len(columns):
        return None
    left, right = int(columns[0]), int(columns[-1]) + 1
    if left < w * .08 or right > w * .92 or not .18 <= (right - left) / w <= .84:
        return None
    rows = np.flatnonzero(mask[:, left:right].mean(axis=1) > .55)
    if not len(rows):
        return None
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    if bottom - top < .5 * h:
        return None
    # Exclude shadows and rounded corners when assessing the vertical edges.
    middle = mask[top + int((bottom - top) * .08):bottom - int((bottom - top) * .08)]
    first = middle.argmax(axis=1)
    last = w - 1 - middle[:, ::-1].argmax(axis=1)
    straight = (np.abs(first - left) <= w * .015) & (np.abs(last + 1 - right) <= w * .015)
    if straight.mean() < .8 or mask[top:bottom, left:right].mean() < .7:
        return None
    return left, top, right, bottom, background


def _refine_edge(image, background, center: int, radius: int, vertical: bool,
                 span: tuple[int, int], threshold: float, last: bool) -> Optional[int]:
    """Full-resolution edge near a coarse estimate, with the coarse criterion."""
    length = image.shape[1] if vertical else image.shape[0]
    lo, hi = max(0, center - radius), min(length, center + radius + 1)
    pad = 2  # 5x5 blur support
    if vertical:
        strip = image[:, max(0, lo - pad):min(length, hi + pad)]
    else:
        strip = image[max(0, lo - pad):min(length, hi + pad), :]
    blurred = cv2.GaussianBlur(strip, (5, 5), 0).astype(np.float32)
    offset = lo - max(0, lo - pad)
    mask = np.max(np.abs(blurred - background), axis=2) > 25
    if vertical:
        share = mask[:, offset:offset + hi - lo].mean(axis=0)
    else:
        share = mask[offset:offset + hi - lo, span[0]:span[1]].mean(axis=1)
    hits = np.flatnonzero(share > threshold)
    if not len(hits):
        return None
    return lo + int(hits[-1]) + 1 if last else lo + int(hits[0])


def detect_content_box(image) -> Optional[Box]:
    """Conservative rectangular inset detection, including textured padding.

    Require matching, mostly uniform side margins and sustained straight
    content edges. A wall behind a person alone is not a rectangular inset.
    Work only on the already-decoded analysis image; no extra provider call.
    Detection runs on a CONTENT_BOX_WIDTH-wide copy (most frames are rejected
    there); a found inset's edges are then located at full resolution.
    """
    h, w = image.shape[:2]
    if w <= CONTENT_BOX_WIDTH * 1.5 or h < 60:
        edges = _content_edges(image)
        if edges is None:
            return None
        left, top, right, bottom, _ = edges
        return Box(left / w, top / h, (right - left) / w, (bottom - top) / h)
    small_h = max(60, round(h * CONTENT_BOX_WIDTH / w))
    coarse = _content_edges(cv2.resize(image, (CONTENT_BOX_WIDTH, small_h), interpolation=cv2.INTER_AREA))
    if coarse is None:
        return None
    left, top, right, bottom, background = coarse
    sx, sy = w / CONTENT_BOX_WIDTH, h / small_h
    rx, ry = math.ceil(sx) + 2, math.ceil(sy) + 2
    guess = [round(left * sx), round(top * sy), round(right * sx), round(bottom * sy)]
    left = _refine_edge(image, background, guess[0], rx, True, (0, h), .45, False)
    right = _refine_edge(image, background, guess[2] - 1, rx, True, (0, h), .45, True)
    left, right = guess[0] if left is None else left, guess[2] if right is None else right
    top = _refine_edge(image, background, guess[1], ry, False, (left, right), .55, False)
    bottom = _refine_edge(image, background, guess[3] - 1, ry, False, (left, right), .55, True)
    top, bottom = guess[1] if top is None else top, guess[3] if bottom is None else bottom
    if right - left < 8 or bottom - top < 8:
        return None
    return Box(left / w, top / h, (right - left) / w, (bottom - top) / h)


def same_content(a: Optional[Box], b: Optional[Box]) -> bool:
    if a is None or b is None:
        return a is b
    return max(abs(x - y) for x, y in zip(a.to_list(), b.to_list())) < .035


def content_boundaries(frames: list[FrameInfo]) -> list[int]:
    """Confirm an inset entering, leaving or moving over three observations."""
    if not frames:
        return []
    current = frames[0].content_box
    pending: list[FrameInfo] = []
    cuts = []
    for frame in frames[1:]:
        if same_content(frame.content_box, current):
            pending = []
            continue
        if pending and not same_content(frame.content_box, pending[0].content_box):
            pending = []
        pending.append(frame)
        if len(pending) >= LAYOUT_CHANGE_SAMPLES and frame.t_ms - pending[0].t_ms >= LAYOUT_CHANGE_MS - SAMPLE_JITTER_MS:
            cuts.append(pending[0].t_ms)
            current, pending = frame.content_box, []
    return cuts


def segment_shots(frames: list[FrameInfo], duration_ms: int) -> list[tuple[int, int]]:
    """Split the timeline into shots at histogram cuts; merge shots < MIN_SHOT_MS."""
    if not frames:
        return [(0, duration_ms)]
    cuts = [0]
    for prev, cur in zip(frames, frames[1:]):
        dist = cv2.compareHist(prev.hist, cur.hist, cv2.HISTCMP_BHATTACHARYYA)
        if dist > SHOT_CUT_THRESHOLD:
            # The cut happened somewhere between the two samples.
            cuts.append((prev.t_ms + cur.t_ms) // 2)
    cuts.append(duration_ms)

    shots = [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]
    merged: list[tuple[int, int]] = []
    for start, end in shots:
        if merged and (end - start < MIN_SHOT_MS or merged[-1][1] - merged[-1][0] < MIN_SHOT_MS):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def track_faces(frames: list[FrameInfo]) -> list[FaceTrack]:
    """Associate nearby faces, retaining an unambiguous subject through jumps.

    A sole visible face can move farther than the normal association radius
    between samples. Keep that track while its size/layout remain compatible;
    never use this shortcut after a multi-person frame or a long dropout.
    """
    tracks: list[FaceTrack] = []
    sole_track: Optional[int] = None
    for frame in frames:
        faces = framing_faces(frame)
        used: set[int] = set()
        for face in sorted(faces, key=lambda b: -b.area):
            best, best_dist = None, 0.12
            if len(faces) == 1 and sole_track is not None:
                last_t, last = tracks[sole_track].samples[-1]
                # A zoom cut can exceed the usual 2x size gate. Webcam moves
                # still need separate tracks for overlay-region classification.
                if (frame.t_ms - last_t <= MAX_TRACK_GAP_MS
                        and 0.25 < face.h / max(last.h, 1e-6) < 4.0
                        and not is_corner_overlay(face) and not is_corner_overlay(last)):
                    best = sole_track
            if best is None:
                for i, track in enumerate(tracks):
                    if i in used or frame.t_ms - track.samples[-1][0] > MAX_TRACK_GAP_MS:
                        continue
                    last = track.samples[-1][1]
                    dist = ((last.cx - face.cx) ** 2 + (last.cy - face.cy) ** 2) ** 0.5
                    size_ratio = face.h / max(last.h, 1e-6)
                    if dist < best_dist and 0.5 < size_ratio < 2.0:
                        best, best_dist = i, dist
            if best is None:
                tracks.append(FaceTrack(samples=[(frame.t_ms, face)]))
                used.add(len(tracks) - 1)
            else:
                tracks[best].samples.append((frame.t_ms, face))
                used.add(best)
        if len(faces) == 1:
            sole_track = next(iter(used))
        elif faces:
            sole_track = None
    return tracks


def is_corner_overlay(face: Box) -> bool:
    """Small face in a frame corner: almost always a webcam overlay."""
    if face.h > OVERLAY_MAX_FACE_HEIGHT:
        return False
    return (face.cx < 0.3 or face.cx > 0.7) and (face.cy < 0.4 or face.cy > 0.6)


# Webcam overlays also sit at the bottom centre or the middle of a side. A
# small face there could as well be a person in a wide shot, so it counts as a
# webcam only when a border around it is found (find_cam_border). The top band
# is left out: that is where a standing person's face is.
EDGE_BAND = 0.3
# A webcam face stays inside its overlay: across a shot its centre wanders
# less than this many face heights (a 16:9 webcam is about six face heights
# wide), and its size changes by less than this ratio (leaning in included).
# Faces in a game roam with the action.
OVERLAY_MAX_WANDER = 2.5
OVERLAY_MAX_SIZE_SPREAD = 2.5
# Between keyframes a live face changes by at least LIVE_FACE_MIN_CHANGE (mean
# gray level) somewhere; a face that never changes by STILL_FACE_MAX_CHANGE
# while the picture around it does is a still picture (a game's character
# portrait). In between, often a dim or soft webcam, nothing is decided.
LIVE_FACE_MIN_CHANGE = 2.5
STILL_FACE_MAX_CHANGE = 0.8
# Border search: a border's contrast must hold in most keyframes, and the
# picture just outside it must move clearly more than just inside it.
BORDER_MIN_CONTRAST = 12.0
BORDER_CONTRAST_OVER_MEDIAN = 2.5
BORDER_MOTION_RATIO = 1.8
BORDER_MOTION_GAIN = 4.0
MAX_BORDER_IMAGES = 8


def is_edge_overlay_position(face: Box) -> bool:
    """A small face along an edge but not in a corner: the bottom centre or the middle of a side."""
    if face.h > OVERLAY_MAX_FACE_HEIGHT or is_corner_overlay(face):
        return False
    return face.cx < EDGE_BAND or face.cx > 1 - EDGE_BAND or face.cy > 1 - EDGE_BAND


def face_wander(track: FaceTrack, src_w: int, src_h: int) -> float:
    """How far the face strays from its usual place, in face heights (80th percentile)."""
    boxes = [box for _, box in track.samples]
    if len(boxes) < 3:
        return 0.0
    mid = track.median_box()
    face_px = max(mid.h * src_h, 1e-6)
    wander = sorted(math.hypot((b.cx - mid.cx) * src_w, (b.cy - mid.cy) * src_h) / face_px for b in boxes)
    return wander[min(len(wander) - 1, int(len(wander) * 0.8))]


def is_stationary(track: FaceTrack, src_w: int, src_h: int) -> bool:
    """Whether a face keeps its place and size across the shot, as a webcam's does."""
    boxes = [box for _, box in track.samples]
    if len(boxes) < 3:
        return True
    heights = sorted(b.h for b in boxes)
    spread = heights[min(len(heights) - 1, int(len(heights) * 0.9))] / max(heights[int(len(heights) * 0.1)], 1e-6)
    return face_wander(track, src_w, src_h) <= OVERLAY_MAX_WANDER and spread <= OVERLAY_MAX_SIZE_SPREAD


# Where the person said their webcam sits in their videos (Settings > Framing >
# Webcam in your videos). "auto" decides from the video. Otherwise only a small
# face whose centre is in that zone can be the webcam, border or not, and small
# faces elsewhere near an edge belong to the game.
WEBCAM_LOCATIONS = ("auto", "top-left", "top-right", "bottom-left", "bottom-right", "bottom-center")


def in_webcam_location(box: Box, location: str) -> bool:
    """Whether a face's (or webcam box's) centre is where the person said the webcam is. "auto" accepts any.

    The zones are Auto's own: the corner bands of is_corner_overlay and the
    bottom band of is_edge_overlay_position, so a presenter or a person in a
    wide shot is never taken for a chosen-place webcam.
    """
    left, right, top, bottom = box.cx < 0.3, box.cx > 0.7, box.cy < 0.4, box.cy > 0.6
    zones = {
        "top-left": left and top, "top-right": right and top,
        "bottom-left": left and bottom, "bottom-right": right and bottom,
        "bottom-center": 0.3 <= box.cx <= 0.7 and box.cy > 1 - EDGE_BAND,
    }
    return zones.get(location, True)


# Ranking webcam candidates in Auto (webcam_score). A person on a webcam moves
# their head a little around one place over a shot (80th-percentile wander, in
# face heights); a game's drawn portrait holds pixel-still, and a face in the
# game drifts away (the second half of the shot centred elsewhere than the first).
PERSON_MIN_WANDER = 0.1
PERSON_MAX_WANDER = 1.0
PERSON_MAX_DRIFT = 0.5
# A face this still over at least PIXEL_STILL_SAMPLES samples is drawn, not a
# person: a game portrait measured 0.004 face heights, a live head 0.15, and
# 0.01 is a quarter of a pixel for a 25-pixel face. Video compression can still
# make such a face look "live" to face_is_live.
PIXEL_STILL_WANDER = 0.01
PIXEL_STILL_SAMPLES = 8
# Webcam faces are usually at least this tall; smaller ones are often HUD portraits.
WEBCAM_TYPICAL_FACE_HEIGHT = 0.06
# A border that stays put while the game moves is the strongest single sign.
BORDER_WEIGHT = 1.5
# Frames sampled across a clip to find where a webcam sits throughout it.
MAX_ANCHOR_FRAMES = 1200


def face_drift(track: FaceTrack, src_w: int, src_h: int) -> float:
    """How far the face's usual place moves from the first half of the shot to the second, in face heights."""
    boxes = [box for _, box in track.samples]
    if len(boxes) < 4:
        return 0.0
    half = len(boxes) // 2
    first, second = FaceTrack(samples=list(enumerate(boxes[:half]))).median_box(), FaceTrack(samples=list(enumerate(boxes[half:]))).median_box()
    return math.hypot((first.cx - second.cx) * src_w, (first.cy - second.cy) * src_h) / max(track.median_box().h * src_h, 1e-6)


def _same_place(a: Box, b: Box) -> bool:
    return abs(a.cx - b.cx) < 0.05 and abs(a.cy - b.cy) < 0.05 and 0.67 <= a.h / max(b.h, 1e-6) <= 1.5


def webcam_anchors(frames: list[FrameInfo]) -> list[tuple[Box, float]]:
    """Places near an edge where a small face shows up across the whole clip, each with
    the share of sampled frames it is seen in. A webcam is there all clip long; a game
    portrait comes and goes, and a face moving through the game spreads over many places."""
    sampled = evenly(frames, MAX_ANCHOR_FRAMES)
    places: list[list] = []  # [first box seen there, frames it was seen in]
    for frame in sampled:
        seen = set()
        for face in framing_faces(frame):
            if not MIN_FACE_HEIGHT <= face.h <= OVERLAY_MAX_FACE_HEIGHT:
                continue
            if not (is_corner_overlay(face) or is_edge_overlay_position(face)):
                continue
            index = next((i for i, (box, _) in enumerate(places) if _same_place(box, face)), None)
            if index is None:
                places.append([face, 0])
                index = len(places) - 1
            if index not in seen:
                seen.add(index)
                places[index][1] += 1
    return [(box, count / max(1, len(sampled))) for box, count in places]


def webcam_score(track: FaceTrack, shot_frames: int, src_w: int, src_h: int, border: bool,
                 live: Optional[bool], anchors: Optional[list[tuple[Box, float]]] = None) -> float:
    """How much a candidate face looks like the streamer's webcam rather than a face in the game:
    a border that stays put while the game moves, being there all clip long and all shot long,
    a head that moves a little around one place, a face that changes, and a usual webcam size."""
    face = track.median_box()
    clip_share = max((share for box, share in anchors or [] if _same_place(box, face)), default=0.0)
    shot_share = min(1.0, len(track.samples) / max(1, shot_frames))
    wander = face_wander(track, src_w, src_h)
    person = PERSON_MIN_WANDER <= wander <= PERSON_MAX_WANDER and face_drift(track, src_w, src_h) <= PERSON_MAX_DRIFT
    drawn = wander < PIXEL_STILL_WANDER and len(track.samples) >= PIXEL_STILL_SAMPLES
    return ((BORDER_WEIGHT if border else 0.0) + clip_share + 0.5 * shot_share
            + (1.0 if person else -1.0 if drawn else 0.0)
            + (0.25 if live else 0.0)
            + (0.25 if face.h >= WEBCAM_TYPICAL_FACE_HEIGHT else 0.0))


@dataclass
class OverlayEvidence:
    """What a shot's keyframes say about a face that may be a webcam."""

    live: Optional[bool] = None  # None: too few images to tell
    border: Optional[Box] = None  # the overlay's bounds, when found


def _gray(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def face_is_live(images: list, face: Box) -> Optional[bool]:
    """Whether the face changes between images (a person) or stays a still picture.

    None when there are too few images, nothing in the frame changes, or the
    change is too small to call either way.
    """
    if len(images) < 3:
        return None
    h, w = images[0].shape[:2]
    x0, y0 = max(0, int(face.x * w)), max(0, int(face.y * h))
    x1, y1 = min(w, math.ceil((face.x + face.w) * w)), min(h, math.ceil((face.y + face.h) * h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    grays = [_gray(image).astype(np.float32) for image in images]
    # A frozen picture (a paused stream) says nothing about the face.
    if max(float(np.mean(np.abs(a - b))) for a, b in zip(grays, grays[1:])) < LIVE_FACE_MIN_CHANGE:
        return None
    crops = [image[y0:y1, x0:x1] for image in grays]
    change = max(float(np.mean(np.abs(a - b))) for a, b in zip(crops, crops[1:]))
    if change >= LIVE_FACE_MIN_CHANGE:
        return True
    return False if change <= STILL_FACE_MAX_CHANGE else None


def sharp_scan_scale(src_w: int, src_h: int) -> float:
    """Scale of the sharp webcam scan: SHARP_SCAN_FACTOR times the analysis scale, at most the source."""
    return min(1.0, SHARP_SCAN_FACTOR * min(ANALYSIS_WIDTH / src_w, MAX_ANALYSIS_HEIGHT / src_h))


def boxes_agree(a: Box, b: Box) -> bool:
    """Two estimates of the same webcam: similar size and mostly overlapping."""
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.area + b.area - inter
    return 0.65 <= a.area / max(b.area, 1e-9) <= 1.4 and inter / max(union, 1e-9) >= 0.5


def find_cam_border(images: list, face: Box) -> Optional[Box]:
    """A webcam overlay's bounds from edges that stay put while the picture around them changes.

    From the face outward on each side, the border is the first line whose
    contrast holds in most images and beyond which the picture moves clearly
    more than inside (the game moves; a room behind the streamer does not).
    The motion step alone also finds an overlay drawn without a border. A
    side with no such line runs to the frame edge. Needs two found lines and a
    plausible webcam shape; otherwise None.
    """
    if len(images) < 3:
        return None
    gray = np.stack([_gray(image) for image in images]).astype(np.float32)
    _, h, w = gray.shape
    motion = gray.std(axis=0)
    fx0, fy0 = face.x * w, face.y * h
    fx1, fy1 = (face.x + face.w) * w, (face.y + face.h) * h
    bounds = {}
    found_sides: list[str] = []
    k_of = {}
    for side in ("left", "right", "top", "bottom"):
        vertical = side in ("left", "right")
        length = w if vertical else h
        # The border must run alongside the face if the face is inside the overlay.
        a, b = (int(fy0), math.ceil(fy1)) if vertical else (int(fx0), math.ceil(fx1))
        a, b = max(0, a), min(h if vertical else w, b)
        if b - a < 4:
            return None
        if vertical:
            band = gray[:, a:b, :]
            per_frame = np.median(np.abs(band[:, :, 2:] - band[:, :, :-2]), axis=1)
            profile = motion[a:b, :].mean(axis=0)
        else:
            band = gray[:, :, a:b]
            per_frame = np.median(np.abs(band[:, 2:, :] - band[:, :-2, :]), axis=2)
            profile = motion[:, a:b].mean(axis=1)
        # Contrast across position p (between p - 1 and p + 1) in most images.
        persistent = np.percentile(per_frame, 25, axis=0)
        k = max(3, round(length * 0.025))
        k_of[side] = k
        size = (fx1 - fx0) if vertical else (fy1 - fy0)
        outward = 1 if side in ("right", "bottom") else -1
        start = ((fx1 if vertical else fy1) + 0.3 * size) if outward > 0 else ((fx0 if vertical else fy0) - 0.3 * size)
        scan = [p for p in (range(math.ceil(start), length - 1) if outward > 0 else range(int(start), 0, -1))
                if 1 <= p <= length - 2]
        edge = length if outward > 0 else 0
        if scan:
            level = float(np.median([persistent[p - 1] for p in scan]))

            def step(p: int, width: int = k) -> tuple[float, float]:
                before, after = profile[max(0, p - width):p].mean(), profile[p:p + width].mean()
                return (before, after) if outward > 0 else (after, before)

            for p in scan:
                inside, outside = step(p)
                moving = outside >= max(inside * BORDER_MOTION_RATIO, inside + BORDER_MOTION_GAIN)
                sharp = persistent[p - 1] >= max(BORDER_MIN_CONTRAST, BORDER_CONTRAST_OVER_MEDIAN * level)
                strong_motion = outside >= max(inside * 3, inside + 10)
                if moving and (sharp or strong_motion):
                    # The motion window reaches the edge before p does: settle on
                    # the sharpest line or steepest motion step just beyond.
                    near = [q for q in range(p, p + outward * k, outward) if 1 <= q <= length - 2]
                    best = max(near, key=lambda q: persistent[q - 1] + 2 * (step(q, 3)[1] - step(q, 3)[0]))
                    # One pixel inside the line keeps border bleed out of the crop.
                    edge = best - 1 if outward > 0 else best + 1
                    found_sides.append(side)
                    break
        bounds[side] = edge
    left, right, top, bottom = bounds["left"], bounds["right"], bounds["top"], bounds["bottom"]
    if len(found_sides) < 2 or right - left < 8 or bottom - top < 8:
        return None
    if not _calm_overlay(motion, (left, top, right, bottom), (fx0, fy0, fx1, fy1), found_sides, k_of):
        return None
    box = Box(left / w, top / h, (right - left) / w, (bottom - top) / h).clamp()
    aspect = (box.w * w) / max(box.h * h, 1e-6)
    if not (box.contains(face.x, face.y) and box.contains(face.x + face.w, face.y + face.h)):
        return None
    if box.area > 0.45 or not 0.45 <= aspect <= 2.6 or not 0.1 <= face.h / max(box.h, 1e-6) <= 0.8:
        return None
    return box


def _calm_overlay(motion, box: tuple[int, int, int, int], face: tuple[float, float, float, float],
                  found_sides: list[str], k_of: dict[str, int]) -> bool:
    """Whether a found box behaves like an overlay: calmer inside (the face aside)
    than in the picture around it, and every found line separating the two along
    most of its length, not just beside the face."""
    h, w = motion.shape
    left, top, right, bottom = box
    inside = motion[top:bottom, left:right].astype(np.float64).copy()
    fx0, fy0, fx1, fy1 = face
    # The person moves; judge the room around them.
    pad_x, pad_y = (fx1 - fx0) * 0.5, (fy1 - fy0) * 0.5
    inside[max(0, int(fy0 - pad_y) - top):max(0, math.ceil(fy1 + pad_y) - top),
           max(0, int(fx0 - pad_x) - left):max(0, math.ceil(fx1 + pad_x) - left)] = np.nan
    if np.isnan(inside).all():
        # A small webcam is mostly face: judge all of it.
        inside = motion[top:bottom, left:right].astype(np.float64)
    calm = float(np.nanmean(inside))
    k = max(k_of.values())
    ring = motion[max(0, top - k):min(h, bottom + k), max(0, left - k):min(w, right + k)].astype(np.float64).copy()
    ring[top - max(0, top - k):top - max(0, top - k) + (bottom - top), left - max(0, left - k):left - max(0, left - k) + (right - left)] = np.nan
    if np.isnan(ring).all():
        return False
    busy = float(np.nanmean(ring))
    if busy < max(calm * BORDER_MOTION_RATIO, calm + BORDER_MOTION_GAIN):
        return False
    for side in found_sides:
        k = k_of[side]
        if side == "left":
            inner, outer = motion[top:bottom, left:left + k], motion[top:bottom, max(0, left - k):left]
            axis = 1
        elif side == "right":
            inner, outer = motion[top:bottom, max(0, right - k):right], motion[top:bottom, right:right + k]
            axis = 1
        elif side == "top":
            inner, outer = motion[top:top + k, left:right], motion[max(0, top - k):top, left:right]
            axis = 0
        else:
            inner, outer = motion[max(0, bottom - k):bottom, left:right], motion[bottom:bottom + k, left:right]
            axis = 0
        if inner.size == 0 or outer.size == 0:
            return False
        a, b = inner.mean(axis=axis), outer.mean(axis=axis)
        if float(np.mean((b >= a * 1.5) & (b >= a + 2))) < 0.6:
            return False
    return True


def frame_layout_evidence(frame: FrameInfo) -> Optional[str]:
    """Conservative temporal evidence, not a replacement for shot classification.

    A visible corner webcam wins over faces inside a game or shared screen.
    Small central faces and groups are ambiguous. A large central face with no
    corner webcam is positive evidence of a speaker, unlike a detector dropout.
    """
    faces = [f for f in framing_faces(frame) if f.h >= MIN_FACE_HEIGHT]
    if not faces:
        return LayoutType.SCREEN
    if any(is_corner_overlay(f) for f in faces):
        return LayoutType.SCREEN_CAM
    if len(faces) == 1:
        face = faces[0]
        in_corner = (face.cx < 0.3 or face.cx > 0.7) and (face.cy < 0.4 or face.cy > 0.6)
        if face.h > OVERLAY_MAX_FACE_HEIGHT * 1.3 and not in_corner:
            return LayoutType.TALKING_HEAD
    return None


def split_layout_segments(
    frames: list[FrameInfo], start_ms: int, end_ms: int,
    boundaries: Optional[list[dict]] = None,
) -> list[tuple[int, int]]:
    """Refine a color-based shot using sustained face-layout evidence.

    Backdate a confirmed transition to its first observation. Ambiguous frames
    and brief dropouts do not change the established layout; persistent missing
    faces get their own segment so vision can distinguish a hidden face from a
    removed webcam. Unlike histogram blips, confirmed layout changes are not
    absorbed by the 1.2-second shot-merging rule.
    """
    cuts = [start_ms]
    current: Optional[str] = None
    pending: Optional[str] = None
    pending_start = start_ms
    count = 0
    for frame in frames:
        evidence = frame_layout_evidence(frame)
        if evidence is None or evidence == current:
            pending, count = None, 0
            continue
        if evidence != pending:
            pending, pending_start, count = evidence, frame.t_ms, 0
        count += 1
        hold_ms = MIN_SHOT_MS if evidence == LayoutType.SCREEN else LAYOUT_CHANGE_MS
        if count >= LAYOUT_CHANGE_SAMPLES and frame.t_ms - pending_start >= hold_ms - SAMPLE_JITTER_MS:
            if current is not None and pending_start > cuts[-1]:
                cuts.append(pending_start)
                if boundaries is not None:
                    boundaries.append({"t_ms": pending_start, "kind": "layout", "accepted": True,
                                       "from_layout": current, "to_layout": evidence,
                                       "hold_ms": frame.t_ms - pending_start, "samples": count})
            current, pending, count = evidence, None, 0
    return list(zip(cuts, cuts[1:] + [end_ms]))


def segment_content_box(frames: list[FrameInfo]) -> Optional[Box]:
    """Stable median geometry, tolerating isolated misses and outliers."""
    content = [f.content_box for f in frames if f.content_box is not None]
    if len(content) >= max(LAYOUT_CHANGE_SAMPLES, len(frames) * .6):
        box = Box(*(median(getattr(b, axis) for b in content) for axis in ('x', 'y', 'w', 'h')))
        if sum(same_content(box, b) for b in content) >= .8 * len(content):
            return box.clamp()
    return None


def inset_crop_width(bounds: Optional[Box], src_w: int, src_h: int) -> float:
    """Width (share of the source) of the 9:16 fill crop inside `bounds`."""
    box = bounds or Box(0, 0, 1, 1)
    crop_h = min(box.h * src_h, box.w * src_w * 16 / 9)
    return min(1.0, crop_h * 9 / 16 / src_w)


def inset_speaker(
    tracks: list[FaceTrack], shot_frames: int, content: Box, src_w: int, src_h: int,
) -> Optional[FaceTrack]:
    """The one sustained on-camera face inside a landscape inset, if any.

    A pillarboxed 4:3 camera is an inset whose composition is a person; follow
    them like any talking head. Other insets (portrait video, screens, groups,
    brief faces) keep their composition.
    """
    present = [t for t in tracks
               if len(t.samples) >= max(1, MIN_TRACK_PRESENCE * shot_frames)
               and t.median_box().h >= MIN_FACE_HEIGHT]
    if (len(present) != 1 or content.h < INSET_FILL_MIN_HEIGHT
            or content.w * src_w < INSET_FILL_MIN_ASPECT * content.h * src_h):
        return None
    track = present[0]
    face = track.median_box()
    if (len(track.samples) < INSET_SPEAKER_PRESENCE * shot_frames or is_corner_overlay(face)
            or face.h <= OVERLAY_MAX_FACE_HEIGHT * 1.3 or not content.contains(face.cx, face.cy)):
        return None
    return track


def heuristic_layout(
    frames: list[FrameInfo], start_ms: int, end_ms: int, src_w: int, src_h: int,
    diagnostic: Optional[dict] = None,
    inspect: Optional[Callable[[Box], OverlayEvidence]] = None,
    location: str = "auto", anchors: Optional[list[tuple[Box, float]]] = None,
) -> ShotLayout:
    """Classify and track only the faces belonging to this layout segment.

    `inspect` looks at the segment's keyframes for a face that may be a webcam;
    `location` and `anchors` go to classify_shot.
    """
    content = segment_content_box(frames)
    tracks = track_faces(frames)
    bounds = None
    if content is not None:
        main_track = inset_speaker(tracks, len(frames), content, src_w, src_h)
        if main_track is None:
            return ShotLayout(start_ms, end_ms, LayoutType.SCREEN, content_box=content)
        shot, bounds = ShotLayout(0, 0, LayoutType.TALKING_HEAD, people=[main_track.median_box()],
                                  crop_bounds=content), content
    else:
        shot, main_track = classify_shot(tracks, len(frames), src_w, src_h, inspect, location, anchors)
    if diagnostic is not None:
        lookup = {(f.t_ms, id(box)): i for f in frames for i, box in enumerate(f.faces)}
        diagnostic["tracks"] = [{"id": i, "selected": track is main_track,
                                 "samples": [[t, lookup[(t, id(box))]] for t, box in track.samples]}
                                for i, track in enumerate(tracks)]
    shot.start_ms, shot.end_ms = start_ms, end_ms
    if main_track is not None:
        samples = [(t - start_ms, box) for t, box in main_track.samples]
        shot.focus_path = smooth_focus_path(samples, end_ms - start_ms, inset_crop_width(bounds, src_w, src_h))
    return shot


def _path_at(path: list[tuple[float, float, float]], t: float) -> tuple[float, float]:
    """Focus at t on a keyframed path, held flat past either end."""
    if t <= path[0][0]:
        return path[0][1], path[0][2]
    for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:]):
        if t < t1:
            f = (t - t0) / max(t1 - t0, 1e-3)
            return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
    return path[-1][1], path[-1][2]


def _same_person(a: ShotLayout, b: ShotLayout) -> bool:
    if not a.people or not b.people:
        return True
    p, q = a.people[0], b.people[0]
    return abs(p.cx - q.cx) < .25 and abs(p.cy - q.cy) < .25 and .5 <= p.h / max(q.h, 1e-6) <= 2


def fills_screen_layout(shot: ShotLayout) -> bool:
    """A screen with no webcam panel: the shots the screen-fill setting can crop."""
    return shot.layout in (LayoutType.SCREEN, LayoutType.SCREEN_CAM) and shot.cam_box is None and not shot.manual_crops


# A screen fills the 9:16 frame only when its largest crop needs no more
# enlargement than this. A 720p source fills (2.7x, as a talking head on 720p
# already does); 480p, or a small screen area, is shown whole instead.
MAX_FILL_UPSCALE = 3.0


def fill_crop_height(shot: ShotLayout, src_w: int, src_h: int) -> float:
    """Source pixels of height in the largest 9:16 crop of the shot's screen.

    A crop around a focus is never shorter than this or than what MAX_UPSCALE
    allows (layout_renderer.screen_crop), so this bounds the enlargement.
    """
    out_w, out_h = get_output_dimensions("9:16")
    bounds = shot.screen_box or shot.content_box or Box(0, 0, 1, 1)
    return min(bounds.h * src_h, bounds.w * src_w * out_h / out_w)


def shows_people(shot: ShotLayout, frames: list[FrameInfo], location: str = "auto") -> bool:
    """Whether people are on camera for much of the shot.

    Shots of three or more people, and insets without one clear speaker, are
    classified as screens too; a fill crop would cut people off, so they stay whole.
    With a chosen webcam `location`, small faces near an edge belong to the game.
    """
    def person(face: Box) -> bool:
        if face.h < MIN_FACE_HEIGHT or is_corner_overlay(face):
            return False
        return location == "auto" or not is_edge_overlay_position(face)
    inside = [f for f in frames if shot.start_ms <= f.t_ms < shot.end_ms]
    seen = sum(1 for f in inside if any(person(face) for face in f.faces))
    return bool(inside) and seen >= MIN_TRACK_PRESENCE * len(inside)


def is_face_dropout(shot: ShotLayout) -> bool:
    """A heuristic 'nobody here' shot that neither vision nor geometry confirmed."""
    return (shot.layout == LayoutType.SCREEN and shot.source == "heuristic" and not shot.people
            and shot.content_box is None and shot.screen_box is None and shot.cam_box is None)


def bridge_face_dropouts(
    shots: list[ShotLayout], scene_cuts: set, boundaries: Optional[list[dict]] = None,
) -> list[ShotLayout]:
    """Keep a talking head through a no-face gap that has no scene cut.

    Missing faces get their own segment so vision can tell a hidden face from
    a removed webcam. When vision is off, fails or is over budget, that gap
    must not letterbox a speaker who looked away: without a cut it is the same
    shot, so keep the neighbouring talking head and hold (or ease) its focus.
    """
    result: list[ShotLayout] = []
    i = 0
    while i < len(shots):
        gap = shots[i]
        before = result[-1] if result else None
        after = shots[i + 1] if i + 1 < len(shots) else None
        if not is_face_dropout(gap):
            result.append(gap)
            i += 1
            continue
        if before is not None and (before.layout != LayoutType.TALKING_HEAD or gap.start_ms in scene_cuts):
            before = None
        if after is not None and (after.layout != LayoutType.TALKING_HEAD or gap.end_ms in scene_cuts):
            after = None
        if before is not None and after is not None and (
                not _same_person(before, after) or before.crop_bounds != after.crop_bounds):
            after = None
        if before is None and after is None:
            result.append(gap)
            i += 1
            continue
        first, last = before or gap, after or gap
        points = [(first.start_ms + t, x, y) for part in (before, after) if part is not None
                  for t, x, y in (part.focus_path or [(0, *(
                      (part.people[0].cx, part.people[0].cy) if part.people else (.5, .5)))])]
        start = first.start_ms
        path = [(t - start, x, y) for t, x, y in points]
        if path[0][0] > 0:
            path.insert(0, (0, *_path_at(path, 0)))
        merged = ShotLayout(
            start, last.end_ms, LayoutType.TALKING_HEAD, source=(before or after).source,
            people=list((before or after).people[:1]), crop_bounds=(before or after).crop_bounds,
            focus_path=simplify_path(path),
        )
        if boundaries is not None:
            boundaries.append({"t_ms": gap.start_ms, "kind": "face_dropout_hold", "accepted": True,
                               "end_ms": gap.end_ms})
        if before is not None:
            result[-1] = merged
        else:
            result.append(merged)
        i += 2 if after is not None else 1
    return result


def estimate_cam_box(face: Box, src_w: int, src_h: int) -> Box:
    """Estimate the webcam overlay rectangle around a face.

    Used when the vision model is unavailable. A streaming webcam frames head
    and shoulders with the face spanning ~28% of its height (the old 50%
    guess gave a box a quarter of the overlay's area, and a 5x-upscaled
    crop). The box stays in the face's corner so it never reaches across the
    middle of the frame into screen content.
    """
    cam_h = min(face.h / CAM_FACE_SHARE, 0.45)
    cam_w = min(cam_h * (16 / 9) * (src_h / src_w), 0.4)
    box = Box(face.cx - cam_w / 2, face.cy - cam_h * 0.40, cam_w, cam_h).clamp()
    x0, x1 = (0.5, 1.0) if face.cx >= 0.5 else (0.0, 0.5)
    y0, y1 = (0.5, 1.0) if face.cy >= 0.5 else (0.0, 0.5)
    # Never cut into the face itself, whatever the corner bounds say.
    x0, x1 = min(x0, face.x), max(x1, face.x + face.w)
    y0, y1 = min(y0, face.y), max(y1, face.y + face.h)
    left, top = max(box.x, x0), max(box.y, y0)
    right, bottom = min(box.x + box.w, x1), min(box.y + box.h, y1)
    return Box(left, top, right - left, bottom - top)


def refine_cam_box(image, cam: Box, face: Optional[Box] = None) -> Optional[Box]:
    """Snap approximate webcam bounds to nearby, continuous image edges.

    Require two independently supported edges. Strong texture everywhere is
    not a boundary; the edge must stand out from neighboring lines. Leave
    uncertain geometry alone and never snap through the detected face.
    """
    h, w = image.shape[:2]
    pixels = cv2.GaussianBlur(image, (3, 3), 0).astype(np.float32)
    box = cam.clamp()
    edges = [box.x * w, box.y * h, (box.x + box.w) * w, (box.y + box.h) * h]
    found = 0
    for side, value in enumerate(edges.copy()):
        vertical = side in (0, 2)
        length = w if vertical else h
        span = box.w * w if vertical else box.h * h
        # Image borders already bound the camera; they are not independent
        # visual evidence of an overlay.
        if value <= 1 or value >= length - 1:
            continue
        radius = max(4, round(min(length * .035, span * .15)))
        lo, hi = (edges[1], edges[3]) if vertical else (edges[0], edges[2])
        inset = (hi - lo) * .12
        a, b = int(lo + inset), int(hi - inset)
        if b - a < 12:
            continue
        scores = []
        for pos in range(max(2, round(value) - radius), min(length - 2, round(value) + radius) + 1):
            if vertical:
                delta = np.max(np.abs(pixels[a:b, pos-2:pos].mean(axis=1) - pixels[a:b, pos:pos+2].mean(axis=1)), axis=1)
            else:
                delta = np.max(np.abs(pixels[pos-2:pos, a:b].mean(axis=0) - pixels[pos:pos+2, a:b].mean(axis=0)), axis=1)
            strength = float(np.median(delta))
            support = float((delta > 18).mean())
            scores.append((pos, strength, support))
        if not scores:
            continue
        baseline = median(score for _, score, _ in scores)
        candidates = [(pos, score * support - abs(pos - value)) for pos, score, support in scores
                      if support >= .65 and score >= max(24, baseline * 1.8 + 8)]
        if not candidates:
            continue
        pos = max(candidates, key=lambda item: item[1])[0]
        # One analysis pixel inside the edge removes compression/border bleed.
        edges[side] = pos + 1 if side < 2 else pos - 1
        found += 1
    left, top, right, bottom = edges
    if found < 2 or right - left < 8 or bottom - top < 8:
        return None
    refined = Box(left / w, top / h, (right - left) / w, (bottom - top) / h).clamp()
    if not .65 <= refined.area / max(box.area, 1e-6) <= 1.4:
        return None
    if face and not (refined.contains(face.x, face.y) and refined.contains(face.x + face.w, face.y + face.h)):
        return None
    return refined


def transfer_cam_box(ref_cam: Box, ref_face: Box, face: Box) -> Box:
    """Move/scale a known webcam box to follow its face.

    Streamers resize and reposition webcam overlays mid-shot; the face inside
    moves and scales with the overlay, so the overlay's geometry relative to
    the face carries over.
    """
    scale = face.h / max(ref_face.h, 1e-6)
    return Box(
        face.cx + (ref_cam.x - ref_face.cx) * scale,
        face.cy + (ref_cam.y - ref_face.cy) * scale,
        ref_cam.w * scale,
        ref_cam.h * scale,
    ).clamp()


def _overlay_moved(face: Box, anchor: Box) -> bool:
    ratio = face.h / max(anchor.h, 1e-6)
    return (
        abs(face.cx - anchor.cx) > 0.6 * anchor.w
        or abs(face.cy - anchor.cy) > 0.6 * anchor.h
        or not 0.8 <= ratio <= 1.25
    )


def split_overlay_segments(
    frames: list[FrameInfo],
    shot_start: int,
    shot_end: int,
    seed: Box,
    region: Optional[Box] = None,
    location: str = "auto",
    exclude: Optional[list[Box]] = None,
) -> list[tuple[int, int, Optional[Box]]]:
    """Split a screen+webcam shot wherever the webcam overlay moves or resizes.

    1. Observe the webcam face per sample: the small face nearest the last one.
       With a known webcam `region` (vision box), any face inside it counts
       too, however large: a close webcam is still the webcam. Faces outside a
       chosen `location`, and faces at the `exclude` places (other faces that sit
       still through the clip, such as a game portrait), never count. The
       follower only moves to a face far from the webcam once it is seen there
       twice in a row, so one missed detection does not hand the webcam to
       another face.
    2. Dropouts lasting MIN_SHOT_MS or more are "no webcam" stretches.
    3. A jump in position/size held for 2+ samples starts a new segment.
    Segments shorter than MIN_SHOT_MS are absorbed by their neighbour.

    Returns contiguous (start_ms, end_ms, median face or None) covering the shot.
    """
    max_face_h = OVERLAY_MAX_FACE_HEIGHT * 1.3
    observations: list[tuple[int, Optional[Box]]] = []
    anchor = seed
    far: Optional[Box] = None  # a face far from the anchor, seen once
    for frame in frames:
        candidates = [
            f for f in framing_faces(frame)
            if MIN_FACE_HEIGHT <= f.h
            and (f.h <= max_face_h or (region is not None and region.contains(f.cx, f.cy) and f.h <= region.h))
            and in_webcam_location(f, location)
            and not any(_same_place(place, f) for place in exclude or [])
        ]
        face = min(
            candidates,
            key=lambda f: (f.cx - anchor.cx) ** 2 + (f.cy - anchor.cy) ** 2,
            default=None,
        )
        observations.append((frame.t_ms, face))
        if face is None:
            continue
        if not _overlay_moved(face, anchor):
            anchor, far = face, None
        elif far is not None and not _overlay_moved(face, far):
            anchor, far = face, None  # seen twice in its new place: the webcam moved
        else:
            far = face

    # Long dropouts -> no-webcam stretches (index ranges).
    in_gap = [False] * len(observations)
    i = 0
    while i < len(observations):
        if observations[i][1] is not None:
            i += 1
            continue
        j = i
        while j < len(observations) and observations[j][1] is None:
            j += 1
        gap_end = observations[j][0] if j < len(observations) else shot_end
        if gap_end - observations[i][0] >= MIN_SHOT_MS:
            for k in range(i, j):
                in_gap[k] = True
        i = j

    segments: list[list] = []  # [start_ms, faces or None]
    current: Optional[list] = None
    reference: Optional[Box] = None
    pending: list[tuple[int, Box]] = []
    for idx, (t_ms, face) in enumerate(observations):
        if in_gap[idx]:
            if current is None or current[1] is not None:
                current = [t_ms, None]
                segments.append(current)
                pending = []
            continue
        if face is None:
            continue  # brief dropout (hand over face, looking away)
        if current is None or current[1] is None:
            current = [t_ms, [face]]
            segments.append(current)
            reference, pending = face, []
            continue
        if _overlay_moved(face, reference):
            pending.append((t_ms, face))
            if len(pending) >= 2:
                current = [pending[0][0], [f for _, f in pending]]
                segments.append(current)
                reference, pending = pending[-1][1], []
            continue
        pending = []
        current[1].append(face)
        reference = Box(
            reference.x * 0.7 + face.x * 0.3, reference.y * 0.7 + face.y * 0.3,
            reference.w * 0.7 + face.w * 0.3, reference.h * 0.7 + face.h * 0.3,
        )

    if not segments:
        return [(shot_start, shot_end, None)]

    result: list[tuple[int, int, Optional[Box]]] = []
    for i, (seg_start, faces) in enumerate(segments):
        seg_start = shot_start if i == 0 else seg_start
        seg_end = segments[i + 1][0] if i + 1 < len(segments) else shot_end
        face = None
        if faces:
            face = Box(
                median(f.x for f in faces), median(f.y for f in faces),
                median(f.w for f in faces), median(f.h for f in faces),
            )
        if result and seg_end - seg_start < MIN_SHOT_MS:
            prev = result[-1]
            result[-1] = (prev[0], seg_end, prev[2])
        else:
            result.append((seg_start, seg_end, face))
    if len(result) > 1 and result[0][1] - result[0][0] < MIN_SHOT_MS:
        first, second = result[0], result[1]
        result[:2] = [(first[0], second[1], second[2] or first[2])]
    return result


def screen_box_excluding_cam(cam: Box) -> Box:
    """Screen region: the full frame, trimmed away from a side-mounted webcam."""
    # Only trim when the cam spans most of a side; corner cams overlap the
    # screen, which is normal for streams.
    if cam.h > 0.8 and cam.x < 0.05:
        return Box(cam.x + cam.w, 0.0, 1.0 - cam.w, 1.0)
    if cam.h > 0.8 and cam.x + cam.w > 0.95:
        return Box(0.0, 0.0, cam.x, 1.0)
    return Box(0.0, 0.0, 1.0, 1.0)


def expanded_webcam_intervals(
    frames: list[FrameInfo], cam: Box, end_ms: int,
) -> list[tuple[int, int]]:
    """Find sustained presenter punch-ins relative to an observed compact webcam.

    A keyed presenter can grow without changing the background histogram, and
    remain on the right/left of the screen. Neither a central-face requirement
    nor searching only inside the old camera box can recognize that edit.
    Require a stable, sole face inside the known camera before using relative
    size/position evidence. This also works when vision sampled the punch-in:
    the compact reference can occur earlier or later in the shot.
    """
    runs: list[list[tuple[int, Box]]] = []
    run: list[tuple[int, Box]] = []
    for frame in frames:
        faces = framing_faces(frame)
        face = faces[0] if len(faces) == 1 else None
        compact = (face is not None and MIN_FACE_HEIGHT <= face.h <= .25
                   and cam.contains(face.cx, face.cy)
                   and (face.cx < .3 or face.cx > .7)
                   and (face.cy < .4 or face.cy > .6))
        if not compact or (run and _overlay_moved(face, run[0][1])):
            if len(run) >= LAYOUT_CHANGE_SAMPLES and run[-1][0] - run[0][0] >= LAYOUT_CHANGE_MS - SAMPLE_JITTER_MS:
                runs.append(run)
            run = []
        if compact:
            run.append((frame.t_ms, face))
    if len(run) >= LAYOUT_CHANGE_SAMPLES and run[-1][0] - run[0][0] >= LAYOUT_CHANGE_MS - SAMPLE_JITTER_MS:
        runs.append(run)
    if not runs:
        return []
    reference = max(runs, key=len)
    anchor = Box(*(median(getattr(f, axis) for _, f in reference) for axis in ('x', 'y', 'w', 'h')))

    intervals = []
    expanded = False
    pending_start, count, start = 0, 0, 0
    for frame in frames:
        faces = framing_faces(frame)
        face = faces[0] if len(faces) == 1 else None
        evidence = (face is not None and face.h >= max(.26, anchor.h * 1.45)
                    and face.w >= anchor.w * 1.3
                    and (abs(face.cx - anchor.cx) > .6 * anchor.w
                         or abs(face.cy - anchor.cy) > .6 * anchor.h))
        if evidence == expanded:
            count = 0
            continue
        if count == 0:
            pending_start = frame.t_ms
        count += 1
        if count >= LAYOUT_CHANGE_SAMPLES and frame.t_ms - pending_start >= LAYOUT_CHANGE_MS - SAMPLE_JITTER_MS:
            if evidence:
                start = pending_start
            else:
                intervals.append((start, pending_start))
            expanded, count = evidence, 0
    if expanded:
        intervals.append((start, end_ms))
    return intervals


def classify_shot(
    tracks: list[FaceTrack], shot_frames: int, src_w: int, src_h: int,
    inspect: Optional[Callable[[Box], OverlayEvidence]] = None,
    location: str = "auto", anchors: Optional[list[tuple[Box, float]]] = None,
) -> tuple[ShotLayout, Optional[FaceTrack]]:
    """Heuristic layout for one shot from its face tracks.

    A webcam face is small, near an edge and stationary. In a corner that is
    enough; in the middle of an edge, `inspect` must also find a border around
    it. A face `inspect` finds to be a still picture is never the webcam, and a
    small face that moves around near an edge is a face in the game: it is
    neither the webcam nor a person to follow.

    With a `location` other than "auto", only a small stationary face in that
    zone can be the webcam (no border needed), and small faces near an edge
    elsewhere are ignored. Several candidates are ranked by webcam_score, using
    `anchors` (webcam_anchors of the whole clip) for presence across the clip.

    Returns the layout (times filled in by the caller) and, for talking-head
    shots, the track the camera should follow.
    """
    present = [
        t for t in tracks
        if len(t.samples) >= max(1, MIN_TRACK_PRESENCE * shot_frames)
        and t.median_box().h >= MIN_FACE_HEIGHT
    ]
    present.sort(key=lambda t: -t.median_box().area)

    explicit = location != "auto"
    overlays: list[FaceTrack] = []
    borders: dict[int, Box] = {}
    lively: dict[int, Optional[bool]] = {}
    stills: list[FaceTrack] = []
    roaming: list[FaceTrack] = []
    elsewhere: list[FaceTrack] = []
    for track in present:
        face = track.median_box()
        corner, edge = is_corner_overlay(face), is_edge_overlay_position(face)
        if explicit and face.h <= OVERLAY_MAX_FACE_HEIGHT and (corner or edge or in_webcam_location(face, location)):
            if not in_webcam_location(face, location):
                elsewhere.append(track)
                continue
        elif not corner and not edge:
            continue
        if not is_stationary(track, src_w, src_h):
            roaming.append(track)
            continue
        evidence = inspect(face) if inspect is not None else None
        if evidence is not None and evidence.live is False:
            stills.append(track)
            continue
        if explicit or corner or (evidence is not None and evidence.live and evidence.border is not None):
            overlays.append(track)
            lively[id(track)] = evidence.live if evidence is not None else None
            if evidence is not None and evidence.border is not None:
                borders[id(track)] = evidence.border
    # Of several candidates the webcam is the most webcam-like (webcam_score),
    # not simply the steadiest, which favours a game's pixel-still portrait.
    overlays.sort(key=lambda t: (-webcam_score(t, shot_frames, src_w, src_h, id(t) in borders, lively.get(id(t)), anchors),
                                 face_wander(t, src_w, src_h)))
    # A still picture is never the webcam nor a person, nor is a small face
    # away from the webcam's chosen place. A face roaming near the edges is the
    # game's when a webcam sits beside it; without one it may be a person
    # moving about, followed as before.
    ignored = stills + elsewhere + (roaming if overlays else [])
    on_camera = [t for t in present if t not in overlays and t not in ignored]

    # A webcam that moves mid-shot splits into several short tracks; judge
    # its presence by all corner-overlay faces together.
    if not overlays:
        def in_place(face: Box) -> bool:
            if explicit:
                return face.h <= OVERLAY_MAX_FACE_HEIGHT and in_webcam_location(face, location)
            return is_corner_overlay(face)
        corner = [t for t in tracks if t.median_box().h >= MIN_FACE_HEIGHT and in_place(t.median_box())
                  and is_stationary(t, src_w, src_h) and t not in ignored]
        if sum(len(t.samples) for t in corner) >= MIN_TRACK_PRESENCE * shot_frames:
            overlays = sorted(corner, key=lambda t: -len(t.samples))

    if len(on_camera) >= 2:
        a, b = on_camera[0].median_box(), on_camera[1].median_box()
        similar = min(a.h, b.h) / max(a.h, b.h) >= 0.5
        separated = abs(a.cx - b.cx) >= 0.22
        if len(on_camera) == 2 and similar and separated:
            people = sorted([a, b], key=lambda box: box.cx)
            return ShotLayout(0, 0, LayoutType.TWO_SHOT, people=people), None
        if len(on_camera) > 2:
            return ShotLayout(0, 0, LayoutType.SCREEN), None
    if on_camera:
        main = on_camera[0]
        return ShotLayout(0, 0, LayoutType.TALKING_HEAD, people=[main.median_box()]), main
    if overlays:
        face = overlays[0].median_box()
        border = borders.get(id(overlays[0]))
        cam = border or estimate_cam_box(face, src_w, src_h)
        return ShotLayout(
            0, 0, LayoutType.SCREEN_CAM,
            cam_box=cam, cam_face=face, screen_box=screen_box_excluding_cam(cam),
            cam_box_refined=border is not None,
        ), None
    return ShotLayout(0, 0, LayoutType.SCREEN), None


def smooth_focus_path(
    samples: list[tuple[int, Box]],
    duration_ms: int,
    crop_w_frac: float,
    step_ms: int = int(1000 / ANALYSIS_FPS),
) -> list[tuple[int, float, float]]:
    """Turn noisy face centers into a calm virtual-camera path.

    Holds still while the subject stays inside a dead zone, then pans with a
    capped speed. Returns keyframes (t_ms, cx, cy) with times relative to the
    first sample's shot start.
    """
    if not samples:
        return []
    samples = sorted(samples, key=lambda s: s[0])

    # Resample onto a regular grid, holding the nearest detection.
    grid: list[tuple[int, float, float]] = []
    j = 0
    for t in range(0, max(math.ceil(duration_ms), 1), step_ms):
        while j + 1 < len(samples) and abs(samples[j + 1][0] - t) <= abs(samples[j][0] - t):
            j += 1
        grid.append((t, samples[j][1].cx, samples[j][1].cy))

    # Median filter to reject single-frame detector jumps.
    k = 2
    xs = [g[1] for g in grid]
    ys = [g[2] for g in grid]
    fx = [median(xs[max(0, i - k): i + k + 1]) for i in range(len(xs))]
    fy = [median(ys[max(0, i - k): i + k + 1]) for i in range(len(ys))]

    deadzone = CAMERA_DEADZONE * crop_w_frac
    max_step = CAMERA_MAX_SPEED * crop_w_frac * step_ms / 1000
    cam_x, cam_y = median(fx[: max(1, 1000 // step_ms)]), median(fy[: max(1, 1000 // step_ms)])
    path = [(0, cam_x, cam_y)]
    for (t, _, _), x, y in zip(grid[1:], fx[1:], fy[1:]):
        dx = x - cam_x
        if abs(dx) > deadzone:
            # Move toward the subject (not all the way: re-center with ease).
            move = max(-max_step, min(max_step, (dx - deadzone * (1 if dx > 0 else -1)) * 0.5))
            cam_x += move
        cam_y += (y - cam_y) * 0.2
        path.append((t, cam_x, cam_y))

    return simplify_path(path)


def simplify_path(path: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Fewest keyframes that reproduce a focus path.

    Preserve the shape in BOTH axes. Keeping only moving/still transitions
    erased reversals and vertical motion, sometimes leaving a static crop.
    Insert the point with the largest interpolation error until the path is
    accurate to 0.2% of the source or reaches FFmpeg's expression budget.
    """
    keep = sorted({0, len(path) - 1})
    while len(keep) < MAX_PATH_KEYFRAMES:
        worst_error, worst_index = 0.002, None
        for a, b in zip(keep, keep[1:]):
            t0, x0, y0 = path[a]
            t1, x1, y1 = path[b]
            for i in range(a + 1, b):
                t, x, y = path[i]
                fraction = (t - t0) / (t1 - t0)
                error = max(abs(x - (x0 + (x1 - x0) * fraction)),
                            abs(y - (y0 + (y1 - y0) * fraction)))
                if error > worst_error:
                    worst_error, worst_index = error, i
        if worst_index is None:
            break
        keep.append(worst_index)
        keep.sort()
    return [path[i] for i in keep]


def apply_style(shot: ShotLayout, style: str, src_w: int, src_h: int) -> ShotLayout:
    """Adjust a detected layout to the user's chosen framing style."""
    shot.detected_layout = shot.detected_layout or shot.layout
    if shot.content_box is not None or shot.crop_bounds is not None:
        # Insets keep their own picture: fill crops inside the inset, fit shows
        # the whole inset. Automatic framing keeps the detected choice.
        if style == LayoutStyle.FILL and shot.content_box is not None:
            bounds = shot.content_box
            focus = shot.people[0] if shot.people else bounds
            shot.layout, shot.source = LayoutType.TALKING_HEAD, "style"
            shot.content_box, shot.crop_bounds = None, bounds
            shot.focus_path = shot.focus_path or [(0, focus.cx, focus.cy)]
        elif style == LayoutStyle.FIT and shot.crop_bounds is not None:
            shot.layout, shot.source = LayoutType.SCREEN, "style"
            shot.content_box, shot.crop_bounds, shot.focus_path = shot.crop_bounds, None, []
        return shot
    if style == LayoutStyle.FIT:
        shot.layout = LayoutType.SCREEN
        shot.source = "style"
    elif style == LayoutStyle.FILL:
        if shot.layout in (LayoutType.TWO_SHOT, LayoutType.SCREEN_CAM, LayoutType.SCREEN):
            focus = shot.people[0] if shot.people else None
            if shot.layout == LayoutType.SCREEN_CAM and shot.cam_face:
                focus = None  # a tiny webcam blown up to full frame looks bad
            shot.layout = LayoutType.TALKING_HEAD
            shot.source = "style"
            if focus and not shot.focus_path:
                shot.focus_path = [(0, focus.cx, focus.cy)]
            elif not shot.focus_path:
                shot.focus_path = [(0, 0.5, 0.5)]
    return shot


def evenly(items: list, count: int) -> list:
    """`count` items spread evenly over the list (its first and last included)."""
    if count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def retain_keyframes(
    regular: list[tuple[float, bytes]], cuts: list[tuple[float, bytes]],
    limit: int = MAX_RETAINED_KEYFRAMES, max_bytes: int = MAX_KEYFRAME_BYTES,
) -> list[tuple[float, bytes]]:
    """At most `limit` keyframes spread over the WHOLE window.

    Images at camera changes are kept ahead of regular samples (a short shot
    may have no other image); each kind is thinned evenly across time, never
    by dropping the end of a long window.
    """
    cut_quota = min(len(cuts), max(limit // 2, limit - len(regular)))
    kept = sorted(evenly(cuts, cut_quota) + evenly(regular, limit - cut_quota), key=lambda k: k[0])
    while len(kept) > 1 and sum(len(image) for _, image in kept) > max_bytes:
        kept = evenly(kept, len(kept) // 2)
    return kept


class _EvenSample:
    """Bounded, evenly spaced subset of a stream of unknown length."""

    def __init__(self, capacity: int):
        self.capacity, self.stride, self.seen = capacity, 1, 0
        self._items: list = []

    def add(self, make):
        if self.seen % self.stride == 0:
            self._items.append(make())
            if len(self._items) >= self.capacity:
                self._items, self.stride = self._items[::2], self.stride * 2
        self.seen += 1

    def items(self) -> list:
        return list(self._items)


def _bgr(image):
    """BGR copy of a planar I420 frame (height * 3/2 rows); BGR passes through."""
    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420) if image.ndim == 2 else image


def _jpeg(image) -> bytes:
    ok, encoded = cv2.imencode(".jpg", _bgr(image), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise MediaProcessError("Layout keyframe could not be encoded")
    return encoded.tobytes()


class _Detections:
    """Face detection for decoded frames on the shared pool, bounded in flight.

    Each worker thread keeps its own detector. Results come back in time
    order; leaving the block early cancels queued work.
    """

    def __init__(self, analyzer: "LayoutAnalyzer", width: int, height: int):
        self.analyzer, self.width, self.height = analyzer, width, height
        self._futures: list = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for future in self._futures:
            future.cancel()
        return False

    def _detect(self, image, t_ms):
        if image.ndim == 2:
            image = _bgr(image)
        detector = self.analyzer._get_detector(self.width, self.height)
        return self.analyzer._frame_info(image, t_ms, detector, self.width, self.height)

    def submit(self, image, t_ms):
        """Queue a BGR image, or a planar I420 one (converted on the worker)."""
        waiting = [f for f in self._futures[-MAX_PENDING_DETECTIONS:] if not f.done()]
        if len(waiting) >= MAX_PENDING_DETECTIONS:
            waiting[0].result()
        self._futures.append(detection_pool().submit(self._detect, image, t_ms))

    def pending(self) -> bool:
        return any(not f.done() for f in self._futures)

    def results(self) -> list[FrameInfo]:
        frames = [future.result() for future in self._futures]
        self._futures = []
        return sorted(frames, key=lambda f: f.t_ms)


# ------------------------------------------------------------------
# Vision refinement
# ------------------------------------------------------------------

VISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "layout": {"type": "string", "enum": list(LayoutType.ALL)},
        "cam_box": {
            "type": "array", "items": {"type": "integer"},
            "description": "Webcam overlay [ymin, xmin, ymax, xmax] 0-1000, or [] if none.",
        },
        "screen_box": {
            "type": "array", "items": {"type": "integer"},
            "description": "Screen/app content region [ymin, xmin, ymax, xmax] 0-1000, or [] if none.",
        },
        "screen_focus": {
            "type": "array", "items": {"type": "integer"},
            "description": "Most important area inside the screen [ymin, xmin, ymax, xmax] 0-1000, or [] if none.",
        },
        "people": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
            "description": "Head-and-shoulders box [ymin, xmin, ymax, xmax] 0-1000 for each on-camera person (not inside the webcam overlay).",
        },
    },
    "required": ["layout", "cam_box", "screen_box", "screen_focus", "people"],
    "additionalProperties": False,
}

# Vision failures that hold for the whole run, and how many rejected frames in
# a row turn vision off for the run. A model too slow for one frame, or a rate
# limit still in force after the client waited, would hold up every later frame.
VISION_FATAL_REASONS = ("auth", "model_not_found", "endpoint_not_found", "model_missing", "url_missing",
                        "timeout", "rate_limit", "quota")
VISION_REJECTIONS_BEFORE_OFF = 2
# Longest wait for one frame check from a custom endpoint, whatever the planner may wait.
VISION_TIMEOUT_LIMIT_SECONDS = 300.0

VISION_PROMPT = """You are framing a frame from a video for a 9:16 vertical short.

Classify the frame's layout:
- "talking_head": one person on camera fills a meaningful part of the frame (no screen recording).
- "two_shot": two people on camera, e.g. a podcast wide shot.
- "screen_cam": a screen recording, slides, browser, code or gameplay WITH a webcam overlay of the presenter.
- "screen": screen content, graphics or b-roll with NO webcam overlay, or a crowd/group where no single framing works.

Return boxes as [ymin, xmin, ymax, xmax] integers from 0 to 1000 relative to the full frame:
- cam_box: the ENTIRE webcam overlay rectangle (its visible border/edges, including background around the person), not just the face. [] if there is no webcam overlay.
- screen_box: the region holding the screen/app content, excluding black bars and the webcam overlay if it sits outside the screen. [] if there is no screen content.
- screen_focus: inside screen_box, the area a viewer should see when the screen is cropped for a phone: the active editor/document pane, chat window, chart, or game view. Leave out sidebars, toolbars and empty space. [] if the whole screen matters equally.
- people: one head-and-shoulders box per on-camera person, left to right. Exclude people inside the webcam overlay and people shown inside screen content.

Detected faces (normalized x, y, w, h, may be incomplete): {faces}"""


def _box_from_1000(values: list[int]) -> Optional[Box]:
    if not isinstance(values, list) or len(values) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (min(1000, max(0, int(v))) / 1000 for v in values)
    except (TypeError, ValueError):
        return None
    if xmax - xmin < 0.02 or ymax - ymin < 0.02:
        return None
    return Box(xmin, ymin, xmax - xmin, ymax - ymin)


def face_from_person_box(person: Box) -> Box:
    """Approximate the face inside a head-and-shoulders box.

    The renderer sizes crops from face boxes, so vision-model person boxes are
    converted to the same convention.
    """
    face_h = person.h * 0.45
    face_w = person.w * 0.5
    return Box(person.cx - face_w / 2, person.y + person.h * 0.08, face_w, face_h).clamp()


def merge_vision_result(heuristic: ShotLayout, result: dict, src_w: int, src_h: int,
                        location: str = "auto") -> ShotLayout:
    """Combine the vision model's layout with locally detected faces.

    A webcam the model places away from the person's chosen `location` is not
    taken: the shot keeps the local webcam if it has one, otherwise it is a screen.
    """
    layout = result.get("layout")
    if layout not in LayoutType.ALL:
        return heuristic

    cam = _box_from_1000(result.get("cam_box", []))
    screen = _box_from_1000(result.get("screen_box", []))
    people = [face_from_person_box(b) for b in (_box_from_1000(p) for p in result.get("people", [])) if b]
    people.sort(key=lambda b: b.cx)

    merged = ShotLayout(0, 0, layout, source="vision")
    if layout == LayoutType.SCREEN_CAM:
        if cam is None or not (0.01 <= cam.area <= 0.5) or not in_webcam_location(cam, location):
            cam = heuristic.cam_box
        if cam is None:
            merged.layout = LayoutType.SCREEN
            merged.screen_box = screen
            merged.screen_focus = _box_from_1000(result.get("screen_focus", []))
            return merged
        merged.cam_box = cam
        merged.screen_box = screen or screen_box_excluding_cam(cam)
        merged.screen_focus = _box_from_1000(result.get("screen_focus", []))
        # Prefer the locally detected face inside the cam for centering.
        face = heuristic.cam_face
        if face is None or not cam.contains(face.cx, face.cy):
            face = next((p for p in heuristic.people if cam.contains(p.cx, p.cy)), None)
        merged.cam_face = face
    elif layout == LayoutType.TWO_SHOT:
        local = heuristic.people if heuristic.layout == LayoutType.TWO_SHOT else []
        merged.people = local if len(local) == 2 else people[:2]
        if len(merged.people) < 2:
            merged.layout = LayoutType.TALKING_HEAD if merged.people else LayoutType.SCREEN
    elif layout == LayoutType.TALKING_HEAD:
        merged.people = heuristic.people[:1] if heuristic.people else people[:1]
        merged.focus_path = heuristic.focus_path
    elif layout == LayoutType.SCREEN:
        merged.screen_box = screen
        merged.screen_focus = _box_from_1000(result.get("screen_focus", []))
    return merged


# ------------------------------------------------------------------
# Service
# ------------------------------------------------------------------


class LayoutAnalyzer:
    """Analyzes a clip's shots and picks a 9:16 framing per shot."""

    def __init__(self):
        self.settings = get_settings()
        # One YuNet detector per executor thread: clips render concurrently,
        # and a shared detector's input size/buffers race (OpenCV 5 asserts).
        self._local = threading.local()
        self._http_client: Optional[httpx.AsyncClient] = None
        # Reuse vision answers across clips of the same video: (hist, face signature, result)
        self._vision_cache: list[tuple[Any, tuple, dict, dict]] = []

    @property
    def available(self) -> bool:
        return cv2 is not None and os.path.isfile(self._model_path())

    def _model_path(self) -> str:
        return os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "models",
            "face_detection_yunet_2023mar.onnx",
        ))

    async def analyze(
        self,
        video_path: str,
        start_ms: int,
        duration_ms: int,
        src_w: int,
        src_h: int,
        style: str = LayoutStyle.AUTO,
        vision: bool = True,
        capture: bool = False,
        progress=None,
        precise: Optional[bool] = None,
    ) -> Optional[ClipLayoutPlan]:
        """Plan the framing for the render window [start_ms, start_ms + duration_ms).

        `vision=False` skips the paid vision model: heuristics only, used when
        the plan only informs pacing (Classic style, 16:9 output).
        `precise` (default: same as `vision`) scans every frame for camera
        changes in the same decode as face sampling, to place cuts on exact
        frames. Pacing-only plans do not need it.
        """
        if style == LayoutStyle.FIT:
            return ClipLayoutPlan(
                shots=[ShotLayout(0, duration_ms, LayoutType.SCREEN, source="style")],
                source_width=src_w, source_height=src_h,
            )
        if not self.available:
            logger.warning("Layout analysis unavailable (OpenCV or face model missing); using letterbox")
            return None

        loop = asyncio.get_running_loop()
        scan = decoded = None
        if vision if precise is None else precise:
            # One decode samples faces, scans every frame for camera changes
            # and examines the frames around them. If it cannot run, sample
            # faces exactly as before.
            try:
                decoded = await loop.run_in_executor(
                    None, self._precise_frames, video_path, start_ms, duration_ms, src_w, src_h, progress)
            except (OSError, ValueError, MediaProcessError):
                logger.warning("Detailed camera analysis unavailable; sampling faces only")
        if decoded is None:
            if progress:
                progress('Sampling faces', None)
            frames, keyframes = await loop.run_in_executor(
                None, self._decode_and_detect, video_path, start_ms, duration_ms, src_w, src_h,
            )
        else:
            scan, frames, keyframes = decoded
        if not frames:
            logger.warning("Layout analysis decoded no frames; using letterbox")
            return None

        shots: list[ShotLayout] = []
        vision_cost = 0.0
        vision_decisions = 0
        sharp_scans = 0
        location = self._webcam_location()
        anchors = webcam_anchors(frames)
        trace = {"samples": [], "boundaries": [], "decisions": []} if capture else None
        color_segments = segment_shots(frames, duration_ms)
        inset_cuts = content_boundaries(frames)
        # Padding boundaries carry geometric evidence even when the color-cut
        # debouncer swallowed a short shot. Prefer their observed timestamps
        # to nearby histogram midpoints to avoid tiny duplicate segments.
        cuts = {0, duration_ms, *inset_cuts}
        cuts.update(a for a, _ in color_segments[1:] if all(abs(a - t) > 250 for t in inset_cuts))
        cuts = sorted(cuts)
        color_segments = list(zip(cuts, cuts[1:]))
        if trace is not None:
            accepted_cuts = {a for a, _ in color_segments[1:]}
            for i, frame in enumerate(frames):
                distance = float(cv2.compareHist(frames[i - 1].hist, frame.hist, cv2.HISTCMP_BHATTACHARYYA)) if i else None
                trace["samples"].append({"t_ms": frame.t_ms,
                    "faces": [{"box": box.to_list(), "score": frame.scores[j] if j < len(frame.scores) else None}
                              for j, box in enumerate(frame.faces)],
                    "content_box": frame.content_box.to_list() if frame.content_box else None,
                    "evidence": frame_layout_evidence(frame), "histogram_distance": distance})
                if distance is not None and distance > SHOT_CUT_THRESHOLD:
                    t = (frames[i - 1].t_ms + frame.t_ms) // 2
                    trace["boundaries"].append({"t_ms": t, "kind": "scene", "accepted": t in accepted_cuts,
                                                "distance": distance})
            trace["boundaries"].extend({"t_ms": t, "kind": "content", "accepted": True} for t in inset_cuts)

        if scan:
            from .layout_precision import confirmed_cuts, align_boundaries
            markers = [{**m, 'at_ms': m['at_ms'] - start_ms} for m in scan['markers']]
            exact = confirmed_cuts(markers, frames, duration_ms, frame_layout_evidence)
            cuts = sorted(set(align_boundaries(cuts, markers, duration_ms) + exact))
            color_segments = list(zip(cuts, cuts[1:]))
            if trace is not None:
                trace['boundaries'].extend({'t_ms': t, 'kind': 'precise_scene', 'accepted': True} for t in cuts[1:-1])
        # Scene cuts end a shot; face-layout splits inside one scene do not.
        scene_cuts = set(cuts)
        segments = []
        for start, end in color_segments:
            local_frames = [f for f in frames if start <= f.t_ms < end]
            # Faces coming and going inside a stable inset do not change the
            # composition of that inset.
            inset = segment_content_box(local_frames)
            segments.extend([(start, end)] if inset else split_layout_segments(
                local_frames, start, end, trace["boundaries"] if trace is not None else None))

        if scan:
            boundaries = align_boundaries([a for a, _ in segments], markers, duration_ms)
            segments = list(zip(boundaries, boundaries[1:]))
        if progress:
            progress('Checking shot layouts', 0)
        for shot_index, (shot_start, shot_end) in enumerate(segments):
            if progress:
                progress('Checking shot layouts', round(100 * shot_index / max(1, len(segments))))
            shot_frames = []
            for frame in frames:
                if shot_start <= frame.t_ms < shot_end and (not shot_frames or frame.t_ms - shot_frames[-1].t_ms >= 200):
                    shot_frames.append(frame)
            decision = {"start_ms": shot_start, "end_ms": shot_end, "tracks": [],
                        "vision": {"status": "disabled" if not vision or not self._vision_enabled() else "unavailable"}} if capture else None
            shot = heuristic_layout(shot_frames, shot_start, shot_end, src_w, src_h, decision,
                                    inspect=self._keyframe_inspector(keyframes, shot_start, shot_end),
                                    location=location, anchors=anchors)
            if decision is not None:
                decision["heuristic"] = shot.summary()
            reference_ms = (shot_start + shot_end) // 2

            inset = shot.content_box is not None or shot.crop_bounds is not None
            if inset and decision is not None:
                decision["vision"] = {"status": "content_region"}
            if vision and self._vision_enabled() and not inset:
                keyframe = self._pick_keyframe(keyframes, reference_ms, shot_start, shot_end)
                if decision is not None:
                    decision["vision"] = {"status": "no_image"}
                if keyframe is not None and shot_frames:
                    reference_ms, image = keyframe
                    cache_only = vision_decisions >= 12
                    vision_decisions += 1
                    # Face hints and the cache signature must describe the
                    # image being sent, not faces pooled from other moments.
                    reference_frame = min(shot_frames, key=lambda f: abs(f.t_ms - reference_ms))
                    if decision is not None:
                        decision["vision"] = {"status": "failed", "t_ms": reference_ms,
                                              "source_ms": start_ms + reference_ms}
                        result, cost = await self._vision_classify(image, [reference_frame], shot, decision["vision"], **({"cache_only": True} if cache_only else {}))
                    else:
                        result, cost = await self._vision_classify(image, [reference_frame], shot, **({"cache_only": True} if cache_only else {}))
                    vision_cost += cost
                    if result:
                        refined = merge_vision_result(shot, result, src_w, src_h, location)
                        if decision is not None:
                            decision["vision"]["validated"] = refined.summary()
                        if refined.layout == LayoutType.TALKING_HEAD and not refined.focus_path:
                            focus = refined.people[0] if refined.people else None
                            refined.focus_path = [(0, focus.cx, focus.cy)] if focus else [(0, 0.5, 0.5)]
                        shot = refined

            shot.start_ms, shot.end_ms = shot_start, shot_end
            sharp = None
            analysis_scale = min(ANALYSIS_WIDTH / src_w, MAX_ANALYSIS_HEIGHT / src_h)
            if (vision and shot.layout == LayoutType.SCREEN and shot.source == "heuristic" and not inset
                    and sharp_scan_scale(src_w, src_h) >= SHARP_SCAN_MIN_GAIN * analysis_scale
                    and not any(f.h >= MIN_FACE_HEIGHT and (location == "auto" or f.h > OVERLAY_MAX_FACE_HEIGHT
                                                            or in_webcam_location(f, location))
                                for frame in shot_frames for f in frame.faces)
                    and shot_end - shot_start >= SHARP_SCAN_MIN_SHOT_MS and sharp_scans < MAX_SHARP_SCANS):
                sharp_scans += 1
                sharp = await loop.run_in_executor(
                    None, self._sharp_overlay, video_path, start_ms, shot_start, shot_end, src_w, src_h, location)
                if decision is not None:
                    decision["sharp_scan"] = {"found": sharp is not None}
                if sharp is not None and not self._overlay_holds(keyframes, shot_start, shot_end, sharp[1] or estimate_cam_box(sharp[0], src_w, src_h), sharp[0]):
                    # Sampled from the middle of the shot only: it must sit in a
                    # calm box over the whole shot, or it is not kept.
                    sharp = None
                    if decision is not None:
                        decision["sharp_scan"] = {"found": False, "dropped": "not held across the shot"}
                if sharp is not None:
                    face, border = sharp
                    cam = border or estimate_cam_box(face, src_w, src_h)
                    shot = ShotLayout(shot_start, shot_end, LayoutType.SCREEN_CAM, source=shot.source,
                                      cam_box=cam, cam_face=face, screen_box=screen_box_excluding_cam(cam),
                                      cam_box_refined=border is not None)
            if shot.layout == LayoutType.SCREEN_CAM and sharp is None:
                sub_shots = self._follow_webcam(shot, shot_frames, src_w, src_h, reference_ms, location)
                self._refine_webcam_regions(sub_shots, keyframes, shot.cam_box)
            else:
                # A webcam found only in sharp frames is invisible to the
                # analysis-size samples that following relies on: keep it whole.
                sub_shots = [shot]
            shots.extend(sub_shots)
            if trace is not None:
                for before, after in zip(sub_shots, sub_shots[1:]):
                    if before.layout != after.layout:
                        trace["boundaries"].append({"t_ms": after.start_ms, "kind": "layout", "accepted": True,
                                                   "from_layout": before.layout, "to_layout": after.layout})
                trace["decisions"].append(decision)

        shots = bridge_face_dropouts(shots, scene_cuts, trace["boundaries"] if trace is not None else None)
        shots = [apply_style(shot, style, src_w, src_h) for shot in shots]
        self._apply_framing_settings(shots, style, src_w, src_h, frames)
        shots = self._merge_adjacent(shots)
        plan = ClipLayoutPlan(
            shots=shots, source_width=src_w, source_height=src_h, vision_cost_usd=vision_cost,
            face_samples=[(f.t_ms, f.faces) for f in frames],
            trace=trace, camera_scan=scan,
        )
        logger.info(
            "Layout plan: " + ", ".join(
                f"{s.layout}[{s.start_ms / 1000:.1f}-{s.end_ms / 1000:.1f}s,{s.source}]" for s in shots
            ) + (f" (vision ${vision_cost:.4f})" if vision_cost else "")
        )
        return plan

    @staticmethod
    def _keyframe_images(keyframes: list[tuple[int, bytes]], start_ms: int, end_ms: int) -> list:
        """Up to MAX_BORDER_IMAGES decoded keyframes spread over [start_ms, end_ms)."""
        inside = evenly([k for k in keyframes if start_ms <= k[0] < end_ms], MAX_BORDER_IMAGES)
        images = [cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) for _, data in inside]
        return [image for image in images if image is not None]

    def _keyframe_inspector(self, keyframes: list[tuple[int, bytes]], start_ms: int, end_ms: int):
        """classify_shot's look at a candidate webcam face, from this shot's keyframes."""
        images: list = []

        def inspect(face: Box) -> OverlayEvidence:
            if not images:
                images.extend(self._keyframe_images(keyframes, start_ms, end_ms))
            return OverlayEvidence(live=face_is_live(images, face), border=find_cam_border(images, face))
        return inspect

    @staticmethod
    def _overlay_holds(keyframes: list[tuple[int, bytes]], start_ms: int, end_ms: int, cam: Box, face: Box) -> bool:
        """Whether a webcam box stays a calm overlay in the shot's keyframes (too few to tell: yes)."""
        images = LayoutAnalyzer._keyframe_images(keyframes, start_ms, end_ms)
        if len(images) < 3:
            return True
        motion = np.stack([_gray(image) for image in images]).astype(np.float32).std(axis=0)
        if float(motion.mean()) < 1.0:
            return True  # a frozen picture says nothing either way
        h, w = motion.shape
        box = (max(0, int(cam.x * w)), max(0, int(cam.y * h)),
               min(w, math.ceil((cam.x + cam.w) * w)), min(h, math.ceil((cam.y + cam.h) * h)))
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            return True
        k = max(3, round(min(w, h) * 0.025))
        return _calm_overlay(motion, box, (face.x * w, face.y * h, (face.x + face.w) * w, (face.y + face.h) * h), [], {"ring": k})

    def _sharp_overlay(self, video_path: str, window_start_ms: int, start_ms: int, end_ms: int,
                       src_w: int, src_h: int, location: str = "auto") -> Optional[tuple[Box, Optional[Box]]]:
        """A webcam too small for the analysis-size detector, from a few sharper frames.

        With a `location` other than "auto", only a face in that zone counts, border or not.

        Returns (face, border or None), or None. Never raises: a missed webcam
        only costs framing quality.
        """
        try:
            scale = sharp_scan_scale(src_w, src_h)
            width, height = max(2, round(src_w * scale / 2) * 2), max(2, round(src_h * scale / 2) * 2)
            # One decode of at most SHARP_SCAN_SPAN_MS around the shot's middle,
            # keeping SHARP_SCAN_SAMPLES evenly spaced frames (separate seeks each
            # decode from the previous keyframe, which is slower).
            span_ms = min(end_ms - start_ms, SHARP_SCAN_SPAN_MS)
            begin_ms = window_start_ms + (start_ms + end_ms - span_ms) / 2
            frame_bytes = width * height * 3
            result = run_media([
                "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{begin_ms / 1000:.3f}", *MEDIA_INPUT_OPTIONS,
                "-i", video_path, "-t", f"{span_ms / 1000:.3f}", "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", f"fps={SHARP_SCAN_SAMPLES * 1000 / span_ms:.6f},scale={width}:{height}",
                "-frames:v", str(SHARP_SCAN_SAMPLES), "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
            ], timeout=120, max_output=frame_bytes * SHARP_SCAN_SAMPLES)
            if result.returncode != 0:
                return None
            images = [np.frombuffer(result.stdout[i:i + frame_bytes], np.uint8).reshape(height, width, 3)
                      for i in range(0, len(result.stdout) - frame_bytes + 1, frame_bytes)]
            if len(images) < SHARP_SCAN_MIN_HITS:
                return None
            detector = self._get_detector(width, height)
            explicit = location != "auto"
            sightings = [[f for f in self._frame_info(image, 0, detector, width, height).faces
                          if SHARP_MIN_FACE_HEIGHT <= f.h and (
                              f.h <= OVERLAY_MAX_FACE_HEIGHT and in_webcam_location(f, location) if explicit
                              else is_corner_overlay(f) or is_edge_overlay_position(f))]
                         for image in images]
            best: list[Box] = []
            for seed in (f for faces in sightings for f in faces):
                group = [min(faces, key=lambda f: (f.cx - seed.cx) ** 2 + (f.cy - seed.cy) ** 2)
                         for faces in sightings if faces]
                group = [f for f in group
                         if math.hypot((f.cx - seed.cx) * src_w, (f.cy - seed.cy) * src_h) <= OVERLAY_MAX_WANDER * seed.h * src_h]
                if len(group) > len(best):
                    best = group
            if len(best) < SHARP_SCAN_MIN_HITS:
                return None
            track = FaceTrack(samples=[(i, f) for i, f in enumerate(best)])
            if not is_stationary(track, src_w, src_h):
                return None
            face = track.median_box()
            live, border = face_is_live(images, face), find_cam_border(images, face)
            # A small face in a static picture (a slide's headshot) is not a webcam.
            if live is not True or (not explicit and not is_corner_overlay(face) and border is None):
                return None
            logger.info("Found a small webcam in sharper frames at %s", face.to_list())
            return face, border
        except Exception:
            logger.warning("Sharp webcam scan failed", exc_info=True)
            return None

    @staticmethod
    def _refine_webcam_regions(shots: list[ShotLayout], keyframes: list[tuple[int, bytes]], reference: Box):
        """Confirm camera edges locally; face motion alone cannot move them.

        First from edges that stay put while the picture outside changes across
        the shot's keyframes (find_cam_border); otherwise by snapping the
        estimate to edges in a few keyframes.
        """
        for shot in shots:
            if shot.layout != LayoutType.SCREEN_CAM or shot.cam_box is None:
                continue
            if shot.cam_face is not None:
                border = find_cam_border(LayoutAnalyzer._keyframe_images(keyframes, shot.start_ms, shot.end_ms), shot.cam_face)
                # A vision box wins over a border that disagrees with it (a still
                # panel beside the webcam can extend the border search).
                if border is not None and (shot.source != "vision" or boxes_agree(border, shot.cam_box)):
                    shot.cam_box, shot.cam_box_refined = border, True
                    continue
            mid = (shot.start_ms + shot.end_ms) // 2
            images = sorted((k for k in keyframes if shot.start_ms <= k[0] < shot.end_ms),
                            key=lambda k: abs(k[0] - mid))[:3]
            decoded = [cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) for _, data in images]
            decoded = [image for image in decoded if image is not None]
            if not decoded:
                continue
            # Try the original camera region first. Leaning or turning one's
            # head can otherwise scale/translate a stationary overlay.
            for guess in [reference, shot.cam_box]:
                matches = [box for image in decoded
                           if (box := refine_cam_box(image, guess, shot.cam_face)) is not None]
                if len(matches) < min(2, len(decoded)):
                    continue
                box = Box(*(median(getattr(b, axis) for b in matches) for axis in ('x', 'y', 'w', 'h')))
                if any(max(abs(a - b) for a, b in zip(box.to_list(), candidate.to_list())) > .015
                       for candidate in matches):
                    continue
                shot.cam_box, shot.cam_box_refined = box, True
                break

    @staticmethod
    def _follow_webcam(
        shot: ShotLayout, frames: list[FrameInfo], src_w: int, src_h: int,
        reference_ms: Optional[int] = None, location: str = "auto",
    ) -> list[ShotLayout]:
        """Split a screen+webcam shot where the overlay moves, resizes or disappears."""
        cam = shot.cam_box
        vision = shot.source == "vision"
        seed = shot.cam_face or Box(cam.cx - cam.w * 0.15, cam.cy - cam.h * 0.25, cam.w * 0.3, cam.h * 0.4)
        # Other faces that sit still through the shot (a game portrait) are never the webcam.
        others = [box for box, share in webcam_anchors(frames)
                  if share >= MIN_TRACK_PRESENCE and not cam.contains(box.cx, box.cy)]
        segments = split_overlay_segments(
            frames, shot.start_ms, shot.end_ms, seed, region=cam if vision else None,
            location=location, exclude=others,
        )

        # Anchor the webcam box to the segment the vision model actually saw.
        mid = reference_ms if reference_ms is not None else (shot.start_ms + shot.end_ms) // 2
        ref_face = next((f for a, b, f in segments if a <= mid < b and f), None) or shot.cam_face

        # Keep these confirmed edits separate from overlay/dropout smoothing:
        # that code deliberately absorbs short segments and preserves a vision
        # camera during missed detections, which otherwise hides punch-ins.
        expanded = expanded_webcam_intervals(frames, cam, shot.end_ms)
        for start, end, face in segments:
            local_frames = [f for f in frames if start <= f.t_ms < end]
            # Preserve the existing central-speaker evidence over its full
            # interval; a noisy scale threshold must not fragment a known
            # talking-head shot or restore the stale camera at its edges.
            speaker_frames = [f for f in local_frames if frame_layout_evidence(f) == LayoutType.TALKING_HEAD]
            if (face is None and (not vision or not start <= mid < end)
                    and len(speaker_frames) >= LAYOUT_CHANGE_SAMPLES
                    and len(speaker_frames) >= 0.7 * len(local_frames)
                    and speaker_frames[-1].t_ms - speaker_frames[0].t_ms >= LAYOUT_CHANGE_MS - SAMPLE_JITTER_MS):
                expanded.append((start, end))
        merged = []
        for start, end in sorted(expanded):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        expanded = merged
        cuts = sorted({shot.start_ms, shot.end_ms,
                       *(t for a, b, _ in segments for t in (a, b)
                         if not any(start < t < end for start, end in expanded)),
                       *(t for a, b in expanded for t in (a, b))})
        segments = [(a, b, next(face for start, end, face in segments if start <= a < end))
                    for a, b in zip(cuts, cuts[1:])]

        result = []
        for start, end, face in segments:
            local_frames = [f for f in frames if start <= f.t_ms < end]
            if any(a <= start < b for a, b in expanded):
                result.append(heuristic_layout(local_frames, start, end, src_w, src_h, location=location))
                continue
            if face is None and vision:
                # The vision model saw the webcam; YuNet missing its face
                # (profile, lighting, a big close-up) doesn't remove it.
                result.append(ShotLayout(
                    start, end, LayoutType.SCREEN_CAM, source=shot.source,
                    screen_box=shot.screen_box, screen_focus=shot.screen_focus,
                    cam_box=cam, cam_face=shot.cam_face,
                ))
                continue
            if face is None:
                result.append(ShotLayout(
                    start, end, LayoutType.SCREEN, source=shot.source,
                    screen_box=shot.screen_box, screen_focus=shot.screen_focus,
                ))
                continue
            if (vision or shot.cam_box_refined) and ref_face is not None:
                # A known overlay (vision, or a border found in the keyframes)
                # moves and scales with its face.
                cam_box = transfer_cam_box(cam, ref_face, face)
            elif vision and cam.contains(face.cx, face.cy):
                cam_box = cam
            else:
                cam_box = estimate_cam_box(face, src_w, src_h)
            result.append(ShotLayout(
                start, end, LayoutType.SCREEN_CAM, source=shot.source,
                screen_box=shot.screen_box, screen_focus=shot.screen_focus,
                cam_box=cam_box, cam_face=face,
            ))
        return result

    # -- decoding & detection -------------------------------------------------

    def _decode_and_detect(
        self, video_path: str, start_ms: int, duration_ms: int, src_w: int, src_h: int,
    ) -> tuple[list[FrameInfo], list[tuple[int, bytes]]]:
        width, height = analysis_dimensions(src_w, src_h)
        if not 0 < duration_ms <= MAX_ANALYSIS_DURATION_MS or start_ms < 0:
            raise MediaProcessError("Layout analysis window exceeds supported limits")
        frame_limit = min(MAX_ANALYSIS_FRAMES, math.ceil(duration_ms / 1000 * ANALYSIS_FPS) + 1)
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error",
            "-ss", f"{start_ms / 1000:.3f}",
            "-protocol_whitelist", "file,pipe,fd", "-format_whitelist", "mov,matroska,webm,avi,flv,mpegts", "-i", video_path,
            "-t", f"{duration_ms / 1000:.3f}",
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"fps={ANALYSIS_FPS},scale={width}:{height}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        frame_bytes = width * height * 3
        keyframes: list[tuple[int, bytes]] = []
        keyframe_every = max(1, int(round(ANALYSIS_FPS / KEYFRAME_FPS)),
                             math.ceil(frame_limit / MAX_RETAINED_KEYFRAMES))
        keyframe_bytes = 0

        with _Detections(self, width, height) as detections, media_process(cmd, timeout=30 * 60) as (proc, _stderr):
            index = 0
            while True:
                raw = proc.stdout.read(frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise MediaProcessError("Incomplete layout analysis frame")
                if index >= frame_limit:
                    raise MediaProcessError("Layout analysis exceeds the frame limit")
                image = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
                t_ms = int(index * 1000 / ANALYSIS_FPS)

                detections.submit(image, t_ms)

                if index % keyframe_every == 0:
                    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        encoded = jpg.tobytes()
                        if len(encoded) > MAX_KEYFRAME_BYTES - keyframe_bytes:
                            raise MediaProcessError("Layout keyframes exceed the size limit")
                        keyframe_bytes += len(encoded)
                        keyframes.append((t_ms, encoded))
                index += 1
            frames = detections.results()
        if proc.returncode:
            raise MediaProcessError("Layout decoding failed")
        return frames, keyframes

    def _frame_info(self, image, t_ms, detector, width, height):
        _, faces = detector.detect(image)
        boxes = []
        scores = []
        candidates = sorted(faces, key=lambda row: float(row[14]), reverse=True)[:MAX_FACES_PER_FRAME] if faces is not None else []
        for row in candidates:
            if float(row[14]) < FACE_SCORE_THRESHOLD:
                continue
            x, y, w, h = (float(v) for v in row[:4])
            boxes.append(Box(x / width, y / height, w / width, h / height).clamp())
            scores.append(float(row[14]))

        hsv = cv2.cvtColor(cv2.resize(image, (160, 90)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return FrameInfo(t_ms=t_ms, faces=boxes, hist=hist, scores=scores,
                        content_box=detect_content_box(image))

    def _precise_frames(self, video_path, start_ms, duration_ms, src_w, src_h, progress=None):
        """Sample faces, scan every frame for camera changes, and examine the
        frames that decide each change, all from ONE decode of the window.

        FFmpeg decodes once and splits the picture: a 320-px branch prints the
        camera scan's timestamps and scene scores (the same chain as
        scan_camera_changes, so the scan is identical), and an analysis-size
        branch streams frames here: every frame up to ~40 fps, fewer above.
        The reader applies the same integer rule to the scan's timestamps, so
        every streamed frame is matched to its scan record and exact time.

        Face samples are the frames nearest each 1/ANALYSIS_FPS tick, labelled
        with their true time. Around a camera change only the frames
        DetailSelector picks are analyzed. Detection runs on a small thread
        pool while decoding continues.

        Returns (scan or None, frames, keyframes). Raises only when the decode
        itself fails; an unusable scan is dropped and sampling is kept.
        """
        from .camera_scan import SCAN_FILTER, ScanParser, check_frame_budget, filter_path, scan_window
        from .layout_precision import DetailSelector
        width, height = analysis_dimensions(src_w, src_h)
        if not 0 < duration_ms <= MAX_ANALYSIS_DURATION_MS or start_ms < 0:
            raise MediaProcessError("Layout analysis window exceeds supported limits")
        scan_start, scan_end = scan_window(start_ms, start_ms + duration_ms)
        check_frame_budget(video_path, scan_start, scan_end)
        frame_limit = min(MAX_ANALYSIS_FRAMES, math.ceil(duration_ms / 1000 * ANALYSIS_FPS) + 1)
        keyframe_every = max(1, int(round(ANALYSIS_FPS / KEYFRAME_FPS)),
                             math.ceil(frame_limit / MAX_RETAINED_KEYFRAMES))
        step = 1000 / ANALYSIS_FPS
        # Every frame crosses the pipe, so stream compact I420 and convert only
        # the frames that are analyzed.
        frame_bytes = width * height * 3 // 2
        parser = ScanParser(scan_start, scan_end, strict=False)
        selector = DetailSelector(duration_ms)
        regular: list[tuple[float, bytes]] = []
        cuts = _EvenSample(2 * MAX_RETAINED_KEYFRAMES)
        with tempfile.TemporaryDirectory(prefix="bridgeclip-scan-") as folder:
            scan_path = os.path.join(folder, "scan.txt")
            open(scan_path, "xb").close()
            graph = (f"[0:v:0]settb=1/1000000,split=2[scan][frames];"
                     f"[scan]{SCAN_FILTER},metadata=mode=print:direct=1:file={filter_path(scan_path)},nullsink;"
                     f"[frames]select='isnan(prev_selected_pts)+gte(pts-prev_selected_pts,{MIN_STREAMED_GAP_US})',"
                     f"scale={width}:{height}[analysis]")
            cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{scan_start / 1000:.6f}",
                   "-t", f"{(scan_end - scan_start) / 1000:.6f}", *MEDIA_INPUT_OPTIONS, "-i", video_path,
                   "-filter_complex", graph, "-map", "[analysis]", "-an", "-sn", "-dn",
                   "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
            with open(scan_path, "rb") as scan_output, _Detections(self, width, height) as detections, \
                    media_process(cmd, timeout=30 * 60) as (proc, _):
                def wait_for(index):
                    """Block until scan record `index` (timestamp and score) is read."""
                    waited = None
                    while parser.complete <= index:
                        data = scan_output.read(65536)
                        if data:
                            parser.feed(data)
                            continue
                        if proc.poll() is not None:
                            parser.feed(scan_output.read())
                            parser.finish()
                            if parser.complete <= index:
                                raise MediaProcessError("Camera scan is missing frames")
                            break
                        waited = waited or time.monotonic()
                        if time.monotonic() - waited > SCAN_OUTPUT_WAIT_S:
                            raise MediaProcessError("Camera scan output stalled")
                        time.sleep(.0005)

                record, streamed_pts = 0, math.nan
                def streamed(pts):
                    # FFmpeg: isnan(prev_selected_pts) + gte(pts - prev_selected_pts, gap)
                    return math.isnan(streamed_pts) or (pts is not None and pts - streamed_pts >= MIN_STREAMED_GAP_US)

                def next_streamed():
                    """Scan record of the next streamed frame, and the highest
                    in-window scene score among the records skipped before it."""
                    nonlocal record, streamed_pts
                    skipped = None
                    while True:
                        wait_for(record)
                        pts, score, at = parser.pts[record], parser.scores[record], parser.records[record]
                        record += 1
                        if streamed(pts):
                            streamed_pts = math.nan if pts is None else pts
                            return record - 1, skipped
                        if at is not None and score is not None and 0 < at - start_ms < duration_ms:
                            skipped = score if skipped is None else max(skipped, score)

                analyzed: set[int] = set()
                samples = 0
                def analyze_frame(index, t, image):
                    if index not in analyzed:
                        analyzed.add(index)
                        detections.submit(image, t)

                previous = None  # (index, t, image) of the last usable frame
                tick, index, reported = 0.0, 0, 0
                while raw := proc.stdout.read(frame_bytes):
                    if len(raw) != frame_bytes:
                        raise MediaProcessError("Incomplete layout analysis frame")
                    current, skipped = next_streamed()
                    at, score = parser.records[current], parser.scores[current]
                    if skipped is not None:
                        # A change on a frame that was not streamed starts on
                        # this one, at most one frame later.
                        score = skipped if score is None else max(score, skipped)
                    t = None if at is None else round(at - start_ms, 3)
                    if t is None or t >= duration_ms:
                        index += 1
                        continue
                    image = np.frombuffer(raw, np.uint8).reshape(height * 3 // 2, width)
                    if t >= 0:
                        # Face samples: the frame nearest each tick, never a duplicate.
                        while tick < duration_ms and tick <= t:
                            nearest = previous if previous is not None and previous[1] >= 0 and tick - previous[1] <= t - tick else (index, t, image)
                            if nearest[0] not in analyzed:
                                if samples >= frame_limit:
                                    raise MediaProcessError("Layout analysis exceeds the frame limit")
                                analyze_frame(*nearest)
                                if samples % keyframe_every == 0:
                                    regular.append((nearest[1], _jpeg(nearest[2])))
                                samples += 1
                            tick += step
                        take_previous, take_current, change = selector.select(
                            t, score, previous[1] if previous is not None else None)
                        if take_previous:
                            analyze_frame(*previous)
                        if take_current:
                            analyze_frame(index, t, image)
                        if change:
                            # The first frame of each camera change: a keyframe
                            # for short shots, which regular samples can miss.
                            cuts.add(lambda image=image, t=t: (t, _jpeg(image)))
                        if progress:
                            percent = min(99, int(100 * t / duration_ms))
                            if percent >= reported + 5:
                                reported = percent
                                progress("Sampling faces", percent)
                    previous = (index, t, image)
                    index += 1
                if previous is not None and previous[1] >= 0 and tick < duration_ms and previous[0] not in analyzed:
                    analyze_frame(*previous)
                    if samples % keyframe_every == 0:
                        regular.append((previous[1], _jpeg(previous[2])))
                if proc.wait():
                    raise MediaProcessError("Layout decoding failed")
                parser.feed(scan_output.read())
                parser.finish()
                if any(streamed(pts) for pts in parser.pts[record:]):
                    raise MediaProcessError("Camera scan and analysis frames differ")
                if progress and detections.pending():
                    progress("Refining face tracking", None)
                frames = detections.results()
        if selector.capped:
            logger.info("Camera change details reached their frame budget; later changes use sampled faces")
        scan = parser.result() if parser.usable and parser.frames else None
        return scan, frames, retain_keyframes(regular, cuts.items())

    def _get_detector(self, width: int, height: int):
        detector = getattr(self._local, "detector", None)
        if detector is None:
            detector = cv2.FaceDetectorYN.create(
                self._model_path(), "", (width, height), FACE_SCORE_THRESHOLD, 0.3, 50,
            )
            self._local.detector = detector
        detector.setInputSize((width, height))
        return detector

    @staticmethod
    def _pick_keyframe(
        keyframes: list[tuple[int, bytes]], t_ms: int, start_ms: int, end_ms: int,
    ) -> Optional[tuple[int, bytes]]:
        # Never borrow an image from the other side of a layout transition.
        candidates = [k for k in keyframes if start_ms <= k[0] < end_ms]
        if not candidates:
            return None
        return min(candidates, key=lambda k: abs(k[0] - t_ms))

    def _webcam_location(self) -> str:
        """Where the person said their webcam sits (Settings > Framing), "auto" when unset."""
        location = getattr(self.settings, "webcam_location", "auto")
        return location if location in WEBCAM_LOCATIONS else "auto"

    def _apply_framing_settings(self, shots: list[ShotLayout], style: str, src_w: int, src_h: int,
                                frames: list[FrameInfo]) -> None:
        """The person's webcam position and screen framing (Smart framing only).

        A screen fills the frame unless it would be enlarged too far or people
        are on camera in it.
        """
        position = "bottom" if getattr(self.settings, "webcam_position", "top") == "bottom" else "top"
        fill = style == LayoutStyle.AUTO and bool(getattr(self.settings, "screen_fill", True))
        _, out_h = get_output_dimensions("9:16")
        for shot in shots:
            shot.cam_position = position
            shot.screen_fill = (fill and fills_screen_layout(shot)
                                and fill_crop_height(shot, src_w, src_h) * MAX_FILL_UPSCALE >= out_h
                                and not shows_people(shot, frames, self._webcam_location()))

    @staticmethod
    def _merge_adjacent(shots: list[ShotLayout]) -> list[ShotLayout]:
        """Join neighbouring shots that ended up with the same static framing."""
        merged: list[ShotLayout] = []
        for shot in shots:
            prev = merged[-1] if merged else None
            same_static = (
                prev is not None
                and prev.layout == shot.layout
                and prev.content_box == shot.content_box
                and prev.cam_box_refined == shot.cam_box_refined
                and (not shot.cam_box_refined or prev.cam_box == shot.cam_box)
                # A filled screen is cropped around its focus; a new focus is a new framing.
                and prev.screen_fill == shot.screen_fill
                and (not shot.screen_fill or (prev.screen_focus == shot.screen_focus
                                              and prev.screen_box == shot.screen_box))
                and shot.layout in (LayoutType.SCREEN, LayoutType.SCREEN_CAM)
                and (shot.layout == LayoutType.SCREEN or (
                    prev.cam_box and shot.cam_box
                    and abs(prev.cam_box.cx - shot.cam_box.cx) < 0.05
                    and abs(prev.cam_box.cy - shot.cam_box.cy) < 0.05
                    and 0.85 <= shot.cam_box.h / max(prev.cam_box.h, 1e-6) <= 1.18
                ))
            )
            if same_static:
                prev.end_ms = shot.end_ms
            else:
                merged.append(shot)
        return merged

    # -- vision ---------------------------------------------------------------

    def _vision_enabled(self) -> bool:
        if getattr(self, "_vision_unavailable", False):
            return False
        # Only Gemini and OpenAI-compatible models the user marked as image
        # capable can classify frames; otherwise framing uses the heuristics.
        return bool(self.settings.layout_vision_enabled and has_credentials(self.settings) and supports_images(self.settings))

    async def _vision_classify(
        self, keyframe: bytes, shot_frames: list[FrameInfo], heuristic: ShotLayout,
        diagnostic: Optional[dict] = None,
        cache_only: bool = False,
    ) -> tuple[Optional[dict], float]:
        """Ask the vision model about one keyframe; cached per visual setup."""
        signature = (heuristic.layout, tuple(sorted(
            (round(b.cx, 1), round(b.cy, 1), round(b.w, 1), round(b.h, 1))
            for f in shot_frames for b in framing_faces(f)
        )))
        image = cv2.imdecode(np.frombuffer(keyframe, np.uint8), cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(cv2.resize(image, (160, 90)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        for cached_hist, cached_sig, cached, provenance in self._vision_cache:
            if cached_sig == signature and cv2.compareHist(hist, cached_hist, cv2.HISTCMP_BHATTACHARYYA) < 0.2:
                if diagnostic is not None:
                    diagnostic.update({**provenance, "status": "success", "cache_hit": True})
                return cached, 0.0

        if cache_only:
            if diagnostic is not None: diagnostic.update(status='budget_limited')
            return None, 0.0

        faces = sorted(
            {tuple(round(v, 3) for v in b.to_list()) for f in shot_frames[:: max(1, len(shot_frames) // 4)] for b in framing_faces(f)}
        )[:6]
        payload: dict[str, Any] = {
            "model": self.settings.layout_vision_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT.replace("{faces}", json.dumps(faces) or "[]")},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(keyframe).decode(),
                    }},
                ],
            }],
            "max_tokens": 4000,
            "response_format": json_schema_format("frame_layout", VISION_SCHEMA),
            "provider": {"require_parameters": True},
        }
        fallbacks = self.settings.get_layout_vision_fallback_models()
        if fallbacks:
            payload["models"] = fallbacks
        apply_reasoning(payload, self.settings.layout_vision_reasoning_effort, temperature=0.0)

        client = await self._get_client()
        for attempt in range(2):
            try:
                if uses_openrouter(self.settings):
                    body, usage = await chat_completion(client, payload)
                    content, _ = message_text(body)
                    result = json.loads(content or "")
                else:
                    body, usage = await ai_chat_completion(client, prepare_payload(self.settings, payload, is_vision=True), self.settings)
                    content, _ = message_text(body)
                    result = tolerant_json_parse(content or "")
                provenance = {"cache_id": hashlib.sha256(keyframe).hexdigest()[:16],
                              "cache_source_ms": diagnostic.get("source_ms") if diagnostic else None,
                              "model": str(body.get("model") or self.settings.layout_vision_model)[:120]}
                self._vision_cache.append((hist, signature, result, provenance))
                self._vision_rejections = 0
                if diagnostic is not None:
                    diagnostic.update({**provenance, "status": "success", "cache_hit": False})
                return result, usage.get("cost") or 0.0
            except OpenRouterError as e:
                reason = getattr(e, "reason", None)
                if reason in VISION_FATAL_REASONS:
                    # A wrong key, model or URL fails every shot. Stop asking.
                    self._vision_unavailable = True
                elif reason == "bad_request":
                    # One rejected frame can be a passing server problem; the
                    # same rejection twice in a row means the model takes no images.
                    self._vision_rejections = getattr(self, "_vision_rejections", 0) + 1
                    if self._vision_rejections >= VISION_REJECTIONS_BEFORE_OFF:
                        self._vision_unavailable = True
                if not e.retryable or attempt == 1:
                    logger.warning(f"Layout vision failed, using heuristics: {e}")
                    return None, 0.0
                await asyncio.sleep(1.5)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning(f"Layout vision returned invalid JSON, using heuristics: {e}")
                return None, 0.0
        return None, 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = get_ai_client(self.settings, timeout=120.0, limit=VISION_TIMEOUT_LIMIT_SECONDS)
        return self._http_client

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
