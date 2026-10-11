"""Finding the webcam: stationary faces near any edge, live faces, borders that stay put, sharper scans."""

import asyncio

import cv2
import numpy as np
import pytest

from clip_engine.services import layout_analyzer as module
from clip_engine.services.layout_analyzer import (
    Box,
    FaceTrack,
    FrameInfo,
    LayoutAnalyzer,
    LayoutType,
    OverlayEvidence,
    ShotLayout,
    classify_shot,
    face_is_live,
    find_cam_border,
    is_edge_overlay_position,
    is_stationary,
)

W, H = 640, 360
# A webcam overlay in pixels at analysis size: x0, y0, x1, y1.
CAM = (448, 220, 624, 352)
FACE = Box(510 / W, 250 / H, 26 / W, 34 / H)


def stream_frames(count=6, cam=CAM, border=True, inner_line=False, game_moves=True, face_moves=True, seed=1):
    """Gameplay that changes every frame, with a webcam overlay whose room stays still."""
    rng = np.random.default_rng(seed)
    room = cv2.GaussianBlur(rng.integers(60, 120, (H, W), dtype=np.uint8), (9, 9), 0)
    still_game = rng.integers(0, 256, (H, W), dtype=np.uint8)
    frames = []
    x0, y0, x1, y1 = cam
    for i in range(count):
        game = rng.integers(0, 256, (H, W), dtype=np.uint8) if game_moves else still_game.copy()
        image = game.copy()
        image[y0:y1, x0:x1] = room[y0:y1, x0:x1]
        if inner_line:
            image[y0:y1, 488:490] = 200  # a door frame in the room, inside the overlay
        fx, fy, fw, fh = round(FACE.x * W), round(FACE.y * H), round(FACE.w * W), round(FACE.h * H)
        shift = (i % 3) * 2 if face_moves else 0
        image[fy:fy + fh, fx + shift:fx + fw + shift] = 170 + (i % 2) * 40  # a face that talks and moves
        if border:
            cv2.rectangle(image, (x0 - 3, y0 - 3), (x1 + 2, y1 + 2), 255, 3)
        frames.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    return frames


def close_to(box, cam, tolerance=4):
    x0, y0, x1, y1 = cam
    got = (box.x * W, box.y * H, (box.x + box.w) * W, (box.y + box.h) * H)
    return all(abs(a - b) <= tolerance for a, b in zip(got, (x0, y0, x1, y1))), got


class TestBorder:
    def test_a_drawn_border_is_found(self):
        box = find_cam_border(stream_frames(), FACE)
        ok, got = close_to(box, CAM)
        assert ok, got

    def test_an_overlay_without_a_border_is_found_by_where_the_picture_moves(self):
        box = find_cam_border(stream_frames(border=False), FACE)
        ok, got = close_to(box, CAM, tolerance=6)
        assert ok, got

    def test_a_still_line_inside_the_overlay_is_not_its_border(self):
        box = find_cam_border(stream_frames(inner_line=True), FACE)
        ok, got = close_to(box, CAM)
        assert ok, got

    def test_an_overlay_in_the_corner_runs_to_the_frame_edges(self):
        cam = (448, 220, W, H)
        box = find_cam_border(stream_frames(cam=cam, border=False), FACE)
        ok, got = close_to(box, cam, tolerance=6)
        assert ok, got

    def test_no_answer_when_nothing_moves_around_it(self):
        assert find_cam_border(stream_frames(game_moves=False, border=False), FACE) is None
        assert find_cam_border(stream_frames()[:2], FACE) is None


class TestLiveFaces:
    def test_a_dim_soft_webcam_is_not_called_a_picture(self):
        frames = stream_frames(face_moves=False)
        for i, frame in enumerate(frames):  # barely changing: a still streamer on a dim camera
            frame[250:284, 510:536] = 150 + (i % 2)
        assert face_is_live(frames, FACE) is None

    def test_a_talking_face_is_live_and_a_still_portrait_is_not(self):
        assert face_is_live(stream_frames(), FACE) is True
        portrait = stream_frames(face_moves=False)
        for frame in portrait:  # the face patch identical in every frame
            frame[250:284, 510:536] = 180
        assert face_is_live(portrait, FACE) is False

    def test_a_frozen_video_says_nothing(self):
        frozen = [stream_frames(count=1)[0]] * 4
        assert face_is_live(frozen, FACE) is None


def track(*centers, h=0.08):
    return FaceTrack(samples=[(i * 250, Box(cx - h / 2, cy - h / 2, h * 9 / 16, h)) for i, (cx, cy) in enumerate(centers)])


class TestClassification:
    def test_stationary_faces_and_faces_that_move(self):
        assert is_stationary(track(*[(0.85, 0.85)] * 8), 1920, 1080)
        assert not is_stationary(track(*[(0.7 + 0.05 * i, 0.85) for i in range(8)]), 1920, 1080)
        # Leaning in and moving about inside the webcam is still stationary.
        assert is_stationary(track(*[(0.85 + 0.01 * (i % 3), 0.85) for i in range(8)]), 1920, 1080)
        zooming = FaceTrack(samples=[(i, Box(0.8, 0.8, 0.03 * (1 + i / 3), 0.05 * (1 + i / 3))) for i in range(8)])
        assert not is_stationary(zooming, 1920, 1080)

    def test_edge_positions(self):
        assert is_edge_overlay_position(Box(0.47, 0.82, 0.04, 0.08))   # bottom centre
        assert is_edge_overlay_position(Box(0.88, 0.46, 0.04, 0.08))   # middle of the right side
        assert not is_edge_overlay_position(Box(0.88, 0.82, 0.04, 0.08))  # a corner
        assert not is_edge_overlay_position(Box(0.48, 0.46, 0.04, 0.08))  # the centre
        assert not is_edge_overlay_position(Box(0.48, 0.12, 0.04, 0.08))  # top centre: where standing people's faces are

    def test_a_face_roaming_beside_the_webcam_is_the_games(self):
        cam = track(*[(0.9, 0.86)] * 8, h=0.07)
        roaming = track(*[(0.62 + 0.05 * i, 0.8) for i in range(8)], h=0.1)  # bigger, so it was picked before
        shot, _ = classify_shot([roaming, cam], 8, 1920, 1080)
        assert shot.layout == LayoutType.SCREEN_CAM and shot.cam_face == cam.median_box()
        # A bigger game face drifting slowly enough to pass as stationary still loses to the webcam,
        # whose head moves a little around one place.
        drifting = track(*[(0.15 + 0.012 * i, 0.82) for i in range(8)], h=0.12)
        live_cam = track(*[(0.9 + (0.01 if i % 2 else -0.01), 0.86) for i in range(8)], h=0.07)
        shot, _ = classify_shot([drifting, live_cam], 8, 1920, 1080)
        assert shot.cam_face == live_cam.median_box()

    def test_without_a_webcam_a_moving_face_is_still_followed(self):
        # A presenter walking across a wide shot, a vlogger walking up to the camera.
        walking = track(*[(0.2 + 0.08 * i, 0.25) for i in range(8)], h=0.1)
        approaching = FaceTrack(samples=[(i, Box(0.45, 0.2, 0.03 * (1 + i / 2), 0.05 * (1 + i / 2))) for i in range(8)])
        for person in (walking, approaching):
            shot, followed = classify_shot([person], 8, 1920, 1080)
            assert shot.layout == LayoutType.TALKING_HEAD and followed is person

    def test_a_still_corner_face_is_the_webcam_unless_it_is_a_picture(self):
        still = track(*[(0.86, 0.84)] * 8)
        assert classify_shot([still], 8, 1920, 1080)[0].layout == LayoutType.SCREEN_CAM
        portrait = classify_shot([still], 8, 1920, 1080, lambda face: OverlayEvidence(live=False))[0]
        assert portrait.layout == LayoutType.SCREEN

    def test_a_bottom_centre_webcam_needs_its_border(self):
        bottom = track(*[(0.5, 0.86)] * 8)
        # Without a border it may be a person in a wide shot, as before.
        assert classify_shot([bottom], 8, 1920, 1080)[0].layout == LayoutType.TALKING_HEAD
        border = Box(0.4, 0.7, 0.2, 0.3)
        shot, _ = classify_shot([bottom], 8, 1920, 1080, lambda face: OverlayEvidence(live=True, border=border))
        assert shot.layout == LayoutType.SCREEN_CAM and shot.cam_box == border and shot.cam_box_refined


def jpeg(image):
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


class TestAnalyzer:
    def test_keyframes_snap_an_estimated_webcam_to_its_border(self):
        keyframes = [(i * 1000, jpeg(frame)) for i, frame in enumerate(stream_frames(count=8))]
        shot = ShotLayout(0, 8000, LayoutType.SCREEN_CAM, cam_box=Box(0.72, 0.62, 0.22, 0.3), cam_face=FACE)
        LayoutAnalyzer._refine_webcam_regions([shot], keyframes, shot.cam_box)
        ok, got = close_to(shot.cam_box, CAM)
        assert ok and shot.cam_box_refined, got

    def sharp_analyzer(self, monkeypatch, faces_per_frame):
        frames = stream_frames(count=5)
        monkeypatch.setattr(module, "run_media", lambda cmd, **kw: type("R", (), {"returncode": 0, "stdout": b"".join(f.tobytes() for f in frames)})())
        detections = iter(faces_per_frame)
        analyzer = LayoutAnalyzer()
        monkeypatch.setattr(analyzer, "_get_detector", lambda w, h: None)
        monkeypatch.setattr(analyzer, "_frame_info", lambda image, t, d, w, h: FrameInfo(t, next(detections), None))
        return analyzer

    def test_a_small_webcam_is_found_in_sharper_frames(self, monkeypatch):
        analyzer = self.sharp_analyzer(monkeypatch, [[FACE]] * 5)
        found = analyzer._sharp_overlay("in.mp4", 0, 0, 10_000, W, H)
        assert found is not None
        face, border = found
        assert face == FACE
        ok, got = close_to(border, CAM)
        assert ok, got

    def test_a_face_seen_in_too_few_sharp_frames_is_ignored(self, monkeypatch):
        analyzer = self.sharp_analyzer(monkeypatch, [[FACE], [FACE], [FACE], [], []])
        assert analyzer._sharp_overlay("in.mp4", 0, 0, 10_000, W, H) is None

    def test_analysis_uses_the_sharp_scan_for_shots_without_a_webcam(self, monkeypatch):
        analyzer = LayoutAnalyzer()
        hist = np.zeros((24, 16), np.float32)
        hist[1, 1] = 1
        frames = [FrameInfo(t, [], hist) for t in range(0, 8000, 250)]
        keyframes = [(t, jpeg(stream_frames(count=1)[0])) for t in range(0, 8000, 1000)]
        border = Box(0.7, 0.6, 0.27, 0.36)
        monkeypatch.setattr(LayoutAnalyzer, "available", property(lambda _: True))
        monkeypatch.setattr(analyzer, "_decode_and_detect", lambda *a: (frames, keyframes))
        monkeypatch.setattr(analyzer, "_vision_enabled", lambda: False)
        seen = []
        monkeypatch.setattr(analyzer, "_sharp_overlay", lambda *a: seen.append(a) or (FACE, border))
        plan = asyncio.run(analyzer.analyze("in.mp4", 0, 8000, 1920, 1080, precise=False))
        assert len(seen) == 1
        [shot] = plan.shots
        assert shot.layout == LayoutType.SCREEN_CAM and shot.cam_box == border and shot.cam_face == FACE
        # Pacing-only plans never pay for it.
        seen.clear()
        asyncio.run(analyzer.analyze("in.mp4", 0, 8000, 1920, 1080, vision=False, precise=False))
        assert seen == []


class TestReviewFixes:
    def test_a_vision_box_wins_over_a_border_that_disagrees(self):
        keyframes = [(i * 1000, jpeg(frame)) for i, frame in enumerate(stream_frames(count=8))]
        vision_box = Box(0.5, 0.1, 0.2, 0.25)  # far from the border the keyframes show
        shot = ShotLayout(0, 8000, LayoutType.SCREEN_CAM, source="vision", cam_box=vision_box, cam_face=FACE)
        LayoutAnalyzer._refine_webcam_regions([shot], keyframes, shot.cam_box)
        assert shot.cam_box == vision_box

    def test_a_found_border_follows_its_face_after_a_split(self):
        border = Box(*(v / d for v, d in zip((CAM[0], CAM[1], CAM[2] - CAM[0], CAM[3] - CAM[1]), (W, H, W, H))))
        shot = ShotLayout(0, 4000, LayoutType.SCREEN_CAM, cam_box=border, cam_face=FACE, cam_box_refined=True)
        hist = np.zeros((24, 16), np.float32)
        frames = [FrameInfo(t, [FACE], hist) for t in range(0, 4000, 250)]
        [sub] = LayoutAnalyzer._follow_webcam(shot, frames, W, H)
        assert sub.cam_box == border

    def test_the_sharp_scan_wants_a_changing_face(self, monkeypatch):
        analyzer = TestAnalyzer().sharp_analyzer(monkeypatch, [[FACE]] * 5)
        monkeypatch.setattr(module, "face_is_live", lambda images, face: None)
        assert analyzer._sharp_overlay("in.mp4", 0, 0, 10_000, W, H) is None

    def test_the_sharp_scan_skips_shots_with_faces_and_vision_verdicts(self, monkeypatch):
        analyzer = LayoutAnalyzer()
        hist = np.zeros((24, 16), np.float32)
        crowd = [FrameInfo(t, [Box(0.3, 0.3, 0.05, 0.09), Box(0.45, 0.3, 0.05, 0.09), Box(0.6, 0.3, 0.05, 0.09)], hist)
                 for t in range(0, 8000, 250)]
        monkeypatch.setattr(LayoutAnalyzer, "available", property(lambda _: True))
        monkeypatch.setattr(analyzer, "_decode_and_detect", lambda *a: (crowd, []))
        monkeypatch.setattr(analyzer, "_vision_enabled", lambda: False)
        seen = []
        monkeypatch.setattr(analyzer, "_sharp_overlay", lambda *a: seen.append(a) or (FACE, None))
        plan = asyncio.run(analyzer.analyze("in.mp4", 0, 8000, 1920, 1080, precise=False))
        assert seen == [] and plan.shots[0].layout == LayoutType.SCREEN

    def test_a_sharp_webcam_that_does_not_hold_across_the_shot_is_dropped(self, monkeypatch):
        analyzer = LayoutAnalyzer()
        hist = np.zeros((24, 16), np.float32)
        frames = [FrameInfo(t, [], hist) for t in range(0, 8000, 250)]
        # The box sits over moving game in every keyframe: no calm overlay there.
        busy = [(t, jpeg(stream_frames(count=1, seed=t)[0])) for t in range(0, 8000, 1000)]
        monkeypatch.setattr(LayoutAnalyzer, "available", property(lambda _: True))
        monkeypatch.setattr(analyzer, "_decode_and_detect", lambda *a: (frames, busy))
        monkeypatch.setattr(analyzer, "_vision_enabled", lambda: False)
        game_spot = Box(0.05, 0.05, 0.04, 0.07)
        monkeypatch.setattr(analyzer, "_sharp_overlay", lambda *a: (game_spot, Box(0.02, 0.02, 0.15, 0.2)))
        plan = asyncio.run(analyzer.analyze("in.mp4", 0, 8000, 1920, 1080, precise=False))
        assert plan.shots[0].layout == LayoutType.SCREEN
