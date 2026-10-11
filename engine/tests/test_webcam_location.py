"""Telling the streamer's webcam from faces in the game: a chosen webcam location, and Auto's ranking."""

from types import SimpleNamespace

import pytest

from clip_engine.config import Settings
from clip_engine.services.layout_analyzer import (
    Box,
    FaceTrack,
    FrameInfo,
    LayoutAnalyzer,
    LayoutType,
    OverlayEvidence,
    ShotLayout,
    classify_shot,
    face_drift,
    in_webcam_location,
    merge_vision_result,
    webcam_anchors,
)

SRC_W, SRC_H = 1920, 1080


def track(*centers, h=0.07):
    return FaceTrack(samples=[(i * 250, Box(cx - h * 9 / 32, cy - h / 2, h * 9 / 16, h)) for i, (cx, cy) in enumerate(centers)])


def still(cx, cy, n=8, h=0.07):
    """A drawn portrait: the same box in every sample."""
    return track(*[(cx, cy)] * n, h=h)


def person(cx, cy, n=8, h=0.07):
    """A streamer on a webcam: the head moves a little around one place."""
    return track(*[(cx + (0.01 if i % 2 else -0.01), cy + (0.005 if i % 3 == 0 else 0)) for i in range(n)], h=h)


class TestLocations:
    @pytest.mark.parametrize("location, inside, outside", [
        ("top-left", (0.12, 0.15), (0.88, 0.15)),
        ("top-right", (0.88, 0.15), (0.12, 0.85)),
        ("bottom-left", (0.12, 0.85), (0.12, 0.15)),
        ("bottom-right", (0.88, 0.85), (0.5, 0.85)),
        ("bottom-center", (0.5, 0.85), (0.88, 0.15)),
    ])
    def test_zones(self, location, inside, outside):
        assert in_webcam_location(Box(inside[0] - 0.02, inside[1] - 0.04, 0.04, 0.08), location)
        assert not in_webcam_location(Box(outside[0] - 0.02, outside[1] - 0.04, 0.04, 0.08), location)
        assert in_webcam_location(Box(outside[0] - 0.02, outside[1] - 0.04, 0.04, 0.08), "auto")

    def test_the_setting_is_read_leniently(self):
        assert Settings(_env_file=None).webcam_location == "auto"
        assert Settings(_env_file=None, webcam_location="Bottom-Right").webcam_location == "bottom-right"
        assert Settings(_env_file=None, webcam_location="middle").webcam_location == "auto"
        analyzer = LayoutAnalyzer.__new__(LayoutAnalyzer)
        analyzer.settings = SimpleNamespace(webcam_location="top-left")
        assert analyzer._webcam_location() == "top-left"
        analyzer.settings = SimpleNamespace()
        assert analyzer._webcam_location() == "auto"


class TestChosenLocation:
    def test_only_a_face_in_the_chosen_corner_is_the_webcam(self):
        hud, cam = still(0.12, 0.15), person(0.88, 0.85)
        shot, _ = classify_shot([hud, cam], 8, SRC_W, SRC_H, location="bottom-right")
        assert shot.layout == LayoutType.SCREEN_CAM and shot.cam_face == cam.median_box()
        shot, _ = classify_shot([hud, cam], 8, SRC_W, SRC_H, location="top-left")
        assert shot.cam_face == hud.median_box()

    def test_a_game_face_elsewhere_is_neither_the_webcam_nor_a_person_to_follow(self):
        shot, followed = classify_shot([still(0.12, 0.15)], 8, SRC_W, SRC_H, location="bottom-right")
        assert shot.layout == LayoutType.SCREEN and followed is None
        # Moving about near another edge, it still belongs to the game.
        roaming = track(*[(0.1 + 0.03 * i, 0.5) for i in range(8)])
        shot, followed = classify_shot([roaming], 8, SRC_W, SRC_H, location="bottom-right")
        assert shot.layout == LayoutType.SCREEN and followed is None

    def test_a_bottom_centre_webcam_needs_no_border_once_chosen(self):
        cam = person(0.5, 0.86)
        assert classify_shot([cam], 8, SRC_W, SRC_H)[0].layout == LayoutType.TALKING_HEAD
        shot, _ = classify_shot([cam], 8, SRC_W, SRC_H, location="bottom-center")
        assert shot.layout == LayoutType.SCREEN_CAM and shot.cam_face == cam.median_box()

    def test_a_still_picture_in_the_chosen_corner_is_still_not_the_webcam(self):
        shot, _ = classify_shot([still(0.88, 0.85)], 8, SRC_W, SRC_H, lambda face: OverlayEvidence(live=False),
                                location="bottom-right")
        assert shot.layout == LayoutType.SCREEN

    def test_a_big_face_in_the_middle_is_still_followed(self):
        speaker = person(0.5, 0.4, h=0.3)
        shot, followed = classify_shot([speaker], 8, SRC_W, SRC_H, location="bottom-right")
        assert shot.layout == LayoutType.TALKING_HEAD and followed is speaker

    def test_the_vision_models_webcam_must_be_in_the_chosen_place(self):
        top_left = {"layout": "screen_cam", "cam_box": [0, 0, 300, 250], "screen_box": [0, 0, 1000, 1000]}
        heuristic = ShotLayout(0, 0, LayoutType.SCREEN_CAM, cam_box=Box(0.75, 0.7, 0.24, 0.28),
                               cam_face=Box(0.84, 0.78, 0.04, 0.08))
        assert merge_vision_result(heuristic, top_left, SRC_W, SRC_H).cam_box == Box(0, 0, 0.25, 0.3)
        kept = merge_vision_result(heuristic, top_left, SRC_W, SRC_H, "bottom-right")
        assert kept.layout == LayoutType.SCREEN_CAM and kept.cam_box == heuristic.cam_box
        screen = merge_vision_result(ShotLayout(0, 0, LayoutType.SCREEN), top_left, SRC_W, SRC_H, "bottom-right")
        assert screen.layout == LayoutType.SCREEN and screen.screen_box == Box(0, 0, 1, 1)


class TestAutoRanking:
    def test_a_moving_streamer_beats_a_pixel_still_game_portrait(self):
        hud, cam = still(0.12, 0.15), person(0.88, 0.85)
        shot, _ = classify_shot([hud, cam], 8, SRC_W, SRC_H)
        assert shot.cam_face == cam.median_box()
        # Even when the portrait has a border and compression makes it look live.
        border = Box(0.02, 0.02, 0.2, 0.28)
        shot, _ = classify_shot([hud, cam], 8, SRC_W, SRC_H,
                                lambda face: OverlayEvidence(live=True, border=border if face.cx < 0.5 else None))
        assert shot.cam_face == cam.median_box()

    def test_a_face_there_all_clip_long_beats_one_that_comes_and_goes(self):
        hud, cam = still(0.12, 0.15), still(0.88, 0.85)
        anchors = [(hud.median_box(), 0.2), (cam.median_box(), 0.9)]
        assert classify_shot([hud, cam], 8, SRC_W, SRC_H, anchors=anchors)[0].cam_face == cam.median_box()
        anchors = [(hud.median_box(), 0.9), (cam.median_box(), 0.2)]
        assert classify_shot([hud, cam], 8, SRC_W, SRC_H, anchors=anchors)[0].cam_face == hud.median_box()

    def test_a_border_found_decides_between_two_moving_faces(self):
        hud, cam = person(0.12, 0.15), person(0.88, 0.85)
        border = Box(0.78, 0.7, 0.2, 0.28)

        def inspect(face):
            return OverlayEvidence(live=True, border=border if face.cx > 0.5 else None)
        shot, _ = classify_shot([hud, cam], 8, SRC_W, SRC_H, inspect)
        assert shot.cam_face == cam.median_box() and shot.cam_box == border

    def test_a_face_drifting_away_gets_no_credit_for_moving(self):
        drifting = track(*[(0.15 + 0.012 * i, 0.82) for i in range(8)], h=0.12)
        assert face_drift(drifting, SRC_W, SRC_H) > 0.5
        assert face_drift(person(0.88, 0.85), SRC_W, SRC_H) < 0.2

    def test_webcam_anchors_measure_presence_across_the_clip(self):
        cam, hud = Box(0.86, 0.8, 0.04, 0.08), Box(0.1, 0.1, 0.04, 0.08)
        frames = [FrameInfo(t_ms=i * 250, faces=[cam] + ([hud] if i < 2 else []), hist=None) for i in range(10)]
        frames.append(FrameInfo(t_ms=2500, faces=[Box(0.45, 0.3, 0.2, 0.3)], hist=None))  # a big face: not a webcam
        anchors = dict((round(box.cx, 2), share) for box, share in webcam_anchors(frames))
        assert anchors == {0.88: pytest.approx(10 / 11), 0.12: pytest.approx(2 / 11)}


class TestReviewFixes:
    def test_a_still_streamer_with_a_border_beats_an_animated_portrait(self):
        streamer = track(*[(0.88 + (0.0004 if i % 2 else -0.0004), 0.85) for i in range(8)])  # about 0.015 face heights
        portrait = person(0.12, 0.15)  # an idle animation moves it like a person
        border = Box(0.78, 0.7, 0.2, 0.28)
        shot, _ = classify_shot([portrait, streamer], 8, SRC_W, SRC_H,
                                lambda face: OverlayEvidence(live=True, border=border if face.cx > 0.5 else None))
        assert shot.cam_face == streamer.median_box()

    def test_a_presenter_beside_a_chosen_corner_is_still_a_presenter(self):
        presenter = person(0.32, 0.38, h=0.15)
        shot, followed = classify_shot([presenter], 8, SRC_W, SRC_H, location="top-left")
        assert shot.layout == LayoutType.TALKING_HEAD and followed is presenter

    def test_the_follower_stays_on_the_webcam_when_its_face_is_missed(self):
        from clip_engine.services.layout_analyzer import split_overlay_segments
        cam, hud = Box(0.84, 0.8, 0.04, 0.08), Box(0.08, 0.06, 0.04, 0.08)

        def frames(missed):
            return [FrameInfo(t_ms=i * 250, faces=([hud] if i in missed else [cam, hud]), hist=None) for i in range(40)]
        # One missed detection: never handed to the portrait.
        segments = split_overlay_segments(frames({10}), 0, 10_000, cam)
        assert len(segments) == 1 and segments[0][2].cx > 0.5
        # A longer miss: the portrait, a face that sits still all clip long, is excluded.
        segments = split_overlay_segments(frames({10, 11, 12}), 0, 10_000, cam, exclude=[hud])
        assert all(face is None or face.cx > 0.5 for _, _, face in segments)
        # A chosen place keeps the follower there too.
        segments = split_overlay_segments(frames({10, 11, 12}), 0, 10_000, cam, location="bottom-right")
        assert all(face is None or face.cx > 0.5 for _, _, face in segments)

    def test_game_faces_at_the_edges_do_not_block_fill_once_a_place_is_chosen(self):
        from clip_engine.services.layout_analyzer import shows_people
        side_face = Box(0.9, 0.45, 0.04, 0.08)
        frames = [FrameInfo(t_ms=i * 250, faces=[side_face], hist=None) for i in range(8)]
        shot = ShotLayout(0, 2000, LayoutType.SCREEN)
        assert shows_people(shot, frames)
        assert not shows_people(shot, frames, "bottom-right")
