"""Webcam position and screen fill: the Smart framing preferences from Settings."""

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clip_engine.config import LayoutStyle, Settings
from clip_engine.services.layout_analyzer import Box, ClipLayoutPlan, FrameInfo, LayoutAnalyzer, LayoutType, ShotLayout
from clip_engine.services.layout_renderer import (
    banner_y,
    caption_anchor,
    face_rects,
    foreground_geometry,
    panel_seam,
    screen_view,
    shot_chain,
    shot_views,
    stacked_panel_heights,
    title_y,
)

SRC_W, SRC_H, OUT_W, OUT_H = 1920, 1080, 1080, 1920
CAM = Box(0.75, 0.70, 0.22, 0.27)
CAM_FACE = Box(0.82, 0.74, 0.07, 0.12)


def screen_cam(position: str) -> ShotLayout:
    return ShotLayout(0, 5000, LayoutType.SCREEN_CAM, source="vision", screen_box=Box(0, 0, 1, 1),
                      cam_box=CAM, cam_face=CAM_FACE, cam_position=position)


def branch(graph: str, label: str) -> str:
    """The filter chain that starts at [label]."""
    return next(part for part in graph.split(";") if part.startswith(f"[{label}]"))


def crop_of(chain: str) -> tuple[int, int, int, int]:
    w, h, x, y = map(int, re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", chain).groups())
    return w, h, x, y


class TestWebcamPosition:
    def test_on_top_the_webcam_panel_comes_first_with_the_same_size_and_crops(self):
        screen_h, cam_h = stacked_panel_heights(screen_cam("top"), SRC_H, OUT_H)
        top, bottom = shot_chain(0, screen_cam("top"), SRC_W, SRC_H, OUT_W, OUT_H), shot_chain(0, screen_cam("bottom"), SRC_W, SRC_H, OUT_W, OUT_H)
        # [sa] is the screen and [sb] the webcam in both; only the stacking order changes.
        assert branch(top, "sb0").endswith(f"scale={OUT_W}:{cam_h}:flags=lanczos[top0]")
        assert branch(top, "sa0").endswith(f"scale={OUT_W}:{screen_h}:flags=lanczos[bot0]")
        assert branch(bottom, "sa0").endswith(f"scale={OUT_W}:{screen_h}:flags=lanczos[top0]")
        assert branch(bottom, "sb0").endswith(f"scale={OUT_W}:{cam_h}:flags=lanczos[bot0]")
        assert crop_of(branch(top, "sb0")) == crop_of(branch(bottom, "sb0"))
        assert crop_of(branch(top, "sa0")) == crop_of(branch(bottom, "sa0"))
        assert "[top0][bot0]vstack=inputs=2" in top and "[top0][bot0]vstack=inputs=2" in bottom

    def test_views_and_captions_follow_the_panels(self):
        screen_h, cam_h = stacked_panel_heights(screen_cam("top"), SRC_H, OUT_H)
        views = shot_views(screen_cam("top"), 0, SRC_W, SRC_H, OUT_W, OUT_H)
        assert [dest for _, dest in views] == [(0, 0, OUT_W, cam_h), (0, cam_h, OUT_W, screen_h)]
        cam_source = views[0][0]
        assert CAM.x * SRC_W <= cam_source[0] and cam_source[0] + cam_source[2] <= (CAM.x + CAM.w) * SRC_W + 2
        # Captions sit on the seam, which is now the bottom of the webcam panel.
        assert caption_anchor(screen_cam("top"), SRC_W, SRC_H, OUT_W, OUT_H) == (5, cam_h)
        assert caption_anchor(screen_cam("bottom"), SRC_W, SRC_H, OUT_W, OUT_H) == (5, screen_h)
        # The face in the webcam lands in the top panel, so captions keep off it there.
        [rect] = face_rects(screen_cam("top"), [CAM_FACE], 0, SRC_W, SRC_H, OUT_W, OUT_H)
        assert rect[3] <= cam_h
        [rect] = face_rects(screen_cam("bottom"), [CAM_FACE], 0, SRC_W, SRC_H, OUT_W, OUT_H)
        assert rect[1] >= screen_h

    def test_saved_plans_without_the_field_keep_the_webcam_at_the_bottom(self):
        shot = ShotLayout(0, 5000, LayoutType.SCREEN_CAM, cam_box=CAM, screen_box=Box(0, 0, 1, 1))
        assert shot.cam_position == "bottom"
        assert ShotLayout(**{**screen_cam("top").__dict__}).cam_position == "top"


class TestScreenFill:
    def test_a_filled_screen_is_one_full_frame_crop_with_no_blur(self):
        shot = ShotLayout(0, 5000, LayoutType.SCREEN, screen_fill=True)
        chain = shot_chain(0, shot, SRC_W, SRC_H, OUT_W, OUT_H)
        assert "gblur" not in chain and "overlay" not in chain
        assert chain.endswith(f"scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1[v0]")
        w, h, x, y = crop_of(chain)
        assert abs(w / h - OUT_W / OUT_H) < 0.01 and h == SRC_H
        assert abs(x + w / 2 - SRC_W / 2) <= 2, "no focus: the centre"
        assert shot_views(shot, 0, SRC_W, SRC_H, OUT_W, OUT_H) == [((x, y, w, h), (0, 0, OUT_W, OUT_H))]

    def test_the_crop_follows_the_area_the_vision_model_marked(self):
        shot = ShotLayout(0, 5000, LayoutType.SCREEN, source="vision", screen_box=Box(0, 0, 1, 1),
                          screen_focus=Box(0.02, 0.1, 0.2, 0.6), screen_fill=True)
        w, _, x, _ = crop_of(shot_chain(0, shot, SRC_W, SRC_H, OUT_W, OUT_H))
        assert x <= 0.02 * SRC_W + 2 and x + w >= 0.22 * SRC_W

    def test_padding_around_an_inset_never_shows(self):
        inset = Box(0.125, 0, 0.75, 1)  # 4:3 pillarboxed in 16:9
        shot = ShotLayout(0, 5000, LayoutType.SCREEN, content_box=inset, screen_fill=True)
        w, _, x, _ = crop_of(shot_chain(0, shot, SRC_W, SRC_H, OUT_W, OUT_H))
        assert x >= inset.x * SRC_W - 2 and x + w <= (inset.x + inset.w) * SRC_W + 2

    def test_overlays_use_full_frame_positions(self):
        filled = ShotLayout(0, 5000, LayoutType.SCREEN, screen_fill=True)
        whole = ShotLayout(0, 5000, LayoutType.SCREEN)
        assert foreground_geometry(filled, SRC_W, SRC_H, OUT_W, OUT_H) == (OUT_H, 0)
        assert caption_anchor(filled, SRC_W, SRC_H, OUT_W, OUT_H) == (5, 1340)
        assert title_y(filled, SRC_W, SRC_H, OUT_W, OUT_H, 120) == 110
        assert banner_y(filled, SRC_W, SRC_H, OUT_W, OUT_H) == 1590
        # Whole screens still put captions and the banner in the bars.
        assert caption_anchor(whole, SRC_W, SRC_H, OUT_W, OUT_H)[0] == 2

    def test_only_screens_without_a_webcam_or_manual_crops_fill(self):
        with_cam = ShotLayout(0, 5000, LayoutType.SCREEN_CAM, cam_box=CAM, screen_box=Box(0, 0, 1, 1), screen_fill=True)
        assert "vstack" in shot_chain(0, with_cam, SRC_W, SRC_H, OUT_W, OUT_H)
        manual = ShotLayout(0, 5000, LayoutType.SCREEN, manual_crops=[(0.1, 0.1, 0.3, 0.5)], screen_fill=True)
        assert shot_chain(0, manual, SRC_W, SRC_H, OUT_W, OUT_H).startswith("[t0]crop=576:540:192:108,")

    def test_a_filled_plan_is_not_a_letterbox_so_the_letterbox_fallback_stays(self):
        assert ClipLayoutPlan([ShotLayout(0, 5000, LayoutType.SCREEN)], SRC_W, SRC_H).is_letterbox_only
        assert not ClipLayoutPlan([ShotLayout(0, 5000, LayoutType.SCREEN, screen_fill=True)], SRC_W, SRC_H).is_letterbox_only


class TestAnalyzerAppliesTheSettings:
    def shots(self):
        return [ShotLayout(0, 1000, LayoutType.SCREEN),
                ShotLayout(1000, 2000, LayoutType.SCREEN_CAM, cam_box=CAM, screen_box=Box(0, 0, 1, 1)),
                ShotLayout(2000, 3000, LayoutType.TALKING_HEAD, people=[CAM_FACE])]

    def analyzer(self, **settings):
        analyzer = LayoutAnalyzer()
        analyzer.settings = SimpleNamespace(**settings)
        return analyzer

    def test_smart_framing_gets_the_position_and_fills_only_bare_screens(self):
        shots = self.shots()
        self.analyzer(webcam_position="top", screen_fill=True)._apply_framing_settings(shots, LayoutStyle.AUTO, SRC_W, SRC_H, [])
        assert [s.cam_position for s in shots] == ["top"] * 3
        assert [s.screen_fill for s in shots] == [True, False, False]

    def test_the_other_choices(self):
        shots = self.shots()
        self.analyzer(webcam_position="bottom", screen_fill=False)._apply_framing_settings(shots, LayoutStyle.AUTO, SRC_W, SRC_H, [])
        assert [s.cam_position for s in shots] == ["bottom"] * 3 and not any(s.screen_fill for s in shots)
        for style in (LayoutStyle.FILL, LayoutStyle.FIT):
            shots = self.shots()
            self.analyzer(webcam_position="top", screen_fill=True)._apply_framing_settings(shots, style, SRC_W, SRC_H, [])
            assert not any(s.screen_fill for s in shots), style

    def test_filled_screens_with_a_new_focus_stay_separate_shots(self):
        def screen(start, focus, fill):
            return ShotLayout(start, start + 1000, LayoutType.SCREEN, screen_box=Box(0, 0, 1, 1), screen_focus=focus, screen_fill=fill)

        left, right = Box(0, 0, .3, .5), Box(.6, 0, .3, .5)
        assert len(LayoutAnalyzer._merge_adjacent([screen(0, left, True), screen(1000, right, True)])) == 2
        assert len(LayoutAnalyzer._merge_adjacent([screen(0, left, True), screen(1000, left, True)])) == 1
        # Shown whole, the focus does not change the picture: one shot, as before.
        assert len(LayoutAnalyzer._merge_adjacent([screen(0, left, False), screen(1000, right, False)])) == 1

    def test_a_full_analysis_uses_the_settings(self):
        from tests.framing_fixture import recorded_fixture

        _, plan, _ = recorded_fixture()
        cams = [s for s in plan.shots if s.layout == LayoutType.SCREEN_CAM]
        assert cams and all(s.cam_position == "top" for s in cams)


class TestConfig:
    def test_defaults_and_unknown_values(self, monkeypatch):
        monkeypatch.delenv("WEBCAM_POSITION", raising=False)
        monkeypatch.delenv("SCREEN_FILL", raising=False)
        settings = Settings(_env_file=None)
        assert settings.webcam_position == "top" and settings.screen_fill is True
        assert Settings(_env_file=None, webcam_position="sideways").webcam_position == "top"

    def test_the_desktop_app_environment(self, monkeypatch):
        monkeypatch.setenv("WEBCAM_POSITION", "bottom")
        monkeypatch.setenv("SCREEN_FILL", "false")
        settings = Settings(_env_file=None)
        assert settings.webcam_position == "bottom" and settings.screen_fill is False


class TestReviewFixes:
    @pytest.mark.parametrize("src_w, src_h, screen", [
        (854, 480, Box(0, 0, 1, 1)),        # low-resolution source
        (1920, 1080, Box(0, 0.25, 1, 0.5)),  # a short screen area from vision
    ])
    def test_a_focused_crop_keeps_the_frame_shape(self, src_w, src_h, screen):
        shot = ShotLayout(0, 5000, LayoutType.SCREEN, screen_box=screen, screen_focus=Box(0.4, 0.3, 0.1, 0.2), screen_fill=True)
        w, h, _, _ = crop_of(shot_chain(0, shot, src_w, src_h, OUT_W, OUT_H))
        assert abs(w / h - OUT_W / OUT_H) < 0.02, (w, h)
        # The same rule keeps the screen panel of a screen + webcam shot in shape.
        stacked = ShotLayout(0, 5000, LayoutType.SCREEN_CAM, screen_box=screen, screen_focus=Box(0.4, 0.3, 0.1, 0.2),
                             cam_box=Box(0.8, 0.8, 0.15, 0.15))
        (w, h, _, _), (_, _, pw, ph) = screen_view(stacked, src_w, src_h, OUT_W, 1100)
        assert abs(w / h - pw / ph) < 0.02, (w, h)

    def test_a_fractional_inset_edge_never_lets_padding_in(self):
        inset = Box(0.1253, 0.0, 0.7494, 1.0)  # 240.6 px to 1679.4 px
        shot = ShotLayout(0, 5000, LayoutType.SCREEN, content_box=inset, screen_focus=Box(0.13, 0.2, 0.1, 0.5), screen_fill=True)
        w, h, x, y = crop_of(shot_chain(0, shot, SRC_W, SRC_H, OUT_W, OUT_H))
        assert x == 242 and x + w <= 1678 and w % 2 == 0 and h % 2 == 0

    @pytest.mark.parametrize("src_w, src_h, screen, fills", [
        (1280, 720, None, True),                     # 720p: 2.7x, like a talking head
        (854, 480, None, False),                     # 480p would be 4x
        (1920, 1080, Box(0.4, 0.4, 0.1, 0.1), False),  # a tiny screen area
    ])
    def test_screens_that_would_be_enlarged_too_far_stay_whole(self, src_w, src_h, screen, fills):
        shot = ShotLayout(0, 1000, LayoutType.SCREEN, screen_box=screen)
        TestAnalyzerAppliesTheSettings().analyzer(webcam_position="top", screen_fill=True)._apply_framing_settings(
            [shot], LayoutStyle.AUTO, src_w, src_h, [])
        assert shot.screen_fill is fills

    def test_shots_with_people_on_camera_stay_whole(self):
        panel = [Box(0.1, 0.3, 0.1, 0.18), Box(0.45, 0.3, 0.1, 0.18), Box(0.8, 0.3, 0.1, 0.18)]
        crowd = [FrameInfo(t, panel, None) for t in range(0, 2000, 250)]
        empty = [FrameInfo(t, [], None) for t in range(2000, 4000, 250)]
        shots = [ShotLayout(0, 2000, LayoutType.SCREEN), ShotLayout(2000, 4000, LayoutType.SCREEN)]
        TestAnalyzerAppliesTheSettings().analyzer(webcam_position="top", screen_fill=True)._apply_framing_settings(
            shots, LayoutStyle.AUTO, SRC_W, SRC_H, crowd + empty)
        assert [s.screen_fill for s in shots] == [False, True]

    def test_the_title_never_covers_the_face_of_a_webcam_on_top(self):
        title_h = 140
        y = title_y(screen_cam("top"), SRC_W, SRC_H, OUT_W, OUT_H, title_h)
        [face] = face_rects(screen_cam("top"), [CAM_FACE], 0, SRC_W, SRC_H, OUT_W, OUT_H)
        assert y >= face[3] or y + title_h <= face[1]
        # It sits below the captions on the seam, inside the frame.
        assert y >= panel_seam(screen_cam("top"), SRC_H, OUT_H) + 100 and y + title_h <= OUT_H
        # A webcam with room above the face keeps the usual place, as does a webcam at the bottom.
        low_face = ShotLayout(**{**screen_cam("top").__dict__, "cam_face": Box(0.82, 0.86, 0.05, 0.08)})
        assert title_y(low_face, SRC_W, SRC_H, OUT_W, OUT_H, title_h) == 110
        assert title_y(screen_cam("bottom"), SRC_W, SRC_H, OUT_W, OUT_H, title_h) == 110

    def test_filled_clips_are_labelled_as_filled(self):
        plan = ClipLayoutPlan([ShotLayout(0, 5000, LayoutType.SCREEN, screen_fill=True),
                               ShotLayout(5000, 6000, LayoutType.TALKING_HEAD)], SRC_W, SRC_H)
        assert plan.dominant_layout == "screen_fill"
        assert ClipLayoutPlan([ShotLayout(0, 5000, LayoutType.SCREEN)], SRC_W, SRC_H).dominant_layout == "screen"


def test_the_editor_suggests_the_webcam_split_and_filled_screens_as_rendered(tmp_path):
    from clip_engine.services.intelligence_planner import ClipPlanSegment
    from clip_engine.services.manual_editor import prepare_project
    from tests.test_coherence_review import reviewer, transcript

    gate, _ = reviewer(lambda state, q: False)
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source")
    plan = ClipLayoutPlan([
        ShotLayout(0, 2000, **{k: v for k, v in screen_cam("top").__dict__.items() if k not in ("start_ms", "end_ms")}),
        ShotLayout(2000, 3500, LayoutType.SCREEN, screen_fill=True),
        ShotLayout(3500, 5000, LayoutType.SCREEN),
    ], SRC_W, SRC_H)

    async def preview(src, dest, **kwargs):
        open(dest, "wb").write(b"preview")

    renderer = SimpleNamespace(_get_video_dimensions=AsyncMock(return_value=(SRC_W, SRC_H)),
                               capture_framing_source=AsyncMock(side_effect=preview), render_clip=AsyncMock(),
                               layout_analyzer=SimpleNamespace(analyze=AsyncMock(return_value=plan)))
    request = SimpleNamespace(aspect_ratio="9:16", layout_style="auto", include_captions=True, caption_preset="pop", video_speed=1)
    out = tmp_path / "run"
    out.mkdir()
    project = asyncio.run(prepare_project(request, [ClipPlanSegment(0, 5000, .9, summary="One")], transcript(),
                                          SimpleNamespace(video_path=str(source), metadata=SimpleNamespace(title="T", duration_seconds=12)),
                                          renderer, gate, str(out), lambda *_: None))
    scenes = project["candidates"][0]["scenes"]
    assert [s["layout"] for s in scenes] == ["split", "fill", "fit"]
    # The first crop of the split is the top panel: the webcam, on the right of the source.
    top, bottom = scenes[0]["crops"]
    assert top[0] > 0.5 > bottom[0]
