"""Local webcam-boundary precision and centered, bounded output crops."""
import os
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from clip_engine.services.layout_analyzer import Box, ClipLayoutPlan, LayoutAnalyzer, LayoutType, ShotLayout, refine_cam_box
from clip_engine.services.layout_renderer import MAX_UPSCALE, cam_crop, shot_views, stacked_panel_heights, build_layout_graph


def image_with_camera(x=481, y=241):
    image = np.full((360, 640, 3), 235, np.uint8)
    image[y:, x:] = (30, 65, 95)
    # Some interior detail; avoid treating every straight object as an edge.
    cv2.rectangle(image, (x + 30, y + 20), (x + 70, y + 55), (80, 110, 170), -1)
    return image


def test_snaps_approximate_bounds_inside_actual_camera():
    image = image_with_camera()
    guess = Box(.748, .66, .252, .34)
    face = Box(.88, .77, .045, .1)
    actual = refine_cam_box(image, guess, face)
    assert actual is not None
    assert actual.x * 640 >= 481 and actual.y * 360 >= 241
    assert actual.x * 640 <= 484 and actual.y * 360 <= 244
    assert actual.x + actual.w == 1 and actual.y + actual.h == 1
    assert actual.contains(face.x + face.w, face.y + face.h)


def test_ambiguous_images_and_edges_crossing_face_do_not_change_bounds():
    guess = Box(.748, .66, .252, .34)
    assert refine_cam_box(np.full((360, 640, 3), 100, np.uint8), guess) is None
    noise = np.random.default_rng(9).integers(0, 256, (360, 640, 3), np.uint8)
    assert refine_cam_box(noise, guess) is None
    assert refine_cam_box(image_with_camera(), guess, Box(.74, .75, .08, .1)) is None


def test_image_edges_hold_static_camera_when_subject_leans():
    original = Box(.748, .66, .252, .34)
    face = Box(.86, .76, .05, .10)
    # Face-based transfer incorrectly enlarged the stationary overlay.
    shot = ShotLayout(0, 4000, LayoutType.SCREEN_CAM, cam_box=Box(.70, .61, .3, .39), cam_face=face)
    _, jpg = cv2.imencode('.jpg', image_with_camera())
    LayoutAnalyzer._refine_webcam_regions([shot], [(0, jpg.tobytes()), (1000, jpg.tobytes()), (2000, jpg.tobytes())], original)
    assert shot.cam_box_refined
    assert shot.cam_box.x > .75 and shot.cam_box.y > .67


def test_real_overlay_move_uses_current_edges_not_original_region():
    original = Box(.748, .66, .252, .34)
    moved = Box(.62, .55, .38, .45)
    shot = ShotLayout(2000, 4000, LayoutType.SCREEN_CAM, cam_box=moved, cam_face=Box(.83, .69, .06, .13))
    _, jpg = cv2.imencode('.jpg', image_with_camera(400, 200))
    LayoutAnalyzer._refine_webcam_regions([shot], [(2000, jpg.tobytes()), (3000, jpg.tobytes())], original)
    assert shot.cam_box_refined
    assert 400 <= shot.cam_box.x * 640 <= 403
    assert 200 <= shot.cam_box.y * 360 <= 203


def test_refinement_requires_temporal_agreement_and_keeps_uncertain_box():
    original = Box(.748, .66, .252, .34)
    shot = ShotLayout(0, 3000, LayoutType.SCREEN_CAM, cam_box=original)
    frames = []
    for t, image in [(0, image_with_camera()), (1000, np.full((360, 640, 3), 100, np.uint8)), (2000, np.full((360, 640, 3), 100, np.uint8))]:
        _, jpg = cv2.imencode('.jpg', image)
        frames.append((t, jpg.tobytes()))
    LayoutAnalyzer._refine_webcam_regions([shot], frames, original)
    assert shot.cam_box == original and not shot.cam_box_refined


@pytest.mark.parametrize('cam', [Box(.755, .671, .245, .329), Box(0, 0, .245, .329), Box(.1023, .2261, .3279, .4213)])
def test_pixel_rounding_never_expands_beyond_camera(cam):
    w, h, x, y = cam_crop(cam, None, 1920, 1080, 1080, 768)
    assert x >= cam.x * 1920 and y >= cam.y * 1080
    assert x + w <= (cam.x + cam.w) * 1920
    assert y + h <= (cam.y + cam.h) * 1080
    assert all(v % 2 == 0 for v in (x, y, w, h))
    if cam.x == 0:
        assert x == 0


def output_face_x(shot, face):
    (sx, _, sw, _), (dx, _, dw, _) = shot_views(shot, 0, 1920, 1080, 1080, 1920)[1]
    return dx + (face.cx * 1920 - sx) / sw * dw


def test_offcenter_subject_moves_toward_center_within_upscale_budget():
    cam, face = Box(.7578125, .672222, .2421875, .327778), Box(.8815, .7753, .0425, .0998)
    w, h, x, y = cam_crop(cam, face, 1920, 1080, 1080, 768)
    untightened = cam_crop(cam, None, 1920, 1080, 1080, 768)
    # Fully centering needs 2.9x; the crop tightens only to MAX_UPSCALE.
    assert 1080 / w <= MAX_UPSCALE + .01 and w < untightened[0]
    assert w / h == pytest.approx(1080 / 768, rel=.02)
    shot = ShotLayout(0, 1000, LayoutType.SCREEN_CAM, cam_box=cam, cam_face=face)
    assert shot_views(shot, 0, 1920, 1080, 1080, 1920)[1][1] == (0, 1152, 1080, 768)
    loose = ShotLayout(0, 1000, LayoutType.SCREEN_CAM, cam_box=cam)
    assert abs(output_face_x(shot, face) - 540) < abs(output_face_x(loose, face) - 540)


def test_offcenter_subject_is_centered_when_the_budget_allows():
    cam, face = Box(.6, .5, .4, .5), Box(.86, .62, .05, .12)
    w, h, x, y = cam_crop(cam, face, 1920, 1080, 1080, 768)
    assert abs((face.cx * 1920 - x) / w - .5) < .01
    assert 1080 / w <= MAX_UPSCALE


@pytest.mark.parametrize('cam', [
    Box(.7865, .7639, .1635, .2102),  # ~314x227 px overlay (DevDay stream)
    Box(.73, .68, .22, .29),          # ~420x313 px overlay
])
def test_small_webcam_with_offcenter_face_fills_the_whole_panel(cam):
    # Regression: these were shown at MAX_UPSCALE, centered over a blurred copy.
    face = Box(cam.x + cam.w * .62, cam.y + cam.h * .25, cam.w * .2, cam.h * .32)
    shot = ShotLayout(0, 1000, LayoutType.SCREEN_CAM, cam_box=cam, cam_face=face, screen_box=Box(0, 0, 1, 1))
    top_h, bottom_h = stacked_panel_heights(shot, 1080, 1920)
    w, h, x, y = cam_crop(cam, face, 1920, 1080, 1080, bottom_h)
    assert w / h == pytest.approx(1080 / bottom_h, rel=.02)
    assert x >= cam.x * 1920 and y >= cam.y * 1080
    assert x + w <= (cam.x + cam.w) * 1920 and y + h <= (cam.y + cam.h) * 1080
    assert shot_views(shot, 0, 1920, 1080, 1080, 1920)[1] == ((x, y, w, h), (0, top_h, 1080, bottom_h))
    graph = build_layout_graph(ClipLayoutPlan([shot], 1920, 1080), 1080, 1920)
    assert 'gblur' not in graph.split('[bot0]')[0].rsplit('[sb0]', 1)[1]
    assert f'[sb0]crop={w}:{h}:{x}:{y},scale=1080:{bottom_h}:flags=lanczos[bot0]' in graph


def test_rendered_webcam_panel_contains_no_surrounding_page(tmp_path):
    ffmpeg = os.environ.get('TEST_FFMPEG') or shutil.which('ffmpeg')
    if not ffmpeg:
        pytest.skip('ffmpeg unavailable')
    image = image_with_camera()
    face = Box(.88, .77, .045, .1)
    cam = refine_cam_box(image, Box(.748, .66, .252, .34), face)
    shot = ShotLayout(0, 1000, LayoutType.SCREEN_CAM, cam_box=cam, cam_face=face, screen_box=Box(0, 0, 1, 1))
    plan = ClipLayoutPlan([shot], 640, 360)
    raw = tmp_path / 'camera.rgb'
    raw.write_bytes(image.tobytes() * 4)
    result = subprocess.run([
        ffmpeg, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x360',
        '-r', '4', '-i', str(raw), '-filter_complex', build_layout_graph(plan, 180, 320, fps='4'),
        '-map', '[base]', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1',
    ], capture_output=True, check=True, timeout=30)
    output = np.frombuffer(result.stdout, np.uint8).reshape(-1, 320, 180, 3)
    assert len(output) == 4
    for frame in output:
        assert not (frame[192:].min(axis=2) > 210).any(), 'white webpage leaked into camera panel'
