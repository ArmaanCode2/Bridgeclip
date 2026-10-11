"""Real decoded frames exercise the pipeline's new refinement pass."""
import asyncio
import math
import numpy as np
import pytest
from clip_engine.services.layout_analyzer import LayoutAnalyzer, Box, FrameInfo, frame_layout_evidence
from clip_engine.services.layout_precision import DetailSelector, align_boundaries, confirmed_cuts
from .test_camera_scan import source_video, ffmpeg, available_encoder  # noqa: F401 (fixture)


def frame(t, x=.2, faces=True):
    return FrameInfo(t, [Box(x, .2, .15, .3)] if faces else [], np.zeros((24, 16), np.float32))


def test_weak_visual_hint_needs_sustained_abrupt_face_change():
    marker = [{'at_ms': 500, 'score': .05}]
    stable = [frame(450), *[frame(t, .65) for t in (500, 550, 600, 700, 800)]]
    assert confirmed_cuts(marker, stable, 1000, frame_layout_evidence) == [500]
    moving = [frame(450), *[frame(t, .2 + i * .04) for i, t in enumerate((500, 550, 600, 700, 800))]]
    assert confirmed_cuts(marker, moving, 1000, frame_layout_evidence) == []
    dropout = [frame(450), frame(500, faces=False), frame(550), frame(700), frame(800)]
    assert confirmed_cuts(marker, dropout, 1000, frame_layout_evidence) == []


def test_flash_pair_is_not_an_automatic_layout_change():
    assert confirmed_cuts([{'at_ms': 500, 'score': .4}, {'at_ms': 550, 'score': .4}], [], 1000, frame_layout_evidence) == []


def selected(selector, times, scores):
    chosen, previous = [], None
    for t in times:
        take_previous, take_current, _ = selector.select(t, scores.get(t), previous)
        chosen += [previous] * take_previous + [t] * take_current
        previous = t
    return chosen


def test_detail_frames_are_the_ones_that_decide_each_change():
    times = [round(i * 1000 / 30, 3) for i in range(90)]
    strong, weak = times[20], times[50]
    chosen = selected(DetailSelector(3000), times, {strong: .4, weak: .05, times[70]: .01})
    # A strong cut is accepted on its score: only its first frame (face samples
    # and a keyframe for the new shot). A weak one needs face evidence: the
    # frame before it and >= 3 frames spanning >= 180 ms within 350 ms.
    after = [t for t in chosen if weak <= t <= weak + 350]
    assert chosen == [strong, times[49], *after]
    assert len(after) == 3 and after[-1] - after[0] >= 180


def test_detail_budget_caps_instead_of_raising():
    times = [i * 10 for i in range(10000)]
    selector = DetailSelector(100000, budget=40)
    chosen = selected(selector, times, {t: .05 for t in times[1::30]})
    assert len(chosen) == 40 and selector.capped
    # Markers beyond the budget still align boundaries and strong cuts still count.
    markers = [{'at_ms': 5000, 'score': .4}, {'at_ms': 7000, 'score': .03}]
    assert confirmed_cuts(markers, [], 10000, frame_layout_evidence) == [5000]
    assert align_boundaries([7100], markers, 10000) == [0, 7000, 10000]


def test_cut_confirmation_stays_fast_for_long_windows():
    import random
    import time
    duration = 60 * 60 * 1000
    frames = [frame(t) for t in range(0, duration, 250)]
    random.seed(1)
    markers = sorted(({'at_ms': random.uniform(0, duration), 'score': random.uniform(.025, .1)} for _ in range(5000)),
                     key=lambda m: m['at_ms'])
    started = time.perf_counter()
    confirmed_cuts(markers, frames, duration, frame_layout_evidence)
    align_boundaries(list(range(0, duration, 5000)), markers, duration)
    assert time.perf_counter() - started < 1


@pytest.mark.parametrize('start', [0, 217, 503])
@pytest.mark.parametrize('vfr', [False, True])
def test_pipeline_refines_real_fractional_frames_and_retains_scan(tmp_path, monkeypatch, start, vfr, available_encoder):
    source = source_video(tmp_path, vfr=vfr)
    analyzer = LayoutAnalyzer()
    monkeypatch.setattr(analyzer, '_get_detector', lambda *_: None)
    original = analyzer._frame_info
    class NoFaces:
        def detect(self, image): return None, None
    def observed(image, t, detector, width, height):
        result = original(image, t, NoFaces(), width, height)
        # Deterministic face cues isolate timestamp correctness from ML accuracy.
        b, g, r = image[height // 2, width // 4].astype(int)
        result.faces = [Box(.65 if r > 200 and g > 200 else .2, .2, .15, .3)]
        result.content_box = None
        return result
    monkeypatch.setattr(analyzer, '_frame_info', observed)
    updates = []
    plan = asyncio.run(analyzer.analyze(str(source), start, 2400-start, 160, 90, vision=False, precise=True, capture=True,
                                        progress=lambda message, value: updates.append((message, value))))
    assert plan.camera_scan
    cuts = [m['at_ms'] - start for m in plan.camera_scan['markers'] if start < m['at_ms'] < 2400]
    assert len(cuts) == 2
    accepted = [b['t_ms'] for b in plan.trace['boundaries'] if b['kind'] == 'precise_scene']
    assert accepted == pytest.approx(cuts, abs=.002)
    for cut in cuts:
        sample = next(f for f in plan.face_samples if abs(f[0]-cut) < .002)
        assert sample[1][0].x == (.65 if cut == cuts[0] else .2)
    percents = [value for message, value in updates if message == 'Sampling faces']
    assert percents and all(isinstance(p, int) for p in percents) and percents == sorted(percents)
    if start == 0 and not vfr:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from clip_engine.services.manual_editor import prepare_project, validate_candidate
        from clip_engine.services.rendering_service import RenderingService
        from clip_engine.services.intelligence_planner import ClipPlanSegment
        from .test_manual_editor import reviewer
        gate, _ = reviewer(lambda state, q: True)
        renderer = RenderingService()
        monkeypatch.setattr(renderer.layout_analyzer, 'analyze', AsyncMock(return_value=plan))
        run = tmp_path / 'run'
        run.mkdir()
        request = SimpleNamespace(aspect_ratio='9:16', layout_style='auto', include_captions=True, caption_preset='pop', video_speed=1)
        project = asyncio.run(prepare_project(request, [ClipPlanSegment(0, 2400, .9)], [],
            SimpleNamespace(video_path=str(source), metadata=SimpleNamespace(title='Test', duration_seconds=2.4)),
            renderer, gate, str(run), lambda *_: None))
        candidate = project['candidates'][0]
        assert candidate['camera_scan'] == plan.camera_scan
        assert candidate['dismissed_camera_markers'] == []
        assert [scene['at_ms'] for scene in candidate['scenes'][1:]] == pytest.approx(cuts, abs=.002)
        validate_candidate(candidate, 2400)

    from clip_engine.services.layout_renderer import build_layout_graph
    graph = build_layout_graph(plan, 40, 44, fps='24000/1001')
    args = ['-ss', str(start/1000), '-t', str((2400-start)/1000), '-i', str(source)]
    raw = ffmpeg([*args, '-filter_complex', graph, '-map', '[base]', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1'])
    output = np.frombuffer(raw, np.uint8).reshape(-1, 44, 40, 3)
    raw = ffmpeg([*args, '-vf', 'fps=24000/1001:start_time=0:round=near', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1'])
    reference = np.frombuffer(raw, np.uint8).reshape(-1, 90, 160, 3)
    for i, image in enumerate(reference[:len(output)]):
        left = image[45, 40].astype(int)
        expected = image[45, 120] if left[0] > 200 and left[1] > 200 else left
        assert output[i, 22, 20] == pytest.approx(expected.astype(int), abs=12), (i, start, vfr)



def test_vision_budget_still_allows_cached_layouts(monkeypatch):
    import cv2
    from clip_engine.services.layout_analyzer import ShotLayout, LayoutType
    analyzer = LayoutAnalyzer()
    image = np.full((90, 160, 3), 100, np.uint8)
    _, encoded = cv2.imencode('.jpg', image)
    hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2HSV)], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    shot = ShotLayout(0, 1000, LayoutType.SCREEN)
    answer = {'layout': 'screen'}
    analyzer._vision_cache = [(hist, (LayoutType.SCREEN, ()), answer, {})]
    diagnostic = {}
    assert asyncio.run(analyzer._vision_classify(encoded.tobytes(), [], shot, diagnostic, cache_only=True)) == (answer, 0)
    assert diagnostic['cache_hit'] is True
    analyzer._vision_cache.clear()
    assert asyncio.run(analyzer._vision_classify(encoded.tobytes(), [], shot, diagnostic, cache_only=True)) == (None, 0)
    assert diagnostic['status'] == 'budget_limited'


def test_analysis_bounds_optional_paid_decisions_for_many_short_shots(monkeypatch):
    analyzer = LayoutAnalyzer()
    frames = [frame(t, .2 if (t // 500) % 2 else .65) for t in range(0, 8000, 50)]
    images = [(t, b'fake') for t in range(0, 8000, 500)]
    scan = {'frames': [f.t_ms for f in frames], 'markers': [{'at_ms': t, 'score': .2} for t in range(500, 8000, 500)]}
    monkeypatch.setattr(analyzer, '_decode_and_detect', lambda *_: (frames, images))
    monkeypatch.setattr(analyzer, '_precise_frames', lambda *_: (scan, frames, images))
    monkeypatch.setattr(analyzer, '_vision_enabled', lambda: True)
    modes = []
    async def classify(*_, cache_only=False):
        modes.append(cache_only)
        return None, 0
    monkeypatch.setattr(analyzer, '_vision_classify', classify)
    asyncio.run(analyzer.analyze('fixture.mp4', 0, 8000, 160, 90))
    assert len(modes) == 16
    assert modes == [False] * 12 + [True] * 4


@pytest.mark.parametrize('fps', ['24000/1001', '60'])
@pytest.mark.parametrize('vfr', [False, True])
def test_one_bounded_decode_yields_samples_scan_and_cut_details(tmp_path, monkeypatch, fps, vfr):
    """Sampling, the every-frame scan and cut details share one input-bounded decode."""
    from contextlib import contextmanager
    from clip_engine.services import layout_analyzer as module
    from clip_engine.services.camera_scan import scan_camera_changes
    source = source_video(tmp_path, fps, vfr=vfr)
    analyzer = LayoutAnalyzer()
    class NoFaces:
        def detect(self, image): return None, None
    monkeypatch.setattr(analyzer, '_get_detector', lambda *_: NoFaces())
    original = module.media_process
    commands = []
    @contextmanager
    def bounded(cmd, **kwargs):
        commands.append(cmd)
        with original(cmd, **kwargs) as process:
            yield process
    monkeypatch.setattr(module, 'media_process', bounded)
    # The 60-frame fixture lasts 2.5 s at 24 fps but 1 s at 60 fps.
    start, duration = (500, 1700) if fps != '60' else (100, 850)
    scan, frames, keyframes = analyzer._precise_frames(str(source), start, duration, 160, 90)
    assert len(commands) == 1
    cmd = commands[0]
    assert cmd.index('-t') < cmd.index('-i')
    assert float(cmd[cmd.index('-t') + 1]) == pytest.approx((start + duration) / 1000)
    # The scan is exactly what the editor's standalone scan reports.
    assert scan == scan_camera_changes(source, start, start + duration)
    cuts = [m['at_ms'] - start for m in scan['markers'] if start < m['at_ms'] < start + duration]
    assert len(cuts) == 2
    times = [f.t_ms for f in frames]
    in_window = [round(t - start, 3) for t in scan['frames'] if t >= start]
    assert all(t in in_window for t in times), 'every analyzed frame carries its exact time'
    # The first frame of each change is analyzed. Above ~40 fps only every
    # other frame is streamed, so it may be the next one.
    step = 1000 / 60 + .002 if fps == '60' else .002
    assert all(any(-.002 < t - cut < step for t in times) for cut in cuts)
    if fps == '60':
        # Only every other frame crosses the pipe; the scan still has them all.
        assert min(b - a for a, b in zip(times, times[1:])) > 1.5 * 1000 / 60
    # About ANALYSIS_FPS samples per second plus the first frame of each cut.
    samples = math.ceil(duration / 250)
    assert samples <= len(frames) <= samples + 2 * len(cuts)
    assert any(-.002 < t - cuts[0] < step for t, _ in keyframes)


def test_window_ending_on_a_cut_keeps_the_scan(tmp_path, monkeypatch):
    """A frame exactly at the window end belongs to the next window, in both passes."""
    from clip_engine.services.camera_scan import scan_camera_changes
    source = source_video(tmp_path, fps='30')
    full = scan_camera_changes(source, 0, 3000)
    end = full['markers'][-1]['at_ms']
    assert end in full['frames']
    clipped = scan_camera_changes(source, 0, end)
    assert clipped['frames'] == [t for t in full['frames'] if t < end]
    analyzer = LayoutAnalyzer()
    class NoFaces:
        def detect(self, image): return None, None
    monkeypatch.setattr(analyzer, '_get_detector', lambda *_: NoFaces())
    scan, frames, _ = analyzer._precise_frames(str(source), 0, end, 160, 90)
    assert scan == clipped
    assert frames and max(f.t_ms for f in frames) < end


def test_precise_pass_runs_only_when_framing_needs_exact_cuts(monkeypatch):
    analyzer = LayoutAnalyzer()
    frames = [frame(t) for t in range(0, 4000, 250)]
    calls = []
    def precise(*_):
        calls.append('precise')
        return None, frames, []
    monkeypatch.setattr(analyzer, '_precise_frames', precise)
    monkeypatch.setattr(analyzer, '_decode_and_detect', lambda *_: (calls.append('sampled'), (frames, []))[1])
    for kwargs, expected in [({}, 'precise'), ({'vision': False}, 'sampled'),
                             ({'vision': False, 'precise': True}, 'precise')]:
        calls.clear()
        asyncio.run(analyzer.analyze('fixture.mp4', 0, 4000, 160, 90, **kwargs))
        assert calls == [expected]


def test_render_progress_omits_unknown_percentages(monkeypatch):
    from clip_engine.services.rendering_service import RenderRequest, RenderingService
    service = RenderingService()
    async def analyze(*_, progress=None, **__):
        progress('Sampling faces', None)
        progress('Sampling faces', 40)
    monkeypatch.setattr(service.layout_analyzer, 'analyze', analyze)
    messages = []
    request = RenderRequest(video_path='src.mp4', output_path='clip.mp4', start_time_ms=0, end_time_ms=1000,
                            source_width=160, source_height=90, progress_callback=lambda text, _: messages.append(text))
    asyncio.run(service._plan_layout(request, 160, 90, 0, 1000))
    assert messages == ['Sampling faces', 'Sampling faces 40%']


def test_long_window_keeps_keyframes_to_its_end():
    """Thin evenly across time; never truncate the end of a long window."""
    from clip_engine.services.layout_analyzer import MAX_RETAINED_KEYFRAMES, retain_keyframes
    regular = [(t, b'r') for t in range(0, 300_000, 1250)]
    cuts = [(t + 10, b'c') for t in range(0, 300_000, 1000)]
    kept = retain_keyframes(regular, cuts)
    assert len(kept) == MAX_RETAINED_KEYFRAMES
    assert kept[0][0] <= 1000 and kept[-1][0] >= 298_000
    gaps = [b[0] - a[0] for a, b in zip(kept, kept[1:])]
    assert max(gaps) < 3000
    # Every cut image survives when there is room for them.
    assert retain_keyframes(regular[:100], cuts[:150]) == sorted(regular[:100] + cuts[:150])
    # The byte budget thins evenly too.
    heavy = retain_keyframes([(t, b'x' * 1000) for t in range(0, 300_000, 1250)], [], max_bytes=64_000)
    assert len(heavy) <= 64 and heavy[-1][0] >= 290_000


def test_real_five_minute_window_keeps_late_cut_keyframes(tmp_path, monkeypatch):
    fps, seconds = 5, 300
    frames = np.zeros((fps * seconds, 90, 160, 3), np.uint8)
    colors = np.array([(200, 40, 40), (40, 200, 40), (40, 40, 200), (200, 200, 40), (40, 200, 200)], np.uint8)
    frames[:] = colors[(np.arange(fps * seconds) // fps) % len(colors)][:, None, None, :]
    source = tmp_path / 'long.mkv'
    ffmpeg(['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '160x90', '-r', str(fps), '-i', 'pipe:0',
            '-c:v', 'mpeg4', '-q:v', '2', str(source)], frames.tobytes())
    analyzer = LayoutAnalyzer()
    class NoFaces:
        def detect(self, image): return None, None
    monkeypatch.setattr(analyzer, '_get_detector', lambda *_: NoFaces())
    scan, analyzed, keyframes = analyzer._precise_frames(str(source), 0, seconds * 1000, 160, 90)
    assert len([m for m in scan['markers'] if m['score'] >= .15]) >= seconds - 2
    assert len(keyframes) <= 256
    assert keyframes[-1][0] >= (seconds - 5) * 1000, 'keyframes after ~251 s were dropped'
    assert max(f.t_ms for f in analyzed) >= (seconds - 1) * 1000
