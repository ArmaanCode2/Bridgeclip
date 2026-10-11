"""Every filter our graphs emit must exist in the shipped LGPL FFmpeg.

CI runs the engine suite with GPL FFmpeg builds, where GPL-only filters such as
`perspective` work, so a render test alone cannot catch them. This check needs
no binary: it compares emitted filter names with an allowlist taken from the
shipped build (engine-bin, FFmpeg 8.1.3 LGPL), and cross-checks that allowlist
against engine-bin when it is present.
"""
import asyncio
import re
import subprocess
from pathlib import Path

import pytest

from clip_engine.services import rendering_service
from clip_engine.services.camera_scan import scan_camera_changes
from clip_engine.services.layout_analyzer import Box, ClipLayoutPlan, LayoutType, ShotLayout
from clip_engine.services.layout_renderer import LOUDNESS_FILTER, build_layout_graph
from clip_engine.services.manual_editor import manual_plan
from clip_engine.services.rendering_service import RenderingService
from clip_engine.services.video_speed import speed_audio_filter, speed_video_filter

FIXTURE = Path(__file__).parent / 'fixtures' / 'lgpl_ffmpeg_filters.txt'
ENGINE_BIN = Path(__file__).resolve().parents[2] / 'engine-bin' / 'ffmpeg'
# Filters FFmpeg's configure marks `gpl`; never available in shipped builds.
GPL_ONLY = {'blackframe', 'boxblur', 'colormatrix', 'cover_rect', 'cropdetect', 'delogo', 'eq', 'find_rect',
            'fspp', 'histeq', 'hqdn3d', 'interlace', 'kerndeint', 'mcdeint', 'mpdecimate', 'owdenoise',
            'perspective', 'phase', 'pp', 'pp7', 'pullup', 'repeatfields', 'sab', 'signature', 'smartblur',
            'spp', 'stereo3d', 'super2xsai', 'tinterlace', 'uspp', 'vaguedenoiser'}


def allowlist():
    return {line.strip() for line in FIXTURE.read_text().splitlines() if line.strip() and not line.startswith('#')}


def filter_names(graph):
    """Filter names in a filtergraph, honoring quotes, escapes and [labels]."""
    names, token, quoted, escaped = [], '', False, False
    for char in graph + ';':
        if escaped:
            token += char; escaped = False
        elif char == '\\':
            token += char; escaped = True
        elif char == "'":
            token += char; quoted = not quoted
        elif char in ',;' and not quoted:
            head = re.sub(r'^(\s*\[[^\]]*\])*\s*', '', token)
            name = re.match(r'[A-Za-z0-9_]+', head)
            if name:
                names.append(name.group(0))
            token = ''
        else:
            token += char
    return names


def manual(scenes, ranges=((1000, 4000), (5000, 9000))):
    return manual_plan({'width': 1920, 'height': 1080}, {'ranges': [list(r) for r in ranges], 'scenes': scenes})


def graphs():
    moving = manual([{'at_ms': 0, 'layout': 'fill', 'crops': [[0, 0, .3164, 1]]},
                     {'at_ms': 2000, 'layout': 'fill', 'crops': [[.5, .5, .1, .3]], 'transition_ms': 1500},
                     {'at_ms': 6000, 'layout': 'split', 'crops': [[0, 0, .5, .5], [.5, .5, .5, .5]]},
                     {'at_ms': 7000, 'layout': 'split', 'crops': [[.1, .1, .4, .4], [0, 0, .6, .6]], 'transition_ms': 800},
                     {'at_ms': 8500, 'layout': 'fit', 'crops': [[0, 0, 1, 1]]}])
    face, cam = Box(.4, .2, .1, .2), Box(.75, .7, .2, .25)
    automatic = ClipLayoutPlan([
        ShotLayout(0, 1000, LayoutType.TALKING_HEAD, focus_path=[(0, .4, .3), (900, .6, .3)]),
        ShotLayout(1000, 2000, LayoutType.TWO_SHOT, people=[Box(.1, .2, .1, .2), Box(.7, .2, .1, .2)]),
        ShotLayout(2000, 3000, LayoutType.SCREEN_CAM, screen_box=Box(0, 0, .7, 1), cam_box=cam, cam_face=face),
        ShotLayout(3000, 4000, LayoutType.SCREEN_CAM, screen_box=Box(0, 0, .7, 1), cam_box=Box(.9, .9, .05, .05)),
        ShotLayout(4000, 5000, LayoutType.SCREEN),
        ShotLayout(5000, 6000, LayoutType.SCREEN, content_box=Box(.1, .1, .6, .6)),
    ], 1920, 1080)
    for plan in (moving, automatic):
        for speed in (1, 1.5):
            yield build_layout_graph(plan, 1080, 1920, [(0, 2500), (3000, 6000)], True, fps='30000/1001',
                                     loudness_filter=LOUDNESS_FILTER, video_speed=speed)
    for size in ((1920, 1080), (1440, 1080)):
        yield build_layout_graph(ClipLayoutPlan([ShotLayout(0, 3000, LayoutType.SCREEN)], *size), 1920, 1080,
                                 None, True, landscape=True, fps='60')
    yield speed_video_filter(1.5, '30') + ';' + speed_audio_filter(1.5, 3000)


def rendering_graphs(tmp_path):
    renderer = RenderingService.__new__(RenderingService)
    renderer._fonts_dir = str(tmp_path)
    captions = tmp_path / 'captions.ass'
    captions.write_text('')
    time_map = type('Identity', (), {'to_output_clamped': staticmethod(lambda ms: ms)})()
    graph = renderer._caption_graph(str(captions), [(i * 200, i * 200 + 100) for i in range(40)], 0, time_map)
    fade = 'format=rgba,fade=t=in:st=0.4:d=0.4:alpha=1,fade=t=out:st=5.1:d=0.4:alpha=1'
    composed, _ = RenderingService._compose_overlays(graph, [('a.png', '0', '0'), ('b.png', '10', '20', 'between(t,0.4,5.5)', fade)],
                                                     speed_video_filter(2, '30'))
    yield composed


def captured_commands(tmp_path, monkeypatch):
    """Command-line -vf/-af graphs of the editor preview and camera scan."""
    commands = []
    renderer = RenderingService.__new__(RenderingService)
    renderer.settings = type('Settings', (), {'local_mode': True, 'ffmpeg_preset': 'fast', 'ffmpeg_crf': 20})()

    async def dimensions(_):
        return 1920, 1080

    async def fps(_):
        return '30'

    async def run(cmd, **_):
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b'')
    monkeypatch.setattr(renderer, '_get_video_dimensions', dimensions, raising=False)
    monkeypatch.setattr(renderer, '_probe_fps', fps, raising=False)
    monkeypatch.setattr(renderer, '_run_cmd', run, raising=False)
    asyncio.run(renderer.capture_framing_source('in.mp4', str(tmp_path / 'preview.mp4')))

    class Scan:
        def __init__(self, cmd, **_):
            commands.append(cmd)

        def __enter__(self):
            raise RuntimeError('captured')

        def __exit__(self, *_):
            return False
    monkeypatch.setattr('clip_engine.services.camera_scan.media_process', Scan)
    with pytest.raises(RuntimeError, match='captured'):
        scan_camera_changes('in.mp4', 0, 1000)
    for cmd in commands:
        for flag in ('-vf', '-af'):
            if flag in cmd:
                yield cmd[cmd.index(flag) + 1]


def test_emitted_filters_are_in_the_shipped_lgpl_build(tmp_path, monkeypatch):
    allowed = allowlist()
    assert not allowed & GPL_ONLY
    emitted = set()
    for graph in [*graphs(), *rendering_graphs(tmp_path), *captured_commands(tmp_path, monkeypatch)]:
        emitted.update(filter_names(graph))
    # Guard the parser itself: these must be seen, or the check is vacuous.
    assert {'scale', 'crop', 'trim', 'concat', 'overlay', 'ass', 'loudnorm', 'atempo', 'select'} <= emitted
    assert emitted <= allowed, f'Not in the shipped LGPL FFmpeg: {sorted(emitted - allowed)}'


def test_filter_parser_handles_quoted_expressions():
    graph = "[a]crop=w=2:h=2:x='clip(t,0,1)':y='if(lt(t\\,1),0,1)'[b];[b]split=2[c][d];[c][d]vstack=inputs=2,setsar=1"
    assert filter_names(graph) == ['crop', 'split', 'vstack', 'setsar']


@pytest.mark.skipif(not ENGINE_BIN.exists(), reason='engine-bin/ffmpeg is not staged')
def test_allowlist_matches_engine_bin():
    output = subprocess.run([str(ENGINE_BIN), '-hide_banner', '-filters'], capture_output=True, text=True,
                            check=True, timeout=30).stdout
    available = {m.group(1) for m in re.finditer(r'^\s*[.TSC|]{2,3}\s+(\S+)\s+\S+->\S+', output, re.M)}
    assert allowlist() <= available, sorted(allowlist() - available)
    assert not GPL_ONLY & available, sorted(GPL_ONLY & available)
