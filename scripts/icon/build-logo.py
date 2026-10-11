"""Builds the BridgeClip logo lockup: the BridgeMind mark beside the wordmark,
with the wordmark set in Geist SemiBold and converted to outlines, so the SVGs
render the same everywhere without the font installed.

  resources/bridgeclip-logo.svg        for dark backgrounds
  resources/bridgeclip-logo-light.svg  for light backgrounds

"Bridge" is drawn at half opacity and the product name at full, so the family
name recedes and the product reads first.

Needs: pip install fonttools brotli uharfbuzz
Usage: python3 scripts/icon/build-logo.py [prefix] [product]   (default: Bridge Clip)
"""
import io
import re
import sys
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ROOT = Path(__file__).resolve().parents[2]
FONT = ROOT / 'src/renderer/assets/fonts/Geist-Variable.woff2'
MARK = ROOT / 'resources/bridgemind-mark.svg'

WEIGHT = 600
TRACKING = -18  # font units (1000 per em): slightly tight, for display size

# Lockup geometry in a 100-unit-tall box: mark 100 tall, capitals 50 tall and
# vertically centred on the mark, 28 units between mark and wordmark.
MARK_H = 100
MARK_W = MARK_H * 142.29 / 129
GAP = 28
CAP_H = 50
BASELINE = 75

COLORS = {'bridgeclip-logo.svg': '#fafafa', 'bridgeclip-logo-light.svg': '#09090b'}


def outline(parts):
    font = instancer.instantiateVariableFont(TTFont(FONT), {'wght': WEIGHT})
    font.flavor = None
    buf = io.BytesIO()
    font.save(buf)

    hb_font = hb.Font(hb.Face(hb.Blob(buf.getvalue())))
    hb_buf = hb.Buffer()
    hb_buf.add_str(''.join(parts))
    hb_buf.guess_segment_properties()
    hb.shape(hb_font, hb_buf, {'kern': True, 'liga': True})

    ends = []
    total = 0
    for part in parts:
        total += len(part)
        ends.append(total)

    glyph_set = font.getGlyphSet()
    order = font.getGlyphOrder()
    paths = [''] * len(parts)
    x = 0.0
    count = len(hb_buf.glyph_infos)
    for i, (info, pos) in enumerate(zip(hb_buf.glyph_infos, hb_buf.glyph_positions)):
        part = next(j for j, end in enumerate(ends) if info.cluster < end)
        pen = SVGPathPen(glyph_set, ntos=lambda v: f'{v:.1f}'.rstrip('0').rstrip('.'))
        glyph_set[order[info.codepoint]].draw(TransformPen(pen, (1, 0, 0, -1, x + pos.x_offset, -pos.y_offset)))
        paths[part] += pen.getCommands()
        x += pos.x_advance + (TRACKING if i < count - 1 else 0)

    return paths, x, font['OS/2'].sCapHeight


def main():
    parts = sys.argv[1:3] if len(sys.argv) >= 3 else ['Bridge', 'Clip']
    (prefix_d, product_d), text_width, cap_height = outline(parts)
    scale = CAP_H / cap_height
    text_x = MARK_W + GAP
    width = text_x + text_width * scale

    mark_src = MARK.read_text()
    mark_inner = re.sub(r'^[\s\S]*?<svg[^>]*>', '', mark_src)
    mark_inner = re.sub(r'</svg>\s*$', '', mark_inner).strip()

    for filename, color in COLORS.items():
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {MARK_H}" role="img" aria-label="{''.join(parts)}">
  <title>{''.join(parts)}</title>
  <svg width="{MARK_W:.2f}" height="{MARK_H}" viewBox="0 0 142.29 129">
    {mark_inner}
  </svg>
  <g transform="translate({text_x:.2f} {BASELINE}) scale({scale:.6f})" fill="{color}">
    <path fill-opacity="0.5" d="{prefix_d}"/>
    <path d="{product_d}"/>
  </g>
</svg>
'''
        (ROOT / 'resources' / filename).write_text(svg)
        print(f'wrote resources/{filename} ({width:.1f} x {MARK_H})')


if __name__ == '__main__':
    main()
