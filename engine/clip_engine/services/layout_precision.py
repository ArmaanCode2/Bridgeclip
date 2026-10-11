"""Bounded, local refinement of layout boundaries using decoded source frames."""
from bisect import bisect_left, bisect_right
from typing import Optional

from .camera_scan import MIN_SCORE

MAX_DETAIL_FRAMES = 6000
# confirmed_cuts accepts a change this strong on its score alone.
STRONG_CUT_SCORE = .15
# Face evidence confirmed_cuts reads around a weaker change.
BEFORE_MS = 250
AFTER_MS = 350


class DetailSelector:
    """Choose, while decoding, the few source frames that decide each camera change.

    A strong visual cut is accepted on its score alone, so only its first frame
    is analyzed (exact face samples and a keyframe for the new shot). A weaker
    change is decided by face evidence: the frame before it plus frames about
    a third and two thirds into the following AFTER_MS, which with the cut
    frame are the three stable observations confirmed_cuts needs.

    Frames are chosen in time order within a hard budget. Past it no more
    detection work is added (it never raises): markers still align layout
    boundaries, and strong cuts are still accepted.
    """

    def __init__(self, duration_ms: float, budget: int = MAX_DETAIL_FRAMES):
        self.duration_ms, self.budget = duration_ms, budget
        self.used = 0
        self.capped = False
        self._targets: list[tuple[float, float]] = []  # (earliest, latest) time

    def _take(self, count: int) -> bool:
        if self.used + count > self.budget:
            self.capped = True
            return False
        self.used += count
        return True

    def select(self, t: float, score: Optional[float], previous_t: Optional[float]) -> tuple[bool, bool, bool]:
        """(analyze the previous frame, analyze this frame, this frame starts a change)."""
        take_previous = take_current = False
        due = [target for target in self._targets if target[0] <= t]
        if due:
            self._targets = [target for target in self._targets if target[0] > t]
            if any(t <= latest for _, latest in due) and self._take(1):
                take_current = True
        marker = score is not None and score >= MIN_SCORE and 0 < t < self.duration_ms
        if not marker:
            return take_previous, take_current, False
        if score >= STRONG_CUT_SCORE:
            if not take_current and self._take(1):
                take_current = True
            return take_previous, take_current, True
        has_previous = previous_t is not None and 0 <= previous_t and t - previous_t <= BEFORE_MS
        if self._take(int(has_previous) + int(not take_current)):
            take_previous, take_current = has_previous, True
            self._targets += [(t + AFTER_MS / 3, t + AFTER_MS), (t + AFTER_MS * 2 / 3, t + AFTER_MS)]
        return take_previous, take_current, True


def confirmed_cuts(markers, frames, duration_ms, evidence):
    """Strong cuts or abrupt, sustained composition changes; keep weak hints for review."""
    frames = sorted(frames, key=lambda f: f.t_ms)
    times = [f.t_ms for f in frames]
    accepted = []
    for i, marker in enumerate(markers):
        t, score = marker['at_ms'], marker['score']
        if not 0 < t < duration_ms:
            continue
        # Flash on/off pairs and very brief inserts should not cause crop flicker.
        if any(abs(t - other['at_ms']) < 150 for other in markers[max(0, i-1):i+2] if other is not marker and other['score'] >= max(.12, score * .6)):
            continue
        first = bisect_left(times, t)
        before = frames[bisect_left(times, t - BEFORE_MS):first]
        after = frames[first:bisect_right(times, t + AFTER_MS)]
        stable = len(after) >= 3 and after[-1].t_ms - after[0].t_ms >= 180
        changed = False
        if before and stable:
            old, new = evidence(before[-1]), evidence(after[0])
            changed = old is not None and new is not None and old != new and all(evidence(f) == new for f in after)
            # A relocated/enlarged face needs a discontinuity followed by a stable
            # position. Ordinary gradual tracking or a momentary dropout is not a cut.
            if not changed and score >= .04 and all(len(f.faces) == 1 for f in [before[-1], *after]):
                a, b = before[-1].faces[0], after[0].faces[0]
                jump = abs(a.cx - b.cx) > .22 or abs(a.cy - b.cy) > .22 or max(a.h, b.h) / max(.001, min(a.h, b.h)) > 1.8
                settled = all(abs(f.faces[0].cx - b.cx) < .05 and abs(f.faces[0].cy - b.cy) < .05 and abs(f.faces[0].h - b.h) < .05 for f in after)
                changed = jump and settled
        if score >= STRONG_CUT_SCORE or changed:
            if not accepted or t - accepted[-1] >= 150:
                accepted.append(t)
    return accepted


def align_boundaries(cuts, markers, duration_ms):
    """Refine existing evidence to a nearby observed change without inventing a cut."""
    usable = sorted((m for m in markers if 0 < m['at_ms'] < duration_ms), key=lambda m: m['at_ms'])
    times = [m['at_ms'] for m in usable]
    aligned = {0, duration_ms}
    for t in cuts:
        if not 0 < t < duration_ms:
            continue
        nearby = usable[bisect_left(times, t - 250):bisect_right(times, t + 250)]
        aligned.add(min(nearby, key=lambda m: (abs(m['at_ms'] - t), -m['score']))['at_ms'] if nearby else t)
    return sorted(aligned)
