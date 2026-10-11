"""Reaction protection in source time for Jev review. Code owns all cuts.

A speech gap is protected only when the words right next to it carry an
explicit cue: an introduction just before it ("watch this") or a reaction just
after it ("did you see that"). Ordinary pauses stay available to tight pacing,
no provider is asked, and many pauses never protect a whole clip. Planner
skips (tangents, sponsor reads) still win over protection; see subtract_intervals.
"""
import re

# A pause at least this long can hold a watched event.
MIN_GAP_MS = 700
# Cues must be spoken within this distance of the gap to count.
CUE_WINDOW_MS = 4000
# Dialogue kept around a gap as audit evidence and for the protected sequence.
CONTEXT_MS = 20000
INTRO = re.compile(
    r"\b(watch (this|that|it)|look at (this|that)|take a look|let.s (watch|take a look)"
    r"|check (this|that|it) out|play (this|that|the (clip|video|footage))"
    r"|roll (the )?(clip|tape|video|footage)|here.s the (clip|video|footage))\b", re.I)
# Only reactions that refer back to something watched; "wow", "that was" or
# "that's why" are ordinary speech and never protect a gap on their own.
REFERENCE = re.compile(
    r"\b(did you (see|hear|catch) (that|this|it)|what was that|look at what (he|she|they|it) did"
    r"|(you|we) (just )?(saw|heard) (that|it)|see what (he|she|they|it) (just )?did)\b", re.I)
AMBIGUOUS_FILLERS = {'hmm', 'hm', 'mm', 'mhm'}
# Audit records kept per clip (the desktop trace reader accepts up to 24).
MAX_CANDIDATES = 24
RULE_VERSION = 'reaction-cues-v2'


def empty_report():
    return {'version': 1, 'candidates': [], 'protected_source': [], 'flags': [],
            'qa': None, 'duplicates': [], 'fillers': []}


def _rules_judgment():
    """The shape of a Jev record, so saved traces stay readable; nothing is sent."""
    return {'status': 'disabled', 'model': 'local-rules', 'requested_model': 'local-rules',
            'rule_version': RULE_VERSION, 'cache_id': '', 'cache_hit': False, 'latency_ms': 0,
            'input_tokens': None, 'output_tokens': None, 'cost_usd': None, 'estimated_cost_usd': None,
            'answers': {}, 'questions': {}}


def _dialogue(segments, start, end):
    result = []
    for s in segments:
        words = [w.word for w in s.words if w.start_time_ms < end and w.end_time_ms > start]
        text = ' '.join(words) if s.words else s.text
        if text:
            result.append({'start_ms': max(start, s.start_time_ms), 'end_ms': min(end, s.end_time_ms), 'text': text[:1600]})
    return result


def reaction_candidates(segments, start_ms, end_ms):
    """Cued word/segment gaps touching the proposed clip, including crossing edges."""
    ordered = sorted(segments, key=lambda s: s.start_time_ms)
    spans = sorted([(w.start_time_ms, w.end_time_ms, w.word) for s in ordered for w in s.words]
                   or [(s.start_time_ms, s.end_time_ms, s.text) for s in ordered])

    # Cue windows only reach this far; keep the per-gap text lookups local.
    lo, hi = start_ms - CONTEXT_MS - CUE_WINDOW_MS, end_ms + CONTEXT_MS + CUE_WINDOW_MS
    nearby = [span for span in spans if span[1] > lo and span[0] < hi]

    def spoken(a, b):
        return ' '.join(text for start, end, text in nearby if start < b and end > a)

    gaps = []
    previous_end = None
    for a, b, _ in spans:
        if previous_end is not None and a - previous_end >= MIN_GAP_MS and previous_end < end_ms + CONTEXT_MS and a > start_ms - CONTEXT_MS:
            intro = INTRO.search(spoken(previous_end - CUE_WINDOW_MS, previous_end))
            reference = REFERENCE.search(spoken(a, a + CUE_WINDOW_MS))
            before = [s for s in ordered if s.start_time_ms < previous_end and s.end_time_ms >= previous_end - CONTEXT_MS][-3:]
            after = [s for s in ordered if s.end_time_ms > a and s.start_time_ms <= a + CONTEXT_MS][:3]
            if (intro or reference) and before and after and before[-1].start_time_ms < end_ms and after[0].end_time_ms > start_ms:
                # Keep the whole setup and reaction lines, not just the pause.
                setup = next((s for s in reversed(before) if INTRO.search(s.text)), before[-1]) if intro else before[-1]
                reaction = next((s for s in after if REFERENCE.search(s.text)), after[0]) if reference else after[0]
                gaps.append({'interval': [previous_end, a],
                             'sequence': [min(setup.start_time_ms, previous_end), max(reaction.end_time_ms, a)],
                             'dialogue_before': _dialogue(before, max(0, previous_end - CONTEXT_MS), previous_end),
                             'dialogue_after': _dialogue(after, a, a + CONTEXT_MS),
                             'reason': 'introduction_cue' if intro else 'reaction_cue'})
        previous_end = max(previous_end or 0, b)
    return gaps


def analyze_reactions(segments, start_ms, end_ms):
    """Return protected source intervals and audit records; never change timing here.

    Every cued gap is protected. Only the audit records are bounded; the clip
    as a whole is never protected in their place.
    """
    report = empty_report()
    for candidate in reaction_candidates(segments, start_ms, end_ms):
        report['protected_source'].append(candidate['sequence'])
        if len(report['candidates']) == MAX_CANDIDATES:
            if 'reaction_records_truncated' not in report['flags']:
                report['flags'].append('reaction_records_truncated')
            continue
        state = {'candidate': candidate['interval'], 'dialogue_before': candidate['dialogue_before'],
                 'dialogue_after': candidate['dialogue_after'],
                 'observed_facts': {'transcript_speech_gap': True}, 'visual_observations': []}
        judgment = _rules_judgment()
        report['candidates'].append({**candidate, 'evidence': state, 'evidence_history': [state],
                                     'judgment': judgment, 'judgment_history': [judgment], 'visual': None,
                                     'decision': 'protect'})
    return report


def repair_context_boundaries(start_ms, end_ms, report, allowed_start, allowed_end, max_duration_ms):
    """Expand only within explicit constraints, otherwise flag the incomplete edit."""
    protected = report['protected_source']
    a = min([start_ms] + [p[0] for p in protected])
    b = max([end_ms] + [p[1] for p in protected])
    if a >= allowed_start and b <= allowed_end and b - a <= max_duration_ms:
        if (a, b) != (start_ms, end_ms):
            report['flags'].append('reaction_boundaries_expanded')
        return a, b
    report['flags'].append('incomplete_reaction_context')
    return start_ms, end_ms


def window_protection(report, window_start, window_ms):
    return [(max(0, a - window_start), min(window_ms, b - window_start))
            for a, b in report.get('protected_source', [])
            if b > window_start and a < window_start + window_ms]


def record_prevented_cuts(report, baseline_keeps, planner_skips, protected, window_start, window_ms, plan):
    cuts, cursor = [], 0
    for a, b in baseline_keeps:
        if a > cursor:
            cuts.append((cursor, a))
        cursor = b
    if cursor < window_ms:
        cuts.append((cursor, window_ms))
    from clip_engine.services.clip_editor import preserve_intervals, remove_intervals
    # Planner skips win over protection, so only pacing cuts can be prevented.
    protections = remove_intervals(preserve_intervals([], protected, window_ms), planner_skips)
    prevented = []
    i = 0
    for a, b in sorted(cuts):
        while i < len(protections) and protections[i][1] <= a:
            i += 1
        j = i
        while j < len(protections) and protections[j][0] < b:
            p, q = protections[j]
            if min(b, q) > max(a, p):
                prevented.append({'interval': [window_start + max(a, p), window_start + min(b, q)], 'kind': 'pacing'})
            j += 1
    report['prevented_cuts'] = prevented
    for candidate in report['candidates']:
        a, b = candidate['interval']
        candidate['layout_segments'] = [
            {'start_ms': window_start + s.start_ms, 'end_ms': window_start + s.end_ms,
             'layout': s.detected_layout or s.layout}
            for s in (plan.shots if plan else []) if window_start + s.start_ms < b and window_start + s.end_ms > a]
