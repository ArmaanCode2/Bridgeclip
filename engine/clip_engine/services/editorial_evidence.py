"""Grounded editorial evidence and bounded discovery coverage (no model calls)."""


def transcript_row(index, segment):
    return {'id': index, 'start_ms': segment.start_time_ms, 'end_ms': segment.end_time_ms,
            'speaker': getattr(segment, 'speaker_label', None), 'text': segment.text}


def parse_moment(value, segments):
    """Resolve model-proposed narrative anchors to actual transcript intervals."""
    if not isinstance(value, dict):
        raise ValueError('Missing moment')
    keys = ('topic_start_segment', 'setup_segment', 'payoff_segment', 'topic_end_segment')
    ids = [value.get(k) for k in keys]
    if any(type(i) is not int or not 0 <= i < len(segments) for i in ids) or ids != sorted(ids):
        raise ValueError('Invalid moment anchors')
    topic = value.get('topic')
    if not isinstance(topic, str) or not 1 <= len(topic.strip()) <= 300 or type(value.get('requires_visual_context')) is not bool:
        raise ValueError('Invalid moment metadata')
    a, s, p, b = ids
    return {'topic': topic.strip(), 'topic_interval': [segments[a].start_time_ms, segments[b].end_time_ms],
            'setup': transcript_row(s, segments[s]), 'payoff': transcript_row(p, segments[p]),
            'requires_visual_context': value['requires_visual_context']}


def preferred_range_ms(preferred_range, duration_ms):
    """The user's preferred range in ms, clamped to the source; None ends mean unbounded."""
    start, end = (list(preferred_range or []) + [None, None])[:2]
    low = 0 if start is None else round(start * 1000)
    high = duration_ms if end is None else round(end * 1000)
    low, high = max(0, min(low, duration_ms)), max(0, min(high, duration_ms))
    return (low, high) if low < high else (0, duration_ms)


def discovery_feedback(entries, duration_ms, preferred_range=None):
    """A second search explores up to six largest unproposed spans, never a quota.

    Spans are limited to the user's preferred range (seconds, as saved in the
    edit audit): the second pass must not look for clips the user excluded.
    """
    low, high = preferred_range_ms(preferred_range, duration_ms)
    previous = []
    for c in entries:
        a, b = c['original_interval']
        accepted = ((c.get('report') or {}).get('coherence') or {}).get('accepted_interval')
        if accepted:
            a, b = min(a, accepted[0]), max(b, accepted[1])
        previous.append({'interval': [a, b], 'title': c['title'], 'status': c['status']})
    covered = sorted([min(max(a, low), high), min(max(b, low), high)] for a, b in (c['interval'] for c in previous))
    gaps, end = [], low
    for a, b in covered + [[high, high]]:
        if a - end >= 30000:
            gaps.append([end, a])
        end = max(end, b)
    gaps = sorted(sorted(gaps, key=lambda p: p[1] - p[0], reverse=True)[:6])
    return {'search_intervals': gaps, 'previous_candidates': previous}


def overlaps(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) > 5000
