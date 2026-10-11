"""Reaction protection (Jev review mode) and planner skips. No provider calls."""
from types import SimpleNamespace

import pytest

from clip_engine.services.clip_editor import MIN_PIECE_MS, subtract_intervals
from clip_engine.services.editorial_context import MAX_CANDIDATES, analyze_reactions


def segment(a, b, text, words=None):
    return SimpleNamespace(start_time_ms=a, end_time_ms=b, text=text, audio_events=[], words=words or [
        SimpleNamespace(start_time_ms=a, end_time_ms=b, word=text)])


def sentences(texts, length=5000, gap=1000):
    """One sentence per text, `gap` ms of silence between them."""
    return [segment(i * (length + gap), i * (length + gap) + length, text) for i, text in enumerate(texts)]


def test_many_plain_pauses_never_protect_the_whole_clip():
    transcript = sentences(['A plain sentence.'] * 40)
    report = analyze_reactions(transcript, 0, transcript[-1].end_time_ms)
    assert report['protected_source'] == []
    assert report['candidates'] == [] and report['flags'] == []


def test_each_cued_gap_is_protected_and_records_stay_bounded():
    texts = ['Watch this.' if i % 3 == 0 else 'A plain sentence.' for i in range(90)]
    transcript = sentences(texts)
    clip = [0, transcript[-1].end_time_ms]
    report = analyze_reactions(transcript, *clip)
    assert len(report['protected_source']) == 30
    assert clip not in report['protected_source']
    assert len(report['candidates']) == MAX_CANDIDATES
    assert report['flags'] == ['reaction_records_truncated']
    # The uncued pause after sentence 1 stays available to tight pacing.
    gap = (transcript[1].end_time_ms, transcript[2].start_time_ms)
    assert not any(a < gap[1] and b > gap[0] for a, b in report['protected_source'])


@pytest.mark.parametrize('before,after', [
    ('A plain sentence.', 'That was great.'),
    ('A plain sentence.', 'Wow.'),
    ('A plain sentence.', "That's why it works."),
    ('A plain sentence.', "That's what I mean."),
    ('A plain sentence.', 'Unbelievable, ridiculous.'),
    ('A plain sentence.', 'Now we react to the numbers.'),
    ("Here's the thing.", 'A plain sentence.'),
    ("Let's see how this goes.", 'A plain sentence.'),
    ("Here's what I think.", 'A plain sentence.'),
])
def test_common_phrases_alone_do_not_protect_a_pause(before, after):
    transcript = sentences([before, after])
    assert analyze_reactions(transcript, 0, transcript[-1].end_time_ms)['protected_source'] == []


@pytest.mark.parametrize('before,after,reason', [
    ('Watch this.', 'A plain sentence.', 'introduction_cue'),
    ('Check this out.', 'A plain sentence.', 'introduction_cue'),
    ('Take a look.', 'A plain sentence.', 'introduction_cue'),
    ("Let's play the clip.", 'A plain sentence.', 'introduction_cue'),
    ('A plain sentence.', 'Did you see that?', 'reaction_cue'),
    ('A plain sentence.', 'What was that?', 'reaction_cue'),
    ('A plain sentence.', 'Look at what he did.', 'reaction_cue'),
])
def test_explicit_cues_protect_the_adjacent_pause(before, after, reason):
    transcript = sentences([before, after])
    report = analyze_reactions(transcript, 0, transcript[-1].end_time_ms)
    assert report['protected_source'] == [[0, transcript[-1].end_time_ms]]
    assert report['candidates'][0]['reason'] == reason


def test_cue_must_be_spoken_next_to_the_pause():
    # "Watch this" opens ten seconds of continuous speech; the later pause is ordinary.
    words = [SimpleNamespace(start_time_ms=0, end_time_ms=300, word='Watch'),
             SimpleNamespace(start_time_ms=350, end_time_ms=650, word='this.')]
    words += [SimpleNamespace(start_time_ms=t, end_time_ms=t + 300, word='word') for t in range(700, 10000, 350)]
    first = segment(0, words[-1].end_time_ms, ' '.join(w.word for w in words), words)
    second = segment(first.end_time_ms + 2000, first.end_time_ms + 6000, 'A plain sentence.')
    assert analyze_reactions([first, second], 0, second.end_time_ms)['protected_source'] == []


class TestPlannerSkipsWinOverProtection:
    def test_protected_fragment_inside_a_skip_is_not_restored(self):
        # Previously returned a 733 ms fragment inside the skipped sponsor read.
        assert subtract_intervals([(0, 30000)], [(10000, 20000)], [(14200, 14900)]) == [(0, 10000), (20000, 30000)]

    def test_protection_straddling_a_skip_edge_stops_at_the_skip(self):
        assert subtract_intervals([(0, 30000)], [(10000, 20000)], [(9000, 11000)], 30000) == [(0, 10000), (20000, 30000)]

    def test_restored_fragment_respects_min_piece(self):
        keeps = subtract_intervals([(0, 5000), (6000, 30000)], [(5500, 29800)], [(29850, 29950)], 30000)
        assert keeps == [(0, 5000)]
        assert all(b - a >= MIN_PIECE_MS for a, b in keeps)

    def test_protection_outside_skips_is_kept(self):
        assert subtract_intervals([(0, 1000)], [], [(5000, 6000)], 10000) == [(0, 1000), (5000, 6000)]
        # A short protected tail may extend a kept piece.
        assert subtract_intervals([(0, 1000)], [], [(1000, 1100)], 10000) == [(0, 1100)]
