"""Score one clip's VAD spans against the benchmark's ground-truth labels.

This used to live inside ``krisp_bench.py``. It moved here when a second driver
needed it, because two copies of a measurement are a silent bug waiting to
happen: the day one of them gains a metric or fixes a boundary condition, the
two sets of published numbers stop meaning the same thing and nothing says so.

Labels, from Krisp-AI/VoiceIsolation-Benchmark-Dataset metadata.jsonl:

    primary    the caller alone
    mix        caller plus background
    secondary  background speech only (text: null)
    noise      non-speech noise

so ``primary``/``mix`` are the spans where the caller is audible, and
``secondary``/``noise`` are the spans where they are not.

Four of the metrics here are the original ones. Two were added because alkove's
maintainer pointed out, correctly, that the original four can all be satisfied
by a filter that is simply destroying the caller:

``primary_recall``
    the fraction of labelled caller time the VAD actually called speech. A
    filter that drops the primary speaker scores perfectly on false triggers
    and on turn-end delay -- there is nothing left to trigger on and nothing
    left to run late. Recall is what catches that, and it is the metric a
    filter cannot game by removing signal.

``background_speech_s``
    VAD speech time *inside* background-only spans, anywhere in the file.
    ``false_triggers`` counts onsets, so it only sees a VAD that switches on
    during background. When the caller goes quiet and a secondary voice carries
    on, a VAD that simply stays on produces no new onset at all, and
    ``turn_end_delay_s`` only looks at the end of the file -- so between them
    the original metrics are blind to exactly the case these filters exist for.
"""

from __future__ import annotations

import numpy as np

SR = 16000

PRIMARY_LABELS = {"primary", "mix"}
BACKGROUND_LABELS = {"secondary", "noise"}

Span = tuple[float, float]


def spans(meta: dict, labels: set[str]) -> list[Span]:
    return [(s["start"], s["end"]) for s in meta.get("segments") or [] if s["label"] in labels]


def total(sp: list[Span]) -> float:
    return sum(e - s for s, e in sp)


def overlap(a: list[Span], b: list[Span]) -> float:
    """Seconds covered by both span lists.

    The label spans do not overlap each other and neither do the VAD spans, so
    summing pairwise intersections cannot double-count.
    """
    seconds = 0.0
    for a0, a1 in a:
        for b0, b1 in b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                seconds += hi - lo
    return seconds


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x**2).mean())) if len(x) else 0.0


def level(x: np.ndarray, sp: list[Span]) -> float:
    """RMS over a set of spans, so the control follows the labels, not a fixed window."""
    if not sp:
        return 0.0
    parts = [x[int(a * SR) : int(b * SR)] for a, b in sp]
    parts = [p for p in parts if len(p)]
    return rms(np.concatenate(parts)) if parts else 0.0


def score(x: np.ndarray, meta: dict, vad: list[Span]) -> dict:
    """Every metric for one clip, given its audio, its labels and its VAD spans."""
    dur = len(x) / SR
    pri = spans(meta, PRIMARY_LABELS)
    bg = spans(meta, BACKGROUND_LABELS)

    # a false trigger is an ONSET inside a background-only span
    false_triggers = sum(1 for s, _ in vad if any(b0 <= s <= b1 for b0, b1 in bg))

    # turn-end delay measured from the end of the last primary-bearing span
    last_primary_end = max((e for _, e in pri), default=None)
    vad_last = max((e for _, e in vad), default=0.0)
    delay = None if last_primary_end is None else round(vad_last - last_primary_end, 2)

    # Turn-end delay is only testable where background outlasts the caller. On a
    # file that ends with the primary still talking there is nothing for a filter
    # to cut, and averaging those in would dilute the effect toward zero.
    segs = meta.get("segments") or []
    last_label = segs[-1]["label"] if segs else None
    ends_with_background = last_label in BACKGROUND_LABELS
    tail_s = None
    if ends_with_background and last_primary_end is not None:
        tail_s = round(segs[-1]["end"] - last_primary_end, 2)

    speech = total(vad)
    primary_s, background_s = total(pri), total(bg)
    kept = overlap(vad, pri)
    leaked = overlap(vad, bg)

    return {
        "ends_with_background": ends_with_background,
        "last_label": last_label,
        "background_tail_s": tail_s,
        "duration_s": round(dur, 2),
        "vad_events": len(vad),
        "speech_s": round(speech, 2),
        "speech_fraction": round(speech / dur, 3) if dur else 0.0,
        "false_triggers": false_triggers,
        "background_spans": len(bg),
        "last_primary_end_s": None if last_primary_end is None else round(last_primary_end, 2),
        "vad_last_speech_s": round(vad_last, 2),
        "turn_end_delay_s": delay,
        "rms_primary": round(level(x, pri), 5),
        "rms_background": round(level(x, bg), 5),
        # the two metrics that catch a filter destroying the caller
        "primary_s": round(primary_s, 2),
        "primary_speech_s": round(kept, 2),
        "primary_recall": round(kept / primary_s, 3) if primary_s else None,
        "background_s": round(background_s, 2),
        "background_speech_s": round(leaked, 2),
        "background_speech_fraction": round(leaked / background_s, 3) if background_s else None,
    }


# --- turn decisions ---------------------------------------------------------
#
# ``turn_end_delay_s`` above is the backdated end of the last span. alkove's
# maintainer pointed out that this is not the moment an agent decides the turn
# is over, and he is right: it measures how long background keeps the VAD
# reporting speech, which is the filter's job, and says nothing about what the
# caller waits through or gets interrupted by.
#
# These two measure the decision instead. They need a timestamp per turn rather
# than a span list, so a VAD that exposes its own turn output can be scored the
# same way as one driven by a silence timeout.


def decisions_from_spans(vad: list[Span], silence_s: float) -> list[float]:
    """When a silence-timeout VAD would actually declare each turn over.

    A VAD declares end of speech only after ``silence_s`` of quiet, so the
    decision lands that much after the span it closes. A filter can look good on
    the backdated number while the caller is still sitting there waiting.
    """
    return [end + silence_s for _, end in vad]


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    mid = len(ys) // 2
    return ys[mid] if len(ys) % 2 else (ys[mid - 1] + ys[mid]) / 2


def turn_metrics(
    meta: dict,
    decisions: list[float],
    vad: list[Span],
    tolerance_s: float = 0.10,
    resume_s: float = 1.0,
    horizon_s: float = 3.0,
) -> dict:
    """Score turn decisions against the labelled caller spans.

    **Lag.** For a decision that lands after the caller stopped, how long they
    waited. This is the dead air a caller actually hears.

    **Cut-off.** A decision taken while the caller was not finished: inside an
    open caller span, and followed by the caller resuming within ``resume_s``.
    Operationally, the agent would have taken the floor and the caller carried
    straight on.

    Two things about this dataset shape the definition, and both of them were
    got wrong before the numbers made them obvious.

    A labelled ``primary`` span is a region where the caller is the active
    speaker, *pauses included* -- not one utterance. So "a decision inside an
    open span" counts every natural pause as an interruption. Requiring the
    caller to resume within ``resume_s`` is what separates a premature endpoint
    from an ordinary turn boundary.

    And severity cannot be "the speech left in the span", because these spans
    run to forty seconds and more. Scored that way the worst case on ``work``
    read as a caller cut off with 39 seconds still to come, which is not a
    thing that happened to anybody. What the agent would actually talk over is
    bounded: the caller's speech in the ``horizon_s`` after the endpoint. That
    is what ``overlap_speech_s`` measures.

    ``tolerance_s`` covers label boundary precision -- a decision within it of a
    span's end is the same instant measured twice, not an interruption.
    """
    pri = sorted(spans(meta, PRIMARY_LABELS))
    vad = sorted(vad)

    lags: list[float] = []
    overlaps: list[float] = []
    gaps: list[float] = []
    worst: dict | None = None

    for t in decisions:
        open_span = next(((a, b) for a, b in pri if a <= t < b - tolerance_s), None)
        if open_span is not None:
            _, b = open_span
            resumes_at = next((s0 for s0, _ in vad if t < s0 <= t + resume_s and s0 < b), None)
            if resumes_at is None:
                continue
            # what the agent would have talked over, bounded rather than
            # running to the end of a region that may be a minute long
            spoken = overlap(vad, [(t, min(t + horizon_s, b))])
            overlaps.append(spoken)
            gaps.append(resumes_at - t)
            if worst is None or spoken > worst["overlap_speech_s"]:
                worst = {
                    "overlap_speech_s": round(spoken, 2),
                    "at_s": round(t, 2),
                    "resumed_after_s": round(resumes_at - t, 2),
                }
            continue
        prev_end = max((b for _, b in pri if b <= t), default=None)
        if prev_end is not None:
            lags.append(t - prev_end)

    turns = len(decisions)
    return {
        "turns": turns,
        "cutoffs": len(overlaps),
        "cutoff_rate": round(len(overlaps) / turns, 3) if turns else None,
        "horizon_s": horizon_s,
        # caller speech in the horizon after a premature endpoint: what the
        # agent would have talked over
        "overlap_speech_s_median": round(_median(overlaps), 2) if overlaps else None,
        "overlap_speech_s_max": round(max(overlaps), 2) if overlaps else None,
        # how quickly they carried on; a very short gap is a clipped word
        "resume_gap_s_median": round(_median(gaps), 2) if gaps else None,
        "cutoff_worst": worst,
        # dead air on the turns that were not cut off
        "lag_s_median": round(_median(lags), 2) if lags else None,
        "lag_s_mean": round(sum(lags) / len(lags), 2) if lags else None,
        "lag_s_max": round(max(lags), 2) if lags else None,
        "lag_turns": len(lags),
    }
