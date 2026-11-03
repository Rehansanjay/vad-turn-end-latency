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

# How far before the caller's labelled end a decision still counts as having
# declared that turn. The labels run past the voice, so a detector that is right
# can decide just inside the last span; see gap_metrics.
FINAL_TOL_S = 1.5


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
    # The last segment in the list is not always the last one in time. Ten of
    # the 96 ``work`` files list theirs out of order -- 194.wav ends its list
    # with a span at 22.2s, four spans before the one that ends at 88.4s. Taking
    # segs[-1] excluded nine files that do end with background from the
    # turn-end population, and that was wrong in the first published run.
    segs = meta.get("segments") or []
    last_seg = max(segs, key=lambda s: s["end"]) if segs else None
    last_label = last_seg["label"] if last_seg else None
    ends_with_background = last_label in BACKGROUND_LABELS
    tail_s = None
    if ends_with_background and last_primary_end is not None:
        tail_s = round(last_seg["end"] - last_primary_end, 2)

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


def gaps_between(meta: dict) -> list[Span]:
    """The pauses between the caller's labelled spans.

    alkove's maintainer pointed out where the signal actually is. A turn cannot
    be declared while the detector still hears the caller, so a decision landing
    *inside* a labelled span only happens when the detector has lost the voice --
    which ``primary_recall`` already measures. What a turn detector is actually
    judged on is the pauses: given a gap, did it declare the turn over before the
    caller carried on, and how far into the gap did it do so.
    """
    pri = sorted(spans(meta, PRIMARY_LABELS))
    return [(pri[i][1], pri[i + 1][0]) for i in range(len(pri) - 1) if pri[i + 1][0] > pri[i][1]]


def gap_metrics(
    meta: dict,
    decisions: list[float],
    buckets: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
    final_tol_s: float = FINAL_TOL_S,
) -> dict:
    """Score turn decisions against the caller's pauses and their final stop.

    Two things, and they are different questions.

    **The final turn.** After the caller's last labelled span there is nothing
    more coming, so a turn should be declared. ``final_delay_s`` is the decision
    timestamp minus that labelled end -- the dead air a caller waits through.

    That number takes only a decision strictly *after* the labelled end, and
    that is too strict to be a measurement. alkove's maintainer found why: these
    labels run about 0.2s past the voice in the median, while Cue decides about
    0.3s after the voice stops -- so its decision can land just inside the last
    span and be discarded as "never declared". The first published run reported
    71 such files on ``Focus -> Cue``; the detector had in fact declared most of
    them.

    ``final_delay_tol_s`` is his rule: the **last** decision at or after
    ``final_end - final_tol_s``, with the delay allowed to go negative. Taking
    the last rather than the first is what "no new speech before the end"
    amounts to in practice -- had the caller carried on, a later decision would
    have closed the turn they actually finished, and that is the one to score.

    ``tail_after_final_s`` keeps the remaining failures honest. When the labels
    stop within a breath of the caller's last word there is no room left in the
    file to declare anything, and a detector should not be marked down for a
    turn that had nowhere to land.

    **The pauses.** Every earlier gap is a chance to be wrong. A decision taken
    in a long gap is a correct handover; one taken in a short gap cut the caller
    off mid-sentence. Length is what separates the two, so gaps are bucketed by
    it rather than pooled: a 0.4s breath and a 6s handover are not the same
    event and averaging them says nothing.

    ``into_gap`` is where in the pause the decision landed. Near 0 means the
    detector fired the moment the voice stopped; near 1 means it waited almost
    until the caller resumed.
    """
    pri = sorted(spans(meta, PRIMARY_LABELS))
    if not pri:
        return {
            "gaps": 0,
            "premature": 0,
            "by_bucket": {},
            "final_delay_s": None,
            "final_delay_tol_s": None,
            "final_tol_s": final_tol_s,
            "tail_after_final_s": None,
        }

    decisions = sorted(decisions)
    gaps = gaps_between(meta)
    final_end = pri[-1][1]

    def bucket_of(length: float) -> str:
        for b in buckets:
            if length < b:
                return f"<{b:g}s"
        return f">={buckets[-1]:g}s"

    by_bucket: dict[str, dict] = {}
    premature = 0
    for g0, g1 in gaps:
        name = bucket_of(g1 - g0)
        row = by_bucket.setdefault(name, {"gaps": 0, "declared": 0, "into_gap_s": [], "into_gap_frac": []})
        row["gaps"] += 1
        hit = next((t for t in decisions if g0 < t <= g1), None)
        if hit is not None:
            row["declared"] += 1
            row["into_gap_s"].append(hit - g0)
            row["into_gap_frac"].append((hit - g0) / (g1 - g0))
            premature += 1

    for row in by_bucket.values():
        row["into_gap_s_median"] = round(_median(row["into_gap_s"]), 2) if row["into_gap_s"] else None
        row["into_gap_frac_median"] = round(_median(row["into_gap_frac"]), 3) if row["into_gap_frac"] else None
        row["declared_rate"] = round(row["declared"] / row["gaps"], 3) if row["gaps"] else None
        del row["into_gap_s"], row["into_gap_frac"]

    final = next((t for t in decisions if t > final_end), None)
    # the maintainer's rule: the last decision from final_tol_s before the
    # labelled end onwards, so a decision that beat the label is still a
    # decision rather than a miss
    near = [t for t in decisions if t >= final_end - final_tol_s]
    tol = near[-1] if near else None
    # max, not segs[-1]: these lists are not always in time order (see score)
    segs = meta.get("segments") or []
    tail = round(max(s["end"] for s in segs) - final_end, 2) if segs else None
    return {
        "gaps": len(gaps),
        "premature": premature,
        "premature_rate": round(premature / len(gaps), 3) if gaps else None,
        "by_bucket": by_bucket,
        "final_end_s": round(final_end, 2),
        "final_delay_s": None if final is None else round(final - final_end, 2),
        "final_delay_tol_s": None if tol is None else round(tol - final_end, 2),
        "final_tol_s": final_tol_s,
        # seconds of labels left after the caller's last word: 0 means the file
        # ends on it, and no detector could have declared anything
        "tail_after_final_s": tail,
    }
