r"""Cases for the turn-decision metrics, run before any real audio goes near them.

    .venv\Scripts\python.exe -m probe.test_turn_metrics

The scorer these test is going into a run for alkove's maintainer, who has
already read the last one against his own core. A boundary condition that
quietly mislabels a cut-off would not announce itself in a table of means.

The case that matters most is the one that was wrong first time round. A
labelled ``primary`` span is a region where the caller is the active speaker,
pauses included, so "a decision inside an open span" counted every natural
pause as an interruption -- on the work split that produced a worst case of a
caller cut off with 41 seconds still to come. A cut-off needs the caller to
actually carry on.
"""

from __future__ import annotations

import sys

from probe.metrics import decisions_from_spans, turn_metrics


def meta(*segments: tuple[float, float, str]) -> dict:
    return {"segments": [{"start": s, "end": e, "label": lab} for s, e, lab in segments]}


CALLER = meta((0.0, 3.0, "primary"), (3.0, 5.0, "secondary"), (5.0, 8.0, "primary"))

# The caller talks, pauses, talks again inside the first labelled span, then
# twice more in the second. Every gap here is a pause, not a turn boundary.
SPEAKING = [(0.0, 0.9), (1.2, 3.0), (5.0, 6.0), (6.4, 8.0)]

FAILED: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n          got  {got}\n          want {want}")
        FAILED.append(name)


print("what counts as a cut-off")

# The whole point: a decision inside an open span, where the caller then carries
# on, is the floor being taken from them.
m = turn_metrics(CALLER, [1.0], SPEAKING)
check("caller resumes -> cut-off", m["cutoffs"], 1)

# ... and what they lost is the speech still to come, not the wall clock to the
# end of the label. 1.0 -> 3.0 is two seconds of span but 1.8s of speech.
check("severity is speech in the horizon, not the whole span", m["overlap_speech_s_max"], 1.8)
check("resume gap is recorded", m["resume_gap_s_median"], 0.2)

# The failure the first version had: a decision in a pause that the caller does
# not come back from inside the window is an ordinary turn boundary.
m = turn_metrics(CALLER, [5.2], SPEAKING)
check("long pause is not a cut-off", m["cutoffs"], 0)

# ... and the same decision is a cut-off once the caller is given time to come
# back, which is what resume_s is for.
m = turn_metrics(CALLER, [5.2], SPEAKING, resume_s=1.5)
check("resume_s widens what counts", m["cutoffs"], 1)

# A short pause they do come back from, in the same span.
m = turn_metrics(CALLER, [6.0], SPEAKING)
check("short pause they return from -> cut-off", (m["cutoffs"], m["overlap_speech_s_max"]), (1, 1.6))

print()
print("what does not count")

# A decision after the caller stopped is dead air, not an interruption.
m = turn_metrics(CALLER, [3.4], SPEAKING)
check("after the caller stops is lag", (m["cutoffs"], m["lag_s_median"]), (0, 0.4))

# Label boundaries are not infinitely precise.
m = turn_metrics(CALLER, [2.95], SPEAKING)
check("inside tolerance is not a cut-off", m["cutoffs"], 0)

# Background is not the caller, so nothing there can be an interruption.
m = turn_metrics(CALLER, [4.0], SPEAKING)
check("decision in background is not a cut-off", m["cutoffs"], 0)

# A decision before any caller speech is neither a cut-off nor a lag, and must
# not be silently counted as either.
m = turn_metrics(meta((2.0, 4.0, "primary")), [1.0], [(2.0, 4.0)])
check("before all speech is neither", (m["cutoffs"], m["lag_turns"]), (0, 0))

print()
print("degenerate inputs")

m = turn_metrics(CALLER, [], SPEAKING)
check("no decisions", (m["turns"], m["cutoff_rate"], m["lag_s_median"]), (0, None, None))

m = turn_metrics(meta((0.0, 2.0, "noise")), [1.0], [(0.0, 2.0)])
check("no caller labels", (m["cutoffs"], m["lag_turns"]), (0, 0))

m = turn_metrics(CALLER, [1.0], [])
check("no vad spans -> nothing resumes", m["cutoffs"], 0)

print()
print("decisions from a silence-timeout VAD")

# The decision lands `silence` after the span closes, not when it closes. This
# is the distinction the maintainer drew, so it is the one that must not drift.
check("silence is added to each span end", decisions_from_spans([(0.0, 2.0), (5.0, 7.0)], 0.35), [2.35, 7.35])
check("no spans, no decisions", decisions_from_spans([], 0.35), [])

# It composes: a VAD closing exactly on the labelled end cuts nobody off.
m = turn_metrics(CALLER, decisions_from_spans([(0.0, 3.0)], 0.35), SPEAKING)
check("closing on the label is a lag of exactly silence", (m["cutoffs"], m["lag_s_median"]), (0, 0.35))

print()
if FAILED:
    print(f"{len(FAILED)} failed: {', '.join(FAILED)}")
    sys.exit(1)
print(f"all cases pass")
