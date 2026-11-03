r"""Cases for the gap-based turn scorer.

    .venv\Scripts\python.exe -m probe.test_gap_metrics

This replaces an earlier scorer that looked for decisions *inside* the caller's
labelled spans. alkove's maintainer pointed out that is the wrong place: a turn
cannot be declared while the detector still hears the caller, so the signal is
in the pauses between spans and in the final stop. These cases pin that down.
"""

from __future__ import annotations

import sys

from probe.metrics import gap_metrics, gaps_between


def meta(*segments: tuple[float, float, str]) -> dict:
    return {"segments": [{"start": s, "end": e, "label": lab} for s, e, lab in segments]}


# caller speaks, pauses 0.4s (a breath), speaks, pauses 3.0s (a handover), speaks
CALLER = meta(
    (0.0, 2.0, "primary"),
    (2.4, 4.0, "primary"),
    (7.0, 9.0, "primary"),
)

FAILED: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n          got  {got}\n          want {want}")
        FAILED.append(name)


print("finding the pauses")
check("two gaps, the speech between them is not one", gaps_between(CALLER), [(2.0, 2.4), (4.0, 7.0)])
check("adjacent spans make no gap", gaps_between(meta((0.0, 1.0, "primary"), (1.0, 2.0, "primary"))), [])
check("one span, no gaps", gaps_between(meta((0.0, 1.0, "primary"))), [])

print()
print("the final turn")

# nothing follows the last span, so a decision after it is the real turn end
m = gap_metrics(CALLER, [9.5])
check("final delay is measured from the last labelled end", m["final_delay_s"], 0.5)
check("a decision after the end is not premature", m["premature"], 0)

m = gap_metrics(CALLER, [])
check("no decision at all leaves the delay unknown", m["final_delay_s"], None)

print()
print("pauses, bucketed by length")

# a decision in the 0.4s breath cut the caller off; one in the 3s gap did not
m = gap_metrics(CALLER, [2.2, 9.5])
check("declared in the short gap", m["by_bucket"]["<0.5s"]["declared"], 1)
check("how far into that gap", m["by_bucket"]["<0.5s"]["into_gap_s_median"], 0.2)
check("halfway through it", m["by_bucket"]["<0.5s"]["into_gap_frac_median"], 0.5)
check("the long gap was left alone", m["by_bucket"][">=5s"]["declared"] if ">=5s" in m["by_bucket"] else m["by_bucket"]["<5s"]["declared"], 0)

# the 3.0s gap falls in the 2-5s bucket
m = gap_metrics(CALLER, [5.0])
check("a 3s gap buckets as <5s", "<5s" in m["by_bucket"], True)
check("declared there", m["by_bucket"]["<5s"]["declared"], 1)
check("1s into a 3s gap", m["by_bucket"]["<5s"]["into_gap_s_median"], 1.0)

print()
print("edges")

# a decision exactly on a span boundary belongs to the gap it opens, not the one before
m = gap_metrics(CALLER, [2.4])
check("a decision at the gap's end still counts", m["by_bucket"]["<0.5s"]["declared"], 1)

m = gap_metrics(CALLER, [2.0])
check("a decision at the gap's start does not", m["by_bucket"]["<0.5s"]["declared"], 0)

# two decisions in one gap is one premature turn, not two
m = gap_metrics(CALLER, [2.1, 2.3])
check("one gap counts once", m["premature"], 1)

# a file with no caller labels has nothing to score
m = gap_metrics(meta((0.0, 2.0, "noise")), [1.0])
check("no caller labels", (m["gaps"], m["final_delay_s"]), (0, None))

print()
if FAILED:
    print(f"{len(FAILED)} failed: {', '.join(FAILED)}")
    sys.exit(1)
print("all cases pass")
