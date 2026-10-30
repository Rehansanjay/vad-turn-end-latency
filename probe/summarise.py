"""Turn the per-file benchmark JSON into the comparison table.

Two populations are kept apart on purpose. Turn-end delay is only meaningful on files whose
last labelled span is background -- elsewhere the caller is still talking at the end and there
is nothing for a filter to cut, so including them would pull every condition toward zero and
hide the difference. False triggers use every file that has a background span at all.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
# alkove_bench.py records the run's own metadata around the per-file results
# (which alkove build produced them, above all), so unwrap that first. The
# Krisp files are the bare mapping.
meta = {k: v for k, v in data.items() if k != "conditions"} if "conditions" in data else {}
if meta:
    data = data["conditions"]
    print("run: " + ", ".join(f"{k}={v}" for k, v in meta.items()))
conds = list(data)

print(f"conditions: {', '.join(conds)}\n")

ref = data[conds[0]]
testable = [n for n, r in ref.items() if r["ends_with_background"] and r["turn_end_delay_s"] is not None]
with_bg = [n for n, r in ref.items() if r["background_spans"] > 0]
print(f"files: {len(ref)} total | {len(with_bg)} contain background | "
      f"{len(testable)} end with background (turn-end testable)\n")

print(f"{'condition':12}{'false trig':>11}{'files w/ FT':>13}{'median delay':>14}{'mean delay':>12}{'speech %':>10}")
print("-" * 72)
rowsum = {}
for c in conds:
    d = data[c]
    ft = sum(d[n]["false_triggers"] for n in with_bg if n in d)
    ftf = sum(1 for n in with_bg if n in d and d[n]["false_triggers"] > 0)
    delays = [d[n]["turn_end_delay_s"] for n in testable if n in d and d[n]["turn_end_delay_s"] is not None]
    sf_ = [d[n]["speech_fraction"] for n in d]
    rowsum[c] = {
        "false_triggers": ft,
        "files_with_false_triggers": ftf,
        "median_turn_end_delay_s": round(st.median(delays), 2) if delays else None,
        "mean_turn_end_delay_s": round(st.fmean(delays), 2) if delays else None,
        "speech_fraction_mean": round(st.fmean(sf_), 3) if sf_ else None,
        "n_delays": len(delays),
    }
    print(f"{c:12}{ft:11d}{ftf:13d}"
          f"{(rowsum[c]['median_turn_end_delay_s'] if delays else float('nan')):13.2f}s"
          f"{(rowsum[c]['mean_turn_end_delay_s'] if delays else float('nan')):11.2f}s"
          f"{100 * rowsum[c]['speech_fraction_mean']:9.1f}%")

print(f"\n{'condition':12}{'rms primary':>13}{'rms background':>16}{'ratio':>10}")
print("-" * 51)
for c in conds:
    d = data[c]
    p = st.fmean([d[n]["rms_primary"] for n in d if d[n]["rms_primary"] > 0] or [0])
    b = st.fmean([d[n]["rms_background"] for n in d if n in d] or [0])
    ratio = (p / b) if b > 0 else float("inf")
    rowsum[c]["rms_primary_mean"] = round(p, 5)
    rowsum[c]["rms_background_mean"] = round(b, 5)
    rowsum[c]["primary_to_background_ratio"] = None if b == 0 else round(ratio, 1)
    print(f"{c:12}{p:13.5f}{b:16.5f}{ratio:10.1f}")

print("\nA filter that only attenuates moves both columns together; the ratio is what separates")
print("an isolator from a volume control.")

# The two columns that catch a filter which is simply destroying the caller.
# Every metric above improves when the primary speaker is removed, so none of
# them can tell a good filter from a destructive one on its own.
print()
print(f"{'condition':12}{'primary recall':>16}{'bg speech %':>13}{'worst recall':>14}{'files recall<0.8':>18}")
print("-" * 73)
for c in conds:
    d = data[c]
    rec = [d[n]["primary_recall"] for n in d if d[n].get("primary_recall") is not None]
    leak = [d[n]["background_speech_fraction"] for n in d if d[n].get("background_speech_fraction") is not None]
    poor = sum(1 for r in rec if r < 0.8)
    rowsum[c]["primary_recall_mean"] = round(st.fmean(rec), 3) if rec else None
    rowsum[c]["primary_recall_worst"] = round(min(rec), 3) if rec else None
    rowsum[c]["files_recall_below_0_8"] = poor
    rowsum[c]["background_speech_fraction_mean"] = round(st.fmean(leak), 3) if leak else None
    print(f"{c:12}{(st.fmean(rec) if rec else float('nan')):16.3f}"
          f"{(100 * st.fmean(leak) if leak else float('nan')):12.1f}%"
          f"{(min(rec) if rec else float('nan')):14.3f}{poor:18d}")

print()
print("Recall is the metric a filter cannot game by removing signal: drop the caller and")
print("false triggers, dead air and background leak all improve at once. Background speech %")
print("is VAD speech inside background-only spans anywhere in the file, which is what a VAD")
print("that never switches off does instead of producing a new onset.")

if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(
        json.dumps({"files_total": len(ref), "files_with_background": len(with_bg),
                    "files_turn_end_testable": len(testable), "conditions": rowsum}, indent=1) + "\n",
        encoding="utf-8")
    print(f"\nwrote {sys.argv[2]}")
