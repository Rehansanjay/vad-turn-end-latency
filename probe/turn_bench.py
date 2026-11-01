r"""Baseline for the two numbers alkove's maintainer asked for next.

    .venv\Scripts\python.exe -m probe.turn_bench audio/krisp-work work
    .venv\Scripts\python.exe -m probe.turn_bench audio/krisp call_center

He is adding end of turn, hesitation and interruption as outputs of Cue, and
asked for a re-run on the same 265 files when it ships. Two numbers matter then:

    when the turn is declared over, against the labelled end
    how often the caller is cut off before they are done

Neither is in the existing tables, because every metric there is scored from
spans and these need a *decision* -- the instant an agent would act. For a
silence-timeout VAD that instant is derivable (`decisions_from_spans`), so the
Silero baseline can be measured now, before his build arrives, and his turn
output can drop into the same scorer when it does.

Raw audio only. Focus is not applied here: the point is a reference for the
VAD-driven decision, not another filter comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from probe import vad_turn_end as vte
from probe.krisp_bench import load_16k, segments_from
from probe.metrics import decisions_from_spans, turn_metrics


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    split = sys.argv[2] if len(sys.argv) > 2 else root.name

    rows = [
        json.loads(line)
        for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"{split}: {len(rows)} labelled files, raw -> silero", flush=True)
    print(f"  silero: activation {vte.ACTIVATION}, deactivation {vte.DEACTIVATION}, "
          f"min_silence {vte.MIN_SILENCE}s -- a turn is declared min_silence after the span closes")
    print()

    per_file: dict[str, dict] = {}
    for i, meta in enumerate(rows, 1):
        name = meta["file_name"]
        path = root / "raw" / name
        if not path.exists():
            continue
        # load_16k resamples; the work split is stored at 32 kHz and the VAD
        # wants 16. vte.load refuses rather than resampling, by design.
        x = load_16k(path)
        spans = segments_from(x)
        m = turn_metrics(meta, decisions_from_spans(spans, vte.MIN_SILENCE), spans)
        per_file[name] = m
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    scored = list(per_file.values())
    turns = sum(m["turns"] for m in scored)
    cutoffs = sum(m["cutoffs"] for m in scored)
    lags = [m["lag_s_median"] for m in scored if m["lag_s_median"] is not None]
    spoken = [m["overlap_speech_s_max"] for m in scored if m["overlap_speech_s_max"] is not None]
    worst = max(
        (m["cutoff_worst"] for m in scored if m["cutoff_worst"]),
        key=lambda w: w["overlap_speech_s"],
        default=None,
    )

    print()
    print("=== turn decisions, raw -> silero ===")
    print(f"  files                 {len(scored)}")
    print(f"  turns declared        {turns}")
    print(f"  cut off mid-span      {cutoffs}  ({cutoffs / turns:.1%} of turns)" if turns else "  no turns")
    print(f"  files with a cut-off  {sum(1 for m in scored if m['cutoffs'])}")
    if spoken:
        spoken.sort()
        print("  caller speech in the 3s after a premature endpoint, per file worst case:")
        print(f"    median {spoken[len(spoken) // 2]:.2f}s   max {spoken[-1]:.2f}s")
    if worst:
        print(f"  worst single cut-off  {worst['overlap_speech_s']:.2f}s spoken in the next 3s, "
              f"decided at {worst['at_s']:.2f}s, carried on {worst['resumed_after_s']:.2f}s later")
    gaps = [m["resume_gap_s_median"] for m in scored if m["resume_gap_s_median"] is not None]
    if gaps:
        gaps.sort()
        print("  how long the caller paused before carrying on, after a premature endpoint:")
        print(f"    median {gaps[len(gaps) // 2]:.2f}s   p90 {gaps[int(0.9 * len(gaps))]:.2f}s   max {gaps[-1]:.2f}s")
        print(f"    (add min_silence {vte.MIN_SILENCE}s for the real pause length)")
    if lags:
        lags.sort()
        print(f"  lag on turns not cut off (per-file median):")
        print(f"    median {lags[len(lags) // 2]:.2f}s   max {lags[-1]:.2f}s")
    print()
    print("  A cut-off is a turn declared while a labelled caller span is still open, with 0.10s")
    print("  tolerance for label boundary precision. The count alone is not the finding: what the")
    print("  caller experiences is how much they still had to say.")

    out = Path("results") / f"turn_{split}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "split": split,
                "vad": "silero v5 via livekit-plugins-silero, LiveKit defaults",
                "activation": vte.ACTIVATION,
                "deactivation": vte.DEACTIVATION,
                "min_silence_s": vte.MIN_SILENCE,
                "files": per_file,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
