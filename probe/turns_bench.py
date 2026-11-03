r"""Cue's end-of-turn head against the labelled pauses, and Silero as the baseline.

    .venv\Scripts\python.exe -m probe.turns_bench audio/krisp call_center

alkove 0.4.0 gives Cue a turn head. Its maintainer asked for a re-run on the
same 265 files with two numbers the first run could not produce: when the turn
is declared against the labelled end, and how often the caller is cut off.

Three conditions:

    raw   -> Cue       alkove.Turns(models, rate)
    Focus -> Cue       alkove.Live(models, rate, turn=True)
    Silero             the VAD-only endpointing a LiveKit agent does today

`at` is the decision timestamp; `end` is the backdated end of speech, which is
what the first run measured. Scoring uses `at`.

The files are read at their own rate, as he asked -- not resampled to 16 kHz as
the first run did. Silero needs 16 kHz, so the baseline alone is resampled.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import alkove
import numpy as np
import soundfile as sf

from probe import vad_turn_end as vte
from probe.krisp_bench import load_16k, segments_from
from probe.metrics import gap_metrics

MODELS = ("focus", "cue")

# What a LiveKit agent waits before acting: Silero declares end of speech after
# MIN_SILENCE of quiet, and the session then holds for min_delay. The caller
# experiences the sum, measured from when they actually stopped.
MIN_DELAY = 0.5


def native(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def turnends(events) -> list[float]:
    return [e["at"] for e in (events or []) if e.get("type") == "turnend" and e.get("at") is not None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("split", nargs="?")
    args = ap.parse_args()
    root = Path(args.root)
    split = args.split or root.name

    rows = [
        json.loads(line)
        for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    models = alkove.Models.load(alkove.fetch(MODELS))
    tp = alkove.TurnParams()
    print(f"{split}: {len(rows)} files | alkove {getattr(alkove, '__version__', '?')} | "
          f"TurnParams(threshold={tp.threshold}, min_pause={tp.min_pause}, max_pause={tp.max_pause})")
    print(f"  baseline: silero end + min_silence {vte.MIN_SILENCE}s + min_delay {MIN_DELAY}s", flush=True)

    out: dict[str, dict] = {c: {} for c in ("raw_cue", "focus_cue", "silero")}
    t0 = time.time()
    for i, meta in enumerate(rows, 1):
        path = root / "raw" / meta["file_name"]
        if not path.exists():
            continue
        x, sr = native(path)

        out["raw_cue"][meta["file_name"]] = gap_metrics(
            meta, turnends(alkove.Turns(models, sr).push(x))
        )
        _audio, events = alkove.Live(models, sr, turn=True).push(x)
        out["focus_cue"][meta["file_name"]] = gap_metrics(meta, turnends(events))

        spans = segments_from(load_16k(path))
        out["silero"][meta["file_name"]] = gap_metrics(
            meta, [e + vte.MIN_SILENCE + MIN_DELAY for _, e in spans]
        )

        if i % 10 == 0:
            per = (time.time() - t0) / i
            print(f"  {i}/{len(rows)}  {per:.1f}s/clip  ~{per * (len(rows) - i) / 60:.0f} min left", flush=True)

    dest = Path("results") / f"turns_{split}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(
        json.dumps({"split": split, "alkove": getattr(alkove, "__version__", "?"),
                    "min_delay_s": MIN_DELAY, "min_silence_s": vte.MIN_SILENCE,
                    "clips": len(out["silero"]), "conditions": out}, indent=1),
        encoding="utf-8",
    )
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
