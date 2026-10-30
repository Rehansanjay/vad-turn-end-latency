"""Four conditions over the Krisp benchmark: alkove's Focus and Cue, against Silero.

    python -m probe.alkove_bench audio/krisp results/alkove_call_center.json [--limit N]

The existing driver compares pre-rendered folders, because Krisp publishes its
filtered output as files. alkove is a library, so the filter runs here and the
conditions are built in process:

    raw      -> Silero    the baseline already in this repo
    Focus    -> Silero    what Focus does to the VAD everyone actually uses
    raw      -> Cue       Cue as a VAD on its own, unfiltered
    Focus    -> Cue       how alkove's own LiveKit plugin runs

The four exist because three of them are not the product. Scoring Focus only
against Silero says nothing about Cue; scoring Cue only on raw audio says
nothing about the pair that ships. The last row is the one a LiveKit user gets.

Two alkove passes per clip, not four: ``focus=True, cue=True`` returns both the
filtered audio and Cue's spans over it, so the filtered audio is computed once
and the Silero pass reuses it.

Each condition is scored against the audio that was fed to its VAD, so the RMS
control describes what the VAD heard rather than what the file held. That
matters here more than in the Krisp comparison: Focus is the thing under test,
and an isolator that separates primary from background has to be told apart
from one that just turns everything down.

Requires alkove >= 0.3.3. Its maintainer reports that 0.3.3 fixes Focus and Cue
on audio with nothing above 4 kHz, which is what the phone_calls split is made
of, so an earlier version would measure the bug rather than the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import alkove
import numpy as np

from probe.krisp_bench import load_16k, segments_from
from probe.metrics import SR, score

# The models Focus and Cue need. Downloaded once into alkove's own cache.
MODELS = ("focus", "cue")


def cue_spans(part) -> list[tuple[float, float]]:
    """Cue's speech spans, in the same shape segments_from returns."""
    return [(s.start, s.end) for s in part.speech]


def conditions(models, raw: np.ndarray) -> dict[str, tuple[np.ndarray, list[tuple[float, float]]]]:
    """The four (audio, vad spans) pairs for one clip."""
    # Cue over the unfiltered audio. No filtering, so the audio it heard is raw.
    raw_cue = alkove.process(models, raw, rate=SR, cue=True)

    # Focus and Cue in one pass: Cue's spans here are over the filtered audio,
    # which is what alkove's LiveKit plugin does, and .audio is that filtered
    # audio -- so Silero can be run on it without filtering a second time.
    both = alkove.process(models, raw, rate=SR, focus=True, cue=True)
    focused = np.asarray(both.audio, dtype=np.float32)

    return {
        "raw_silero": (raw, segments_from(raw)),
        "focus_silero": (focused, segments_from(focused)),
        "raw_cue": (raw, cue_spans(raw_cue)),
        "focus_cue": (focused, cue_spans(both)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="a folder holding metadata.jsonl and raw/")
    parser.add_argument("out", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N clips")
    args = parser.parse_args()

    models = alkove.Models.load(alkove.fetch(MODELS))

    meta_by_name = {
        row["file_name"]: row
        for row in (
            json.loads(line)
            for line in (args.root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    files = sorted((args.root / "raw").glob("*.wav"))
    if args.limit:
        files = files[: args.limit]

    results: dict[str, dict] = {name: {} for name in ("raw_silero", "focus_silero", "raw_cue", "focus_cue")}
    started = time.monotonic()
    for n, path in enumerate(files, 1):
        meta = meta_by_name.get(path.name)
        if meta is None:
            continue
        raw = load_16k(path)
        for condition, (audio, vad) in conditions(models, raw).items():
            results[condition][path.name] = score(audio, meta, vad)
        if n % 5 == 0 or n == len(files):
            per = (time.monotonic() - started) / n
            left = per * (len(files) - n)
            print(f"  {n}/{len(files)}  {per:.1f}s/clip  ~{left / 60:.0f} min left", flush=True)

    payload = {
        "alkove": getattr(alkove, "__version__", "0.3.3"),
        "sample_rate": SR,
        "clips": len(files),
        "conditions": results,
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
