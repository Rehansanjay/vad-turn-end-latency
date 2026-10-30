"""Fetch one split of the Krisp Voice Isolation Benchmark.

    python -m probe.fetch_krisp audio/krisp                  # call_center, raw + vi25
    python -m probe.fetch_krisp audio/krisp-work work --raw-only

The split is an argument because there are three and they do not behave alike:
`phone_calls` is band-limited to 4 kHz despite being stored at 32 kHz, which is
where filters tend to fail.

`--raw-only` skips Krisp's own processed output. The Krisp comparison needs it,
because that is the filter under test there. The alkove run does not: Focus is
applied in process by `alkove_bench.py`, so fetching vi25 would double the
transfer for files nothing reads.

Each split is a few hundred megabytes and `audio/` is gitignored, so a split
that has been measured and committed can be deleted and refetched later.
"""

from __future__ import annotations

import json
import sys
import urllib.request as u
from pathlib import Path

RAW = "https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset/resolve/main"
PRO = "https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset-Processed/resolve/main"

SPLITS = ("call_center", "work", "phone_calls")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raw_only = "--raw-only" in sys.argv

    root = Path(args[0])
    fold = args[1] if len(args) > 1 else "call_center"
    if fold not in SPLITS:
        raise SystemExit(f"unknown split {fold!r}; expected one of {', '.join(SPLITS)}")

    sources = [(RAW, "", "raw")]
    if not raw_only:
        sources.append((PRO, "vi_2_5_default/", "vi25"))
    for _, _, dest in sources:
        (root / dest).mkdir(parents=True, exist_ok=True)

    meta = root / "metadata.jsonl"
    if not meta.exists():
        u.urlretrieve(f"{RAW}/{fold}/metadata.jsonl", meta)
    rows = [json.loads(line) for line in meta.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"{fold}: {len(rows)} labelled files into {root}", flush=True)

    for i, row in enumerate(rows, 1):
        name = row["file_name"]
        for base, sub, dest in sources:
            out = root / dest / name
            if out.exists() and out.stat().st_size > 0:
                continue
            try:
                u.urlretrieve(f"{base}/{fold}/{sub}{name}", out)
            except Exception as e:
                print(f"  MISS {dest}/{name}: {e}", flush=True)
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
