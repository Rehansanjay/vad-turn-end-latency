r"""Turn the per-file turn-decision JSON into the published tables.

    .venv\Scripts\python.exe -m probe.turns_summary results/turns_*.json
    .venv\Scripts\python.exe -m probe.turns_summary --rescore results/turns_*_decisions.json

The first published turn tables were assembled by hand, and that cost a full
re-run over all 265 files the first time a scoring rule changed -- the models
had to be driven again for numbers that were already on disk. So two things
live here now: the tables themselves, and ``--rescore``, which rebuilds them
from the saved decision timestamps alone. A rule change is then arithmetic, and
the audio is only needed when the models change.

Three tables, and they answer different questions:

**Final-turn delay.** How long the caller waits after their last word. Scored on
``final_delay_tol_s``, which accepts a decision from 1.5s before the labelled
end onwards -- these labels run past the voice, so a correct detector can decide
just inside the last span.

**Pauses, by length.** How often a turn is declared in a gap the caller was
going to speak through. A decision in a 0.4s breath is an interruption; one in a
6s handover is the job. Pooling them says nothing, so they stay bucketed.

**Never declared.** Files where no turn was declared at all. Reported both
strictly and with the tolerance, because the difference between those two is
itself the finding, and split by whether the labels left any room to decide in.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from probe.metrics import FINAL_TOL_S, gap_metrics

# Where each split's labels live, for --rescore. The decision files carry
# timestamps, not labels, and scoring needs both.
SPLIT_ROOT = {
    "call_center": "audio/krisp",
    "work": "audio/krisp-work",
    "phone_calls": "audio/krisp-phone",
}

# A file whose labels stop this close to the caller's last word had nowhere to
# put a decision, so it is not evidence of a detector failing to make one.
NO_ROOM_S = 0.45

ORDER = ("silero", "raw_cue", "focus_cue")
LABEL = {"silero": "silero", "raw_cue": "raw -> Cue", "focus_cue": "Focus -> Cue"}


def load(path: Path) -> tuple[str, dict, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["split"], d["conditions"], {k: v for k, v in d.items() if k != "conditions"}


def rescore(path: Path) -> tuple[str, dict, dict]:
    """Rebuild the scores from saved decisions, without touching the audio."""
    d = json.loads(path.read_text(encoding="utf-8"))
    split = d["split"]
    root = Path(SPLIT_ROOT[split])
    rows = {
        json.loads(line)["file_name"]: json.loads(line)
        for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    conds = {
        cond: {name: gap_metrics(rows[name], ts) for name, ts in files.items() if name in rows}
        for cond, files in d["decisions"].items()
    }
    return split, conds, {k: v for k, v in d.items() if k != "decisions"}


def conds_in(splits: dict) -> list[str]:
    seen = {c for _, by_cond in splits.items() for c in by_cond}
    return [c for c in ORDER if c in seen] + sorted(seen - set(ORDER))


def table(header: str, cols: list[str], rows: list[tuple]) -> None:
    print(f"\n{header}\n")
    print("| " + " | ".join([cols[0]] + [LABEL.get(c, c) for c in cols[1:]]) + " |")
    print("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--rescore", action="store_true", help="rebuild from saved decisions, no audio")
    args = ap.parse_args()

    splits: dict[str, dict] = {}
    runs: dict[str, dict] = {}
    for p in args.paths:
        split, conds, meta = (rescore if args.rescore else load)(Path(p))
        splits[split], runs[split] = conds, meta
    order = [s for s in ("call_center", "work", "phone_calls") if s in splits]
    order += [s for s in splits if s not in order]
    cols = conds_in(splits)

    for s in order:
        print(f"{s}: " + ", ".join(f"{k}={v}" for k, v in runs[s].items()))

    # --- final-turn delay ---------------------------------------------------
    rows = []
    pooled: dict[str, list[float]] = {c: [] for c in cols}
    for s in order:
        row = [f"{s} ({len(splits[s][cols[0]])})"]
        for c in cols:
            xs = [r["final_delay_tol_s"] for r in splits[s][c].values() if r.get("final_delay_tol_s") is not None]
            pooled[c] += xs
            row.append(f"{st.median(xs):.2f}s ({len(xs)})" if xs else "--")
        rows.append(tuple(row))
    total = sum(len(splits[s][cols[0]]) for s in order)
    rows.append(tuple([f"**all {total}**"] + [
        f"**{st.median(pooled[c]):.2f}s** ({len(pooled[c])})" if pooled[c] else "--" for c in cols
    ]))
    table("### Dead air after the caller's last word", ["split"] + cols, rows)

    # --- pauses, by length --------------------------------------------------
    buckets: dict[str, dict[str, list[int]]] = {}
    for s in order:
        for c in cols:
            for r in splits[s][c].values():
                for name, b in (r.get("by_bucket") or {}).items():
                    cell = buckets.setdefault(name, {}).setdefault(c, [0, 0])
                    cell[0] += b["declared"]
                    cell[1] += b["gaps"]

    def bkey(n: str) -> float:
        return float(n.lstrip("<>=").rstrip("s")) + (0.5 if n.startswith(">") else 0.0)

    rows = []
    for name in sorted(buckets, key=bkey):
        gaps = max(buckets[name][c][1] for c in cols if c in buckets[name])
        row = [f"{name} ({gaps})"]
        for c in cols:
            d, g = buckets[name].get(c, [0, 0])
            row.append(f"{d / g:.0%}" if g else "--")
        rows.append(tuple(row))
    table("### Declared inside a pause, by how long the caller paused", ["pause length"] + cols, rows)

    # --- never declared -----------------------------------------------------
    rows = []
    for c in cols:
        strict = tol = no_room = 0
        for s in order:
            for r in splits[s][c].values():
                if r.get("final_delay_s") is None:
                    strict += 1
                if r.get("final_delay_tol_s") is None:
                    tol += 1
                    if (r.get("tail_after_final_s") or 0.0) < NO_ROOM_S:
                        no_room += 1
        rows.append((LABEL.get(c, c), strict, tol, no_room, tol - no_room))
    table(
        "### Never declared a final turn",
        ["condition", "strictly after the end", f"within {FINAL_TOL_S}s of it",
         f"of those, no room (<{NO_ROOM_S}s tail)", "unexplained"],
        rows,
    )


if __name__ == "__main__":
    main()
