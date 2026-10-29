import json, sys, urllib.request as u
from pathlib import Path

RAW = "https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset/resolve/main"
PRO = "https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset-Processed/resolve/main"
root = Path(sys.argv[1]); fold = "call_center"

meta = root / "metadata.jsonl"
if not meta.exists():
    u.urlretrieve(f"{RAW}/{fold}/metadata.jsonl", meta)
rows = [json.loads(l) for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"{len(rows)} labelled files", flush=True)

for i, r in enumerate(rows, 1):
    name = r["file_name"]
    for base, sub, dest in ((RAW, "", "raw"), (PRO, "vi_2_5_default/", "vi25")):
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
