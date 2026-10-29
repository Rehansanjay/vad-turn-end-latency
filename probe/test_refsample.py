"""Does VoxLock lose the caller because loud_sample picks the wrong speaker?

On the phone_calls split VoxLock removes the primary on a lot of files. Its documented
protocol takes the reference voice from loud_sample(): the loudest 3s inside the first 15s.
If the other party dominates that window, the reference locks onto them and VoxLock then
isolates the wrong person.

Test: re-run the failing files with a reference taken from a span the dataset LABELS as
primary, instead of from loud_sample. If preservation jumps, the protocol is the problem,
not the model.
"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf, voxlock

root = Path(sys.argv[1])
files = sys.argv[2:]
rows = {json.loads(l)["file_name"]: json.loads(l)
        for l in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
vl = voxlock.VoxLock()
SR = voxlock.SR

def rms(x): return float(np.sqrt((x**2).mean())) if len(x) else 0.0

print(f"{'file':10}{'raw':>10}{'loud_sample':>13}{'labelled ref':>14}   verdict")
print("-"*62)
for name in files:
    audio = voxlock.load_audio(str(root / "raw" / name))
    meta = rows[name]
    pri = [(s["start"], s["end"]) for s in meta["segments"] if s["label"] in ("primary", "mix")]
    if not pri:
        print(f"  {name}: no primary span"); continue
    # level over labelled primary spans
    def level(x):
        parts=[x[int(a*SR):int(b*SR)] for a,b in pri]
        parts=[p for p in parts if len(p)]
        return rms(np.concatenate(parts)) if parts else 0.0

    base = level(audio)
    out_loud = vl.isolate(audio, sample=voxlock.loud_sample(audio))
    # reference taken from the longest labelled primary span, capped at 3s
    a,b = max(pri, key=lambda s: s[1]-s[0])
    ref = audio[int(a*SR):min(int(b*SR), int(a*SR)+3*SR)]
    out_ref = vl.isolate(audio, sample=ref)

    l_loud, l_ref = level(out_loud), level(out_ref)
    r1, r2 = (l_loud/base if base else 0), (l_ref/base if base else 0)
    verdict = "FIXED by labelled ref" if r2 > 0.6 and r1 < 0.4 else ("still lost" if r2 < 0.4 else "")
    print(f"{name:10}{base:10.5f}{l_loud:13.5f}{l_ref:14.5f}   loud={r1:.3f} ref={r2:.3f}  {verdict}", flush=True)
