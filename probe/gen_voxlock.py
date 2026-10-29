"""Produce VoxLock outputs for the benchmark, using VoxLock's own documented protocol.

loud_sample() is described upstream as "the voice sample VoxLock is benchmarked with: the
loudest `sec` seconds in the first `within` seconds", so using it keeps this comparable to
the numbers VoxLock publishes rather than to a convention I invented. load_audio() returns
16 kHz mono, which is also the rate Krisp publishes its outputs at.
"""

import sys
import time
from pathlib import Path

import soundfile as sf
import voxlock

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

vl = voxlock.VoxLock()
files = sorted(src.glob("*.wav"))
print(f"{len(files)} files, SR={voxlock.SR}", flush=True)

t0 = time.time()
for n, f in enumerate(files, 1):
    out = dst / f.name
    if out.exists() and out.stat().st_size > 0:
        continue
    audio = voxlock.load_audio(str(f))
    sample = voxlock.loud_sample(audio)
    cleaned = vl.isolate(audio, sample=sample)
    sf.write(str(out), cleaned, voxlock.SR)
    if n % 10 == 0:
        print(f"  {n}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
print(f"done in {time.time() - t0:.0f}s", flush=True)
