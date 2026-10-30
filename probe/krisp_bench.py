"""Turn-end latency and true VAD false triggers on the Krisp Voice Isolation Benchmark.

The demo-pair measurement in this repo had one real weakness: no ground truth for where the
primary speaker stops, so "turn end" was VAD-attributed rather than scored. The benchmark's
raw dataset carries per-segment labels, which removes that.

Labels, from Krisp-AI/VoiceIsolation-Benchmark-Dataset metadata.jsonl:

    primary    the caller alone
    mix        caller plus background
    secondary  background speech only (text: null)
    noise      non-speech noise

So two metrics become scoreable rather than inferred:

  false triggers   VAD speech onsets that begin inside a `secondary` or `noise` span.
                   The caller is not speaking there, so an onset is wrong by construction.

  turn-end delay   how long the VAD keeps calling it speech after the END of the last
                   primary-bearing span (`primary` or `mix`). That is the dead air a caller
                   sits through before the agent believes they finished.

A filter that merely attenuates would improve both, which is why rms_primary/rms_background
are reported alongside: an isolator separates them, an attenuator moves both.

Everything is resampled to 16 kHz first. The raw mixes are 32 kHz and Krisp publishes 16 kHz
output, so comparing them untouched would measure sample rate, not isolation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from livekit.plugins.silero.onnx_model import OnnxModel, new_inference_session

from probe.metrics import score

SR = 16000
WINDOW = 512
ACTIVATION = 0.5
DEACTIVATION = 0.35
MIN_SPEECH = 0.05
MIN_SILENCE = 0.55



def load_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = audio.mean(axis=1)
    return x if sr == SR else soxr.resample(x, sr, SR).astype(np.float32)


def segments_from(x: np.ndarray) -> list[tuple[float, float]]:
    """VAD spans, using LiveKit's own inference and its own thresholds."""
    model = OnnxModel(onnx_session=new_inference_session(force_cpu=True), sample_rate=SR)
    step = WINDOW / SR
    out: list[tuple[float, float]] = []
    speaking = False
    start = silence = 0.0
    i = 0
    for off in range(0, len(x) - WINDOW + 1, WINDOW):
        p = model(x[off : off + WINDOW][None, :])
        t = i * step
        i += 1
        if not speaking:
            if p >= ACTIVATION:
                speaking, start, silence = True, t, 0.0
        elif p < DEACTIVATION:
            silence += step
            if silence >= MIN_SILENCE:
                end = t - silence
                if end - start >= MIN_SPEECH:
                    out.append((start, end))
                speaking = False
        else:
            silence = 0.0
    if speaking:
        end = i * step
        if end - start >= MIN_SPEECH:
            out.append((start, end))
    return out


def main() -> None:
    root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    conditions = {name: root / name for name in sys.argv[3:]}

    rows = [
        json.loads(line)
        for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta_by_name = {r["file_name"]: r for r in rows}

    results: dict[str, dict] = {}
    for cond, folder in conditions.items():
        per_file = {}
        files = sorted(folder.glob("*.wav"))
        for n, f in enumerate(files, 1):
            meta = meta_by_name.get(f.name)
            if meta is None:
                continue
            x = load_16k(f)
            per_file[f.name] = score(x, meta, segments_from(x))
            if n % 20 == 0:
                print(f"  {cond}: {n}/{len(files)}", flush=True)
        results[cond] = per_file
        print(f"  {cond}: done ({len(per_file)} files)", flush=True)

    out_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
