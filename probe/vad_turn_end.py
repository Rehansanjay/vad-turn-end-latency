"""Measure what a voice-isolation filter does to VAD turn-end latency.

Word error rate tells you what the transcriber got wrong. It does not tell you how long
the agent sat mute because the VAD could still hear somebody talking. On a voice agent
that second number is the one the caller feels, on every turn.

This drives the silero VAD that livekit-plugins-silero ships, through LiveKit's own
OnnxModel wrapper and at LiveKit's own defaults, over matched unfiltered/filtered pairs of
the same clip. It reports, for each pair:

  * the last moment the VAD still calls the audio speech   -> turn-end latency
  * speech-labelled time as a fraction of the clip
  * segment boundaries, so a merge or a re-trigger is visible
  * an RMS control, split into a window where the primary speaker is talking and a window
    where only background remains

The control exists because a filter that simply attenuates everything would also reduce
VAD speech time, and would deserve no credit for it. Report it next to the result.

Usage:
    python probe/vad_turn_end.py <clip>_off.<ext> <clip>_on.<ext> [...] [--json out.json]

Pairs are matched on the filename stem before the final _off / _on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from livekit.plugins.silero.onnx_model import OnnxModel, new_inference_session

SR = 16000
WINDOW = 512  # silero v5 at 16k; OnnxModel prepends 64 samples of context itself

# LiveKit's silero VAD defaults (livekit-plugins-silero/vad.py). deactivation follows
# their rule: max(activation - 0.15, 0.01).
ACTIVATION = 0.5
DEACTIVATION = 0.35
MIN_SPEECH = 0.05
MIN_SILENCE = 0.55

PRIMARY_WINDOW = 5.0     # seconds from the start, primary speaker active
BACKGROUND_WINDOW = 3.0  # seconds at the end, background only


def load(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise SystemExit(f"{path.name}: expected {SR} Hz, got {sr}")
    return audio.mean(axis=1)


def frame_probs(x: np.ndarray) -> np.ndarray:
    """Per-frame speech probability, using LiveKit's inference rather than our own.

    Driving the ONNX graph by hand is the easy way to get this wrong: silero v5 wants a
    576-sample buffer (64 samples of carried context + a 512-sample window), and feeding
    it a bare 512 yields probabilities that never leave the noise -- zero speech even on
    unfiltered speech. OnnxModel owns that detail.
    """
    model = OnnxModel(onnx_session=new_inference_session(force_cpu=True), sample_rate=SR)
    return np.asarray(
        [model(x[i : i + WINDOW][None, :]) for i in range(0, len(x) - WINDOW + 1, WINDOW)]
    )


def segments(p: np.ndarray) -> list[tuple[float, float]]:
    """Hysteresis and duration gating, the way LiveKit decides a turn has ended."""
    step = WINDOW / SR
    out: list[tuple[float, float]] = []
    speaking = False
    start = silence = 0.0
    for i, v in enumerate(p):
        t = i * step
        if not speaking:
            if v >= ACTIVATION:
                speaking, start, silence = True, t, 0.0
        elif v < DEACTIVATION:
            silence += step
            if silence >= MIN_SILENCE:
                end = t - silence
                if end - start >= MIN_SPEECH:
                    out.append((start, end))
                speaking = False
        else:
            silence = 0.0
    if speaking:
        end = len(p) * step
        if end - start >= MIN_SPEECH:
            out.append((start, end))
    return out


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x**2).mean())) if len(x) else 0.0


def analyse(path: Path) -> dict:
    x = load(path)
    p = frame_probs(x)
    segs = segments(p)
    duration = len(p) * WINDOW / SR
    speech = sum(e - s for s, e in segs)
    n = len(x)
    return {
        "file": path.name,
        "duration_s": round(duration, 2),
        "events": len(segs),
        "speech_s": round(speech, 2),
        "speech_fraction": round(speech / duration, 3) if duration else 0.0,
        "turn_end_s": round(max((e for _, e in segs), default=0.0), 2),
        "segments": [[round(s, 2), round(e, 2)] for s, e in segs],
        "rms_overall": round(rms(x), 5),
        "rms_primary_window": round(rms(x[: int(PRIMARY_WINDOW * SR)]), 5),
        "rms_background_window": round(rms(x[max(0, n - int(BACKGROUND_WINDOW * SR)) :]), 5),
    }


def main(argv: list[str]) -> None:
    out_path = None
    if "--json" in argv:
        i = argv.index("--json")
        out_path = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2 :]
    if not argv:
        raise SystemExit(__doc__)

    clips = {}
    for raw in argv:
        path = Path(raw)
        clips[path.name] = analyse(path)

    print(
        f"VAD: silero v5 via livekit-plugins-silero, LiveKit defaults "
        f"(activation {ACTIVATION}, deactivation {DEACTIVATION}, "
        f"min_speech {MIN_SPEECH}s, min_silence {MIN_SILENCE}s)\n"
    )
    print(f"{'clip':26}{'events':>7}{'speech s':>10}{'% clip':>8}{'turn end':>10}")
    print("-" * 61)
    for c in clips.values():
        print(
            f"{c['file']:26}{c['events']:7d}{c['speech_s']:10.2f}"
            f"{100 * c['speech_fraction']:7.1f}%{c['turn_end_s']:10.2f}"
        )

    pairs = []
    stems = sorted({n.rsplit("_", 1)[0] for n in clips if "_" in n})
    print()
    for stem in stems:
        off = next((c for n, c in clips.items() if n.startswith(f"{stem}_off")), None)
        on = next((c for n, c in clips.items() if n.startswith(f"{stem}_on")), None)
        if not off or not on:
            continue
        delta = round(on["turn_end_s"] - off["turn_end_s"], 2)
        pairs.append({"clip": stem, "off": off, "on": on, "turn_end_delta_s": delta})
        print(f"{stem}")
        print(
            f"   turn end   {off['turn_end_s']:6.2f}s -> {on['turn_end_s']:6.2f}s  "
            f"({delta:+.2f}s)"
        )
        print(
            f"   speech     {100 * off['speech_fraction']:5.1f}%  -> "
            f"{100 * on['speech_fraction']:5.1f}%  of clip"
        )
        print(
            f"   rms primary    {off['rms_primary_window']:.5f} -> "
            f"{on['rms_primary_window']:.5f}   (control: filter must keep this)"
        )
        print(
            f"   rms background {off['rms_background_window']:.5f} -> "
            f"{on['rms_background_window']:.5f}   (control: filter should kill this)"
        )

    if out_path:
        out_path.write_text(
            json.dumps(
                {
                    "vad": {
                        "model": "silero_vad.onnx (livekit-plugins-silero)",
                        "activation_threshold": ACTIVATION,
                        "deactivation_threshold": DEACTIVATION,
                        "min_speech_duration": MIN_SPEECH,
                        "min_silence_duration": MIN_SILENCE,
                        "sample_rate": SR,
                    },
                    "control_windows_s": {
                        "primary": PRIMARY_WINDOW,
                        "background": BACKGROUND_WINDOW,
                    },
                    "clips": list(clips.values()),
                    "pairs": pairs,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
