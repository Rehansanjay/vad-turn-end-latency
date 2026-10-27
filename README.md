# VAD turn-end latency

**Word error rate does not tell you how long your agent sat mute.**

A voice isolation filter is usually sold on transcription accuracy: this much lower WER,
this many fewer background words per minute. Those are real numbers, and they describe the
transcript. They do not describe the thing a caller actually experiences on a voice agent,
which is the pause.

When background speech leaks through to voice activity detection, the VAD keeps reporting
speech after the caller has stopped talking. The agent believes the turn is still running,
so it waits. The caller hears nothing and assumes the agent is broken. That happens on
every turn, whether or not a single word was mis-transcribed.

This repo measures that pause.

## Method

The VAD is the `silero_vad.onnx` that ships with `livekit-plugins-silero`, driven through
LiveKit's own `OnnxModel` wrapper, at LiveKit's own defaults:

| parameter | value |
| --- | --- |
| activation threshold | 0.5 |
| deactivation threshold | 0.35 (their rule: `max(activation - 0.15, 0.01)`) |
| min speech duration | 0.05s |
| min silence duration | 0.55s |
| sample rate | 16000 |

So a speech frame here means what it means to a LiveKit agent: the turn has not ended yet.

Given matched unfiltered/filtered pairs of the same clip, `probe/vad_turn_end.py` reports
**turn end** (the last moment the VAD still calls the audio speech), speech-labelled time
as a fraction of the clip, the segment boundaries, and an RMS control.

### The control matters more than the result

A filter that simply turned the volume down would also reduce VAD speech time, and would
deserve no credit for it. So every measurement is reported with RMS in two windows: the
first 5 seconds, where the primary speaker is talking and a good filter must *keep* the
signal, and the last 3 seconds, where only background remains and a good filter should
*remove* it. A filter that moves both by the same factor is an attenuator, not an isolator.

## Measured: VoxLock

[VoxLock](https://github.com/arbazsiddiqui/voxlock) is an open-weights voice isolator for
voice agents (27M parameters, Apache-2.0, real-time on CPU, ships a Pipecat audio filter).
It publishes before/after demo pairs, which is what is measured here -- its own material,
not a case chosen to make it look bad.

| clip | turn end, unfiltered | turn end, VoxLock | change |
| --- | --- | --- | --- |
| `bystander_000` | 13.98s | 9.79s | **-4.19s** |
| `tail_024` | 11.97s | 7.65s | **-4.32s** |
| `tail_028` | 11.36s | 7.04s | **-4.32s** |

Speech-labelled time falls from 90-97% of each clip to 51-67%.

**It is not an attenuator.** The control separates cleanly:

| clip | RMS, primary speaker talking | RMS, background only |
| --- | --- | --- |
| `bystander_000` | 0.06318 -> 0.04716 | 0.03374 -> **0.00013** |
| `tail_024` | 0.06409 -> 0.05248 | 0.03618 -> **0.00017** |
| `tail_028` | 0.05904 -> 0.05072 | 0.01986 -> **0.00004** |

The primary is held within roughly a quarter of its original level while the background
tail goes to the noise floor -- a selectivity ratio of two to three orders of magnitude.
That separation, not the turn-end number, is the reason the turn-end number exists.

One side effect worth knowing about: the 15-25% drop on the primary is consistent enough
that anything downstream doing its own gain control or energy thresholding will see a
quieter signal than it did before the filter.

### What this is not

- **Three clips**, from the project's own demo set.
- **Lossy input.** The published pairs are mp3, not raw model output.
- **No ground-truth labels** for where the primary speaker actually stops. These are
  VAD-attributed speech times, not a scored false-trigger rate.
- **Not a false-trigger count.** At `min_silence = 0.55s` continuous background merges into
  one long segment instead of producing new events, so event counts barely move
  (2 -> 1, 1 -> 1, 1 -> 2). The latency is what moves. If you want a false-trigger rate,
  lower `min_silence` and count onsets -- the harness prints segment boundaries so merges
  and re-triggers stay visible.

## Reproducing

```
pip install livekit-plugins-silero soundfile numpy
python probe/fetch_voxlock_demo.py audio/voxlock
python probe/vad_turn_end.py audio/voxlock/*.mp3 --json results/voxlock.json
```

The demo audio belongs to the VoxLock project and is not redistributed here; the fetch
script pulls it from upstream. Raw output is in [`results/voxlock.json`](results/voxlock.json).

## A note on getting the VAD right

The first version of this harness drove the ONNX graph directly and reported **zero speech
events on the unfiltered clips** -- audio with an RMS of 0.053 and a clear human voice in
it. silero v5 expects a 576-sample buffer: 64 samples of context carried from the previous
window, then the 512-sample window itself. Feed it a bare 512 and the probabilities never
leave the noise floor (max 0.003 across a 14-second clip).

The reason that got caught rather than published is the control. A filter cannot be
credited for silencing audio that the unfiltered baseline also calls silent. If your
baseline reports nothing, the baseline is broken -- not the audio.

`probe/vad_turn_end.py` now calls LiveKit's `OnnxModel`, which owns that detail.

## Adding another filter

Any filter that can produce a matched `<clip>_off` / `<clip>_on` pair at 16 kHz can be
measured the same way; pairs are matched on the filename stem. Krisp, ElevenLabs and Sanas
all publish comparable material.
