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

## Measured: the Krisp Voice Isolation Benchmark

The demo-pair section above has one real weakness — no ground truth for where the caller stops,
so "turn end" was VAD-attributed rather than scored. The benchmark removes that. Its raw dataset
labels every span `primary`, `mix`, `secondary` or `noise`, so a VAD onset inside a
background-only span is wrong *by construction*, and turn-end delay can be measured from the end
of the last primary-bearing span instead of guessed.

106 files, the `call_center` split, 163 minutes of audio per condition. Three conditions: the raw
mix, **Krisp VI 2.5 default** as Krisp publishes it, and **VoxLock** run through its own
documented protocol (`load_audio` -> `loud_sample` -> `isolate`). Everything resampled to 16 kHz
first, because the raw mixes are 32 kHz and Krisp publishes at 16 kHz — comparing them untouched
would measure sample rate rather than isolation.

### False triggers: VAD onsets inside background-only spans

| condition | false triggers | files affected (of 105) |
| --- | --- | --- |
| raw mix | **229** | 68 |
| Krisp VI 2.5 | **16** | 10 |
| **VoxLock** | **2** | **2** |

### Turn-end delay: dead air after the caller's last labelled word

The median is near zero in every condition and it is the wrong statistic. This is a tail
phenomenon: most calls have no trailing background loud enough to matter, and about a third are
ruined by it. Across the 52 files whose last labelled span is background:

| condition | median | p90 | worst | files over 0.5s | total dead air |
| --- | --- | --- | --- | --- | --- |
| raw mix | -0.06s | 5.22s | **26.64s** | **18 of 52** | **99.3s** |
| Krisp VI 2.5 | -0.21s | -0.02s | 3.51s | 1 | 3.5s |
| **VoxLock** | -0.20s | -0.04s | **0.17s** | **0** | **0.0s** |

Worst cases, raw to VoxLock: `047.wav` 26.64s -> -0.59s, `059.wav` 14.72s -> -0.16s,
`087.wav` 10.23s -> -0.20s. Twenty-six seconds is an agent that believes the caller is still
talking for nearly half a minute after they stopped.

### The attenuator control

| condition | rms primary | rms background | separation |
| --- | --- | --- | --- |
| raw mix | 0.04473 | 0.00361 | 12.4 |
| Krisp VI 2.5 | 0.04371 | 0.00029 | 149.0 |
| **VoxLock** | **0.04726** | **0.00016** | **288.7** |

Levels are taken over the labelled spans rather than fixed windows. Neither filter is an
attenuator, and **VoxLock's primary comes out slightly louder than the raw mix** while its
background is the quietest of the three.

That last point corrects something stated earlier in this repo. On the three demo pairs the
primary appeared to drop 15-25%, and that was reported as a caveat for downstream gain control.
Over 106 labelled files it does not hold: measured across the actual primary spans, VoxLock
preserves the caller at or slightly above the input level. The earlier figure was an artefact of
three clips and a fixed 5-second window.

### Reproducing

```
python probe/fetch_krisp.py audio/krisp            # raw + vi_2_5_default + labels
python probe/gen_voxlock.py audio/krisp/raw audio/krisp/voxlock
python probe/krisp_bench.py audio/krisp results/krisp_call_center.json raw vi25 voxlock
python probe/summarise.py results/krisp_call_center.json
```

Audio is not redistributed here. Both datasets are
[Krisp-AI/VoiceIsolation-Benchmark-Dataset](https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset)
and its `-Processed` companion, **CC BY-NC 4.0**, credit to Krisp — including for publishing
their own model's output, which is what makes a side-by-side comparison possible at all.

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
