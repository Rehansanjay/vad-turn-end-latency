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

### The other two splits, and where VoxLock fails

`call_center` is one of three splits. Running the same measurement over all 265 files changes
the picture in a way one split did not show.

**`work`** (96 files, 95 with background, 83 turn-end testable) — the richest background material
in the dataset:

| condition | false triggers | files affected | mean turn-end delay | rms primary | rms background | separation |
| --- | --- | --- | --- | --- | --- | --- |
| raw mix | **263** | 75 | 2.05s | 0.05528 | 0.01125 | 4.9 |
| Krisp VI 2.5 | 19 | 14 | 0.35s | 0.05229 | 0.00058 | 90.2 |
| **VoxLock** | **12** | **9** | **-0.29s** | **0.05550** | **0.00029** | **194.5** |

Same shape as `call_center`: VoxLock ahead on false triggers, and the only condition whose mean
turn-end delay is negative — Krisp still leaves 0.35s on average.

**`phone_calls`** (63 files) is where it breaks, and the honest answer is two-sided.

First, the split cannot carry much weight: only **9 of 63 files contain any background span** and
only **4 are turn-end testable**, so the turn-end means there are a single file's behaviour. That
file, `143.wav`, has a 46-second background tail and is a clean loss for VoxLock — raw 26.20s,
Krisp **0.16s**, VoxLock **13.02s**.

Second, and more seriously, **VoxLock removes the primary speaker on a large minority of this
split.** Primary level preserved relative to the raw mix, measured over the labelled primary
spans:

| split | median preserved | files below 80% | worst |
| --- | --- | --- | --- |
| call_center | 1.066 | **0 of 106** | 0.916 |
| work | 1.038 | 4 of 96 | 0.442 |
| **phone_calls** | **0.855** | **27 of 63** | **0.022** |

On `161.wav` the caller is reduced to 2.2% of input level and VAD-detected speech falls from 93%
to 12% — the speaker is gone. `160.wav`: 86% to 5%. Krisp preserves all of these.

The obvious explanation is that VoxLock's documented protocol takes its voice reference from
`loud_sample()` — the loudest 3s of the first 15s — and on a phone call the other party can
dominate that window, locking the reference onto the wrong speaker. **That was tested and it is
mostly not the cause.** Re-running the five worst files with a reference taken from a span the
dataset labels as primary recovered only one of them (`160.wav`, 0.095 -> 0.760). The other four
stayed lost with ground-truth reference audio, so the failure is in the isolation itself rather
than in sample selection.

So: VoxLock is clearly ahead on the two splits with real background, and has a reproducible
failure mode on the third that Krisp does not share.

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

## Measured: alkove Focus and Cue

[alkove](https://github.com/alkove-ai/alkove) is open-weights voice isolation (**Focus**) and a VAD
(**Cue**), with LiveKit and Pipecat plugins. Its maintainer asked for this benchmark and asked
specifically for the parts that do not flatter the models, so those are in here with the rest.

Unlike Krisp, alkove is a library rather than published output, so the filter runs in process and
four conditions are built per clip:

| condition | what it answers |
| --- | --- |
| raw -> Silero | the baseline the rest of this repo uses |
| Focus -> Silero | what Focus does to the VAD most people actually run |
| raw -> Cue | Cue on its own, unfiltered |
| **Focus -> Cue** | **how alkove's own LiveKit plugin runs** |

Three splits of the Krisp benchmark, 265 labelled files, `alkove==0.3.3`. Two alkove passes per
clip rather than four: `focus=True, cue=True` returns the filtered audio and Cue's spans over it
together, so the filtered audio is computed once and the Silero pass reuses it. Each condition is
scored against the audio its own VAD heard, so the RMS control describes what the VAD got.

`phone_calls` is fed **as-is at its stored 32 kHz** on the maintainer's instruction: the core
resamples everything to 16 kHz before Focus and Cue anyway, and as-is is what a LiveKit agent sees
on a SIP call — phone content at the session rate.

### Primary recall: the metric a filter cannot game

Every other metric here improves when a filter destroys the caller. Remove the primary speaker and
false triggers, dead air and background leak all fall at once. Recall is what catches that, and it
is measurable on all 265 files.

| split | raw -> Silero | Focus -> Silero | raw -> Cue | Focus -> Cue |
| --- | --- | --- | --- | --- |
| call_center (106) | **0.923** | 0.901 | 0.931 | 0.899 |
| work (96) | **0.967** | 0.948 | 0.973 | 0.946 |
| phone_calls (63) | **0.927** | 0.906 | **0.844** | 0.897 |

Focus costs recall on the two wideband splits — on `call_center`, **94 of 106 clips get worse, 11
get better and 1 is unchanged**, mean 0.923 to 0.901, worst case 0.819 to 0.743. That is the price of the numbers below and
it should be read alongside them.

### Background speech: VAD speech inside background-only spans

`false_triggers` counts onsets, so it only sees a VAD that switches *on* during background. When the
caller goes quiet and a secondary voice carries on, a VAD that simply stays on produces no new onset
at all. This counts the time instead.

| split | raw -> Silero | Focus -> Silero | raw -> Cue | Focus -> Cue |
| --- | --- | --- | --- | --- |
| call_center | 31.2% | 17.6% | **47.7%** | **14.9%** |
| work | 42.4% | 24.6% | **53.4%** | **23.8%** |

**Cue alone hears more background than Silero alone, not less** — 47.7% against 31.2%, and 53.4%
against 42.4%. Same direction on both splits, so it is a property of Cue rather than a quirk of one
set. Paired with Focus it becomes the best of the four.

### Turn-end delay: dead air after the caller's last labelled word

Reported only where the sample supports it. `phone_calls` has **4 testable files** out of 63, so its
turn-end numbers are omitted rather than published.

| split | raw -> Silero | Focus -> Silero | raw -> Cue | Focus -> Cue | n |
| --- | --- | --- | --- | --- | --- |
| call_center, mean | 1.80s | 1.29s | 2.05s | **1.04s** | 52 |
| work, mean | 2.05s | 1.50s | 2.89s | **0.83s** | 83 |
| work, median | 0.06s | -0.11s | **0.78s** | **-0.19s** | 83 |

That `work` median is the clearest result in the set. **Cue on its own runs late on more than half
the files** — a median of 0.78s, against Silero's 0.06s. Focus takes it to -0.19s. So Cue needs
Focus in a way Silero does not.

### The control: is Focus separating or just attenuating?

| split | condition | RMS primary | RMS background | ratio |
| --- | --- | --- | --- | --- |
| call_center | raw | 0.04473 | 0.00361 | 12.4 |
| call_center | Focus | 0.04606 | 0.00219 | **21.0** |
| work | raw | 0.05528 | 0.01125 | 4.9 |
| work | Focus | 0.05528 | 0.00564 | **9.8** |

On `work` Focus leaves the mean primary level unchanged to the **seventh** decimal — 0.05528062
against 0.05527958, a difference of 0.002% — while halving the background. The two round to the same
five decimals, which looks like a copy-paste error and is not one; all 96 files differ individually
and the means converge. On `call_center` the primary actually rises 3%. Either way the ratio roughly
doubles, so this is separation, not a volume control.

### Where Focus earns its place, and it is not the same reason twice

On `call_center` and `work` it suppresses background the VAD would otherwise hear.

On `phone_calls` there is almost no background to suppress — 9 of 63 files contain a background span
at all. There it does a different job: **it makes the caller visible to Cue.**

`raw -> Cue` on band-limited audio has a healthy median of 0.927 and a brutal tail: **13 of 63 files
below 0.8 recall, 6 below 0.5**, worst **0.078**, where Cue misses 92% of the caller's labelled
speech. Focus brings those back:

| file | raw -> Cue | Focus -> Cue | raw -> Silero |
| --- | --- | --- | --- |
| `154.wav` | **0.078** | 0.756 | 0.995 |
| `109.wav` | 0.141 | 0.853 | 0.842 |
| `152.wav` | 0.275 | 0.965 | 0.966 |
| `157.wav` | 0.284 | 0.902 | 0.871 |

That is 0.3.3 doing what it was built for: it checks the power above ~4.6 kHz and, when that band is
empty, fills it with faint noise. Silero needs none of it — the right-hand column is its unfiltered
recall on the same files. **So Silero is robust to band-limiting and Cue is not**, and on this split
Focus *costs* Silero: files below 0.8 recall go from 1 to 6.

### What this does not establish

- **`phone_calls` is underpowered for everything except recall.** 9 of 63 files have a background
  span, so its background share, false triggers and RMS ratio rest on those 9; 4 are turn-end
  testable, so its turn-end numbers are not reported at all.
- **No voiceprint.** Focus runs in keep-the-closest-voice mode throughout, which is what the LiveKit
  plugin does by default and what suits the headset recordings in `work`. Enrolled-voiceprint
  performance is not measured here.
- **One alkove build.** 0.3.3, released the day these runs happened. Earlier builds fail on
  band-limited audio by the maintainer's own account, so these numbers do not transfer backwards.
- **Three clips are not a result.** A pilot on three `call_center` clips read 2.1% background speech
  for `Focus -> Cue`; over 106 it is 14.9%. The pilot was not wrong so much as meaningless, and the
  same mistake is why `phone_calls` turn-end is absent above.

Raw output: [`results/alkove_call_center.json`](results/alkove_call_center.json),
[`alkove_work.json`](results/alkove_work.json),
[`alkove_phone_calls.json`](results/alkove_phone_calls.json), with summaries beside each.

## Reproducing

```
pip install livekit-plugins-silero soundfile numpy soxr

# the VoxLock demo pairs
python probe/fetch_voxlock_demo.py audio/voxlock
python probe/vad_turn_end.py audio/voxlock/*.mp3 --json results/voxlock.json

# the Krisp benchmark, one split at a time
python -m probe.fetch_krisp audio/krisp call_center
python probe/krisp_bench.py audio/krisp results/krisp_call_center.json raw vi25
python probe/summarise.py results/krisp_call_center.json

# alkove, four conditions, the filter applied in process
pip install "alkove>=0.3.3"
python -m probe.fetch_krisp audio/krisp-work work --raw-only
python -m probe.alkove_bench audio/krisp-work results/alkove_work.json
python probe/summarise.py results/alkove_work.json results/alkove_work_summary.json
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
