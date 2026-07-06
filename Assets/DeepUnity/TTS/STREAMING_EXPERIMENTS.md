# Windowed / true-streaming S3Gen — experiment log (PAUSED 2026-07-03)

Goal: replace the clause-level pipeline (ChatterboxVoice pumps whole clauses through
T3→S3Gen) with **true streaming**: S3Gen renders 1s mel windows while T3 is still decoding
the same utterance, audio pushed to the ring buffer per window. Feasibility was probed
Python-side only (`validation/s3_stream_probe.py`, WSL `chatterbox` env) — **nothing of this
is ported to Unity**. Verdict when paused: *close but an audible onset/seam artifact remains*.

## Probe design (controlled A/B)

One SAMPLED T3 token sequence (hooked out of a seeded `tts.generate`; greedy decoding
collapses into stretched near-silence on multi-sentence texts — do NOT use greedy), one
global z-noise indexed by absolute mel position, elder_ref voice, 13s two-sentence line.
Baseline `wav_full` = full-context S3Gen on the same tokens. All windowed variants differ
from it only by windowing.

Run: `conda activate chatterbox && export HF_HOME=/mnt/c/Users/Radu/.cache/huggingface &&
python s3_stream_probe.py` (script self-contained; writes wavs + numeric gate to scratchpad
path at top of file — repoint `SCRATCH` before running).

## What was established (keep these — they're measured, not guessed)

1. **Vocoder (HiFT) windowing is SOLVED and numerically transparent.**
   - The last frames of ANY `voc.decode` render are conv/iSTFT edge-garbage. Measured
     contamination profile (frames from a cut edge → artifact rel. to signal):
     right edge 0f:−2dB 4f:−15dB **8f:−44dB** 16f:−59dB; left edge similar (8f:−45dB).
   - Keeping rendered samples that touch the window edge was THE dominant artifact
     (+3–5 dB LOUDER than the speech in the 40–100ms before every seam — this was the
     original "Ah | Ah another" stutter; flow-window size never mattered for it).
   - Fix: window layout `[st−CM … st … en … en+RM]`, decode all of it, keep only `[st,en)`,
     with **CM=32 left context, RM=16 right render margin** (mel frames; 20ms/frame), 10ms
     overlap-crossfade of two renderings of the SAME instant at each seam.
   - NSF phase continuity: generate the source GLOBALLY (or carry cumsum phase + f0-RNN
     state in a real streaming impl) and slice per window — never restart phase per window.
   - Numeric gate after fix: worst 20ms bin of (windowed−full) = **−40 dB** (pause-floored
     denominator). Inaudible; vocoder side needs no further work.
   - Cost: vocoder trails the flow by RM=16 frames (320ms). Vocoder is the cheap stage.

2. **Flow (meanflow estimator) windowing works mid-stream but the onset artifact remains.**
   - Naive windowing (each window re-renders its left context fresh from noise) → the
     window continues its OWN take, not the emitted one → audible "take switch" at seams.
   - v9 = **teacher-forced context**: emitted mel of the context region goes into the
     estimator's `cond` channel (same pathway as the voice-prompt mel; prompt always
     prepended). `run_flow(tokens[a:b], z_abs=2a, ctx_mel=emitted[2a:2s])`.
     This is CosyVoice2's chunked-flow mechanism and is the right design — keep it.
   - Config when paused: W=25 tok (1s) windows, C=50 (2s) context, la=5 lookahead,
     W0=50 first window. mel-MSE vs full ≈ 0.08–0.10 (NOT a quality metric — teacher-forced
     continuations legitimately diverge; ear is the gate).
   - User verdict on v9 (`wav_stream_G`): "meh.. the bug still exists" — a subtle
     discontinuity around the onset/first seam ("Ah, | another"), everything after smooth.

## Where the remaining bug most likely lives (next steps, in order)

1. **The encoder is still windowed.** `mu` comes from UpsampleConformerEncoder over
   prompt+window tokens only — mu at/near the seam differs between windows even though the
   estimator context is teacher-forced. NEXT PROBE: windowed estimator + FULL-history
   encoder (encoder is cheap; in a real impl it can re-run over all tokens-so-far each
   window). If that kills the artifact → port design = full encoder + windowed estimator.
2. Localize the artifact objectively before more ear rounds: spectrogram-diff stream_G vs
   full around the reported region (the probe's diff/spectrogram snippets in the session
   scratchpad show how); confirm whether it sits INSIDE the first window (then it's
   right-truncation of the estimator, try la=15–25) or AT the first seam (encoder suspect).
3. z-noise semantics for the context region: we keep global noise there; CosyVoice2's cache
   semantics differ — worth checking their exact chunked inference once more.

## Trap log (cost real time — don't repeat)

- **Never batch-open audio files** (`Start-Process` × N = simultaneous playback): produced
  phantom "AH Ah AH" stutter reports for two full iterations. One file at a time.
- **Greedy T3 on long text collapses** (934 tok / 37s of quiet mumble for a 10s line, peak
  0.003). All-variant near-silence == degenerate tokens, not a pipeline bug.
- The manual NSF-source math in the probe is verified EXACT vs `voc.inference`
  (peak-identical on same mel) — don't re-suspect it.
- Per-bin artifact/signal dB needs a pause floor (≥30% of global RMS) or silent bins
  dominate the ranking.

## Latency picture (why this is worth resuming)

Clause pipeline today: first sound after first clause fully through T3+S3Gen. Streaming
target: first sound after W0+la tokens (~2s of tokens ≈ 1.4–1.9s wall on the 4060) and
CONTINUOUS audio after — no inter-clause gaps, no burstiness on long replies.

## Unity port sketch (when quality passes)

S3GenModel: window driver (teacher-forced cond = emitted mel slice; prompt prepended per
window; global-noise slice per absolute position), full-history encoder if probe #1 above
confirms, vocoder with CM=32/RM=16 + crossfade, carried NSF phase (cumsum offset) + f0
predictor state, ChatterboxVoice pushes per window instead of per clause. T3 side needs
nothing — tokens already stream one by one (`onSpeechToken`).
