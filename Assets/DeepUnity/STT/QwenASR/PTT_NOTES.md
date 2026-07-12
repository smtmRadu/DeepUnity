# Qwen3-ASR push-to-talk wiring (design notes — demo task, not built in D1)

D1 ships the model side (`QwenASRSTT.Transcribe(float[] samples, ...)`). This documents how the
demo should feed it from Unity's Microphone API.

## Capture loop (16 kHz mono)

```csharp
// setup (device null = default mic). Loop clip, 30 s ring — utterances are capped anyway.
AudioClip clip = Microphone.Start(null, loop: true, lengthSec: 30, frequency: 16000);
int lastPos = 0;
List<float> utterance = new();       // grows only while PTT is held

// per frame while PTT held:
int pos = Microphone.GetPosition(null);
if (pos != lastPos)
{
    int count = (pos - lastPos + clip.samples) % clip.samples;
    float[] chunk = new float[count];
    clip.GetData(chunk, lastPos);    // wraps: for the wrap case read in two GetData calls
    utterance.AddRange(chunk);
    lastPos = pos;
}
```

Notes:
- `Microphone.Start(freq: 16000)` — Unity resamples if the device can't do 16 kHz natively; if the
  platform ignores the request (check `clip.frequency`), capture at 48000 and decimate ×3 with a
  small FIR (the model REQUIRES 16 kHz — SPEC §1).
- Mono is the default for mic clips; assert `clip.channels == 1`.
- Keep `Microphone.Start` running across utterances (starting the mic is the slow part, ~100+ ms);
  gate only the copy loop on the PTT key.

## Utterance lifecycle

1. **PTT down**: reset `utterance`, snapshot `lastPos = Microphone.GetPosition(null)`.
2. **held**: drain the ring each frame (above). Hard cap ~20 s (`utterance.Count > 320_000` → force
   release) — decoder KV capacity (default 1024) fits ~20 s ⇒ 13·20+scaffold+output ≈ 400 tokens.
3. **PTT up**: `StartCoroutine(stt.Transcribe(utterance.ToArray(), OnTranscript))`. Optionally trim
   leading/trailing silence (RMS < threshold for > 200 ms) to cut audio tokens — accuracy-neutral,
   latency-positive. Clips < 0.5 s are auto-padded by the model (min 8000 samples).
4. Ignore PTT presses while a transcription is in flight (or queue one), `QwenASRSTT` is
   single-stream — one Transcribe coroutine at a time per instance.

## Latency budget (0.6B fp16, 4060-laptop class — SPEC §9)

| stage | 5 s utterance |
|---|---|
| mel+conv+encoder+projector | ~0.2-0.4 s (65 tokens through 18 layers) |
| prefill (~80 tok) | <1 s |
| greedy decode (~20 tok) | ~0.7-1 s |
| **total after release** | **~1.5-2.5 s** |

`Transcribe` yields between every layer/step — no frame is blocked; the AsyncGPUReadback per decode
step keeps the main thread free.

## Options

- `stt.Language = "English"` — skips language detection (forced-language prefill, SPEC §5); worth
  setting in a known-locale game for a few tokens of latency and zero LID errors.
- `stt.Context = "Eldoria, Vael'thar, mana potion"` — context injection biases rare names/jargon
  (goes into the system slot verbatim).
- Warmup: run `StartCoroutine(stt.Warmup())` behind the loading screen — first transcription
  otherwise pays every kernel's driver ISA compile (~hundreds of ms).

## v2 — live captions (deferred; reference semantics in SPEC §7)

Re-run the full pipeline every 2 s of accumulated audio with the previously decoded text (minus the
last 5 tokens) prefilled after `<asr_text>`; first 2 chunks use no prefix. Cost per update = one
fresh prefill (audio tokens grow 26/update) — acceptable ≤15 s but strictly worse than offline for
final accuracy; ship v1 first.
