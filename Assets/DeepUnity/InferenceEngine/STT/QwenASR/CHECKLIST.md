# Qwen3-ASR — orchestrator verification checklist

Sections: D1 (Unity build-out) first, then the original D0 (research/ground truth) gates.

## D1. Build-out verification

### D1.1 CPU parity twin (the graded implementation — WS-B/Kokoro harness pattern)
- [ ] `cd Assets/DeepUnity/STT/QwenASR/validation/harness && dotnet run -c Release` prints
      **ALL GATES PASS** — verified in D1 for 0.6b × 3 clips AND 1.7b spot-check (clip2):
      mel corr 1.000000 · prompt_ids EXACT · enc_out corr 1.000000 · proj_out corr 1.000000 ·
      logits_step0 corr 1.000000 + argmax match · tokens_greedy EXACT · transcript EXACT.
- [ ] Optional full 1.7b sweep: `dotnet run -c Release -- 1.7b` (~40 s/clip, ~8 GB RAM).

### D1.2 Files delivered
- [ ] `QwenASRConfig.cs`, `QwenASRTokenizer.cs`, `QwenASRTensors.cs`, `QwenASRCPU.cs` — pure C#
      (no UnityEngine), shared by Unity and the harness.
- [ ] `QwenASRWeights.cs` — VERBATIM CosyVoiceWeights residency pattern (live budget, BeginLoad/
      Defetch epoch, every-exit-path cleanup); only names/messages differ.
- [ ] `QwenASRModel.cs` — GPU orchestration (Qwen3_5Model dispatch idioms; FP32 KV v1).
- [ ] `QwenASR.cs` — `QwenASRSTT : STT` (ModelBase residency contract + Transcribe coroutine,
      Language forcing + Context injection per SPEC §5).
- [ ] `Assets/Resources/ComputeShaders/QwenASRCS.compute` — 26 kernels; **all compile under
      `fxc /T cs_5_0` incl. the INT8_WEIGHTS variant** (verified in D1 without launching Unity).
- [ ] `PTT_NOTES.md` — Microphone wiring design (demo task, deferred).

### D1.3 Orchestrator actions (files this workstream may NOT touch)
- [ ] **Register the shader in `Main/DeepUnityMeta.cs`** (main workstream owns it): add
      `internal static ComputeShader QwenASRCS => Get(ref _qwenASRCS, "QwenASRCS");` + backing
      field, then switch `QwenASRModel`'s ctor from its `Resources.Load` fallback to
      `DeepUnityMeta.QwenASRCS`. (The fallback works as-is; registration is for convention/lazy-load
      uniformity.)
- [ ] Offline C# compile check ran against Unity 2022.3.43f1 + Assembly-CSharp: only expected
      cross-assembly `internal` errors (ConsoleMessage/DeepUnityDispatcher) — same-assembly Unity
      compile is expected clean. Confirm by opening the project once.

### D1.4 Unity-side parity probe (first editor session; NOT run in D1 — no Unity launches allowed)
- [ ] Editor script: load `clip1_hello.wav` (16 kHz mono, `validation/clips/`), run
      `QwenASRSTT.Transcribe`, assert transcript equals
      `validation/reference_dumps/0.6b/clip1_hello/transcript.txt`.
- [ ] Stage-level (only if end-to-end fails): AsyncGPUReadback `melBuf` / `encX` / `projBuf` /
      `logitsBuf` after the corresponding dispatches, corr-compare vs `mel.npy` / `enc_out.npy` /
      `proj_out.npy` / `logits_step0.npy` (>0.999; the CPU twin passes 1.000000, so any GPU
      divergence is a dispatch/layout bug, not math).
- [ ] Boot: `new QwenASRSTT()` streams `weights_qwen3asr_0.6b_fp16` without frame drops
      (LoadProgress ramps to 1), `Warmup()` completes, `Defetch()` mid-load doesn't leak
      (re-`Prefetch()` works after).

## D0. Research + ground truth (original gates — all verified in D0)

## 1. License (LICENSE_CHECK.md)
- [ ] Verdict is PASS / Apache-2.0 with per-repo evidence (HF front-matter + API tags + source headers).
- [ ] Spot-check one link, e.g. `https://huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/README.md`
      begins with `license: apache-2.0`.

## 2. Spec (SPEC.md)
- [ ] Mel frontend: 16 kHz / n_fft 400 / hop 160 / hann / center-reflect / 128 slaney bins /
      log10 → global-max−8 clamp → (x+4)/4 / min 8000 samples / pad-to-100-frames.
- [ ] Encoder: per-1s-chunk conv2d ×3 (stride 2, 480 ch) → Linear 7680→d → +13-pos sinusoid PE;
      block-diagonal attention windows of 104 tokens (8 s); pre-LN; erf-GELU; q/k/v/out with bias;
      13 tokens per second of audio.
- [ ] Decoder: stock Qwen3 28L, GQA 16/8×128, QK-norm before full-dim RoPE θ=1e6, RMSNorm 1e-6,
      SwiGLU, tied embeddings, vocab 151936. Dims per size match the config.json blocks in
      `C:/dev/_model_staging/qwen3asr/*/config.json`.
- [ ] Prompt scaffold + special-token ids table (151644/151645/151669/151670/151676/151704) and the
      two usage modes (language-in-system vs context-in-system + `language X<asr_text>` prefill).
- [ ] VRAM table: 0.6B fits 8 GB co-resident in all modes; 1.7B flagged fp16-DOES-NOT-FIT / int8-OK.

## 3. Weights staged (outside Assets)
- [ ] `C:/dev/_model_staging/qwen3asr/Qwen3-ASR-0.6B-hf/model.safetensors` (~1.56 GB)
- [ ] `C:/dev/_model_staging/qwen3asr/Qwen3-ASR-1.7B-hf/model.safetensors` (~4.08 GB)
- [ ] No raw .safetensors/.pt anywhere under `Assets/`.

## 4. Exporter ran (validation/import_qwen3asr.py)
- [ ] `Assets/Resources/DeepUnity/STT/QwenASR/weights_qwen3asr_0.6b_fp16/manifest.tsv` — 628 entries.
- [ ] `Assets/Resources/DeepUnity/STT/QwenASR/weights_qwen3asr_1.7b_fp16/manifest.tsv` — 724 entries.
- [ ] Every manifest line `name\tfile\tdtype\tnumel\tshape` and file size == numel × dtype-bytes
      (verified in D0; re-run: parse manifest, `os.path.getsize` each file).
- [ ] Synthesized tensors present: `frontend/mel_filters` [201,128], `enc/pos_emb` [13,d]
      (pos_emb row 0 = sin-half 0.0 / cos-half 1.0 — checked).
- [ ] `tokenizer/vocab.txt` (151936 lines) + `merges.txt` + `specials.tsv` next to the manifests.
- [ ] fp16 packing/int8-scales conventions identical to ChatterboxWeights.cs expectations
      (loader can be a near-copy; no loader changes needed).

## 5. Reference dumps (validation/dump_reference.py)
- [ ] `validation/reference_dumps/0.6b/<clip>/` and `.../1.7b/<clip>/` exist for the 3 clips in
      `validation/clips/` with: mel / mel_mask / enc_out / proj_out / input_ids / logits_step0 /
      tokens_greedy .npy + raw_output.txt + transcript.txt + meta.json.
- [ ] Each transcript.txt matches the spoken text of its clip
      (clip1: "Hello world, this is a test of the speech recognition system." etc.).
- [ ] meta.json `audio_tokens == expected_audio_tokens_formula` (validates the C#-side token-count
      formula in SPEC §2.1).
- [ ] raw_output.txt shows the `language English<asr_text>...` format from SPEC §5.

## 6. Gate for D1 (build-out) — decision inputs
- [ ] Feasibility verdict + risk list in the D0 final report accepted.
- [ ] Port order agreed (frontend → encoder → decoder-config reuse → prompt/decode → PTT wiring).
- [ ] Default ship target agreed (0.6B fp16/int8 primary; 1.7B int8 optional).
