# Parakeet-TDT E0+E1 — orchestrator verification checklist

E0 = research + ground truth (sections 1-6). E1 = build-out (section 7), green-lit 2026-07-11.

## 1. License (LICENSE_CHECK.md)
- [ ] Verdict PASS / CC-BY-4.0 with per-repo evidence (README front-matter + HF API tags), and the
      **attribution text draft** present (CC-BY requires credit — stricter than the Apache repos).
- [ ] Spot-check: `https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/raw/main/README.md` begins
      `license: cc-by-4.0`.
- [ ] Gitignored-weights nuance noted (repo ships no CC-BY material; NOTICE shipped anyway).

## 2. Spec (SPEC.md)
- [ ] Frontend: preemph 0.97 → STFT 512/**400 SYMMETRIC hann**/160, center **zero**-pad, |·|² →
      slaney mel 128 (baked tensor) → ln(x+2⁻²⁴) → **per-feature mean/std over the utterance**
      (unbiased, +1e-5); T_mel = floor(len/160) (last stft frame masked).
- [ ] Subsampling: conv 1→256 k3 s2 + 2×(dw+pw) s2, channel-major flatten [256·16=4096] →
      Linear→1024; 80 ms/frame (12.5 fps).
- [ ] Encoder: 24× (½FF 4096 SiLU · rel-pos MHSA 8×128 (bias_u/v + rel-shift) · conv GLU-k9-**BN
      folded**-SiLU · ½FF · LN_out), **NO linear biases** (attention_bias=convolution_bias=false),
      pre-LN LayerNorms.
- [ ] TDT decoder: Embedding(V,640) blank-row-zero, LSTM 2×640, pred_proj 640→640,
      enc_proj 1024→640, joint = head(relu(enc+pred)) → V+5 logits (last 5 = durations 0–4).
- [ ] §6 greedy loop pseudocode with blank/dur-0 forcing + max_symbols=10 guard, LSTM advances on
      non-blank only, timestamps = frame × 80 ms.
- [ ] v2-vs-v3 table (vocab 1025/8193) + v3-first recommendation + STT.cs/ModelBase C# design +
      GPU-encoder/CPU-decoder split + VRAM (~1.26 GB fp16, fits 8 GB co-resident) + latency table.

## 3. Weights staged (outside Assets)
- [ ] `C:/dev/_model_staging/parakeet/parakeet-tdt-0.6b-v3-hf/model.safetensors` (2.51 GB, HF-native)
- [ ] `C:/dev/_model_staging/parakeet/parakeet-tdt-0.6b-v2-nemo/parakeet-tdt-0.6b-v2.nemo` (2.47 GB)
- [ ] `C:/dev/_model_staging/parakeet/parakeet-tdt-0.6b-v2-hf/` (converted via
      `validation/convert_v2_nemo.py`, missing-keys == none)
- [ ] No raw .nemo/.safetensors anywhere under `Assets/`.

## 4. Exporter ran (validation/import_parakeet.py)
- [ ] `Assets/Resources/DeepUnity/STT/Parakeet/weights_parakeet_tdt_0.6b_v3_fp16/manifest.tsv`
      — 653 entries, 1254 MB, ZERO size mismatches (file bytes == numel × dtype).
- [ ] `…/weights_parakeet_tdt_0.6b_v2_fp16/manifest.tsv` — same layout, v2 vocab (dec/embedding
      [1025,640], joint/head.w [1030,640]).
- [ ] Exporter printed `decoder embedding blank row max|w| = 0.00e+00` (both variants) and
      `all checkpoint tensors consumed`.
- [ ] Synthesized tensors present: `frontend/mel_filters` [128,257] (== librosa slaney, checked to
      4e-9), `frontend/window` [400] symmetric hann; BN folded to `conv.bn.scale/.shift`.
- [ ] `tokenizer/vocab.txt` + `specials.tsv` next to each manifest; v3 = 8193 ids, 275 specials,
      **0 byte-fallback tokens** (plain string decode in C#).

## 5. Reference dumps (validation/dump_reference.py)
- [ ] `validation/reference_dumps/{v3,v2}/<clip>/` for the 3 clips in `validation/clips/` with:
      mel / pos_emb / sub_out / enc_layer0 / enc_out / enc_proj / joint_logits_first8 / tokens /
      durations / frames .npy + emissions.tsv + transcript.txt + meta.json.
- [ ] Every meta.json has `manual_matches_generate: true` — the SPEC §6 loop reproduces
      transformers' own decoding verbatim.
- [ ] Transcripts correct incl. punctuation/caps (clip1: "Hello world, this is a test of the
      speech recognition system.").
- [ ] TDT frame-skipping visible: T_enc≈58 covered in ≈23–30 steps (emissions.tsv).

## 6. Gate for E1 (build-out) — decision inputs
- [x] Feasibility verdict + risk list in the E0 final report accepted (coordinator, 2026-07-11).
- [x] Port order agreed: mel → subsampling → 1 conformer block (rel-shift settled vs Chatterbox
      kernel) → ×24 + enc_proj → CPU LSTM/joint loop → tokenizer decode → PTT wiring.
- [x] v3 first (HF-native, multilingual incl. RO), v2 second via same code path; fp16 default,
      int8 encoder = headroom mode.

## 7. E1 build-out — verification

### 7.1 dotnet parity harness (the pre-Unity gate — run it yourself)
- [ ] `cd Assets/DeepUnity/STT/Parakeet/validation/harness && dotnet run -c Release`
      prints **ALL PASS**: for BOTH variants × 3 clips — mel/sub_out/pos_emb/enc_layer0/enc_out/
      enc_proj/joint_first8 all corr ≥ 0.999 (most 1.000000), token+duration+frame sequences
      EXACT, transcripts EXACT string matches vs the HF reference.
      (Achieved 2026-07-11: 0 failures, fp16 weights / fp32 math.)
- [ ] The harness compiles ParakeetConfig/Tensors/Tokenizer/CPU.cs straight from the package —
      any future edit to those files is re-gated by rerunning it.

### 7.2 Files delivered
- [ ] `ParakeetSTT.cs` (: DeepUnity.STT), `ParakeetWeights.cs` (CosyVoiceWeights-pattern
      residency incl. epoch-guarded Defetch + reload-after-defetch; CPU-side tensors excluded
      from GPU upload), `ParakeetConfig.cs`, `ParakeetCPU.cs`, `ParakeetTensors.cs`,
      `ParakeetTokenizer.cs` (pure C#, harness-shared),
      `Assets/Resources/ComputeShaders/ParakeetCS.compute` (9 kernels).
- [ ] Unity-side files stub-compile clean (checked against UnityEngine API stubs at E1; the real
      Unity compile is the orchestrator's first step).

### 7.3 Orchestrator-side steps (things this workstream could NOT do)
- [ ] **Register ParakeetCS.compute in DeepUnityMeta.cs** (shader registry = main-only edits,
      not touched by this workstream per constraints).
- [ ] Open Unity, let it compile; fix any Unity-version API nits (AsyncGPUReadback etc.).
- [ ] In-Unity parity probe: `new ParakeetSTT(ParakeetVariant.V3)` + `Transcribe()` on the
      samples of validation/clips/clip1_hello.wav; expect EXACTLY "Hello world, this is a test
      of the speech recognition system." — this isolates GPU-kernel bugs (the CPU math the
      kernels mirror is already transcript-exact vs HF).
- [ ] Optional deeper probe: readback the residual stream after block 0 and compare with
      validation/reference_dumps/v3/clip1_hello/enc_layer0.npy (corr > 0.999).
- [ ] Measure 4060 latency for 5 s / 15 s clips vs the SPEC §9 table; decide whether the int8
      encoder export (`--quant int8` — exporter + q8 packing already supported) becomes the
      default co-residency mode.
- [ ] Wire a PTT demo (Microphone ring buffer → Transcribe → NPC prompt).
- [ ] Ship attribution: LICENSE_CHECK.md draft into the repo credits/NOTICE.
