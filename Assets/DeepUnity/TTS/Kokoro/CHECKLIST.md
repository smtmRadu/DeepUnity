# Kokoro port — orchestrator checklist (what to run/verify in Unity)

Everything below is run by the ORCHESTRATOR (this workstream never launches Unity).
Python prerequisites are already done and verified if the marked artifacts exist.

## B0 artifacts (done, verify presence only)
- [ ] `Assets/DeepUnity/TTS/Kokoro/SPEC.md`, `G2P_PROPOSAL.md` (decision: option (b), approved)
- [ ] `Assets/Resources/DeepUnity/TTS/Kokoro/weights_kokoro_fp16/` — manifest.tsv (460 lines),
      manifest.json, vocab.txt (178 lines), .bin files ≈155 MB. Exporter self-verified: all 548
      checkpoint keys consumed, shape spot-checks passed, per-file byte sizes == numel*2.
- [ ] `Assets/DeepUnity/TTS/Kokoro/validation/dump/` — t{0,1,2}_* stage tensors + t{i}.wav
      (listen: 3.7 s / 7.2 s / 8.4 s, intelligible female voice = af_heart), t{i}_phonemes.txt,
      t{i}_meta.json. Python self-check printed `max|manual - reference| = 0.000e+00` for all 3.
- After first Unity import: confirm the weights folder + dump folder got .meta files and no
  import errors (all .bin are DefaultAssets; .npy must NOT be imported as anything else).

## B1 STATUS (pre-Unity verification already done via the repo dotnet harness)
Run it yourself anytime (no Unity needed):
    cd C:/dev/DeepUnity/Assets/DeepUnity/TTS/Kokoro/validation/harness~ && dotnet run -c Release
Latest result — ALL PASS:
  - G2P: t0/t1/t2 byte-exact + **99/99 exact on the B2 sweep corpus** (g2p_corpus.tsv, 99 varied
    sentences: quotes, contractions, currency, clock times, ordinals, years, a.m., heteronyms).
    Four tagger/tokenizer fixes landed during the sweep (quote-parity ``/'' tags, complementizer
    "that"→IN, irregular-past VBD set, dotted-abbreviation + d:dd clock-time handling).
  - FULL MODEL fp32 CPU forward (KokoroCPU.cs + KokoroTensors.cs, fp16 export weights) vs the
    python dumps: bert_dur/d_en/d/duration/en/F0/N/t_en/asr/dec_x ALL corr = 1.000000 on all 3
    texts; pred_dur EXACT (0 tokens differ, override provision never triggered); t0 wav
    corr = 0.9973 with injected rand_ini+sine_noise (gate is 0.99). Only drift source = fp16
    weight rounding (maxabs ~1e-3 features / 4.5e-2 wav).
  - KokoroCPU is the parity ORACLE for the upcoming KokoroCS.compute kernels AND the runtime
    home of the 6 CPU biLSTMs. CPU forward timing (scalar C#, partial Parallel.For): ~5.7x
    real-time slow (20.8s for 3.7s audio) — GPU port is the fix, CPU path stays as oracle.
  - NOTE: t*_d_en.npy / t*_d.npy are stored FORTRAN-order (numpy kept the transposed stride
    layout) — any new .npy reader must honor the `fortran_order` flag (harness does).

## Remaining B1 — GPU PORT CODE WRITTEN (this session), Unity runs pending
- [x] KokoroCS.compute WRITTEN — Assets/Resources/ComputeShaders/KokoroCS.compute, all kernels
  of the plan below (+ EmbedText and CopySlice/AddBuf/AddScale/ScaleBuf utils). MUST be
  registered in DeepUnityMeta.cs AT MERGE (shared file — owned by the orchestrator; this
  workstream does NOT edit it; until then KokoroModel Resources.Load's it directly).
  KERNEL PLAN (all written; grade in THIS order; each has a direct KokoroCPU oracle method):
    1. LinearBias (matmul+bias, readH fp16)             <- KokoroCPU.LinearT / StyleFc
    2. LayerNormAffine (eps uniform: 1e-12 bert / 1e-5; strided — covers [T,C] rows AND
       [C,T] per-time channel norm)                     <- LayerNorm
    3. EmbedAlbert (word+pos+tok lookup, 128)           <- EmbedAlbert
    4. AttentionBi 12x64 (no mask, no cache; cribbed Chatterbox estimator MHA) <- AttentionBi
    5. GeluNew elementwise (Activate act=1)             <- GeluNew
    6. Conv1D generic (stride/pad/dilation/no-bias uniforms, channel-major) <- Conv1d
    7. InstanceNormStyle (= AdaIN: 2-pass mean/var over T + (1+g)x+b from style fc) <- AdaIN
    8. LeakyRelu (Activate act=2, slope 0.2/0.1/0.01!)  <- LRelu
    9. ConvTranspose1D (grouped: ups groups=1 + depthwise pool groups=C) <- ConvTranspose1d
   10. GatherTime (nearest x2 mode + idx mode: frame2tok aln expand, reflection_pad(1,0))
                                                        <- AdainBlock shortcut / asr build
   11. Snake (per-channel alpha, exact 1/a)             <- Snake
   12. Stft20 (DFT-20 mag/angle) + Istft20 (inv-DFT OLA + wsum + trim) <- Stft / Istft
   Stays CPU inside the GPU pipeline (KokoroCPU, worker Tasks): 6 biLSTMs, duration
   round/alignment, SineGen phase pipeline + RNG (NsfHar), chunker, G2P.
- [x] KokoroModel.cs WRITTEN — GPU dispatch along KokoroCPU.Forward's exact call sequence;
  GPU<->CPU handoffs: d_en readback -> CPU DurationEncode/head/shared-LSTM -> xf upload;
  tenc conv readback -> CPU tenc biLSTM -> t_en upload; CPU NsfHar (overlaps GPU decoder) ->
  har upload. Parity hooks: InjectPredDur, CaptureStages/LastStages (KokoroCPU.Stages layout).
- [x] KokoroTTS.cs SWITCHED TO GPU-ONLY — Synthesize drives KokoroModel.ForwardYielding on the
  main thread (public API unchanged; residency forwarding untouched). KokoroCPU is now strictly
  the validation oracle + CPU-stage host, never a runtime backend.
- [x] KokoroKernelProbe.cs + validation/Editor/KokoroKernelBatchRunner.cs WRITTEN — part A
  grades EVERY kernel vs the oracle (random [-2,2] inputs, real fp16 weights, maxabs < 1e-3;
  STFT angles wrap-aware), part B runs the full stage probe vs the dumps (B2 gates below) and
  writes ProbeLogs/kokoro_kernel_report.md + ProbeLogs/kokoro_kernel.done + kokoro_gpu_t0.wav.
- ORCHESTRATOR RUN (Unity closed):
      Unity.exe -batchmode -projectPath C:\dev\DeepUnity ^
        -executeMethod DeepUnity.KokoroModeling.KokoroKernelBatchRunner.Run ^
        -logFile ProbeLogs/kokoro_kernel.log
  (NO -nographics; exit 0 PASS / 1 FAIL / 2 timeout. Unity open: menu
  DeepUnity/TTS/Run Kokoro Kernel Probe.) Report failures back to WS-B for kernel fixes.
- NOTE: KokoroCPU/KokoroTensors were refactored (public oracle methods extracted: EmbedAlbert,
  AttentionBi, DurationEncode, DurationHead, ConvTranspose1d, NsfHar; D() lock for concurrent
  worker tasks) — dotnet harness re-run after refactor: ALL PASS (wav corr 0.9973 unchanged).
- PROBE RUN 1 (2026-07-11): 23/24 kernels PASS, ALL stage gates PASS (t0 wav corr 0.9972,
  pred_dur exact x3, GPU audio saved). Fix round applied, RE-RUN PENDING:
  (a) 12a Stft20 angle grading now magnitude-gated (>1e-2) + wrap-aware — atan2 at
      near-zero-mag bins is fp-noise-conditioned on BOTH sides (kernel math correct: mag
      maxabs 3.8e-6; the stage gates already covered the real-signal path).
  (b) Perf: run-1 predictor 3.6-7.1 s = the CPU biLSTMs (worker-task wall, not GPU stalls).
      KokoroCPU.BiLstm parallelized BIT-EXACTLY (input-proj Parallel-over-t in original op
      order, directions concurrent, per-step whh.h chunk-parallel): harness re-run ALL PASS
      with IDENTICAL maxabs values; isolated LSTM chain at t2 sizes 7089 ms -> 333 ms (~20x).
      Expected end-to-end RTF ~0.2. KokoroModel now reports PredCpuMs/TencCpuMs/NsfWaitMs in
      the probe perf line to confirm on re-run.
- B3 CODE DONE: KokoroVoice.cs exists (clip + chunk-streaming ring buffer off
  KokoroTTS.Synthesize onChunk; ChatterboxVoice pattern). Unity checks: compile, then play-mode
  Say() in both modes — verify prebuffer gate, starvation silence (no clicks), StopSpeaking,
  and shared-TTS reuse across two KokoroVoice instances.

## B1 gates (Unity side — the GPU port now exists, run these)
1. Compile: no errors with ONLY the Kokoro folder added (no edits outside TTS/Kokoro + Resources;
   new files this session: KokoroModel.cs, validation/KokoroKernelProbe.cs,
   validation/Editor/KokoroKernelBatchRunner.cs, Resources/ComputeShaders/KokoroCS.compute).
2. G2P EXACT-MATCH GATE (blocks everything downstream):
   `KokoroG2P.Phonemize(text_i)` must equal `dump/t{i}_phonemes.txt` byte-for-byte for i=0,1,2.
   Do NOT trim — trailing punctuation and inner spaces are part of the string.
   t1 is the heteronym/number gauntlet (read→ɹˈɛd, present noun pɹˈɛzᵊnt vs verb pɹizˈɛnt,
   record→ɹˈɛkəɹd, 42→fˈɔɹTi tˈu, 2024→twˈɛnti twˈɛnti fˈɔɹ).
   STATUS: ALREADY PASSING outside Unity — KokoroG2P.cs is pure C# (no UnityEngine) and was
   gate-tested with a dotnet harness: t0/t1/t2 byte-exact, PLUS 12/12 exact on the extended
   tricky corpus `validation/dump/g2p_corpus.tsv` (contractions, $4.50, ordinals, 1,500, 1987,
   100%, heteronym cluster "use the record to track uses of the lead pipes"). In Unity, re-run
   the same comparison inside the parity probe (stage A) to confirm identical behavior under
   Unity's Mono/IL2CPP string handling.
3. Vocab mapping: ids from vocab.txt (line i = id i, id0='$'; read WITHOUT trimming — line 16 is
   a single space) must reproduce `t{i}_input_ids.npy` given the reference phoneme strings.

## B2 parity probe
STATUS: implemented as part B of KokoroKernelProbe.cs (one batch run covers kernels + stages;
no separate KokoroParityProbe file needed — same gates, same report/marker).
Stages + expected tolerances (fp16 weights vs fp32 reference; corr = Pearson):
- [ ] A  G2P + ids            — exact (see B1 gates)
- [ ] B  bert_dur             — corr ≥ 0.999, report maxabs/mae
- [ ] C  d_en / d             — corr ≥ 0.999
- [ ] D  duration / pred_dur  — pred_dur exact or ±1 frame on ≤2 tokens (rounding boundary);
        F = Σ pred_dur must match meta.json "F" (else all downstream shapes shift — if off,
        OVERRIDE pred_dur with the dumped one and continue grading later stages)
- [ ] E  en / F0_pred / N_pred— corr ≥ 0.995 (LSTM fp32 CPU + fp16 conv stacks)
- [ ] F  t_en / asr           — corr ≥ 0.999
- [ ] G  dec_x                — corr ≥ 0.99
- [ ] H  wav (t0, INJECTED t0_rand_ini + t0_sine_noise) — corr ≥ 0.99 vs t0_wav.npy
        (noise injection replaces the C# RNG draws: rand_ini [9], nz [1,S,9]; noise dumped RAW —
        multiply by noise_amp = uv*0.003+(1-uv)*0.1/3 at the use site)
- [ ] Report to `ProbeLogs/kokoro_parity_report.md` + `ProbeLogs/kokoro_parity.done` marker
      (batch-runner pattern = ChatterboxParityBatchRunner).
Numeric traps to check when a stage fails: SPEC.md §12 gotchas (esp. (1+γ) style affines,
identity InstanceNorm affine, lrelu 0.01-vs-0.1, mag/ANGLE har_cat, reflection_pad index-1).

## B3 runtime/product checks
- [ ] KokoroListenProbe: synthesize t0-t2 fresh (own RNG), save wavs to ProbeLogs, human listen
      (no clicks at chunk joins, correct pacing, matches reference prosody).
- [ ] KokoroVoice component: clip mode + streaming ring-buffer mode (prebuffer 0.4 s), sentence
      chunker respects the ≤510-phoneme bound; Say() during Say() queues, Release() leak-free
      (ComputeBuffer count reported by DeepUnity diagnostics = 0 after Release).
- [ ] Warmup(): weights streamed under UploadBudgetBytes without frame hitches > 33 ms.
- [ ] Perf on RTX 4060 laptop: report RTF for t2 (8.4 s audio), VRAM delta after Warmup, and
      time-to-first-audio in streaming mode. Target: RTF < 0.3 fp16 (82M model, non-AR — should
      be far faster than Chatterbox); flag if LSTM CPU time > 20% of total (then Burst/job it).
- [ ] BENCHMARK.md row for low-end tier (Pavilion GTX 1660 Ti) once available.

## Voice add flow (later, optional)
Download more `voices/*.pt` into `C:/dev/_model_staging/kokoro/hf/voices/` and re-run
`validation/import_kokoro.py` (idempotent, rewrites manifest with the new `voices/<name>` rows).
