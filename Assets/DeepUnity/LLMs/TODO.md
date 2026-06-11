# LLMs — TODO

## Refactor: shared Base layer for all LLMs — DONE

Goal: cleaner, more general code so future models drop in easily, and Gemma/Qwen expose near-identical APIs.

- [x] Created `LLMs/Base/`.
- [x] Moved the BPE tokenizer (`BPETokenizer.cs`) into `Base/`; deleted the old `Tokenizers/` folder.
- [x] Added abstract `LLM` base class (`Base/LLM.cs`) — `Qwen3_5ForCausalLM` and `Gemma3ForCausalLM` inherit it.
      Shared surface: `Config`, `IsReady`, `TokensPerSecond`, `Warmup`, `Predict`, `Generate`, `InitializeChat`,
      `Chat`, `Release`. The editor play-mode hook + finalizer live in the base and both route through `Release()`.
- [x] Added abstract config base (`Base/LLMConfig.cs`): hidden size, vocab, num layers, max-pos, head dim, RMS eps,
      tie-embedding, special token ids, and sampling defaults. Per-model descriptors (`Qwen3_5ConfigDescriptor`,
      `Gemma3ConfigDescriptor`) extend it and forward to the existing static `*Config` (single source of truth).
- [x] Aligned Gemma + Qwen public APIs (same method names/signatures/order). Gemma accepts the penalty / thinking
      knobs for parity and ignores them (its sampler has no such support); Warmup is a base no-op Gemma inherits.

Note: the per-model static `*Config` classes are unchanged (the modeling/weights/cache code still reads them
directly) — the descriptor is only the uniform, model-agnostic view exposed via `llm.Config`.

---

## Future: MTP (multi-token prediction) speculative decoding for Qwen3.5-0.8B

Research result (confirmed): Qwen3.5 ships an **MTP draft head** in the official checkpoint — config declares
`mtp_num_hidden_layers: 1` (a 1-layer next-token draft module). It is the same family used by vLLM/SGLang for
speculative decoding:
- vLLM: `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` (1–5 tokens, latency vs accept-rate trade-off).
- SGLang: `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`.

- [ ] Export the MTP head weights alongside the main model (extend the param-export script).
- [ ] Implement the draft head + a verify/accept loop on the GPU (propose N tokens, verify with the main model, accept the longest matching prefix).
- [ ] Expose `num_speculative_tokens` (default ~2) on the generate/chat API.
Payoff: meaningfully faster single-stream generation (the tavern/NPC use case).

---

## System-prompt KV disk cache (implemented 2026-06-10) — NEEDS TESTING

`InitializeChat` now persists the system prompt's KV + SSM state to
`persistentDataPath/DeepUnity/qwen35_prompt_<hash>.kv` after the first prefill, and restores it on later
inits with the same prompt/weights/capacity instead of recomputing (toggle:
`Qwen3_5ForCausalLM.SystemPromptDiskCache`, default ON). Mechanics: `Qwen3_5Cache.SaveYielding`
(AsyncGPUReadback → worker-thread file write) / `LoadYielding` (worker-thread parse → one `SetData` per
frame). K/V layout is token-major so the prefix-partial save is valid; linear layers save full conv +
recurrent SSM states.

2026-06-11 playtest: the restore dropped play mode to ~48 fps — the old code uploaded a full layer's K+V
(several MB of SetData) per frame. Reworked to be frame-budgeted; the tuning knobs are SHARED across all
models and live in `Base/LLM.cs` ("prompt-cache hitch tuning" block), turn them DOWN if frames ever drop:
- `LLM.UploadFrameBudgetMs` (default 0.5) — max ms of SetData copy work per frame on restore.
- `LLM.UploadChunkFloats` (default 64k = 256 KB) — SetData granularity the budget cuts at.
- `LLM.SaveReadbacksInFlight` (default 1) — concurrent GPU readbacks while saving; caps per-frame
  readback→managed copies (results must be copied the frame they complete, so deferring isn't an option).
SEPARATE knob for boot: `UPLOAD_BUDGET_BYTES` in Qwen3_5Weights.cs / Gemma3Weights.cs (per-frame bytes of
weight streaming; was 24 MB → ~10.7 ms worst slice and a mid-game fps dip, now 8 MB). 2026-06-11 second
nudge after a 58-fps-minimum report: KV knobs 1.0→0.5 ms / 128k→64k / 2→1 + the 24→8 MB weight budget.
Boot log simplified to one line (`Qwen3.5 ready — load X ms, system prompt restored/computed (N tokens, Y ms)`);
the old per-step breakdown is commented out inside LogBootSummary for debugging.
Gemma3Cache.SaveAsync/TryLoadAsync were reworked to the same pattern (worker-thread IO + FP16 conversion,
budgeted SetData, readback window) honoring the same knobs — on-disk format unchanged, old caches load.
Gemma DIFFERENCES kept as-is for now: cache folder is `Assets/Resources/Cache/<sha256>` (editor-only path,
breaks in player builds) and the key hashes the prompt TEXT only (not weights/capacity → swap checkpoints
and a stale cache can hit). Align with the Qwen scheme (persistentDataPath + FNV over ids+path+capacity)
if Gemma ever ships in a build.
Next lever if the knobs stop being enough: stream the restore (worker parses layer i while layer i-1
uploads, two reused scratch arrays) instead of parsing the whole file up front — less transient managed
memory → less GC pressure.

2026-06-11 (fps still dipping, now 49 — knobs alone can't explain it): added PHASE ATTRIBUTION instead of
more blind nudging. `LLM.CurrentPhase` static tag ("idle"/"kernel-prewarm"/"boot (weights+warmup)"/
"kv-restore"/"prefill"/"kv-save"/"decode") written by Qwen3_5.cs around each long phase (reset in
LLM.OnReleased so interrupts can't leave it stale); ChatDemo3D got a FrameSpikeProbe (logs every frame
>18 ms with phase + whether GC landed that frame → ProbeLogs/frame_spikes.csv + 10 s console summaries)
and LLMPrewarm (runs the static Qwen3_5Model.PrewarmKernels at scene start — the per-session driver
compiles, incl. the long DeltaNet ones, now hitch during scene load instead of mid-game on first chat).
Suspects the budgets CANNOT fix: lazy GPU buffer creation/first-touch commits (driver alloc is atomic) and
GPU contention during decode (frames wait on inference dispatches). Read frame_spikes.csv after the next
playtest to see which phase owns the dips before optimizing further.

- [ ] TEST the restore path properly: generation quality after a disk-restored init must be byte-identical
      (or statistically indistinguishable) vs a freshly prefilled one — same prompt, greedy decode, compare outputs.
- [ ] RE-verify frame pacing with the profiler after the 2026-06-11 budgeted rework (target: no frame over
      budget during save on first init and load on subsequent inits).
- [ ] Test invalidation: change the prompt / weights path / maxModelLength → must recompute, not load stale state.
- [ ] Test corrupt/truncated cache file → must fall back to prefill (and delete the bad file).
- [ ] Consider versioning the format if the cache layout ever changes (FILE_VERSION already in the header).

---

## Sampling penalties (implemented)
GPU presence + repetition penalty added to the Qwen3.5 sampler. Recommended preset values are documented in the
`Qwen3_5ForCausalLM` constructor XML doc, and the per-model defaults now also live in the shared `LLMConfig`
descriptor (`DefaultPresencePenalty`, `DefaultRepetitionPenalty`, etc.) lifted in during the Base refactor.
