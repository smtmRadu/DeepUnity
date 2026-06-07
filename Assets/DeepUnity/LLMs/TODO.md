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

## Sampling penalties (implemented)
GPU presence + repetition penalty added to the Qwen3.5 sampler. Recommended preset values are documented in the
`Qwen3_5ForCausalLM` constructor XML doc, and the per-model defaults now also live in the shared `LLMConfig`
descriptor (`DefaultPresencePenalty`, `DefaultRepetitionPenalty`, etc.) lifted in during the Base refactor.
