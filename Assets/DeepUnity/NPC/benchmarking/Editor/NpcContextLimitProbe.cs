#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    // Edit-mode probe for the context-limit primitives the NPC history modes are built on:
    //   - LLM.MaxContextTokens  — the per-NPC maxContextLength reaching the KV capacity
    //   - LLM.CurrentContextTokens — live KV occupancy (system prompt + every turn so far)
    //   - crossing maxContextLength is detected (ResumeFromCompact fires the compaction)
    //   - re-seeding a SHORT prefix (exactly what CompactConversationRoutine does after summarizing)
    //     drops the occupancy back under the limit → the conversation keeps going
    // Deliberately SYNCHRONOUS (model.Forward + SampleGreedy, like QwenDecodeProfileProbe) — it never
    // pumps the async-readback coroutines, so it is safe in edit mode (no play loop to service them).
    // A small allocated context (MAXLEN) makes the limit hit in a handful of tokens, per the ask.
    public static class NpcContextLimitProbe
    {
        const int MAXLEN = 256;   // allocated KV capacity for the test model (small on purpose)
        const int LIMIT  = 128;   // the "maxContextLength" threshold the modes would act on
        const int PROMPT = 40;    // pretend system-prompt length
        const int RESEED = 18;    // pretend [system + HISTORY] compacted prefix length

        [MenuItem("DeepUnity/NPC/Context-Limit Probe (int8, edit mode)")]
        public static void Run()
        {
            Qwen3_5ForCausalLM llm = null;
            var log = new System.Text.StringBuilder();
            int pass = 0, fail = 0;
            void Check(bool ok, string what)
            { log.AppendLine($"    [{(ok ? "PASS" : "FAIL")}] {what}"); if (ok) pass++; else fail++; }

            try
            {
                EditorUtility.DisplayProgressBar("NPC context-limit probe", "Loading Qwen3.5-0.8B int8…", 0.15f);
                llm = new Qwen3_5ForCausalLM(Qwen3_5Size.B0_8, LLMQuant.INT8,
                                             maxModelLength: MAXLEN, kv_quant: KVQuant.FP16);
                var model = llm.model;
                model.LoadBlockingForProbe();

                // 1. capacity plumbs through (NPCChatBase.maxContextLength → Acquire → ctor → here)
                Check(llm.MaxContextTokens == MAXLEN, $"MaxContextTokens == {MAXLEN} (got {llm.MaxContextTokens})");

                // 2. fresh occupancy is zero
                model.ResetCache();
                Check(llm.CurrentContextTokens == 0, $"fresh CurrentContextTokens == 0 (got {llm.CurrentContextTokens})");

                // 3. a P-token prompt occupies exactly P
                var ids = new float[PROMPT];
                for (int i = 0; i < PROMPT; i++) ids[i] = 1000 + i;
                model.Forward(Tensor.Constant(ids), useCache: true, lastPosOnly: true);
                Check(llm.CurrentContextTokens == PROMPT,
                      $"after prefill CurrentContextTokens == {PROMPT} (got {llm.CurrentContextTokens})");

                EditorUtility.DisplayProgressBar("NPC context-limit probe", "Decoding to the limit…", 0.5f);
                // 4. decode turn-by-turn; occupancy grows one per token and CROSSES the limit
                int tok = model.SampleGreedy();
                int crossedAt = -1, prev = llm.CurrentContextTokens;
                bool monotonic = true;
                for (int n = 0; n < MAXLEN - PROMPT - 2; n++)
                {
                    model.Forward(Tensor.Constant((float)tok), useCache: true, lastPosOnly: true);
                    tok = model.SampleGreedy();
                    int now = llm.CurrentContextTokens;
                    if (now != prev + 1) monotonic = false;
                    prev = now;
                    if (crossedAt < 0 && now >= LIMIT) { crossedAt = now; break; }
                }
                Check(monotonic, "CurrentContextTokens grows exactly +1 per decoded token");
                Check(crossedAt >= LIMIT,
                      $"context crossed the {LIMIT}-token limit (at {crossedAt}) → ContextFull() fires here");
                Check(crossedAt < MAXLEN,
                      $"crossing happened with headroom below the {MAXLEN} allocation (compact pass has room)");

                EditorUtility.DisplayProgressBar("NPC context-limit probe", "Compaction re-seed…", 0.85f);
                // 5. compaction shrink: re-seeding a short [system + HISTORY] prefix drops occupancy
                //    back under the limit (this is the KV effect of CompactConversationRoutine).
                model.ResetCache();
                var shortIds = new float[RESEED];
                for (int i = 0; i < RESEED; i++) shortIds[i] = 2000 + i;
                model.Forward(Tensor.Constant(shortIds), useCache: true, lastPosOnly: true);
                Check(llm.CurrentContextTokens == RESEED,
                      $"after compact re-seed CurrentContextTokens == {RESEED} (got {llm.CurrentContextTokens})");
                Check(llm.CurrentContextTokens < LIMIT,
                      "re-seeded context is back under the limit → ResumeFromCompact keeps talking");

                string verdict = fail == 0 ? "ALL PASS" : "FAILURES";
                Debug.Log($"[NpcContextLimit] {verdict} — {pass} pass / {fail} fail (MAXLEN {MAXLEN}, LIMIT {LIMIT}) " +
                          $"| GPU {SystemInfo.graphicsDeviceName}\n{log}");
                if (fail > 0) Debug.LogError("[NpcContextLimit] FAILED — the history-mode context primitives are wrong; do NOT ship.");
            }
            finally
            {
                llm?.Release();
                EditorUtility.ClearProgressBar();
            }
        }
    }
}
#endif
