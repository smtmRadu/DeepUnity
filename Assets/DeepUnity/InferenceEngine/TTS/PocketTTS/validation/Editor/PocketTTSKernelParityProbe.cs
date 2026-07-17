#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        // Kernel-parity probe, NO reference dumps needed (the .npy dumps live on the main dev box
        // only) — everything is legacy-vs-new on DETERMINISTIC pseudo-random inputs.
        //
        // Gate set A (#30, menu "Kernel Parity (tiled vs legacy)"): tiled conv/attention vs the
        // pre-#30 kernels — BIT-exact contract (accumulation order preserved). #31 kernels are
        // forced OFF during these gates so the axis stays pure.
        //
        // Gate set B (#31-P, menus "Kernel Parity #31 ..."): coalesced GEMV/GEMM + fused flow head
        // vs the legacy LinearBias path. Tree/lane reductions REORDER float sums, so the contract
        // here is TOLERANCE gates (per-kernel maxAbs/corr like Kokoro's probe), plus an
        // end-to-end offline generate A/B (latents/EOS-step/wav-mel) and a same-run [perf] print.
        public static class PocketTTSKernelParityProbe
        {
            // ---------------- gate set A: #30 tiled vs legacy (bit-exact) ----------------
            [MenuItem("DeepUnity/PocketTTS/Kernel Parity (tiled vs legacy)")]
            public static void Run()
            {
                PocketTTSWeights weights = null;
                PocketTTSMimi mimi = null;
                bool fk2 = PocketTTS.FastKernels2;
                try
                {
                    // #31 OFF for the whole gate set: its reordered sums would break the #30
                    // bit-exact contract (the #31 axis is graded separately below).
                    PocketTTS.FastKernels2 = false;
                    EditorUtility.DisplayProgressBar("pocket-tts kernel parity", "Loading mimi weights…", 0.1f);
                    weights = new PocketTTSWeights(PocketTTSConfig.WEIGHTS_DIR_FP16, beginLoad: false);
                    weights.LoadBlocking("mimi/");
                    mimi = new PocketTTSMimi(weights);

                    const int T = 41;                       // one streaming window (CTX 40 + 1): odd, exercises every stage shape
                    float[] latents = RandVec(T * PocketTTSConfig.LDIM, 0x9E3779B9u);

                    EditorUtility.DisplayProgressBar("pocket-tts kernel parity", "Legacy decode…", 0.3f);
                    PocketTTSMimi.ForceLegacyKernels = true;
                    mimi.Decode(latents, T);                              // warmup: kernel compiles
                    float[] wavLegacy = mimi.Decode(latents, T);
                    float legacyMs = mimi.DecodeMs;

                    EditorUtility.DisplayProgressBar("pocket-tts kernel parity", "Tiled decode…", 0.6f);
                    PocketTTSMimi.ForceLegacyKernels = false;
                    mimi.Decode(latents, T);                              // warmup
                    float[] wavTiled = mimi.Decode(latents, T);
                    float tiledMs = mimi.DecodeMs;

                    double maxAbs = 0, sumSq = 0, dotAB = 0, sumA2 = 0, sumB2 = 0;
                    int worstIdx = -1;
                    for (int i = 0; i < wavLegacy.Length; i++)
                    {
                        double d = System.Math.Abs(wavLegacy[i] - (double)wavTiled[i]);
                        if (d > maxAbs) { maxAbs = d; worstIdx = i; }
                        sumSq += d * d;
                        dotAB += (double)wavLegacy[i] * wavTiled[i];
                        sumA2 += (double)wavLegacy[i] * wavLegacy[i];
                        sumB2 += (double)wavTiled[i] * wavTiled[i];
                    }
                    double corr = dotAB / System.Math.Max(System.Math.Sqrt(sumA2 * sumB2), 1e-30);
                    bool pass = maxAbs == 0.0;
                    string verdict = pass ? "PASS (bit-exact)"
                                   : maxAbs < 1e-5 ? $"PASS-ish (maxAbs {maxAbs:E2} @ {worstIdx} — expected 0, investigate)"
                                   : $"FAIL (maxAbs {maxAbs:E2} @ sample {worstIdx})";
                    Debug.Log($"[PocketParity] tiled vs legacy, T={T} ({wavLegacy.Length} samples): {verdict} | " +
                              $"corr {corr:F9} | legacy {legacyMs:F0} ms -> tiled {tiledMs:F0} ms ({legacyMs / Mathf.Max(tiledMs, 0.01f):F2}x) | " +
                              $"GPU {SystemInfo.graphicsDeviceName}");
                    if (!pass && maxAbs >= 1e-5) Debug.LogError("[PocketParity] kernel parity FAILED — do not ship the tiled kernels.");

                    // ---- gate 2 (#30): tail-restricted window decode vs full decode of the SAME
                    // window. The kept tail must be BIT-exact; the context region is garbage by
                    // contract and is not compared. Mirrors the streaming flush (CTX 40 + chunk 12).
                    EditorUtility.DisplayProgressBar("pocket-tts kernel parity", "Tail-restricted decode…", 0.8f);
                    const int T2 = 52, TAIL = 12;
                    float[] lat2 = RandVec(T2 * PocketTTSConfig.LDIM, 0x2545F491u);
                    float[] wavFull = mimi.Decode(lat2, T2);
                    float fullMs = mimi.DecodeMs;
                    float[] wavTail = mimi.Decode(lat2, T2, tailLatents: TAIL);
                    float tailMs = mimi.DecodeMs;
                    int tailN = TAIL * PocketTTSConfig.SAMPLES_PER_LATENT;
                    int off = wavFull.Length - tailN;
                    double maxAbs2 = 0; int worst2 = -1;
                    for (int i = 0; i < tailN; i++)
                    {
                        double d = System.Math.Abs(wavFull[off + i] - (double)wavTail[off + i]);
                        if (d > maxAbs2) { maxAbs2 = d; worst2 = i; }
                    }
                    bool pass2 = maxAbs2 == 0.0;
                    Debug.Log($"[PocketParity] tail-restricted (T={T2}, tail={TAIL}): " +
                              (pass2 ? "PASS (bit-exact tail)" : $"FAIL (maxAbs {maxAbs2:E2} @ tail sample {worst2})") +
                              $" | full {fullMs:F0} ms -> tail {tailMs:F0} ms ({fullMs / Mathf.Max(tailMs, 0.01f):F2}x)");
                    if (!pass2) Debug.LogError("[PocketParity] tail-restriction parity FAILED — do not ship.");
                }
                finally
                {
                    PocketTTSMimi.ForceLegacyKernels = false;
                    PocketTTS.FastKernels2 = fk2;
                    mimi?.Dispose();
                    weights?.Dispose();
                    EditorUtility.ClearProgressBar();
                }
            }

            // ---------------- gate set B: #31-P coalesced GEMV/GEMM + fused flow head ----------------
            [MenuItem("DeepUnity/PocketTTS/Kernel Parity #31 (coal GEMV+flow, fp16)")]
            public static void RunCoalFp16() => RunCoal(PocketTTSConfig.WEIGHTS_DIR_FP16, "fp16");

            [MenuItem("DeepUnity/PocketTTS/Kernel Parity #31 (coal GEMV+flow, int8)")]
            public static void RunCoalInt8() => RunCoal(PocketTTSConfig.WEIGHTS_DIR_INT8, "int8");

            // per-kernel tolerance gates (#31 convention: reordered sums, NOT bit-exact)
            const float K_CORR = 0.999999f;   // per-kernel output corr
            const float K_MAXABS = 1e-2f;     // per-kernel absolute cap (outputs are O(1)-O(10))
            const float FH_CORR = 0.9999f;    // whole flow-head velocity
            const float E2E_LAT_CORR = 0.999f;      // e2e latents, all frames (AR feedback drift allowed)
            const float E2E_EARLY_CORR = 0.9999f;   // e2e latents, frames 0-3 (drift-free zone)
            const float E2E_MEL_CORR = 0.99f;       // e2e wav mel-corr (phase-invariant)

            static bool failed;

            static void RunCoal(string dir, string tag)
            {
                if (!System.IO.Directory.Exists(dir))
                {
                    Debug.LogWarning($"[PocketParity31] {tag}: weights dir missing ({dir}) — skipped.");
                    return;
                }
                failed = false;
                bool fk2 = PocketTTS.FastKernels2;
                bool fk3 = PocketTTS.FastKernels3;
                PocketTTSWeights weights = null;
                PocketTTSFlowLM flm = null;
                PocketTTSMimi mimi = null;
                PocketTTS tts = null;
                try
                {
                    // R1-axis purity: FastKernels3 stays OFF for the whole gate set — its GPU-frame
                    // path replaces the code these gates compare (graded by its own #31-R2 menus).
                    PocketTTS.FastKernels3 = false;
                    EditorUtility.DisplayProgressBar($"pocket-tts #31 parity ({tag})", "Loading weights…", 0.05f);
                    weights = new PocketTTSWeights(dir, beginLoad: false);
                    weights.LoadBlocking();
                    flm = new PocketTTSFlowLM(weights);
                    mimi = new PocketTTSMimi(weights);
                    Debug.Log($"[PocketParity31] {tag} — GPU {SystemInfo.graphicsDeviceName}");

                    // ---- B1: per-kernel GEMV (T=1, the AR-loop shape) ----
                    EditorUtility.DisplayProgressBar($"pocket-tts #31 parity ({tag})", "GEMV kernels…", 0.2f);
                    KernelAB(flm, "flow_lm/transformer/layers/0/self_attn/in_proj", 1, 1024, 3072, false, 0, 0x1111u);
                    KernelAB(flm, "flow_lm/transformer/layers/0/self_attn/out_proj", 1, 1024, 1024, false, 0, 0x2222u);
                    KernelAB(flm, "flow_lm/transformer/layers/0/linear1", 1, 1024, 4096, false, 2, 0x3333u);
                    KernelAB(flm, "flow_lm/transformer/layers/0/linear2", 1, 4096, 1024, false, 0, 0x4444u);
                    KernelAB(flm, "flow_lm/flow_net/cond_embed", 1, 1024, 512, true, 0, 0x5555u);
                    KernelAB(flm, "flow_lm/flow_net/res_blocks/0/adaLN_modulation/1", 1, 512, 1536, true, 0, 0x6666u);
                    KernelAB(flm, "flow_lm/flow_net/res_blocks/0/mlp/0", 1, 512, 512, true, 1, 0x7777u);
                    KernelAB(flm, "flow_lm/flow_net/time_embed/0/mlp/0", 1, 256, 512, true, 1, 0x8888u);
                    KernelAB(flm, "flow_lm/flow_net/final_layer/linear", 1, 512, 32, true, 0, 0x9999u);

                    // ---- B2: per-kernel GEMM (T=61: odd -> ragged 8-token tail tile + row guards) ----
                    EditorUtility.DisplayProgressBar($"pocket-tts #31 parity ({tag})", "GEMM kernels…", 0.4f);
                    KernelAB(flm, "flow_lm/transformer/layers/0/self_attn/in_proj", 61, 1024, 3072, false, 0, 0xAAA1u);
                    KernelAB(flm, "flow_lm/transformer/layers/0/linear1", 61, 1024, 4096, false, 2, 0xAAA2u);
                    KernelAB(flm, "flow_lm/transformer/layers/0/linear2", 61, 4096, 1024, false, 0, 0xAAA3u);
                    KernelAB(flm, "mimi/decoder_transformer/transformer/layers/0/self_attn/in_proj", 61, 512, 1536, false, 0, 0xAAA4u);
                    KernelAB(flm, "mimi/decoder_transformer/transformer/layers/0/linear2", 61, 2048, 512, false, 0, 0xAAA5u);

                    // ---- B3: whole flow head — fused persistent kernels vs the legacy op storm ----
                    EditorUtility.DisplayProgressBar($"pocket-tts #31 parity ({tag})", "Fused flow head…", 0.55f);
                    {
                        float[] c = RandVec(PocketTTSConfig.DIM, 0xC0FFEEu);
                        float[] noise = RandGauss(PocketTTSConfig.LDIM, 0xBADD1Eu,
                                                  Mathf.Sqrt(PocketTTSConfig.TEMPERATURE));
                        PocketTTS.FastKernels2 = false;
                        float[] velLegacy = flm.FlowHead(c, noise, 0f, 1f);
                        PocketTTS.FastKernels2 = true;
                        float[] velFused = flm.FlowHead(c, noise, 0f, 1f);
                        Grade("flow head (fused vs legacy)", velLegacy, velFused, FH_CORR, K_MAXABS);
                    }

                    // ---- B4: Mimi window decode A/B (#31 axis only: tiled convs BOTH ways) ----
                    EditorUtility.DisplayProgressBar($"pocket-tts #31 parity ({tag})", "Mimi decode A/B…", 0.65f);
                    {
                        const int T = 41;
                        float[] lat = RandVec(T * PocketTTSConfig.LDIM, 0x51DE0u);
                        PocketTTS.FastKernels2 = false;
                        mimi.Decode(lat, T);                          // warmup (legacy path compiles)
                        float[] wavA = mimi.Decode(lat, T);
                        float msA = mimi.DecodeMs;
                        PocketTTS.FastKernels2 = true;
                        mimi.Decode(lat, T);                          // warmup (coal path compiles)
                        float[] wavB = mimi.Decode(lat, T);
                        float msB = mimi.DecodeMs;
                        Grade("mimi window decode (coal vs legacy linears)", wavA, wavB, 0.99999f, 1e-3f);
                        Debug.Log($"[PocketParity31] [perf] {tag} mimi window T={T}: legacy-linear {msA:F0} ms -> " +
                                  $"coal {msB:F0} ms ({msA / Mathf.Max(msB, 0.01f):F2}x)");

                        // B4b: streaming-flush contract UNDER the coal path — tail-restricted decode
                        // must reproduce the full decode's kept tail BIT-exactly (the GEMM's per-token
                        // value is elem_offset-invariant: tokens never share math across a tile).
                        const int TAIL = 12;
                        float[] wavTail = mimi.Decode(lat, T, tailLatents: TAIL);
                        int tailN = TAIL * PocketTTSConfig.SAMPLES_PER_LATENT;
                        int toff = wavB.Length - tailN;
                        double mx4 = 0;
                        for (int i = 0; i < tailN; i++)
                            mx4 = System.Math.Max(mx4, System.Math.Abs(wavB[toff + i] - (double)wavTail[toff + i]));
                        bool tailBad = mx4 != 0.0;
                        failed |= tailBad;
                        Debug.Log($"[PocketParity31] mimi tail-restricted under coal (T={T}, tail={TAIL}): " +
                                  (tailBad ? $"FAIL (maxAbs {mx4:E2} — expected bit-exact)" : "PASS (bit-exact tail)"));
                    }

                    // dispose the standalone pair before the e2e instance (bounds VRAM to one copy)
                    flm.Dispose(); flm = null;
                    mimi.Dispose(); mimi = null;
                    weights.Dispose(); weights = null;

                    // ---- B5: e2e offline generate A/B — latents + EOS step + wav + [perf] ----
                    EditorUtility.DisplayProgressBar($"pocket-tts #31 parity ({tag})", "E2E offline A/B…", 0.75f);
                    tts = new PocketTTS(dir);
                    tts.LoadBlocking();
                    int[] ids = tts.Tokenize(
                        "The old lighthouse keeper climbed the spiral stairs every evening at dusk.");
                    var inject = new float[160][];
                    for (int n = 0; n < inject.Length; n++)
                        inject[n] = RandGauss(PocketTTSConfig.LDIM, 0xE2E000u + (uint)n,
                                              Mathf.Sqrt(PocketTTSConfig.TEMPERATURE));

                    PocketTTS.FastKernels2 = false;
                    tts.GenerateOffline(ids, inject, useKvCache: true);           // warmup/compile
                    float[] wavL = tts.GenerateOffline(ids, inject, useKvCache: true);
                    int framesL = tts.LastFrames;
                    float[] latL = (float[])tts.LastLatentsRaw.Clone();
                    float preL = tts.PrefillMs, loopL = tts.LoopMs, decL = tts.DecodeMs;

                    PocketTTS.FastKernels2 = true;
                    tts.GenerateOffline(ids, inject, useKvCache: true);           // warmup/compile
                    float[] wavN = tts.GenerateOffline(ids, inject, useKvCache: true);
                    int framesN = tts.LastFrames;
                    float[] latN = (float[])tts.LastLatentsRaw.Clone();
                    float preN = tts.PrefillMs, loopN = tts.LoopMs, decN = tts.DecodeMs;

                    if (framesL != framesN)
                    {
                        failed = true;
                        Debug.LogError($"[PocketParity31] E2E FRAME COUNT MISMATCH: legacy {framesL} vs new {framesN} " +
                                       "(EOS step flipped — inspect the out_eos margin before shipping).");
                    }
                    else
                        Debug.Log($"[PocketParity31] E2E frames: {framesL} == {framesN}  PASS (same EOS step)");
                    int cmpFrames = Mathf.Min(framesL, framesN);
                    Grade("E2E latents early(0-3)", Slice(latL, 0, 4 * PocketTTSConfig.LDIM),
                          Slice(latN, 0, 4 * PocketTTSConfig.LDIM), E2E_EARLY_CORR, float.MaxValue);
                    Grade("E2E latents (all frames)", Slice(latL, 0, cmpFrames * PocketTTSConfig.LDIM),
                          Slice(latN, 0, cmpFrames * PocketTTSConfig.LDIM), E2E_LAT_CORR, float.MaxValue);
                    float mel = PocketTTSMel.MelCorr(wavL, wavN, PocketTTSConfig.SAMPLE_RATE);
                    bool melBad = mel < E2E_MEL_CORR;
                    failed |= melBad;
                    var (wx, _, wcorr) = Diff(wavL, wavN);
                    Debug.Log($"[PocketParity31] E2E wav: mel-corr {mel:F6} (gate {E2E_MEL_CORR}){(melBad ? "  <-- FAIL" : "  PASS")} " +
                              $"| raw corr {wcorr:F6} maxAbs {wx:F4} (informational — AR drift shows as phase)");

                    float totL = preL + loopL + decL, totN = preN + loopN + decN;
                    Debug.Log($"[PocketParity31] [perf] {tag} offline A/B (same run, {framesN} frames): " +
                              $"prefill {preL:F0} -> {preN:F0} ms ({preL / Mathf.Max(preN, 0.01f):F2}x) | " +
                              $"AR loop {loopL:F0} -> {loopN:F0} ms ({loopL / Mathf.Max(loopN, 0.01f):F2}x) | " +
                              $"mimi {decL:F0} -> {decN:F0} ms ({decL / Mathf.Max(decN, 0.01f):F2}x) | " +
                              $"total {totL:F0} -> {totN:F0} ms ({totL / Mathf.Max(totN, 0.01f):F2}x)");

                    Debug.Log(failed
                        ? $"[PocketParity31] {tag} RESULT: FAIL — flip PocketTTS.FastKernels2 = false and report the failing gate."
                        : $"[PocketParity31] {tag} RESULT: PASS (all #31 gates)");
                    if (failed) Debug.LogError("[PocketParity31] #31 parity FAILED — do not ship FastKernels2.");
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"[PocketParity31] EXCEPTION: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    PocketTTS.FastKernels2 = fk2;
                    PocketTTS.FastKernels3 = fk3;
                    flm?.Dispose();
                    mimi?.Dispose();
                    weights?.Dispose();
                    tts?.Dispose();
                    EditorUtility.ClearProgressBar();
                }
            }

            // ---------------- gate set C: #31-R2 GPU-resident AR frame (FastKernels3) ----------------
            // C0: instrumentation report (per-frame dispatch/readback/upload counts + CPU-ms
            //     attribution) for the R1 loop vs the R2 loop — confirms the sync-point diagnosis
            //     with numbers before grading anything.
            // C1: GemvLN targeted kernel gate (LN-folded GEMV vs LayerNormT + routed Linear).
            // C2: composite single-frame gate — one R1 frame vs one GPU-resident frame on IDENTICAL
            //     KV state (re-prefilled): c / eos logit / latent. Covers ARQkvPrep (verbatim-RoPE),
            //     AREosNorm(Q8), ARCommit, the mode-1/2 Gemv epilogues and the on-GPU input_linear
            //     in one attributable comparison.
            // C3: E2E offline A/B (FastKernels2-only vs +FastKernels3): frames/EOS-step equality,
            //     latent corr, mel-corr, [perf] with the AR-loop speedup — the R2 acceptance number.
            // C4 (#31-R3): mimi/AR overlap determinism (bitwise wav vs the sequential windowed
            //     schedule; bitwise latent non-interference) + TTFA-ramp effect + overlap [perf]
            //     on the normal AND a forced 160-frame clip (where windows actually interleave).
            [MenuItem("DeepUnity/PocketTTS/Kernel Parity #31-R2 (GPU-resident AR, fp16)")]
            public static void RunR2Fp16() => RunR2(PocketTTSConfig.WEIGHTS_DIR_FP16, "fp16");

            [MenuItem("DeepUnity/PocketTTS/Kernel Parity #31-R2 (GPU-resident AR, int8)")]
            public static void RunR2Int8() => RunR2(PocketTTSConfig.WEIGHTS_DIR_INT8, "int8");

            static void RunR2(string dir, string tag)
            {
                if (!System.IO.Directory.Exists(dir))
                {
                    Debug.LogWarning($"[PocketParity31R2] {tag}: weights dir missing ({dir}) — skipped.");
                    return;
                }
                failed = false;
                bool fk2 = PocketTTS.FastKernels2, fk3 = PocketTTS.FastKernels3;
                bool ovm = PocketTTS.OverlapMimi;
                int[] ramp = PocketTTS.ArBatchRamp;
                PocketTTSWeights weights = null;
                PocketTTSFlowLM flm = null;
                PocketTTS tts = null;
                try
                {
                    PocketTTS.FastKernels2 = true;   // R2 layers on R1 — both tiers required
                    // C0-C3 grade the PURE R2 axis: the R3 knobs (mimi overlap + TTFA ramp) are
                    // pinned OFF here and graded separately by gate set C4 below.
                    PocketTTS.OverlapMimi = false;
                    PocketTTS.ArBatchRamp = null;
                    EditorUtility.DisplayProgressBar($"pocket-tts #31-R2 ({tag})", "Loading weights…", 0.05f);
                    weights = new PocketTTSWeights(dir, beginLoad: false);
                    weights.LoadBlocking();
                    flm = new PocketTTSFlowLM(weights);
                    Debug.Log($"[PocketParity31R2] {tag} — GPU {SystemInfo.graphicsDeviceName}");

                    // ---- C1: LN-folded GEMV vs LayerNormT + routed Linear (real weights) ----
                    EditorUtility.DisplayProgressBar($"pocket-tts #31-R2 ({tag})", "GemvLN kernels…", 0.15f);
                    LNKernelAB(flm, "flow_lm/transformer/layers/0/norm1",
                               "flow_lm/transformer/layers/0/self_attn/in_proj", 1024, 3072, 0, 0xC1A1u);
                    LNKernelAB(flm, "flow_lm/transformer/layers/0/norm2",
                               "flow_lm/transformer/layers/0/linear1", 1024, 4096, 2, 0xC1A2u);
                    LNKernelAB(flm, "flow_lm/transformer/layers/3/norm2",
                               "flow_lm/transformer/layers/3/linear1", 1024, 4096, 2, 0xC1A3u);

                    // ---- C2: composite single-frame — R1 chain vs GPU-resident frame, same KV state ----
                    EditorUtility.DisplayProgressBar($"pocket-tts #31-R2 ({tag})", "Single-frame composite…", 0.35f);
                    {
                        const int Lp = 40;
                        float[] prefix = RandVec(Lp * PocketTTSConfig.DIM, 0xF1B0u);
                        float[] noise = RandGauss(PocketTTSConfig.LDIM, 0xF1B1u,
                                                  Mathf.Sqrt(PocketTTSConfig.TEMPERATURE));
                        // R1 baseline frame (CPU token/eos/latent glue, two blocking readbacks)
                        PocketTTS.FastKernels3 = false;
                        flm.ResetKV();
                        flm.PrefillKV(prefix, Lp, Lp + 4);
                        float[] cL = flm.DecodeStepKV(flm.BosLatentEmbedding());
                        float eosL = flm.OutEos(cL);
                        float[] velL = flm.FlowHead(cL, noise, 0f, 1f);
                        var latL1 = new float[PocketTTSConfig.LDIM];
                        for (int i = 0; i < latL1.Length; i++) latL1[i] = noise[i] + velL[i];
                        // R2 GPU-resident frame on a RE-PREFILLED (identical) KV state
                        PocketTTS.FastKernels3 = true;
                        flm.ResetKV();
                        flm.PrefillKV(prefix, Lp, Lp + 4);
                        flm.UploadNoiseBlock(new[] { noise }, 1);
                        flm.DecodeFrameGpuIssue(0, 0);
                        var slot = new float[PocketTTSConfig.LDIM + 1];
                        flm.ReadEosLatBlock(1, slot);
                        float[] cG = flm.ReadCondForProbe();
                        Grade("R2 frame c (out_norm, tree-LN vs serial)", cL, cG, 0.9999f, 5e-3f);
                        float eosDiff = Mathf.Abs(eosL - slot[0]);
                        bool eosBad = eosDiff > 1e-2f;
                        failed |= eosBad;
                        Debug.Log($"[PocketParity31R2] R2 frame eos: legacy {eosL:F5} vs gpu {slot[0]:F5} " +
                                  $"(|d| {eosDiff:E2}, gate 1e-2){(eosBad ? "  <-- FAIL" : "  PASS")}");
                        var latG1 = new float[PocketTTSConfig.LDIM];
                        System.Array.Copy(slot, 1, latG1, 0, latG1.Length);
                        Grade("R2 frame latent", latL1, latG1, 0.9999f, K_MAXABS);
                    }
                    flm.Dispose(); flm = null;
                    weights.Dispose(); weights = null;

                    // ---- C0 + C3: instrumented E2E A/B (FastKernels2-only vs +FastKernels3) ----
                    EditorUtility.DisplayProgressBar($"pocket-tts #31-R2 ({tag})", "E2E offline A/B…", 0.6f);
                    tts = new PocketTTS(dir);
                    tts.LoadBlocking();
                    int[] ids = tts.Tokenize(
                        "The old lighthouse keeper climbed the spiral stairs every evening at dusk.");
                    var inject = new float[160][];
                    for (int n = 0; n < inject.Length; n++)
                        inject[n] = RandGauss(PocketTTSConfig.LDIM, 0xE2E000u + (uint)n,
                                              Mathf.Sqrt(PocketTTSConfig.TEMPERATURE));

                    PocketTTS.PerfCounting = true;
                    PocketTTS.FastKernels3 = false;                                // R1 loop
                    tts.GenerateOffline(ids, inject, useKvCache: true);            // warmup/compile
                    PocketTTS.StatReset();
                    float[] wavL = tts.GenerateOffline(ids, inject, useKvCache: true);
                    int framesL = tts.LastFrames;
                    float[] latL = (float[])tts.LastLatentsRaw.Clone();
                    float preL = tts.PrefillMs, loopL = tts.LoopMs, decL = tts.DecodeMs;
                    PrintLoopStats($"{tag} R1 loop (FastKernels3 off)", framesL);

                    PocketTTS.FastKernels3 = true;                                 // R2 loop
                    tts.GenerateOffline(ids, inject, useKvCache: true);            // warmup/compile
                    PocketTTS.StatReset();
                    float[] wavN = tts.GenerateOffline(ids, inject, useKvCache: true);
                    int framesN = tts.LastFrames;
                    float[] latN = (float[])tts.LastLatentsRaw.Clone();
                    float preN = tts.PrefillMs, loopN = tts.LoopMs, decN = tts.DecodeMs;
                    PrintLoopStats($"{tag} R2 loop (GPU-resident, K={PocketTTS.ArBatchFrames})", framesN);
                    PocketTTS.PerfCounting = false;

                    if (framesL != framesN)
                    {
                        failed = true;
                        Debug.LogError($"[PocketParity31R2] E2E FRAME COUNT MISMATCH: R1 {framesL} vs R2 {framesN} " +
                                       "(EOS step flipped — inspect the eos margin in the C2 gate before shipping).");
                    }
                    else
                        Debug.Log($"[PocketParity31R2] E2E frames: {framesL} == {framesN}  PASS (same EOS step)");
                    int cmpFrames = Mathf.Min(framesL, framesN);
                    Grade("R2 E2E latents early(0-3)", Slice(latL, 0, 4 * PocketTTSConfig.LDIM),
                          Slice(latN, 0, 4 * PocketTTSConfig.LDIM), E2E_EARLY_CORR, float.MaxValue);
                    Grade("R2 E2E latents (all frames)", Slice(latL, 0, cmpFrames * PocketTTSConfig.LDIM),
                          Slice(latN, 0, cmpFrames * PocketTTSConfig.LDIM), E2E_LAT_CORR, float.MaxValue);
                    float mel = PocketTTSMel.MelCorr(wavL, wavN, PocketTTSConfig.SAMPLE_RATE);
                    bool melBad = mel < E2E_MEL_CORR;
                    failed |= melBad;
                    var (wx, _, wcorr) = Diff(wavL, wavN);
                    Debug.Log($"[PocketParity31R2] E2E wav: mel-corr {mel:F6} (gate {E2E_MEL_CORR}){(melBad ? "  <-- FAIL" : "  PASS")} " +
                              $"| raw corr {wcorr:F6} maxAbs {wx:F4} (informational)");

                    float totL = preL + loopL + decL, totN = preN + loopN + decN;
                    Debug.Log($"[PocketParity31R2] [perf] {tag} offline A/B (R1 -> R2, {framesN} frames): " +
                              $"prefill {preL:F0} -> {preN:F0} ms | " +
                              $"AR loop {loopL:F0} -> {loopN:F0} ms ({loopL / Mathf.Max(loopN, 0.01f):F2}x  <- the R2 number) | " +
                              $"mimi {decL:F0} -> {decN:F0} ms | " +
                              $"total {totL:F0} -> {totN:F0} ms ({totL / Mathf.Max(totN, 0.01f):F2}x)");

                    // ---- C4 (#31-R3): mimi/AR overlap determinism + TTFA ramp + [perf] ----
                    // Overlap issues DecodeWindowed's EXACT window schedule interleaved with the
                    // AR blocks — bit-identical wav is the contract, gated three ways:
                    //  (a) normal clip (T <= 64): overlap wav == the R2 tail's plain full decode;
                    //  (b) forced long clip (all 160 inject frames): overlap wav == the SAME
                    //      schedule run sequentially (DecodeWindowed, chunk 64, multi-window);
                    //  (c) latents bitwise-identical overlap-on vs overlap-off (AR untouched by
                    //      interleaved mimi — disjoint buffers; also proves ArBatchRamp neutrality).
                    EditorUtility.DisplayProgressBar($"pocket-tts #31-R2 ({tag})", "R3 overlap + ramp…", 0.85f);
                    PocketTTS.FastKernels3 = true;
                    // R3 OFF side (pure R2, ramp off) — timed + captured
                    PocketTTS.OverlapMimi = false; PocketTTS.ArBatchRamp = null;
                    float[] wavOff = tts.GenerateOffline(ids, inject, useKvCache: true);
                    float[] rawOff = (float[])tts.LastLatentsRaw.Clone();
                    float loopOff = tts.LoopMs, decOff = tts.DecodeMs, preOff = tts.PrefillMs, ttfaOff = tts.TtfaMs;
                    // R3 ON side (overlap + ramp {2,4})
                    PocketTTS.OverlapMimi = true; PocketTTS.ArBatchRamp = new[] { 2, 4 };
                    tts.GenerateOffline(ids, inject, useKvCache: true);            // warmup (alloc paths)
                    float[] wavOn = tts.GenerateOffline(ids, inject, useKvCache: true);
                    float[] rawOn = (float[])tts.LastLatentsRaw.Clone();
                    int framesR3 = tts.LastFrames;
                    float loopOn = tts.LoopMs, decOn = tts.DecodeMs, preOn = tts.PrefillMs, ttfaOn = tts.TtfaMs;
                    BitGate("R3 latents unperturbed by overlap+ramp", rawOff, rawOn);
                    if (framesR3 <= 64)
                        BitGate("R3 overlap wav == R2 full decode (T<=64, single window)", wavOff, wavOn);
                    else
                    {
                        float mel3 = PocketTTSMel.MelCorr(wavOff, wavOn, PocketTTSConfig.SAMPLE_RATE);
                        bool m3Bad = mel3 < E2E_MEL_CORR;   // 64 < T <= 128: full vs windowed = the
                        failed |= m3Bad;                    // established past-RF fp-noise relation
                        Debug.Log($"[PocketParity31R2] R3 wav (windowed vs full, T={framesR3}): mel {mel3:F6}" +
                                  (m3Bad ? "  <-- FAIL" : "  PASS"));
                    }
                    float totOff = preOff + loopOff + decOff, totOn = preOn + loopOn + decOn;
                    Debug.Log($"[PocketParity31R2] [perf] {tag} R3 A/B (overlap+ramp OFF -> ON, {framesR3} frames): " +
                              $"loop+mimi {loopOff:F0}+{decOff:F0} -> {loopOn:F0}+{decOn:F0} ms " +
                              $"(split shifts under overlap — TOTAL is the number) | " +
                              $"total {totOff:F0} -> {totOn:F0} ms ({totOff / Mathf.Max(totOn, 0.01f):F2}x) | " +
                              $"TTFA proxy {ttfaOff:F0} -> {ttfaOn:F0} ms (ramp {{2,4}})");

                    // (b) forced long clip: run ALL 160 inject frames (framesAfterEos never trips)
                    // -> 3 windows (64|64|32) interleaved with 20+ AR blocks; reference = the same
                    // windows sequentially. Also re-proves latent non-interference at length, and
                    // its [perf] is the overlap's REAL showcase (the 46-frame clip has no in-loop
                    // window to hide — its first window only completes at end-of-stream).
                    float[] wavLong = tts.GenerateOffline(ids, inject, 512, 999, useKvCache: true);
                    int Tlong = tts.LastFrames;
                    float[] rawLong = (float[])tts.LastLatentsRaw.Clone();
                    float longLoopOn = tts.LoopMs, longDecOn = tts.DecodeMs, longPreOn = tts.PrefillMs;
                    float[] wavSeq = tts.DecodeWindowed(rawLong, Tlong);   // chunk 64, T=160 -> windowed
                    BitGate($"R3 overlap wav == sequential windowed (bitwise, T={Tlong})", wavLong, wavSeq);
                    PocketTTS.OverlapMimi = false;
                    tts.GenerateOffline(ids, inject, 512, 999, useKvCache: true);
                    BitGate("R3 long-run latents unperturbed by overlap",
                            rawLong, (float[])tts.LastLatentsRaw.Clone());
                    float longLoopOff = tts.LoopMs, longDecOff = tts.DecodeMs, longPreOff = tts.PrefillMs;
                    Debug.Log($"[PocketParity31R2] [perf] {tag} R3 long-run A/B (T={Tlong}, overlap OFF -> ON): " +
                              $"loop+mimi {longLoopOff:F0}+{longDecOff:F0} -> {longLoopOn:F0}+{longDecOn:F0} ms | " +
                              $"total {longPreOff + longLoopOff + longDecOff:F0} -> " +
                              $"{longPreOn + longLoopOn + longDecOn:F0} ms " +
                              $"({(longPreOff + longLoopOff + longDecOff) / Mathf.Max(longPreOn + longLoopOn + longDecOn, 0.01f):F2}x)");
                    PocketTTS.OverlapMimi = true;

                    Debug.Log(failed
                        ? $"[PocketParity31R2] {tag} RESULT: FAIL — flip PocketTTS.FastKernels3 = false (and/or PocketTTS.OverlapMimi = false) and report the failing gate."
                        : $"[PocketParity31R2] {tag} RESULT: PASS (all #31-R2 + #31-R3 gates)");
                    if (failed) Debug.LogError("[PocketParity31R2] #31-R2/R3 parity FAILED — do not ship.");
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"[PocketParity31R2] EXCEPTION: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    PocketTTS.PerfCounting = false;
                    PocketTTS.FastKernels2 = fk2;
                    PocketTTS.FastKernels3 = fk3;
                    PocketTTS.OverlapMimi = ovm;
                    PocketTTS.ArBatchRamp = ramp;
                    flm?.Dispose();
                    weights?.Dispose();
                    tts?.Dispose();
                    EditorUtility.ClearProgressBar();
                }
            }

            // bitwise-equality gate (#31-R3 determinism contracts)
            static void BitGate(string tag, float[] a, float[] b)
            {
                if (a.Length != b.Length)
                {
                    failed = true;
                    Debug.LogError($"[PocketParity31R2] {tag}: LENGTH MISMATCH {a.Length} vs {b.Length}  <-- FAIL");
                    return;
                }
                double mx = 0; int wi = -1;
                for (int i = 0; i < a.Length; i++)
                {
                    double d = System.Math.Abs(a[i] - (double)b[i]);
                    if (d > mx) { mx = d; wi = i; }
                }
                bool bad = mx != 0.0;
                failed |= bad;
                Debug.Log($"[PocketParity31R2] {tag}: " +
                          (bad ? $"FAIL (maxAbs {mx:E2} @ {wi} — expected bit-exact)" : "PASS (bit-exact)"));
            }

            static void PrintLoopStats(string tag, int frames)
            {
                long d = PocketTTS.StatDispatches - PocketTTS.StatLoopStartDisp;
                long r = PocketTTS.StatBlockingReads - PocketTTS.StatLoopStartReads;
                long u = PocketTTS.StatUploads - PocketTTS.StatLoopStartUps;
                float fr = Mathf.Max(frames, 1);
                Debug.Log($"[PocketParity31R2] [instrument] {tag}: {frames} frames | " +
                          $"AR-loop dispatches {d} ({d / fr:F1}/frame) | " +
                          $"blocking reads {r} ({r / fr:F2}/frame, wait {PocketTTS.StatReadWaitMs:F1} ms total) | " +
                          $"async reads {PocketTTS.StatAsyncReads} | uploads {u} ({u / fr:F2}/frame) | " +
                          $"legacy-loop CPU split: token {PocketTTS.StatTokenCpuMs:F1} / decode-call " +
                          $"{PocketTTS.StatDecodeCallMs:F1} / flow-call {PocketTTS.StatFlowCallMs:F1} ms");
            }

            // LN-folded GEMV (GemvLN16/Q8) vs the legacy LayerNormT + routed Linear composite
            static void LNKernelAB(PocketTTSFlowLM flm, string lnName, string wName,
                                   int inDim, int outDim, int act, uint seed)
            {
                float[] x = RandVec(inDim, seed);
                float[] a = flm.RunLNLinearForProbe(lnName, wName, x, inDim, outDim, act, 1e-5f, fused: false);
                float[] b = flm.RunLNLinearForProbe(lnName, wName, x, inDim, outDim, act, 1e-5f, fused: true);
                Grade($"GemvLN {lnName}+{wName} [{inDim}->{outDim}]", a, b, K_CORR, K_MAXABS);
            }

            // legacy-vs-coal A/B of one manifest Linear via the FlowLM router (T==1 -> GEMV, else GEMM)
            static void KernelAB(PocketTTSFlowLM flm, string name, int T, int inDim, int outDim,
                                 bool bias, int act, uint seed)
            {
                float[] x = RandVec(T * inDim, seed);
                PocketTTS.FastKernels2 = false;
                float[] a = flm.RunLinearForProbe(name, x, T, inDim, outDim, bias, act);
                PocketTTS.FastKernels2 = true;
                float[] b = flm.RunLinearForProbe(name, x, T, inDim, outDim, bias, act);
                Grade($"{(T == 1 ? "GEMV" : $"GEMM T={T}")} {name} [{inDim}->{outDim}]", a, b, K_CORR, K_MAXABS);
            }

            static void Grade(string tag, float[] a, float[] b, float corrGate, float maxAbsGate)
            {
                var (mx, mae, corr) = Diff(a, b);
                bool bad = corr < corrGate || mx > maxAbsGate;
                failed |= bad;
                Debug.Log($"[PocketParity31] {tag}: maxAbs {mx:E2} MAE {mae:E2} corr {corr:F8} " +
                          $"(gates corr>={corrGate}, maxAbs<={maxAbsGate:E1}){(bad ? "  <-- FAIL" : "  PASS")}");
            }

            static float[] Slice(float[] src, int off, int n)
            {
                n = Mathf.Min(n, src.Length - off);
                var r = new float[n];
                System.Array.Copy(src, off, r, 0, n);
                return r;
            }

            // xorshift32 uniform in [-2, 2] — machine-independent
            static float[] RandVec(int n, uint seed)
            {
                var v = new float[n];
                uint rng = seed | 1u;
                for (int i = 0; i < n; i++)
                {
                    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                    v[i] = (rng / 4294967295f) * 4f - 2f;
                }
                return v;
            }

            // deterministic Box-Muller gaussian (std given) off xorshift32
            static float[] RandGauss(int n, uint seed, float std)
            {
                var a = new float[n];
                uint rng = seed | 1u;
                float Next()
                {
                    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                    return rng / 4294967295f;
                }
                for (int i = 0; i < n; i += 2)
                {
                    double u1 = 1.0 - Next() * 0.9999999, u2 = Next();
                    double r = System.Math.Sqrt(-2.0 * System.Math.Log(u1)) * std;
                    a[i] = (float)(r * System.Math.Cos(2.0 * System.Math.PI * u2));
                    if (i + 1 < n) a[i + 1] = (float)(r * System.Math.Sin(2.0 * System.Math.PI * u2));
                }
                return a;
            }

            static (float, float, float) Diff(float[] a, float[] b)
            {
                int n = Mathf.Min(a.Length, b.Length);
                double mx = 0, mae = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                for (int i = 0; i < n; i++)
                {
                    double dd = System.Math.Abs(a[i] - b[i]); mx = System.Math.Max(mx, dd); mae += dd;
                    sa += a[i]; sb += b[i]; saa += (double)a[i] * a[i]; sbb += (double)b[i] * b[i]; sab += (double)a[i] * b[i];
                }
                double cov = sab / n - (sa / n) * (sb / n);
                double va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n);
                return ((float)mx, (float)(mae / n), (float)(cov / System.Math.Sqrt(System.Math.Max(va * vb, 1e-20))));
            }
        }
    }
}
#endif
