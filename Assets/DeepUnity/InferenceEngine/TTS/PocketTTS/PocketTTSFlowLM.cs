using System;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        using Cfg = PocketTTSConfig;

        // FlowLM front half: text ids -> embed -> [voice ; text ; latent] causal transformer
        // (6L/1024d/16h RoPE, UNBOUNDED causal, no layer_scale) -> out_norm -> SimpleMLPAdaLN flow
        // head (input_proj, 6 res_blocks with adaLN modulate/gate, FinalLayer) -> velocity [32];
        // latent = noise + velocity (1 Euler step). Reuses PocketTTSCS kernels. FIRST DRAFT.
        public class PocketTTSFlowLM : IDisposable
        {
            readonly ComputeShader cs;
            readonly PocketTTSWeights w;
            int kCopy, kSliceCols, kAdd, kAct, kLinear, kLinearQ8, kLN, kRope, kAttn, kAttnKV, kAppendKV, kMod, kGate, kRms;
            int kLinearCoal, kLinearQ8Coal, kLinearGemm, kLinearQ8Gemm;          // #31-P coalesced
            int kFlowRB, kFlowRBQ8, kFlowFinal, kFlowFinalQ8;                    // #31-P fused flow head
            int kGemv16, kGemvQ8, kGemvLN16, kGemvLNQ8;                          // #31-R2 mode/LN GEMVs
            int kQkvPrep, kEosNorm, kEosNormQ8, kCommit, kSlice;                 // #31-R2 AR-frame kernels
            ComputeBuffer tfIn, tfNorm, qkv, q, k, v, attn, ff, tmp, onesB, zerosB;
            ComputeBuffer fx, fy, fh, fmod, ftmp, ftime0, ftime1;
            int cap;

            // ---- #31-R2 GPU-resident AR frame state ----
            // Zero per-frame CPU<->GPU crossings: the previous latent lives in d1Lat (feedback for
            // the on-GPU input_linear), noise for a K-frame block is uploaded ONCE into noiseK, and
            // each frame's [eos | latent[32]] lands in its eosLat slot (stride 33) — read back once
            // per block (offline) / once per frame async (streaming).
            public const int EOSLAT_STRIDE = PocketTTSConfig.LDIM + 1;           // [eos | latent]
            ComputeBuffer bosLat, d1Noise, d1Lat, noiseK, eosLat;
            float[] _noiseFlat;                                                  // upload scratch
            int _arCap;                                                          // allocated K slots

            // instrumentation funnel (#31-R2 item 1): every FlowLM dispatch goes through Disp so
            // PocketTTS.PerfCounting can report EXACT per-frame dispatch counts. Zero cost when off.
            void Disp(int kernel, int gx, int gy, int gz)
            {
                cs.Dispatch(kernel, gx, gy, gz);
                if (PocketTTS.PerfCounting) PocketTTS.StatDispatches++;
            }
            static void CountUpload() { if (PocketTTS.PerfCounting) PocketTTS.StatUploads++; }
            void BlockingRead(ComputeBuffer b, float[] dst, int n)
            {
                if (!PocketTTS.PerfCounting) { b.GetData(dst, 0, 0, n); return; }
                var sw = System.Diagnostics.Stopwatch.StartNew();
                b.GetData(dst, 0, 0, n);
                PocketTTS.StatBlockingReads++;
                PocketTTS.StatReadWaitMs += sw.Elapsed.TotalMilliseconds;
            }

            // #31-P routing gate: coalesced GEMV/GEMM for every K that is a multiple of 128 (the
            // GEMM stages 128-column chunks; the GEMV requires K <= COAL_KMAX groupshared floats).
            // Pocket Ks: 32 (input_proj — stays legacy), 256, 512, 1024, 4096 — all %128==0 above 32.
            const int COAL_KMAX = 4096;                                          // = PVC_XMAX in the shader
            static bool CoalEligible(int inDim)
                => PocketTTS.FastKernels2 && inDim % 128 == 0 && inDim <= COAL_KMAX;

            // ---- KV-cache incremental decode (P5) ----
            ComputeBuffer[] kCache, vCache;   // per-layer [maxLen, DIM]
            ComputeBuffer d1In, d1Norm, d1Qkv, d1Q, d1K, d1V, d1Attn, d1Ff, d1Tmp, d1Out;  // 1-row scratch
            int kvLen;                        // rows currently cached (prefill + generated tokens)
            int kvCap;                        // allocated cache length

            // ---- #32: retained voice-prompt KV (cross-clause) ----
            // Cache rows [0, promptRows) hold the [bbv | voicePrompt] speaker conditioning. Those
            // rows are IDENTICAL in content AND in absolute position on every clause of a reply, so
            // re-prefilling them is pure waste (the measured 392-604 ms clause dead window).
            // promptKey is the IDENTITY of the array those rows were built from — NOT a "have I
            // prefilled before" flag: this FlowLM is SHARED between voices (two NPCs with different
            // audio prompts alternate on one engine), so a rebind MUST fall back to a full prefill.
            // Only the speaker conditioning is retained — no text, no latents, no EOS — so the model
            // still sees exactly one utterance per clause.
            object promptKey;
            int promptRows;

            /// <summary>Flow-head intermediate tap for P3 localization (name, readback values).</summary>
            public Action<string, float[]> FlowTap;
            void Tap(string name, ComputeBuffer b, int n)
            {
                if (FlowTap == null) return;
                var a = new float[n]; b.GetData(a, 0, 0, n); FlowTap(name, a);
            }

            public PocketTTSFlowLM(PocketTTSWeights weights)
            {
                w = weights;
                cs = DeepUnityMeta.PocketTTSCS;
                kCopy = cs.FindKernel("CopyBuffer");
                kSliceCols = cs.FindKernel("SliceCols");
                kAdd = cs.FindKernel("AddResidual");
                kAct = cs.FindKernel("Activate");
                kLinear = cs.FindKernel("LinearBias");
                kLinearQ8 = cs.FindKernel("LinearBiasQ8");
                kLN = cs.FindKernel("LayerNormT");
                kRope = cs.FindKernel("ApplyRoPE");
                kAttn = cs.FindKernel("CausalAttention");
                kAttnKV = cs.FindKernel("CausalAttentionKV");
                kAppendKV = cs.FindKernel("AppendKV");
                kMod = cs.FindKernel("Modulate");
                kGate = cs.FindKernel("GateAdd");
                kRms = cs.FindKernel("RMSNormFlow");
                kLinearCoal = cs.FindKernel("LinearBiasCoal");
                kLinearQ8Coal = cs.FindKernel("LinearBiasQ8Coal");
                kLinearGemm = cs.FindKernel("LinearBiasGemm");
                kLinearQ8Gemm = cs.FindKernel("LinearBiasQ8Gemm");
                kFlowRB = cs.FindKernel("FlowResBlockFused");
                kFlowRBQ8 = cs.FindKernel("FlowResBlockFusedQ8");
                kFlowFinal = cs.FindKernel("FlowFinalFused");
                kFlowFinalQ8 = cs.FindKernel("FlowFinalFusedQ8");
                kGemv16 = cs.FindKernel("Gemv16");
                kGemvQ8 = cs.FindKernel("GemvQ8");
                kGemvLN16 = cs.FindKernel("GemvLN16");
                kGemvLNQ8 = cs.FindKernel("GemvLNQ8");
                kQkvPrep = cs.FindKernel("ARQkvPrep");
                kEosNorm = cs.FindKernel("AREosNorm");
                kEosNormQ8 = cs.FindKernel("AREosNormQ8");
                kCommit = cs.FindKernel("ARCommit");
                kSlice = cs.FindKernel("CopySlice");
            }

            static int Div256(int n) => (n + 255) / 256;
            static void Grow(ref ComputeBuffer b, int n) { if (b != null && b.count >= n) return; b?.Release(); b = new ComputeBuffer(Math.Max(n, 1), 4, ComputeBufferType.Structured); }

            // ---------- generic ops (fp16 OR int8 weights via manifest Get) ----------
            // A '<name>.weight.scales' sibling in the manifest => the weight is q8 (int8 4-per-uint
            // + per-row fp16 scale): route to LinearBiasQ8. Chosen PER TENSOR (fp16 dirs have no
            // .scales, so this is a no-op there). All pocket q8 in_dims are % 4 == 0.
            // #31-P: eligible shapes (K % 128 == 0) route to the coalesced GEMV (T==1, the AR-loop
            // shape) / GEMM (T>1, RunTransformer) kernels behind PocketTTS.FastKernels2. Same
            // uniforms/buffers, different thread layout — reductions reorder float sums, so the new
            // path is parity-gated (maxAbs/corr), NOT bit-exact (see DEEPOPT.md).
            void Linear(string name, ComputeBuffer x, ComputeBuffer y, int T, int inDim, int outDim, bool bias, int act = 0)
            {
                ComputeBuffer scales = w.Has(name + ".weight.scales") ? w.Get(name + ".weight.scales") : null;
                if (CoalEligible(inDim))
                {
                    int kc = T == 1 ? (scales != null ? kLinearQ8Coal : kLinearCoal)
                                    : (scales != null ? kLinearQ8Gemm : kLinearGemm);
                    cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                    cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                    cs.SetBuffer(kc, "X", x); cs.SetBuffer(kc, "W", w.Get(name + ".weight"));
                    cs.SetBuffer(kc, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                    if (scales != null) cs.SetBuffer(kc, "W_scales", scales);
                    cs.SetBuffer(kc, "Y", y);
                    if (T == 1) Disp(kc, (outDim + 7) / 8, 1, 1);
                    else
                    {
                        cs.SetInt("elem_offset", 0);   // GEMM token offset — whole-op dispatch
                        Disp(kc, (outDim + 7) / 8, (T + 7) / 8, 1);
                    }
                    return;
                }
                int k = scales != null ? kLinearQ8 : kLinear;
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                cs.SetBuffer(k, "X", x); cs.SetBuffer(k, "W", w.Get(name + ".weight"));
                cs.SetBuffer(k, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                if (scales != null) cs.SetBuffer(k, "W_scales", scales);
                cs.SetBuffer(k, "Y", y);
                Disp(k, 1, (T + 7) / 8, (outDim + 31) / 32);
            }

            // #29: row-sliced Linear — splits T rows into ~PocketTTS.GpuMacsPerTick sub-dispatches
            // (via the shader's elem_offset row offset) with a yield between them, so one fat
            // prefill matmul never owns a whole frame's GPU. The MAC budget is runtime
            // self-calibrated (no GPU-specific tuning). Re-applies every uniform per sub-dispatch
            // (the shader object is shared with Mimi, which may dispatch between our yields) and
            // resets elem_offset synchronously — it must never stay non-zero across a yield.
            System.Collections.IEnumerator LinearRows(string name, ComputeBuffer x, ComputeBuffer y,
                int T, int inDim, int outDim, bool bias, int act = 0)
            {
                long macs = (long)T * inDim * outDim;
                int slices = (int)Math.Min(T, (macs + PocketTTS.GpuMacsPerTick - 1) / PocketTTS.GpuMacsPerTick);
                int rows = (T + slices - 1) / slices;
                for (int r0 = 0; r0 < T; r0 += rows)
                {
                    ComputeBuffer scales = w.Has(name + ".weight.scales") ? w.Get(name + ".weight.scales") : null;
                    int span = Math.Min(rows, T - r0);
                    // #31-P: eligible slices go through the coalesced GEMM — elem_offset is the
                    // FIRST TOKEN ROW; a ragged tail tile recomputes <=7 rows also covered by the
                    // next slice with the SAME kernel -> identical values, parity-neutral (#29 rule).
                    if (CoalEligible(inDim))
                    {
                        int kc = scales != null ? kLinearQ8Gemm : kLinearGemm;
                        cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                        cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                        cs.SetBuffer(kc, "X", x); cs.SetBuffer(kc, "W", w.Get(name + ".weight"));
                        cs.SetBuffer(kc, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                        if (scales != null) cs.SetBuffer(kc, "W_scales", scales);
                        cs.SetBuffer(kc, "Y", y);
                        cs.SetInt("elem_offset", r0);
                        Disp(kc, (outDim + 7) / 8, (span + 7) / 8, 1);
                        cs.SetInt("elem_offset", 0);
                        if (r0 + rows < T) yield return null;
                        continue;
                    }
                    int k = scales != null ? kLinearQ8 : kLinear;
                    cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                    cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                    cs.SetBuffer(k, "X", x); cs.SetBuffer(k, "W", w.Get(name + ".weight"));
                    cs.SetBuffer(k, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                    if (scales != null) cs.SetBuffer(k, "W_scales", scales);
                    cs.SetBuffer(k, "Y", y);
                    cs.SetInt("elem_offset", r0);
                    Disp(k, 1, (span + 7) / 8, (outDim + 31) / 32);
                    cs.SetInt("elem_offset", 0);
                    if (r0 + rows < T) yield return null;
                }
            }

            void LayerNorm(ComputeBuffer gamma, ComputeBuffer beta, ComputeBuffer x, ComputeBuffer y, int T, int dim, float eps)
            {
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", dim); cs.SetFloat("norm_eps", eps);
                cs.SetBuffer(kLN, "norm_input", x); cs.SetBuffer(kLN, "norm_output", y);
                cs.SetBuffer(kLN, "ln_gamma", gamma); cs.SetBuffer(kLN, "ln_beta", beta);
                Disp(kLN, Div256(T), 1, 1);
            }

            void Act(ComputeBuffer b, int n, int act) { cs.SetInt("buffer_size", n); cs.SetInt("activation_type", act); cs.SetFloat("leaky_slope", 0.01f); cs.SetBuffer(kAct, "inout_buf", b); Disp(kAct, Div256(n), 1, 1); }
            void AddR(ComputeBuffer a, ComputeBuffer b, int n) { cs.SetInt("buffer_size", n); cs.SetBuffer(kAdd, "buf_a", a); cs.SetBuffer(kAdd, "buf_b", b); Disp(kAdd, Div256(n), 1, 1); }
            void Copy(ComputeBuffer a, ComputeBuffer b, int n) { cs.SetInt("buffer_size", n); cs.SetBuffer(kCopy, "buf_a", a); cs.SetBuffer(kCopy, "buf_b", b); Disp(kCopy, Div256(n), 1, 1); }

            // ================= P2: text embed lookup (CPU gather) =================
            // #29: the table MUST be cached — ReadFloats re-reads the 8 MB file and fp16-decodes
            // 4.1M values (~90 ms), and EmbedLookup runs at EVERY clause start. Uncached, this was
            // the once-per-clause GEN+SPK spike in the talk-perf report.
            float[] _embTable;
            public float[] EmbedLookup(int[] ids)
            {
                _embTable ??= w.ReadFloats("flow_lm/conditioner/embed.weight");   // [4001,1024] widened
                float[] outv = new float[ids.Length * Cfg.DIM];
                for (int i = 0; i < ids.Length; i++)
                    Array.Copy(_embTable, ids[i] * Cfg.DIM, outv, i * Cfg.DIM, Cfg.DIM);
                return outv;
            }

            // ================= P2: 6L causal transformer over a full [L,1024] sequence =================
            // Returns the transformer output buffer (PRE out_norm) [L,1024]; last row = xformer_out.
            public ComputeBuffer RunTransformer(float[] inputSeq, int L)
            {
                int dim = Cfg.DIM, heads = Cfg.TF_HEADS, hd = Cfg.HEAD_DIM;
                Grow(ref tfIn, L * dim); Grow(ref tfNorm, L * dim); Grow(ref qkv, L * 3 * dim);
                Grow(ref q, L * dim); Grow(ref k, L * dim); Grow(ref v, L * dim);
                Grow(ref attn, L * dim); Grow(ref ff, L * Cfg.TF_FFN); Grow(ref tmp, L * dim);
                if (cap < L) cap = L;
                tfIn.SetData(inputSeq, 0, 0, L * dim);
                float attScale = 1f / Mathf.Sqrt(hd);
                for (int li = 0; li < Cfg.TF_LAYERS; li++)
                {
                    string lp = $"flow_lm/transformer/layers/{li}";
                    // self-attn: x + attn(norm1(x))  (no layer_scale)
                    LayerNorm(w.Get(lp + "/norm1.weight"), w.Get(lp + "/norm1.bias"), tfIn, tfNorm, L, dim, 1e-5f);
                    Linear(lp + "/self_attn/in_proj", tfNorm, qkv, L, dim, 3 * dim, bias: false);
                    Slice(qkv, q, L, 3 * dim, dim, 0);
                    Slice(qkv, k, L, 3 * dim, dim, dim);
                    Slice(qkv, v, L, 3 * dim, dim, 2 * dim);
                    RoPE(q, L, heads, hd); RoPE(k, L, heads, hd);
                    cs.SetInt("seq_len", L); cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd);
                    cs.SetInt("rope_on", 1); cs.SetFloat("scale", attScale); cs.SetInt("attn_context", 0); // 0 = unbounded
                    cs.SetBuffer(kAttn, "Q", q); cs.SetBuffer(kAttn, "K", k); cs.SetBuffer(kAttn, "V", v);
                    cs.SetBuffer(kAttn, "AttendedValues", attn);
                    Disp(kAttn, L, heads, 1);
                    Linear(lp + "/self_attn/out_proj", attn, tmp, L, dim, dim, bias: false);
                    AddR(tfIn, tmp, L * dim);
                    // ffn: x + linear2(gelu(linear1(norm2(x))))
                    LayerNorm(w.Get(lp + "/norm2.weight"), w.Get(lp + "/norm2.bias"), tfIn, tfNorm, L, dim, 1e-5f);
                    Linear(lp + "/linear1", tfNorm, ff, L, dim, Cfg.TF_FFN, bias: false, act: 2);  // GELU exact
                    Linear(lp + "/linear2", ff, tmp, L, Cfg.TF_FFN, dim, bias: false);
                    AddR(tfIn, tmp, L * dim);
                }
                return tfIn;   // [L,1024] pre out_norm
            }

            void Slice(ComputeBuffer src, ComputeBuffer dst, int T, int inDim, int outDim, int colOff)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim); cs.SetInt("copy_src_offset", colOff);
                cs.SetBuffer(kSliceCols, "X", src); cs.SetBuffer(kSliceCols, "Y", dst);
                Disp(kSliceCols, Div256(T * outDim), 1, 1);
            }
            void RoPE(ComputeBuffer b, int T, int heads, int hd, int posOffset = 0)
            {
                cs.SetInt("seq_len", T); cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd);
                cs.SetInt("pos_offset", posOffset); cs.SetFloat("rope_theta", Cfg.ROPE_THETA);
                cs.SetBuffer(kRope, "inout_buf", b); Disp(kRope, Div256(T * heads * (hd / 2)), 1, 1);
            }

            // ================= P5: KV-cache incremental decode =================
            // KV decode ≡ the full causal forward of RunTransformer (unbounded causal + interleaved
            // RoPE at absolute positions), just amortized: each row's K/V is written once and reused.
            // Prefill populates the caches over the prompt; PrefillKV/DecodeStepKV replace the
            // full-forward loop in the AR generation path. RunTransformer stays untouched (P2 gate).

            static void GrowKV(ref ComputeBuffer b, int n) { if (b != null && b.count >= n) return; b?.Release(); b = new ComputeBuffer(Math.Max(n, 1), 4, ComputeBufferType.Structured); }

            /// <summary>Pre-allocate every clause-lifetime buffer at the caller's worst-case sizes,
            /// ONE actual driver allocation per MoveNext — so neither the warmup synthesis nor the
            /// first real clause ever allocates or grows mid-frame. The 2026-07-30 spike hunt
            /// measured exactly that: a 174 ms frame (warmup's cold EnsureKV + prefill scratch) and
            /// a 286 ms frame (the first real clause regrowing it all, kvCap ~138 -> ~700) — the
            /// same driver-stall class as the 250-550 ms monolithic frees documented on DisposeSlow.
            /// <para>Cost honesty (verifier 2026-07-30): allocation cost calibrated from that log is
            /// ~6-14 ms per create (byte-model up to ~12 ms/MB, i.e. ~40 ms for a 3 MB KV buffer),
            /// so these frames are NOT free — they are simply placed in the walk-up, where a
            /// 20-40 ms frame is invisible next to the same bytes landing mid-conversation. A
            /// buffer is atomic; it cannot be split finer than one per frame.</para>
            /// <para>Yields happen ONLY on frames that actually allocated: a re-run over covered
            /// buffers completes without yielding once, so re-prewarming is a same-frame no-op.</para>
            /// <para>Growth stays possible afterwards (a clause beyond the caller's bound pays the
            /// old cost once). Value-neutral: fresh buffers carry no state until a prefill writes
            /// them, and a bigger kvCap only means later EnsureKV calls no-op — which also fixes a
            /// real #32 miss: CanReusePromptKV demands kvCap >= maxTotal, so before this, EVERY
            /// clause longer than all previous ones was refused and re-prefilled cold (the hunt log
            /// shows those as mid-reply `prefill` spikes at t=32 and t=39). Pinned at the bound,
            /// the retained-prompt path now serves every clause under it.</para>
            /// <para>TWO ordering/ownership rules, both load-bearing:
            /// (1) kvCap is published LAST, after every layer's buffers exist — published first, a
            /// prefill racing this coroutine would see "capacity ok" in EnsureKV and dispatch
            /// against still-small caches.
            /// (2) NEVER grow while kvLen > 0 (verifier finding A): GrowKV releases-and-recreates,
            /// and a concurrent clause on the shared engine owns live rows in these buffers —
            /// growing under it replaces its prompt K/V with zeroed memory that RetainPromptKV
            /// would then mark as the retained prompt (silent corrupt voice for up to
            /// PROMPT_REUSE_LIMIT clauses). Preallocation is for idle engines; a live one keeps
            /// the sizes it already negotiated.</para></summary>
            public System.Collections.IEnumerator PreallocateYielding(int maxLp, int maxTotal)
            {
                int dim = Cfg.DIM;
                if (kCache == null) { kCache = new ComputeBuffer[Cfg.TF_LAYERS]; vCache = new ComputeBuffer[Cfg.TF_LAYERS]; }
                if (kvCap < maxTotal && kvLen == 0)   // rule (2): idle engines only
                {
                    InvalidatePromptKV();   // same contract as EnsureKV: growth drops retained rows
                    for (int li = 0; li < Cfg.TF_LAYERS; li++)
                    {
                        bool grewK = kCache[li] == null || kCache[li].count < maxTotal * dim;
                        GrowKV(ref kCache[li], maxTotal * dim);
                        if (grewK) yield return null;            // one ~3 MB create per frame
                        bool grewV = vCache[li] == null || vCache[li].count < maxTotal * dim;
                        GrowKV(ref vCache[li], maxTotal * dim);
                        if (grewV) yield return null;
                    }
                    kvCap = maxTotal;        // published last — rule (1) above
                }
                // single-row scratch: ~40 KB all told, one frame is plenty
                GrowKV(ref d1In, dim); GrowKV(ref d1Norm, dim); GrowKV(ref d1Qkv, 3 * dim);
                GrowKV(ref d1Q, dim); GrowKV(ref d1K, dim); GrowKV(ref d1V, dim);
                GrowKV(ref d1Attn, dim); GrowKV(ref d1Ff, Cfg.TF_FFN); GrowKV(ref d1Tmp, dim);
                GrowKV(ref d1Out, dim);
                // the block-prefill scratch, sized by prompt rows — without this the first clause
                // LONGER than the warmup's regrows every one of these mid-conversation. One
                // create per frame, skipping frames for buffers already covered.
                var scratch = new (System.Func<ComputeBuffer> get, System.Action grow)[]
                {
                    (() => tfIn,   () => Grow(ref tfIn,   maxLp * dim)),
                    (() => tfNorm, () => Grow(ref tfNorm, maxLp * dim)),
                    (() => qkv,    () => Grow(ref qkv,    maxLp * 3 * dim)),
                    (() => q,      () => Grow(ref q,      maxLp * dim)),
                    (() => k,      () => Grow(ref k,      maxLp * dim)),
                    (() => v,      () => Grow(ref v,      maxLp * dim)),
                    (() => attn,   () => Grow(ref attn,   maxLp * dim)),
                    (() => tmp,    () => Grow(ref tmp,    maxLp * dim)),
                    (() => ff,     () => Grow(ref ff,     maxLp * Cfg.TF_FFN)),
                };
                foreach (var (get, grow) in scratch)
                {
                    var before = get();   // Grow releases-and-news on real growth -> reference moves
                    grow();
                    if (!ReferenceEquals(before, get())) yield return null;   // it actually allocated
                }
                EnsureAr(Mathf.Clamp(PocketTTS.StreamArBatchFrames, 1, 8));
            }

            void EnsureKV(int maxLen)
            {
                int dim = Cfg.DIM;
                if (kCache == null) { kCache = new ComputeBuffer[Cfg.TF_LAYERS]; vCache = new ComputeBuffer[Cfg.TF_LAYERS]; }
                if (kvCap < maxLen)
                {
                    // #32: GrowKV RELEASES and re-creates the caches — every retained prompt row is
                    // gone. Invalidate here so a longer clause (maxTotal = Lp + maxFrames grows with
                    // the text) can never silently decode against a dropped prompt. CanReusePromptKV
                    // also refuses up front when kvCap is too small, so on the retained path this
                    // branch is unreachable; it is the safety net for every other caller.
                    InvalidatePromptKV();
                    kvCap = maxLen;
                    for (int li = 0; li < Cfg.TF_LAYERS; li++)
                    {
                        GrowKV(ref kCache[li], maxLen * dim);
                        GrowKV(ref vCache[li], maxLen * dim);
                    }
                }
                GrowKV(ref d1In, dim); GrowKV(ref d1Norm, dim); GrowKV(ref d1Qkv, 3 * dim);
                GrowKV(ref d1Q, dim); GrowKV(ref d1K, dim); GrowKV(ref d1V, dim);
                GrowKV(ref d1Attn, dim); GrowKV(ref d1Ff, Cfg.TF_FFN); GrowKV(ref d1Tmp, dim);
                GrowKV(ref d1Out, dim);
            }

            // Compute this layer's K,V for one row (already in d1Norm = norm1(x)) and store them at
            // absolute cache row `pos`. Returns nothing; caches now hold [0..pos] valid rows.
            void AppendRowKV(string lp, int layer, int pos, int heads, int hd)
            {
                int dim = Cfg.DIM;
                Linear(lp + "/self_attn/in_proj", d1Norm, d1Qkv, 1, dim, 3 * dim, bias: false);
                Slice(d1Qkv, d1Q, 1, 3 * dim, dim, 0);
                Slice(d1Qkv, d1K, 1, 3 * dim, dim, dim);
                Slice(d1Qkv, d1V, 1, 3 * dim, dim, 2 * dim);
                RoPE(d1Q, 1, heads, hd, pos);   // query RoPE at its absolute position
                RoPE(d1K, 1, heads, hd, pos);   // key RoPE at its absolute position
                cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd); cs.SetInt("pos_offset", pos);
                cs.SetBuffer(kAppendKV, "K", d1K); cs.SetBuffer(kAppendKV, "V", d1V);
                cs.SetBuffer(kAppendKV, "KCache", kCache[layer]); cs.SetBuffer(kAppendKV, "VCache", vCache[layer]);
                Disp(kAppendKV, Div256(dim), 1, 1);
            }

            /// <summary>Prefill the KV caches over the prompt rows [bos_before_voice ; voice ; text]
            /// (flattened [Lp,1024]). After this kvLen == Lp. The prompt outputs are discarded (the
            /// AR loop only ever needs the last-position output of each subsequent single-token step).
            /// Synchronous form (probes/offline) — drains the yielding enumerator: IDENTICAL dispatches
            /// in identical order, so P2/P4 parity is untouched.</summary>
            public void PrefillKV(float[] promptSeq, int Lp, int maxTotal)
            {
                var e = PrefillKVYielding(promptSeq, Lp, maxTotal);
                while (e.MoveNext()) { }
            }

            /// <summary>Streaming form (bug C): yields after each of the 6 transformer layers so the
            /// ~90 ms prefill burst spreads across pump ticks (~15 ms per MoveNext) instead of
            /// freezing the frame that starts a conversation.</summary>
            public System.Collections.IEnumerator PrefillKVYielding(float[] promptSeq, int Lp, int maxTotal)
            {
                int dim = Cfg.DIM, heads = Cfg.TF_HEADS, hd = Cfg.HEAD_DIM;
                InvalidatePromptKV();   // #32: this rewrites rows from 0 — whatever was retained is stale
                EnsureKV(maxTotal);
                // Inline the exact P2 full-forward over the prompt, additionally snapshotting each
                // layer's RoPE'd K/V ([Lp, H*D], same layout as the caches) via a whole-block Copy.
                Grow(ref tfIn, Lp * dim); Grow(ref tfNorm, Lp * dim); Grow(ref qkv, Lp * 3 * dim);
                Grow(ref q, Lp * dim); Grow(ref k, Lp * dim); Grow(ref v, Lp * dim);
                Grow(ref attn, Lp * dim); Grow(ref ff, Lp * Cfg.TF_FFN); Grow(ref tmp, Lp * dim);
                tfIn.SetData(promptSeq, 0, 0, Lp * dim);
                float attScale = 1f / Mathf.Sqrt(hd);
                // #29 it.3: 4 ticks per layer (QKV | attention | linear1 | linear2), each ≲800 MMAC —
                // the old half-layer ticks (in_proj+attn ~0.8 G, linear1+linear2 ~1.5 G at Lp≈180)
                // were the 33-40 ms prefill spikes when a clause start collided with LLM decode.
                // LinearRows additionally splits any single matmul that outgrows the cap (long text).
                for (int li = 0; li < Cfg.TF_LAYERS; li++)
                {
                    string lp = $"flow_lm/transformer/layers/{li}";
                    LayerNorm(w.Get(lp + "/norm1.weight"), w.Get(lp + "/norm1.bias"), tfIn, tfNorm, Lp, dim, 1e-5f);
                    var lr = LinearRows(lp + "/self_attn/in_proj", tfNorm, qkv, Lp, dim, 3 * dim, bias: false);
                    while (lr.MoveNext()) yield return null;
                    Slice(qkv, q, Lp, 3 * dim, dim, 0);
                    Slice(qkv, k, Lp, 3 * dim, dim, dim);
                    Slice(qkv, v, Lp, 3 * dim, dim, 2 * dim);
                    RoPE(q, Lp, heads, hd); RoPE(k, Lp, heads, hd);          // positions 0..Lp-1
                    // store this layer's K/V rows [0..Lp-1] into the caches (CopyBuffer whole block)
                    Copy(kCache[li], k, Lp * dim);
                    Copy(vCache[li], v, Lp * dim);
                    yield return null;   // QKV tick | attention tick
                    cs.SetInt("seq_len", Lp); cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd);
                    cs.SetInt("rope_on", 1); cs.SetFloat("scale", attScale); cs.SetInt("attn_context", 0);
                    cs.SetBuffer(kAttn, "Q", q); cs.SetBuffer(kAttn, "K", k); cs.SetBuffer(kAttn, "V", v);
                    cs.SetBuffer(kAttn, "AttendedValues", attn);
                    Disp(kAttn, Lp, heads, 1);
                    Linear(lp + "/self_attn/out_proj", attn, tmp, Lp, dim, dim, bias: false);
                    AddR(tfIn, tmp, Lp * dim);
                    yield return null;   // attention tick | ffn ticks
                    LayerNorm(w.Get(lp + "/norm2.weight"), w.Get(lp + "/norm2.bias"), tfIn, tfNorm, Lp, dim, 1e-5f);
                    lr = LinearRows(lp + "/linear1", tfNorm, ff, Lp, dim, Cfg.TF_FFN, bias: false, act: 2);
                    while (lr.MoveNext()) yield return null;
                    yield return null;   // linear1 tick | linear2 tick
                    lr = LinearRows(lp + "/linear2", ff, tmp, Lp, Cfg.TF_FFN, dim, bias: false);
                    while (lr.MoveNext()) yield return null;
                    AddR(tfIn, tmp, Lp * dim);
                    yield return null;   // layer done
                }
                kvLen = Lp;
            }

            // Issue ALL of one decode step's dispatches (transformer + out_norm into d1Out) and
            // advance kvLen — NO readback. The sync/async wrappers below own the readback.
            void DecodeStepKVIssue(float[] tokenEmb)
            {
                int dim = Cfg.DIM, heads = Cfg.TF_HEADS, hd = Cfg.HEAD_DIM;
                int pos = kvLen;                 // absolute position of this new token
                float attScale = 1f / Mathf.Sqrt(hd);
                d1In.SetData(tokenEmb, 0, 0, dim); CountUpload();
                for (int li = 0; li < Cfg.TF_LAYERS; li++)
                {
                    string lp = $"flow_lm/transformer/layers/{li}";
                    // self-attn: x + out_proj(attn(norm1(x)))
                    LayerNorm(w.Get(lp + "/norm1.weight"), w.Get(lp + "/norm1.bias"), d1In, d1Norm, 1, dim, 1e-5f);
                    AppendRowKV(lp, li, pos, heads, hd);   // writes K/V at row `pos`, RoPE'd; d1Q RoPE'd too
                    cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd);
                    cs.SetFloat("scale", attScale); cs.SetInt("kv_len", pos + 1);
                    cs.SetBuffer(kAttnKV, "Q", d1Q);
                    cs.SetBuffer(kAttnKV, "KCache", kCache[li]); cs.SetBuffer(kAttnKV, "VCache", vCache[li]);
                    cs.SetBuffer(kAttnKV, "AttendedValues", d1Attn);
                    Disp(kAttnKV, heads, 1, 1);
                    Linear(lp + "/self_attn/out_proj", d1Attn, d1Tmp, 1, dim, dim, bias: false);
                    AddR(d1In, d1Tmp, dim);
                    // ffn
                    LayerNorm(w.Get(lp + "/norm2.weight"), w.Get(lp + "/norm2.bias"), d1In, d1Norm, 1, dim, 1e-5f);
                    Linear(lp + "/linear1", d1Norm, d1Ff, 1, dim, Cfg.TF_FFN, bias: false, act: 2);
                    Linear(lp + "/linear2", d1Ff, d1Tmp, 1, Cfg.TF_FFN, dim, bias: false);
                    AddR(d1In, d1Tmp, dim);
                }
                kvLen = pos + 1;
                // out_norm -> c (in d1Out)
                LayerNorm(w.Get("flow_lm/out_norm.weight"), w.Get("flow_lm/out_norm.bias"), d1In, d1Out, 1, dim, 1e-5f);
            }

            /// <summary>Decode ONE token: feed embedding `tokenEmb` [1024] (= input_linear output),
            /// attend over the cache, return c = out_norm(transformer_out) [1024]. Advances kvLen.
            /// Synchronous readback (probes/offline — deterministic timing).</summary>
            public float[] DecodeStepKV(float[] tokenEmb)
            {
                DecodeStepKVIssue(tokenEmb);
                float[] c = new float[Cfg.DIM];
                BlockingRead(d1Out, c, Cfg.DIM);   // per-frame pipeline drain #1 (legacy loop)
                return c;
            }

            /// <summary>Streaming form (bug C): same dispatches, but the 4 KB readback is ASYNC —
            /// the coroutine yields instead of blocking the main thread until the whole queued GPU
            /// work (prefill + decode) drains. Writes c into cOut [1024].</summary>
            public System.Collections.IEnumerator DecodeStepKVYielding(float[] tokenEmb, float[] cOut, bool async)
            {
                DecodeStepKVIssue(tokenEmb);
                var rb = ReadbackYielding(d1Out, cOut, Cfg.DIM, async);
                while (rb.MoveNext()) yield return rb.Current;   // forwards GpuWait to the pump
            }

            // Async-with-fallback GPU readback of `count` floats into dst. #29: the fallback cap
            // counts FRAMES, not MoveNexts — the voice pump re-enters tens of thousands of times
            // per frame inside its budget loop, so a MoveNext-counted cap tripped after ~2 frames
            // of waiting and the WaitForCompletion stalled the main thread on the WHOLE GPU queue
            // (the 85-116 ms outliers in the talk-perf report). Frame-counted, the hard-wait is
            // truly pathological (~10 s of no readback), never a routine slow frame.
            System.Collections.IEnumerator ReadbackYielding(ComputeBuffer buf, float[] dst, int count, bool async)
            {
                if (async && SystemInfo.supportsAsyncGPUReadback)
                {
                    var req = UnityEngine.Rendering.AsyncGPUReadback.Request(buf, count * 4, 0);
                    int startFrame = UnityEngine.Time.frameCount;
                    while (!req.done)
                    {
                        if (UnityEngine.Time.frameCount - startFrame > 600)
                        { PocketTTS.LastHeavyTick = "readback_hardwait"; req.WaitForCompletion(); break; }
                        yield return PocketTTS.GpuWait;
                    }
                    if (!req.hasError)
                    {
                        // Explicit-length copy, NOT CopyTo(dst): CopyTo demands dst.Length == the
                        // native array's length, and #StreamArBatch reads ramped blocks (count =
                        // blk*33) into a steady-K-sized dst — the mismatch threw here on the very
                        // first prewarm block (2026-07-30) and killed the voice for the session.
                        // The probes never saw it: they run AsyncReadback=false, and the sync
                        // GetData below is partial-copy tolerant. Copying `count` matches this
                        // method's contract ("readback of `count` floats into dst") on both paths.
                        Unity.Collections.NativeArray<float>.Copy(req.GetData<float>(), 0, dst, 0, count);
                        yield break;
                    }
                }
                buf.GetData(dst, 0, 0, count);   // unsupported/error fallback: sync
            }

            // #32: the cursor going back to 0 means the next appended row lands ON the prompt rows,
            // so the retained marker cannot survive a reset (every ResetKV caller re-prefills).
            public void ResetKV() { kvLen = 0; InvalidatePromptKV(); }

            // ================= #32: retained voice-prompt KV =================

            /// <summary>Rows of prompt currently retained (0 = nothing). Diagnostics/probes.</summary>
            public int RetainedPromptRows => promptRows;

            /// <summary>True when cache rows [0, rows) still hold the prompt built from
            /// <paramref name="voiceKey"/> AND the caches are big enough for <paramref name="maxTotal"/>
            /// rows — i.e. the caller may skip the prompt prefill and append only its text rows.
            /// Every clause of this check must hold or the caller MUST do the full prefill:
            /// <list type="bullet">
            /// <item>identity, not "warm": the FlowLM is shared, so a different voicePrompt array
            /// (SetVoice / CloneVoice both assign a fresh one) fails ReferenceEquals and falls back;</item>
            /// <item>row count, so a prompt of a different length can never be read as this one;</item>
            /// <item>kvCap, because EnsureKV reallocates when it grows (see there);</item>
            /// <item>buffer liveness, because play-mode exit / device loss destroys ComputeBuffers
            /// under us while this managed object still holds the references.</item>
            /// </list></summary>
            /// <para>BOUNDED SELF-HEAL (added 2026-07-28 after review). Every knob above detects a
            /// released buffer or a swapped prompt; NONE of them detects a buffer that is still alive
            /// with LOST CONTENTS. This box takes GPU device resets — PocketTTS.cs documents that during
            /// one "every dispatch silently no-ops and GetData returns zeros" — and after a reset
            /// promptKey still matches and IsValid() is still true, so a retained cache would decode
            /// every later clause against zeroed prompt rows: a corrupt voice, no error, persisting
            /// until the voice changed or the dialogue ended. Before retention, the next clause
            /// re-prefilled and healed within one clause. So force a full prefill every
            /// PROMPT_REUSE_LIMIT clauses: it restores that self-healing property at a cost of one
            /// clause in N (~4% at 24), and it is the only defence here that does not require the
            /// failure to announce itself.</para>
            public bool CanReusePromptKV(object voiceKey, int rows, int maxTotal)
                => promptRows > 0 && rows == promptRows && voiceKey != null
                   && ReferenceEquals(promptKey, voiceKey) && kvCap >= maxTotal
                   && promptReuseCount < PROMPT_REUSE_LIMIT
                   && kCache != null && kCache[0] != null && kCache[0].IsValid()
                   && vCache != null && vCache[0] != null && vCache[0].IsValid();

            const int PROMPT_REUSE_LIMIT = 24;
            int promptReuseCount;

            /// <summary>Count a retained-path clause. Called by the reuse branch so the limit above is
            /// driven by actual reuses, not by elapsed time or clause attempts.</summary>
            public void NotePromptKVReuse() => promptReuseCount++;

            /// <summary>Mark rows [0, rows) as the retained prompt for <paramref name="voiceKey"/>.
            /// Call right after a COMPLETED full prefill whose first `rows` rows were exactly
            /// [bbv | voicePrompt] — those cache rows already hold the right K/V, so retaining them
            /// costs nothing beyond this bookkeeping.</summary>
            public void RetainPromptKV(object voiceKey, int rows)
            {
                if (voiceKey == null || rows <= 0 || rows > kvLen) { InvalidatePromptKV(); return; }
                promptKey = voiceKey;
                promptRows = rows;
                promptReuseCount = 0;   // a fresh full prefill restarts the self-heal window
            }

            /// <summary>Forget the retained prompt (weight defetch/dispose, unverifiable rebind).</summary>
            public void InvalidatePromptKV() { promptKey = null; promptRows = 0; promptReuseCount = 0; }

            /// <summary>Retained-prompt clause start: park the append cursor just after the prompt
            /// rows. Everything past them is stale and gets overwritten by this clause's text rows
            /// and AR frames — the caches are a plain append region, not a ring.</summary>
            public void BeginFromRetainedPromptKV() { kvLen = promptRows; }

            /// <summary>#32: append <paramref name="count"/> already-embedded rows (flattened
            /// [count,1024] — the clause's text embeddings) at the current cursor through the PER-ROW
            /// decode path, yielding on the same MAC budget the block prefill uses. This is the
            /// retained-prompt clause start: rows [0, promptRows) are already cached, so only the
            /// ~25 text rows are computed instead of the whole ~151-row prefix.
            /// <para>Precondition: CanReusePromptKV said yes (so kvCap covers the clause and the
            /// 1-row scratch is allocated by the earlier prefill's EnsureKV).</para>
            /// <para>BIT-EXACT vs the block prefill, and not by luck:
            /// (a) every matmul routes to the same kernel tier — CoalEligible keys on in_dim only —
            /// and LinearBiasCoal's per-lane order (4 consecutive, stride 128, same PVC_REDUCE tree)
            /// is exactly LinearBiasGemm's for a single token;
            /// (b) CausalAttentionKV over kv_len rows IS CausalAttention's last-row output (same
            /// j-ascending accumulation, same online-softmax constants — see the kernel comment);
            /// (c) LinearBiasGemm keeps one accumulator per token, so a row's value is independent of
            /// Lp and of the LinearRows slicing — rows cached during a SHORTER prefill are the rows a
            /// longer prefill would have written.
            /// Proven end-to-end (sample-exact, maxAbs 0) by PocketTTSPromptCacheProbe.</para>
            /// <para>DecodeStepKVIssue also writes out_norm into d1Out; for a text row that output is
            /// meaningless and DISCARDED (1 extra LayerNorm dispatch per row) — reusing the decode
            /// path verbatim is what buys the bit-exactness above.</para></summary>
            public System.Collections.IEnumerator AppendRowsKVYielding(float[] rows, int count)
            {
                int dim = Cfg.DIM;
                // One tick per MAC-budget batch, mirroring LinearRows: a text row is ~76 MMAC
                // (6 layers x [in_proj 3.1M, out_proj 1.0M, linear1 4.2M, linear2 4.2M]), so the tick
                // dial buys 11 rows at Smooth / 19 at Balanced / 52 at Very Fast — a 25-row clause is
                // 1-3 ticks against the block prefill's 24.
                //
                // ROWS_HARD_CAP is there because a text row is CPU-ISSUE bound, not MAC bound: ~40
                // tiny dispatches per row, measured ~0.4 ms of issue on the GTX 1650 box against
                // ~76 MMAC of GPU. The MAC dial alone would hand the Very Fast tier a 52-row tick =
                // ~21 ms of uninterruptible issue, past the pump's 12 ms clause-start budget
                // (gpuBudgetMs 6 x TtsSilentRefillBudgetScale 2) — exactly the frame spike #29 sliced
                // the block prefill to avoid. 24 rows ≈ 10 ms fits that budget and never binds at the
                // smoother tiers, so the table stays the dial everywhere it can be.
                const int ROWS_HARD_CAP = 24;
                long macsPerRow = Cfg.TF_LAYERS * ((long)dim * 3 * dim + (long)dim * dim
                                                   + 2L * dim * Cfg.TF_FFN);
                int perTick = (int)Math.Max(1, PocketTTS.GpuMacsPerTick / Math.Max(macsPerRow, 1));
                perTick = Math.Min(perTick, ROWS_HARD_CAP);
                var row = new float[dim];
                for (int i = 0; i < count; i++)
                {
                    Array.Copy(rows, i * dim, row, 0, dim);
                    DecodeStepKVIssue(row);
                    if ((i + 1) % perTick == 0 && i + 1 < count) yield return null;
                }
            }

            // out_norm on the LAST row -> c [1024] (readback)
            public float[] OutNormLastRow(ComputeBuffer tfOut, int L)
            {
                int dim = Cfg.DIM;
                LayerNorm(w.Get("flow_lm/out_norm.weight"), w.Get("flow_lm/out_norm.bias"), tfOut, tfNorm, L, dim, 1e-5f);
                float[] all = new float[L * dim];
                tfNorm.GetData(all, 0, 0, L * dim);
                float[] c = new float[dim];
                Array.Copy(all, (L - 1) * dim, c, 0, dim);
                return c;
            }

            // ================= P3: SimpleMLPAdaLN flow head =================
            // velocity = flow_net(c[1024], s, t, x=noise[32]). latent = noise + velocity.
            // Split into Issue (all dispatches -> persistent fOut) + sync/async readback wrappers
            // (bug C: the sync 128-byte GetData blocked the main thread on the whole GPU queue).
            ComputeBuffer fOut;   // velocity [32] (persistent; released in Dispose)
            ComputeBuffer fNoiseIn, fCondIn;   // #29: persistent FlowHead input uploads (no per-frame alloc)
            bool _fhConstInit;                 // ones/zeros fp16 constants uploaded once

            // ---- #31-P fused flow head state ----
            // fTimeComb caches 0.5*(TimeEmbed(s,0)+TimeEmbed(t,1)) — (s,t) are CONSTANT (0,1) for
            // every AR frame, yet the legacy path recomputed both embeds (6 dispatches + a CPU
            // cos/sin table + a transient ComputeBuffer) per frame. Same kernels compute it ONCE;
            // bit-identical values thereafter. Invalidated if (s,t) ever change.
            ComputeBuffer fTimeComb;
            float _tcS, _tcT; bool _tcValid;
            // tri-state fusion capability: 0 unknown, 1 fusable, -1 mixed-quant fallback (never
            // happens with the real exporter — every flow_net 2D .weight is uniformly fp16 or q8).
            int _fuseState;
            bool CanFuseFlowHead()
            {
                if (_fuseState != 0) return _fuseState > 0;
                bool ok = true;
                bool q0 = w.Has("flow_lm/flow_net/res_blocks/0/adaLN_modulation/1.weight.scales");
                for (int i = 0; i < Cfg.FLOW_DEPTH && ok; i++)
                {
                    string p = $"flow_lm/flow_net/res_blocks/{i}";
                    ok = w.Has(p + "/adaLN_modulation/1.weight.scales") == q0
                      && w.Has(p + "/mlp/0.weight.scales") == q0
                      && w.Has(p + "/mlp/2.weight.scales") == q0;
                }
                ok = ok && w.Has("flow_lm/flow_net/final_layer/adaLN_modulation/1.weight.scales") ==
                           w.Has("flow_lm/flow_net/final_layer/linear.weight.scales");
                _fuseState = ok ? 1 : -1;
                return ok;
            }

            /// <summary>Synchronous form (probes/offline — P3/P4 parity + deterministic timing).</summary>
            public float[] FlowHead(float[] c, float[] noise, float s, float t)
            {
                FlowHeadIssue(c, noise, s, t);
                float[] vel = new float[Cfg.LDIM];
                BlockingRead(fOut, vel, Cfg.LDIM);   // per-frame pipeline drain #2 (legacy loop)
                if (FlowTap != null) FlowTap("flow_final", vel);
                return vel;   // velocity; latent = noise + vel
            }

            /// <summary>Streaming form: async readback of the velocity (yields, no main-thread stall).</summary>
            public System.Collections.IEnumerator FlowHeadYielding(float[] c, float[] noise, float s, float t,
                                                                    float[] velOut, bool async)
            {
                FlowHeadIssue(c, noise, s, t);
                var rb = ReadbackYielding(fOut, velOut, Cfg.LDIM, async);
                while (rb.MoveNext()) yield return rb.Current;   // forwards GpuWait to the pump
            }

            void FlowHeadIssue(float[] c, float[] noise, float s, float t)
            {
                // #31-P: the fused path collapses the ~50-dispatch SimpleMLPAdaLN storm into ~10
                // dispatches (input_proj + cond_embed + 2 assembles + 6 res_block kernels + 1 final
                // kernel). The legacy body below is untouched and remains the A/B + FlowTap path
                // (the persistent kernels cannot surface per-stage taps).
                if (PocketTTS.FastKernels2 && FlowTap == null && CanFuseFlowHead())
                {
                    FlowHeadIssueFused(c, noise, s, t);
                    return;
                }
                int D = Cfg.FLOW_DIM;   // 512
                Grow(ref fx, D); Grow(ref fy, D); Grow(ref fh, D); Grow(ref fmod, 3 * D);
                Grow(ref ftmp, D); Grow(ref ftime0, D); Grow(ref ftime1, D);
                // norm_final is no-affine: emulate via LayerNormT with gamma=1/beta=0. ln_gamma/ln_beta
                // are read with readH (fp16, 2-per-uint), so these MUST be fp16-PACKED, not float32.
                // #29: constants — uploaded ONCE (this ran per AR frame: 2 allocs + 2 SetDatas).
                int packed = (D + 1) / 2;
                if (!_fhConstInit)
                {
                    Grow(ref onesB, packed); Grow(ref zerosB, packed);
                    uint[] onesPk = new uint[packed]; for (int i = 0; i < packed; i++) onesPk[i] = 0x3C003C00u; // two fp16 1.0
                    onesB.SetData(onesPk); zerosB.SetData(new uint[packed]);   // zeros = fp16 0.0 pairs
                    _fhConstInit = true;
                }

                // x = input_proj(noise)  [32->512] (bias). #29: persistent input buffers — the old
                // per-call new ComputeBuffer + Release churned the D3D11 allocator every AR frame
                // (the ar_flowhead 28-32 ms spikes in the talk-perf report).
                Grow(ref fNoiseIn, Cfg.LDIM); fNoiseIn.SetData(noise); CountUpload();
                Linear("flow_lm/flow_net/input_proj", fNoiseIn, fx, 1, Cfg.LDIM, D, bias: true);
                Tap("flow_inproj", fx, D);

                // t_combined = (TimeEmbed(s,0) + TimeEmbed(t,1)) / 2
                TimeEmbed(s, 0, ftime0);
                Tap("flow_temb_s", ftime0, D);
                TimeEmbed(t, 1, ftime1);
                Tap("flow_temb_t", ftime1, D);
                Copy(fy, ftime0, D); AddR(fy, ftime1, D);   // fy = t0 + t1
                // c_emb = cond_embed(c) [1024->512]; y = t_combined/2 + c_emb  -> fold /2: y = (t0+t1)*0.5 + c_emb
                cs.SetInt("buffer_size", D); cs.SetFloat("scale_val", 0.5f);
                cs.SetBuffer(cs.FindKernel("ScaleBuf"), "inout_buf", fy); Disp(cs.FindKernel("ScaleBuf"), Div256(D), 1, 1);
                Grow(ref fCondIn, Cfg.DIM); fCondIn.SetData(c); CountUpload();
                Linear("flow_lm/flow_net/cond_embed", fCondIn, ftmp, 1, Cfg.DIM, D, bias: true);
                AddR(fy, ftmp, D);   // y = 0.5*(t0+t1) + c_emb
                Tap("flow_cond_vec", fy, D);

                // 6 res_blocks: x = x + gate * mlp(modulate(in_ln(x), shift, scale)).
                // NOTE: Modulate is OUT-OF-PLACE (read fh -> write ftime1); binding one buffer as
                // both SRV (norm_input) and UAV (norm_output) is illegal in D3D11 (was the P3 bug).
                // ftime0 = SiLU(y) scratch (temb buffers free after fy is built); ftime1 = modulated.
                for (int i = 0; i < Cfg.FLOW_DEPTH; i++)
                {
                    string p = $"flow_lm/flow_net/res_blocks/{i}";
                    Copy(ftime0, fy, D); Act(ftime0, D, 1);   // SiLU(y)
                    Linear(p + "/adaLN_modulation/1", ftime0, fmod, 1, D, 3 * D, bias: true);  // [1536]=[shift|scale|gate]
                    if (i == 0) Tap("flow_rb0_adaln", fmod, 3 * D);
                    LayerNorm(w.Get(p + "/in_ln.weight"), w.Get(p + "/in_ln.bias"), fx, fh, 1, D, 1e-6f);
                    Modulate(fh, ftime1, fmod, D, 0, D);      // ftime1 = in_ln(x)*(1+scale)+shift  (out-of-place)
                    if (i == 0) Tap("flow_rb0_modulated", ftime1, D);
                    Linear(p + "/mlp/0", ftime1, fh, 1, D, D, bias: true, act: 1);   // fh = SiLU(Linear0(mod))
                    Linear(p + "/mlp/2", fh, ftmp, 1, D, D, bias: true);            // ftmp = Linear2(...)
                    if (i == 0) Tap("flow_rb0_mlp", ftmp, D);
                    GateAdd(fx, ftmp, fmod, D, 2 * D);        // x += gate * mlp
                    if (i == 0 || i == 1 || i == 3 || i == 5) Tap($"flow_resblock{i}", fx, D);
                }

                // final: modulate(norm_final_noaffine(x), shift, scale) -> linear [512->32]
                Copy(ftime0, fy, D); Act(ftime0, D, 1);   // SiLU(y)
                Linear("flow_lm/flow_net/final_layer/adaLN_modulation/1", ftime0, fmod, 1, D, 2 * D, bias: true);  // [1024]=[shift|scale] (2 chunks, NO gate)
                LayerNorm(onesB, zerosB, fx, fh, 1, D, 1e-6f);   // norm_final (no affine)
                Modulate(fh, ftime1, fmod, D, 0, D);   // ftime1 = modulate (out-of-place)
                Tap("flow_final_prelinear", ftime1, D);
                Grow(ref fOut, Cfg.LDIM);
                Linear("flow_lm/flow_net/final_layer/linear", ftime1, fOut, 1, D, Cfg.LDIM, bias: true);
            }

            // #31-P fused flow head: identical math, restructured dispatches.
            //   x = input_proj(noise)                       (legacy GEMV — K=32 not coal-eligible)
            //   y = [cached 0.5*(temb_s+temb_t)] + cond_embed(c)
            //   6x FlowResBlockFused(y, x)                  (adaLN+LN+modulate+mlp+gate, 1 dispatch each)
            //   vel = FlowFinalFused(y, x)                  (adaLN+noaffine-LN+modulate+linear, 1 dispatch)
            // SiLU(y) is computed inside the fused kernels (the legacy Copy+Activate per block dies).
            void FlowHeadIssueFused(float[] c, float[] noise, float s, float t)
            {
                int D = Cfg.FLOW_DIM;   // 512
                Grow(ref fx, D); Grow(ref fy, D); Grow(ref ftmp, D);

                // x = input_proj(noise)  [32->512] (bias)
                Grow(ref fNoiseIn, Cfg.LDIM); fNoiseIn.SetData(noise); CountUpload();
                Linear("flow_lm/flow_net/input_proj", fNoiseIn, fx, 1, Cfg.LDIM, D, bias: true);

                EnsureTimeComb(s, t);

                // y = tcomb + cond_embed(c)  [1024->512] (bias; coal GEMV when eligible)
                Grow(ref fCondIn, Cfg.DIM); fCondIn.SetData(c); CountUpload();
                Linear("flow_lm/flow_net/cond_embed", fCondIn, ftmp, 1, Cfg.DIM, D, bias: true);
                Copy(fy, fTimeComb, D); AddR(fy, ftmp, D);

                FlowBlocksIssue();
            }

            // cached time embedding: 0.5*(TimeEmbed(s,0) + TimeEmbed(t,1)) — constant across
            // AR frames ((s,t) is always (0,1)); computed once with the SAME legacy dispatches.
            void EnsureTimeComb(float s, float t)
            {
                if (_tcValid && s == _tcS && t == _tcT) return;
                int D = Cfg.FLOW_DIM;
                Grow(ref ftmp, D); Grow(ref ftime0, D); Grow(ref ftime1, D); Grow(ref fTimeComb, D);
                TimeEmbed(s, 0, ftime0);
                TimeEmbed(t, 1, ftime1);
                Copy(fTimeComb, ftime0, D); AddR(fTimeComb, ftime1, D);
                int kScale = cs.FindKernel("ScaleBuf");
                cs.SetInt("buffer_size", D); cs.SetFloat("scale_val", 0.5f);
                cs.SetBuffer(kScale, "inout_buf", fTimeComb); Disp(kScale, Div256(D), 1, 1);
                _tcS = s; _tcT = t; _tcValid = true;
            }

            // Shared tail of the fused flow head (R1 fused path AND the R2 GPU-resident frame):
            // 6x FlowResBlockFused (x in fx updated in place, cond vector in fy) + FlowFinalFused
            // -> velocity in fOut. Identical dispatches from both callers (R1 parity intact).
            void FlowBlocksIssue()
            {
                int D = Cfg.FLOW_DIM;
                cs.SetInt("norm_dim", D); cs.SetFloat("norm_eps", 1e-6f);
                for (int i = 0; i < Cfg.FLOW_DEPTH; i++)
                {
                    string p = $"flow_lm/flow_net/res_blocks/{i}";
                    bool q8 = w.Has(p + "/adaLN_modulation/1.weight.scales");
                    int kk = q8 ? kFlowRBQ8 : kFlowRB;
                    cs.SetBuffer(kk, "X", fy);
                    cs.SetBuffer(kk, "inout_buf", fx);
                    cs.SetBuffer(kk, "W", w.Get(p + "/adaLN_modulation/1.weight"));
                    cs.SetBuffer(kk, "W_bias", w.Get(p + "/adaLN_modulation/1.bias"));
                    cs.SetBuffer(kk, "W2", w.Get(p + "/mlp/0.weight"));
                    cs.SetBuffer(kk, "W_bias2", w.Get(p + "/mlp/0.bias"));
                    cs.SetBuffer(kk, "W3", w.Get(p + "/mlp/2.weight"));
                    cs.SetBuffer(kk, "W_bias3", w.Get(p + "/mlp/2.bias"));
                    cs.SetBuffer(kk, "ln_gamma", w.Get(p + "/in_ln.weight"));
                    cs.SetBuffer(kk, "ln_beta", w.Get(p + "/in_ln.bias"));
                    if (q8)
                    {
                        cs.SetBuffer(kk, "W_scales", w.Get(p + "/adaLN_modulation/1.weight.scales"));
                        cs.SetBuffer(kk, "W_scales2", w.Get(p + "/mlp/0.weight.scales"));
                        cs.SetBuffer(kk, "W_scales3", w.Get(p + "/mlp/2.weight.scales"));
                    }
                    Disp(kk, 1, 1, 1);
                }

                // fused final layer -> velocity [32]
                Grow(ref fOut, Cfg.LDIM);
                bool q8f = w.Has("flow_lm/flow_net/final_layer/adaLN_modulation/1.weight.scales");
                int kf = q8f ? kFlowFinalQ8 : kFlowFinal;
                cs.SetInt("out_dim", Cfg.LDIM);
                cs.SetBuffer(kf, "X", fy);
                cs.SetBuffer(kf, "buf_b", fx);
                cs.SetBuffer(kf, "W", w.Get("flow_lm/flow_net/final_layer/adaLN_modulation/1.weight"));
                cs.SetBuffer(kf, "W_bias", w.Get("flow_lm/flow_net/final_layer/adaLN_modulation/1.bias"));
                cs.SetBuffer(kf, "W2", w.Get("flow_lm/flow_net/final_layer/linear.weight"));
                cs.SetBuffer(kf, "W_bias2", w.Get("flow_lm/flow_net/final_layer/linear.bias"));
                if (q8f)
                {
                    cs.SetBuffer(kf, "W_scales", w.Get("flow_lm/flow_net/final_layer/adaLN_modulation/1.weight.scales"));
                    cs.SetBuffer(kf, "W_scales2", w.Get("flow_lm/flow_net/final_layer/linear.weight.scales"));
                }
                cs.SetBuffer(kf, "Y", fOut);
                Disp(kf, 1, 1, 1);
            }

            /// <summary>Probe-only (#31-P parity): run one manifest Linear on x [T*inDim] and read
            /// back y [T*outDim]. Routing (legacy vs coalesced) follows PocketTTS.FastKernels2 —
            /// the parity probe toggles it around two calls and diffs.</summary>
            public float[] RunLinearForProbe(string name, float[] x, int T, int inDim, int outDim, bool bias, int act = 0)
            {
                var xb = new ComputeBuffer(T * inDim, 4, ComputeBufferType.Structured);
                var yb = new ComputeBuffer(T * outDim, 4, ComputeBufferType.Structured);
                xb.SetData(x);
                Linear(name, xb, yb, T, inDim, outDim, bias, act);
                var y = new float[T * outDim];
                yb.GetData(y);
                xb.Release(); yb.Release();
                return y;
            }

            // ======================= #31-R2 (FastKernels3): GPU-resident AR frame =======================
            // One frame = ~49 dispatches, ZERO CPU<->GPU crossings. The legacy loop paid 2 blocking
            // readbacks (c for the CPU EOS check, velocity for the CPU latent add) + 3 uploads (token
            // embedding, noise, c re-uploaded for cond_embed) EVERY frame — each readback drains the
            // whole GPU pipe (the measured ~50 us/dispatch was mostly this serialization, see DEEPOPT
            // §R2). Here the AR feedback (latent -> input_linear -> next token) stays on the GPU
            // (d1Lat), noise for a K-frame block is uploaded once (noiseK), and each frame writes
            // [eos | latent[32]] into its eosLat slot — read back ONCE per block (offline, blocking)
            // or once per frame (streaming, async, replacing TWO async waits).

            /// <summary>True when the GPU-resident frame path is usable: both kernel tiers on, no
            /// FlowTap (needs legacy per-op taps), flow head fusable (uniform quant, always true
            /// with the real exporter).</summary>
            public bool CanRunGpuFrames()
                => PocketTTS.FastKernels2 && PocketTTS.FastKernels3 && FlowTap == null && CanFuseFlowHead();

            void EnsureAr(int kFrames)
            {
                GrowKV(ref d1Noise, Cfg.LDIM);
                GrowKV(ref d1Lat, Cfg.LDIM);
                if (_arCap < kFrames)
                {
                    _arCap = kFrames;
                    GrowKV(ref noiseK, kFrames * Cfg.LDIM);
                    GrowKV(ref eosLat, kFrames * EOSLAT_STRIDE);
                }
                if (bosLat == null)
                {
                    bosLat = new ComputeBuffer(Cfg.LDIM, 4, ComputeBufferType.Structured);
                    bosLat.SetData(w.ReadFloats("flow_lm.bos_emb"));   // [32] — one-time upload
                }
                Grow(ref fx, Cfg.FLOW_DIM); Grow(ref fy, Cfg.FLOW_DIM); Grow(ref ftmp, Cfg.FLOW_DIM);
                Grow(ref fOut, Cfg.LDIM);
            }

            /// <summary>Upload `count` frames of noise ([32] each) into the block buffer in ONE
            /// SetData (the legacy path uploaded per frame). Rows are consumed by slot index.</summary>
            public void UploadNoiseBlock(float[][] rows, int count)
            {
                EnsureAr(Math.Max(count, 1));
                if (_noiseFlat == null || _noiseFlat.Length < count * Cfg.LDIM)
                    _noiseFlat = new float[Math.Max(count, 1) * Cfg.LDIM];
                for (int f = 0; f < count; f++)
                    Array.Copy(rows[f], 0, _noiseFlat, f * Cfg.LDIM, Cfg.LDIM);
                noiseK.SetData(_noiseFlat, 0, 0, count * Cfg.LDIM); CountUpload();
            }

            // R2 GEMV with epilogue mode (0 write / 1 Y += r / 2 Y = r + addSrc[row]). addSrc == null
            // binds X to the unread buf_b slot (same-buffer double-SRV is legal; never leave it unbound).
            void Gemv(string name, ComputeBuffer x, ComputeBuffer y, int inDim, int outDim,
                      bool bias, int act, int mode, ComputeBuffer addSrc)
            {
                ComputeBuffer scales = w.Has(name + ".weight.scales") ? w.Get(name + ".weight.scales") : null;
                int kg = scales != null ? kGemvQ8 : kGemv16;
                cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                cs.SetInt("gemv_mode", mode);
                cs.SetBuffer(kg, "X", x); cs.SetBuffer(kg, "W", w.Get(name + ".weight"));
                cs.SetBuffer(kg, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                if (scales != null) cs.SetBuffer(kg, "W_scales", scales);
                cs.SetBuffer(kg, "buf_b", addSrc ?? x);
                cs.SetBuffer(kg, "Y", y);
                Disp(kg, (outDim + 7) / 8, 1, 1);
            }

            // R2 GEMV with the preceding LayerNorm folded into the staging pass (norm1/norm2 fold).
            void GemvLN(string lnName, string wName, ComputeBuffer x, ComputeBuffer y,
                        int inDim, int outDim, bool bias, int act, float eps, int mode)
            {
                ComputeBuffer scales = w.Has(wName + ".weight.scales") ? w.Get(wName + ".weight.scales") : null;
                int kg = scales != null ? kGemvLNQ8 : kGemvLN16;
                cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                cs.SetInt("gemv_mode", mode); cs.SetFloat("norm_eps", eps);
                cs.SetBuffer(kg, "X", x); cs.SetBuffer(kg, "W", w.Get(wName + ".weight"));
                cs.SetBuffer(kg, "W_bias", bias ? w.Get(wName + ".bias") : w.Get(wName + ".weight"));
                if (scales != null) cs.SetBuffer(kg, "W_scales", scales);
                cs.SetBuffer(kg, "ln_gamma", w.Get(lnName + ".weight"));
                cs.SetBuffer(kg, "ln_beta", w.Get(lnName + ".bias"));
                cs.SetBuffer(kg, "buf_b", x);   // unread in mode 0; keep the slot bound
                cs.SetBuffer(kg, "Y", y);
                Disp(kg, (outDim + 7) / 8, 1, 1);
            }

            /// <summary>Issue ONE fully GPU-resident AR frame into slot `slot` of eosLat/noiseK.
            /// absFrame 0 sources the token from bos_emb, else from d1Lat (the previous frame's
            /// committed latent — GPU feedback, no readback). Advances kvLen. NO readback here —
            /// callers batch K frames then read eosLat once. UploadNoiseBlock must have filled the
            /// slot's noise row first. Dispatches: 1 token GEMV + 6x(GemvLN, ARQkvPrep, AttnKV,
            /// Gemv+add, GemvLN, Gemv+add) + AREosNorm + noise CopySlice + input_proj + cond_embed
            /// + 6 res_blocks + final + ARCommit = 49.</summary>
            public void DecodeFrameGpuIssue(int slot, int absFrame)
            {
                int dim = Cfg.DIM, heads = Cfg.TF_HEADS, hd = Cfg.HEAD_DIM;
                int pos = kvLen;
                float attScale = 1f / Mathf.Sqrt(hd);
                EnsureTimeComb(0f, 1f);

                // token = input_linear(prev latent | bos)  [32 -> 1024], no bias (K=32 -> legacy kernel)
                Linear("flow_lm/input_linear", absFrame == 0 ? bosLat : d1Lat, d1In, 1, Cfg.LDIM, dim, bias: false);

                for (int li = 0; li < Cfg.TF_LAYERS; li++)
                {
                    string lp = $"flow_lm/transformer/layers/{li}";
                    // norm1 folded into the qkv GEMV
                    GemvLN(lp + "/norm1", lp + "/self_attn/in_proj", d1In, d1Qkv, dim, 3 * dim,
                           bias: false, act: 0, eps: 1e-5f, mode: 0);
                    // slice q|k|v + RoPE q,k @pos + append k,v to the caches — one dispatch
                    cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd); cs.SetInt("pos_offset", pos);
                    cs.SetFloat("rope_theta", Cfg.ROPE_THETA);
                    cs.SetBuffer(kQkvPrep, "X", d1Qkv); cs.SetBuffer(kQkvPrep, "Y", d1Q);
                    cs.SetBuffer(kQkvPrep, "KCache", kCache[li]); cs.SetBuffer(kQkvPrep, "VCache", vCache[li]);
                    Disp(kQkvPrep, 1, 1, 1);
                    // attention over the cache (unchanged kernel)
                    cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd);
                    cs.SetFloat("scale", attScale); cs.SetInt("kv_len", pos + 1);
                    cs.SetBuffer(kAttnKV, "Q", d1Q);
                    cs.SetBuffer(kAttnKV, "KCache", kCache[li]); cs.SetBuffer(kAttnKV, "VCache", vCache[li]);
                    cs.SetBuffer(kAttnKV, "AttendedValues", d1Attn);
                    Disp(kAttnKV, heads, 1, 1);
                    // out_proj with the residual add folded into the epilogue (d1In += r)
                    Gemv(lp + "/self_attn/out_proj", d1Attn, d1In, dim, dim, bias: false, act: 0, mode: 1, addSrc: null);
                    // ffn: norm2 folded into linear1 (GELU), residual add folded into linear2
                    GemvLN(lp + "/norm2", lp + "/linear1", d1In, d1Ff, dim, Cfg.TF_FFN,
                           bias: false, act: 2, eps: 1e-5f, mode: 0);
                    Gemv(lp + "/linear2", d1Ff, d1In, Cfg.TF_FFN, dim, bias: false, act: 0, mode: 1, addSrc: null);
                }
                kvLen = pos + 1;

                // out_norm + eos logit in one dispatch: d1Out = out_norm(d1In); eosLat[slot*33] = eos
                bool q8e = w.Has("flow_lm/out_eos.weight.scales");
                int ke = q8e ? kEosNormQ8 : kEosNorm;
                cs.SetInt("in_dim", dim); cs.SetFloat("norm_eps", 1e-5f);
                cs.SetInt("elem_offset", slot * EOSLAT_STRIDE);
                cs.SetBuffer(ke, "X", d1In); cs.SetBuffer(ke, "Y", d1Out);
                cs.SetBuffer(ke, "ln_gamma", w.Get("flow_lm/out_norm.weight"));
                cs.SetBuffer(ke, "ln_beta", w.Get("flow_lm/out_norm.bias"));
                cs.SetBuffer(ke, "W", w.Get("flow_lm/out_eos.weight"));
                cs.SetBuffer(ke, "W_bias", w.Get("flow_lm/out_eos.bias"));
                if (q8e) cs.SetBuffer(ke, "W_scales", w.Get("flow_lm/out_eos.weight.scales"));
                cs.SetBuffer(ke, "buf_a", eosLat);
                Disp(ke, 1, 1, 1);
                cs.SetInt("elem_offset", 0);

                // this slot's noise row -> d1Noise (input_proj/commit source)
                cs.SetInt("buffer_size", Cfg.LDIM);
                cs.SetInt("copy_src_offset", slot * Cfg.LDIM); cs.SetInt("copy_dst_offset", 0);
                cs.SetBuffer(kSlice, "buf_a", d1Noise); cs.SetBuffer(kSlice, "buf_b", noiseK);
                Disp(kSlice, 1, 1, 1);

                // flow head, cond read DIRECTLY from d1Out (the legacy path read c back and
                // re-uploaded it): x = input_proj(noise); y = cond_embed(c) + tcomb (mode-2 epilogue
                // replaces the legacy Copy+AddR assemble); then the shared R1 fused blocks.
                Linear("flow_lm/flow_net/input_proj", d1Noise, fx, 1, Cfg.LDIM, Cfg.FLOW_DIM, bias: true);
                Gemv("flow_lm/flow_net/cond_embed", d1Out, fy, Cfg.DIM, Cfg.FLOW_DIM,
                     bias: true, act: 0, mode: 2, addSrc: fTimeComb);
                FlowBlocksIssue();

                // commit: d1Lat = velocity + noise (AR feedback) + the slot's latent part
                cs.SetInt("out_dim", Cfg.LDIM);
                cs.SetInt("elem_offset", slot * EOSLAT_STRIDE + 1);
                cs.SetBuffer(kCommit, "X", fOut); cs.SetBuffer(kCommit, "buf_b", d1Noise);
                cs.SetBuffer(kCommit, "Y", d1Lat); cs.SetBuffer(kCommit, "buf_a", eosLat);
                Disp(kCommit, 1, 1, 1);
                cs.SetInt("elem_offset", 0);
            }

            /// <summary>Blocking readback of `count` frame slots ([eos | latent[32]] each) into dst
            /// [count * 33]. The offline loop's ONE sync point per K-frame block.</summary>
            public void ReadEosLatBlock(int count, float[] dst) => BlockingRead(eosLat, dst, count * EOSLAT_STRIDE);

            /// <summary>Async form (streaming): one combined [eos | latent] readback per frame —
            /// replaces the legacy pair of waits (c then velocity). Yields GpuWait to the pump.</summary>
            public System.Collections.IEnumerator ReadEosLatYielding(int count, float[] dst, bool async)
            {
                if (PocketTTS.PerfCounting) PocketTTS.StatAsyncReads++;
                var rb = ReadbackYielding(eosLat, dst, count * EOSLAT_STRIDE, async);
                while (rb.MoveNext()) yield return rb.Current;
            }

            /// <summary>Probe-only: read back the current c (= d1Out, post out_norm) [1024].</summary>
            public float[] ReadCondForProbe()
            {
                var c = new float[Cfg.DIM];
                d1Out.GetData(c, 0, 0, Cfg.DIM);
                return c;
            }

            /// <summary>Probe-only (#31-R2): LN-folded GEMV vs the legacy LayerNormT + routed Linear
            /// composite on real weights. Returns y [outDim].</summary>
            public float[] RunLNLinearForProbe(string lnName, string wName, float[] x, int inDim, int outDim,
                                               int act, float eps, bool fused)
            {
                var xb = new ComputeBuffer(inDim, 4, ComputeBufferType.Structured);
                var nb = new ComputeBuffer(inDim, 4, ComputeBufferType.Structured);
                var yb = new ComputeBuffer(outDim, 4, ComputeBufferType.Structured);
                xb.SetData(x);
                if (fused)
                    GemvLN(lnName, wName, xb, yb, inDim, outDim, bias: false, act: act, eps: eps, mode: 0);
                else
                {
                    LayerNorm(w.Get(lnName + ".weight"), w.Get(lnName + ".bias"), xb, nb, 1, inDim, eps);
                    Linear(wName, nb, yb, 1, inDim, outDim, bias: false, act: act);
                }
                var y = new float[outDim];
                yb.GetData(y);
                xb.Release(); nb.Release(); yb.Release();
                return y;
            }

            // TimestepEmbedder(scalar tau): emb=cat(cos(tau*freqs),sin(tau*freqs)) [256];
            // mlp: Linear(256->512)+SiLU -> Linear(512->512) -> RMSNorm(alpha). Output in `dst` [512].
            void TimeEmbed(float tau, int idx, ComputeBuffer dst)
            {
                string p = $"flow_lm/flow_net/time_embed/{idx}";
                float[] freqs = w.ReadFloats(p + ".freqs");   // [128] (dot leaf per manifest)
                int half = freqs.Length;
                float[] emb = new float[2 * half];
                for (int i = 0; i < half; i++) { float a = tau * freqs[i]; emb[i] = Mathf.Cos(a); emb[half + i] = Mathf.Sin(a); }
                var eb = new ComputeBuffer(2 * half, 4, ComputeBufferType.Structured); eb.SetData(emb);
                Linear(p + "/mlp/0", eb, dst, 1, 2 * half, Cfg.FLOW_DIM, bias: true, act: 1);   // SiLU
                eb.Release();
                Linear(p + "/mlp/2", dst, ftmp, 1, Cfg.FLOW_DIM, Cfg.FLOW_DIM, bias: true);
                // RMSNorm(mlp/3/alpha)
                cs.SetInt("seq_len", 1); cs.SetInt("norm_dim", Cfg.FLOW_DIM); cs.SetFloat("rms_eps", 1e-5f);
                cs.SetBuffer(kRms, "norm_input", ftmp); cs.SetBuffer(kRms, "norm_output", dst);
                cs.SetBuffer(kRms, "rms_alpha", w.Get(p + "/mlp/3.alpha"));
                Disp(kRms, 1, 1, 1);
            }

            void Modulate(ComputeBuffer x, ComputeBuffer y, ComputeBuffer mod, int dim, int shiftOff, int scaleOff)
            {
                cs.SetInt("seq_len", 1); cs.SetInt("norm_dim", dim);
                cs.SetInt("mod_shift_off", shiftOff); cs.SetInt("mod_scale_off", scaleOff);
                cs.SetBuffer(kMod, "norm_input", x); cs.SetBuffer(kMod, "norm_output", y); cs.SetBuffer(kMod, "mod_vec", mod);
                Disp(kMod, Div256(dim), 1, 1);
            }
            void GateAdd(ComputeBuffer a, ComputeBuffer h, ComputeBuffer mod, int dim, int gateOff)
            {
                cs.SetInt("seq_len", 1); cs.SetInt("norm_dim", dim); cs.SetInt("mod_gate_off", gateOff);
                cs.SetBuffer(kGate, "buf_a", a); cs.SetBuffer(kGate, "buf_b", h); cs.SetBuffer(kGate, "mod_vec", mod);
                Disp(kGate, Div256(dim), 1, 1);
            }

            // input_linear [32->1024] CPU (no bias). Cached weight for the AR loop.
            float[] _inLin;
            public float[] InputLinear(float[] latent)
            {
                _inLin ??= w.ReadFloats("flow_lm/input_linear.weight");   // [1024,32]
                float[] outv = new float[Cfg.DIM];
                for (int o = 0; o < Cfg.DIM; o++)
                {
                    float s = 0f; int b = o * Cfg.LDIM;
                    for (int i = 0; i < Cfg.LDIM; i++) s += _inLin[b + i] * latent[i];
                    outv[o] = s;
                }
                return outv;
            }
            // input_linear(bos_emb) — the decode BOS latent embedding (cached: runs per clause; callers read-only)
            float[] _bosTok;
            public float[] BosLatentEmbedding() => _bosTok ??= InputLinear(w.ReadFloats("flow_lm.bos_emb"));   // bos_emb [32] (dot leaf)

            // out_eos: Linear(1024->1) + bias on the post-out_norm condition c. EOS when > threshold.
            float[] _eosW; float _eosB; bool _eosLoaded;
            public float OutEos(float[] c)
            {
                if (!_eosLoaded) { _eosW = w.ReadFloats("flow_lm/out_eos.weight"); _eosB = w.ReadFloats("flow_lm/out_eos.bias")[0]; _eosLoaded = true; }
                float s = _eosB;
                for (int i = 0; i < Cfg.DIM; i++) s += _eosW[i] * c[i];
                return s;
            }

            public void Dispose()
            {
                InvalidatePromptKV();   // #32: the cache buffers go away with this object
                kvLen = 0; kvCap = 0;
                tfIn?.Release(); tfNorm?.Release(); qkv?.Release(); q?.Release(); k?.Release(); v?.Release();
                attn?.Release(); ff?.Release(); tmp?.Release(); onesB?.Release(); zerosB?.Release();
                fx?.Release(); fy?.Release(); fh?.Release(); fmod?.Release(); ftmp?.Release(); ftime0?.Release(); ftime1?.Release();
                fOut?.Release(); fNoiseIn?.Release(); fCondIn?.Release(); fTimeComb?.Release();
                bosLat?.Release(); d1Noise?.Release(); d1Lat?.Release(); noiseK?.Release(); eosLat?.Release();
                if (kCache != null) foreach (var b in kCache) b?.Release();
                if (vCache != null) foreach (var b in vCache) b?.Release();
                d1In?.Release(); d1Norm?.Release(); d1Qkv?.Release(); d1Q?.Release(); d1K?.Release(); d1V?.Release();
                d1Attn?.Release(); d1Ff?.Release(); d1Tmp?.Release(); d1Out?.Release();
            }
        }
    }
}
