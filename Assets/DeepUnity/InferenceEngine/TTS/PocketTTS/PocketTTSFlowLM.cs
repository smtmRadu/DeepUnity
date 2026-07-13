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
            ComputeBuffer tfIn, tfNorm, qkv, q, k, v, attn, ff, tmp, onesB, zerosB;
            ComputeBuffer fx, fy, fh, fmod, ftmp, ftime0, ftime1;
            int cap;

            // ---- KV-cache incremental decode (P5) ----
            ComputeBuffer[] kCache, vCache;   // per-layer [maxLen, DIM]
            ComputeBuffer d1In, d1Norm, d1Qkv, d1Q, d1K, d1V, d1Attn, d1Ff, d1Tmp, d1Out;  // 1-row scratch
            int kvLen;                        // rows currently cached (prefill + generated tokens)
            int kvCap;                        // allocated cache length

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
            }

            static int Div256(int n) => (n + 255) / 256;
            static void Grow(ref ComputeBuffer b, int n) { if (b != null && b.count >= n) return; b?.Release(); b = new ComputeBuffer(Math.Max(n, 1), 4, ComputeBufferType.Structured); }

            // ---------- generic ops (fp16 OR int8 weights via manifest Get) ----------
            // A '<name>.weight.scales' sibling in the manifest => the weight is q8 (int8 4-per-uint
            // + per-row fp16 scale): route to LinearBiasQ8. Chosen PER TENSOR (fp16 dirs have no
            // .scales, so this is a no-op there). All pocket q8 in_dims are % 4 == 0.
            void Linear(string name, ComputeBuffer x, ComputeBuffer y, int T, int inDim, int outDim, bool bias, int act = 0)
            {
                ComputeBuffer scales = w.Has(name + ".weight.scales") ? w.Get(name + ".weight.scales") : null;
                int k = scales != null ? kLinearQ8 : kLinear;
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                cs.SetBuffer(k, "X", x); cs.SetBuffer(k, "W", w.Get(name + ".weight"));
                cs.SetBuffer(k, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                if (scales != null) cs.SetBuffer(k, "W_scales", scales);
                cs.SetBuffer(k, "Y", y);
                cs.Dispatch(k, 1, (T + 7) / 8, (outDim + 31) / 32);
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
                    int k = scales != null ? kLinearQ8 : kLinear;
                    cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                    cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                    cs.SetBuffer(k, "X", x); cs.SetBuffer(k, "W", w.Get(name + ".weight"));
                    cs.SetBuffer(k, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                    if (scales != null) cs.SetBuffer(k, "W_scales", scales);
                    cs.SetBuffer(k, "Y", y);
                    cs.SetInt("elem_offset", r0);
                    cs.Dispatch(k, 1, (Math.Min(rows, T - r0) + 7) / 8, (outDim + 31) / 32);
                    cs.SetInt("elem_offset", 0);
                    if (r0 + rows < T) yield return null;
                }
            }

            void LayerNorm(ComputeBuffer gamma, ComputeBuffer beta, ComputeBuffer x, ComputeBuffer y, int T, int dim, float eps)
            {
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", dim); cs.SetFloat("norm_eps", eps);
                cs.SetBuffer(kLN, "norm_input", x); cs.SetBuffer(kLN, "norm_output", y);
                cs.SetBuffer(kLN, "ln_gamma", gamma); cs.SetBuffer(kLN, "ln_beta", beta);
                cs.Dispatch(kLN, Div256(T), 1, 1);
            }

            void Act(ComputeBuffer b, int n, int act) { cs.SetInt("buffer_size", n); cs.SetInt("activation_type", act); cs.SetFloat("leaky_slope", 0.01f); cs.SetBuffer(kAct, "inout_buf", b); cs.Dispatch(kAct, Div256(n), 1, 1); }
            void AddR(ComputeBuffer a, ComputeBuffer b, int n) { cs.SetInt("buffer_size", n); cs.SetBuffer(kAdd, "buf_a", a); cs.SetBuffer(kAdd, "buf_b", b); cs.Dispatch(kAdd, Div256(n), 1, 1); }
            void Copy(ComputeBuffer a, ComputeBuffer b, int n) { cs.SetInt("buffer_size", n); cs.SetBuffer(kCopy, "buf_a", a); cs.SetBuffer(kCopy, "buf_b", b); cs.Dispatch(kCopy, Div256(n), 1, 1); }

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
                    cs.Dispatch(kAttn, L, heads, 1);
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
                cs.Dispatch(kSliceCols, Div256(T * outDim), 1, 1);
            }
            void RoPE(ComputeBuffer b, int T, int heads, int hd, int posOffset = 0)
            {
                cs.SetInt("seq_len", T); cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd);
                cs.SetInt("pos_offset", posOffset); cs.SetFloat("rope_theta", Cfg.ROPE_THETA);
                cs.SetBuffer(kRope, "inout_buf", b); cs.Dispatch(kRope, Div256(T * heads * (hd / 2)), 1, 1);
            }

            // ================= P5: KV-cache incremental decode =================
            // KV decode ≡ the full causal forward of RunTransformer (unbounded causal + interleaved
            // RoPE at absolute positions), just amortized: each row's K/V is written once and reused.
            // Prefill populates the caches over the prompt; PrefillKV/DecodeStepKV replace the
            // full-forward loop in the AR generation path. RunTransformer stays untouched (P2 gate).

            static void GrowKV(ref ComputeBuffer b, int n) { if (b != null && b.count >= n) return; b?.Release(); b = new ComputeBuffer(Math.Max(n, 1), 4, ComputeBufferType.Structured); }

            void EnsureKV(int maxLen)
            {
                int dim = Cfg.DIM;
                if (kCache == null) { kCache = new ComputeBuffer[Cfg.TF_LAYERS]; vCache = new ComputeBuffer[Cfg.TF_LAYERS]; }
                if (kvCap < maxLen)
                {
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
                cs.Dispatch(kAppendKV, Div256(dim), 1, 1);
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
                    cs.Dispatch(kAttn, Lp, heads, 1);
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
                d1In.SetData(tokenEmb, 0, 0, dim);
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
                    cs.Dispatch(kAttnKV, heads, 1, 1);
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
                d1Out.GetData(c, 0, 0, Cfg.DIM);
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
                        req.GetData<float>().CopyTo(dst);
                        yield break;
                    }
                }
                buf.GetData(dst, 0, 0, count);   // unsupported/error fallback: sync
            }

            public void ResetKV() { kvLen = 0; }

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

            /// <summary>Synchronous form (probes/offline — P3/P4 parity + deterministic timing).</summary>
            public float[] FlowHead(float[] c, float[] noise, float s, float t)
            {
                FlowHeadIssue(c, noise, s, t);
                float[] vel = new float[Cfg.LDIM];
                fOut.GetData(vel, 0, 0, Cfg.LDIM);
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
                Grow(ref fNoiseIn, Cfg.LDIM); fNoiseIn.SetData(noise);
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
                cs.SetBuffer(cs.FindKernel("ScaleBuf"), "inout_buf", fy); cs.Dispatch(cs.FindKernel("ScaleBuf"), Div256(D), 1, 1);
                Grow(ref fCondIn, Cfg.DIM); fCondIn.SetData(c);
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
                cs.Dispatch(kRms, 1, 1, 1);
            }

            void Modulate(ComputeBuffer x, ComputeBuffer y, ComputeBuffer mod, int dim, int shiftOff, int scaleOff)
            {
                cs.SetInt("seq_len", 1); cs.SetInt("norm_dim", dim);
                cs.SetInt("mod_shift_off", shiftOff); cs.SetInt("mod_scale_off", scaleOff);
                cs.SetBuffer(kMod, "norm_input", x); cs.SetBuffer(kMod, "norm_output", y); cs.SetBuffer(kMod, "mod_vec", mod);
                cs.Dispatch(kMod, Div256(dim), 1, 1);
            }
            void GateAdd(ComputeBuffer a, ComputeBuffer h, ComputeBuffer mod, int dim, int gateOff)
            {
                cs.SetInt("seq_len", 1); cs.SetInt("norm_dim", dim); cs.SetInt("mod_gate_off", gateOff);
                cs.SetBuffer(kGate, "buf_a", a); cs.SetBuffer(kGate, "buf_b", h); cs.SetBuffer(kGate, "mod_vec", mod);
                cs.Dispatch(kGate, Div256(dim), 1, 1);
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
                tfIn?.Release(); tfNorm?.Release(); qkv?.Release(); q?.Release(); k?.Release(); v?.Release();
                attn?.Release(); ff?.Release(); tmp?.Release(); onesB?.Release(); zerosB?.Release();
                fx?.Release(); fy?.Release(); fh?.Release(); fmod?.Release(); ftmp?.Release(); ftime0?.Release(); ftime1?.Release();
                fOut?.Release(); fNoiseIn?.Release(); fCondIn?.Release();
                if (kCache != null) foreach (var b in kCache) b?.Release();
                if (vCache != null) foreach (var b in vCache) b?.Release();
                d1In?.Release(); d1Norm?.Release(); d1Qkv?.Release(); d1Q?.Release(); d1K?.Release(); d1V?.Release();
                d1Attn?.Release(); d1Ff?.Release(); d1Tmp?.Release(); d1Out?.Release();
            }
        }
    }
}
