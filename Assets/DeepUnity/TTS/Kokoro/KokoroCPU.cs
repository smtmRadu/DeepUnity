using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics;

namespace DeepUnity
{
    namespace KokoroModeling
    {
        // Full fp32 CPU forward of Kokoro-82M (SPEC.md graph, tensor-for-tensor vs
        // validation/dump_reference.py). Pure C# (no UnityEngine) on purpose. Roles:
        //   1. VALIDATION ORACLE for KokoroCS.compute — every kernel and every stage of the GPU
        //      path (KokoroModel) is graded against this implementation (KokoroKernelProbe in
        //      Unity, validation/harness~ outside). It is NEVER a selectable runtime backend:
        //      the runtime is GPU-only.
        //   2. Runtime home of the CPU-side stages INSIDE the GPU pipeline (SPEC §9): the 6
        //      biLSTMs (DurationEncode/DurationHead/BiLstm), the duration round/alignment, and
        //      the NSF phase pipeline + RNG (NsfHar). These are microsecond-scale sequential
        //      chains where per-step GPU dispatch would dominate; KokoroModel calls them here.
        // Weights come from KokoroTensors (fp16 export widened to fp32 — so numeric drift vs the
        // python fp32 reference is the fp16 weight rounding only).
        public class KokoroCPU
        {
            // These stages run inside Task.Run WHILE the game renders — cap the fan-out so
            // Parallel.For can't grab every worker and starve Unity's main/render threads
            // mid-frame (observed as fps dips during synthesis even though the work is
            // off-thread). Half the cores keeps the LSTMs fast without contending.
            public static readonly System.Threading.Tasks.ParallelOptions ParOpts =
                new System.Threading.Tasks.ParallelOptions
                { MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount / 2) };

            readonly KokoroTensors W;
            public readonly Dictionary<char, int> Vocab;

            public KokoroCPU(KokoroTensors tensors)
            {
                W = tensors;
                Vocab = tensors.LoadVocab();
            }

            public class Stages
            {
                public int T, F;                       // tokens (incl. $ bounds), duration frames
                public float[] bertDur, dEn, d, duration, en, F0, N, tEn, asr, decX, wav;
                public int[] predDur;
            }

            public int[] PhonemesToIds(string ps)
            {
                var ids = new List<int>(ps.Length + 2) { 0 };
                foreach (char c in ps) if (Vocab.TryGetValue(c, out int id)) ids.Add(id);
                ids.Add(0);
                return ids.ToArray();
            }

            /// <summary>Full forward. refS = voicepack row [256]. randU01/randN01 supply the NSF
            /// randomness (parity probes inject the python dumps; production passes an RNG).</summary>
            public Stages Forward(int[] ids, float[] refS, float speed,
                                  Func<int, float[]> randU01, Func<int, float[]> randN01)
            {
                var S = new Stages { T = ids.Length };
                int T = ids.Length;
                float[] sd = new float[128], sp = new float[128];
                Array.Copy(refS, 0, sd, 0, 128);
                Array.Copy(refS, 128, sp, 0, 128);

                // ---------------- PLBERT (ALBERT, one shared layer ×12) ----------------
                S.bertDur = Bert(ids);                                        // [T,768]
                S.dEn = Transpose(LinearT(S.bertDur, T, 768, "benc"), T, 512); // [512,T]

                // ---------------- DurationEncoder + duration head ----------------
                S.d = DurationEncode(Transpose(S.dEn, 512, T), T, sp);        // [T,640]
                (S.duration, S.predDur) = DurationHead(S.d, T, speed);
                int F = 0;
                foreach (int pd in S.predDur) F += pd;
                S.F = F;

                return ForwardFromDurations(S, ids, sd, sp, randU01, randN01);
            }

            /// <summary>Continues the forward from S.predDur (parity probes may override the
            /// durations with the reference dump to keep downstream shapes comparable).</summary>
            public Stages ForwardFromDurations(Stages S, int[] ids, float[] sd, float[] sp,
                                               Func<int, float[]> randU01, Func<int, float[]> randN01)
            {
                int T = S.T, F = 0;
                foreach (int pd in S.predDur) F += pd;
                S.F = F;
                int[] frame2tok = new int[F];
                for (int t = 0, f = 0; t < T; t++)
                    for (int k = 0; k < S.predDur[t]; k++) frame2tok[f++] = t;

                // en = d^T @ aln  -> [640,F]
                S.en = new float[640 * F];
                for (int c = 0; c < 640; c++)
                    for (int f = 0; f < F; f++)
                        S.en[c * F + f] = S.d[frame2tok[f] * 640 + c];

                // ---------------- F0/N (shared biLSTM + AdainResBlk stacks) ----------------
                float[] sh = BiLstm(Transpose(S.en, 640, F), F, 640, "pred/shared");  // [F,512]
                float[] xf = Transpose(sh, F, 512);                                    // [512,F]
                (S.F0, _) = FoNPath(xf, F, sp, "F0");
                (S.N, _) = FoNPath(xf, F, sp, "N");

                // ---------------- TextEncoder ----------------
                float[] emb = W.D("tenc/embedding.w");                                 // [178,512]
                float[] te = new float[512 * T];
                for (int t = 0; t < T; t++)
                    for (int c = 0; c < 512; c++) te[c * T + t] = emb[ids[t] * 512 + c];
                for (int i = 0; i < 3; i++)
                {
                    (te, _) = Conv1d(te, 512, T, $"tenc/cnn{i}/conv", 1, 2, 1);
                    float[] tt = Transpose(te, 512, T);
                    LayerNorm(tt, T, 512, W.D($"tenc/cnn{i}/ln.w"), W.D($"tenc/cnn{i}/ln.b"), 1e-5f);
                    te = Transpose(tt, T, 512);
                    LRelu(te, 0.2f);
                }
                S.tEn = Transpose(BiLstm(Transpose(te, 512, T), T, 512, "tenc/lstm"), T, 512); // [512,T]
                S.asr = new float[512 * F];
                for (int c = 0; c < 512; c++)
                    for (int f = 0; f < F; f++) S.asr[c * F + f] = S.tEn[c * T + frame2tok[f]];

                // ---------------- Decoder trunk ----------------
                float[] f0c = StridedConv1(S.F0, 2 * F, "dec/F0_conv");                // [F]
                float[] nc = StridedConv1(S.N, 2 * F, "dec/N_conv");
                float[] xd = new float[514 * F];
                Array.Copy(S.asr, xd, 512 * F);
                Array.Copy(f0c, 0, xd, 512 * F, F);
                Array.Copy(nc, 0, xd, 513 * F, F);
                int C, L;
                (xd, C, L) = AdainBlock(xd, 514, F, sd, "dec/encode", false);          // [1024,F]
                (float[] asrRes, _) = Conv1d(S.asr, 512, F, "dec/asr_res", 1, 0, 1);   // [64,F]
                for (int b = 0; b < 4; b++)
                {
                    float[] cat = new float[1090 * L];
                    Array.Copy(xd, cat, C * L);
                    Array.Copy(asrRes, 0, cat, 1024 * L, 64 * L);
                    Array.Copy(f0c, 0, cat, 1088 * L, L);
                    Array.Copy(nc, 0, cat, 1089 * L, L);
                    (xd, C, L) = AdainBlock(cat, 1090, L, sd, $"dec/decode{b}", b == 3);
                }
                S.decX = xd;                                                           // [512,2F]

                // ---------------- Generator ----------------
                S.wav = Generator(xd, 2 * F, S.F0, sd, randU01, randN01);
                return S;
            }

            // ================= extracted CPU-side stages (GPU pipeline hosts) =================
            /// <summary>DurationEncoder (3×{biLSTM, AdaLayerNorm, cat-style}) on d_en rows
            /// [T,512] -> d [T,640]. CPU stage of the GPU pipeline (biLSTM home).</summary>
            public float[] DurationEncode(float[] dEnRows, int T, float[] sp)
            {
                float[] x = CatStyle(dEnRows, T, 512, sp);                    // [T,640]
                for (int i = 0; i < 3; i++)
                {
                    x = BiLstm(x, T, 640, $"pred/durenc/lstm{i}");            // [T,512]
                    AdaLayerNorm(x, T, 512, sp, $"pred/durenc/adaln{i}_fc");
                    x = CatStyle(x, T, 512, sp);                              // [T,640]
                }
                return x;
            }

            /// <summary>Duration head: biLSTM + dur_proj + sigmoid-sum/speed -> (pre-round
            /// durations [T], pred_dur [T] rounded/clamped). CPU stage of the GPU pipeline.</summary>
            public (float[] duration, int[] predDur) DurationHead(float[] d, int T, float speed)
            {
                float[] dx = BiLstm(d, T, 640, "pred/lstm");                  // [T,512]
                float[] durLogits = LinearT(dx, T, 512, "pred/dur_proj");     // [T,50]
                float[] duration = new float[T];
                int[] predDur = new int[T];
                for (int t = 0; t < T; t++)
                {
                    double sum = 0;
                    for (int k = 0; k < 50; k++) sum += 1.0 / (1.0 + Math.Exp(-durLogits[t * 50 + k]));
                    duration[t] = (float)(sum / speed);
                    predDur[t] = Math.Max(1, (int)Math.Round(duration[t], MidpointRounding.ToEven));
                }
                return (duration, predDur);
            }

            // ================= PLBERT =================
            /// <summary>ALBERT embedding lookups pre-LN: word+pos+tok -> [T,128]
            /// (EmbedAlbert kernel oracle).</summary>
            public float[] EmbedAlbert(int[] ids)
            {
                int T = ids.Length;
                float[] we = W.D("bert/emb/word.w"), pe = W.D("bert/emb/pos.w"), te = W.D("bert/emb/tok.w");
                float[] e = new float[T * 128];
                for (int t = 0; t < T; t++)
                    for (int c = 0; c < 128; c++)
                        e[t * 128 + c] = we[ids[t] * 128 + c] + pe[t * 128 + c] + te[c];
                return e;
            }

            /// <summary>Bidirectional MHA, no mask: q,k,v [T, H·Dh] head-major -> [T, H·Dh],
            /// softmax(QK^T/sqrt(Dh))V per head (AttentionBi kernel oracle).</summary>
            public static float[] AttentionBi(float[] q, float[] k, float[] v, int T, int H, int Dh)
            {
                int C = H * Dh;
                float rsqrtD = 1f / (float)Math.Sqrt(Dh);
                float[] att = new float[T * C];
                var scores = new float[T];
                for (int hd = 0; hd < H; hd++)
                {
                    int off = hd * Dh;
                    for (int i = 0; i < T; i++)
                    {
                        float max = float.NegativeInfinity;
                        for (int j = 0; j < T; j++)
                        {
                            float sdot = 0;
                            for (int c = 0; c < Dh; c++) sdot += q[i * C + off + c] * k[j * C + off + c];
                            scores[j] = sdot * rsqrtD;
                            if (scores[j] > max) max = scores[j];
                        }
                        float denom = 0;
                        for (int j = 0; j < T; j++) { scores[j] = (float)Math.Exp(scores[j] - max); denom += scores[j]; }
                        for (int c = 0; c < Dh; c++)
                        {
                            float acc = 0;
                            for (int j = 0; j < T; j++) acc += scores[j] * v[j * C + off + c];
                            att[i * C + off + c] = acc / denom;
                        }
                    }
                }
                return att;
            }

            float[] Bert(int[] ids)
            {
                int T = ids.Length;
                float[] e = EmbedAlbert(ids);
                LayerNorm(e, T, 128, W.D("bert/emb/ln.w"), W.D("bert/emb/ln.b"), 1e-12f);
                float[] h = LinearT(e, T, 128, "bert/map");                            // [T,768]

                for (int layer = 0; layer < 12; layer++)
                {
                    float[] q = LinearT(h, T, 768, "bert/layer/attn_q");
                    float[] k = LinearT(h, T, 768, "bert/layer/attn_k");
                    float[] v = LinearT(h, T, 768, "bert/layer/attn_v");
                    float[] att = AttentionBi(q, k, v, T, 12, 64);
                    float[] ao = LinearT(att, T, 768, "bert/layer/attn_o");
                    for (int i = 0; i < T * 768; i++) ao[i] += h[i];
                    LayerNorm(ao, T, 768, W.D("bert/layer/attn_ln.w"), W.D("bert/layer/attn_ln.b"), 1e-12f);
                    float[] ff = LinearT(ao, T, 768, "bert/layer/ffn");                // [T,2048]
                    for (int i = 0; i < ff.Length; i++) ff[i] = GeluNew(ff[i]);
                    float[] fo = LinearT(ff, T, 2048, "bert/layer/ffn_out");
                    for (int i = 0; i < T * 768; i++) fo[i] += ao[i];
                    LayerNorm(fo, T, 768, W.D("bert/layer/ln.w"), W.D("bert/layer/ln.b"), 1e-12f);
                    h = fo;
                }
                return h;
            }

            // ================= F0/N path =================
            (float[], int) FoNPath(float[] x, int F, float[] sp, string fam)
            {
                int C, L;
                float[] y;
                (y, C, L) = AdainBlock(x, 512, F, sp, $"pred/{fam}_0", false);
                (y, C, L) = AdainBlock(y, C, L, sp, $"pred/{fam}_1", true);            // ×2 time
                (y, C, L) = AdainBlock(y, C, L, sp, $"pred/{fam}_2", false);
                (float[] proj, _) = Conv1d(y, C, L, $"pred/{fam}_proj", 1, 0, 1);      // [1,2F]
                return (proj, L);
            }

            // ================= building blocks (public: kernel-probe oracles) =================
            public float[] LinearT(float[] x, int T, int I, string name)   // x [T,I] -> [T,O]
            {
                float[] w = W.D(name + ".w"), b = W.D(name + ".b");
                int O = w.Length / I;
                float[] y = new float[T * O];
                for (int t = 0; t < T; t++)
                    for (int o = 0; o < O; o++)
                    {
                        float acc = b[o];
                        int wo = o * I, xo = t * I;
                        for (int i = 0; i < I; i++) acc += w[wo + i] * x[xo + i];
                        y[t * O + o] = acc;
                    }
                return y;
            }

            /// <summary>torch bidirectional 1-layer LSTM, H=256/dir, gates i,f,g,o. x [T,I] -> [T,512].
            /// SIMD-PARALLELIZED but BIT-EXACT vs the original scalar loop (the LSTMs dominated
            /// PredCpuMs at ~1 s/utterance): input projections (bias + wih·x_t, ~70% of the
            /// FLOPs, no recurrence) precompute Parallel-over-t and the two directions run
            /// concurrently, with every GEMV vectorized ACROSS independent gate rows
            /// (System.Numerics Vector&lt;float&gt; lanes are separate outputs — each output's
            /// float op SEQUENCE stays bias-first + i-ascending mul-THEN-add, no FMA), so
            /// results are bit-identical to the pre-optimization oracle. Weight matrices are
            /// repacked once into a row-blocked i-major layout (PackLstm) and cached; the gate
            /// nonlinearities keep the exact scalar Math.Exp/Tanh calls.</summary>
            public float[] BiLstm(float[] x, int T, int I, string prefix)
            {
                float[] y = new float[T * 512];
                float[] preF = LstmInputProj(x, T, I, $"{prefix}/wih", $"{prefix}/bih", $"{prefix}/bhh");
                float[] preR = LstmInputProj(x, T, I, $"{prefix}/wih_r", $"{prefix}/bih_r", $"{prefix}/bhh_r");
                System.Threading.Tasks.Parallel.Invoke(
                    () => RunLstmDir(preF, T, $"{prefix}/whh", y, 0, false),
                    () => RunLstmDir(preR, T, $"{prefix}/whh_r", y, 256, true));
                return y;
            }

            // ---- SIMD GEMV plumbing (bit-identical: lanes = rows, per-row order unchanged) ----
            const int LSTM_G = 4 * 256;                       // gate rows per direction
            static readonly int VW = Vector<float>.Count;
            static readonly bool Simd = Vector.IsHardwareAccelerated && LSTM_G % (4 * VW) == 0;
            readonly ConcurrentDictionary<string, float[]> lstmPack = new ConcurrentDictionary<string, float[]>();

            // w [R,I] row-major -> blocked [(R/BR) blocks][I][BR] so the hot loop streams
            // BR consecutive rows per input i. ~44 MB extra across the 6 LSTMs, built lazily.
            float[] PackLstm(string name, int R, int I)
            {
                return lstmPack.GetOrAdd(name, _ =>
                {
                    float[] w = W.D(name);
                    int BR = 4 * VW;
                    float[] p = new float[R * I];
                    for (int b = 0; b < R / BR; b++)
                        for (int i = 0; i < I; i++)
                            for (int j = 0; j < BR; j++)
                                p[(b * I + i) * BR + j] = w[(b * BR + j) * I + i];
                    return p;
                });
            }

            // y[yOff+r] = init[initOff+r] + Σ_i wB[r,i]·x[xOff+i], r in [0,R). Vector lanes are
            // independent rows; per row: init first, i ascending, mul then add — the exact
            // original accumulation order (Vector * and + are per-lane IEEE mul/add, no fusion).
            static void GemvBlocked(float[] wB, int I, int R, float[] x, int xOff,
                                    float[] init, int initOff, float[] y, int yOff)
            {
                int vw = VW, BR = 4 * vw;
                for (int b = 0; b < R / BR; b++)
                {
                    int rb = b * BR;
                    var a0 = new Vector<float>(init, initOff + rb);
                    var a1 = new Vector<float>(init, initOff + rb + vw);
                    var a2 = new Vector<float>(init, initOff + rb + 2 * vw);
                    var a3 = new Vector<float>(init, initOff + rb + 3 * vw);
                    int wo = b * I * BR;
                    for (int i = 0; i < I; i++)
                    {
                        var xb = new Vector<float>(x[xOff + i]);
                        a0 += new Vector<float>(wB, wo) * xb;
                        a1 += new Vector<float>(wB, wo + vw) * xb;
                        a2 += new Vector<float>(wB, wo + 2 * vw) * xb;
                        a3 += new Vector<float>(wB, wo + 3 * vw) * xb;
                        wo += BR;
                    }
                    a0.CopyTo(y, yOff + rb);
                    a1.CopyTo(y, yOff + rb + vw);
                    a2.CopyTo(y, yOff + rb + 2 * vw);
                    a3.CopyTo(y, yOff + rb + 3 * vw);
                }
            }

            // scalar fallback (no HW SIMD): 4 row streams share each x[i] load; per-row order
            // identical to the original loop, on the native [R,I] layout.
            static void Gemv4(float[] w, int I, int R, float[] x, int xOff,
                              float[] init, int initOff, float[] y, int yOff)
            {
                for (int r = 0; r < R; r += 4)
                {
                    float a0 = init[initOff + r], a1 = init[initOff + r + 1],
                          a2 = init[initOff + r + 2], a3 = init[initOff + r + 3];
                    int w0 = r * I, w1 = w0 + I, w2 = w1 + I, w3 = w2 + I;
                    for (int i = 0; i < I; i++)
                    {
                        float xv = x[xOff + i];
                        a0 += w[w0 + i] * xv;
                        a1 += w[w1 + i] * xv;
                        a2 += w[w2 + i] * xv;
                        a3 += w[w3 + i] * xv;
                    }
                    y[yOff + r] = a0; y[yOff + r + 1] = a1;
                    y[yOff + r + 2] = a2; y[yOff + r + 3] = a3;
                }
            }

            // pre[t,r] = bih[r] + bhh[r] + wih[r,:]·x[t,:]  (exact original accumulation order;
            // the bias pre-sum is the same single addition the original did per row)
            float[] LstmInputProj(float[] x, int T, int I, string wihName, string bihName, string bhhName)
            {
                float[] bsum = lstmPack.GetOrAdd(wihName + "#bsum", _ =>
                {
                    float[] bih = W.D(bihName), bhh = W.D(bhhName);
                    float[] s = new float[LSTM_G];
                    for (int r = 0; r < LSTM_G; r++) s[r] = bih[r] + bhh[r];
                    return s;
                });
                float[] pre = new float[T * LSTM_G];
                if (Simd)
                {
                    float[] wB = PackLstm(wihName, LSTM_G, I);
                    System.Threading.Tasks.Parallel.For(0, T, ParOpts, t =>
                        GemvBlocked(wB, I, LSTM_G, x, t * I, bsum, 0, pre, t * LSTM_G));
                }
                else
                {
                    float[] wih = W.D(wihName);
                    System.Threading.Tasks.Parallel.For(0, T, ParOpts, t =>
                        Gemv4(wih, I, LSTM_G, x, t * I, bsum, 0, pre, t * LSTM_G));
                }
                return pre;
            }

            void RunLstmDir(float[] pre, int T, string whhName, float[] y, int yOff, bool reverse)
            {
                const int H = 256;
                float[] whhB = Simd ? PackLstm(whhName, LSTM_G, H) : W.D(whhName);
                float[] h = new float[H], c = new float[H], g = new float[LSTM_G];
                for (int step = 0; step < T; step++)
                {
                    int t = reverse ? T - 1 - step : step;
                    // g[r] = pre[t,r] + whh[r,:]·h — SIMD across rows replaces the old per-step
                    // Parallel.For(0,4) (same math, no fork/join per timestep)
                    if (Simd) GemvBlocked(whhB, H, LSTM_G, h, 0, pre, t * LSTM_G, g, 0);
                    else Gemv4(whhB, H, LSTM_G, h, 0, pre, t * LSTM_G, g, 0);
                    for (int i = 0; i < H; i++)
                    {
                        float ig = Sigmoid(g[i]), fg = Sigmoid(g[H + i]),
                              gg = (float)Math.Tanh(g[2 * H + i]), og = Sigmoid(g[3 * H + i]);
                        c[i] = fg * c[i] + ig * gg;
                        h[i] = og * (float)Math.Tanh(c[i]);
                        y[t * 512 + yOff + i] = h[i];
                    }
                }
            }

            public static void LayerNorm(float[] x, int T, int C, float[] g, float[] b, float eps)
            {
                for (int t = 0; t < T; t++)
                {
                    double mean = 0, var = 0;
                    int o = t * C;
                    for (int c = 0; c < C; c++) mean += x[o + c];
                    mean /= C;
                    for (int c = 0; c < C; c++) { double d = x[o + c] - mean; var += d * d; }
                    float rstd = (float)(1.0 / Math.Sqrt(var / C + eps));
                    for (int c = 0; c < C; c++) x[o + c] = (float)((x[o + c] - mean) * rstd) * g[c] + b[c];
                }
            }

            public void AdaLayerNorm(float[] x, int T, int C, float[] s, string fcName)
            {
                float[] h = StyleFc(s, fcName);                    // [2C]: gamma, beta
                for (int t = 0; t < T; t++)
                {
                    double mean = 0, var = 0;
                    int o = t * C;
                    for (int c = 0; c < C; c++) mean += x[o + c];
                    mean /= C;
                    for (int c = 0; c < C; c++) { double d = x[o + c] - mean; var += d * d; }
                    float rstd = (float)(1.0 / Math.Sqrt(var / C + 1e-5));
                    for (int c = 0; c < C; c++)
                        x[o + c] = (1f + h[c]) * (float)((x[o + c] - mean) * rstd) + h[C + c];
                }
            }

            public float[] StyleFc(float[] s, string name)
            {
                float[] w = W.D(name + ".w"), b = W.D(name + ".b");
                int O = b.Length;
                float[] h = new float[O];
                for (int o = 0; o < O; o++)
                {
                    float acc = b[o];
                    int wo = o * 128;
                    for (int i = 0; i < 128; i++) acc += w[wo + i] * s[i];
                    h[o] = acc;
                }
                return h;
            }

            /// <summary>AdaIN1d: per-channel InstanceNorm (identity affine, SPEC §12.1) + style.</summary>
            public void AdaIN(float[] x, int C, int T, float[] s, string fcName)
            {
                float[] h = StyleFc(s, fcName);
                for (int c = 0; c < C; c++)
                {
                    double mean = 0, var = 0;
                    int o = c * T;
                    for (int t = 0; t < T; t++) mean += x[o + t];
                    mean /= T;
                    for (int t = 0; t < T; t++) { double d = x[o + t] - mean; var += d * d; }
                    float rstd = (float)(1.0 / Math.Sqrt(var / T + 1e-5));
                    float gm = 1f + h[c], bt = h[C + c];
                    for (int t = 0; t < T; t++) x[o + t] = gm * (float)((x[o + t] - mean) * rstd) + bt;
                }
            }

            public (float[], int) Conv1d(float[] x, int Cin, int T, string name, int stride, int pad, int dil)
            {
                float[] w = W.D(name + ".w"), b = W.D(name + ".b");
                int K = W.Shape(name + ".w")[2], Cout = W.Shape(name + ".w")[0];
                int Tout = (T + 2 * pad - dil * (K - 1) - 1) / stride + 1;
                float[] y = new float[Cout * Tout];
                System.Threading.Tasks.Parallel.For(0, Cout, ParOpts, co =>
                {
                    for (int to = 0; to < Tout; to++)
                    {
                        float acc = b[co];
                        int start = to * stride - pad;
                        for (int ci = 0; ci < Cin; ci++)
                        {
                            int wo = (co * Cin + ci) * K, xo = ci * T;
                            for (int k = 0; k < K; k++)
                            {
                                int ti = start + k * dil;
                                if (ti >= 0 && ti < T) acc += w[wo + k] * x[xo + ti];
                            }
                        }
                        y[co * Tout + to] = acc;
                    }
                });
                return (y, Tout);
            }

            float[] StridedConv1(float[] x, int T, string name)   // Conv1d(1,1,k3,s2,p1) on [T]
            {
                float[] w = W.D(name + ".w"); float b = W.D(name + ".b")[0];
                int Tout = (T + 2 - 3) / 2 + 1;
                float[] y = new float[Tout];
                for (int to = 0; to < Tout; to++)
                {
                    float acc = b;
                    int start = to * 2 - 1;
                    for (int k = 0; k < 3; k++)
                    {
                        int ti = start + k;
                        if (ti >= 0 && ti < T) acc += w[k] * x[ti];
                    }
                    y[to] = acc;
                }
                return y;
            }

            /// <summary>torch ConvTranspose1d on [Cin,T] channel-major (weight [Cin, Cout/groups, K],
            /// bias [Cout]); Tout = (T-1)·stride - 2·pad + K + outPad. Covers both the AdainBlock
            /// depthwise pool (s2,p1,outPad1,groups=Cin) and the Generator ups (groups=1).
            /// (ConvTranspose1D kernel oracle.)</summary>
            public float[] ConvTranspose1d(float[] x, int Cin, int T, string name,
                                           int stride, int pad, int outPad, int groups)
            {
                float[] w = W.D(name + ".w"); float[] b = W.D(name + ".b");
                int K = W.Shape(name + ".w")[2];
                int coutPerG = W.Shape(name + ".w")[1];
                int cinPerG = Cin / groups;
                int Tout = (T - 1) * stride - 2 * pad + K + outPad;
                float[] y = new float[coutPerG * groups * Tout];
                System.Threading.Tasks.Parallel.For(0, coutPerG * groups, ParOpts, co =>
                {
                    int g = co / coutPerG, cog = co % coutPerG;
                    for (int to = 0; to < Tout; to++)
                    {
                        float acc = b[co];
                        // out[to] += w[k]·in[ti] where ti*stride - pad + k == to
                        for (int k = 0; k < K; k++)
                        {
                            int num = to + pad - k;
                            if (num < 0 || num % stride != 0) continue;
                            int ti = num / stride;
                            if (ti >= T) continue;
                            for (int icg = 0; icg < cinPerG; icg++)
                            {
                                int ci = g * cinPerG + icg;
                                acc += w[(ci * coutPerG + cog) * K + k] * x[ci * T + ti];
                            }
                        }
                        y[co * Tout + to] = acc;
                    }
                });
                return y;
            }

            /// <summary>AdainResBlk1d (SPEC §4.4): pre-act residual + nearest×2/conv1x1 shortcut.</summary>
            public (float[], int, int) AdainBlock(float[] x, int Cin, int T, float[] s, string p, bool up)
            {
                int Cout = W.Shape(p + "/conv1.w")[0];
                // residual
                float[] r = (float[])x.Clone();
                AdaIN(r, Cin, T, s, p + "/norm1_fc");
                LRelu(r, 0.2f);
                int Tr = T;
                if (up)
                {   // depthwise ConvT(k3,s2,p1,outpad1,groups=Cin): Tout = 2T
                    r = ConvTranspose1d(r, Cin, T, p + "/pool", 2, 1, 1, Cin);
                    Tr = 2 * T;
                }
                (r, Tr) = Conv1d(r, Cin, Tr, p + "/conv1", 1, 1, 1);      // -> [Cout,Tr]
                AdaIN(r, Cout, Tr, s, p + "/norm2_fc");
                LRelu(r, 0.2f);
                (r, Tr) = Conv1d(r, Cout, Tr, p + "/conv2", 1, 1, 1);
                // shortcut
                float[] sc = x;
                int Ts = T;
                if (up)
                {
                    float[] u = new float[Cin * 2 * T];
                    for (int c = 0; c < Cin; c++)
                        for (int t = 0; t < 2 * T; t++) u[c * 2 * T + t] = x[c * T + t / 2];
                    sc = u; Ts = 2 * T;
                }
                if (W.Has(p + "/conv1x1.w"))
                {
                    float[] w = W.D(p + "/conv1x1.w");                    // [Cout,Cin,1], no bias
                    float[] y1 = new float[Cout * Ts];
                    for (int co = 0; co < Cout; co++)
                        for (int t = 0; t < Ts; t++)
                        {
                            float acc = 0;
                            for (int ci = 0; ci < Cin; ci++) acc += w[co * Cin + ci] * sc[ci * Ts + t];
                            y1[co * Ts + t] = acc;
                        }
                    sc = y1;
                }
                const float RS2 = 0.70710678f;
                for (int i = 0; i < r.Length; i++) r[i] = (r[i] + sc[i]) * RS2;
                return (r, Cout, Tr);
            }

            /// <summary>Generator AdaINResBlock1 (Snake, dil 1/3/5) — in-place on x.</summary>
            public void SnakeResBlock(ref float[] x, int C, int T, float[] s, string p)
            {
                int K = W.Shape(p + "/c1_0.w")[2];
                int[] dil = { 1, 3, 5 };
                for (int j = 0; j < 3; j++)
                {
                    float[] xt = (float[])x.Clone();
                    AdaIN(xt, C, T, s, $"{p}/ada1_{j}_fc");
                    Snake(xt, C, T, W.D($"{p}/a1_{j}"));
                    (xt, _) = Conv1d(xt, C, T, $"{p}/c1_{j}", 1, (K * dil[j] - dil[j]) / 2, dil[j]);
                    AdaIN(xt, C, T, s, $"{p}/ada2_{j}_fc");
                    Snake(xt, C, T, W.D($"{p}/a2_{j}"));
                    (xt, _) = Conv1d(xt, C, T, $"{p}/c2_{j}", 1, (K - 1) / 2, 1);
                    for (int i = 0; i < x.Length; i++) x[i] += xt[i];
                }
            }

            public static void Snake(float[] x, int C, int T, float[] a)
            {
                for (int c = 0; c < C; c++)
                {
                    float al = a[c], inv = 1f / al;
                    int o = c * T;
                    for (int t = 0; t < T; t++)
                    {
                        float sn = (float)Math.Sin(al * x[o + t]);
                        x[o + t] += inv * sn * sn;
                    }
                }
            }

            public static void LRelu(float[] x, float slope)
            {
                for (int i = 0; i < x.Length; i++) if (x[i] < 0) x[i] *= slope;
            }

            static float Sigmoid(float v) => 1f / (1f + (float)Math.Exp(-v));
            public static float GeluNew(float v) =>
                0.5f * v * (1f + (float)Math.Tanh(0.7978845608f * (v + 0.044715f * v * v * v)));

            public static float[] Transpose(float[] x, int R, int C)   // [R,C] -> [C,R]
            {
                float[] y = new float[x.Length];
                for (int r = 0; r < R; r++)
                    for (int c = 0; c < C; c++) y[c * R + r] = x[r * C + c];
                return y;
            }

            public static float[] CatStyle(float[] x, int T, int C, float[] s)   // [T,C] -> [T,C+128]
            {
                float[] y = new float[T * (C + 128)];
                for (int t = 0; t < T; t++)
                {
                    Array.Copy(x, t * C, y, t * (C + 128), C);
                    Array.Copy(s, 0, y, t * (C + 128) + C, 128);
                }
                return y;
            }

            // torch F.interpolate(mode='linear', align_corners=False)
            static float[] InterpLinear(float[] x, int C, int Tin, int Tout)
            {
                float[] y = new float[C * Tout];
                double scale = (double)Tin / Tout;
                for (int t = 0; t < Tout; t++)
                {
                    double src = (t + 0.5) * scale - 0.5;
                    if (src < 0) src = 0;
                    int i0 = (int)src;
                    int i1 = Math.Min(i0 + 1, Tin - 1);
                    float w1 = (float)(src - i0);
                    for (int c = 0; c < C; c++)
                        y[c * Tout + t] = x[c * Tin + i0] * (1f - w1) + x[c * Tin + i1] * w1;
                }
                return y;
            }

            // ================= Generator (SPEC §7) =================
            /// <summary>NSF harmonic source (SPEC §7.1): F0 [T80] -> har [S = 300·T80]. CPU stage
            /// of the GPU pipeline (keeps the RNG on CPU: consumes randU01(9) then randN01(S·9) —
            /// parity probes inject the python dumps in that exact order).</summary>
            public float[] NsfHar(float[] F0, int T80, Func<int, float[]> randU01, Func<int, float[]> randN01)
            {
                int S = 300 * T80;                                       // samples @24k
                // NSF source: f0 nearest ×300, 9 harmonics
                float[] rad = new float[9 * S];                          // [h,S] (channel-major)
                for (int t = 0; t < S; t++)
                {
                    float f0 = F0[t / 300];
                    for (int h = 0; h < 9; h++) rad[h * S + t] = (f0 * (h + 1) / 24000f) % 1f;
                }
                float[] rini = randU01(9); rini[0] = 0;
                for (int h = 0; h < 9; h++) rad[h * S] += rini[h];       // first sample only
                int Td = S / 300;                                        // == T80
                float[] radDs = InterpLinear(rad, 9, S, Td);
                for (int h = 0; h < 9; h++)                              // cumsum ×2π then ×300
                {
                    double acc = 0;
                    for (int t = 0; t < Td; t++)
                    {
                        acc += radDs[h * Td + t];
                        radDs[h * Td + t] = (float)(acc * 2 * Math.PI) * 300f;
                    }
                }
                float[] phase = InterpLinear(radDs, 9, Td, S);
                float[] noise = randN01(S * 9);                          // [S,9] python order
                float[] har = new float[S];
                float[] lw = W.D("dec/gen/nsf_linear.w"); float lb = W.D("dec/gen/nsf_linear.b")[0];
                for (int t = 0; t < S; t++)
                {
                    float f0 = F0[t / 300];
                    float uv = f0 > 10f ? 1f : 0f;
                    float namp = uv * 0.003f + (1f - uv) * 0.1f / 3f;
                    float acc = lb;
                    for (int h = 0; h < 9; h++)
                    {
                        float sine = 0.1f * (float)Math.Sin(phase[h * S + t]);
                        acc += lw[h] * (sine * uv + namp * noise[t * 9 + h]);
                    }
                    har[t] = (float)Math.Tanh(acc);
                }
                return har;
            }

            float[] Generator(float[] x, int T80, float[] F0, float[] sd,
                              Func<int, float[]> randU01, Func<int, float[]> randN01)
            {
                x = (float[])x.Clone();   // trunk LRelu is in-place; never mutate caller's dec_x
                int S = 300 * T80;                                       // samples @24k
                float[] har = NsfHar(F0, T80, randU01, randN01);

                // STFT(har): n_fft 20, hop 5, hann periodic, center reflect -> [22, S/5+1]
                int frames = S / 5 + 1;
                float[] harCat = Stft(har, S, frames);

                // trunk
                int C = 512, L = T80;
                int[] upsK = { 20, 12 }, upsS = { 10, 6 }, upsP = { 5, 3 };
                for (int u = 0; u < 2; u++)
                {
                    LRelu(x, 0.1f);
                    // x_source
                    (float[] xs, int Ls) = u == 0 ? Conv1d(harCat, 22, frames, "dec/gen/noise_conv0", 6, 3, 1)
                                                  : Conv1d(harCat, 22, frames, "dec/gen/noise_conv1", 1, 0, 1);
                    int Cs = u == 0 ? 256 : 128;
                    SnakeResBlock(ref xs, Cs, Ls, sd, $"dec/gen/noise_res{u}");
                    // ups (ConvTranspose1d [Cin,Cout,K])
                    int Cout = C / 2, K = upsK[u], st = upsS[u], pd = upsP[u];
                    int Lout = (L - 1) * st - 2 * pd + K;
                    x = ConvTranspose1d(x, C, L, $"dec/gen/ups{u}", st, pd, 0, 1);
                    C = Cout; L = Lout;
                    if (u == 1)
                    {   // reflection_pad (1,0): prepend x[:,1]
                        float[] rp = new float[C * (L + 1)];
                        for (int c = 0; c < C; c++)
                        {
                            rp[c * (L + 1)] = x[c * L + 1];
                            Array.Copy(x, c * L, rp, c * (L + 1) + 1, L);
                        }
                        x = rp; L = L + 1;
                    }
                    for (int i = 0; i < C * L; i++) x[i] += xs[i];
                    // 3 resblocks mean
                    float[] sum = null;
                    for (int j = 0; j < 3; j++)
                    {
                        float[] rb = (float[])x.Clone();
                        SnakeResBlock(ref rb, C, L, sd, $"dec/gen/rb{u * 3 + j}");
                        if (sum == null) sum = rb;
                        else for (int i = 0; i < sum.Length; i++) sum[i] += rb[i];
                    }
                    for (int i = 0; i < sum.Length; i++) sum[i] /= 3f;
                    x = sum;
                }
                LRelu(x, 0.01f);                                          // DEFAULT slope here!
                (x, L) = Conv1d(x, C, L, "dec/gen/conv_post", 1, 3, 1);   // [22,L]
                // spec/phase -> iSTFT
                return Istft(x, L);
            }

            static readonly float[] Hann20 = BuildHann();
            static float[] BuildHann()
            {
                var w = new float[20];
                for (int n = 0; n < 20; n++) w[n] = 0.5f * (1f - (float)Math.Cos(2 * Math.PI * n / 20));
                return w;
            }

            public static float[] Stft(float[] x, int S, int frames)   // -> [22, frames]: mag(11)+angle(11)
            {
                int P = S + 20;                                  // reflect pad 10 each side
                float[] xp = new float[P];
                for (int i = 0; i < P; i++)
                {
                    int t = i - 10;
                    if (t < 0) t = -t;
                    else if (t >= S) t = 2 * S - 2 - t;
                    xp[i] = x[t];
                }
                float[] y = new float[22 * frames];
                for (int f = 0; f < frames; f++)
                {
                    int start = f * 5;
                    for (int b = 0; b <= 10; b++)
                    {
                        double re = 0, im = 0;
                        for (int n = 0; n < 20; n++)
                        {
                            double v = Hann20[n] * xp[start + n];
                            double ang = 2 * Math.PI * b * n / 20;
                            re += v * Math.Cos(ang);
                            im -= v * Math.Sin(ang);
                        }
                        y[b * frames + f] = (float)Math.Sqrt(re * re + im * im);
                        y[(11 + b) * frames + f] = (float)Math.Atan2(im, re);
                    }
                }
                return y;
            }

            public static float[] Istft(float[] x, int frames)          // x [22,frames]: log-mag(11)+sin-phase(11)
            {
                int S = (frames - 1) * 5;
                float[] outw = new float[S + 20];                // padded OLA buffer
                float[] wsum = new float[S + 20];
                for (int f = 0; f < frames; f++)
                {
                    int start = f * 5;
                    for (int n = 0; n < 20; n++)
                    {
                        double acc = 0;
                        for (int b = 0; b <= 10; b++)
                        {
                            double mag = Math.Exp(Math.Min(x[b * frames + f], 100f));
                            double ph = Math.Sin(x[(11 + b) * frames + f]);
                            double re = mag * Math.Cos(ph), im = mag * Math.Sin(ph);
                            double ang = 2 * Math.PI * b * n / 20;
                            double term = re * Math.Cos(ang) - im * Math.Sin(ang);
                            acc += (b == 0 || b == 10) ? term : 2 * term;
                        }
                        float v = (float)(acc / 20.0) * Hann20[n];
                        outw[start + n] += v;
                        wsum[start + n] += Hann20[n] * Hann20[n];
                    }
                }
                float[] wav = new float[S];
                for (int i = 0; i < S; i++)
                    wav[i] = outw[i + 10] / Math.Max(wsum[i + 10], 1e-11f);
                return wav;
            }
        }
    }
}
