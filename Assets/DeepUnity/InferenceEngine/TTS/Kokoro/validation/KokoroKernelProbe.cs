using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    namespace KokoroModeling
    {
        // Play-mode probe that grades the Kokoro GPU port (KokoroCS.compute via KokoroModel)
        // against the parity-verified fp32 oracle KokoroCPU. Two parts:
        //
        //   A. KERNEL GRADING (CHECKLIST kernel plan, build order 1..12): every kernel gets
        //      random [-2,2] inputs (seeded) pushed through the SHIPPED KokoroModel dispatch
        //      helper and through the exact KokoroCPU oracle method, with the REAL fp16 weights
        //      (both sides read the identical widened values, so the gate is pure kernel math):
        //      maxabs < 1e-3 per kernel. STFT angle rows are compared with wrap-around.
        //
        //   B. STAGE PROBE: full ForwardYielding on t0/t1/t2 with the reference dumps
        //      (validation/dump/*.npy, fortran_order honored) — G2P+ids exact, then corr gates
        //      per CHECKLIST B2 (bert_dur/d_en/d ≥.999, en/F0/N ≥.995, t_en/asr ≥.999,
        //      dec_x ≥.99, t0 wav ≥.99 with injected rand_ini+sine_noise; pred_dur exact or
        //      ±1 on ≤2 tokens, else overridden with the dump to keep grading downstream).
        //
        // Report: ProbeLogs/kokoro_kernel_report.md + ProbeLogs/kokoro_kernel.done (PASS/FAIL).
        // Run via KokoroKernelBatchRunner (Unity closed) or drop on a GameObject and play.
        public class KokoroKernelProbe : MonoBehaviour
        {
            public string weightsDir = "Assets/Resources/Weights/weights_kokoro_fp16";
            public string dumpDir = "Assets/DeepUnity/InferenceEngine/TTS/Kokoro/validation/dump";
            public string g2pPath = "Assets/DeepUnity/InferenceEngine/TTS/Kokoro/KokoroG2P";
            public string reportPath = "ProbeLogs/kokoro_kernel_report.md";
            public string doneMarker = "ProbeLogs/kokoro_kernel.done";
            [Tooltip("Bisect: false = legacy (pre-optimization) kernel routing. Serialized so it survives the play-mode domain reload.")]
            public bool fastKernels = true;

            readonly StringBuilder report = new StringBuilder();
            bool failed;
            KokoroWeights weights;
            KokoroTensors tensors;
            KokoroCPU cpu;
            KokoroModel model;
            KokoroG2P g2p;
            System.Random rng;
            readonly List<ComputeBuffer> scoped = new List<ComputeBuffer>();

            void Start()
            {
                KokoroModel.FastKernels = fastKernels;
                StartCoroutine(Run());
            }

            // ---------------- minimal .npy loader (v1/v2, little-endian, honors fortran_order) --
            static Array LoadNpy(string path, out int[] shape)
            {
                byte[] all = File.ReadAllBytes(path);
                if (all[0] != 0x93) throw new Exception($"not npy: {path}");
                int major = all[6];
                int headerLen = major >= 2 ? BitConverter.ToInt32(all, 8) : BitConverter.ToUInt16(all, 8);
                int dataStart = (major >= 2 ? 12 : 10) + headerLen;
                string header = Encoding.ASCII.GetString(all, major >= 2 ? 12 : 10, headerLen);
                bool fortran = header.Contains("'fortran_order': True");

                string shapeStr = header.Substring(header.IndexOf("'shape':", StringComparison.Ordinal) + 8);
                shapeStr = shapeStr.Substring(shapeStr.IndexOf('(') + 1);
                shapeStr = shapeStr.Substring(0, shapeStr.IndexOf(')'));
                var dims = new List<int>();
                foreach (string s in shapeStr.Split(','))
                    if (!string.IsNullOrWhiteSpace(s)) dims.Add(int.Parse(s.Trim()));
                if (dims.Count == 0) dims.Add(1);
                shape = dims.ToArray();
                long count = 1; foreach (int d in shape) count *= d;

                if (header.Contains("f4"))
                {
                    float[] r = new float[count];
                    Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4);
                    if (fortran && shape.Length >= 2)
                    {   // dumps are [1,A,B]; F-order flat = B-major -> convert to C-order
                        int A = shape[shape.Length - 2], B = shape[shape.Length - 1];
                        float[] c = new float[count];
                        for (int a = 0; a < A; a++)
                            for (int b2 = 0; b2 < B; b2++) c[a * B + b2] = r[b2 * A + a];
                        return c;
                    }
                    return r;
                }
                if (header.Contains("i8"))
                {
                    long[] r = new long[count];
                    Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 8);
                    return r;
                }
                if (header.Contains("i4"))
                {
                    int[] r = new int[count];
                    Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4);
                    return r;
                }
                throw new Exception($"unsupported npy dtype in {path}");
            }

            float[] F(string n) => (float[])LoadNpy(Path.Combine(dumpDir, n + ".npy"), out _);
            int[] I(string n)
            {
                Array a = LoadNpy(Path.Combine(dumpDir, n + ".npy"), out _);
                if (a is int[] i) return i;
                long[] l = (long[])a;
                int[] r = new int[l.Length];
                for (int j = 0; j < l.Length; j++) r[j] = (int)l[j];
                return r;
            }

            // ---------------- helpers ------------------------------------------------------------
            void Log(string line)
            {
                report.AppendLine(line);
                Debug.Log("[KokoroKernelProbe] " + line);
            }

            float[] Rand(int n, float lo = -2f, float hi = 2f)
            {
                var a = new float[n];
                for (int i = 0; i < n; i++) a[i] = lo + (hi - lo) * (float)rng.NextDouble();
                return a;
            }

            int[] RandIds(int n, int max)
            {
                var a = new int[n];
                for (int i = 0; i < n; i++) a[i] = rng.Next(max);
                return a;
            }

            ComputeBuffer Up(float[] a)
            {
                var b = new ComputeBuffer(a.Length, 4, ComputeBufferType.Structured);
                b.SetData(a);
                scoped.Add(b);
                return b;
            }

            ComputeBuffer Alloc(int n)
            {
                var b = new ComputeBuffer(n, 4, ComputeBufferType.Structured);
                scoped.Add(b);
                return b;
            }

            float[] Down(ComputeBuffer b, int n)
            {
                var a = new float[n];
                b.GetData(a, 0, 0, n);
                return a;
            }

            void ReleaseScoped()
            {
                foreach (var b in scoped) b.Release();
                scoped.Clear();
            }

            static float MaxAbs(float[] a, float[] b, int lo = 0, int hi = -1)
            {
                if (hi < 0) hi = Math.Min(a.Length, b.Length);
                float m = 0;
                for (int i = lo; i < hi; i++)
                {
                    float d = Math.Abs(a[i] - b[i]);
                    if (d > m) m = d;
                }
                return m;
            }

            static (float maxAbs, float corr) Diff(float[] a, float[] b)
            {
                int n = Math.Min(a.Length, b.Length);
                double ma = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                for (int i = 0; i < n; i++)
                {
                    double d = Math.Abs(a[i] - b[i]);
                    if (d > ma) ma = d;
                    sa += a[i]; sb += b[i]; saa += (double)a[i] * a[i];
                    sbb += (double)b[i] * b[i]; sab += (double)a[i] * b[i];
                }
                double cov = sab / n - (sa / n) * (sb / n);
                double va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n);
                return ((float)ma, (float)(cov / Math.Sqrt(Math.Max(va * vb, 1e-30))));
            }

            void GradeKernel(string name, float[] gpu, float[] cpuRef, float tol = 1e-3f)
            {
                if (gpu.Length != cpuRef.Length)
                {
                    failed = true;
                    Log($"| {name} | len {gpu.Length} vs {cpuRef.Length} | **FAIL** |");
                    return;
                }
                float m = MaxAbs(gpu, cpuRef);
                bool ok = m < tol;
                if (!ok) failed = true;
                Log($"| {name} | {m:E2} | {(ok ? "PASS" : "**FAIL**")} |");
            }

            void GradeStage(string name, float[] mine, float[] reference, double minCorr)
            {
                var (ma, corr) = Diff(mine, reference);
                bool ok = corr >= minCorr && mine.Length == reference.Length;
                if (!ok) failed = true;
                Log($"| {name} | {corr:F6} | {ma:E2} | {mine.Length}/{reference.Length} | {(ok ? "OK" : "**FAIL**")} |");
            }

            // ================================ main ================================
            IEnumerator Run()
            {
                Directory.CreateDirectory("ProbeLogs");
                Log($"# Kokoro kernel + stage report — {DateTime.Now:yyyy-MM-dd HH:mm}");
                Log("");
                Log($"weights: {weightsDir}");

                weights = new KokoroWeights(weightsDir);
                weights.BudgetBytesPerFrame = 512L * 1024 * 1024;   // probe: no hitch concerns
                tensors = new KokoroTensors(weightsDir);
                cpu = new KokoroCPU(tensors);
                model = new KokoroModel(weights, cpu);
                g2p = new KokoroG2P(g2pPath);
                while (!weights.IsReady || !g2p.IsReady) yield return null;
                Log("weights streamed to GPU.");
                Log("");

                // ---------------- A. kernel grading ----------------
                Log("## A. Kernel grading (random inputs, real fp16 weights, maxabs < 1e-3)");
                Log("");
                Log("| kernel | maxabs | verdict |");
                Log("|---|---|---|");
                rng = new System.Random(1234);
                try { KernelTests(); }
                finally { ReleaseScoped(); }
                Log("");
                yield return null;

                // ---------------- B. stage probe vs dumps ----------------
                for (int i = 0; i < 3; i++)
                {
                    var st = StageProbe(i);
                    while (st.MoveNext()) yield return st.Current;
                }

                model.Dispose();
                weights.Dispose();
                Finish(!failed);
            }

            // ================================ A. kernels ================================
            void KernelTests()
            {
                // ---- 1. LinearBias <- KokoroCPU.LinearT / StyleFc ----
                {
                    int T = 33;
                    float[] x = Rand(T * 768);
                    var xb = Up(x); var yb = Alloc(T * 768);
                    model.Linear("bert/layer/attn_q", xb, yb, T, 768, 768);
                    GradeKernel("1a LinearBias 768>768", Down(yb, T * 768),
                                cpu.LinearT(x, T, 768, "bert/layer/attn_q"));

                    float[] s = Rand(128);
                    var sb = Up(s); var gb = Alloc(1024);
                    model.StyleFc("pred/F0_0/norm1_fc", sb, gb);
                    GradeKernel("1b LinearBias styleFc 128>1024", Down(gb, 1024),
                                cpu.StyleFc(s, "pred/F0_0/norm1_fc"));
                }
                ReleaseScoped();

                // ---- 2. LayerNormAffine (row-major bert eps 1e-12 / channel-major tenc 1e-5) ----
                {
                    int T = 29;
                    float[] x = Rand(T * 768);
                    var xb = Up(x); var yb = Alloc(T * 768);
                    model.LayerNorm("bert/layer/ln", xb, yb, T, 768, 1e-12f, 768, 1);
                    float[] r = (float[])x.Clone();
                    KokoroCPU.LayerNorm(r, T, 768, tensors.D("bert/layer/ln.w"), tensors.D("bert/layer/ln.b"), 1e-12f);
                    GradeKernel("2a LayerNormAffine rows eps1e-12", Down(yb, T * 768), r);

                    int T2 = 41;
                    float[] xc = Rand(512 * T2);                       // [512, T] channel-major
                    var xcb = Up(xc); var ycb = Alloc(512 * T2);
                    model.LayerNorm("tenc/cnn0/ln", xcb, ycb, T2, 512, 1e-5f, 1, T2);
                    float[] rows = KokoroCPU.Transpose(xc, 512, T2);   // -> [T,512]
                    KokoroCPU.LayerNorm(rows, T2, 512, tensors.D("tenc/cnn0/ln.w"), tensors.D("tenc/cnn0/ln.b"), 1e-5f);
                    GradeKernel("2b LayerNormAffine chans eps1e-5", Down(ycb, 512 * T2),
                                KokoroCPU.Transpose(rows, T2, 512));
                }
                ReleaseScoped();

                // ---- 3. EmbedAlbert (word+pos+tok lookup) ----
                {
                    int T = 15;
                    int[] ids = RandIds(T, 178);
                    model.UploadIds(ids);
                    var yb = Alloc(T * 128);
                    model.EmbedAlbert(yb, T);
                    GradeKernel("3 EmbedAlbert", Down(yb, T * 128), cpu.EmbedAlbert(ids));
                }
                ReleaseScoped();

                // ---- 4. AttentionBi 12x64 ----
                {
                    int T = 50;
                    float[] q = Rand(T * 768), k = Rand(T * 768), v = Rand(T * 768);
                    var qb = Up(q); var kb = Up(k); var vb = Up(v); var ob = Alloc(T * 768);
                    model.AttentionBi(qb, kb, vb, ob, T, 12, 64);
                    GradeKernel("4 AttentionBi 12x64", Down(ob, T * 768),
                                KokoroCPU.AttentionBi(q, k, v, T, 12, 64));
                }
                ReleaseScoped();

                // ---- 5. GeluNew elementwise ----
                {
                    float[] x = Rand(4096, -4f, 4f);
                    var b = Up(x);
                    model.Activate(b, x.Length, 1);
                    float[] r = new float[x.Length];
                    for (int i = 0; i < x.Length; i++) r[i] = KokoroCPU.GeluNew(x[i]);
                    GradeKernel("5 GeluNew", Down(b, x.Length), r);
                }
                ReleaseScoped();

                // ---- 6. Conv1d generic (k5 / strided / dilated / k1-nobias) ----
                {
                    int T = 37;
                    float[] x = Rand(512 * T);
                    var xb = Up(x); var yb = Alloc(512 * T);
                    model.Conv("tenc/cnn0/conv", xb, yb, 512, T, 512, T, 5, 1, 2, 1);
                    (float[] r, _) = cpu.Conv1d(x, 512, T, "tenc/cnn0/conv", 1, 2, 1);
                    GradeKernel("6a Conv1D k5 p2", Down(yb, 512 * T), r);
                }
                ReleaseScoped();
                {
                    int T = 64, To = 32;
                    float[] x = Rand(T);
                    var xb = Up(x); var yb = Alloc(To);
                    model.Conv("dec/F0_conv", xb, yb, 1, T, 1, To, 3, 2, 1, 1);
                    (float[] r, _) = cpu.Conv1d(x, 1, T, "dec/F0_conv", 2, 1, 1);
                    GradeKernel("6b Conv1D k3 s2", Down(yb, To), r);
                }
                ReleaseScoped();
                {
                    int T = 40;
                    float[] x = Rand(256 * T);
                    var xb = Up(x); var yb = Alloc(256 * T);
                    model.Conv("dec/gen/rb2/c1_1", xb, yb, 256, T, 256, T, 11, 1, 15, 3);
                    (float[] r, _) = cpu.Conv1d(x, 256, T, "dec/gen/rb2/c1_1", 1, 15, 3);
                    GradeKernel("6c Conv1D k11 dil3", Down(yb, 256 * T), r);
                }
                ReleaseScoped();
                {
                    int T = 23;
                    float[] x = Rand(512 * T);
                    var xb = Up(x); var yb = Alloc(256 * T);
                    model.Conv("pred/F0_1/conv1x1", xb, yb, 512, T, 256, T, 1, 1, 0, 1, bias: false);
                    float[] w = tensors.D("pred/F0_1/conv1x1.w");      // [256,512,1], no bias
                    float[] r = new float[256 * T];
                    for (int co = 0; co < 256; co++)
                        for (int t = 0; t < T; t++)
                        {
                            float acc = 0;
                            for (int ci = 0; ci < 512; ci++) acc += w[co * 512 + ci] * x[ci * T + t];
                            r[co * T + t] = acc;
                        }
                    GradeKernel("6d Conv1D k1 no-bias", Down(yb, 256 * T), r);
                }
                ReleaseScoped();

                // ---- 7. InstanceNormStyle (= AdaIN: IN over T + (1+g)x+b from style fc) ----
                {
                    int C = 256, T = 77;
                    float[] x = Rand(C * T);
                    float[] s = Rand(128);
                    var xb = Up(x); var sb = Up(s); var gb = Alloc(2 * C);
                    model.StyleFc("pred/F0_2/norm1_fc", sb, gb);
                    model.InstanceNormStyle(xb, gb, C, T);
                    float[] r = (float[])x.Clone();
                    cpu.AdaIN(r, C, T, s, "pred/F0_2/norm1_fc");
                    GradeKernel("7 InstanceNormStyle (AdaIN)", Down(xb, C * T), r);
                }
                ReleaseScoped();

                // ---- 8. LeakyRelu slopes 0.2 / 0.1 / 0.01 ----
                foreach (float slope in new[] { 0.2f, 0.1f, 0.01f })
                {
                    float[] x = Rand(4096);
                    var b = Up(x);
                    model.Activate(b, x.Length, 2, slope);
                    float[] r = (float[])x.Clone();
                    KokoroCPU.LRelu(r, slope);
                    GradeKernel($"8 LeakyRelu {slope}", Down(b, x.Length), r);
                    ReleaseScoped();
                }

                // ---- 9. ConvTranspose1d (depthwise pool + full ups) ----
                {
                    int C = 512, T = 25, To = 2 * T;
                    float[] x = Rand(C * T);
                    var xb = Up(x); var yb = Alloc(C * To);
                    model.ConvT("pred/F0_1/pool", xb, yb, C, T, C, To, 3, 2, 1, C);
                    GradeKernel("9a ConvT depthwise pool", Down(yb, C * To),
                                cpu.ConvTranspose1d(x, C, T, "pred/F0_1/pool", 2, 1, 1, C));
                }
                ReleaseScoped();
                {
                    int T = 20, To = (T - 1) * 6 - 6 + 12;             // 120
                    float[] x = Rand(256 * T);
                    var xb = Up(x); var yb = Alloc(128 * To);
                    model.ConvT("dec/gen/ups1", xb, yb, 256, T, 128, To, 12, 6, 3, 1);
                    GradeKernel("9b ConvT ups k12 s6", Down(yb, 128 * To),
                                cpu.ConvTranspose1d(x, 256, T, "dec/gen/ups1", 6, 3, 0, 1));
                }
                ReleaseScoped();

                // ---- 10. GatherTime (nearest x2 + index gather) ----
                {
                    int C = 17, T = 30, To = 2 * T;
                    float[] x = Rand(C * T);
                    var xb = Up(x); var yb = Alloc(C * To);
                    model.GatherUp2(xb, yb, C, T, To);
                    float[] r = new float[C * To];
                    for (int c = 0; c < C; c++)
                        for (int t = 0; t < To; t++) r[c * To + t] = x[c * T + t / 2];
                    GradeKernel("10a GatherTime up2", Down(yb, C * To), r);

                    int Tg = 45;
                    int[] idx = RandIds(Tg, T);
                    uint[] idxU = new uint[Tg];
                    for (int t = 0; t < Tg; t++) idxU[t] = (uint)idx[t];
                    var ib = Alloc(Tg); ib.SetData(idxU);
                    var yg = Alloc(C * Tg);
                    model.GatherIdx(xb, yg, ib, C, T, Tg);
                    float[] rg = new float[C * Tg];
                    for (int c = 0; c < C; c++)
                        for (int t = 0; t < Tg; t++) rg[c * Tg + t] = x[c * T + idx[t]];
                    GradeKernel("10b GatherTime idx", Down(yg, C * Tg), rg);
                }
                ReleaseScoped();

                // ---- 11. Snake (per-channel alpha, exact 1/a) ----
                {
                    int C = 256, T = 50;
                    float[] x = Rand(C * T);
                    var b = Up(x);
                    model.SnakeAct("dec/gen/rb0/a1_0", b, C, T);
                    float[] r = (float[])x.Clone();
                    KokoroCPU.Snake(r, C, T, tensors.D("dec/gen/rb0/a1_0"));
                    GradeKernel("11 Snake", Down(b, C * T), r);
                }
                ReleaseScoped();

                // ---- 12. DFT20 STFT fwd (mag/angle) + iSTFT OLA ----
                {
                    int S = 2000, frames = S / 5 + 1;
                    float[] x = Rand(S, -1f, 1f);
                    var xb = Up(x); var yb = Alloc(22 * frames);
                    model.Stft(xb, yb, S, frames);
                    float[] gpu = Down(yb, 22 * frames);
                    float[] r = KokoroCPU.Stft(x, S, frames);
                    float magDiff = MaxAbs(gpu, r, 0, 11 * frames);
                    // Angle rows: wrap-aware AND magnitude-gated — atan2 is ill-conditioned where
                    // |X_bf| ~ 0 (fp32-vs-double noise in re/im flips the angle arbitrarily; the
                    // reference has the same arbitrariness vs torch there, and the stage gates
                    // cover the real-signal path). Grade only bins with oracle mag > 1e-2: the
                    // observed re/im noise (~4e-6) then bounds the angle error at ~4e-4 < 1e-3.
                    const float MAG_GATE = 1e-2f;
                    float angDiff = 0;
                    int angSkipped = 0;
                    for (int i = 11 * frames; i < 22 * frames; i++)
                    {
                        if (r[i - 11 * frames] <= MAG_GATE) { angSkipped++; continue; }
                        float d = Math.Abs(gpu[i] - r[i]);
                        d = Math.Min(d, 2f * Mathf.PI - d);
                        if (d > angDiff) angDiff = d;
                    }
                    bool ok = magDiff < 1e-3f && angDiff < 1e-3f;
                    if (!ok) failed = true;
                    Log($"| 12a Stft20 mag/angle | {magDiff:E2} / {angDiff:E2} " +
                        $"(angle mag-gated >{MAG_GATE}, skipped {angSkipped}/{11 * frames}) | " +
                        $"{(ok ? "PASS" : "**FAIL**")} |");

                    float[] spec = Rand(22 * frames);
                    var sb2 = Up(spec); var wb = Alloc(S);
                    model.Istft(sb2, wb, frames, S);
                    GradeKernel("12b Istft20", Down(wb, S), KokoroCPU.Istft(spec, frames));
                }
                ReleaseScoped();

                // ---- 13. composite blocks (grades whichever path KokoroModel.FastKernels
                //          selects: fused InstanceNormStats + Conv1DTile prologue when ON, the
                //          legacy unfused dispatch list when OFF — both must pass) ----
                {
                    int C = 256, T = 60;
                    float[] x = Rand(C * T);
                    float[] s = Rand(128);
                    var xb = Up(x); var sb = Up(s);
                    var t1 = Alloc(C * T); var t2 = Alloc(C * T);
                    var it = model.SnakeResBlockY("dec/gen/rb1", xb, C, T, sb, t1, t2);
                    while (it.MoveNext()) { }
                    float[] r = (float[])x.Clone();
                    cpu.SnakeResBlock(ref r, C, T, s, "dec/gen/rb1");
                    GradeKernel("13a SnakeResBlockY (fused)", Down(xb, C * T), r);
                }
                ReleaseScoped();
                {
                    int T = 25;                                        // dec/encode: 514 -> 1024
                    float[] x = Rand(514 * T);
                    float[] s = Rand(128);
                    var xb = Up(x); var sb = Up(s);
                    var t1 = Alloc(1024 * T); var t2 = Alloc(1024 * T); var t3 = Alloc(1024 * T);
                    var ob = Alloc(1024 * T);
                    var it = model.AdainBlockY("dec/encode", xb, ob, 514, T, false, sb, t1, t2, t3);
                    while (it.MoveNext()) { }
                    (float[] r, _, _) = cpu.AdainBlock(x, 514, T, s, "dec/encode", false);
                    GradeKernel("13b AdainBlockY 514>1024 (fused)", Down(ob, 1024 * T), r, 2e-3f);
                }
                ReleaseScoped();
                {
                    int T = 30;                                        // pred/F0_1: 512 -> 256 up x2
                    float[] x = Rand(512 * T);
                    float[] s = Rand(128);
                    var xb = Up(x); var sb = Up(s);
                    var t1 = Alloc(512 * 2 * T); var t2 = Alloc(512 * 2 * T); var t3 = Alloc(512 * 2 * T);
                    var ob = Alloc(256 * 2 * T);
                    var it = model.AdainBlockY("pred/F0_1", xb, ob, 512, T, true, sb, t1, t2, t3);
                    while (it.MoveNext()) { }
                    (float[] r, _, _) = cpu.AdainBlock(x, 512, T, s, "pred/F0_1", true);
                    GradeKernel("13c AdainBlockY up (pool+fused)", Down(ob, 256 * 2 * T), r, 2e-3f);
                }
                ReleaseScoped();

                // ---- extras: EmbedText + util kernels ----
                {
                    int T = 21;
                    int[] ids = RandIds(T, 178);
                    model.UploadIds(ids);
                    var yb = Alloc(512 * T);
                    model.EmbedText(yb, T);
                    float[] emb = tensors.D("tenc/embedding.w");
                    float[] r = new float[512 * T];
                    for (int t = 0; t < T; t++)
                        for (int c = 0; c < 512; c++) r[c * T + t] = emb[ids[t] * 512 + c];
                    GradeKernel("x1 EmbedText", Down(yb, 512 * T), r);
                }
                ReleaseScoped();
                {
                    int n = 1000;
                    float[] a = Rand(n), b = Rand(n);
                    var ab = Up(a); var bb = Up(b);
                    model.AddScaleOp(ab, bb, n, 0.70710678f);
                    float[] r = new float[n];
                    for (int i = 0; i < n; i++) r[i] = (a[i] + b[i]) * 0.70710678f;
                    GradeKernel("x2 AddScale rsqrt2", Down(ab, n), r);

                    var cb = Alloc(2 * n);
                    model.CopySliceOp(cb, n, bb, 0, n);
                    model.CopySliceOp(cb, 0, bb, 0, n);
                    float[] got = Down(cb, 2 * n);
                    float m = Math.Max(MaxAbs(got, b, 0, n), 0f);
                    for (int i = 0; i < n; i++) m = Math.Max(m, Math.Abs(got[n + i] - b[i]));
                    bool ok = m < 1e-6f;
                    if (!ok) failed = true;
                    Log($"| x3 CopySlice | {m:E2} | {(ok ? "PASS" : "**FAIL**")} |");
                }
                ReleaseScoped();
            }

            // ================================ B. stages ================================
            IEnumerator StageProbe(int i)
            {
                Log($"## B. Stage probe — t{i}");
                Log("");

                // stage A: G2P + ids exact (gate)
                string meta = File.ReadAllText(Path.Combine(dumpDir, $"t{i}_meta.json"));
                string text = System.Text.RegularExpressions.Regex
                    .Match(meta, "\"text\": \"(.*?)\",\n").Groups[1].Value.Replace("\\\"", "\"");
                string expectedPs = File.ReadAllText(Path.Combine(dumpDir, $"t{i}_phonemes.txt"));
                string ps = g2p.Phonemize(text);
                bool g2pOk = ps == expectedPs;
                int[] refIds = I($"t{i}_input_ids");
                int[] ourIds = cpu.PhonemesToIds(expectedPs);
                bool idsOk = ourIds.Length == refIds.Length;
                if (idsOk) for (int t = 0; t < ourIds.Length; t++) idsOk &= ourIds[t] == refIds[t];
                if (!g2pOk || !idsOk) failed = true;
                Log($"G2P: {(g2pOk ? "EXACT MATCH" : "**MISMATCH**")}; vocab ids: {(idsOk ? "EXACT MATCH" : "**MISMATCH**")}");

                float[] refS = F($"t{i}_ref_s");
                Func<int, float[]> u01, n01;
                if (i == 0)
                {   // inject the python noise -> wav directly comparable
                    u01 = n => F("t0_rand_ini");
                    n01 = n => F("t0_sine_noise");
                }
                else
                {   // fresh RNG (no reference noise dumped for t1/t2 — stages only)
                    var r2 = new System.Random(1234);
                    u01 = n => { var a = new float[n]; for (int j = 0; j < n; j++) a[j] = (float)r2.NextDouble(); return a; };
                    n01 = n =>
                    {
                        var a = new float[n];
                        for (int j = 0; j < n; j += 2)
                        {
                            double r1 = 1 - r2.NextDouble(), rr = r2.NextDouble();
                            double m = Math.Sqrt(-2 * Math.Log(r1));
                            a[j] = (float)(m * Math.Cos(2 * Math.PI * rr));
                            if (j + 1 < n) a[j + 1] = (float)(m * Math.Sin(2 * Math.PI * rr));
                        }
                        return a;
                    };
                }

                model.CaptureStages = true;
                model.InjectPredDur = null;
                var fwd = model.ForwardYielding(refIds, refS, 1f, u01, n01, _ => { });
                while (fwd.MoveNext()) yield return fwd.Current;
                var S = model.LastStages;
                if (S == null || S.wav == null)
                {
                    failed = true;
                    Log("**FORWARD FAILED** — no stages captured.");
                    yield break;
                }

                // pred_dur gate (D): exact or +-1 on <=2 tokens, else override with the dump
                int[] pdRef = I($"t{i}_pred_dur");
                int pdDiff = 0;
                for (int t = 0; t < S.predDur.Length; t++) if (S.predDur[t] != pdRef[t]) pdDiff++;
                Log($"pred_dur: {pdDiff} of {S.predDur.Length} tokens differ");
                if (pdDiff > 2) failed = true;
                if (pdDiff > 0)
                {
                    Log("(override with reference pred_dur, re-run for downstream stages)");
                    model.InjectPredDur = pdRef;
                    fwd = model.ForwardYielding(refIds, refS, 1f, u01, n01, _ => { });
                    while (fwd.MoveNext()) yield return fwd.Current;
                    S = model.LastStages;
                    model.InjectPredDur = null;
                }

                Log("");
                Log("| stage | corr | maxabs | len | verdict |");
                Log("|---|---|---|---|---|");
                GradeStage("bert_dur", S.bertDur, F($"t{i}_bert_dur"), 0.999);
                GradeStage("d_en", S.dEn, F($"t{i}_d_en"), 0.999);
                GradeStage("d", S.d, F($"t{i}_d"), 0.999);
                GradeStage("duration", S.duration, F($"t{i}_duration"), 0.999);
                GradeStage("en", S.en, F($"t{i}_en"), 0.995);
                GradeStage("F0_pred", S.F0, F($"t{i}_F0_pred"), 0.995);
                GradeStage("N_pred", S.N, F($"t{i}_N_pred"), 0.995);
                GradeStage("t_en", S.tEn, F($"t{i}_t_en"), 0.999);
                GradeStage("asr", S.asr, F($"t{i}_asr"), 0.999);
                if (i == 0)
                {
                    GradeStage("dec_x", S.decX, F("t0_dec_x"), 0.99);
                    GradeStage("wav", S.wav, F("t0_wav"), 0.99);
                    SaveWav("ProbeLogs/kokoro_gpu_t0.wav", S.wav, KokoroTTS.SAMPLE_RATE);
                    Log("");
                    Log("t0 GPU audio written to ProbeLogs/kokoro_gpu_t0.wav");
                }
                float audioSec = S.wav.Length / 24000f;
                Log($"[perf] bert {model.BertMs:F0} ms | predictor {model.PredictorMs:F0} ms " +
                    $"(cpu-lstm: pred {model.PredCpuMs:F0} ms, tenc {model.TencCpuMs:F0} ms) | " +
                    $"decoder {model.DecoderMs:F0} ms | generator {model.GeneratorMs:F0} ms " +
                    $"(nsf-wait {model.NsfWaitMs:F0} ms) | " +
                    $"end-to-end {model.EndToEndMs:F0} ms for {audioSec:F2}s audio -> " +
                    $"RTF {model.EndToEndMs / 1000f / audioSec:F3}");
                Log("");
            }

            static void SaveWav(string path, float[] samples, int sr)
            {
                using var fs = new FileStream(path, FileMode.Create);
                using var w = new BinaryWriter(fs);
                int byteLen = samples.Length * 2;
                w.Write(Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + byteLen);
                w.Write(Encoding.ASCII.GetBytes("WAVEfmt ")); w.Write(16);
                w.Write((short)1); w.Write((short)1); w.Write(sr); w.Write(sr * 2);
                w.Write((short)2); w.Write((short)16);
                w.Write(Encoding.ASCII.GetBytes("data")); w.Write(byteLen);
                foreach (float s in samples)
                    w.Write((short)Mathf.Clamp(Mathf.RoundToInt(s * 32767f), short.MinValue, short.MaxValue));
            }

            void Finish(bool ok)
            {
                Log("");
                Log(ok ? "## RESULT: PASS" : "## RESULT: FAIL (see stages above)");
                File.WriteAllText(reportPath, report.ToString());
                File.WriteAllText(doneMarker, ok ? "PASS" : "FAIL");
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
#endif
            }
        }
    }
}
