using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace DeepUnity
{
    namespace ParakeetModeling
    {
        // Pure-C# fp32 Parakeet-TDT pipeline (SPEC.md §1-§7). No UnityEngine — shared by the
        // dotnet parity harness (validation/harness/) and the Unity runtime:
        //   - HARNESS: full chain (mel -> subsampling -> 24 conformer blocks -> enc_proj ->
        //     TDT greedy) graded stage-by-stage against validation/reference_dumps/*.
        //   - UNITY (ParakeetSTT): Mel() runs on the CPU (trivial cost, radix-2 FFT), the encoder
        //     runs on the GPU (ParakeetCS.compute mirrors EncoderLayer verbatim), and Decode()
        //     (LSTM 2x640 + joint head) runs here on the CPU — zero GPU dispatches per step.
        // All activation buffers are [T, C] row-major (t*C + c), matching the shader convention.
        public class ParakeetCPU
        {
            readonly ParakeetTensors W;
            readonly int V, blank;

            public ParakeetCPU(ParakeetTensors tensors, ParakeetVariant variant)
            {
                W = tensors;
                V = ParakeetConfig.Vocab(variant);
                blank = ParakeetConfig.Blank(variant);
            }

            // ==================================================================== §1 mel frontend
            /// <summary>16 kHz mono clip -> per-feature-normalized log-mel [tMel, 128].</summary>
            public float[] Mel(float[] samples, out int tMel)
            {
                int n = samples.Length, nfft = ParakeetConfig.NFft, hop = ParakeetConfig.HopLength;
                tMel = ParakeetConfig.MelFrames(n);
                if (tMel <= 0) { tMel = 0; return Array.Empty<float>(); }

                // pre-emphasis, then center pad (256 zeros both sides; constant/zero pad — SPEC §1)
                float[] pad = new float[n + nfft];
                pad[nfft / 2] = samples[0];
                for (int t = 1; t < n; t++)
                    pad[nfft / 2 + t] = samples[t] - ParakeetConfig.Preemphasis * samples[t - 1];

                // symmetric hann(400) zero-embedded centered in 512
                float[] win = new float[nfft];
                int off = (nfft - ParakeetConfig.WinLength) / 2;                 // 56
                for (int i = 0; i < ParakeetConfig.WinLength; i++)
                    win[off + i] = 0.5f - 0.5f * (float)Math.Cos(2.0 * Math.PI * i / (ParakeetConfig.WinLength - 1));

                float[] filters = W.D("frontend/mel_filters");                   // [128, 257]
                int bins = nfft / 2 + 1;
                int nm = ParakeetConfig.NMels;
                float[] mel = new float[tMel * nm];
                int frames = tMel;

                // twiddle tables (double precision trig, float storage)
                float[] cosT = new float[nfft / 2], sinT = new float[nfft / 2];
                for (int i = 0; i < nfft / 2; i++)
                {
                    cosT[i] = (float)Math.Cos(-2.0 * Math.PI * i / nfft);
                    sinT[i] = (float)Math.Sin(-2.0 * Math.PI * i / nfft);
                }

                Parallel.For(0, frames, t =>
                {
                    float[] re = new float[nfft], im = new float[nfft];
                    int s0 = t * hop;
                    for (int i = 0; i < nfft; i++) re[i] = pad[s0 + i] * win[i];
                    Fft(re, im, cosT, sinT);
                    for (int m = 0; m < nm; m++)
                    {
                        float acc = 0f;
                        int fb = m * bins;
                        for (int b = 0; b < bins; b++)
                            acc += filters[fb + b] * (re[b] * re[b] + im[b] * im[b]);
                        mel[t * nm + m] = (float)Math.Log(acc + ParakeetConfig.LogGuard);
                    }
                });

                // per-feature normalization over the utterance (unbiased std, +1e-5)
                for (int m = 0; m < nm; m++)
                {
                    double sum = 0;
                    for (int t = 0; t < tMel; t++) sum += mel[t * nm + m];
                    double mean = sum / tMel;
                    double var = 0;
                    for (int t = 0; t < tMel; t++) { double d = mel[t * nm + m] - mean; var += d * d; }
                    float std = (float)Math.Sqrt(var / (tMel - 1));
                    float inv = 1f / (std + ParakeetConfig.NormEps);
                    for (int t = 0; t < tMel; t++)
                        mel[t * nm + m] = (mel[t * nm + m] - (float)mean) * inv;
                }
                return mel;
            }

            /// <summary>In-place iterative radix-2 FFT (n = 512, unnormalized, torch-compatible).</summary>
            static void Fft(float[] re, float[] im, float[] cosT, float[] sinT)
            {
                int n = re.Length;
                for (int i = 1, j = 0; i < n; i++)          // bit reversal
                {
                    int bit = n >> 1;
                    for (; (j & bit) != 0; bit >>= 1) j ^= bit;
                    j |= bit;
                    if (i < j) { (re[i], re[j]) = (re[j], re[i]); (im[i], im[j]) = (im[j], im[i]); }
                }
                for (int len = 2; len <= n; len <<= 1)
                {
                    int half = len >> 1, step = n / len;
                    for (int i = 0; i < n; i += len)
                        for (int k = 0; k < half; k++)
                        {
                            float wr = cosT[k * step], wi = sinT[k * step];
                            int a = i + k, b = i + k + half;
                            float xr = re[b] * wr - im[b] * wi;
                            float xi = re[b] * wi + im[b] * wr;
                            re[b] = re[a] - xr; im[b] = im[a] - xi;
                            re[a] += xr; im[a] += xi;
                        }
                }
            }

            // ==================================================================== §2 subsampling
            /// <summary>Normalized mel [tMel,128] -> [tEnc,1024] (dw-striding conv x8 + linear).</summary>
            public float[] Subsample(float[] mel, int tMel, out int tEnc)
            {
                int C = ParakeetConfig.SubChannels;
                // stage 0: full conv 1->256
                float[] x = Conv2dStride2(mel, 1, tMel, ParakeetConfig.NMels,
                                          W.D("sub/conv0.w"), W.D("sub/conv0.b"), C, depthwise: false, relu: true,
                                          out int t1, out int f1);
                // stages 1..2: depthwise + pointwise
                for (int s = 1; s <= 2; s++)
                {
                    x = Conv2dStride2(x, C, t1, f1, W.D($"sub/conv{s}_dw.w"), W.D($"sub/conv{s}_dw.b"),
                                      C, depthwise: true, relu: false, out t1, out f1);
                    x = Pointwise2d(x, C, t1, f1, W.D($"sub/conv{s}_pw.w"), W.D($"sub/conv{s}_pw.b"), relu: true);
                }
                tEnc = t1;                                   // f1 == 16
                // flatten channel-major [t, c*f1+f] then linear 4096 -> 1024
                int flat = C * f1;
                float[] xf = new float[tEnc * flat];
                for (int c = 0; c < C; c++)
                    for (int t = 0; t < tEnc; t++)
                        for (int f = 0; f < f1; f++)
                            xf[t * flat + c * f1 + f] = x[(c * t1 + t) * f1 + f];
                return Linear(xf, tEnc, flat, W.D("sub/linear.w"), W.D("sub/linear.b"), ParakeetConfig.Dim);
            }

            /// <summary>3x3 stride-2 pad-1 conv over a [Cin, T, F] stack (Cin=1 reads mel [T,F]).
            /// depthwise: one 3x3 filter per channel (Cout == Cin).</summary>
            static float[] Conv2dStride2(float[] x, int cin, int tin, int fin,
                                         float[] w, float[] b, int cout, bool depthwise, bool relu,
                                         out int tout, out int fout)
            {
                int to = (tin - 1) / 2 + 1, fo = (fin - 1) / 2 + 1;
                tout = to; fout = fo;
                float[] y = new float[cout * to * fo];
                int tinL = tin, finL = fin, cinL = cin;
                Parallel.For(0, cout, oc =>
                {
                    for (int t = 0; t < to; t++)
                        for (int f = 0; f < fo; f++)
                        {
                            float acc = b[oc];
                            int nIn = depthwise ? 1 : cinL;
                            for (int ic = 0; ic < nIn; ic++)
                            {
                                int srcC = depthwise ? oc : ic;
                                int wBase = (oc * nIn + ic) * 9;
                                for (int kt = 0; kt < 3; kt++)
                                {
                                    int ti = t * 2 + kt - 1;
                                    if (ti < 0 || ti >= tinL) continue;
                                    for (int kf = 0; kf < 3; kf++)
                                    {
                                        int fi = f * 2 + kf - 1;
                                        if (fi < 0 || fi >= finL) continue;
                                        acc += w[wBase + kt * 3 + kf] * x[(srcC * tinL + ti) * finL + fi];
                                    }
                                }
                            }
                            y[(oc * to + t) * fo + f] = relu && acc < 0f ? 0f : acc;
                        }
                });
                return y;
            }

            static float[] Pointwise2d(float[] x, int c, int t, int f, float[] w, float[] b, bool relu)
            {
                float[] y = new float[c * t * f];
                int tf = t * f;
                Parallel.For(0, c, oc =>
                {
                    int wBase = oc * c;
                    for (int i = 0; i < tf; i++)
                    {
                        float acc = b[oc];
                        for (int ic = 0; ic < c; ic++)
                            acc += w[wBase + ic] * x[ic * tf + i];
                        y[oc * tf + i] = relu && acc < 0f ? 0f : acc;
                    }
                });
                return y;
            }

            // ==================================================================== §3 rel-pos emb
            /// <summary>[2*tEnc-1, 1024]: index 0 = relative distance +(T-1) ... last = -(T-1),
            /// interleaved [sin, cos] per frequency (SPEC §3).</summary>
            public float[] PosEmb(int tEnc)
            {
                int d = ParakeetConfig.Dim, half = d / 2, len = 2 * tEnc - 1;
                float[] pe = new float[len * d];
                for (int idx = 0; idx < len; idx++)
                {
                    double p = (tEnc - 1) - idx;
                    for (int i = 0; i < half; i++)
                    {
                        double freq = p * Math.Exp(-Math.Log(10000.0) * (2.0 * i) / d);
                        pe[idx * d + 2 * i] = (float)Math.Sin(freq);
                        pe[idx * d + 2 * i + 1] = (float)Math.Cos(freq);
                    }
                }
                return pe;
            }

            // ==================================================================== §4 encoder
            /// <summary>Full 24-block encoder. layer0Out (grading tap) = block 0 output copy.</summary>
            public float[] Encoder(float[] x, int tEnc, out float[] layer0Out)
            {
                float[] pos = PosEmb(tEnc);
                layer0Out = null;
                for (int l = 0; l < ParakeetConfig.Layers; l++)
                {
                    x = EncoderLayer(x, tEnc, pos, l);
                    if (l == 0) layer0Out = (float[])x.Clone();
                }
                return x;
            }

            public float[] EncProj(float[] encOut, int tEnc)
                => Linear(encOut, tEnc, ParakeetConfig.Dim,
                          W.D("joint/enc_proj.w"), W.D("joint/enc_proj.b"), ParakeetConfig.PredDim);

            float[] EncoderLayer(float[] x, int T, float[] pos, int l)
            {
                int D = ParakeetConfig.Dim, F = ParakeetConfig.FfnDim;
                string p = $"layer_{l}/";

                // ---- FF1 (half residual)
                float[] h = LayerNorm(x, T, D, W.D(p + "ff1.ln.w"), W.D(p + "ff1.ln.b"));
                h = Linear(h, T, D, W.D(p + "ff1.lin1.w"), null, F); Silu(h);
                h = Linear(h, T, F, W.D(p + "ff1.lin2.w"), null, D);
                for (int i = 0; i < x.Length; i++) x[i] += 0.5f * h[i];

                // ---- rel-pos MHSA
                float[] n = LayerNorm(x, T, D, W.D(p + "attn.ln.w"), W.D(p + "attn.ln.b"));
                float[] q = Linear(n, T, D, W.D(p + "attn.q.w"), null, D);
                float[] k = Linear(n, T, D, W.D(p + "attn.k.w"), null, D);
                float[] v = Linear(n, T, D, W.D(p + "attn.v.w"), null, D);
                float[] P = Linear(pos, 2 * T - 1, D, W.D(p + "attn.pos.w"), null, D);
                float[] att = RelPosAttention(q, k, v, P, T, W.D(p + "attn.bias_u"), W.D(p + "attn.bias_v"));
                att = Linear(att, T, D, W.D(p + "attn.o.w"), null, D);
                for (int i = 0; i < x.Length; i++) x[i] += att[i];

                // ---- conv module (GLU -> depthwise k9 -> folded BN -> SiLU -> pointwise)
                n = LayerNorm(x, T, D, W.D(p + "conv.ln.w"), W.D(p + "conv.ln.b"));
                float[] g2 = Linear(n, T, D, W.D(p + "conv.pw1.w"), null, 2 * D);
                float[] g = new float[T * D];
                for (int t = 0; t < T; t++)
                    for (int c = 0; c < D; c++)
                    {
                        float a = g2[t * 2 * D + c], b2 = g2[t * 2 * D + D + c];
                        g[t * D + c] = a / (1f + (float)Math.Exp(-b2)) * 1f;   // a * sigmoid(b)
                    }
                float[] dw = W.D(p + "conv.dw.w");                              // [1024, 9]
                float[] bnS = W.D(p + "conv.bn.scale"), bnB = W.D(p + "conv.bn.shift");
                float[] cvo = new float[T * D];
                int K = ParakeetConfig.ConvKernel, padL = K / 2;
                Parallel.For(0, D, c =>
                {
                    for (int t = 0; t < T; t++)
                    {
                        float acc = 0f;
                        for (int kk = 0; kk < K; kk++)
                        {
                            int ti = t + kk - padL;
                            if (ti < 0 || ti >= T) continue;
                            acc += dw[c * K + kk] * g[ti * D + c];
                        }
                        acc = acc * bnS[c] + bnB[c];                            // folded BatchNorm
                        cvo[t * D + c] = acc / (1f + (float)Math.Exp(-acc));    // SiLU
                    }
                });
                cvo = Linear(cvo, T, D, W.D(p + "conv.pw2.w"), null, D);
                for (int i = 0; i < x.Length; i++) x[i] += cvo[i];

                // ---- FF2 (half residual)
                h = LayerNorm(x, T, D, W.D(p + "ff2.ln.w"), W.D(p + "ff2.ln.b"));
                h = Linear(h, T, D, W.D(p + "ff2.lin1.w"), null, F); Silu(h);
                h = Linear(h, T, F, W.D(p + "ff2.lin2.w"), null, D);
                for (int i = 0; i < x.Length; i++) x[i] += 0.5f * h[i];

                // ---- per-block final norm
                return LayerNorm(x, T, D, W.D(p + "out_ln.w"), W.D(p + "out_ln.b"));
            }

            /// <summary>score(i,j) = [(q_i+u)·k_j + (q_i+v)·P_{(T-1)-(i-j)}] / sqrt(head_dim) — the
            /// Transformer-XL rel-shift folded into direct P indexing (Chatterbox kernel convention;
            /// numerically identical to HF's pad-shift-slice, proven by the layer-0 dump gate).</summary>
            static float[] RelPosAttention(float[] q, float[] k, float[] v, float[] P, int T,
                                           float[] biasU, float[] biasV)
            {
                int H = ParakeetConfig.Heads, hd = ParakeetConfig.HeadDim, D = ParakeetConfig.Dim;
                float scale = 1f / (float)Math.Sqrt(hd);
                float[] outv = new float[T * D];
                Parallel.For(0, T * H, ih =>
                {
                    int i = ih / H, h = ih % H;
                    int qBase = i * D + h * hd, uBase = h * hd;
                    float[] qu = new float[hd], qv = new float[hd];
                    for (int d = 0; d < hd; d++)
                    {
                        float qd = q[qBase + d];
                        qu[d] = qd + biasU[uBase + d];
                        qv[d] = qd + biasV[uBase + d];
                    }
                    float[] s = new float[T];
                    float m = float.NegativeInfinity;
                    for (int j = 0; j < T; j++)
                    {
                        int kBase = j * D + h * hd;
                        int pBase = ((T - 1) - i + j) * D + h * hd;
                        float acc = 0f;
                        for (int d = 0; d < hd; d++)
                            acc += qu[d] * k[kBase + d] + qv[d] * P[pBase + d];
                        s[j] = acc * scale;
                        if (s[j] > m) m = s[j];
                    }
                    float lsum = 0f;
                    for (int j = 0; j < T; j++) { s[j] = (float)Math.Exp(s[j] - m); lsum += s[j]; }
                    float inv = 1f / lsum;
                    for (int d = 0; d < hd; d++)
                    {
                        float acc = 0f;
                        for (int j = 0; j < T; j++) acc += s[j] * v[j * D + h * hd + d];
                        outv[i * D + h * hd + d] = acc * inv;
                    }
                });
                return outv;
            }

            // ==================================================================== §5-§6 TDT decode
            public sealed class TdtResult
            {
                public List<int> Tokens = new List<int>();
                public List<int> Frames = new List<int>();
                public List<int> Durs = new List<int>();
                public int Steps;
                public List<float[]> FirstLogits = new List<float[]>();   // first 8 joint evals
            }

            /// <summary>SPEC §6 greedy loop over enc_proj [tEnc, 640]. CPU-only; LSTM state advances
            /// on non-blank emissions only; blank+dur0 forces +1; max_symbols guard forces +1.</summary>
            public TdtResult Decode(float[] encProj, int tEnc)
            {
                int PD = ParakeetConfig.PredDim;
                float[] emb = W.D("dec/embedding");
                float[] h0 = new float[PD], c0 = new float[PD], h1 = new float[PD], c1 = new float[PD];
                float[] x = new float[PD];
                Array.Copy(emb, blank * PD, x, 0, PD);       // blank row (== zeros, asserted at export)
                float[] predOut = PredStep(x, h0, c0, h1, c1);

                var r = new TdtResult();
                float[] logits = new float[V + ParakeetConfig.Durations];
                int t = 0;
                while (t < tEnc)
                {
                    int symbols = 0;
                    while (true)
                    {
                        Joint(encProj, t, predOut, logits);
                        if (r.Steps < 8) { r.FirstLogits.Add((float[])logits.Clone()); }
                        r.Steps++;
                        int k = 0;
                        float best = logits[0];
                        for (int i = 1; i < V; i++) if (logits[i] > best) { best = logits[i]; k = i; }
                        int d = 0;
                        best = logits[V];
                        for (int i = 1; i < ParakeetConfig.Durations; i++)
                            if (logits[V + i] > best) { best = logits[V + i]; d = i; }

                        if (k != blank)
                        {
                            r.Tokens.Add(k); r.Frames.Add(t); r.Durs.Add(d);
                            Array.Copy(emb, k * PD, x, 0, PD);
                            predOut = PredStep(x, h0, c0, h1, c1);
                        }
                        symbols++;
                        t += d;
                        if (d > 0) break;
                        if (k == blank) { t += 1; break; }                        // blank+dur0 spin guard
                        if (symbols >= ParakeetConfig.MaxSymbolsPerStep) { t += 1; break; }
                    }
                }
                return r;
            }

            /// <summary>2-layer LSTM step (torch gate order i,f,g,o) + pred_proj. States updated in place.</summary>
            float[] PredStep(float[] x, float[] h0, float[] c0, float[] h1, float[] c1)
            {
                float[] y0 = LstmCell(x, h0, c0, W.D("dec/lstm.wih0"), W.D("dec/lstm.whh0"),
                                                 W.D("dec/lstm.bih0"), W.D("dec/lstm.bhh0"));
                float[] y1 = LstmCell(y0, h1, c1, W.D("dec/lstm.wih1"), W.D("dec/lstm.whh1"),
                                                  W.D("dec/lstm.bih1"), W.D("dec/lstm.bhh1"));
                float[] pw = W.D("dec/pred_proj.w"); float[] pb = W.D("dec/pred_proj.b");
                int PD = ParakeetConfig.PredDim;
                float[] o = new float[PD];
                for (int r2 = 0; r2 < PD; r2++)
                {
                    float acc = pb[r2];
                    int wb = r2 * PD;
                    for (int i = 0; i < PD; i++) acc += pw[wb + i] * y1[i];
                    o[r2] = acc;
                }
                return o;
            }

            static float[] LstmCell(float[] x, float[] h, float[] c, float[] wih, float[] whh,
                                    float[] bih, float[] bhh)
            {
                int PD = ParakeetConfig.PredDim;
                float[] gates = new float[4 * PD];
                Parallel.For(0, 4 * PD, r =>
                {
                    float acc = bih[r] + bhh[r];
                    int wi = r * PD;
                    for (int i = 0; i < PD; i++) acc += wih[wi + i] * x[i] + whh[wi + i] * h[i];
                    gates[r] = acc;
                });
                for (int i = 0; i < PD; i++)
                {
                    float ig = Sigmoid(gates[i]);
                    float fg = Sigmoid(gates[PD + i]);
                    float gg = (float)Math.Tanh(gates[2 * PD + i]);
                    float og = Sigmoid(gates[3 * PD + i]);
                    c[i] = fg * c[i] + ig * gg;
                    h[i] = og * (float)Math.Tanh(c[i]);
                }
                return (float[])h.Clone();
            }

            /// <summary>logits[V+5] = head( relu(encProj[t] + predOut) ).</summary>
            void Joint(float[] encProj, int t, float[] predOut, float[] logits)
            {
                int PD = ParakeetConfig.PredDim;
                float[] z = new float[PD];
                int eb = t * PD;
                for (int i = 0; i < PD; i++)
                {
                    float s = encProj[eb + i] + predOut[i];
                    z[i] = s > 0f ? s : 0f;
                }
                float[] hw = W.D("joint/head.w"); float[] hb = W.D("joint/head.b");
                int O = logits.Length;
                Parallel.For(0, O, o =>
                {
                    float acc = hb[o];
                    int wb = o * PD;
                    for (int i = 0; i < PD; i++) acc += hw[wb + i] * z[i];
                    logits[o] = acc;
                });
            }

            // ==================================================================== full chain
            public string Transcribe(float[] samples, ParakeetTokenizer tok)
            {
                float[] mel = Mel(samples, out int tMel);
                float[] sub = Subsample(mel, tMel, out int tEnc);
                float[] enc = Encoder(sub, tEnc, out _);
                float[] proj = EncProj(enc, tEnc);
                return tok.Decode(Decode(proj, tEnc).Tokens);
            }

            // ==================================================================== helpers
            static float Sigmoid(float x) => 1f / (1f + (float)Math.Exp(-x));

            static void Silu(float[] x)
            {
                for (int i = 0; i < x.Length; i++) x[i] = x[i] / (1f + (float)Math.Exp(-x[i]));
            }

            /// <summary>[T,cin] x W[cout,cin] (+bias) -> [T,cout].</summary>
            static float[] Linear(float[] x, int T, int cin, float[] w, float[] b, int cout)
            {
                float[] y = new float[T * cout];
                Parallel.For(0, T, t =>
                {
                    int xb = t * cin, yb = t * cout;
                    for (int o = 0; o < cout; o++)
                    {
                        float acc = b != null ? b[o] : 0f;
                        int wb = o * cin;
                        for (int i = 0; i < cin; i++) acc += w[wb + i] * x[xb + i];
                        y[yb + o] = acc;
                    }
                });
                return y;
            }

            static float[] LayerNorm(float[] x, int T, int C, float[] gamma, float[] beta)
            {
                float[] y = new float[T * C];
                Parallel.For(0, T, t =>
                {
                    int b = t * C;
                    double mean = 0;
                    for (int i = 0; i < C; i++) mean += x[b + i];
                    mean /= C;
                    double var = 0;
                    for (int i = 0; i < C; i++) { double d = x[b + i] - mean; var += d * d; }
                    float rstd = (float)(1.0 / Math.Sqrt(var / C + ParakeetConfig.LnEps));
                    for (int i = 0; i < C; i++)
                        y[b + i] = ((x[b + i] - (float)mean) * rstd) * gamma[i] + beta[i];
                });
                return y;
            }
        }
    }
}
