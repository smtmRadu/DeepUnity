using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace DeepUnity
{
    namespace QwenASRModeling
    {
        // Pure-C# fp32 reference implementation of the full Qwen3-ASR pipeline
        // (mel → encoder → projector → Qwen3 decoder → greedy). No UnityEngine.
        //
        // Role (WS-B/Kokoro pattern): this is the PARITY-GRADED twin of the GPU port — the net8.0
        // harness (validation/harness) runs it stage-by-stage against the D0 python dumps, so every
        // formula the QwenASRCS.compute kernels implement is validated here first, outside Unity.
        // It is NOT the runtime inference path (too slow for the decoder); QwenASRModel (GPU) is.
        //
        // All math follows SPEC.md §1-§6; the erf-GELU / reflect-pad STFT / global-max mel clamp /
        // per-chunk conv / 104-token attention windows / QK-norm-before-RoPE details live here in
        // executable form.
        public class QwenASRCPU
        {
            readonly QwenASRTensors T;
            readonly int D, EncLayers, EncHeads, EncHd, Ffn;   // encoder dims
            readonly int H, Interm;                            // decoder dims

            public QwenASRCPU(QwenASRTensors tensors)
            {
                T = tensors;
                D = QwenASRConfig.ENC_D_MODEL; EncLayers = QwenASRConfig.ENC_LAYERS;
                EncHeads = QwenASRConfig.ENC_HEADS; EncHd = D / EncHeads; Ffn = QwenASRConfig.ENC_FFN;
                H = QwenASRConfig.HIDDEN_SIZE; Interm = QwenASRConfig.MLP_INTERMEDIATE_SIZE;
            }

            // ---------------------------------------------------------------- helpers
            // erf-GELU (activation_function "gelu" = exact erf, NOT GPT-2's tanh — SPEC risk #1).
            // erf via Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7, plenty for fp32 parity).
            public static float GeluErf(float x)
            {
                double z = x / 1.4142135623730951;
                double t = 1.0 / (1.0 + 0.3275911 * Math.Abs(z));
                double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                                   - 0.284496736) * t + 0.254829592) * t * Math.Exp(-z * z);
                double erf = z >= 0 ? y : -y;
                return (float)(0.5 * x * (1.0 + erf));
            }

            static float Silu(float x) => x / (1f + MathF.Exp(-x));

            // y[t,o] = act(Σ_i x[t,i] w[o,i] + b[o]); w row-major [O, I]; b may be null.
            static float[] MatMul(float[] x, int rows, int I, float[] w, int O, float[] b, bool geluAct = false)
            {
                float[] y = new float[rows * O];
                Parallel.For(0, O, o =>
                {
                    int wBase = o * I;
                    for (int t = 0; t < rows; t++)
                    {
                        float acc = b != null ? b[o] : 0f;
                        int xBase = t * I;
                        for (int i = 0; i < I; i++) acc += x[xBase + i] * w[wBase + i];
                        y[t * O + o] = geluAct ? GeluErf(acc) : acc;
                    }
                });
                return y;
            }

            static void LayerNorm(float[] x, int rows, int dim, float[] g, float[] b, float eps, float[] dst)
            {
                for (int t = 0; t < rows; t++)
                {
                    int o = t * dim;
                    float mean = 0f;
                    for (int i = 0; i < dim; i++) mean += x[o + i];
                    mean /= dim;
                    float var = 0f;
                    for (int i = 0; i < dim; i++) { float d = x[o + i] - mean; var += d * d; }
                    float inv = 1f / MathF.Sqrt(var / dim + eps);
                    for (int i = 0; i < dim; i++) dst[o + i] = (x[o + i] - mean) * inv * g[i] + b[i];
                }
            }

            // Standard (llama/Qwen3) RMSNorm: x_hat * gamma  — NOT Qwen3.5's (1+gamma) variant.
            static void RmsNorm(float[] x, int rows, int dim, float[] g, float eps, float[] dst)
            {
                for (int t = 0; t < rows; t++)
                {
                    int o = t * dim;
                    float ss = 0f;
                    for (int i = 0; i < dim; i++) ss += x[o + i] * x[o + i];
                    float inv = 1f / MathF.Sqrt(ss / dim + eps);
                    for (int i = 0; i < dim; i++) dst[o + i] = x[o + i] * inv * g[i];
                }
            }

            // ---------------------------------------------------------------- §1 mel frontend
            /// <summary>Normalized log-mel, layout [128, paddedFrames] (matches the mel.npy dump).
            /// validFrames = floor(max(n,8000)/160); paddedFrames = next multiple of 100.</summary>
            public float[] Mel(float[] samples, out int validFrames, out int paddedFrames)
            {
                int n = Math.Max(samples.Length, QwenASRConfig.MIN_SAMPLES);
                float[] x = new float[n];                       // min-length zero pad (SPEC §1)
                Array.Copy(samples, x, samples.Length);

                const int NFFT = QwenASRConfig.N_FFT, HOP = QwenASRConfig.HOP, NF = QwenASRConfig.N_FREQS;
                int T_ = n / HOP;                               // stft frames minus the dropped last
                validFrames = T_;
                paddedFrames = (T_ + QwenASRConfig.MEL_CHUNK - 1) / QwenASRConfig.MEL_CHUNK * QwenASRConfig.MEL_CHUNK;

                // reflect pad 200 both sides (torch.stft center=True)
                int pad = NFFT / 2;
                float[] xp = new float[n + 2 * pad];
                for (int i = 0; i < pad; i++) xp[i] = x[pad - i];
                Array.Copy(x, 0, xp, pad, n);
                for (int j = 1; j <= pad; j++) xp[pad + n - 1 + j] = x[n - 1 - j];

                // periodic hann + DFT basis
                float[] win = new float[NFFT];
                for (int j = 0; j < NFFT; j++) win[j] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * j / NFFT));
                float[] cosB = new float[NF * NFFT], sinB = new float[NF * NFFT];
                for (int f = 0; f < NF; f++)
                    for (int j = 0; j < NFFT; j++)
                    {
                        double a = 2.0 * Math.PI * f * j / NFFT;
                        cosB[f * NFFT + j] = (float)Math.Cos(a);
                        sinB[f * NFFT + j] = (float)Math.Sin(a);
                    }

                float[] melFb = T.F("frontend/mel_filters");    // [201, 128] exported slaney bank
                const int M = QwenASRConfig.N_MELS;
                float[] logMel = new float[M * T_];             // [mel, t] layout
                Parallel.For(0, T_, t =>
                {
                    float[] fr = new float[NFFT];
                    int s0 = t * HOP;
                    for (int j = 0; j < NFFT; j++) fr[j] = xp[s0 + j] * win[j];
                    float[] pow = new float[NF];
                    for (int f = 0; f < NF; f++)
                    {
                        float re = 0f, im = 0f;
                        int fb = f * NFFT;
                        for (int j = 0; j < NFFT; j++) { re += fr[j] * cosB[fb + j]; im += fr[j] * sinB[fb + j]; }
                        pow[f] = re * re + im * im;
                    }
                    for (int m = 0; m < M; m++)
                    {
                        float acc = 0f;
                        for (int f = 0; f < NF; f++) acc += melFb[f * M + m] * pow[f];
                        logMel[m * T_ + t] = MathF.Log10(MathF.Max(acc, 1e-10f));
                    }
                });

                float gmax = float.MinValue;                    // GLOBAL max clamp (whole clip)
                for (int i = 0; i < logMel.Length; i++) gmax = MathF.Max(gmax, logMel[i]);
                float[] mel = new float[M * paddedFrames];      // pad-to-100 with zeros AFTER normalize
                for (int m = 0; m < M; m++)
                    for (int t = 0; t < T_; t++)
                        mel[m * paddedFrames + t] = (MathF.Max(logMel[m * T_ + t], gmax - 8f) + 4f) / 4f;
                return mel;
            }

            // ---------------------------------------------------------------- §2 encoder
            /// <summary>Encoder output after ln_post: [nTokens, D]. mel layout [128, paddedFrames].</summary>
            public float[] Encode(float[] mel, int validFrames, int paddedFrames, out int nTokens)
            {
                const int M = QwenASRConfig.N_MELS, CH = QwenASRConfig.ENC_CONV_CH;
                int chunks = paddedFrames / QwenASRConfig.MEL_CHUNK;
                nTokens = QwenASRConfig.AudioTokenCount(validFrames);
                float[] posEmb = T.F("enc/pos_emb");            // [13, D]
                float[] w1 = T.F("enc/conv2d1.w"); float[] b1 = T.F("enc/conv2d1.b");
                float[] w2 = T.F("enc/conv2d2.w"); float[] b2 = T.F("enc/conv2d2.b");
                float[] w3 = T.F("enc/conv2d3.w"); float[] b3 = T.F("enc/conv2d3.b");
                float[] wOut = T.F("enc/conv_out.w");           // [D, 7680], no bias

                float[] tokens = new float[nTokens * D];
                int written = 0;
                for (int c = 0; c < chunks; c++)
                {
                    // chunk [1, 128, 100] — conv padding is per-chunk (SPEC §2.1)
                    float[] cin = new float[M * 100];
                    for (int m = 0; m < M; m++)
                        Array.Copy(mel, m * paddedFrames + c * 100, cin, m * 100, 100);

                    float[] h1 = Conv2dS2(cin, 1, M, 100, w1, b1, CH);       // [480, 64, 50]
                    float[] h2 = Conv2dS2(h1, CH, 64, 50, w2, b2, CH);       // [480, 32, 25]
                    float[] h3 = Conv2dS2(h2, CH, 32, 25, w3, b3, CH);       // [480, 16, 13]

                    // valid post-CNN positions of this chunk (partial final chunk keeps Ceil3(r))
                    int validInChunk = Math.Min(Math.Max(validFrames - c * 100, 0), 100);
                    int keep = QwenASRConfig.Ceil3(validInChunk);

                    // permute [480,16,13] -> [13, 480*16] (channel-major, then freq) -> conv_out -> +PE
                    for (int t = 0; t < keep; t++)
                    {
                        float[] flat = new float[QwenASRConfig.ENC_CONV_FLAT];
                        for (int ch = 0; ch < CH; ch++)
                            for (int fb = 0; fb < 16; fb++)
                                flat[ch * 16 + fb] = h3[(ch * 16 + fb) * 13 + t];
                        int oBase = (written + t) * D;
                        Parallel.For(0, D, d =>
                        {
                            float acc = 0f;
                            int wb = d * QwenASRConfig.ENC_CONV_FLAT;
                            for (int i = 0; i < QwenASRConfig.ENC_CONV_FLAT; i++) acc += flat[i] * wOut[wb + i];
                            tokens[oBase + d] = acc + posEmb[t * D + d];
                        });
                    }
                    written += keep;
                }

                // transformer layers with 104-token block-diagonal windows
                float[] xbuf = tokens, norm = new float[nTokens * D];
                for (int li = 0; li < EncLayers; li++)
                {
                    string lp = $"enc/layer_{li}/";
                    LayerNorm(xbuf, nTokens, D, T.F(lp + "ln1.w"), T.F(lp + "ln1.b"), QwenASRConfig.ENC_LN_EPS, norm);
                    float[] q = MatMul(norm, nTokens, D, T.F(lp + "attn_q.w"), D, T.F(lp + "attn_q.b"));
                    float[] k = MatMul(norm, nTokens, D, T.F(lp + "attn_k.w"), D, T.F(lp + "attn_k.b"));
                    float[] v = MatMul(norm, nTokens, D, T.F(lp + "attn_v.w"), D, T.F(lp + "attn_v.b"));
                    float[] att = WindowedAttention(q, k, v, nTokens);
                    float[] attO = MatMul(att, nTokens, D, T.F(lp + "attn_out.w"), D, T.F(lp + "attn_out.b"));
                    for (int i = 0; i < xbuf.Length; i++) xbuf[i] += attO[i];

                    LayerNorm(xbuf, nTokens, D, T.F(lp + "ln2.w"), T.F(lp + "ln2.b"), QwenASRConfig.ENC_LN_EPS, norm);
                    float[] f1 = MatMul(norm, nTokens, D, T.F(lp + "fc1.w"), Ffn, T.F(lp + "fc1.b"), geluAct: true);
                    float[] f2 = MatMul(f1, nTokens, Ffn, T.F(lp + "fc2.w"), D, T.F(lp + "fc2.b"));
                    for (int i = 0; i < xbuf.Length; i++) xbuf[i] += f2[i];
                    // keep harness RAM flat — encoder layer weights are not revisited
                    T.Evict(lp + "attn_q.w"); T.Evict(lp + "attn_k.w"); T.Evict(lp + "attn_v.w");
                    T.Evict(lp + "attn_out.w"); T.Evict(lp + "fc1.w"); T.Evict(lp + "fc2.w");
                }
                float[] outp = new float[nTokens * D];
                LayerNorm(xbuf, nTokens, D, T.F("enc/ln_post.w"), T.F("enc/ln_post.b"), QwenASRConfig.ENC_LN_EPS, outp);
                return outp;
            }

            // conv2d k3 s2 p1 + erf-GELU; in [IC, Hh, Ww] -> out [OC, ceil(Hh/2), ceil(Ww/2)]
            static float[] Conv2dS2(float[] input, int IC, int Hh, int Ww, float[] w, float[] b, int OC)
            {
                int OH = (Hh - 1) / 2 + 1, OW = (Ww - 1) / 2 + 1;
                float[] output = new float[OC * OH * OW];
                Parallel.For(0, OC, oc =>
                {
                    for (int oy = 0; oy < OH; oy++)
                        for (int ox = 0; ox < OW; ox++)
                        {
                            float acc = b[oc];
                            for (int ic = 0; ic < IC; ic++)
                            {
                                int wb = ((oc * IC + ic) * 3) * 3;
                                int ib = ic * Hh * Ww;
                                for (int ky = 0; ky < 3; ky++)
                                {
                                    int iy = 2 * oy - 1 + ky;
                                    if (iy < 0 || iy >= Hh) continue;
                                    for (int kx = 0; kx < 3; kx++)
                                    {
                                        int ix = 2 * ox - 1 + kx;
                                        if (ix < 0 || ix >= Ww) continue;
                                        acc += w[wb + ky * 3 + kx] * input[ib + iy * Ww + ix];
                                    }
                                }
                            }
                            output[(oc * OH + oy) * OW + ox] = GeluErf(acc);
                        }
                });
                return output;
            }

            // non-causal MHA restricted to 104-token windows (blocks of ENC_WINDOW_TOKENS; SPEC §2.2)
            float[] WindowedAttention(float[] q, float[] k, float[] v, int n)
            {
                int hd = EncHd;
                float scale = 1f / MathF.Sqrt(hd);
                float[] outp = new float[n * D];
                Parallel.For(0, EncHeads, h =>
                {
                    int hOff = h * hd;
                    float[] scores = new float[n];
                    for (int i = 0; i < n; i++)
                    {
                        int w0 = i / QwenASRConfig.ENC_WINDOW_TOKENS * QwenASRConfig.ENC_WINDOW_TOKENS;
                        int w1 = Math.Min(w0 + QwenASRConfig.ENC_WINDOW_TOKENS, n);
                        float mx = float.MinValue;
                        for (int j = w0; j < w1; j++)
                        {
                            float dot = 0f;
                            for (int d = 0; d < hd; d++) dot += q[i * D + hOff + d] * k[j * D + hOff + d];
                            scores[j] = dot * scale;
                            mx = MathF.Max(mx, scores[j]);
                        }
                        float sum = 0f;
                        for (int j = w0; j < w1; j++) { scores[j] = MathF.Exp(scores[j] - mx); sum += scores[j]; }
                        float inv = 1f / sum;
                        for (int d = 0; d < hd; d++)
                        {
                            float acc = 0f;
                            for (int j = w0; j < w1; j++) acc += scores[j] * v[j * D + hOff + d];
                            outp[i * D + hOff + d] = acc * inv;
                        }
                    }
                });
                return outp;
            }

            // ---------------------------------------------------------------- §3 projector
            public float[] Project(float[] encOut, int nTokens)
            {
                float[] h1 = MatMul(encOut, nTokens, D, T.F("proj/linear_1.w"), D, T.F("proj/linear_1.b"), geluAct: true);
                return MatMul(h1, nTokens, D, T.F("proj/linear_2.w"), H, T.F("proj/linear_2.b"));
            }

            // ---------------------------------------------------------------- §5 prompt
            /// <summary>Chat scaffold (chat_template.jinja, exact): system text = context or language
            /// or empty; assistantPrefix = "language X&lt;asr_text&gt;" for forced-language mode.</summary>
            public static int[] BuildPromptIds(QwenASRTokenizer tok, int nAudioTokens,
                                               string system = "", string assistantPrefix = null)
            {
                var ids = new List<int> { QwenASRConfig.IM_START_TOKEN_ID };
                ids.AddRange(tok.Encode("system\n" + (system ?? "")));
                ids.Add(QwenASRConfig.IM_END_TOKEN_ID);
                ids.AddRange(tok.Encode("\n"));
                ids.Add(QwenASRConfig.IM_START_TOKEN_ID);
                ids.AddRange(tok.Encode("user\n"));
                ids.Add(QwenASRConfig.AUDIO_START_TOKEN_ID);
                for (int i = 0; i < nAudioTokens; i++) ids.Add(QwenASRConfig.AUDIO_PAD_TOKEN_ID);
                ids.Add(QwenASRConfig.AUDIO_END_TOKEN_ID);
                ids.Add(QwenASRConfig.IM_END_TOKEN_ID);
                ids.AddRange(tok.Encode("\n"));
                ids.Add(QwenASRConfig.IM_START_TOKEN_ID);
                ids.AddRange(tok.Encode("assistant\n"));
                if (!string.IsNullOrEmpty(assistantPrefix))
                {
                    // "language French" BPE + the <asr_text> special id (never BPE-encoded)
                    ids.AddRange(tok.Encode(assistantPrefix));
                    ids.Add(QwenASRConfig.ASR_TEXT_TOKEN_ID);
                }
                return ids.ToArray();
            }

            /// <summary>Transcript = text after &lt;asr_text&gt; (id cut, then decode) — SPEC §5.</summary>
            public static string ParseTranscript(QwenASRTokenizer tok, List<int> generated)
            {
                int cut = generated.IndexOf(QwenASRConfig.ASR_TEXT_TOKEN_ID);
                var tail = cut >= 0 ? generated.GetRange(cut + 1, generated.Count - cut - 1) : generated;
                return tok.Decode(tail).Trim();
            }

            // ---------------------------------------------------------------- §4+§6 decoder
            float[][] kCache, vCache;   // [layer][pos * HEADS_KV * HEAD_DIM]
            int cachedLen;
            float[] embed;              // tied embed/lm_head [vocab, H]

            void EnsureDecoder(int capacity)
            {
                embed ??= T.Embedding(QwenASRConfig.VOCAB_SIZE, H);
                int kvw = QwenASRConfig.HEADS_KV * QwenASRConfig.HEAD_DIM;
                kCache = new float[QwenASRConfig.NUM_LAYERS][];
                vCache = new float[QwenASRConfig.NUM_LAYERS][];
                for (int i = 0; i < QwenASRConfig.NUM_LAYERS; i++)
                {
                    kCache[i] = new float[capacity * kvw];
                    vCache[i] = new float[capacity * kvw];
                }
                cachedLen = 0;
            }

            static void RopeSplitHalf(float[] x, int rows, int heads, int posOffset)
            {
                const int hd = QwenASRConfig.HEAD_DIM;
                int half = hd / 2;
                for (int t = 0; t < rows; t++)
                    for (int h = 0; h < heads; h++)
                    {
                        int b = (t * heads + h) * hd;
                        for (int i = 0; i < half; i++)
                        {
                            float inv = MathF.Pow(QwenASRConfig.ROPE_THETA, -2f * i / hd);
                            float a = (posOffset + t) * inv;
                            float c = MathF.Cos(a), s = MathF.Sin(a);
                            float x1 = x[b + i], x2 = x[b + i + half];
                            x[b + i] = x1 * c - x2 * s;
                            x[b + i + half] = x2 * c + x1 * s;
                        }
                    }
            }

            // One decoder forward over `rows` new positions (prefill or single-step). Returns the
            // final hidden states [rows, H] BEFORE the final norm; KV cache is appended.
            float[] DecoderForward(float[] hidden, int rows)
            {
                const int hq = QwenASRConfig.HEADS_Q, hkv = QwenASRConfig.HEADS_KV, hd = QwenASRConfig.HEAD_DIM;
                int qDim = hq * hd, kvDim = hkv * hd;
                float scale = 1f / MathF.Sqrt(hd);
                float[] norm = new float[rows * H];

                for (int li = 0; li < QwenASRConfig.NUM_LAYERS; li++)
                {
                    string lp = $"dec/layer_{li}/";
                    RmsNorm(hidden, rows, H, T.F(lp + "input_ln"), QwenASRConfig.RMS_EPS, norm);
                    float[] q = MatMul(norm, rows, H, T.F(lp + "q_proj"), qDim, null);
                    float[] k = MatMul(norm, rows, H, T.F(lp + "k_proj"), kvDim, null);
                    float[] v = MatMul(norm, rows, H, T.F(lp + "v_proj"), kvDim, null);

                    // QK-norm (RMS over head_dim, standard gamma) BEFORE RoPE — SPEC §4
                    RmsNorm(q, rows * hq, hd, T.F(lp + "q_norm"), QwenASRConfig.RMS_EPS, q);
                    RmsNorm(k, rows * hkv, hd, T.F(lp + "k_norm"), QwenASRConfig.RMS_EPS, k);
                    RopeSplitHalf(q, rows, hq, cachedLen);
                    RopeSplitHalf(k, rows, hkv, cachedLen);

                    Array.Copy(k, 0, kCache[li], cachedLen * kvDim, rows * kvDim);
                    Array.Copy(v, 0, vCache[li], cachedLen * kvDim, rows * kvDim);
                    int kvLen = cachedLen + rows;

                    float[] att = new float[rows * qDim];
                    float[] kc = kCache[li], vc = vCache[li];
                    Parallel.For(0, hq, h =>
                    {
                        int kvh = h / (hq / hkv);
                        float[] scores = new float[kvLen];
                        for (int i = 0; i < rows; i++)
                        {
                            int absQ = cachedLen + i;
                            float mx = float.MinValue;
                            for (int j = 0; j <= absQ; j++)
                            {
                                float dot = 0f;
                                int qb = (i * hq + h) * hd, kb = (j * hkv + kvh) * hd;
                                for (int d = 0; d < hd; d++) dot += q[qb + d] * kc[kb + d];
                                scores[j] = dot * scale;
                                mx = MathF.Max(mx, scores[j]);
                            }
                            float sum = 0f;
                            for (int j = 0; j <= absQ; j++) { scores[j] = MathF.Exp(scores[j] - mx); sum += scores[j]; }
                            float inv = 1f / sum;
                            int ob = (i * hq + h) * hd;
                            for (int d = 0; d < hd; d++)
                            {
                                float acc = 0f;
                                for (int j = 0; j <= absQ; j++) acc += scores[j] * vc[(j * hkv + kvh) * hd + d];
                                att[ob + d] = acc * inv;
                            }
                        }
                    });

                    float[] attO = MatMul(att, rows, qDim, T.F(lp + "o_proj"), H, null);
                    for (int i = 0; i < rows * H; i++) hidden[i] += attO[i];

                    RmsNorm(hidden, rows, H, T.F(lp + "post_attn_ln"), QwenASRConfig.RMS_EPS, norm);
                    float[] g = MatMul(norm, rows, H, T.F(lp + "mlp_gate"), Interm, null);
                    float[] u = MatMul(norm, rows, H, T.F(lp + "mlp_up"), Interm, null);
                    for (int i = 0; i < g.Length; i++) g[i] = Silu(g[i]) * u[i];
                    float[] dn = MatMul(g, rows, Interm, T.F(lp + "mlp_down"), H, null);
                    for (int i = 0; i < rows * H; i++) hidden[i] += dn[i];
                }
                cachedLen += rows;
                return hidden;
            }

            float[] LogitsAt(float[] hidden, int rows, int pos)
            {
                float[] last = new float[H];
                Array.Copy(hidden, pos * H, last, 0, H);
                float[] normed = new float[H];
                RmsNorm(last, 1, H, T.F("dec/norm"), QwenASRConfig.RMS_EPS, normed);
                float[] logits = new float[QwenASRConfig.VOCAB_SIZE];
                Parallel.For(0, QwenASRConfig.VOCAB_SIZE, vv =>
                {
                    float acc = 0f;
                    long b = (long)vv * H;
                    for (int i = 0; i < H; i++) acc += normed[i] * embed[b + i];
                    logits[vv] = acc;
                });
                return logits;
            }

            float[] EmbedPrompt(int[] ids, float[] projOut)
            {
                float[] hidden = new float[ids.Length * H];
                int audioRow = 0;
                for (int t = 0; t < ids.Length; t++)
                {
                    if (ids[t] == QwenASRConfig.AUDIO_PAD_TOKEN_ID)
                        Array.Copy(projOut, audioRow++ * H, hidden, t * H, H);   // scatter, in order
                    else
                        Array.Copy(embed, (long)ids[t] * H, hidden, t * H, H);
                }
                return hidden;
            }

            /// <summary>Prefill only; returns logits at the last prompt position (parity stage).</summary>
            public float[] PrefillLogits(int[] promptIds, float[] projOut)
            {
                EnsureDecoder(promptIds.Length + 160);
                float[] hidden = DecoderForward(EmbedPrompt(promptIds, projOut), promptIds.Length);
                return LogitsAt(hidden, promptIds.Length, promptIds.Length - 1);
            }

            /// <summary>Greedy decode (temperature 0, generation_config) until im_end/endoftext.</summary>
            public List<int> Greedy(int[] promptIds, float[] projOut, int maxNew = 128)
            {
                EnsureDecoder(promptIds.Length + maxNew + 8);
                float[] hidden = DecoderForward(EmbedPrompt(promptIds, projOut), promptIds.Length);
                float[] logits = LogitsAt(hidden, promptIds.Length, promptIds.Length - 1);

                var outIds = new List<int>();
                for (int step = 0; step < maxNew; step++)
                {
                    int best = 0;
                    for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
                    outIds.Add(best);
                    if (best == QwenASRConfig.IM_END_TOKEN_ID || best == QwenASRConfig.ENDOFTEXT_TOKEN_ID) break;

                    float[] h1 = new float[H];
                    Array.Copy(embed, (long)best * H, h1, 0, H);
                    hidden = DecoderForward(h1, 1);
                    logits = LogitsAt(hidden, 1, 0);
                }
                return outIds;
            }
        }
    }
}
