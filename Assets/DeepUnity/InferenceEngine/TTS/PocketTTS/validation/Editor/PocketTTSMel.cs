using System;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        // Shared perceptual metric for the pocket-tts probes: a log-mel spectrogram + its
        // correlation between two waveforms. Raw-sample correlation is PHASE-sensitive — a
        // quantized autoregressive model desyncs the waveform phase (a fractional-sample time
        // shift drops sample-corr to ~0.17) while sounding identical; the log-mel magnitude is
        // phase-invariant and is the right "does it sound the same" proxy (verified in WSL:
        // 40-sample shift => sample-corr 0.17 but mel-corr 0.998). fp16 stays bit-exact so it
        // still gates on sample-corr; int8 gates on THIS.
        //
        // Matches the sanity-check reference: nfft 1024, hop 256, 80 mel bins, Hann window,
        // log(mel + 1e-5). Standalone (no torch) — plain FFT-free DFT-per-frame is fine for the
        // short probe clips (few seconds), and keeps the metric self-contained in the editor.
        public static class PocketTTSMel
        {
            const int NFFT = 1024, HOP = 256, NMEL = 80;
            static float[][] _fb;   // [NMEL][NFFT/2+1] triangular filterbank (cached)

            // log-mel spectrogram flattened [NMEL * frames]
            public static float[] LogMel(float[] wav, int sampleRate)
            {
                int bins = NFFT / 2 + 1;
                if (_fb == null) _fb = BuildFilterbank(sampleRate, bins);
                int frames = wav.Length >= NFFT ? 1 + (wav.Length - NFFT) / HOP : 1;
                var window = new float[NFFT];
                for (int i = 0; i < NFFT; i++) window[i] = 0.5f - 0.5f * Mathf.Cos(2f * Mathf.PI * i / (NFFT - 1));

                var mag = new float[bins];
                var outv = new float[NMEL * frames];
                for (int f = 0; f < frames; f++)
                {
                    int start = f * HOP;
                    // magnitude spectrum of the windowed frame (naive DFT over the real signal)
                    for (int k = 0; k < bins; k++)
                    {
                        double re = 0, im = 0;
                        double w = -2.0 * Math.PI * k / NFFT;
                        for (int n = 0; n < NFFT; n++)
                        {
                            int idx = start + n;
                            float s = idx < wav.Length ? wav[idx] * window[n] : 0f;
                            if (s == 0f) continue;
                            double a = w * n;
                            re += s * Math.Cos(a);
                            im += s * Math.Sin(a);
                        }
                        mag[k] = (float)Math.Sqrt(re * re + im * im);
                    }
                    // apply filterbank -> log
                    for (int m = 0; m < NMEL; m++)
                    {
                        float acc = 0f; float[] row = _fb[m];
                        for (int k = 0; k < bins; k++) acc += row[k] * mag[k];
                        outv[m * frames + f] = Mathf.Log(acc + 1e-5f);
                    }
                }
                return outv;
            }

            /// <summary>Correlation of the log-mel spectrograms of a and b (phase-invariant "sounds
            /// the same" score). ~1.0 = perceptually identical.</summary>
            public static float MelCorr(float[] a, float[] b, int sampleRate)
            {
                float[] ma = LogMel(a, sampleRate), mb = LogMel(b, sampleRate);
                int n = Math.Min(ma.Length, mb.Length);
                double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                for (int i = 0; i < n; i++)
                {
                    sa += ma[i]; sb += mb[i];
                    saa += (double)ma[i] * ma[i]; sbb += (double)mb[i] * mb[i]; sab += (double)ma[i] * mb[i];
                }
                double cov = sab / n - (sa / n) * (sb / n);
                double va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n);
                return (float)(cov / Math.Sqrt(Math.Max(va * vb, 1e-20)));
            }

            static float[][] BuildFilterbank(int sr, int bins)
            {
                var freq = new float[bins];
                for (int k = 0; k < bins; k++) freq[k] = (float)k * sr / NFFT;
                float mmax = 2595f * Mathf.Log10(1f + sr / 2f / 700f);
                var hz = new float[NMEL + 2];
                for (int i = 0; i < NMEL + 2; i++)
                {
                    float mel = mmax * i / (NMEL + 1);
                    hz[i] = 700f * (Mathf.Pow(10f, mel / 2595f) - 1f);
                }
                var fb = new float[NMEL][];
                for (int m = 0; m < NMEL; m++)
                {
                    fb[m] = new float[bins];
                    float lo = hz[m], ce = hz[m + 1], hi = hz[m + 2];
                    for (int k = 0; k < bins; k++)
                        fb[m][k] = Mathf.Clamp(Mathf.Min((freq[k] - lo) / (ce - lo + 1e-9f), (hi - freq[k]) / (hi - ce + 1e-9f)), 0f, 1f);
                }
                return fb;
            }
        }
    }
}
