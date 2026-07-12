namespace DeepUnity
{
    namespace ParakeetModeling
    {
        public enum ParakeetVariant
        {
            /// <summary>parakeet-tdt-0.6b-v2 — English only, best English WER. Vocab 1025.</summary>
            V2,
            /// <summary>parakeet-tdt-0.6b-v3 — 25 European languages (incl. Romanian). Vocab 8193.</summary>
            V3,
        }

        // Compile-time architecture constants for Parakeet-TDT 0.6B (SPEC.md; frozen from
        // nvidia/parakeet-tdt-0.6b-v3 config.json + the v2 .nemo model_config.yaml, 2026-07-11).
        // §1-§4 (frontend + encoder) are IDENTICAL across v2/v3; only the vocab differs.
        public static class ParakeetConfig
        {
            // ---- mel frontend (§1)
            public const int SampleRate = 16000;
            public const int NFft = 512;
            public const int WinLength = 400;      // symmetric hann, zero-padded centered to NFft
            public const int HopLength = 160;
            public const int NMels = 128;
            public const float Preemphasis = 0.97f;
            public const float LogGuard = 5.9604645e-8f;   // 2^-24
            public const float NormEps = 1e-5f;            // std + eps in per-feature norm

            // ---- subsampling (§2)
            public const int SubChannels = 256;
            public const int SubStages = 3;                // x8 time reduction, 128->16 mel bins
            public const int SubFlat = SubChannels * (NMels >> SubStages);  // 4096
            public const float SecondsPerFrame = 0.08f;    // encoder frame = 80 ms

            // ---- encoder (§3-§4)
            public const int Layers = 24;
            public const int Dim = 1024;
            public const int Heads = 8;
            public const int HeadDim = 128;                // Dim / Heads
            public const int FfnDim = 4096;
            public const int ConvKernel = 9;               // depthwise, pad 4
            public const float LnEps = 1e-5f;              // torch nn.LayerNorm default

            // ---- prediction net + joint (§5)
            public const int PredDim = 640;
            public const int PredLayers = 2;               // LSTM layers
            public const int Durations = 5;                // duration bins 0..4 (value == index)
            public const int MaxSymbolsPerStep = 10;

            public static int Vocab(ParakeetVariant v) => v == ParakeetVariant.V3 ? 8193 : 1025;
            public static int Blank(ParakeetVariant v) => Vocab(v) - 1;
            public static int JointOut(ParakeetVariant v) => Vocab(v) + Durations;
            public static string WeightsFolder(ParakeetVariant v, string quant = "fp16")
                => $"weights_parakeet_tdt_0.6b_{(v == ParakeetVariant.V3 ? "v3" : "v2")}_{quant}";

            /// <summary>Mel frames for a clip: floor(samples/hop) (torch.stft's extra last frame is
            /// masked off by the reference — SPEC §1).</summary>
            public static int MelFrames(int samples) => samples / HopLength;

            /// <summary>Encoder frames after the three stride-2 convs: L -> (L-1)/2 + 1, x3.</summary>
            public static int EncFrames(int melFrames)
            {
                int l = melFrames;
                for (int s = 0; s < SubStages; s++) l = (l - 1) / 2 + 1;
                return l;
            }
        }
    }
}
