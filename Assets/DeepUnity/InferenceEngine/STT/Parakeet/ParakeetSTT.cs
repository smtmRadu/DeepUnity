using System;
using System.Collections;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;
using DeepUnity.ParakeetModeling;

namespace DeepUnity
{
    // Parakeet-TDT 0.6B speech-to-text (push-to-talk: 16 kHz mono clip -> transcript).
    // SPEC: Assets/DeepUnity/InferenceEngine/STT/Parakeet/SPEC.md. Math is transcript-exact vs the HF
    // reference (validation/harness, ALL PASS on both variants).
    //
    // Execution split (SPEC §8):
    //   CPU  mel frontend (ParakeetCPU.Mel — radix-2 FFT, ~10 ms, background task)
    //   GPU  subsampling + 24 FastConformer blocks + enc_proj (ParakeetCS.compute,
    //        ~450 dispatches enqueued in one burst, AsyncGPUReadback of [T,640])
    //   CPU  TDT greedy loop (LSTM 2x640 + joint head, background task) + detokenize
    //
    // Residency: ParakeetWeights (CosyVoiceWeights pattern) streams the encoder to the GPU
    // under the ModelBase budget contract; the small decode-side tensors (~35 MB) stay
    // CPU-resident in ParakeetTensors for the app's lifetime.
    //
    // VRAM: ~1.26 GB fp16 weights + transient per-utterance scratch (~50 MB @ 15 s clip,
    // released after each Transcribe). fp32 activations throughout (SPEC §12: subsampling
    // activations reach ±5.4e3 — never store encoder activations as fp16).
    public class ParakeetSTT : STT
    {
        public readonly ParakeetVariant Variant;

        readonly ParakeetWeights weights;       // GPU-resident encoder
        readonly ParakeetTensors cpuTensors;    // CPU-side: dec/*, joint/head, frontend/*
        readonly ParakeetCPU cpu;               // mel + TDT decode (harness-validated)
        readonly ParakeetTokenizer tokenizer;
        readonly ComputeShader cs;
        readonly int kConv, kPw, kFlat, kLin, kLn, kAdd, kGlu, kDwBn, kAtt;

        Task cpuPreload;                        // decode-side tensor pre-cache (thread-safety)
        bool busy;

        public override int InputSampleRate => ParakeetConfig.SampleRate;
        public override bool IsReady => weights.IsReady && cpuPreload != null && cpuPreload.IsCompleted;
        public override ModelResidency Residency => weights.Residency;
        public override long TotalWeightBytes => weights.BytesTotal;
        public override long UploadedWeightBytes => weights.BytesUploaded;
        public override long LoadBudgetBytesPerFrame
        {
            get => weights.BudgetBytesPerFrame;
            set => weights.BudgetBytesPerFrame = value;
        }

        /// <param name="variant">V3 = 25 languages (default), V2 = English-best.</param>
        /// <param name="paramsPath">Override the weights folder (default resolves
        /// Assets/Resources/Weights/weights_parakeet_tdt_0.6b_{v2,v3}_fp16).</param>
        public ParakeetSTT(ParakeetVariant variant = ParakeetVariant.V3, string paramsPath = null)
        {
            Variant = variant;
            string root = paramsPath ?? Application.dataPath + "/Resources/Weights/"
                                      + ParakeetConfig.WeightsFolder(variant);
            weights = new ParakeetWeights(root);
            cpuTensors = new ParakeetTensors(root);
            cpu = new ParakeetCPU(cpuTensors, variant);
            tokenizer = new ParakeetTokenizer(root);

            cs = Resources.Load<ComputeShader>("ComputeShaders/ParakeetCS");
            if (cs == null)
                throw new InvalidOperationException("ParakeetCS.compute missing from Resources/ComputeShaders.");
            kConv = cs.FindKernel("Conv2dSub");
            kPw = cs.FindKernel("Pointwise2d");
            kFlat = cs.FindKernel("FlattenSub");
            kLin = cs.FindKernel("LinearBias");
            kLn = cs.FindKernel("LayerNormT");
            kAdd = cs.FindKernel("AddScaled");
            kGlu = cs.FindKernel("GLU");
            kDwBn = cs.FindKernel("DepthwiseConvBnSilu");
            kAtt = cs.FindKernel("RelPosAttention");

            // pre-cache every CPU-side tensor once, off the main thread — afterwards all
            // ParakeetTensors.D() calls are cache hits (no cross-thread dict mutation)
            cpuPreload = Task.Run(() =>
            {
                string[] names = {
                    "frontend/mel_filters", "dec/embedding", "dec/pred_proj.w", "dec/pred_proj.b",
                    "joint/head.w", "joint/head.b",
                    "dec/lstm.wih0", "dec/lstm.whh0", "dec/lstm.bih0", "dec/lstm.bhh0",
                    "dec/lstm.wih1", "dec/lstm.whh1", "dec/lstm.bih1", "dec/lstm.bhh1" };
                foreach (string n in names) cpuTensors.D(n);
            });
        }

        protected override void StartPrefetch(long bytesPerFrame)
        {
            weights.BudgetBytesPerFrame = bytesPerFrame;
            weights.BeginLoad();
        }

        public override void Defetch(DefetchMode mode)
            => weights.Defetch(mode == DefetchMode.Slow ? LLM.UploadBudgetBytes : 0);

        public override IEnumerator Warmup()
        {
            while (!IsReady) yield return null;
            if (warmedUp) yield break;
            // half a second of silence through the full pipeline compiles/pre-runs every kernel
            var done = false;
            DeepUnityDispatcher.Run(Transcribe(new float[InputSampleRate / 2], _ => done = true));
            while (!done) yield return null;
            warmedUp = true;
        }
        bool warmedUp;

        /// <summary>Transcribe a 16 kHz mono utterance. onTranscript receives the final text
        /// ("" for empty/too-short audio, null on internal failure).</summary>
        public override IEnumerator Transcribe(float[] samples, Action<string> onTranscript)
        {
            while (!IsReady) yield return null;
            while (busy) yield return null;     // one utterance at a time (shared scratch)
            busy = true;
            try
            {
                int tMel = ParakeetConfig.MelFrames(samples?.Length ?? 0);
                if (tMel < 8) { onTranscript?.Invoke(""); yield break; }
                int tEnc = ParakeetConfig.EncFrames(tMel);

                // ---- CPU mel (+ pos emb) off the main thread
                float[] mel = null, posEmb = null;
                var melTask = Task.Run(() => { mel = cpu.Mel(samples, out _); posEmb = cpu.PosEmb(tEnc); });
                while (!melTask.IsCompleted) yield return null;
                if (melTask.IsFaulted) { Debug.LogException(melTask.Exception); onTranscript?.Invoke(null); yield break; }

                // ---- GPU encoder
                float[] encProj = null;
                foreach (var step in EncodeOnGpu(mel, tMel, tEnc, posEmb, r => encProj = r))
                    yield return step;
                if (encProj == null) { onTranscript?.Invoke(null); yield break; }

                // ---- CPU TDT greedy decode + detokenize, off the main thread
                string text = null;
                var decTask = Task.Run(() => text = tokenizer.Decode(cpu.Decode(encProj, tEnc).Tokens));
                while (!decTask.IsCompleted) yield return null;
                if (decTask.IsFaulted) { Debug.LogException(decTask.Exception); onTranscript?.Invoke(null); yield break; }

                onTranscript?.Invoke(text);
            }
            finally { busy = false; }
        }

        // ==================================================================== GPU pipeline
        IEnumerable EncodeOnGpu(float[] mel, int tMel, int tEnc, float[] posEmb, Action<float[]> onProj)
        {
            const int D = ParakeetConfig.Dim, F = ParakeetConfig.FfnDim, C = ParakeetConfig.SubChannels;
            int t1 = (tMel - 1) / 2 + 1, f1 = ParakeetConfig.NMels / 2;
            int t2 = (t1 - 1) / 2 + 1, f2 = f1 / 2;
            int t3 = (t2 - 1) / 2 + 1, f3 = f2 / 2;          // t3 == tEnc, f3 == 16
            int posLen = 2 * tEnc - 1;

            var bMel = new ComputeBuffer(tMel * ParakeetConfig.NMels, 4);
            var bC0 = new ComputeBuffer(C * t1 * f1, 4);      // the big one (~50 MB @ 15 s)
            var bC1 = new ComputeBuffer(C * t2 * f2, 4);
            var bC2 = new ComputeBuffer(C * t3 * f3, 4);
            var bX = new ComputeBuffer(tEnc * D, 4);          // residual stream
            var bN = new ComputeBuffer(tEnc * D, 4);          // post-norm scratch
            var bH = new ComputeBuffer(tEnc * F, 4);          // ffn / glu-in scratch (largest [T,C])
            var bH2 = new ComputeBuffer(tEnc * D, 4);
            var bQ = new ComputeBuffer(tEnc * D, 4);
            var bK = new ComputeBuffer(tEnc * D, 4);
            var bV = new ComputeBuffer(tEnc * D, 4);
            var bPos = new ComputeBuffer(posLen * D, 4);
            var bP = new ComputeBuffer(posLen * D, 4);
            var bAtt = new ComputeBuffer(tEnc * D, 4);
            var bProj = new ComputeBuffer(tEnc * ParakeetConfig.PredDim, 4);
            try
            {
                bMel.SetData(mel);
                bPos.SetData(posEmb);
                cs.SetFloat("norm_eps", ParakeetConfig.LnEps);
                cs.SetFloat("att_scale", 1f / Mathf.Sqrt(ParakeetConfig.HeadDim));
                cs.SetInt("num_heads", ParakeetConfig.Heads);
                cs.SetInt("head_dim", ParakeetConfig.HeadDim);
                cs.SetInt("conv_kernel", ParakeetConfig.ConvKernel);

                // ---- subsampling (SPEC §2)
                Conv2d("sub/conv0", bMel, bC0, 1, C, tMel, ParakeetConfig.NMels, t1, f1, false, act: 2);
                Conv2d("sub/conv1_dw", bC0, bC1, C, C, t1, f1, t2, f2, true, act: 0);
                Pointwise("sub/conv1_pw", bC1, bC0, C, t2 * f2, act: 2);       // reuse bC0 as scratch
                Conv2d("sub/conv2_dw", bC0, bC1, C, C, t2, f2, t3, f3, true, act: 0);
                Pointwise("sub/conv2_pw", bC1, bC2, C, t3 * f3, act: 2);
                cs.SetInt("t_out", t3); cs.SetInt("f_out", f3); cs.SetInt("in_dim", C);
                cs.SetBuffer(kFlat, "X", bC2); cs.SetBuffer(kFlat, "Y", bC1);
                cs.Dispatch(kFlat, (t3 * C * f3 + 255) / 256, 1, 1);
                Linear("sub/linear.w", "sub/linear.b", bC1, bX, tEnc, ParakeetConfig.SubFlat, D, act: 0);

                // ---- 24 conformer blocks (SPEC §4) — mirrors ParakeetCPU.EncoderLayer
                for (int l = 0; l < ParakeetConfig.Layers; l++)
                {
                    string p = $"layer_{l}/";
                    // FF1 (half residual)
                    LayerNorm(p + "ff1.ln", bX, bN, tEnc, D);
                    Linear(p + "ff1.lin1.w", null, bN, bH, tEnc, D, F, act: 1);
                    Linear(p + "ff1.lin2.w", null, bH, bH2, tEnc, F, D, act: 0);
                    Add(bH2, bX, tEnc * D, 0.5f);
                    // rel-pos MHSA
                    LayerNorm(p + "attn.ln", bX, bN, tEnc, D);
                    Linear(p + "attn.q.w", null, bN, bQ, tEnc, D, D, act: 0);
                    Linear(p + "attn.k.w", null, bN, bK, tEnc, D, D, act: 0);
                    Linear(p + "attn.v.w", null, bN, bV, tEnc, D, D, act: 0);
                    Linear(p + "attn.pos.w", null, bPos, bP, posLen, D, D, act: 0);
                    cs.SetInt("seq_len", tEnc);
                    cs.SetBuffer(kAtt, "Q", bQ); cs.SetBuffer(kAtt, "K", bK); cs.SetBuffer(kAtt, "V", bV);
                    cs.SetBuffer(kAtt, "P", bP);
                    cs.SetBuffer(kAtt, "pos_bias_u", weights.Get(p + "attn.bias_u"));
                    cs.SetBuffer(kAtt, "pos_bias_v", weights.Get(p + "attn.bias_v"));
                    cs.SetBuffer(kAtt, "AttendedValues", bAtt);
                    cs.Dispatch(kAtt, tEnc, ParakeetConfig.Heads, 1);
                    Linear(p + "attn.o.w", null, bAtt, bH2, tEnc, D, D, act: 0);
                    Add(bH2, bX, tEnc * D, 1f);
                    // conv module
                    LayerNorm(p + "conv.ln", bX, bN, tEnc, D);
                    Linear(p + "conv.pw1.w", null, bN, bH, tEnc, D, 2 * D, act: 0);
                    cs.SetInt("seq_len", tEnc); cs.SetInt("norm_dim", D);
                    cs.SetBuffer(kGlu, "X", bH); cs.SetBuffer(kGlu, "Y", bH2);
                    cs.Dispatch(kGlu, (tEnc * D + 255) / 256, 1, 1);
                    cs.SetBuffer(kDwBn, "W", weights.Get(p + "conv.dw.w"));
                    cs.SetBuffer(kDwBn, "bn_scale", weights.Get(p + "conv.bn.scale"));
                    cs.SetBuffer(kDwBn, "bn_shift", weights.Get(p + "conv.bn.shift"));
                    cs.SetBuffer(kDwBn, "X", bH2); cs.SetBuffer(kDwBn, "Y", bN);
                    cs.Dispatch(kDwBn, (tEnc * D + 255) / 256, 1, 1);
                    Linear(p + "conv.pw2.w", null, bN, bH2, tEnc, D, D, act: 0);
                    Add(bH2, bX, tEnc * D, 1f);
                    // FF2 (half residual)
                    LayerNorm(p + "ff2.ln", bX, bN, tEnc, D);
                    Linear(p + "ff2.lin1.w", null, bN, bH, tEnc, D, F, act: 1);
                    Linear(p + "ff2.lin2.w", null, bH, bH2, tEnc, F, D, act: 0);
                    Add(bH2, bX, tEnc * D, 0.5f);
                    // per-block final norm (in place via bN copy-back)
                    LayerNorm(p + "out_ln", bX, bN, tEnc, D);
                    (bX, bN) = (bN, bX);                      // swap: bX holds the block output
                }

                // ---- enc_proj (the CPU decode loop's input)
                Linear("joint/enc_proj.w", "joint/enc_proj.b", bX, bProj, tEnc, D, ParakeetConfig.PredDim, act: 0);

                var req = AsyncGPUReadback.Request(bProj);
                while (!req.done) yield return null;
                if (req.hasError)
                {   // fallback: synchronous stall readback
                    float[] sync = new float[tEnc * ParakeetConfig.PredDim];
                    bProj.GetData(sync);
                    onProj(sync);
                }
                else
                    onProj(req.GetData<float>().ToArray());
            }
            finally
            {
                bMel.Release(); bC0.Release(); bC1.Release(); bC2.Release();
                bX.Release(); bN.Release(); bH.Release(); bH2.Release();
                bQ.Release(); bK.Release(); bV.Release(); bPos.Release(); bP.Release();
                bAtt.Release(); bProj.Release();
            }
        }

        void Conv2d(string w, ComputeBuffer x, ComputeBuffer y, int cin, int cout,
                    int tin, int fin, int tout, int fout, bool depthwise, int act)
        {
            cs.SetInt("in_dim", cin); cs.SetInt("out_dim", cout);
            cs.SetInt("t_in", tin); cs.SetInt("f_in", fin);
            cs.SetInt("t_out", tout); cs.SetInt("f_out", fout);
            cs.SetInt("is_depthwise", depthwise ? 1 : 0);
            cs.SetInt("activation_type", act);
            cs.SetBuffer(kConv, "W", weights.Get(w + ".w"));
            cs.SetBuffer(kConv, "W_bias", weights.Get(w + ".b"));
            cs.SetBuffer(kConv, "X", x); cs.SetBuffer(kConv, "Y", y);
            cs.Dispatch(kConv, (cout * tout * fout + 255) / 256, 1, 1);
        }

        void Pointwise(string w, ComputeBuffer x, ComputeBuffer y, int c, int tf, int act)
        {
            cs.SetInt("in_dim", c); cs.SetInt("out_dim", c); cs.SetInt("t_in", tf);
            cs.SetInt("activation_type", act);
            cs.SetBuffer(kPw, "W", weights.Get(w + ".w"));
            cs.SetBuffer(kPw, "W_bias", weights.Get(w + ".b"));
            cs.SetBuffer(kPw, "X", x); cs.SetBuffer(kPw, "Y", y);
            cs.Dispatch(kPw, (c * tf + 255) / 256, 1, 1);
        }

        void Linear(string w, string b, ComputeBuffer x, ComputeBuffer y, int T, int cin, int cout, int act)
        {
            cs.SetInt("seq_len", T); cs.SetInt("in_dim", cin); cs.SetInt("out_dim", cout);
            cs.SetInt("activation_type", act); cs.SetInt("has_bias", b != null ? 1 : 0);
            cs.SetBuffer(kLin, "W", weights.Get(w));
            cs.SetBuffer(kLin, "W_bias", b != null ? weights.Get(b) : weights.Get(w));
            cs.SetBuffer(kLin, "X", x); cs.SetBuffer(kLin, "Y", y);
            cs.Dispatch(kLin, 1, (T + 7) / 8, (cout + 31) / 32);
        }

        void LayerNorm(string w, ComputeBuffer x, ComputeBuffer y, int T, int C)
        {
            cs.SetInt("seq_len", T); cs.SetInt("norm_dim", C);
            cs.SetBuffer(kLn, "ln_gamma", weights.Get(w + ".w"));
            cs.SetBuffer(kLn, "ln_beta", weights.Get(w + ".b"));
            cs.SetBuffer(kLn, "norm_input", x); cs.SetBuffer(kLn, "norm_output", y);
            cs.Dispatch(kLn, (T + 255) / 256, 1, 1);
        }

        void Add(ComputeBuffer src, ComputeBuffer dst, int n, float scale)
        {
            cs.SetInt("seq_len", (n + ParakeetConfig.Dim - 1) / ParakeetConfig.Dim);
            cs.SetInt("norm_dim", ParakeetConfig.Dim);
            cs.SetFloat("residual_scale", scale);
            cs.SetBuffer(kAdd, "X", src); cs.SetBuffer(kAdd, "inout_buf", dst);
            cs.Dispatch(kAdd, (n + 255) / 256, 1, 1);
        }

        public override void Release()
        {
            weights.Dispose();
        }
    }
}
