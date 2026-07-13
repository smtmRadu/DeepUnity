using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;

namespace DeepUnity
{
    // Chatterbox-Turbo text-to-speech for DeepUnity — full-GPU, fp16 weights, 24kHz output.
    // Mirrors the LLM API surface (ctor -> Prewarm/Warmup -> coroutine inference -> Release).
    //
    //   var tts = new ChatterboxTTS();
    //   yield return tts.Warmup();                       // behind a loading screen
    //   yield return tts.Speak("Hello world!", clip => audioSource.PlayOneShot(clip));
    //
    // Speak() runs the reference offline pipeline (SPEC.md §1): punc_norm -> GPT2 BPE ->
    // T3 speech tokens (streamed per token via onSpeechToken) -> S3Gen meanflow (2 estimator
    // passes) -> HiFT vocoder -> AudioClip. The per-token callback is the tap for the future
    // streaming mode (chunked S3Gen); v1 synthesizes the full utterance.
    // Voice: the baked default voice from conds.pt (exported into the weights folder).
    public class ChatterboxTTS
    {
        readonly ChatterboxModeling.ChatterboxWeights weights;
        readonly ChatterboxModeling.T3Model t3;
        readonly ChatterboxModeling.S3GenModel s3gen;
        readonly ChatterboxModeling.ChatterboxTokenizer tokenizer;

        float[] speakerEmb;      // <voice>/t3_speaker_emb [256]
        int[] t3PromptTokens;    // <voice>/t3_prompt_tokens [375]
        bool condsLoaded;
        bool prefixBuilt;        // T3 cond-prefix KV computed (reused across utterances)
        readonly string voice;   // conds subfolder in the manifest ("conds" = baked default;
                                 // alternative voices exported by validation/make_voice.py)

        public bool IsReady => weights.IsReady && tokenizer.IsReady;
        /// <summary>Speech tokens per second during T3 decode (25 tokens ≈ 1s of audio).</summary>
        public float TokensPerSecond { get; private set; }
        public const int SampleRate = ChatterboxModeling.ChatterboxConfig.SAMPLE_RATE;

        public ChatterboxTTS(
            string params_path = null,       // null resolves Resources-first (import_params.py convention)
            string tokenizer_path = "Assets/DeepUnity/InferenceEngine/TTS/Chatterbox/ChatterboxTokenizer",
            int maxContextLength = 2048,
            string voice = "conds",          // e.g. "conds_elder" after make_voice.py
            LLMQuant quantization = LLMQuant.FP16)  // INT8 = T3 matmuls int8 (~300 MB less), s3gen stays fp16
        {
            params_path ??= ResolveParamsDir(quantization);
            weights = new ChatterboxModeling.ChatterboxWeights(params_path);
            // fall back to the baked default when the requested voice isn't in the manifest
            // (alternative voices are added by validation/make_voice.py)
            if (voice != "conds" && !weights.Has($"{voice}/t3_speaker_emb"))
            {
                ConsoleMessage.Warning($"ChatterboxTTS: voice '{voice}' not found in the weights manifest — " +
                                       "using the default voice. Bake it with validation/make_voice.py.");
                voice = "conds";
            }
            this.voice = voice;
            tokenizer = new ChatterboxModeling.ChatterboxTokenizer(tokenizer_path);
            t3 = new ChatterboxModeling.T3Model(weights, maxContextLength);
            s3gen = new ChatterboxModeling.S3GenModel(weights) { CondsPrefix = voice };

#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged += OnPlayModeChanged;
#endif
        }

#if UNITY_EDITOR
        void OnPlayModeChanged(UnityEditor.PlayModeStateChange state)
        {
            if (state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
                Release();
        }
#endif

        static string ResolveParamsDir(LLMQuant quant)
        {
            string q = quant == LLMQuant.INT8 ? "int8" : quant == LLMQuant.INT4 ? "int4" : "fp16";
            string res = DeepUnityMeta.ResolvePath($"Assets/Resources/Weights/weights_chatterbox_turbo_{q}");   // player builds: StreamingAssets
            return System.IO.Directory.Exists(res) ? res
                 : DeepUnityMeta.ResolvePath($"Assets/DeepUnity/InferenceEngine/TTS/Chatterbox/weights_chatterbox_turbo_{q}");
        }

        void LoadConds()
        {
            if (condsLoaded) return;
            speakerEmb = weights.ReadFloats($"{voice}/t3_speaker_emb");
            float[] pt = weights.ReadFloats($"{voice}/t3_prompt_tokens");
            t3PromptTokens = new int[pt.Length];
            for (int i = 0; i < pt.Length; i++) t3PromptTokens[i] = (int)pt[i];
            t3.SetSpeakerEmbedding(speakerEmb);
            condsLoaded = true;
        }

        /// <summary>Waits for the weight stream and runs a throwaway forward so the first real
        /// Speak() call is hitch-free. Idempotent.</summary>
        public IEnumerator Warmup()
        {
            while (!IsReady) yield return null;
            LoadConds();
            yield return null;
        }

        /// <summary>
        /// Full text -> speech. Streams T3 speech tokens through <paramref name="onSpeechToken"/>
        /// (25Hz, the real-time tap), then synthesizes the waveform and returns it as an
        /// <see cref="AudioClip"/> via <paramref name="onClip"/>.
        /// Defaults = the turbo reference sampling config.
        /// </summary>
        public IEnumerator Speak(
            string text,
            Action<AudioClip> onClip,
            Action<int> onSpeechToken = null,
            float temperature = ChatterboxModeling.ChatterboxConfig.DEFAULT_TEMPERATURE,
            int top_k = ChatterboxModeling.ChatterboxConfig.DEFAULT_TOP_K,
            float top_p = ChatterboxModeling.ChatterboxConfig.DEFAULT_TOP_P,
            float repetition_penalty = ChatterboxModeling.ChatterboxConfig.DEFAULT_REPETITION_PENALTY,
            int max_speech_tokens = ChatterboxModeling.ChatterboxConfig.MAX_SPEECH_TOKENS)
        {
            float[] wav = null;
            var e = Synthesize(text, w => wav = w, onSpeechToken, temperature, top_k, top_p,
                               repetition_penalty, max_speech_tokens);
            while (e.MoveNext()) yield return e.Current;
            if (wav == null) { onClip?.Invoke(null); yield break; }

            AudioClip clip = AudioClip.Create("ChatterboxTTS", wav.Length, 1, SampleRate, false);
            clip.SetData(wav, 0);
            onClip?.Invoke(clip);
        }

        /// <summary>Same as <see cref="Speak"/> but returns the raw 24kHz mono samples.</summary>
        public IEnumerator Synthesize(
            string text,
            Action<float[]> onWav,
            Action<int> onSpeechToken = null,
            float temperature = ChatterboxModeling.ChatterboxConfig.DEFAULT_TEMPERATURE,
            int top_k = ChatterboxModeling.ChatterboxConfig.DEFAULT_TOP_K,
            float top_p = ChatterboxModeling.ChatterboxConfig.DEFAULT_TOP_P,
            float repetition_penalty = ChatterboxModeling.ChatterboxConfig.DEFAULT_REPETITION_PENALTY,
            int max_speech_tokens = ChatterboxModeling.ChatterboxConfig.MAX_SPEECH_TOKENS)
        {
            var tokens = new List<int>();
            var g = GenerateSpeechTokens(text, tokens, onSpeechToken, temperature, top_k, top_p,
                                         repetition_penalty, max_speech_tokens, syncSample: false);
            while (g.MoveNext()) yield return g.Current;
            var s = SynthesizeFromTokens(tokens, onWav);
            while (s.MoveNext()) yield return s.Current;
        }

        /// <summary>
        /// STAGE 1 (T3): text -> speech tokens into <paramref name="outTokens"/> (already filtered
        /// to flow vocab). Yields once per decode step, so a budget-pump can advance it several
        /// tokens per frame. syncSample=true samples with a blocking 4-byte readback (fps-decoupled,
        /// for the real-time pump); false uses async readback (one frame per token, gentle).
        /// The text-independent cond prefix is prefilled once and its KV reused across calls.
        /// </summary>
        public IEnumerator GenerateSpeechTokens(
            string text, List<int> outTokens, Action<int> onSpeechToken = null,
            float temperature = ChatterboxModeling.ChatterboxConfig.DEFAULT_TEMPERATURE,
            int top_k = ChatterboxModeling.ChatterboxConfig.DEFAULT_TOP_K,
            float top_p = ChatterboxModeling.ChatterboxConfig.DEFAULT_TOP_P,
            float repetition_penalty = ChatterboxModeling.ChatterboxConfig.DEFAULT_REPETITION_PENALTY,
            int max_speech_tokens = ChatterboxModeling.ChatterboxConfig.MAX_SPEECH_TOKENS,
            bool syncSample = true)
        {
            while (!IsReady) yield return null;
            LoadConds();
            LLM.CurrentPhase = "tts-t3";

            int[] textTokens = tokenizer.Encode(ChatterboxModeling.ChatterboxTokenizer.PuncNorm(text));

            if (!prefixBuilt)
            {
                t3.ResetCache();
                int condLen = t3.BuildCondEmbeds(t3PromptTokens);
                var cf = t3.PrefillYielding(condLen);
                while (cf.MoveNext()) yield return cf.Current;
                t3.SavePrefix();
                prefixBuilt = true;
            }
            else t3.RestoreToPrefix();

            int segLen = t3.BuildTextEmbeds(textTokens, t3.PrefixTokenCount);
            var pf = t3.PrefillYielding(segLen);
            while (pf.MoveNext()) yield return pf.Current;

            // autoregressive decode (inference_turbo semantics): first sample's repetition-penalty
            // domain = {BOS}; afterwards = the generated tokens.
            var uniq = new HashSet<int>();
            int[] genArr = new int[max_speech_tokens + 1];
            int[] sampled = new int[1];

            genArr[0] = ChatterboxModeling.ChatterboxConfig.START_SPEECH_TOKEN;
            if (syncSample) sampled[0] = t3.SampleNow(genArr, 1, temperature, top_k, top_p, repetition_penalty);
            else { var s0 = t3.SampleYielding(genArr, 1, temperature, top_k, top_p, repetition_penalty, sampled);
                   while (s0.MoveNext()) yield return s0.Current; }

            int token = sampled[0];
            int produced = 0;
            var sw = Stopwatch.StartNew();
            while (produced < max_speech_tokens)
            {
                if (token == ChatterboxModeling.ChatterboxConfig.STOP_SPEECH_TOKEN) break;
                produced++;
                if (token < ChatterboxModeling.ChatterboxConfig.FLOW_VOCAB) outTokens.Add(token);
                if (uniq.Add(token)) genArr[uniq.Count - 1] = token;
                onSpeechToken?.Invoke(token);

                var d = t3.DecodeStepYielding(token);
                while (d.MoveNext()) yield return d.Current;
                if (syncSample) token = t3.SampleNow(genArr, uniq.Count, temperature, top_k, top_p, repetition_penalty);
                else { var sm = t3.SampleYielding(genArr, uniq.Count, temperature, top_k, top_p, repetition_penalty, sampled);
                       while (sm.MoveNext()) yield return sm.Current; token = sampled[0]; }

                TokensPerSecond = produced / Mathf.Max((float)sw.Elapsed.TotalSeconds, 1e-3f);
                yield return null;   // ONE yield per token — the pump decides how many per frame
            }
            TokensPerSecond = 0f;
            LLM.CurrentPhase = "idle";
        }

        /// <summary>STAGE 2 (S3Gen): speech tokens -> 24kHz samples. Pump-able (internal yields);
        /// runs fully on the GPU and can be in flight WHILE stage 1 decodes the next chunk.</summary>
        public IEnumerator SynthesizeFromTokens(List<int> flowTokens, Action<float[]> onWav)
        {
            if (flowTokens == null || flowTokens.Count == 0)
            {
                ConsoleMessage.Warning("ChatterboxTTS: no valid speech tokens to synthesize.");
                onWav?.Invoke(null);
                yield break;
            }
            LLM.CurrentPhase = "tts-s3gen";
            float[] result = null;
            var syn = s3gen.SynthesizeYielding(flowTokens.ToArray(), w => result = w);
            while (syn.MoveNext()) yield return syn.Current;
            LLM.CurrentPhase = "idle";
            onWav?.Invoke(result);
        }

        public void Release()
        {
            t3?.Dispose();
            s3gen?.Dispose();
            weights?.Dispose();
#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged -= OnPlayModeChanged;
#endif
            ConsoleMessage.Info("ChatterboxTTS released from GPU");
        }
    }
}
