using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    using QwenASRModeling;

    /// <summary>
    /// Qwen3-ASR speech-to-text (0.6B / 1.7B, Apache-2.0). Push-to-talk transcription of 16 kHz
    /// mono utterances (SPEC.md; parity chain validated in validation/harness — ALL GATES PASS).
    ///
    ///     var stt = new QwenASRSTT(QwenASRSize.B0_6);
    ///     StartCoroutine(stt.Warmup());
    ///     ...
    ///     StartCoroutine(stt.Transcribe(pcm16k, text => Debug.Log(text)));
    ///
    /// Optional: <see cref="Language"/> forces the output language (skips language detection via
    /// the "language X&lt;asr_text&gt;" assistant prefill); <see cref="Context"/> injects free
    /// biasing text (names, jargon) into the system slot — both per SPEC §5.
    /// </summary>
    public class QwenASRSTT : STT
    {
        public override int InputSampleRate => QwenASRConfig.SAMPLE_RATE;

        /// <summary>Forced output language (full name, e.g. "English", "Romanian") or null = auto-detect.</summary>
        public string Language;
        /// <summary>Context-injection string (vocabulary/name biasing), empty = none.</summary>
        public string Context = "";
        /// <summary>Generation cap per utterance (~40 tokens ≈ 15 s of speech; 128 is generous).</summary>
        public int MaxNewTokens = 128;

        readonly QwenASRModel model;
        readonly QwenASRTokenizer tokenizer;
        readonly QwenASRSize size;
        bool released;

        public override bool IsReady => model.IsReady;
        public override ModelResidency Residency => model.weights.Residency;
        public override long TotalWeightBytes => model.weights.BytesTotal;
        public override long UploadedWeightBytes => model.weights.BytesUploaded;
        public override long LoadBudgetBytesPerFrame
        {
            get => model.weights.BudgetBytesPerFrame;
            set => model.weights.BudgetBytesPerFrame = value;
        }

        public QwenASRSTT(QwenASRSize size = QwenASRSize.B0_6, string paramsPath = null, int cacheCapacity = 1024)
        {
            this.size = size;
            QwenASRConfig.ApplySize(size);
            paramsPath ??= ResolveParamsDir(size);
            model = new QwenASRModel(paramsPath, cacheCapacity);
            tokenizer = new QwenASRTokenizer(System.IO.Path.Combine(paramsPath, "tokenizer"));
        }

        static string ResolveParamsDir(QwenASRSize size)
        {
            string dir = $"weights_qwen3asr_{QwenASRConfig.SizeLabel(size)}_fp16";
            string res = $"Assets/Resources/Weights/{dir}";
            return System.IO.Directory.Exists(res) ? res
                 : $"Assets/DeepUnity/InferenceEngine/STT/QwenASR/{dir}";   // legacy fallback, mirrors LLM.ResolveParamsDir
        }

        protected override void StartPrefetch(long bytesPerFrame)
        {
            model.weights.BudgetBytesPerFrame = bytesPerFrame;
            model.weights.BeginLoad();
        }

        public override void Defetch(DefetchMode mode)
            => model.weights.Defetch(mode == DefetchMode.Slow ? Math.Max(LoadBudgetBytesPerFrame, 64 * 1024) : 0);

        /// <summary>Transcribe a mono 16 kHz utterance. onTranscript receives the parsed transcript
        /// (empty string for silent audio, null on failure).</summary>
        public override IEnumerator Transcribe(float[] samples, Action<string> onTranscript)
        {
            if (released) { onTranscript?.Invoke(null); yield break; }
            if (samples == null || samples.Length == 0) { onTranscript?.Invoke(""); yield break; }
            while (!IsReady) yield return null;

            // §1-§3: mel -> encoder -> projector (fills model.projBuf with [nTokens, hidden])
            int[] nTok = new int[1];
            var enc = model.EncodeAudioYielding(samples, nTok);
            while (enc.MoveNext()) yield return enc.Current;

            // §5: prompt scaffold — context in the system slot; forced language via assistant prefill
            string prefix = string.IsNullOrEmpty(Language) ? null : $"language {Language}";
            int[] promptIds = QwenASRCPU.BuildPromptIds(tokenizer, nTok[0], Context ?? "", prefix);
            int audioPadStart = Array.IndexOf(promptIds, QwenASRConfig.AUDIO_PAD_TOKEN_ID);

            // §6: greedy decode
            model.ResetCache();
            var generated = new List<int>(MaxNewTokens);
            var dec = model.GreedyDecodeYielding(promptIds, audioPadStart, nTok[0], generated, MaxNewTokens);
            while (dec.MoveNext()) yield return dec.Current;

            // parse "language X<asr_text>{transcript}" (+ the reference repetition post-fix)
            string transcript = QwenASRCPU.ParseTranscript(tokenizer, generated);
            onTranscript?.Invoke(FixRepetitions(transcript));
        }

        // Reference post-processing (qwen_asr utils._detect_and_fix_repetitions, threshold 20):
        // collapse >threshold identical consecutive chars; longer pattern loops are rare in
        // greedy PTT output and are additionally capped by MaxNewTokens.
        static string FixRepetitions(string s, int threshold = 20)
        {
            if (string.IsNullOrEmpty(s)) return s;
            var sb = new System.Text.StringBuilder(s.Length);
            int i = 0;
            while (i < s.Length)
            {
                int run = 1;
                while (i + run < s.Length && s[i + run] == s[i]) run++;
                sb.Append(s[i], run > threshold ? 1 : run);
                i += run;
            }
            return sb.ToString();
        }

        bool _warmedUp;

        /// <summary>Waits for weights + tables, then runs one throwaway 0.5 s transcription so
        /// every kernel's driver ISA compile happens behind the loading screen. Idempotent.</summary>
        public override IEnumerator Warmup()
        {
            if (_warmedUp) yield break;
            while (!IsReady) yield return null;
            var t = Transcribe(new float[QwenASRConfig.MIN_SAMPLES], _ => { });
            while (t.MoveNext()) yield return t.Current;
            model.ResetCache();
            _warmedUp = true;
        }

        public override void Release()
        {
            if (released) return;
            released = true;
            model.Dispose();
        }
    }
}
