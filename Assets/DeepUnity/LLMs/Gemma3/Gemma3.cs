using System;
using System.Collections;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Assertions;

namespace DeepUnity
{
    // Default Gemma3 implementation in DeepUnity. FP16 weights, full-GPU inference
    // (every kernel reads/writes ComputeBuffers — no per-step CPU<->GPU bounces).
    // System-prompt KV cache is persisted as packed FP16 .bin under Assets/Resources/Cache/.
    // Supersedes the earlier hybrid GPU/CPU `Gemma3OriginalForCausalLM`.
    public class Gemma3ForCausalLM
    {
        private string path;
        public Gemma3Modeling.Gemma3Model model;
        public Gemma3TokenizerFast tokenizer;
        private bool isFreshlyInitialized;

        public bool IsReady => model.IsReady && tokenizer.IsReady;
        public float TokensPerSecond { get; private set; }

        public Gemma3ForCausalLM(
            string params_path = "Assets/DeepUnity/LLMs/Gemma3/params_it",
            string tokenizer_path = "Assets/DeepUnity/LLMs/Gemma3/Gemma3TokenizerFast.json",
            int cacheCapacity = 2048)
        {
            this.path = params_path;
            this.tokenizer = new Gemma3TokenizerFast(tokenizer_path, load_async: true);

#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged += OnPlayModeChanged;
#endif
            Stopwatch sw = Stopwatch.StartNew();
            model = new Gemma3Modeling.Gemma3Model(params_path, cacheCapacity);
            ConsoleMessage.Info($"Gemma3 model created ({sw.Elapsed.TotalSeconds:0.00} s)");
        }

        ~Gemma3ForCausalLM()
        {
            model?.Dispose();
            ConsoleMessage.Info("Gemma3 released from GPU");
        }

#if UNITY_EDITOR
        private void OnPlayModeChanged(UnityEditor.PlayModeStateChange state)
        {
            if (state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
            {
                model?.Dispose();
                ConsoleMessage.Info("Gemma3 released from GPU");
            }
        }
#endif

        public int ParameterCount()
        {
            int p = Gemma3Modeling.Gemma3Config.VOCAB_SIZE * Gemma3Modeling.Gemma3Config.HIDDEN_SIZE;
            int H = Gemma3Modeling.Gemma3Config.HIDDEN_SIZE;
            int D = Gemma3Modeling.Gemma3Config.HEAD_DIM;
            float exp = Gemma3Modeling.Gemma3Config.ATTN_EXPANSION_FACTOR;
            int innerEmb = (int)(H * exp);
            int Hq = Gemma3Modeling.Gemma3Config.HEADS_Q;
            int Hkv = Gemma3Modeling.Gemma3Config.HEADS_KV;
            int I = Gemma3Modeling.Gemma3Config.MLP_INTERMEDIATE_SIZE;

            int perLayer = 0;
            perLayer += H * innerEmb;
            perLayer += H * innerEmb * Hkv / Hq;
            perLayer += H * innerEmb * Hkv / Hq;
            perLayer += innerEmb * H;
            perLayer += D + D;
            perLayer += H * I * 3;
            perLayer += H * 4;

            p += perLayer * Gemma3Modeling.Gemma3Config.NUM_LAYERS;
            p += H;
            return p;
        }

        public Tensor Predict(Tensor input_ids, Tensor attn_mask = null)
        {
            if (!IsReady)
                throw new Exception("Gemma3 is not ready. Check IsReady first.");
            int seqLen = input_ids.Size(-1);
            model.Forward(input_ids, useCache: false, lastPosOnly: false);
            return model.ReadLogits(seqLen);
        }

        public IEnumerator Generate(Tensor input_ids, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = -1, float top_p = 1f, float min_p = 0f)
        {
            while (!IsReady) yield return new WaitForSeconds(0.01f);

            model.ResetCache();

            var e = model.ForwardYielding(input_ids, useCache: true, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            int tokenId = model.Sample(temperature, top_k, top_p, min_p);
            string tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
            onTokenGenerated?.Invoke(tokenStr);
            yield return null;

            for (int t = 0; t < max_new_tokens - 1; t++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                Tensor nextInput = Tensor.Constant(tokenId);
                e = model.ForwardYielding(nextInput, useCache: true, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;

                tokenId = model.Sample(temperature, top_k, top_p, min_p);
                if (tokenId == Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID) break;

                tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
                onTokenGenerated?.Invoke(tokenStr);
                TokensPerSecond = sw.ElapsedMilliseconds > 0 ? 1000f / sw.ElapsedMilliseconds : 0f;
                yield return null;
            }

            TokensPerSecond = 0f;
            yield return true;
        }

        public IEnumerator InitializeChat(string system_prompt = "")
        {
            while (!IsReady) yield return new WaitForSeconds(0.01f);
            Assert.AreNotEqual(system_prompt, null);

            model.ResetCache();

            string cacheFolder = SystemPromptCacheFolder(system_prompt);
            bool loaded = false;
            yield return model.cache.TryLoadAsync(cacheFolder, r => loaded = r);
            if (loaded)
            {
                ConsoleMessage.Info($"Gemma3 system prompt cache hit ({model.cache.CachedTokenCount} tokens).");
                isFreshlyInitialized = true;
                yield return true;
                yield break;
            }

            Stopwatch sw = Stopwatch.StartNew();

            (Tensor, Tensor) tok = tokenizer.Encode(system_prompt, add_special_tokens: false, truncation: true, max_length: 2048);
            Tensor prefix = Tensor.Constant(new float[]
            {
                Gemma3TokenizerFast.BOS_TOKEN_ID,
                Gemma3TokenizerFast.START_OF_TURN_TOKEN_ID,
                2364f, 107f
            });
            Tensor input_ids = Tensor.Concat(-1, prefix, tok.Item1);

            var e = model.ForwardYielding(input_ids, useCache: true, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            ConsoleMessage.Info($"Gemma3 system prompt computed ({sw.Elapsed.TotalSeconds:0.00} s).");

            yield return model.cache.SaveAsync(cacheFolder);
            ConsoleMessage.Info($"Gemma3 system prompt cache saved ({model.cache.CachedTokenCount} tokens).");

            isFreshlyInitialized = true;
            yield return true;
        }

        static string SystemPromptCacheFolder(string system_prompt)
        {
            using (var sha = System.Security.Cryptography.SHA256.Create())
            {
                byte[] h = sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(system_prompt ?? string.Empty));
                var sb = new System.Text.StringBuilder(64);
                for (int i = 0; i < h.Length; i++) sb.Append(h[i].ToString("x2"));
                return System.IO.Path.Combine("Assets/Resources/Cache", sb.ToString());
            }
        }

        public IEnumerator Chat(string prompt, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = -1, float top_p = 1f, float min_p = 0f)
        {
            if (!IsReady) throw new Exception("Call InitializeChat before Chat.");

            Tensor prefixT, postfixT;
            if (isFreshlyInitialized)
            {
                isFreshlyInitialized = false;
                prefixT = Tensor.Constant(new float[] { 108f });
                postfixT = Tensor.Constant(new float[]
                {
                    Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID, 107f,
                    Gemma3TokenizerFast.START_OF_TURN_TOKEN_ID, 4368f, 107f
                });
            }
            else
            {
                prefixT = Tensor.Constant(new float[]
                {
                    Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID, 107f,
                    Gemma3TokenizerFast.START_OF_TURN_TOKEN_ID, 2364f, 107f
                });
                postfixT = Tensor.Constant(new float[]
                {
                    Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID, 107f,
                    Gemma3TokenizerFast.START_OF_TURN_TOKEN_ID, 4368f, 107f
                });
            }

            (Tensor, Tensor) tok = tokenizer.Encode(prompt, add_special_tokens: false, truncation: true, max_length: 2048);
            Tensor input_ids = Tensor.Concat(-1, prefixT, tok.Item1, postfixT);

            var e = model.ForwardYielding(input_ids, useCache: true, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            int tokenId = model.Sample(temperature, top_k, top_p, min_p);
            string tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
            onTokenGenerated?.Invoke(tokenStr);
            yield return null;

            for (int t = 0; t < max_new_tokens - 1; t++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                Tensor nextInput = Tensor.Constant(tokenId);
                e = model.ForwardYielding(nextInput, useCache: true, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;

                tokenId = model.Sample(temperature, top_k, top_p, min_p);
                if (tokenId == Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID)
                {
                    ConsoleMessage.Info("Gemma3 ended the response.");
                    break;
                }

                tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
                onTokenGenerated?.Invoke(tokenStr);
                TokensPerSecond = sw.ElapsedMilliseconds > 0 ? 1000f / sw.ElapsedMilliseconds : 0f;
                yield return null;
            }

            TokensPerSecond = 0f;
            yield return true;
        }
    }
}
