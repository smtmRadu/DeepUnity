using System;
using System.Collections;
using System.Diagnostics;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace DeepUnity
{
    public class Gemma3GPUFP32ForCausalLM
    {
        private string path;
        public Gemma3GPUFP32Modeling.Gemma3GPUFP32Model model;
        public Gemma3TokenizerFast tokenizer;
        private bool isFreshlyInitialized;

        public bool IsReady => model.IsReady && tokenizer.IsReady;
        public float TokensPerSecond { get; private set; }

        [System.Obsolete("Gemma3GPUFP32ForCausalLM is deprecated. Use Gemma3GPUFP16ForCausalLM instead.")]
        public Gemma3GPUFP32ForCausalLM(
            string params_path = "Assets/DeepUnity/LLMs/Gemma3/params_it",
            string tokenizer_path = "Assets/DeepUnity/LLMs/Gemma3/Gemma3TokenizerFast.json",
            int cacheCapacity = 8192)
        {
            throw new System.Exception("Gemma3GPUFP32ForCausalLM is deprecated. Use Gemma3GPUFP16ForCausalLM instead.");
            this.path = params_path;
            this.tokenizer = new Gemma3TokenizerFast(tokenizer_path, load_async: true);

#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged += OnPlayModeChanged;
#endif
            Stopwatch sw = Stopwatch.StartNew();
            model = new Gemma3GPUFP32Modeling.Gemma3GPUFP32Model(params_path, cacheCapacity);
            ConsoleMessage.Info($"Gemma3GPUFP32model created ({sw.Elapsed.TotalSeconds:0.00} s)");
        }

        ~Gemma3GPUFP32ForCausalLM()
        {
            model?.Dispose();
            ConsoleMessage.Info("Gemma3GPUFP32released from GPU");
        }

#if UNITY_EDITOR
        private void OnPlayModeChanged(UnityEditor.PlayModeStateChange state)
        {
            if (state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
            {
                model?.Dispose();
                ConsoleMessage.Info("Gemma3GPUFP32released from GPU");
            }
        }
#endif

        public int ParameterCount()
        {
            int p = Gemma3Modeling.Gemma3Config.VOCAB_SIZE * Gemma3Modeling.Gemma3Config.HIDDEN_SIZE; // embed/lm_head (tied)
            int H = Gemma3Modeling.Gemma3Config.HIDDEN_SIZE;
            int D = Gemma3Modeling.Gemma3Config.HEAD_DIM;
            float exp = Gemma3Modeling.Gemma3Config.ATTN_EXPANSION_FACTOR;
            int innerEmb = (int)(H * exp);
            int Hq = Gemma3Modeling.Gemma3Config.HEADS_Q;
            int Hkv = Gemma3Modeling.Gemma3Config.HEADS_KV;
            int I = Gemma3Modeling.Gemma3Config.MLP_INTERMEDIATE_SIZE;

            int perLayer = 0;
            perLayer += H * innerEmb;                          // q_proj
            perLayer += H * innerEmb * Hkv / Hq;              // k_proj
            perLayer += H * innerEmb * Hkv / Hq;              // v_proj
            perLayer += innerEmb * H;                          // o_proj
            perLayer += D + D;                                 // q_norm + k_norm
            perLayer += H * I * 3;                             // mlp (gate+up+down)
            perLayer += H * 4;                                 // 4 layer norms

            p += perLayer * Gemma3Modeling.Gemma3Config.NUM_LAYERS;
            p += H; // final norm
            return p;
        }

        // ---- Predict: stateless forward, returns full logits ----
        public Tensor Predict(Tensor input_ids, Tensor attn_mask = null)
        {
            if (!IsReady)
                throw new Exception("Gemma3GPUFP32is not ready. Check IsReady first.");

            int seqLen = input_ids.Size(-1);
            model.Forward(input_ids, useCache: false, lastPosOnly: false);
            return model.ReadLogits(seqLen);
        }

        // ---- Generate: autoregressive token generation with KV cache ----
        public IEnumerator Generate(Tensor input_ids, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = -1, float top_p = 1f, float min_p = 0f)
        {
            while (!IsReady)
                yield return new WaitForSeconds(0.01f);

            model.ResetCache();

            // prefill
            var e = model.ForwardYielding(input_ids, useCache: true, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            int tokenId = model.Sample(temperature, top_k, top_p, min_p);
            string tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
            onTokenGenerated?.Invoke(tokenStr);
            yield return null;

            // decode loop
            for (int t = 0; t < max_new_tokens - 1; t++)
            {
                Stopwatch sw = Stopwatch.StartNew();

                Tensor nextInput = Tensor.Constant(tokenId);
                e = model.ForwardYielding(nextInput, useCache: true, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;

                tokenId = model.Sample(temperature, top_k, top_p, min_p);

                if (tokenId == Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID)
                    break;

                tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
                onTokenGenerated?.Invoke(tokenStr);
                TokensPerSecond = sw.ElapsedMilliseconds > 0 ? 1000f / sw.ElapsedMilliseconds : 0f;
                yield return null;
            }

            TokensPerSecond = 0f;
            yield return true;
        }

        // ---- InitializeChat: prefill system prompt + build KV cache ----
        public IEnumerator InitializeChat(string system_prompt = "")
        {
            while (!IsReady)
                yield return new WaitForSeconds(0.01f);

            Assert.AreNotEqual(system_prompt, null);

            model.ResetCache();

            Stopwatch sw = Stopwatch.StartNew();

            (Tensor, Tensor) tok = tokenizer.Encode(system_prompt, add_special_tokens: false, truncation: true, max_length: 2048);
            Tensor prefix = Tensor.Constant(new float[]
            {
                Gemma3TokenizerFast.BOS_TOKEN_ID,
                Gemma3TokenizerFast.START_OF_TURN_TOKEN_ID,
                2364f, // "user"
                107f   // "\n"
            });
            Tensor input_ids = Tensor.Concat(-1, prefix, tok.Item1);

            var e = model.ForwardYielding(input_ids, useCache: true, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            ConsoleMessage.Info($"Gemma3GPUFP32system prompt computed ({sw.Elapsed.TotalSeconds:0.00} s).");

            isFreshlyInitialized = true;
            yield return true;
        }

        // ---- Chat: continue conversation with KV cache ----
        public IEnumerator Chat(string prompt, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = -1, float top_p = 1f, float min_p = 0f)
        {
            if (!IsReady)
                throw new Exception("Call InitializeChat before Chat.");

            Tensor prefixT, postfixT;

            if (isFreshlyInitialized)
            {
                isFreshlyInitialized = false;
                prefixT = Tensor.Constant(new float[] { 108f }); // "\n\n"
                postfixT = Tensor.Constant(new float[]
                {
                    Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID,
                    107f,
                    Gemma3TokenizerFast.START_OF_TURN_TOKEN_ID,
                    4368f, // "model"
                    107f   // "\n"
                });
            }
            else
            {
                prefixT = Tensor.Constant(new float[]
                {
                    Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID,
                    107f,
                    Gemma3TokenizerFast.START_OF_TURN_TOKEN_ID,
                    2364f, // "user"
                    107f
                });
                postfixT = Tensor.Constant(new float[]
                {
                    Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID,
                    107f,
                    Gemma3TokenizerFast.START_OF_TURN_TOKEN_ID,
                    4368f, // "model"
                    107f
                });
            }

            (Tensor, Tensor) tok = tokenizer.Encode(prompt, add_special_tokens: false, truncation: true, max_length: 2048);
            Tensor input_ids = Tensor.Concat(-1, prefixT, tok.Item1, postfixT);

            // prefill user prompt
            var e = model.ForwardYielding(input_ids, useCache: true, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            int tokenId = model.Sample(temperature, top_k, top_p, min_p);
            string tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
            onTokenGenerated?.Invoke(tokenStr);
            yield return null;

            // decode loop
            for (int t = 0; t < max_new_tokens - 1; t++)
            {
                Stopwatch sw = Stopwatch.StartNew();

                Tensor nextInput = Tensor.Constant(tokenId);
                e = model.ForwardYielding(nextInput, useCache: true, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;

                tokenId = model.Sample(temperature, top_k, top_p, min_p);

                if (tokenId == Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID)
                {
                    ConsoleMessage.Info("Gemma3GPUFP32ended the response.");
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
