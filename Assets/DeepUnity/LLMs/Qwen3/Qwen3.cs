using DeepUnity.Activations;
using DeepUnity.Gemma3Modelling;
using DeepUnity.Modules;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;


namespace DeepUnity
{
    namespace Qwen3Modeling
    {
        public class Qwen3Model
        {
            private int pad_idx = 0;
            private int vocab_size = 0;
            public Embedding embed_tokens;
            public List<Qwen3Modeling.Qwen3DecoderLayer> layers; 
            private Qwen3RMSNorm norm;
            private RotaryPositionalEmbeddings rope;
            public bool IsInitialized
            {
                get
                {
                    if (!IsEmbeddingInitialized)
                        return false;
                    if (!norm.IsInitialized)
                        return false;

                    foreach (var layer in layers)
                    {
                        if (!layer.self_attn.IsInitialized)
                            return false;
                        if (!layer.mlp.IsInitialized)
                            return false;
                        if (!layer.input_layernorm.IsInitialized)
                            return false;
                        if (!layer.post_attention_layernorm.IsInitialized)
                            return false;
                    }
                    return true;
                }
            }
            private bool IsEmbeddingInitialized { get; set; } = false;

            public Qwen3Model(string params_path, ref ComputeBuffer lm_head)
            {
                this.pad_idx = Qwen3Modeling.Qwen3Config.PAD_IDX;
                this.vocab_size = Qwen3Modeling.Qwen3Config.VOCAB_SIZE;
                this.embed_tokens = new Embedding(
                    Qwen3Modeling.Qwen3Config.VOCAB_SIZE,
                    Qwen3Modeling.Qwen3Config.HIDDEN_SIZE,
                    Qwen3Modeling.Qwen3Config.PAD_IDX,
                    init: InitType.Zeros);

                lm_head = new ComputeBuffer(Gemma3Config.VOCAB_SIZE * Gemma3Config.HIDDEN_SIZE, 4, ComputeBufferType.Structured);

                _ = LoadEmbeddingWeightsAsync(params_path, lm_head);

                this.layers = new();
                for (int i = 0; i < Qwen3Modeling.Qwen3Config.NUM_LAYERS; i++)
                {
                    layers.Add(new Qwen3Modeling.Qwen3DecoderLayer(i, rope, params_path));
                }
                this.norm = new Qwen3RMSNorm(Qwen3Modeling.Qwen3Config.HIDDEN_SIZE, Qwen3Config.RMS_EPS, params_path + "/norm.bin"); 
                this.rope = new RotaryPositionalEmbeddings(Qwen3Modeling.Qwen3Config.HIDDEN_SIZE, Qwen3Modeling.Qwen3Config.CONTEXT_LENGTH, Qwen3Modeling.Qwen3Config.ROPE_THETA);
            }

            private async Task LoadEmbeddingWeightsAsync(string paramsPath, ComputeBuffer lm_head)
            {
                int[] partSizes = new int[]
                {
                    12_965_206, 12_965_206,12_965_206, 12_965_206,
                    12_965_206, 12_965_206, 12_965_206, 12_965_206,
                    12_965_206, 12_965_206, 12_965_206, 12_965_198
                };

                string[] files = new string[partSizes.Length];
                for (int i = 0; i < partSizes.Length; i++)
                    files[i] = $"{paramsPath}/lm_head/part_{i}.bin";

                Task<float[]>[] tasks = new Task<float[]>[partSizes.Length];
                for (int i = 0; i < partSizes.Length; i++)
                {
                    int size = partSizes[i];
                    string path = files[i];
                    tasks[i] = Task.Run(() => Utils.ReadWeights(path, size));
                }

                float[][] results = await Task.WhenAll(tasks);

                Parallel.For(0, partSizes.Length, part =>
                {
                    for (int i = 0; i < results[part].Length; i++)
                    {
                        this.embed_tokens.embeddings[part * partSizes[0] + i] = results[part][i];
                    }
                }); // faster with parallel for believe me.

                IsEmbeddingInitialized = true;
                // ConsoleMessage.Info($"Loaded {paramsPath}/embeddings");
                lm_head.SetData(this.embed_tokens.embeddings.Data);
                // ConsoleMessage.Info($"Loaded {paramsPath}/lm_head");
            }


            public Tensor Predict(Tensor input_ids, Tensor attention_mask = null)
            {
                Tensor hid = embed_tokens.Predict(input_ids);// * MathF.Sqrt(Qwen3Config.HIDDEN_SIZE);
                UnityEngine.Debug.Log("embed_out * sqrt(hid):" + hid);
                foreach (var layer in layers)
                {
                    hid = layer.Predict(hid, attention_mask);
                }

                return norm.Predict(hid);
            }

            public int ParameterCount()
            {
                int @params = embed_tokens.embeddings.Count();
                //UnityEngine.Debug.Log($"model.embed_tokens:{embed_tokens.embeddings.Shape.ToCommaSeparatedString()}");
                foreach (var item in layers)
                {
                    @params += item.ParameterCount();
                }
                //UnityEngine.Debug.Log($"model.norm:{norm.gamma.Length}");
                @params += norm.gamma.Length;
                return @params;
            }
        }
    }

    public class Qwen3ForCausalLM
    {
        private int vocab_size;
        private int hidden_size;
        private Qwen3Modeling.Qwen3Model model;

        private ComputeBuffer lm_head;
        private List<Dictionary<string, string>> conversation_cache;



        public bool IsReady => model.IsInitialized;
        public Qwen3ForCausalLM(string params_path = "Assets/DeepUnity/LLMs/Qwen3/params")
        {
            this.vocab_size = Qwen3Modeling.Qwen3Config.VOCAB_SIZE;
            this.hidden_size = Qwen3Modeling.Qwen3Config.HIDDEN_SIZE;
            
#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged += DeallocQwen;

#endif

            model = new Qwen3Modeling.Qwen3Model(params_path, ref lm_head);

            // in_features: Qwen3Config.HIDDEN_SIZE,
            // out_features: Qwen3Config.VOCAB_SIZE,
            // weight_init: InitType.Zeros,
            // bias: false, device: Device.GPU);
        }

        ~Qwen3ForCausalLM()
        {
            foreach (var item in model.layers)
            {
                item.mlp.weights.Release();
                item.self_attn.W_QKV.Release();
                item.self_attn.W_O.Release();
            }

            lm_head.Release();
            Debug.Log("Qwen3 released from gpu");
        }

#if UNITY_EDITOR
        private void DeallocQwen(UnityEditor.PlayModeStateChange state)
        {
            if (state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
            {
                foreach (var item in model.layers)
                {
                    item.mlp.weights.Release();
                    item.self_attn.W_QKV.Release();
                    item.self_attn.W_O.Release();
                }
        
                lm_head.Release();
                ConsoleMessage.Info("Qwen3 released from GPU"); 
            }
        }
#endif
        public int ParameterCount()
        {
            int @params  = model.ParameterCount();
            // UnityEngine.Debug.Log($"model.lm_head:{lm_head.)}");
            return @params;
        }
        public Tensor Predict(Tensor input_ids, Tensor attn_mask = null)
        {
            if (!IsReady)
            {
                throw new System.Exception("The model was not loaded yet. Please verify if IsReady.");
                // never put while(true) because model will not initialize otherwise.
            }

            int seq_len = input_ids.Size(-1);
            bool is_batched = input_ids.Rank == 3;
            int batch_size = is_batched ? input_ids.Size(-3) : 1;
            Tensor hid = model.Predict(input_ids, attn_mask);
   

            
            //Benckmark.Start(); // lm head takes 0.01645s per 1 token.

            ComputeShader lm_head_cs = DeepUnityMeta.LmHeadInferenceCS;
            int k = seq_len == 1 && batch_size == 1? lm_head_cs.FindKernel("Predict1Vec") : lm_head_cs.FindKernel("Predict");

            lm_head_cs.SetBuffer(k, "weights", this.lm_head);

            ComputeBuffer lm_head_input = new ComputeBuffer(hid.Count(), 4);
            lm_head_input.SetData(hid.ToArray());
            lm_head_cs.SetBuffer(k, "input", lm_head_input);

            ComputeBuffer lm_head_output = new ComputeBuffer(hid.Count(), 4);
            lm_head_cs.SetBuffer(k, "output", lm_head_output);

            lm_head_cs.SetInt("batch_size", batch_size);
            lm_head_cs.SetInt("seq_len", seq_len);
            lm_head_cs.SetInt("hidden_size", this.hidden_size);
            lm_head_cs.SetInt("vocab_size", this.vocab_size);

            if(seq_len == 1 && batch_size == 1)
                lm_head_cs.Dispatch(k, (vocab_size + 511) / 512, batch_size * seq_len, 1);
            else
                lm_head_cs.Dispatch(k, (vocab_size + 31) / 32, (batch_size * seq_len + 7) / 8, 1);
            Tensor output_probs = is_batched ?
                Tensor.Constant(lm_head_output, batch_size, seq_len, hidden_size) :
                Tensor.Constant(lm_head_output, seq_len, hidden_size);

            lm_head_output.Release();
            lm_head_input.Release();
            
            // Benckmark.Stop("lm head");
            return output_probs;
        }

        /// <summary>
        /// Expects (L, V) or (B, L, V) for batched inference,
        /// where V = vocab size, L (>= 1) = sequence length and B = batch_size
        /// Returns (1, 1) or (B, 1, 1) for batched inference.
        /// </summary>
        /// <param name="logits"></param>
        /// <param name="temperature"></param>
        /// <param name="top_k"></param>
        /// <param name="top_p"></param>
        /// <param name="min_p"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        private Tensor SampleToken(Tensor logits, float temperature = 1f, int top_k = -1, float top_p = 1, float min_p = 0f)
        {
            if (logits.Rank == 1 || logits.Rank == 2)
            {
                int vocab_size = logits.Size(-1);
                int last_vec_idx = logits.Size(0) - 1;

                Softmax sm = new Softmax(temperature: temperature);
                Tensor probs = sm.Predict(logits);

                // max prob for min_p
                float minProbTresh = probs.Max() * min_p;
                int candidatesCount = probs.Count(x => x >= minProbTresh);

                if (candidatesCount == 0)
                    return logits.ArgMax(-1);

                List<int> candidates = new List<int>();
                for (int i = 0; i < vocab_size; i++)
                {
                    if (probs[i] >= minProbTresh)
                        candidates.Add(i);
                }

                candidates.Sort((a, b) => probs[a].CompareTo(probs[b])); //descending

                /// TOP_K
                if (top_k > 0 && top_k < candidatesCount)
                {
                    candidatesCount = top_k;
                }

                /// TOP_P
                if (top_p < 1)
                {
                    float cumProb = 0f;
                    int cutoff = candidatesCount;
                    for (int i = 0; i < candidatesCount; i++)
                    {
                        cumProb += probs[candidates[i]];
                        if (cumProb >= top_p)
                        {
                            cutoff = i + 1;
                            break;
                        }
                    }
                    candidatesCount = cutoff;
                }

                // renormalize the new dist
                float newSum = 0.0f;
                for (int i = 0; i < candidatesCount; i++)
                {
                    newSum += probs[candidates[i]];
                }
                if (newSum == 0.0)
                {
                    return logits.ArgMax(-1);
                }

                // Sample 
                return Tensor.Constant(Utils.Random.Sample(candidates, candidates.Select(x => probs[x])));
            }
            else if (logits.Rank == 3)
            {
                throw new Exception("Batched sampling not handled");

            }
            else
            {
                throw new Exception("Cannot sample 4D tensors");

            }
        }

        public IEnumerator Generate(string prompt, Qwen3TokenizerFast tokenizer, int max_new_tokens = 128, float temperature = 0.7f, int top_k = 20, float top_p = 0.95f, float min_p = 0)
        {
            Debug.Log("Generating...");
            if (!this.IsReady)
                yield return new WaitForSeconds(2f);

            Debug.Log("Model Ready");
            if (!tokenizer.IsReady)
                yield return new WaitForSeconds(0.01f);


            foreach (var item in model.layers)
            {
                item.self_attn.BuildKVCache = true;
            }
            (Tensor, Tensor) tokenized_prompt = tokenizer.Encode(prompt);


            List<string> response = new List<string>();

            Tensor y = model.Predict(tokenized_prompt.Item1, tokenized_prompt.Item2);

            Debug.Log("y:" + y);
            Tensor sampled_token_id = SampleToken(y, temperature: temperature, top_k: top_k, top_p: top_p, min_p: min_p);
            Debug.Log("sampled_tok:" + sampled_token_id);
            for (int i = 0; i < max_new_tokens; i++)
            {
                string sampled_token_str = tokenizer.Decode(sampled_token_id)[0];
                Debug.Log(sampled_token_str);
                response.Add(sampled_token_str);

                y = model.Predict(sampled_token_id, null);
                sampled_token_id = SampleToken(y, temperature: temperature, top_k: top_k, top_p: top_p, min_p: min_p);
            }
        }

        public Dictionary<string, string> Chat(List<Dictionary<string, string>> messages, Qwen3TokenizerFast tokenizer, float temperature = 0.7f, int top_k = 20, float top_p = 0.95f, float min_p = 0)
        {
            List<Dictionary<string, string>> cached_conv = new();
            List<Dictionary<string, string>> noncached_conv = new();

            bool counting_cached_messages = true;
            foreach (var item in messages)
            {
                if(conversation_cache.Contains(item) && counting_cached_messages)
                {
                    cached_conv.Add(item);
                }
                else
                {
                    counting_cached_messages = false;
                    noncached_conv.Add(item);
                }
            }
            string input = tokenizer.ApplyChatTemplate(messages, add_generation_prompt:true);
            var x = tokenizer.Encode(input);
            Tensor input_ids = x.Item1;
            Tensor attn_mask = x.Item2;

            // Set attention mask to all gqa modules..
            // Keep length of KV cache equal to the cached conv len
            // Start passing 1 element at a time.
            

            StringBuilder generated_output = new StringBuilder();

            Tensor output = this.model.Predict(input_ids, attn_mask);

            // do it un

            return new Dictionary<string, string>
            {
                {"role","assistant"},
                {"content", generated_output.ToString()}
            };

        }
    }
}

