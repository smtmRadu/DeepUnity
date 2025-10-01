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
    namespace Gemma3Modeling
    {
        public class Gemma3Model 
        {
            public int eos_idx = 1;
            public int bos_idx = 2;
            public int pad_idx = 0;
            public int vocab_size = 0;
            public Embedding embed_tokens;
            public List<Gemma3DecoderLayer> layers;
            public Gemma3RMSNorm norm;
            public RotaryPositionalEmbeddings rotary_emb; // used for FullGQA
            public RotaryPositionalEmbeddings rotary_emb_local; // used for SWA
            public bool IsInitialized { get 
                {
                    if (!IsEmbeddingInitialized)
                        return false;
                    if (!norm.IsInitialized)
                        return false;

                    foreach (var layer in layers)
                    {
                        if(!layer.self_attn.IsInitialized)
                            return false;
                        if(!layer.mlp.IsInitialized)
                            return false;
                        if(!layer.input_layernorm.IsInitialized)
                            return false;
                        if (!layer.post_attention_layernorm.IsInitialized)
                            return false;
                        if (!layer.pre_feedforward_layernorm.IsInitialized)
                            return false;
                        if (!layer.post_attention_layernorm.IsInitialized)
                            return false;
                    }
                    return true;
                } 
            }
            private bool IsEmbeddingInitialized { get; set; } = false;
            public Gemma3Model(string params_path, ref ComputeBuffer lm_head)
            {
                this.eos_idx = Gemma3Config.EOS_IDX;
                this.bos_idx = Gemma3Config.BOS_IDX;
                this.pad_idx = Gemma3Config.PAD_IDX;
                this.vocab_size = Gemma3Config.VOCAB_SIZE;
                rotary_emb = new RotaryPositionalEmbeddings(
                    Gemma3Config.HEAD_DIM,
                    max_seq_len:Gemma3Config.MAX_POSITION_EMBEDDINGS,
                    theta:Gemma3Config.ROPE_THETA); // use only 1 rope module shared by all layers.
                rotary_emb_local = new RotaryPositionalEmbeddings(
                    Gemma3Config.HEAD_DIM,
                    max_seq_len: Gemma3Config.MAX_POSITION_EMBEDDINGS,
                    theta: Gemma3Config.ROPE_LOCAL_BASE_FREQUENCY);
                this.embed_tokens = new Embedding(
                    Gemma3Config.VOCAB_SIZE,
                    Gemma3Config.HIDDEN_SIZE,
                    Gemma3Config.PAD_IDX,
                    init: InitType.Zeros);

                lm_head = new ComputeBuffer(Gemma3Config.VOCAB_SIZE * Gemma3Config.HIDDEN_SIZE, 4, ComputeBufferType.Structured);

                _ = LoadEmbeddingWeightsAsync(params_path, lm_head);

                this.layers = new();
                for (int i = 0; i < Gemma3Config.NUM_LAYERS; i++)
                {
                    layers.Add(
                        new Gemma3DecoderLayer(
                            i, 
                            Gemma3Config.layer_types[i] == GemmaLayerType.SlidingWindowAttention? rotary_emb_local : rotary_emb, 
                            params_path));
                }
                
                this.norm = new Gemma3RMSNorm(Gemma3Config.HIDDEN_SIZE, Gemma3Config.RMS_EPS, params_path + "/norm.bin"); //new RMSNorm(Qwen3Modeling.Qwen3Config.HIDDEN_SIZE, Qwen3Modeling.Qwen3Config.RMS_EPS, elementwise_affine: true);
            
            }

            private async Task LoadEmbeddingWeightsAsync(string paramsPath, ComputeBuffer lm_head)
            {
                int[] partSizes = new int[]
                {
                    11_983_726, 11_983_726, 11_983_726, 11_983_726,
                    11_983_726, 11_983_726, 11_983_726, 11_983_726,
                    11_983_726, 11_983_726, 11_983_726, 11_983_726,
                    11_983_726, 11_983_722
                };

                string[] files = new string[14];
                for (int i = 0; i < 14; i++)
                    files[i] = $"{paramsPath}/lm_head/part_{i}.bin";

                Task<float[]>[] tasks = new Task<float[]>[14];
                for (int i = 0; i < 14; i++)
                {
                    int size = partSizes[i];
                    string path = files[i];
                    tasks[i] = Task.Run(() => Utils.ReadWeights(path, size));
                }

                float[][] results = await Task.WhenAll(tasks);

                Parallel.For(0, 14, part =>
                {
                    for (int i = 0; i < results[part].Length; i++)
                    {
                        this.embed_tokens.embeddings[part * 11_983_726 + i] = results[part][i];
                    }
                }); // faster with parallel for believe me.

                IsEmbeddingInitialized = true;
                // ConsoleMessage.Info($"Loaded {paramsPath}/embeddings");
                lm_head.SetData(this.embed_tokens.embeddings.Data);
                // ConsoleMessage.Info($"Loaded {paramsPath}/lm_head");
            }

            public Tensor Predict(Tensor input_ids, Tensor attention_mask = null)
            {
                
                Tensor hid = embed_tokens.Predict(input_ids) * MathF.Sqrt(Gemma3Config.HIDDEN_SIZE); // input embeddings are normalized by sqrt(model_size) in hf transformers.

                // Debug.Log("Embeddings: " + hid);

                for (int i = 0; i < layers.Count; i++)
                {
                    hid = layers[i].Predict(hid, attention_mask);
                    // Debug.Log($"Layer {i} output: " + hid);
                }
                Tensor post_norm = norm.Predict(hid);
                return post_norm.Squeeze(-2);
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

    public class Gemma3ForCausalLM
    {
        private string params_path;
        private int vocab_size;
        private int hidden_size;
        public Gemma3Modeling.Gemma3Model model;
        public ComputeBuffer lm_head;
        public Gemma3TokenizerFast tokenizer;
        private List<Dictionary<string, string>> conversation_cache;

        public bool IsReady => model.IsInitialized;
        public Gemma3ForCausalLM(string params_path= "Assets/DeepUnity/LLMs/Gemma3/params", string tokenizer_path = "Assets/DeepUnity/LLMs/Gemma3/Gemma3TokenizerFast.json")
        {
            this.params_path = params_path;
            this.vocab_size = Gemma3Config.VOCAB_SIZE;
            this.hidden_size = Gemma3Config.HIDDEN_SIZE;
            this.tokenizer = new Gemma3TokenizerFast(tokenizer_path, load_async: true);

#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged += DeallocGemma;
#endif
            // to initialize lm_head async as well it must be parsed with ref, but async methods does not allow ref arguments.. fuck em..
            // lm head will be initialized in gemma3 model
            model = new Gemma3Modeling.Gemma3Model(params_path, ref lm_head);
        }

        ~Gemma3ForCausalLM()
        {
            foreach (var item in model.layers)
            {
                item.mlp.weights.Release();
                item.self_attn.W_QKV.Release();
                item.self_attn.W_O.Release();
            }

            lm_head.Release();
            ConsoleMessage.Info("Gemma3 released from GPU");
        }

#if UNITY_EDITOR
        private void DeallocGemma(UnityEditor.PlayModeStateChange state)
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
                ConsoleMessage.Info("Gemma3 released from GPU");
            }
        }
#endif
        public int ParameterCount()
        {
            int @params = model.ParameterCount();
            return @params;
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
            if(logits.Rank == 1)
            {
                if(temperature == 0)
                    return logits.ArgMax(-1);
                
                int vocab_size = logits.Size(-1);
                

                Softmax sm = new Softmax(temperature : temperature);
                Tensor probs = sm.Predict(logits);

                // max prob for min_p
                float minProbTresh = probs.Max() * min_p;
                int candidatesCount = probs.Count(x => x >= minProbTresh);

                if (candidatesCount == 0)
                    return logits.ArgMax(-1);

                List<int> candidates = new List<int>();
                for (int i = 0; i < vocab_size ; i++)
                {
                    if (probs[i] >= minProbTresh)
                        candidates.Add(i);
                }

                candidates.Sort((a, b) => probs[a].CompareTo(probs[b])); //descending

                /// TOP_K
                if(top_k > 0 && top_k < candidatesCount)
                {
                    candidatesCount = top_k;
                }

                /// TOP_P
                if(top_p < 1)
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
            else if(logits.Rank == 2)
            {
                if (logits.Size(-2) == 1)
                    return SampleToken(logits.Squeeze(-2), temperature, top_k, top_p, min_p);
                else
                    return SampleToken(logits.Slice(-2, logits.Size(-2) - 1, logits.Size(-2), 1).Squeeze(), temperature, top_k, top_p, min_p);
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
        
        public Tensor Predict(Tensor input_ids, Tensor attn_mask = null)
        {
            if(!IsReady)
            {
                throw new System.Exception("The model was not loaded yet. Please verify if IsReady.");
                // never put while(true) because model will not initialize otherwise.
            }

            int seq_len = input_ids.Size(-1);
            bool is_batched = input_ids.Rank == 3;
            int batch_size = is_batched ? input_ids.Size(-3) : 1;
            Tensor hid = model.Predict(input_ids, attn_mask);


            // LM HEAD Infer
            ComputeShader lm_head_cs = DeepUnityMeta.LmHeadInferenceCS;
            int k = seq_len == 1 && batch_size == 1 ? lm_head_cs.FindKernel("Predict1Vec") : lm_head_cs.FindKernel("Predict");

            lm_head_cs.SetBuffer(k, "weights", this.lm_head);

            ComputeBuffer lm_head_input = new ComputeBuffer(hid.Count(), 4);
            lm_head_input.SetData(hid.ToArray());
            lm_head_cs.SetBuffer(k, "input", lm_head_input);

            ComputeBuffer lm_head_output = new ComputeBuffer(batch_size * seq_len * vocab_size, 4);
            lm_head_cs.SetBuffer(k, "output", lm_head_output);

            lm_head_cs.SetInt("batch_size", batch_size);
            lm_head_cs.SetInt("seq_len", seq_len);
            lm_head_cs.SetInt("hidden_size", this.hidden_size);
            lm_head_cs.SetInt("vocab_size", this.vocab_size);

            if (seq_len == 1 && batch_size == 1)
                lm_head_cs.Dispatch(k, (vocab_size + 511) / 512, batch_size * seq_len, 1);
            else
                lm_head_cs.Dispatch(k, (vocab_size + 31) / 32, (batch_size * seq_len + 7) / 8, 1);

            Tensor output_logits = is_batched ?
                Tensor.Constant(lm_head_output, batch_size, seq_len, vocab_size) :
                Tensor.Constant(lm_head_output, seq_len, vocab_size);

            lm_head_output.Release();
            lm_head_input.Release();

            return output_logits;
        }
        /// <summary>
        /// Generate("user: What's the capital of France? assistant: ", onTokenGenerated = (x) => { Debug.Log(x); }, tokenizer=tok)
        /// </summary>
        /// <param name="prompt"></param>
        /// <param name="onTokenGenerated"></param>
        /// <param name="tokenizer"></param>
        /// <param name="max_new_tokens"></param>
        /// <param name="temperature"></param>
        /// <param name="top_k"></param>
        /// <param name="top_p"></param>
        /// <param name="min_p"></param>
        /// <returns></returns>
        public IEnumerator Generate(string prompt, Action<string> onTokenGenerated, int max_new_tokens = 128, float temperature = 1f, int top_k = -1, float top_p = 1f, float min_p = 0)
        {
            // Debug.Log("Generating...");
            while(!this.IsReady)
                yield return new WaitForSeconds(0.01f);

            // Debug.Log("Model Ready");
            while (!tokenizer.IsReady)
                yield return new WaitForSeconds(0.01f);
            // Debug.Log("Tokenizer Ready");

            foreach (var item in model.layers)
            {
                item.self_attn.BuildKVCache = true;
            }
            (Tensor, Tensor) tokenized_prompt = tokenizer.Encode(prompt);

            // Debug.Log("x: " + tokenized_prompt.Item1);

            // forward + lm_head ============================================================================================================
            Tensor y = Predict(tokenized_prompt.Item1);
            // ============================================================================================================


            // Debug.Log("y:" + y);
            Tensor sampled_token_id = SampleToken(y, temperature: temperature, top_k: top_k, top_p: top_p, min_p: min_p);
            string sampled_token_str = tokenizer.Decode(sampled_token_id)[0];
            // Debug.Log("Next token: " + sampled_token_str + $" ({sampled_token_id[0]})");
            onTokenGenerated?.Invoke(sampled_token_str);
            yield return null;

            //Debug.Log("sampled_tok:" + sampled_token_id);
            for (int new_tok = 0; new_tok < max_new_tokens - 1; new_tok++)
            {
               
                // forward + lm_head ============================================================================================================
                y = Predict(sampled_token_id);
                // Debug.Log("y:" + y);
                // ============================================================================================================
                sampled_token_id = SampleToken(y, temperature: temperature, top_k: top_k, top_p: top_p, min_p: min_p);

                sampled_token_str = tokenizer.Decode(sampled_token_id)[0];
                // Debug.Log(sampled_token_str);
                onTokenGenerated?.Invoke(sampled_token_str);
                yield return null;
                if (sampled_token_id[0] == model.eos_idx)
                    break;
                
            }
        }
        
        public Dictionary<string, string> Chat(List<Dictionary<string, string>> messages, Gemma3TokenizerFast tokenizer, float temperature = 0.7f, int top_k = 20, float top_p = 0.95f, float min_p = 0)
        {
            List<Dictionary<string, string>> cached_conv = new();
            List<Dictionary<string, string>> noncached_conv = new();

            bool counting_cached_messages = true;
            foreach (var item in messages)
            {
                if (conversation_cache.Contains(item) && counting_cached_messages)
                {
                    cached_conv.Add(item);
                }
                else
                {
                    counting_cached_messages = false;
                    noncached_conv.Add(item);
                }
            }
            string input = tokenizer.ApplyChatTemplate(messages, add_generation_prompt: true);
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

