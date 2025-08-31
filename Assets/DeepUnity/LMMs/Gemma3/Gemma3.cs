using DeepUnity.Gemma3Modelling;
using DeepUnity.Models;
using DeepUnity.Modules;
using System.Collections.Generic;
using System.Drawing.Printing;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

// NOTES
// TODO: 
//       test if a a kernel rms norm will work (try make the entire decoder layer on gpu.. that s the idea)
//       when copying the weights out from the model to unity, load them in separate file each so that they will not be over 50MB each (each module separate file) -> embedding layer might be placed in 2 probably.
//       verify again the tokenizer of gemma to work. (optional)

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
            public RotaryPositionalEmbeddings rope;
            public bool IsInitialized { get 
                {
                    if (!IsEmbeddingInitialized)
                        return false;
                    if (!norm.IsInitialized)
                        return false;

                    foreach (var layer in layers)
                    {
                        if(!layer.gqa.IsInitialized)
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
                rope = new RotaryPositionalEmbeddings(
                    Gemma3Config.HEAD_DIM,
                    Gemma3Config.CONTEXT_LENGTH,
                    Gemma3Config.ROPE_THETA); // use only 1 rope module shared by all layers.

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
                    layers.Add(new Gemma3DecoderLayer(i, rope, params_path));
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
                Tensor hid = embed_tokens.Predict(input_ids);
                Debug.Log("Embeddings: " + hid);
                foreach (var layer in layers)
                {
                    hid = layer.Predict(hid, attention_mask);
                    Debug.Log("Layer output: " + hid);
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

    public class Gemma3ForCausalLM
    {
        private string params_path;
        private int vocab_size;
        private int hidden_size;
        public Gemma3Modeling.Gemma3Model model;
        public ComputeBuffer lm_head;
        private List<Dictionary<string, string>> conversation_cache;

        public bool IsReady => model.IsInitialized;
        public Gemma3ForCausalLM(string params_path= "Assets/DeepUnity/LMMs/Gemma3/params")
        {
            this.params_path = params_path;
            this.vocab_size = Gemma3Config.VOCAB_SIZE;
            this.hidden_size = Gemma3Config.HIDDEN_SIZE;

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
                item.gqa.W_QKV.Release();
                item.gqa.W_O.Release();
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
                    item.gqa.W_QKV.Release();
                    item.gqa.W_O.Release();  
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
        public int SampleToken(Tensor y, float temperature = 1f, int top_k = -1, float top_p = 1, float min_p = 0f)
        {
            // Y = (B, L, VOCAB_SIZE) or (L, VOCAB_SIZE)
            return (int)y.Max(-1)[0];
        }
        public Tensor Predict(Tensor input_ids, Tensor attn_mask = null)
        {
            if(!IsReady)
            {
                throw new System.Exception("The model was not loaded yet. Please verify if IsReady.");
            }

            int seq_len = input_ids.Size(-1);
            bool is_batched = input_ids.Rank == 3;
            int batch_size = is_batched ? input_ids.Size(-3) : 1;
            //Benckmark.Start();
            Tensor hid = model.Predict(input_ids, attn_mask);
            //Benckmark.Stop("Layers");


            // Benckmark.Start(); // lm head takes 0.01645s per 1 token.

            ComputeShader lm_head_cs = DeepUnityMeta.LmHeadInferenceCS;
            int k = lm_head_cs.FindKernel("Predict");

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

            lm_head_cs.Dispatch(k, (vocab_size + 31) / 32, (batch_size * seq_len + 7) / 8, 1);
            Tensor output_probs = is_batched ?
                Tensor.Constant(lm_head_output, batch_size, seq_len, hidden_size) :
                Tensor.Constant(lm_head_output, seq_len, hidden_size);

            lm_head_output.Release();
            lm_head_input.Release();

            // Benckmark.Stop("lm head");
            return output_probs;
        }
        public string Generate(string prompt, GemmaTokenizerFast tokenizer, float temperature = 0.7f, int top_k = 20, float top_p = 0.95f, float min_p = 0)
        {
            foreach (var item in model.layers)
            {
                item.gqa.BuildKVCache = true;
            }
            List<int> reponse_ids = new List<int>();
            (Tensor, Tensor) x = tokenizer.Encode(prompt);

            Tensor y = model.Predict(x.Item1, x.Item2);

            return null;

            // sample y
            // int sampled_token = 100;
            // while(sampled_token != model.eos_idx)
            // {
            //     y = model.Predict(Tensor.Constant(sampled_token));
            // 
            //     sampled_token = 100;
            // }
            // 
            // return string.Join("", reponse_ids.Select(x => tokenizer.id2token[x]));
        }
        public Dictionary<string, string> Chat(List<Dictionary<string, string>> messages, GemmaTokenizerFast tokenizer, float temperature = 0.7f, int top_k = 20, float top_p = 0.95f, float min_p = 0)
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

