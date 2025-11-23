using DeepUnity.Gemma3Modeling;
using DeepUnity.Modules;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace Gemma3Modeling
    {
        public class EmbeddingGemmaModel
        {
            public int eos_idx = 1;
            public int bos_idx = 2;
            public int pad_idx = 0;
            public int vocab_size = 0;
            public Embedding embed_tokens;
            public List<EmbeddingGemmaDecoderLayer> layers;
            public Gemma3RMSNorm norm;
            public RotaryPositionalEmbeddings rotary_emb; // used for FullGQA
            public RotaryPositionalEmbeddings rotary_emb_local; // used for SWA
            public bool IsInitialized
            {
                get
                {
                    if (!IsEmbeddingInitialized)
                    {
                        //Debug.Log("Embedding not initialized");
                        return false;
                    }
                    if (!IsFFNInitialized)
                    {
                        //Debug.Log("FFN not initialized");
                        return false;
                    }
                        
                    if (!norm.IsInitialized)
                    {
                        //Debug.Log("Norm not initialized");
                        return false;
                    }
                       

                    foreach (var layer in layers)
                    {
                        if (!layer.self_attn.IsInitialized)
                        {
                            //Debug.Log("Layer not initialized.");
                            return false;
                        }
                            
                        if (!layer.mlp.IsInitialized)
                        {
                            //Debug.Log("Layer not initialized.");
                            return false;
                        }
                        if (!layer.input_layernorm.IsInitialized)
                        {
                            //Debug.Log("Layer not initialized.");
                            return false;
                        }
                        if (!layer.post_attention_layernorm.IsInitialized)
                        {
                            //Debug.Log("Layer not initialized.");
                            return false;
                        }
                        if (!layer.pre_feedforward_layernorm.IsInitialized)
                        {
                            //Debug.Log("Layer not initialized.");
                            return false;
                        }
                        if (!layer.post_attention_layernorm.IsInitialized)
                        {
                            //Debug.Log("Layer not initialized.");
                            return false;
                        }
                    }
                    return true;
                }
            }
            private bool IsEmbeddingInitialized { get; set; } = false;
            private bool IsFFNInitialized { get; set; } = false;
            public EmbeddingGemmaModel(string params_path, ref ComputeBuffer dense_weights)
            {
                this.eos_idx = EmbeddingGemmaConfig.EOS_IDX;
                this.bos_idx = EmbeddingGemmaConfig.BOS_IDX;
                this.pad_idx = EmbeddingGemmaConfig.PAD_IDX;
                this.vocab_size = EmbeddingGemmaConfig.VOCAB_SIZE;
                rotary_emb = new RotaryPositionalEmbeddings(
                        EmbeddingGemmaConfig.HEAD_DIM,
                    max_seq_len: EmbeddingGemmaConfig.MAX_POSITION_EMBEDDINGS,
                    theta: EmbeddingGemmaConfig.ROPE_THETA); // use only 1 rope module shared by all layers.
                rotary_emb_local = new RotaryPositionalEmbeddings(
                    EmbeddingGemmaConfig.HEAD_DIM,
                    max_seq_len: EmbeddingGemmaConfig.MAX_POSITION_EMBEDDINGS,
                    theta: EmbeddingGemmaConfig.ROPE_LOCAL_BASE_FREQUENCY);
                this.embed_tokens = new Embedding(
                    EmbeddingGemmaConfig.VOCAB_SIZE,
                    EmbeddingGemmaConfig.HIDDEN_SIZE,
                    EmbeddingGemmaConfig.PAD_IDX,
                    init: InitType.Zeros);

                dense_weights = new ComputeBuffer(EmbeddingGemmaConfig.HIDDEN_SIZE * EmbeddingGemmaConfig.HEAD_FFN_INTERMEDIATE_SIZE * 2, 4, ComputeBufferType.Structured);
                _ = LoadEmbeddingWeightsAsync(params_path, dense_weights);

                this.layers = new();
                for (int i = 0; i < EmbeddingGemmaConfig.NUM_LAYERS; i++)
                {
                    layers.Add(
                        new EmbeddingGemmaDecoderLayer(
                            i,
                            EmbeddingGemmaConfig.layer_types[i] == GemmaLayerType.SlidingWindowAttention ? rotary_emb_local : rotary_emb,
                            params_path));
                }

                this.norm = new Gemma3RMSNorm(EmbeddingGemmaConfig.HIDDEN_SIZE, EmbeddingGemmaConfig.RMS_EPS, params_path + "/norm.bin"); //new RMSNorm(Qwen3Modeling.Qwen3Config.HIDDEN_SIZE, Qwen3Modeling.Qwen3Config.RMS_EPS, elementwise_affine: true);

            }

            private async Task LoadEmbeddingWeightsAsync(string paramsPath, ComputeBuffer dense_weights)
            {
                int[] partSizes = new int[]
                {
                    12_582_912, 12_582_912,12_582_912,12_582_912,
                    12_582_912, 12_582_912,12_582_912,12_582_912,
                    12_582_912, 12_582_912, 12_582_912, 12_582_912,
                    12_582_912, 12_582_912, 12_582_912, 12_582_912
                };

                string[] files = new string[18];
                for (int i = 0; i < 16; i++)
                    files[i] = $"{paramsPath}/embed_tokens/part_{i}.bin";

                Task<float[]>[] tasks = new Task<float[]>[18];
                for (int i = 0; i < 16; i++)
                {
                    int size = partSizes[i];
                    string path = files[i];
                    tasks[i] = Task.Run(() => Utils.ReadWeights(path, size));
                }
                
                tasks[16] = Task.Run(() => Utils.ReadWeights($"{paramsPath}/dense_1.bin", EmbeddingGemmaConfig.HIDDEN_SIZE * EmbeddingGemmaConfig.HEAD_FFN_INTERMEDIATE_SIZE));
                tasks[17] = Task.Run(() => Utils.ReadWeights($"{paramsPath}/dense_2.bin", EmbeddingGemmaConfig.HIDDEN_SIZE * EmbeddingGemmaConfig.HEAD_FFN_INTERMEDIATE_SIZE));

                float[][] results = await Task.WhenAll(tasks);

                Parallel.For(0, 16, part =>
                {
                    for (int i = 0; i < results[part].Length; i++)
                    {
                        this.embed_tokens.embeddings[part * 12_582_912 + i] = results[part][i];
                    }
                }); // faster with parallel for believe me.

                IsEmbeddingInitialized = true;

                var dense_down = Tensor.Constant(results[16]).Reshape(3072, 768).ToArray(); // these values were saved improperly and must be reread. (now it is fine)
                var dense_up = Tensor.Constant(results[17]).Reshape(768, 3072).ToArray();

                dense_weights.SetData(dense_down.Concat(dense_up).ToArray());



                IsFFNInitialized = true;
                // ConsoleMessage.Info($"Loaded {paramsPath}/embeddings");
                // ConsoleMessage.Info($"Loaded {paramsPath}/lm_head");
            }

            public Tensor Predict(Tensor input_ids, Tensor attention_mask = null)
            {

                Tensor hid = embed_tokens.Predict(input_ids) * MathF.Sqrt(EmbeddingGemmaConfig.HIDDEN_SIZE); // input embeddings are normalized by sqrt(model_size) in hf transformers.

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

    public class Gemma3ForEmbeddings
    {
        public EmbeddingGemmaModel model;
        public ComputeBuffer dense_weights; // up + down
        private ComputeBuffer dense_IO_buffer;
        private ComputeBuffer dense_intermediate_buffer;
        public Gemma3TokenizerFast tokenizer;

        public bool IsReady => model.IsInitialized;
        public float TokensPerSecond { get; private set; }
        public Gemma3ForEmbeddings(string params_path = "Assets/DeepUnity/LLMs/Gemma3/params_embedding", string tokenizer_path = "Assets/DeepUnity/LLMs/Gemma3/Gemma3TokenizerFast.json")
        {
            this.tokenizer = new Gemma3TokenizerFast(tokenizer_path, load_async: true);

#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged += DeallocGemma;
#endif
            // to initialize lm_head async as well it must be parsed with ref, but async methods does not allow ref arguments.. fuck em..
            // lm head will be initialized in gemma3 model
            model = new EmbeddingGemmaModel(params_path, ref dense_weights);
        }

        ~Gemma3ForEmbeddings()
        {
            foreach (var item in model.layers)
            {
                item.mlp.weights?.Release();
                item.mlp.intermediateBuffer?.Release();
                item.mlp.inputOutputBuffer?.Release();
                item.self_attn.W_QKV?.Release();
                item.self_attn.W_O?.Release();
                item.self_attn.output_buffer?.Release();
                item.self_attn.attended_values_buffer?.Release();
                item.self_attn.input_buffer?.Release();
                item.self_attn.qkv_proj_buffer?.Release();
                item.self_attn.q_buffer?.Release();
            }

            dense_weights?.Release();
            dense_IO_buffer?.Release();
            dense_intermediate_buffer?.Release();
            ConsoleMessage.Info("Gemma3ForEmbeddings released from GPU");
        }

#if UNITY_EDITOR
        private void DeallocGemma(UnityEditor.PlayModeStateChange state)
        {
            if (state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
            {
                foreach (var item in model.layers)
                {
                    item.mlp.weights?.Release();
                    item.mlp.intermediateBuffer?.Release();
                    item.mlp.inputOutputBuffer?.Release();
                    item.self_attn.W_QKV?.Release();
                    item.self_attn.W_O?.Release();
                    item.self_attn.input_buffer?.Release();
                    item.self_attn.qkv_proj_buffer?.Release();
                    item.self_attn.output_buffer?.Release();
                    item.self_attn.attended_values_buffer?.Release();
                    item.self_attn.q_buffer?.Release();
                }

                dense_weights?.Release();
                dense_IO_buffer?.Release();
                dense_intermediate_buffer?.Release();
                ConsoleMessage.Info("Gemma3ForEmbeddings released from GPU");
            }
        }
#endif
        private void PrepareFFNBuffers(int io_buffer_size, int intermediate_buffer_size)
        {
            if (dense_IO_buffer == null || dense_IO_buffer.count != io_buffer_size)
            {
                dense_IO_buffer?.Release();
                dense_IO_buffer = new ComputeBuffer(io_buffer_size, 4, ComputeBufferType.Structured);
            }
            
            if(dense_intermediate_buffer == null || dense_intermediate_buffer.count != intermediate_buffer_size)
            {
                dense_intermediate_buffer?.Release();
                dense_intermediate_buffer = new ComputeBuffer(intermediate_buffer_size, 4, ComputeBufferType.Structured);
            }
        }

        /// <summary>
        /// onEmbeddingReceived returns => sentence_embedding
        /// </summary>
        /// <param name="prompt"></param>
        /// <param name="onEmbeddingReceived"></param>
        /// <returns></returns>
        public IEnumerator EncodeQuery(string prompt, Action<Tensor> onEmbeddingReceived)
        {
            
            // Debug.Log("Generating...");
            //Debug.Log("Initializing model...");
            while (!this.IsReady)
                yield return new WaitForSeconds(0.01f);

            //Debug.Log("Model ready");
            // Debug.Log("Model Ready");
            while (!tokenizer.IsReady)
                yield return new WaitForSeconds(0.01f);
            // Debug.Log("Tokenizer Ready");

            
            Stopwatch stopwatch = Stopwatch.StartNew();

            //Debug.Log("Tokenizer ready");
            foreach (var item in model.layers)
            {
                item.self_attn.BuildKVCache = false;
            }

            //Debug.Log("Prompt: " + prompt);

            prompt += "<eos>"; // it seems like when tested with sentence-transformers it adds <bos> and <eos> tags at the start and end
            (Tensor, Tensor) tokenized_prompt = tokenizer.Encode(prompt);
            yield return null;

            // Debug.Log("x: " + tokenized_prompt.Item1);

            // forward + lm_head (with frame generation allowed) ============================================================================================================
            Tensor y = null;

            var input_ids = tokenized_prompt.Item1;
            // Debug.Log("Input IDS: " + input_ids);
            Tensor attn_mask = null;
            bool is_batched = input_ids.Rank == 3;
            int batch_size = is_batched ? input_ids.Size(-3) : 1;



            Tensor hid = model.embed_tokens.Predict(input_ids) * MathF.Sqrt(EmbeddingGemmaConfig.HIDDEN_SIZE);
            // Debug.Log("Embeddings: "  + hid);
            yield return null;
            for (int i = 0; i < model.layers.Count; i++) // do not put yield return null between layer modules because you get a strange range of fps (60 to 140) - better just 60
            {
                hid = model.layers[i].Predict(hid, attn_mask);
                // Debug.Log($"Layer {i}: "+ hid);
                yield return null;
            }
            hid = model.norm.Predict(hid);
            // Debug.Log("Post Norm: " + hid);
            yield return null;


            // Mean pooling
            hid = hid.Mean(-2); // SEQ_LEN gets 1


            // Debug.Log("FFN input: " + hid);


            // LM HEAD Infer
            ComputeShader ffn_infer_cs = DeepUnityMeta.FFNInferenceCS;
            
            
            PrepareFFNBuffers(io_buffer_size:hid.Count(),intermediate_buffer_size: hid.Count() * EmbeddingGemmaConfig.HEAD_FFN_INTERMEDIATE_SIZE/EmbeddingGemmaConfig.HIDDEN_SIZE ); // 4 times larger
            dense_IO_buffer.SetData(hid.ToArray());

            yield return null;

            // ========================================================================= FFN UP ===================================================================

            int k_up = ffn_infer_cs.FindKernel("Up1Vec");
            ffn_infer_cs.SetInt("activation_type", -1); // linear
            ffn_infer_cs.SetInt("hidden_size", EmbeddingGemmaConfig.HIDDEN_SIZE);
            ffn_infer_cs.SetInt("intermediate_size", EmbeddingGemmaConfig.HEAD_FFN_INTERMEDIATE_SIZE);
            ffn_infer_cs.SetInt("batch_size", batch_size);
            ffn_infer_cs.SetInt("seq_len", 1);
            ffn_infer_cs.SetBuffer(k_up, "input", dense_IO_buffer);
            ffn_infer_cs.SetBuffer(k_up, "intermediate", dense_intermediate_buffer);
            ffn_infer_cs.SetBuffer(k_up, "weights", dense_weights);

            
            // GO UP
            ffn_infer_cs.Dispatch(k_up,
                (EmbeddingGemmaConfig.HEAD_FFN_INTERMEDIATE_SIZE + 255)/256,
                1,
                batch_size
                );

            var hidden = Tensor.Constant(dense_intermediate_buffer, EmbeddingGemmaConfig.HIDDEN_SIZE);
            // Debug.Log("Intermediate FFN value:" + hidden);



            // ========================================================================= FFN DOWN ===================================================================
            int k_down = ffn_infer_cs.FindKernel("Down1Vec");
            ffn_infer_cs.SetBuffer(k_down, "input", dense_IO_buffer);
            ffn_infer_cs.SetBuffer(k_down, "intermediate", dense_intermediate_buffer);
            ffn_infer_cs.SetBuffer(k_down, "weights", dense_weights);
            // GO DOWN
            ffn_infer_cs.Dispatch(k_down,
                (EmbeddingGemmaConfig.HEAD_FFN_INTERMEDIATE_SIZE + 319) / 320,
                1,
                 batch_size
                );
            // Always 1vec inference


            hid = is_batched ?
                Tensor.Constant(dense_IO_buffer, batch_size, EmbeddingGemmaConfig.HIDDEN_SIZE) :
                Tensor.Constant(dense_IO_buffer, EmbeddingGemmaConfig.HIDDEN_SIZE);

            // NORMALIZE THE FINAL THING TO Norm 1

            hid = hid / hid.Norm()[0];

            ConsoleMessage.Info("Time elapsed for encoding: " + stopwatch.Elapsed);
            onEmbeddingReceived(hid);   
        }

    }
}

