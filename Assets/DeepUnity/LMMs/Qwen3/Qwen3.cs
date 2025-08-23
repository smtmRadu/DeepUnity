using DeepUnity.Modules;
using System.Collections.Generic;
using System.Text;
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

            public Qwen3Model()
            {
                this.pad_idx = Qwen3Modeling.Qwen3Config.PAD_IDX;
                this.vocab_size = Qwen3Modeling.Qwen3Config.VOCAB_SIZE;
                this.embed_tokens = new Embedding(
                    Qwen3Modeling.Qwen3Config.VOCAB_SIZE,
                    Qwen3Modeling.Qwen3Config.HIDDEN_SIZE,
                    Qwen3Modeling.Qwen3Config.PAD_IDX,
                    init: InitType.Zeros);
                this.layers = new();
                for (int i = 0; i < Qwen3Modeling.Qwen3Config.NUM_LAYERS; i++)
                {
                    layers.Add(new Qwen3Modeling.Qwen3DecoderLayer(i));
                }
                this.norm = new Qwen3RMSNorm(Qwen3Modeling.Qwen3Config.HIDDEN_SIZE, Qwen3Config.RMS_EPS); //new RMSNorm(Qwen3Modeling.Qwen3Config.HIDDEN_SIZE, Qwen3Modeling.Qwen3Config.RMS_EPS, elementwise_affine: true);
                this.rope = new RotaryPositionalEmbeddings(Qwen3Modeling.Qwen3Config.HIDDEN_SIZE, Qwen3Modeling.Qwen3Config.CONTEXT_LENGTH, Qwen3Modeling.Qwen3Config.ROPE_THETA);


            }

            public Tensor Predict(Tensor input_ids, Tensor attention_mask = null)
            {
                Tensor hid = embed_tokens.Predict(input_ids);

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
        public Qwen3ForCausalLM()
        {
            this.vocab_size = Qwen3Modeling.Qwen3Config.VOCAB_SIZE;
            this.hidden_size = Qwen3Modeling.Qwen3Config.HIDDEN_SIZE;
            model = new Qwen3Modeling.Qwen3Model();


            // lm_head = TensorGPU.Zeros(model.embed_tokens.embeddings.Shape);// new ComputeBuffer(model.embed_tokens.embeddings.Count(), 4);
            lm_head = new ComputeBuffer(model.embed_tokens.embeddings.Shape[0] * model.embed_tokens.embeddings.Shape[1], 4, ComputeBufferType.Structured);
             //Debug.Log("LmHead loaded to gpu!");
            UnityEditor.EditorApplication.playModeStateChanged += DeallocQwen;
            // in_features: Qwen3Config.HIDDEN_SIZE,
            // out_features: Qwen3Config.VOCAB_SIZE,
            // weight_init: InitType.Zeros,
            // bias: false, device: Device.GPU);
        }

        // ~Qwen3ForCausalLM()
        // {
        //     Debug.Log("Lm head disposed from within");
        //     lm_head.Dispose();
        // }
        private void DeallocQwen(UnityEditor.PlayModeStateChange state)
        {
            if (state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
            {
                foreach (var item in model.layers)
                {
                    item.mlp.weights_Cb.Release();
                }
        
                lm_head.Release();
                Debug.Log("Qwen3 released from gpu");
            }
        }
        public int ParameterCount()
        {
            int @params  = model.ParameterCount();
            // UnityEngine.Debug.Log($"model.lm_head:{lm_head.)}");
            return @params;
        }
        public Tensor Predict(Tensor input_ids, Tensor attn_mask = null)
        {
            int seq_len = input_ids.Size(-1);
            bool is_batched = input_ids.Rank == 3;
            int batch_size = is_batched ? input_ids.Size(-3) : 1;
            Benckmark.Start();
            Tensor hid = model.Predict(input_ids, attn_mask);
            Benckmark.Stop("Layers");

            
            Benckmark.Start(); // lm head takes 0.01645s per 1 token.

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
            
            Benckmark.Stop("lm head");
            return output_probs;
        }
        public Dictionary<string, string> Chat(List<Dictionary<string, string>> messages, Qwen2TokenizerFast tokenizer, float temperature = 0.7f, int top_k = 20, float top_p = 0.95f, float min_p = 0)
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
        public void LoadParameters()
        {

        }
    }
}

