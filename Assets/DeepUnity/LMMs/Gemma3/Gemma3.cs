using DeepUnity.Gemma3Modelling;
using DeepUnity.Modules;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

// NOTES
// TODO: set gqa weights as compute buffers (so they live in gpu)
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

            public Gemma3Model()
            {
                this.eos_idx = Gemma3Config.EOS_IDX;
                this.bos_idx = Gemma3Config.BOS_IDX;
                this.pad_idx = Gemma3Config.PAD_IDX;
                this.vocab_size = Gemma3Config.VOCAB_SIZE;
                this.embed_tokens = new Embedding(
                    Gemma3Config.VOCAB_SIZE,
                    Gemma3Config.HIDDEN_SIZE,
                    Gemma3Config.PAD_IDX,
                    init: InitType.Zeros);
                this.layers = new();
                for (int i = 0; i < Gemma3Config.NUM_LAYERS; i++)
                {
                    layers.Add(new Gemma3DecoderLayer(i));
                }
                
                this.norm = new Gemma3RMSNorm(Gemma3Config.HIDDEN_SIZE, Gemma3Config.RMS_EPS); //new RMSNorm(Qwen3Modeling.Qwen3Config.HIDDEN_SIZE, Qwen3Modeling.Qwen3Config.RMS_EPS, elementwise_affine: true);

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

    public class Gemma3ForCausalLM
    {
        private int vocab_size;
        private int hidden_size;
        private Gemma3Modeling.Gemma3Model model;

        private ComputeBuffer lm_head;
        private List<Dictionary<string, string>> conversation_cache;
        public Gemma3ForCausalLM()
        {
            this.vocab_size = Gemma3Config.VOCAB_SIZE;
            this.hidden_size = Gemma3Config.HIDDEN_SIZE;
            model = new Gemma3Modeling.Gemma3Model();


            // lm_head = TensorGPU.Zeros(model.embed_tokens.embeddings.Shape);// new ComputeBuffer(model.embed_tokens.embeddings.Count(), 4);
            lm_head = new ComputeBuffer(model.embed_tokens.embeddings.Shape[0] * model.embed_tokens.embeddings.Shape[1], 4, ComputeBufferType.Structured);
            //Debug.Log("LmHead loaded to gpu!");
            UnityEditor.EditorApplication.playModeStateChanged += DeallocGemma;
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
                Debug.Log("Gemma3 released from gpu");
            }
        }
        public int ParameterCount()
        {
            int @params = model.ParameterCount();
            // UnityEngine.Debug.Log($"model.lm_head:{lm_head.count}");
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
        public void LoadParameters()
        {

        }
    }
}

