
using DeepUnity.MobileLLMModeling;

namespace DeepUnity
{
    namespace MobileLLMModeling
    {
        public class LlamaDecoderLayer
        {
            private int layer_idx;
           
            public LlamaMLP mlp;
            public LlamaGQA self_attn;
            public LlamaRMSNorm input_layernorm;
            public LlamaRMSNorm post_attention_layernorm;

            public LlamaDecoderLayer(int layer_index, RotaryPositionalEmbeddings rope, string params_path)
            {
                this.layer_idx = layer_index;
                this.mlp = new LlamaMLP(
                    hidden_size: LlamaConfig.HIDDEN_SIZE,
                    intermediate_size: LlamaConfig.MLP_INTERMEDIATE_SIZE,
                    params_path + $"/layer_{layer_idx}");



                this.self_attn = new LlamaGQA(
                    embed_dim: LlamaConfig.HIDDEN_SIZE,
                    num_heads_q: LlamaConfig.HEADS_Q,
                    num_heads_kv: LlamaConfig.HEADS_KV,
                    expansion_factor: LlamaConfig.ATTN_EXPANSION_FACTOR,
                    weight_init: InitType.Zeros,
                    rope: rope,
                    layer_params_path: params_path + $"/layer_{layer_idx}");

                input_layernorm = new LlamaRMSNorm(
                    num_features: LlamaConfig.HIDDEN_SIZE,
                    eps: LlamaConfig.RMS_EPS,
                    params_path + $"/layer_{layer_idx}/input_layernorm.bin");
                post_attention_layernorm = new LlamaRMSNorm(
                    num_features: LlamaConfig.HIDDEN_SIZE,
                    eps: LlamaConfig.RMS_EPS,
                    params_path + $"/layer_{layer_idx}/post_attention_layernorm.bin");

            }

            public Tensor Predict(Tensor hidden_states, Tensor attention_mask = null)
            {
                // self attn
                var skip = hidden_states.Clone() as Tensor;
                hidden_states = input_layernorm.Predict(hidden_states);
                UnityEngine.Debug.Log("input_layernorm_OUT:" + hidden_states);
                hidden_states = self_attn.Predict(hidden_states);
                UnityEngine.Debug.Log("self_attn_OUT:" + hidden_states);
                hidden_states = hidden_states + skip;


                // mlp
                skip = hidden_states.Clone() as Tensor;
                hidden_states = post_attention_layernorm.Predict(hidden_states);
                UnityEngine.Debug.Log("post_attention_OUT:" + hidden_states);
                hidden_states = this.mlp.Predict(hidden_states);
                UnityEngine.Debug.Log("mlp_OUT:" + hidden_states);
                hidden_states = hidden_states + skip;
                return hidden_states;
            }

            public int ParameterCount()
            {
                int @params = 0;
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.self_attn.qkv_proj:{gqa.W_QKV.count}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.self_attn.o_proj:{gqa.W_O.count}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.self_attn.q_norm:{gqa.q_rmsn.gamma.Length}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.self_attn.k_norm:{gqa.k_rmsn.gamma.Length}");
                // 
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.mlp.u_proj+g_proj+d_proj:{mlp.weights.count}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.input_layernorm:{input_ln.gamma.Length}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.post_attention_layernorm:{post_attention_layernorm.gamma.Length}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.pre_feedforward_layernorm:{pre_feedforward_layernorm.gamma.Length}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.post_feedforward_layernorm:{post_feedforward_layernorm.gamma.Length}");

                // @params += mlp.weights.Count();
                @params += mlp.weights.count;
                @params += self_attn.W_QKV.count + self_attn.W_O.count;
                @params +=
                    input_layernorm.gamma.Length +
                    post_attention_layernorm.gamma.Length;

                return @params;
            }
        }
    }
}
