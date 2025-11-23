using DeepUnity.Modules;
using Unity.VisualScripting;


namespace DeepUnity
{
    namespace Qwen3Modeling
    {
        // OK SO NEVER USE TensorGPU (use only compute buffers)
        public class Qwen3DecoderLayer
        {
            private int layer_idx;
            public Qwen3MLP mlp;
            public Qwen3GQA self_attn;
            public Qwen3RMSNorm input_layernorm;
            public Qwen3RMSNorm post_attention_layernorm;

            public Qwen3DecoderLayer(int layer_index, RotaryPositionalEmbeddings rope, string params_path)
            {
                this.layer_idx = layer_index;
                this.mlp = new Qwen3MLP(
                    hidden_size:Qwen3Modeling.Qwen3Config.HIDDEN_SIZE,
                    intermediate_size:Qwen3Modeling.Qwen3Config.MLP_INTERMEDIATE_SIZE,
                    params_path + $"/layer_{layer_idx}");
                this.self_attn = new Qwen3GQA(
                    embed_dim: Qwen3Modeling.Qwen3Config.HIDDEN_SIZE,
                    num_heads_q: Qwen3Modeling.Qwen3Config.HEADS_Q,
                    num_heads_kv: Qwen3Modeling.Qwen3Config.HEADS_KV,
                    expansion_factor: Qwen3Modeling.Qwen3Config.ATTN_EXPANSION_FACTOR,
                    qk_norm_eps: Qwen3Modeling.Qwen3Config.RMS_EPS,
                    rope: rope,
                    layer_params_path: params_path + $"/layer_{layer_idx}");
                input_layernorm = new Qwen3RMSNorm(
                    num_features: Qwen3Modeling.Qwen3Config.HIDDEN_SIZE,
                    eps: Qwen3Modeling.Qwen3Config.RMS_EPS,
                    params_path + $"/layer_{layer_idx}/input_layernorm.bin");
                post_attention_layernorm = new Qwen3RMSNorm(
                    num_features: Qwen3Modeling.Qwen3Config.HIDDEN_SIZE,
                    eps: Qwen3Modeling.Qwen3Config.RMS_EPS,
                    params_path + $"/layer_{layer_idx}/post_attention_layernorm.bin");
            }
            //public Qwen3DecoderLayer()

            public Tensor Predict(Tensor hidden_states, Tensor attention_mask = null)
            {
                // self attn
                var skip = hidden_states.Clone() as Tensor;
                UnityEngine.Debug.Log($"layer_{layer_idx}.input_ln_in:" + hidden_states);
                hidden_states = input_layernorm.Predict(hidden_states);
                UnityEngine.Debug.Log($"layer_{layer_idx}.input_ln_out:" + hidden_states);
                hidden_states = self_attn.Predict(hidden_states); // here to set the attention mask for this layer if not null.
                UnityEngine.Debug.Log($"layer_{layer_idx}.self_attn:" + hidden_states);
                hidden_states = hidden_states + skip;

                // mlp
                skip = hidden_states.Clone() as Tensor;
                hidden_states = post_attention_layernorm.Predict(hidden_states);
                UnityEngine.Debug.Log($"layer_{layer_idx}.post_self_attn_ln:" + hidden_states);
                hidden_states = this.mlp.Predict(hidden_states);
                UnityEngine.Debug.Log($"layer_{layer_idx}.mlp:" + hidden_states);
                hidden_states = hidden_states + skip;
                return hidden_states;
            }

            public int ParameterCount()
            {
                int @params = 0;

                @params += mlp.weights.count;
                @params += self_attn.q_norm.gamma.Length + self_attn.k_norm.gamma.Length;
                @params += self_attn.W_QKV.count + self_attn.W_O.count;
                @params += input_layernorm.gamma.Length + post_attention_layernorm.gamma.Length;

                return @params;
            }
        }
    }
}
