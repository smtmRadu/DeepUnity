using DeepUnity.Gemma3Modelling;
using DeepUnity.Modules;

namespace DeepUnity
{
    namespace Gemma3Modeling
    {
        // OK SO NEVER USE TensorGPU (use only compute buffers)
        public class Gemma3DecoderLayer
        {
            private int layer_idx;
           
            public Gemma3MLP mlp;
            public Gemma3GQA gqa;
            public Gemma3RMSNorm input_layernorm;
            public Gemma3RMSNorm post_attention_layernorm;
            public Gemma3RMSNorm pre_feedforward_layernorm;
            public Gemma3RMSNorm post_feedforward_layernorm;

            public Gemma3DecoderLayer(int layer_index, RotaryPositionalEmbeddings rope, string params_path)
            {
                this.layer_idx = layer_index;
                this.mlp = new Gemma3MLP(
                    hidden_size: Gemma3Config.HIDDEN_SIZE,
                    intermediate_size: Gemma3Config.MLP_INTERMEDIATE_SIZE,
                    params_path + $"/layer_{layer_idx}");
                this.gqa = new Gemma3GQA(embed_dim: Gemma3Config.HIDDEN_SIZE,
                    num_heads_q: Gemma3Config.HEADS_Q,
                    num_heads_kv: Gemma3Config.HEADS_KV,
                    expansion_factor: Gemma3Config.ATTN_EXPANSION_FACTOR,
                    qk_norm_eps: Gemma3Config.RMS_EPS,
                    weight_init: InitType.Zeros,
                    device: Device.GPU,
                    rope: rope,
                    layer_params_path: params_path + $"/layer_{layer_idx}");
                input_layernorm = new Gemma3RMSNorm(
                    num_features: Gemma3Config.HIDDEN_SIZE,
                    eps: Gemma3Config.RMS_EPS,
                    params_path + $"/layer_{layer_idx}/input_layernorm.bin");
                post_attention_layernorm = new Gemma3RMSNorm(
                    num_features: Gemma3Config.HIDDEN_SIZE,
                    eps: Gemma3Config.RMS_EPS,
                    params_path + $"/layer_{layer_idx}/post_attention_layernorm.bin");
                pre_feedforward_layernorm = new Gemma3RMSNorm(
                    num_features: Gemma3Config.HIDDEN_SIZE,
                    eps: Gemma3Config.RMS_EPS,
                    params_path + $"/layer_{layer_idx}/pre_feedforward_layernorm.bin");
                post_feedforward_layernorm = new Gemma3RMSNorm(
                    num_features: Gemma3Config.HIDDEN_SIZE,
                    eps: Gemma3Config.RMS_EPS,
                    params_path + $"/layer_{layer_idx}/post_feedforward_layernorm.bin");

            }
            
            public Tensor Predict(Tensor hidden_states, Tensor attention_mask = null)
            {
                // self attn
                var skip = hidden_states.Clone() as Tensor;
                hidden_states = input_layernorm.Predict(hidden_states);
                hidden_states = gqa.Predict(hidden_states); // here to set the attention mask for this layer if not null.
                hidden_states = post_attention_layernorm.Predict(hidden_states);
                hidden_states = hidden_states + skip ;

                // mlp
                skip = hidden_states.Clone() as Tensor;
                hidden_states = pre_feedforward_layernorm.Predict(hidden_states);
                hidden_states = this.mlp.Predict(hidden_states);
                hidden_states = post_feedforward_layernorm.Predict(hidden_states);
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
                @params += gqa.q_norm.gamma.Length + gqa.k_norm.gamma.Length;
                @params += gqa.W_QKV.count + gqa.W_O.count;
                @params += 
                    input_layernorm.gamma.Length + 
                    post_attention_layernorm.gamma.Length + 
                    pre_feedforward_layernorm.gamma.Length + 
                    post_feedforward_layernorm.gamma.Length;

                return @params;
            }
        }
    }
}
