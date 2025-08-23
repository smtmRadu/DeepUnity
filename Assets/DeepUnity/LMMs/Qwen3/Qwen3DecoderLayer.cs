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
            public GroupedQueryAttention gqa;
            public Qwen3RMSNorm input_ln;
            public Qwen3RMSNorm post_attention_ln;

            public Qwen3DecoderLayer(int layer_index)
            {
                this.layer_idx = layer_index;
                this.mlp = new Qwen3MLP(
                    hidden_size:Qwen3Modeling.Qwen3Config.HIDDEN_SIZE,
                    intermediate_size:Qwen3Modeling.Qwen3Config.MLP_INTERMEDIATE_SIZE);
                this.gqa = new GroupedQueryAttention(embed_dim: Qwen3Modeling.Qwen3Config.HIDDEN_SIZE,
                    num_heads_q: Qwen3Modeling.Qwen3Config.HEADS_Q,
                    num_heads_kv: Qwen3Modeling.Qwen3Config.HEADS_KV,
                    expansion_factor: Qwen3Modeling.Qwen3Config.ATTN_EXPANSION_FACTOR,
                    is_causal: true,
                    dropout: 0,
                    qk_norm: true,
                    qk_norm_eps: 1e-6f,
                    use_rope: true,
                    rope_max_seq_len: Qwen3Modeling.Qwen3Config.CONTEXT_LENGTH,
                    rope_theta: Qwen3Modeling.Qwen3Config.ROPE_THETA,
                    weight_init: InitType.Zeros,
                    device: Device.GPU);
                input_ln = new Qwen3RMSNorm(
                    num_features: Qwen3Modeling.Qwen3Config.HIDDEN_SIZE,
                    eps: Qwen3Modeling.Qwen3Config.RMS_EPS);
                post_attention_ln = new Qwen3RMSNorm(
                    num_features: Qwen3Modeling.Qwen3Config.HIDDEN_SIZE,
                    eps: Qwen3Modeling.Qwen3Config.RMS_EPS);
            }
            //public Qwen3DecoderLayer()

            public Tensor Predict(Tensor hidden_states, Tensor attention_mask = null)
            {
                // self attn
                var skip = hidden_states.Clone() as Tensor;
                hidden_states = input_ln.Predict(hidden_states);
                hidden_states = gqa.Predict(hidden_states); // here to set the attention mask for this layer if not null.
                hidden_states = hidden_states + skip;

                // mlp
                skip = hidden_states.Clone() as Tensor;

                hidden_states = post_attention_ln.Predict(hidden_states);
                hidden_states = this.mlp.Predict(hidden_states);
                hidden_states = hidden_states + skip;
                return hidden_states;
            }

            public int ParameterCount()
            {
                int @params = 0;
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.self_attn.qkv_proj:{gqa.W_QKV.weights.Shape.ToCommaSeparatedString()}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.self_attn.o_proj:{gqa.W_O.weights.Shape.ToCommaSeparatedString()}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.self_attn.q_norm:{gqa.q_rmsn.gamma.Shape.ToCommaSeparatedString()}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.self_attn.k_norm:{gqa.k_rmsn.gamma.Shape.ToCommaSeparatedString()}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.mlp.u_proj+g_proj+d_proj:{mlp.weights.Length}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.input_layernorm:{input_ln.gamma.Length}");
                // UnityEngine.Debug.Log($"model.layers.{layer_idx}.post_attention_layernorm:{post_attention_ln.gamma.Length}");


                // @params += mlp.weights.Count();
                @params += mlp.weights_Cb.count;
                @params += gqa.q_rmsn.gamma.Count() + gqa.k_rmsn.gamma.Count();
                @params += gqa.W_QKV.weights.Count() + gqa.W_O.weights.Count();
                @params += input_ln.gamma.Length + post_attention_ln.gamma.Length;

                return @params;
            }
        }
    }
}
