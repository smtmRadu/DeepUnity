
using DeepUnity.Modules;

namespace DeepUnity
{
    namespace Gemma3Modeling
    {
        public class EmbeddingGemmaDecoderLayer
        {
            private int layer_idx;

            public Gemma3MLP mlp;
            public Gemma3GQA self_attn;
            public Gemma3RMSNorm input_layernorm;
            public Gemma3RMSNorm post_attention_layernorm;
            public Gemma3RMSNorm pre_feedforward_layernorm;
            public Gemma3RMSNorm post_feedforward_layernorm;

            public EmbeddingGemmaDecoderLayer(int layer_index, RotaryPositionalEmbeddings rope, string params_path)
            {
                this.layer_idx = layer_index;
                this.mlp = new Gemma3MLP(
                    hidden_size: EmbeddingGemmaConfig.HIDDEN_SIZE,
                    intermediate_size: EmbeddingGemmaConfig.MLP_INTERMEDIATE_SIZE,
                    params_path + $"/layer_{layer_idx}");
                this.self_attn = new Gemma3GQA(embed_dim: EmbeddingGemmaConfig.HIDDEN_SIZE,
                    num_heads_q: EmbeddingGemmaConfig.HEADS_Q,
                    num_heads_kv: EmbeddingGemmaConfig.HEADS_KV,
                    expansion_factor: EmbeddingGemmaConfig.ATTN_EXPANSION_FACTOR,
                    qk_norm_eps: EmbeddingGemmaConfig.RMS_EPS,
                    sliding_window: EmbeddingGemmaConfig.layer_types[layer_index] == GemmaLayerType.SlidingWindowAttention ? EmbeddingGemmaConfig.SLIDING_WINDOW : -1,
                    query_pre_attention_scalar: EmbeddingGemmaConfig.QUERY_PRE_ATTENTION_SCALAR,
                    softcap: EmbeddingGemmaConfig.ATTN_LOGIT_SOFTCAPPING,
                    is_causal: !EmbeddingGemmaConfig.USE_BIDIRECTIONAL_ATTENTION,
                    rope: rope,
                    layer_params_path: params_path + $"/layer_{layer_idx}");
                input_layernorm = new Gemma3RMSNorm(
                    num_features: EmbeddingGemmaConfig.HIDDEN_SIZE,
                    eps: EmbeddingGemmaConfig.RMS_EPS,
                    params_path + $"/layer_{layer_idx}/input_layernorm.bin");
                post_attention_layernorm = new Gemma3RMSNorm(
                    num_features: EmbeddingGemmaConfig.HIDDEN_SIZE,
                    eps: EmbeddingGemmaConfig.RMS_EPS,
                    params_path + $"/layer_{layer_idx}/post_attention_layernorm.bin");
                pre_feedforward_layernorm = new Gemma3RMSNorm(
                    num_features: EmbeddingGemmaConfig.HIDDEN_SIZE,
                    eps: EmbeddingGemmaConfig.RMS_EPS,
                    params_path + $"/layer_{layer_idx}/pre_feedforward_layernorm.bin");
                post_feedforward_layernorm = new Gemma3RMSNorm(
                    num_features: EmbeddingGemmaConfig.HIDDEN_SIZE,
                    eps: EmbeddingGemmaConfig.RMS_EPS,
                    params_path + $"/layer_{layer_idx}/post_feedforward_layernorm.bin");

            }

            public Tensor Predict(Tensor hidden_states, Tensor attention_mask = null)
            {
                // self attn

                var skip = hidden_states.Clone() as Tensor;
                // UnityEngine.Debug.Log($"layer_{layer_idx}.INPUT:" + hidden_states);
                hidden_states = input_layernorm.Predict(hidden_states);
                // UnityEngine.Debug.Log($"layer_{layer_idx}.input_ln:" + hidden_states);
                hidden_states = self_attn.Predict(hidden_states); // here to set the attention mask for this layer if not null.
                // UnityEngine.Debug.Log($"layer_{layer_idx}.self_attn:" + hidden_states);
                hidden_states = post_attention_layernorm.Predict(hidden_states);
                // UnityEngine.Debug.Log($"layer_{layer_idx}.post_self_attn_ln:" + hidden_states);
                hidden_states = hidden_states + skip;


                // mlp
                skip = hidden_states.Clone() as Tensor;
                hidden_states = pre_feedforward_layernorm.Predict(hidden_states);
                // UnityEngine.Debug.Log($"layer_{layer_idx}.pre_feedforward_ln:" + hidden_states);
                hidden_states = this.mlp.Predict(hidden_states);
                // UnityEngine.Debug.Log($"layer_{layer_idx}.mlp:" + hidden_states);
                hidden_states = post_feedforward_layernorm.Predict(hidden_states);
                // UnityEngine.Debug.Log($"layer_{layer_idx}.post_feedforward_ln:" + hidden_states);
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
                @params += self_attn.q_norm.gamma.Length + self_attn.k_norm.gamma.Length;
                @params += self_attn.W_QKV.count + self_attn.W_O.count;
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
