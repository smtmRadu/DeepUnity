using DeepUnity.Activations;
using DeepUnity.Gemma3Modeling;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// Input: <b>(B, L, E)</b> or <b>(L, E)</b>.<br></br>
    /// Output: <b>(B, L, E)</b> or <b>(L, E)</b>.<br></br>
    /// where B = batch_size, L = sequence_length, E = embed_dim
    /// </summary>
    [Serializable]
    public class Gemma3GQA
    {
        private float? attn_logit_softcapping;
        private float scaling;
        private int sliding_window;
        private int qkv_proj_dim;
        private int embedding_dim;
        private int inner_embedding_dim;
        private int num_heads_q;
        private int num_heads_kv;
        private int head_dim;
        private bool is_causal;
        private bool qk_norm;

        

        
        public ComputeBuffer W_QKV;
        public ComputeBuffer W_O;
        public ComputeBuffer input_buffer;
        public ComputeBuffer qkv_proj_buffer;
        public ComputeBuffer q_buffer;
        public ComputeBuffer attended_values_buffer;
        public ComputeBuffer output_buffer;
        [SerializeField] public RotaryPositionalEmbeddings rope;
        [SerializeField] public Gemma3RMSNorm q_norm;
        [SerializeField] public Gemma3RMSNorm k_norm;
        [SerializeField] private Softmax softmax;

        
        
        private bool buildKVCache = false;  // Backing field
        private Tensor attentionMask = null;
        private int cachedTokensNum = 0;
        private List<Tensor> kCache; // Rope + Norm
        private List<Tensor> vCache;
        
        public bool IsInitialized { get; private set; } = false;
        public bool BuildKVCache
        {
            set
            {
                if (value)
                {
                    cachedTokensNum = 0;
                    kCache = new List<Tensor>();
                    vCache = new List<Tensor>();
                    buildKVCache = true;
                }
                else
                {
                    cachedTokensNum = 0;
                    kCache = null;
                    vCache = null;
                    buildKVCache = false;
                }
            }
        }// when build kv cache is ON, Q and K (roped) will be cached and the model must receive one input(B,1,E) at a time (only 1 elem)

        /// <summary>
        ///  <b>(B, L)</b> or <b>(L)</b>.<br></br>
        /// </summary>
        public Tensor AttentionMask { get => attentionMask; set => attentionMask = value; } // This is the second input that must have entered in the forward function.
        public Tensor KCache
        {
            get 
            {
                if(kCache == null)
                    return null;
                if(kCache.Count == 1)
                    return kCache[0];
                return Tensor.Concat(0, kCache.ToArray());
            }
            set
            {
                if(value.Rank != 3)
                    throw new ArgumentException("KV Cache tensor must be of shape (L, 1, H_dim)");
                buildKVCache = true;
                cachedTokensNum = value.Size(0);
                kCache = new List<Tensor>(){value};

            }
        }
        public Tensor VCache
        {
            get 
            {
                if(vCache == null)
                    return null;
                if(vCache.Count == 1)
                    return vCache[0];
                return Tensor.Concat(0, vCache.ToArray());
            }
            set
            {
                if(value.Rank != 3)
                    throw new ArgumentException("KV Cache tensor must be of shape (L, 1, H_dim)");
                buildKVCache = true;
                cachedTokensNum = value.Size(0);
                vCache = new List<Tensor>(){value};
            }
        }

        /// <summary>
        /// Input: <b>(B, L, E)</b> or <b>(L, E)</b>.<br></br>
        /// Output: <b>(B, L, E)</b> or <b>(L, E)</b>.<br></br>
        /// where B = batch_size, L = sequence_length, E = embed_dim
        /// </summary>
        /// <param name="embed_dim">Total dimension of the model.</param>
        /// <param name="num_heads_q">Number of parallel Query attention heads. Note that embed_dim will be split across num_heads (i.e. each head will have dimension embed_dim // num_heads).</param>
        /// <param name="num_heads_kv">Number of parallel Key and Value attention heads For MHA num_heads_q == num_heads_kv.</param>
        /// <param name="is_causal"></param>
        /// <param name="dropout">Dropout probability on attn_output_weights. Default: 0.0 (no dropout).</param>
        /// <param name="qk_norm">Apply RMSNorm on Q and K before SDPA</param>
        /// <param name="use_rope"></param>
        /// <param name="rope_max_seq_len"></param>
        /// <param name="rope_theta"></param>
        /// <param name="weight_init"></param>
        /// <param name="device"></param>
        public Gemma3GQA(
            int embed_dim,
            int num_heads_q,
            int? num_heads_kv = null,
            float expansion_factor = 1f,
            float qk_norm_eps = 1e-6f,
            int sliding_window = -1,
            bool is_causal = true,
            float query_pre_attention_scalar = 256f,
            float? softcap = null,
            RotaryPositionalEmbeddings rope = null,
            string layer_params_path = null)
        {
            if (embed_dim % num_heads_q != 0)
                throw new ArgumentException("embed_dim must be divisible by num_heads_q.");

            
            this.embedding_dim = embed_dim;
            this.inner_embedding_dim = (int)(embedding_dim * expansion_factor);
            this.num_heads_q = num_heads_q;
            if (num_heads_kv is null)
                this.num_heads_kv = num_heads_q;
            else
                this.num_heads_kv = num_heads_kv.Value;

            if (num_heads_q % num_heads_kv != 0)
                throw new ArgumentException("num_heads_q must be an integer multiple of num_heads_kv for GQA.");

            this.is_causal = is_causal;
            // UnityEngine.Debug.Log("is_causal " +  is_causal);
            this.head_dim = this.inner_embedding_dim / num_heads_q;

            qkv_proj_dim = this.inner_embedding_dim + 2 * (this.inner_embedding_dim * this.num_heads_kv / num_heads_q);

            W_QKV = new ComputeBuffer(this.embedding_dim * qkv_proj_dim, 4, ComputeBufferType.Structured);
            W_O = new ComputeBuffer(this.inner_embedding_dim * this.embedding_dim, 4, ComputeBufferType.Structured);
            
            this.qk_norm = true;
            // line 182 https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
            this.q_norm = new Gemma3RMSNorm(this.head_dim, eps: qk_norm_eps, layer_params_path + "/self_attn_q_norm.bin");
            this.k_norm = new Gemma3RMSNorm(this.head_dim, eps: qk_norm_eps, layer_params_path + "/self_attn_k_norm.bin");
            this.sliding_window = sliding_window;

            this.scaling = MathF.Pow(query_pre_attention_scalar, -0.5f);
            this.attn_logit_softcapping = softcap;
            this.softmax = new Softmax();
            this.rope = rope;
           
            if (!string.IsNullOrEmpty(layer_params_path))
            {
                _ = LoadWeightsAsync(layer_params_path);
            }
        }
        private Gemma3GQA() { }
        ~Gemma3GQA()
        {
           W_O.Release();
           W_QKV.Release();
        }

        private async Task LoadWeightsAsync(string path)
        {
            Task<float[]>[] tasks = new Task<float[]>[4];
            tasks[0] = Task.Run(() => Utils.ReadWeights(path + "/self_attn_q_proj.bin", this.embedding_dim * this.inner_embedding_dim));
            tasks[1] = Task.Run(() => Utils.ReadWeights(path + "/self_attn_k_proj.bin", this.embedding_dim * this.inner_embedding_dim * this.num_heads_kv / this.num_heads_q));
            tasks[2] = Task.Run(() => Utils.ReadWeights(path + "/self_attn_v_proj.bin", this.embedding_dim * this.inner_embedding_dim * this.num_heads_kv / this.num_heads_q));
            tasks[3] = Task.Run(() => Utils.ReadWeights(path + "/self_attn_o_proj.bin", this.inner_embedding_dim * this.embedding_dim));
            float[][] results = await Task.WhenAll(tasks);
            float[] flat_qkv = new float[this.embedding_dim * this.qkv_proj_dim];
            try
            {
                Array.Copy(results[0], 0, flat_qkv,
                0,
                this.embedding_dim * this.inner_embedding_dim);
                Array.Copy(results[1], 0, flat_qkv,
                    this.embedding_dim * this.inner_embedding_dim,
                    this.embedding_dim * this.inner_embedding_dim * this.num_heads_kv / this.num_heads_q);
                Array.Copy(results[2], 0, flat_qkv,
                    this.embedding_dim * this.inner_embedding_dim + this.embedding_dim * this.inner_embedding_dim * this.num_heads_kv / this.num_heads_q,
                    this.embedding_dim * this.inner_embedding_dim * this.num_heads_kv / this.num_heads_q);
            }
            catch (Exception ex)
            {

                Debug.LogException(ex);
            }
            

            W_QKV.SetData(flat_qkv);
            W_O.SetData(results[3]);

            IsInitialized = true;
            // ConsoleMessage.Info($"Loaded {path}/self_attn");
        }
        public Tensor Predict(Tensor x)
        {
            if (x.Rank > 3 || x.Rank < 2)
                throw new ArgumentException($"GQA input must be of shape (B, L, E) or (L, E). Received tensor of rank {x.Rank} | shape ({x.Shape.ToCommaSeparatedString()}).");
            bool batched = x.Rank == 3;
            int B = batched ? x.Size(-3) : 1;
            int L_x = x.Size(-2);

            // ========================================================== QKV PROJ =========================================================================
            ComputeShader cs = DeepUnityMeta.GQAInferenceCS;
            int qkvKernel = cs.FindKernel("QKVProj");
            cs.SetBuffer(qkvKernel, "W_QKV", W_QKV);

            ComputeBuffer xBuff = new ComputeBuffer(x.Count(), 4, ComputeBufferType.Structured);
            xBuff.SetData(x.ToArray());
            // Debug.Log("X (gemma):" + x);
            cs.SetBuffer(qkvKernel, "X", xBuff);

            ComputeBuffer qkvBuff = new ComputeBuffer(B * L_x * qkv_proj_dim, 4, ComputeBufferType.Structured);
            cs.SetBuffer(qkvKernel, "QKV", qkvBuff);

            cs.SetInt("batch_size", B);
            cs.SetInt("sequence_length_q", L_x);
            cs.SetInt("embedding_dim", embedding_dim);
            cs.SetInt("qkv_proj_dim", qkv_proj_dim);
            
            cs.Dispatch(qkvKernel, 
                B, 
                (L_x + 7) / 8, 
                (qkv_proj_dim + 31) / 32);

            Tensor QKV = batched ? Tensor.Constant(qkvBuff, B, L_x, qkv_proj_dim) : Tensor.Constant(qkvBuff, L_x, qkv_proj_dim);
            qkvBuff.Release();
            xBuff.Release();

            // UnityEngine.Debug.Log("QKV (gemma):" + QKV);
            //Debug.Log(QKV);
            Tensor Q = batched ? Tensor.Zeros(B, L_x, this.num_heads_q, this.head_dim) : Tensor.Zeros(L_x, this.num_heads_q, this.head_dim);
            Tensor K = batched ? Tensor.Zeros(B, L_x, this.num_heads_kv, this.head_dim) : Tensor.Zeros(L_x, this.num_heads_kv, this.head_dim);
            Tensor V = batched ? Tensor.Zeros(B, L_x, this.num_heads_kv, this.head_dim) : Tensor.Zeros(L_x, this.num_heads_kv, this.head_dim);

            // Unpack QKV
            if (L_x > 1)
            {
                Parallel.For(0, L_x, l =>
                {
                    for (int b = 0; b < B; b++)
                    {
                        for (int h = 0; h < num_heads_q; h++)
                        {
                            for (int e = 0; e < this.head_dim; e++)
                            {
                                Q[b, l, h, e] = QKV[b, l, h * head_dim + e];
                            }
                        }
                        for (int h = 0; h < num_heads_kv; h++)
                        {
                            for (int e = 0; e < this.head_dim; e++)
                            {
                                K[b, l, h, e] = QKV[b, l, num_heads_q * head_dim + h * head_dim + e];
                                V[b, l, h, e] = QKV[b, l, num_heads_q * head_dim + num_heads_kv * head_dim + h * head_dim + e];
                            }
                        }
                    }
                });

            }
            else if (L_x == 1)
            {
                for (int b = 0; b < B; b++)
                {
                    for (int h = 0; h < num_heads_q; h++)
                    {
                        for (int e = 0; e < this.head_dim; e++)
                        {
                            Q[b, 0, h, e] = QKV[b, 0, h * head_dim + e];
                        }
                    }
                    for (int h = 0; h < num_heads_kv; h++)
                    {
                        for (int e = 0; e < this.head_dim; e++)
                        {
                            K[b, 0, h, e] = QKV[b, 0, num_heads_q * head_dim + h * head_dim + e];
                            V[b, 0, h, e] = QKV[b, 0, num_heads_q * head_dim + num_heads_kv * head_dim + h * head_dim + e];
                        }
                    }
                }
            }

            //Debug.Log("Q (gemma): " + Q);
            //Debug.Log("K (gemma): " + K);
            //Debug.Log("V (gemma): " + V);

            // ==================================================================== QK Norm =================================================================
            if (qk_norm)
            {
                // implement fast rmsnorm on head_num, head_dim tensors using the qk norm weights

                bool is_batched = Q.Rank == 4;
                Tensor variance = Q.Square().Mean(-1, keepDim: true).Expand(-1, Q.Size(-1));
                Q = Q / Tensor.Sqrt(variance + 1e-6f);

                variance = K.Square().Mean(-1, keepDim: true).Expand(-1, K.Size(-1));
                K = K / Tensor.Sqrt(variance + 1e-6f);
                 
                // affine
                for (int b = 0; b < B; b++)
                {
                    for (int l = 0; l < L_x; l++)
                    {
                        for (int e = 0; e < this.head_dim; e++)
                        {
                            for (int h = 0; h < this.num_heads_q; h++)
                            {
                                Q[b, l, h, e] *= (1f + q_norm.gamma[e]);
                            }

                            for (int h = 0; h < this.num_heads_kv; h++)
                            {
                                K[b, l, h, e] *= (1f + k_norm.gamma[e]);
                            }
                        }
                    }
                }
            }

            //Debug.Log("Q norm (gemma): " + Q);
            //Debug.Log("K norm (gemma): " + K);

            // =============================================================== RoPE + Caching ==========================================================
            if (buildKVCache)
            {
                Q = rope == null ? Q : rope.ApplyRotaryEmbeddings(Q, input_pos: Enumerable.Range(cachedTokensNum, L_x).ToArray(), type: RotaryPositionalEmbeddings.RoPEType.SplitHalf);
                K = rope == null ? K : rope.ApplyRotaryEmbeddings(K, input_pos: Enumerable.Range(cachedTokensNum, L_x).ToArray(), type: RotaryPositionalEmbeddings.RoPEType.SplitHalf);

                kCache.Add(K);
                vCache.Add(V);
                cachedTokensNum += L_x;

                K = Tensor.Concat(-3, kCache.ToArray());// Update the concat function so it allows for tensor with different sizes on the axis that is merged on.... plaese..
                V = Tensor.Concat(-3, vCache.ToArray());
            }
            else
            {
                Q = rope == null ? Q : rope.ApplyRotaryEmbeddings(Q);
                K = rope == null ? K : rope.ApplyRotaryEmbeddings(K);
            }

            // ======================================================  GQA ===================================================================

            int L_k = K.Size(-3); // L_k is the actual length of the sequence. x might have a sequence length of only 1 when doing inference.

            Tensor mask = BuildMask(is_causal, sliding_window, attentionMask, batched ? new int[] { B, num_heads_q, L_x, L_k } : new int[] { num_heads_q, L_x, L_k });
            
            Tensor scores = ComputeAttentionScoresGPU(Q, K, scale: scaling); // B, Hq,  L_x, L_k

            //Debug.Log("Scores (gemma): " + scores);

            if (this.attn_logit_softcapping is not null)
            {
                scores = scores / this.attn_logit_softcapping.Value;
                scores = scores.Tanh(); // in gemma (line 256 - https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py), scores receive a softcap and are passed through tanh 
                scores = scores * this.attn_logit_softcapping.Value;
            }

            // Debug.Log("scores:" + scores);
            Tensor scores_masked = mask == null ? scores : scores + mask;// B, Hq, L_x, L_k
            Tensor attention_weights = softmax.Predict(scores_masked);  // B, Hq, L_x, L_k

            // Debug.Log("Attention weights (gemma): " + attention_weights);
            // ============================================================ V projection ==================================================================
            {
                int Hq = attention_weights.Size(batched ? 1 : 0);
                int L_q = attention_weights.Size(-2);
                int L_v = attention_weights.Size(-1);
                int D = V.Size(-1);
                int Hkv = V.Size(-2);

                cs = DeepUnityMeta.GQAInferenceCS;
                int attendValueskernel = cs.FindKernel("AttendValues");

                ComputeBuffer awBuf = new ComputeBuffer(attention_weights.Count(), 4);
                awBuf.SetData(attention_weights.ToArray());
                cs.SetBuffer(attendValueskernel, "AttentionWeights", awBuf);

                ComputeBuffer vBuf = new ComputeBuffer(V.Count(), 4);
                vBuf.SetData(V.ToArray());
                cs.SetBuffer(attendValueskernel, "V", vBuf);

                int attendedValuesCount = B * L_v * this.inner_embedding_dim;
                ComputeBuffer attndValsBuff = new ComputeBuffer(attendedValuesCount, 4);
                cs.SetBuffer(attendValueskernel, "AttendedValues", attndValsBuff);

                cs.SetInt("batch_size", B);
                cs.SetInt("sequence_length_v", L_v);
                cs.SetInt("num_heads_q", Hq);
                cs.SetInt("num_heads_kv", Hkv);
                cs.SetInt("head_dim", D);

                int gx = (D + 63) / 64;
                int gy = (L_q + 3) / 4;
                int gz = (B * Hq + 3) / 4;
                cs.Dispatch(attendValueskernel, gx, gy, gz);
                awBuf.Release();
                vBuf.Release();


                // ======================================================================= OUTPUT projection ===============================================

                int oProjKernel = cs.FindKernel("OProj");

                cs.SetBuffer(oProjKernel, "AttendedValues", attndValsBuff);
                cs.SetBuffer(oProjKernel, "W_O", W_O);
                cs.SetInt("inner_embedding_dim", inner_embedding_dim);

                ComputeBuffer oBuff = new ComputeBuffer(B * L_q * embedding_dim, 4, ComputeBufferType.Structured);
                cs.SetBuffer(oProjKernel, "O", oBuff);
                cs.Dispatch(oProjKernel, B, (L_q + 3) / 4, (embedding_dim + 31) / 32);


                Tensor output = batched ? Tensor.Constant(oBuff, B, L_q, embedding_dim) : Tensor.Constant(oBuff, L_q, embedding_dim);

                attndValsBuff.Release();
                oBuff.Release();
                return output;
            }

        }

        private static Tensor BuildMask(bool is_causal, int sliding_window, Tensor attention_mask, params int[] mask_shape)
        {
            Tensor mask = Tensor.Zeros(mask_shape);

            bool is_batched = mask_shape.Length == 4;

            if (is_batched)
            {
                int B = mask_shape[0];
                int num_heads = mask_shape[1];
                int L_q = mask_shape[2];  // Query sequence length
                int L_k = mask_shape[3];  // Key sequence length

                for (int b = 0; b < B; b++)
                {
                    for (int h = 0; h < num_heads; h++)
                    {
                        for (int q_pos = 0; q_pos < L_q; q_pos++)
                        {
                            for (int k_pos = 0; k_pos < L_k; k_pos++)
                            {
                                // Causal mask: query position can only attend to key positions up to (cache_len + q_pos)
                                // When decoding with cache, L_k = cache_len + L_q
                                // So query at position q_pos can attend to keys at positions [0, cache_len + q_pos]
                                int current_q_abs_pos = (L_k - L_q) + q_pos;  // Absolute position of this query token

                                if (is_causal && k_pos > current_q_abs_pos)
                                    mask[b, h, q_pos, k_pos] = -1e10f;

                                if (attention_mask is not null)
                                    mask[b, h, q_pos, k_pos] += (1f - attention_mask[b, k_pos]) * -1e25f;

                                if (sliding_window > 0 && Math.Abs(current_q_abs_pos - k_pos) > sliding_window)
                                    mask[b, h, q_pos, k_pos] = -1e10f;
                            }
                        }
                    }
                }
            }
            else
            {
                int num_heads = mask_shape[0];
                int L_q = mask_shape[1];  // Query sequence length
                int L_k = mask_shape[2];  // Key sequence length

                for (int h = 0; h < num_heads; h++)
                {
                    for (int q_pos = 0; q_pos < L_q; q_pos++)
                    {
                        for (int k_pos = 0; k_pos < L_k; k_pos++)
                        {
                            int current_q_abs_pos = (L_k - L_q) + q_pos;

                            if (is_causal && k_pos > current_q_abs_pos)
                                mask[h, q_pos, k_pos] = -1e10f;

                            if (attention_mask is not null)
                                mask[h, q_pos, k_pos] += (1f - attention_mask[k_pos]) * -1e25f;

                            if (sliding_window > 0 && Math.Abs(current_q_abs_pos - k_pos) > sliding_window)
                                mask[h, q_pos, k_pos] = -1e10f;
                        }
                    }
                }
            }

            return mask;
        }
        /// <summary>
        /// Returns (B, num_heads_q, len(Q), len(K))
        /// </summary>
        /// <param name="Q"></param>
        /// <param name="K"></param>
        /// <param name="scale"></param>
        /// <returns></returns>
        public static Tensor ComputeAttentionScoresGPU(Tensor Q, Tensor K, float scale)
        {
            //Debug.Log("Q:" + Q);
            //Debug.Log("K:" + K);
            bool batched = Q.Rank == 4;
            int B = batched ? Q.Size(0) : 1;
            int L_q = Q.Size(-3);
            int L_k = K.Size(-3);
            int Hq = Q.Size(-2);
            int D = Q.Size(-1);
            int Hkv = K.Size(-2);

            ComputeShader cs = DeepUnityMeta.GQAInferenceCS;
            int kernel = cs.FindKernel("ComputeAttentionScores");

            ComputeBuffer qBuf = new ComputeBuffer(Q.Count(), 4);
            qBuf.SetData(Q.ToArray());
            cs.SetBuffer(kernel, "Q", qBuf);

            ComputeBuffer kBuf = new ComputeBuffer(K.Count(), 4);
            kBuf.SetData(K.ToArray());
            cs.SetBuffer(kernel, "K", kBuf);

            int awCount = B * Hq * L_q * L_k;
            ComputeBuffer awBuf = new ComputeBuffer(awCount, 4);
            cs.SetBuffer(kernel, "AttentionWeights", awBuf);

            cs.SetInt("batch_size", B);
            cs.SetInt("sequence_length_q", L_q);
            cs.SetInt("sequence_length_k", L_k);
            cs.SetInt("num_heads_q", Hq);
            cs.SetInt("num_heads_kv", Hkv);
            cs.SetInt("head_dim", D);
            cs.SetFloat("scale", scale);

            int gx = (L_q + 3) / 4;
            int gy = (L_k + 31) / 32;
            int gz = (B * Hq + 3) / 4;
            cs.Dispatch(kernel, gx, gy, gz);

            Tensor scores = batched ?
                Tensor.Constant(awBuf, B, Hq, L_q, L_k) :
                Tensor.Constant(awBuf, Hq, L_q, L_k);

            qBuf.Release();
            kBuf.Release();
            awBuf.Release();

            return scores;
        }
    }
}