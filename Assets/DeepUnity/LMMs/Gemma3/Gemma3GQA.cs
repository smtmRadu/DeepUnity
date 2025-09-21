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

        public bool IsInitialized { get; private set; } = false;

        [SerializeField] public RotaryPositionalEmbeddings rope;
        public ComputeBuffer W_QKV;
        public ComputeBuffer W_O;
        [SerializeField] public Gemma3RMSNorm q_norm;
        [SerializeField] public Gemma3RMSNorm k_norm;
        [SerializeField] private Softmax softmax;

        private bool _buildKVCache = false;  // Backing field

        public bool BuildKVCache
        {
            set
            {
                if (value)
                {
                    CachedTokensNum = 0;
                    KCache = new List<Tensor>();
                    VCache = new List<Tensor>();
                    _buildKVCache = true;
                }
                else
                {
                    CachedTokensNum = 0;
                    KCache = null;
                    VCache = null;
                    _buildKVCache = false;
                }
            }
        }// when build kv cache is ON, Q and K (roped) will be cached and the model must receive one inpu t(B,1,E) at a time (only 1 elem)

        private int CachedTokensNum = 0;
        private List<Tensor> KCache { get; set; } = null; // Rope + Norm
        private List<Tensor> VCache { get; set; } = null;
        /// <summary>
        ///  <b>(B, L)</b> or <b>(L)</b>.<br></br>
        /// </summary>
        public Tensor AttentionMask { get; set; } = null; // This is the second input that must have entered in the forward function.




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
            float query_pre_attention_scalar = 256f,
            float? softcap = null,
            InitType weight_init = InitType.LeCun_Normal,
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

            this.is_causal = true;
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
            Array.Copy(results[0], 0, flat_qkv, 
                0,
                this.embedding_dim * this.inner_embedding_dim);
            Array.Copy(results[1], 0, flat_qkv,
                this.embedding_dim * this.inner_embedding_dim,
                this.embedding_dim * this.inner_embedding_dim * this.num_heads_kv / this.num_heads_q);
            Array.Copy(results[2], 0, flat_qkv,
                this.embedding_dim * this.inner_embedding_dim + this.embedding_dim * this.inner_embedding_dim * this.num_heads_kv / this.num_heads_q,
                this.embedding_dim * this.inner_embedding_dim * this.num_heads_kv / this.num_heads_q);
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
            if (_buildKVCache)
            {
                Q = rope == null ? Q : rope.ApplyRotaryEmbeddings(Q, input_pos: Enumerable.Range(CachedTokensNum, L_x).ToArray());
                K = rope == null ? K : rope.ApplyRotaryEmbeddings(K, input_pos: Enumerable.Range(CachedTokensNum, L_x).ToArray());

                KCache.Add(K);
                VCache.Add(V);
                CachedTokensNum += L_x;

                K = Tensor.Concat(-3, KCache.ToArray());// Update the concat function so it allows for tensor with different sizes on the axis that is merged on.... plaese..
                V = Tensor.Concat(-3, VCache.ToArray());
            }
            else
            {
                Q = rope == null ? Q : rope.ApplyRotaryEmbeddings(Q);
                K = rope == null ? K : rope.ApplyRotaryEmbeddings(K);
            }

            // ======================================================  GQA ===================================================================

            int L_k = K.Size(-3); // L_k is the actual length of the sequence. x might have a sequence length of only 1 when doing inference.

            Tensor mask = BuildMask(is_causal, AttentionMask, batched ? new int[] { B, num_heads_q, L_x, L_k } : new int[] { num_heads_q, L_x, L_k });
            
            Tensor scores = ComputeAttentionScoresGPU(Q, K, scale: scaling); // B, Hq,  L_x, L_k

            if(this.attn_logit_softcapping is not null)
            {
                scores = scores / this.attn_logit_softcapping.Value;
                scores = scores.Tanh(); // in gemma (line 256 - https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py), scores receive a softcap and are passed through tanh 
                scores = scores * this.attn_logit_softcapping.Value;
            }

            // Debug.Log("scores:" + scores);
            Tensor scores_masked = mask == null ? scores : scores + mask;// B, Hq, L_x, L_k
            Tensor attention_weights = softmax.Predict(scores_masked);  // B, Hq, L_x, L_k

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

                int gx = (D + 31) / 32;
                int gy = (L_q + 31) / 32;
                int gz = B * Hq;
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
        private static Tensor BuildMask(bool is_causal, Tensor attention_mask, params int[] mask_shape)
        {
            Tensor mask = Tensor.Zeros(mask_shape);

            bool is_batched = mask_shape.Length == 4;

            if (is_batched)
            {
                for (int b = 0; b < mask_shape[0]; b++)
                {
                    for (int l = 0; l < mask_shape[1]; l++)
                    {
                        for (int h = 0; h < mask_shape[2]; h++)
                        {
                            for (int e = 0; e < mask_shape[3]; e++)
                            {
                                if (is_causal && h < e)
                                    mask[b, l, h, e] = -1e10f;


                                if (attention_mask is not null)
                                    mask[b, l, h, e] += (1f - attention_mask[b, l]) * -1e25f;

                            }
                        }
                    }
                }
            }
            else
            {
                for (int l = 0; l < mask_shape[0]; l++)
                {
                    for (int h = 0; h < mask_shape[1]; h++)
                    {
                        for (int e = 0; e < mask_shape[2]; e++)
                        {
                            if (is_causal && h < e)
                                mask[l, h, e] = -1e10f;



                            if (attention_mask is not null)
                                mask[l, h, e] += (1f - attention_mask[l]) * -1e25f;
                        }
                    }
                }
            }

            return mask;
        }
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

            int gx = (L_q + 31) / 32;
            int gy = (L_k + 31) / 32;
            int gz = B * Hq;
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
