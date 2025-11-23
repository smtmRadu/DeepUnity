using DeepUnity.Activations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.MobileLLMModeling
{
    /// <summary>
    /// Input: <b>(B, L, E)</b> or <b>(L, E)</b>.<br></br>
    /// Output: <b>(B, L, E)</b> or <b>(L, E)</b>.<br></br>
    /// where B = batch_size, L = sequence_length, E = embed_dim
    /// </summary>
    [Serializable]
    public class LlamaGQA
    {
        private float scaling;
        private int qkv_proj_dim;
        private int embedding_dim;
        private int inner_embedding_dim;
        private int num_heads_q;
        private int num_heads_kv;
        private int head_dim;
        private bool is_causal;

        public bool IsInitialized { get; private set; } = false;

        [SerializeField] public RotaryPositionalEmbeddings rope;
        public ComputeBuffer W_QKV;
        public ComputeBuffer W_O;
        public ComputeBuffer input_buffer;
        public ComputeBuffer qkv_proj_buffer;
        public ComputeBuffer q_buffer;
        public ComputeBuffer attended_values_buffer;
        public ComputeBuffer output_buffer;
        
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
        public LlamaGQA(
            int embed_dim,
            int num_heads_q,
            int? num_heads_kv = null,
            float expansion_factor = 1f,
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

            this.scaling = MathF.Pow(this.inner_embedding_dim, -0.5f);
            this.softmax = new Softmax();
            this.rope = rope;

            if (!string.IsNullOrEmpty(layer_params_path))
            {
                _ = LoadWeightsAsync(layer_params_path);
            }
        }
        private LlamaGQA() { }
        ~LlamaGQA()
        {
            W_O.Release();
            W_QKV.Release();
            input_buffer.Release();
            qkv_proj_buffer.Release();
            output_buffer.Release();
            attended_values_buffer.Release();
            q_buffer.Release();
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

        public void PrepareInputBuffer(int x_count)
        {
            if (input_buffer == null || input_buffer.count != x_count)
            {
                input_buffer?.Release();
                input_buffer = new ComputeBuffer(x_count, 4, ComputeBufferType.Structured);
            }
        }
        public void PrepareQKVProjBuffer(int B, int L)
        {
            if (qkv_proj_buffer == null || qkv_proj_buffer.count != B * L * qkv_proj_dim)
            {
                qkv_proj_buffer?.Release();
                qkv_proj_buffer = new ComputeBuffer(B * L * qkv_proj_dim, 4, ComputeBufferType.Structured);
            }

        }

        public void PrepareQBuffer(int q_count)
        {
            if (q_buffer == null || q_buffer.count != q_count)
            {
                q_buffer?.Release();
                q_buffer = new ComputeBuffer(q_count, 4, ComputeBufferType.Structured);
            }
        }
        public void PrepareAttendedValuesBuffer(int B, int L)
        {
            if (attended_values_buffer == null || attended_values_buffer.count != B * L * this.inner_embedding_dim)
            {
                attended_values_buffer?.Release();
                attended_values_buffer = new ComputeBuffer(B * L * this.inner_embedding_dim, 4, ComputeBufferType.Structured);
            }
        }
        public void PrepareOutputBuffer(int B, int L)
        {
            if (output_buffer == null || output_buffer.count != B * L * this.embedding_dim)
            {
                output_buffer?.Release();
                output_buffer = new ComputeBuffer(B * L * this.embedding_dim, 4, ComputeBufferType.Structured);
            }
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
            int qkvKernel = cs.FindKernel(B == 1 && L_x == 1 ? "QKVProj1Vec" : "QKVProj");
            cs.SetBuffer(qkvKernel, "W_QKV", W_QKV);

            PrepareInputBuffer(x.Count());
            input_buffer.SetData(x.ToArray());
            // Debug.Log("X (gemma):" + x);
            cs.SetBuffer(qkvKernel, "X", input_buffer);

            PrepareQKVProjBuffer(B: B, L: L_x);
            cs.SetBuffer(qkvKernel, "QKV", qkv_proj_buffer);

            cs.SetInt("batch_size", B);
            cs.SetInt("sequence_length_q", L_x);
            cs.SetInt("embedding_dim", embedding_dim);
            cs.SetInt("qkv_proj_dim", qkv_proj_dim);

            if (B == 1 && L_x == 1)
                cs.Dispatch(qkvKernel,
                            (qkv_proj_dim + 255) / 256,
                            L_x,
                            B);

            else
                cs.Dispatch(qkvKernel,
                    B,
                    (L_x + 15) / 16,
                    (qkv_proj_dim + 31) / 32);

            Tensor QKV = batched ? Tensor.Constant(qkv_proj_buffer, B, L_x, qkv_proj_dim) : Tensor.Constant(qkv_proj_buffer, L_x, qkv_proj_dim);

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

            Debug.Log("Q (llama): " + Q);
            Debug.Log("K (llama): " + K);
            Debug.Log("V (llama): " + V);


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

            //Debug.Log("Scores (gemma): " + scores);


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

                PrepareAttendedValuesBuffer(B: B, L: L_q);
                cs.SetBuffer(attendValueskernel, "AttendedValues", attended_values_buffer);

                cs.SetInt("batch_size", B);
                cs.SetInt("sequence_length_v", L_v);
                cs.SetInt("sequence_length_q", L_q);
                cs.SetInt("num_heads_q", Hq);
                cs.SetInt("num_heads_kv", Hkv);
                cs.SetInt("head_dim", D);

                int gx = (D + 127) / 128;
                int gy = L_q;
                int gz = (B * Hq + 3) / 4;
                cs.Dispatch(attendValueskernel, gx, gy, gz);
                awBuf.Release();
                vBuf.Release();


                // ======================================================================= OUTPUT projection ===============================================

                int oProjKernel = cs.FindKernel("OProj");

                cs.SetBuffer(oProjKernel, "AttendedValues", attended_values_buffer);
                cs.SetBuffer(oProjKernel, "W_O", W_O);
                cs.SetInt("inner_embedding_dim", inner_embedding_dim);

                //ComputeBuffer oBuff = new ComputeBuffer(B * L_q * embedding_dim, 4, ComputeBufferType.Structured);
                PrepareOutputBuffer(B: B, L: L_q);
                cs.SetBuffer(oProjKernel, "O", output_buffer);
                cs.Dispatch(oProjKernel, B, (L_q + 3) / 4, (embedding_dim + 31) / 32);


                Tensor output = batched ? Tensor.Constant(output_buffer, B, L_q, embedding_dim) : Tensor.Constant(output_buffer, L_q, embedding_dim);
                return output;
            }

        }

        private static Tensor BuildMask(bool is_causal, Tensor attention_mask, params int[] mask_shape)
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
        public Tensor ComputeAttentionScoresGPU(Tensor Q, Tensor K, float scale)
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

            PrepareQBuffer(Q.Count());
            q_buffer.SetData(Q.ToArray());
            cs.SetBuffer(kernel, "Q", q_buffer);

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

            int gx = L_q;
            int gy = (L_k + 31) / 32;
            int gz = (B * Hq + 3) / 4;
            cs.Dispatch(kernel, gx, gy, gz);

            Tensor scores = batched ?
                Tensor.Constant(awBuf, B, Hq, L_q, L_k) :
                Tensor.Constant(awBuf, Hq, L_q, L_k);

            kBuf.Release();
            awBuf.Release();

            return scores;
        }
    }
}
