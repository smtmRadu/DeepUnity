using DeepUnity.Activations;
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
    public class GroupedQueryAttention : ILearnable, IModule, ICloneable
    {
        [SerializeField]
        public Device Device
        {
            get => W_O.Device;
            set { W_QKV.Device = value; W_O.Device = value; }
        }
        [SerializeField]
        public bool RequiresGrad
        {
            get => W_O.RequiresGrad;
            set { W_QKV.RequiresGrad = value; W_O.RequiresGrad = value; }
        }

        private int embedding_dim;
        private int inner_embedding_dim;
        private int num_heads_q;            
        private int num_heads_kv;                     
        private int head_dim;               
        private bool is_causal;
        private bool qk_norm;


        
        [SerializeField] public RotaryPositionalEmbeddings rope;
        [SerializeField] public Dense W_QKV;
        [SerializeField] public Dense W_O;
        [SerializeField] public RMSNorm q_rmsn;
        [SerializeField] public RMSNorm k_rmsn;
        [SerializeField] public Dropout drop;
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
        public GroupedQueryAttention(
            int embed_dim,
            int num_heads_q,
            int? num_heads_kv = null,
            float expansion_factor = 1f,
            bool is_causal = false,
            float dropout = 0.0f,
            bool qk_norm=false,
            float qk_norm_eps = 1e-6f,
            bool use_rope = false,
            int rope_max_seq_len = 4096,
            int rope_theta = 10_000,
            InitType weight_init = InitType.LeCun_Normal,
            Device device = Device.CPU)
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


            this.head_dim = this.inner_embedding_dim / num_heads_q;
            this.is_causal = is_causal;

            int proj_out = this.inner_embedding_dim + 2 * (this.inner_embedding_dim * this.num_heads_kv / num_heads_q);

            W_QKV = new Dense(this.embedding_dim, proj_out, bias: false, weight_init, device:device);
            W_O = new Dense(this.inner_embedding_dim, this.embedding_dim, bias: false, weight_init, device:device);
            if(qk_norm)
            {
                this.qk_norm = true;
                // line 182 https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
                this.q_rmsn = new RMSNorm(this.head_dim, eps:qk_norm_eps);
                this.k_rmsn = new RMSNorm(this.head_dim, eps:qk_norm_eps);
            }
            softmax = new Softmax();
            drop = new Dropout(dropout);
            rope = use_rope ? new RotaryPositionalEmbeddings(this.head_dim,
                                                                rope_max_seq_len,
                                                                rope_theta) : null;
        }
        private GroupedQueryAttention() { }

        public Tensor Predict(Tensor x)
        {
            if (x.Rank > 3 || x.Rank < 2)
                throw new ArgumentException($"GQA input must be of shape (B, L, E) or (L, E). Received tensor of rank {x.Rank} - shape {x.Shape.ToCommaSeparatedString()}");
            bool batched = x.Rank == 3;
            int B = batched ? x.Size(-3) : 1;
            int L_x = x.Size(-2);

            //Debug.Log("X (base):" + x);
            Tensor QKV = W_QKV.Predict(x);
            //Debug.Log("QKV (base):" + QKV);


            //Debug.Log(QKV);
            Tensor Q = batched ? Tensor.Zeros(B, L_x, this.num_heads_q, this.head_dim)  : Tensor.Zeros(L_x, this.num_heads_q, this.head_dim);
            Tensor K = batched ? Tensor.Zeros(B, L_x, this.num_heads_kv, this.head_dim) : Tensor.Zeros(L_x, this.num_heads_kv, this.head_dim);
            Tensor V = batched ? Tensor.Zeros(B, L_x, this.num_heads_kv, this.head_dim) : Tensor.Zeros(L_x, this.num_heads_kv, this.head_dim);

            // Unpack QKV
            if(L_x > 1)
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
            else if(L_x == 1)
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

            //Debug.Log("Q (base): " + Q);
            //Debug.Log("K (base): " + K);
            //Debug.Log("V (base): " + V);

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
                                Q[b, l, h, e] *= q_rmsn.gamma[e];  
                            }

                            for (int h = 0; h < this.num_heads_kv; h++)
                            {
                                K[b, l, h, e] *= k_rmsn.gamma[e];   
                            }
                        }
                    }
                }
            }

            //Debug.Log("Q norm (base): " + Q);
            //Debug.Log("K norm (base): " + K);


            // K is cached with rotation.

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
                

            int L_k = K.Size(-3); // L_k is the actual length of the sequence. x might have a sequence length of only 1 when doing inference.

            Tensor mask = BuildMask(is_causal,AttentionMask, batched ? new int[] { B, num_heads_q, L_x, L_k } : new int[] { num_heads_q, L_x, L_k });

            // Tensor mask = null;
            // if(is_causal && L_x > 1)
            // {
            //     mask = Tensor.Tril(Tensor.Fill(1, shape: batched ? new int[] { B, num_heads_q, L_x, L_k } : new int[] { num_heads_q, L_x, L_k }));
            // }
            // else // bidirectional attention / 1 token w/ kvcache
            //     mask = Tensor.Fill(value:1, shape: batched ? new int[] { B, num_heads_q, L_x, L_k } : new int[] { num_heads_q, L_x, L_k });
            // 

            // mask = AttentionMask == null ? mask : mask * BroadcastAttentionMask(AttentionMask, mask.Shape);

            // Debug.Log("Q: " + Q); // problem is at Q generation
            // Debug.Log("K: " + K);
            // Debug.Log("V: " + V);
            if (Device == Device.CPU)
            {
                // qkv generated correctly (+norm and rope) but scores are wrong idk why.
                // test please also the gpu generations after this.
                Tensor scores = ComputeAttentionScoresCPU(Q, K,scale: 1f / MathF.Sqrt(head_dim)); // B, Hq, L_x, L_k
                // Debug.Log("scores:" + scores);
                Tensor scores_masked = mask == null? scores : scores + mask;// B, Hq, L_x, L_k
                //Debug.Log("scores_masked:" + scores_masked);
                Tensor attention_weights = softmax.Predict(scores_masked);  // B, Hq, L_x, L_k
                // on predict no dropout.
                //Debug.Log("Att_weights:" + attention_weights);
                Tensor attended_values = AttendValuesCPU(attention_weights, V); // B, L, E_inner

                return W_O.Predict(attended_values);
            }
            else
            {
                Tensor scores = ComputeAttentionScoresGPU(Q, K, scale: 1f/MathF.Sqrt(head_dim)); // B, Hq,  L_x, L_k
                // Debug.Log("scores:" + scores);
                Tensor scores_masked = mask == null ? scores : scores + mask;// B, Hq, L_x, L_k
                Tensor attention_weights = softmax.Predict(scores_masked);  // B, Hq, L_x, L_k

                //Debug.Log("attention_weights:" + attention_weights);
                // on predict no dropout.
                //Debug.Log("Att_weights:" + attention_weights);
                Tensor attended_values = AttendValuesGPU(attention_weights, V); // B, L, E_inner

                //Debug.Log("attented_values:" + attended_values);
                return W_O.Predict(attended_values);
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
                                    mask[b, l, h, e] += (1f-attention_mask[b, l]) * -1e10f;
                                    
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
                        for (int e = 0; e < mask_shape[2]; e++ )
                        {
                            if(is_causal && h < e)
                                mask[l, h, e] = -1e10f;

                            

                            if (attention_mask is not null)
                                mask[l, h, e] += (1f - attention_mask[l]) * -1e10f;
                        }
                    }
                }
            }

            return mask;
        }
        /// <summary>
        /// Q: (B, L, num_heads_q, head_dim)
        /// K: (B, L, num_heads_k, head_dim)
        /// Returns: (B, num_heads_q, L, L)
        /// </summary>
        /// <param name="Q"></param>
        /// <param name="K"></param>
        /// <returns></returns>
        public static Tensor ComputeAttentionScoresCPU(Tensor Q, Tensor K, float scale)
        {
            bool batched = Q.Rank == 4;
            int B = batched ? Q.Size(0) : 1;
            int L_q = Q.Size(-3);
            int L_k = K.Size(-3);
            int num_heads_q = Q.Size(-2);
            int head_dim = Q.Size(-1);
            int num_heads_k = K.Size(-2);

            // (B, Hq, L_q, L_k)
            Tensor scores = batched ? Tensor.Zeros(B, num_heads_q, L_q, L_k) : Tensor.Zeros(num_heads_q, L_q, L_k);

            Parallel.For(0, num_heads_q, h =>
            {
                for (int b = 0; b < B; b++)
                {
                    int group_size = num_heads_q / num_heads_k;
                    int hk = h / group_size;
                    for (int i = 0; i < L_q; i++)
                    {
                        for (int j = 0; j < L_k; j++)
                        {
                            float dot = 0f;
                            for (int d = 0; d < head_dim; d++)
                                dot += Q[b, i, h, d] * K[b, j, hk, d];

                            scores[b, h, i, j] = dot * scale;
                        }
                    }
                }
            });

            return scores;
        }
        /// <summary>
        /// Attention weights: (B, Hq, L, L)
        /// V: (B, L, num_heads_v, head_dim)
        /// Returns (B, L, embedding dim)
        /// </summary>
        /// <param name="attention_weights"></param>
        /// <param name="V"></param>
        /// <returns></returns>
        public static Tensor AttendValuesCPU(Tensor attention_weights, Tensor V)
        {
            bool is_batched = attention_weights.Rank == 4;           // (B,Hq,L_q,L_v) || (Hq,L_q,L_v)
            int B = is_batched ? attention_weights.Size(0) : 1;
            int Hq = attention_weights.Size(is_batched ? 1 : 0);
            int L_q = attention_weights.Size(-2);
            int L_v = attention_weights.Size(-1);
            int num_heads_v = V.Size(-2);
            int head_dim = V.Size(-1);

            int inner_embedding_dim = Hq * head_dim;

            Tensor y = is_batched ? Tensor.Zeros(B, L_q, inner_embedding_dim) : Tensor.Zeros(L_q, inner_embedding_dim);

            Parallel.For(0, Hq, h =>
            {
                int group_size = Hq / num_heads_v;
                int hv = h / group_size;

                for (int b = 0; b < B; b++)
                {
                    for (int i = 0; i < L_q; i++) 
                    {
                        for (int d = 0; d < head_dim; d++)  
                        {
                            float weighted_sum = 0f;

                            for (int j = 0; j < L_v; j++)
                            {
                                float weight = attention_weights[b, h, i, j];
                                float value = V[b, j, hv, d];
                                weighted_sum += weight * value;
                            }

                            int output_dim = h * head_dim + d;
                            y[b, i, output_dim] = weighted_sum;

                        }
                    }
                }
            });

            return y;
        }


        public static Tensor ComputeAttentionScoresGPU(Tensor Q, Tensor K, float scale)
        {
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


        public static Tensor AttendValuesGPU(Tensor attention_weights, Tensor V)
        {
            bool batched = attention_weights.Rank == 4;
            int B = batched ? attention_weights.Size(0) : 1;
            int Hq = attention_weights.Size(batched ? 1 : 0);
            int L_q = attention_weights.Size(-2);
            int L_v = attention_weights.Size(-1);
            int D = V.Size(-1);
            int Hkv = V.Size(-2);
            int inner_embed_dim = Hq * D;

            ComputeShader cs = DeepUnityMeta.GQAInferenceCS;
            int kernel = cs.FindKernel("AttendValues");

            ComputeBuffer awBuf = new ComputeBuffer(attention_weights.Count(), 4);
            awBuf.SetData(attention_weights.ToArray());
            cs.SetBuffer(kernel, "AttentionWeights", awBuf);

            ComputeBuffer vBuf = new ComputeBuffer(V.Count(), 4);
            vBuf.SetData(V.ToArray());
            cs.SetBuffer(kernel, "V", vBuf);

            int yCount = B * L_v * inner_embed_dim;
            ComputeBuffer yBuf = new ComputeBuffer(yCount, 4);
            cs.SetBuffer(kernel, "AttendedValues", yBuf);

            cs.SetInt("batch_size", B);
            cs.SetInt("sequence_length_v", L_v);
            cs.SetInt("num_heads_q", Hq);
            cs.SetInt("num_heads_kv", Hkv);
            cs.SetInt("head_dim", D);

            int gx = (D + 31) / 32;
            int gy = (L_q + 31) / 32;
            int gz = B * Hq;                         
            cs.Dispatch(kernel, gx, gy, gz);

            Tensor Y = batched ?
                Tensor.Constant(yBuf, B, L_q, inner_embed_dim) :
                Tensor.Constant(yBuf, L_q, inner_embed_dim);

            awBuf.Release();
            vBuf.Release();
            yBuf.Release();

            return Y;
        }

        public Tensor Forward(Tensor x)
        {
            throw new Exception("This module was implemented only for inference.");
        }

        public Tensor Backward(Tensor dLdY)
        {
            throw new Exception("This module was implemented only for inference.");
        }

        public Parameter[] Parameters() =>
            W_QKV.Parameters().Concat(W_O.Parameters()).ToArray();

        public void OnBeforeSerialize() { }
        public void OnAfterDeserialize() { W_QKV.OnAfterDeserialize(); W_O.OnAfterDeserialize(); }

        public object Clone()
        {
            var matt = new GroupedQueryAttention();
            matt.num_heads_q = this.num_heads_q;
            matt.num_heads_kv = this.num_heads_kv;
            matt.is_causal = this.is_causal;
            matt.embedding_dim = this.embedding_dim;
            matt.inner_embedding_dim = this.inner_embedding_dim;
            matt.head_dim = this.head_dim;
            matt.rope = this.rope;
            matt.Device = Device;
            matt.RequiresGrad = RequiresGrad;
            matt.softmax = softmax.Clone() as Softmax;
            matt.Device = Device;
            matt.W_O = W_O.Clone() as Dense;
            matt.W_QKV = W_QKV.Clone() as Dense;
            return matt;
        }
    }
}
