using DeepUnity.Activations;
using System;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// NOTE IT REQUIRES A FLASH IMPLEMENTATION ON BOTH CPU AND GPU BECAUSE IS SLOW AF
    
    /// <summary>
    /// <b>Applied a Self Scaled Dot-Product Attention with multiple heads.</b> <br></br>
    /// Input: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
    /// Output: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
    /// where B = batch_size, L = sequence_length and H = num_features<br></br>
    /// <b>Placed after the non-linear activation function.</b> <br></br>
    /// </summary>
    /// <param name="embed_dim"></param>
    /// <param name="heads_num"></param>
    [Serializable]
    public class MultiheadAttention : ILearnable, IModule
    {
        [SerializeField] public Device Device
        { 
            get => W_O.Device; 
            set
                {
                W_QKV.Device = value;
                W_O.Device = value;
                } 
        }
        [SerializeField]
        public bool RequiresGrad
        {
            get => W_O.RequiresGrad;
            set
            {
                W_QKV.RequiresGrad = value;
                W_O.RequiresGrad = value;
            }
        }
        private int HeadDim => embedding_dim / num_heads;
        
        private Tensor CausalMask => Tensor.Triu(Tensor.Ones()) + Tensor.Tril(Tensor.Fill(float.MinValue));

        [SerializeField] private int embedding_dim;
        [SerializeField] private int num_heads;
        [SerializeField] private int head_dim;
        [SerializeField] private bool is_causal;
       
        [SerializeField] private Dense W_QKV;
        [SerializeField] private Dense W_O;
        [SerializeField] private Dropout drop;
        [SerializeField] private Softmax softmax;

        private Tensor Q { get; set; }
        private Tensor K { get; set; }
        private Tensor V { get; set; }
        private Tensor qkt { get; set; }
        private Tensor qkt_srdk { get; set; }
        private Tensor qkt_srdk_causal { get; set; }
        private Tensor qkt_srdk_causal_sm { get; set; }
        private Tensor qkt_srdk_causal_sm_drop { get; set; }
        private Tensor qkt_srdk_causal_sm_drop_v { get; set; }

        /// <summary>
        /// <b>Applied a Self Scaled Dot-Product Attention on multiple heads.</b> <br></br>
        /// Input: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
        /// Output: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
        /// where B = batch_size, L = sequence_length and H = num_features<br></br>
        /// <b>Placed after the non-linear activation function.</b> <br></br>
        /// </summary>
        /// <param name="embed_dim">Must be divisible by heads_num.</param>
        /// <param name="heads_num">Must divide embed_dim exactly.</param>
        /// <param name="is_causal">Causal attention.</param>
        /// <exception cref="ArgumentException">Embedding dimension must be divisible by heads number</exception>
        public MultiheadAttention(int embed_dim, int heads_num, float dropout=0.0F, bool is_causal = false, InitType weight_init = InitType.LeCun_Normal, Device device = Device.CPU)
        {
            if (embed_dim % heads_num != 0)
            {
                throw new ArgumentException("Embedding dimension must be divisible by heads number");
            }
            this.num_heads = heads_num;
            this.embedding_dim = embed_dim;
            this.head_dim = embedding_dim / heads_num;
            this.is_causal = false;
            W_QKV = new Dense(embed_dim, embed_dim * 3, bias: false, weight_init, device:device);
            W_O = new Dense(embed_dim, embed_dim, bias: false, weight_init, device: device);
            softmax = new Softmax();
            drop = new Dropout(dropout);
        }
        private MultiheadAttention() { }

        public Tensor Predict(Tensor input)
        {
            if (input.Size(-1) != this.embedding_dim)
                throw new ShapeException($"Input dim {input.Size(-1)} must be {this.embedding_dim}");

            bool is_batched = input.Rank == 3;
            int L = input.Size(-2);
            int B = is_batched ? input.Size(-3) : 1;

            var qkv = this.W_QKV.Predict(input).Split(-1, embedding_dim);
            Q = qkv[0].Reshape(B * L, this.num_heads, embedding_dim);
            K = qkv[1].Reshape(B * L, this.num_heads, embedding_dim);
            V = qkv[2].Reshape(B * L, this.num_heads, embedding_dim);

            // SDPA
            var scores = Tensor.BatchedMatMul(Q, K.Transpose(-1, -2), device:Device) / MathF.Sqrt(embedding_dim);
            
            if (is_causal)
                scores = scores * CausalMask;

            var att = this.softmax.Predict(scores);

            att = this.drop.Predict(att); // apply dropout

            att = att.Reshape(B, L, embedding_dim);
            att = Tensor.BatchedMatMul(att, V, Device);

            var result = this.W_O.Predict(att);

            if (!is_batched)
                result.Squeeze(0);

            return result;
        }

        public Tensor Forward(Tensor input)
        {
            if (input.Size(-1) != this.embedding_dim)
                throw new ShapeException($"Input dim {input.Size(-1)} must be {this.embedding_dim}");

            bool is_batched = input.Rank == 3;
            int L = input.Size(-2);
            int B = is_batched ? input.Size(-3) : 1;

            var qkv = this.W_QKV.Forward(input).Chunk(-1, 3); // (B, L, E) * 3
            Q = qkv[0].Reshape(B, L, this.num_heads, this.head_dim).Permute(0, 2, 1, 3).Reshape(B * this.num_heads, L , this.head_dim); // (B*H, L, H_d)
            K = qkv[1].Reshape(B, L, this.num_heads, this.head_dim).Permute(0, 2, 1, 3).Reshape(B * this.num_heads, L , this.head_dim);
            V = qkv[2].Reshape(B, L, this.num_heads, this.head_dim).Permute(0, 2, 1, 3).Reshape(B * this.num_heads, L , this.head_dim);

            // SDPA
            qkt = Tensor.BatchedMatMul(Q, K.Transpose(-1, -2), device: Device); // (BL, L, L)
            qkt_srdk = qkt / MathF.Sqrt(head_dim);
            
            qkt_srdk_causal = is_causal ? qkt_srdk * CausalMask : qkt_srdk;
            
            qkt_srdk_causal_sm = this.softmax.Forward(qkt_srdk_causal); // (B*H, L, L)
            qkt_srdk_causal_sm_drop = this.drop.Forward(qkt_srdk_causal_sm);
            qkt_srdk_causal_sm_drop_v = Tensor.BatchedMatMul(qkt_srdk_causal_sm, V, Device); // (B*H, L, L) @ (B*H, L, H_d) = (B*H, L, H_d)
            qkt_srdk_causal_sm_drop_v = qkt_srdk_causal_sm_drop_v.Reshape(B, this.num_heads, L, this.head_dim).Permute(0, 2, 1, 3).Reshape(B, L, this.embedding_dim);
            var qkt_srdk_causal_sm_drop_v_wo = this.W_O.Forward(qkt_srdk_causal_sm_drop_v);
            

            if (!is_batched)
                qkt_srdk_causal_sm_drop_v_wo.Squeeze(0);

            return qkt_srdk_causal_sm_drop_v_wo;
        }

        public Tensor Backward(Tensor dLdY)
        {
            bool is_batched = dLdY.Rank == 3;
            int L = dLdY.Size(-2);
            int B = is_batched ? dLdY.Size(-3) : 1;

            Tensor dLdqkt_srdk_causal_sm_drop_v = this.W_O.Backward(dLdY);
            Tensor dLdqkt_srdk_causal_sm_drop_v_RESHAPED = dLdqkt_srdk_causal_sm_drop_v.
                                        Reshape(B, L, this.num_heads, this.head_dim).
                                        Permute(0, 2, 1, 3).
                                        Reshape(B * this.num_heads, L, this.head_dim);

            Tensor dLdqkt_srdk_causal_sm_drop = 
                Tensor.BatchedMatMul(dLdqkt_srdk_causal_sm_drop_v_RESHAPED, V.Transpose(-1, -2), Device);
            Tensor dLdqkt_srdk_causal_sm = this.drop.Backward(dLdqkt_srdk_causal_sm_drop);
            Tensor dLdqkt_srdk_causal = this.softmax.Backward(dLdqkt_srdk_causal_sm);
            Tensor dLdqkt_srdk = this.is_causal ? dLdqkt_srdk_causal * CausalMask : dLdqkt_srdk_causal;
            Tensor dLdqkt = dLdqkt_srdk / MathF.Sqrt(this.head_dim);
            
            Tensor dLdV = Tensor.BatchedMatMul(qkt_srdk_causal_sm.Transpose(-1, -2), dLdqkt_srdk_causal_sm_drop_v_RESHAPED, Device);
            Tensor dLdQ = Tensor.BatchedMatMul(dLdqkt, K, Device); 
            Tensor dLdK = Tensor.BatchedMatMul(dLdqkt.Transpose(-1, -2), Q, Device);
            Debug.Log(dLdK.Shape.ToCommaSeparatedString());

            dLdV = dLdV.Reshape(B, this.num_heads, L, this.head_dim).Permute(0, 2, 1, 3).Reshape(B, L, this.embedding_dim);
            dLdQ = dLdQ.Reshape(B, this.num_heads, L, this.head_dim).Permute(0, 2, 1, 3).Reshape(B, L, this.embedding_dim);
            dLdK = dLdK.Reshape(B, this.num_heads, L, this.head_dim).Permute(0, 2, 1, 3).Reshape(B, L, this.embedding_dim);
            Tensor dLdQKV = Tensor.Concat(-1, dLdQ, dLdK, dLdV);

            return this.W_QKV.Backward(dLdQKV);
        }


        public Parameter[] Parameters()
        {
            var attpar = W_QKV.Parameters();
            var woparr = W_O.Parameters();
            return woparr.Concat(attpar).ToArray();
        }


        public virtual void OnBeforeSerialize()
        {
        }

        public virtual void OnAfterDeserialize()
        {
            W_O.OnAfterDeserialize();

            W_QKV.OnAfterDeserialize();
           
        }

        public object Clone()
        {
            var matt = new MultiheadAttention();
            matt.num_heads = this.num_heads;
            matt.is_causal = this.is_causal;
            matt.embedding_dim = this.embedding_dim;
            matt.head_dim = this.head_dim;
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


