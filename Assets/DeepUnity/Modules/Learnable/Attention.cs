using System;
using Unity.VisualScripting;
using UnityEngine;
using System.Collections.Generic;
using System.Linq;

namespace DeepUnity
{
    /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention
    /// https://arxiv.org/pdf/1706.03762.pdf
    /// https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    /// https://www.youtube.com/watch?v=aw3H-wPuRcw
    /// https://machinelearningmastery.com/the-attention-mechanism-from-scratch/
    /// A. Liang Transformer March 21, 2023 Transformer Attention Derivative - https://say-hello2y.github.io/2022-09-07/attention-gradient

    /// <summary>
    /// <b>Applies a Scaled Dot-Product Attention over the input.</b> <br></br>
    /// Input: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
    /// Output: <b>(B, L, D)</b> or <b>(L, D)</b> for unbatched input. <br></br>
    /// where B = batch_size, L = sequence_length, H = num_features and D = embed_dim.<br></br>
    /// <b>Placed after the non-linear activation function.</b> <br></br>
    /// </summary>
    [Serializable]
    public class Attention : ILearnable, IModule
    {
        [SerializeField] private Tensor W_Q;
        [SerializeField] private Tensor W_K;
        [SerializeField] private Tensor W_V;

        [NonSerialized] private Tensor W_Q_grad;
        [NonSerialized] private Tensor W_K_grad;
        [NonSerialized] private Tensor W_V_grad;

        [SerializeField] private int d;

        private Stack<Softmax> SoftmaxCache { get; set; }
        private Tensor InputCache { get; set; }

        /// <summary>
        /// <b>Applies a Scaled Dot-Product Attention over the input.</b> <br></br>
        /// Input: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
        /// Output: <b>(B, L, D)</b> or <b>(L, D)</b> for unbatched input. <br></br>
        /// where B = batch_size, L = sequence_length, H = num_features and D = embed_dim.<br></br>
        /// <b>Placed after the non-linear activation function.</b> <br></br>
        /// </summary>
        /// <param name="input_shape">(T, H)</param>
        /// <param name="embed_dim">Total dimension of the model.</param>
        /// <param name="scale">Apply scale in self dot-product attention.</param>
        public Attention(int in_features, int embed_dim)
        {
            this.d = embed_dim;
            
            // For the sake of simplicity d_q, d_k and d_v are the same. 
            // This will produce the same output shape as the input.
            int H = in_features;
            float range = MathF.Sqrt(1f / embed_dim);
            W_Q = Tensor.RandomRange((-range, range), H, d);
            W_K = Tensor.RandomRange((-range, range), H, d);
            W_V = Tensor.RandomRange((-range, range), H, d);

            SoftmaxCache = new();
        }
        private Attention() { }

        public Tensor Predict(Tensor input)
        {
            // softmax(Q * KT / Sqrt(D)) * V [Scaled Dot-Product Attention, page 4]
            // input shape (B, L, H) or (L, H)
            if (input.Size(-1) != W_Q.Size(0))
            {
                throw new ShapeException($"Input features ({input.Size(-1)}) does not match with the Dense Layer features_num ({W_Q.Size(0)}).");

            }
            if (input.Rank != 3 && input.Rank != 2)
            {
                throw new ShapeException($"Input must be of shape (B, L, H) or (L, H) for unabatches input, and input received has shape ({input.Shape.ToCommaSeparatedString()}).");
            }

            bool isBatched = input.Rank == 3;
            int batch_size = isBatched ? input.Size(0) : 1;

            Tensor[] batch_elem = isBatched ? input.Split(0, 1) : new Tensor[] {input};
            Tensor[] SDPA = new Tensor[batch_size];

            for (int i = 0; i < batch_size; i++)
            {
                Tensor x = isBatched ? batch_elem[i].Squeeze(1) : batch_elem[i];
                Tensor Q = Tensor.MatMul(x, W_Q); // (L, H) * (H, D) = (L, D)
                Tensor K = Tensor.MatMul(x, W_K); // (L, D)
                Tensor V = Tensor.MatMul(x, W_V); // (L, D)
                Tensor sdpa = Tensor.MatMul(Q, K.Transpose(0, 1)); //(L, D) * (D, L) = (L, L)
                sdpa /= MathF.Sqrt(d);
                sdpa = new Softmax().Forward(sdpa);
                SDPA[i] = Tensor.MatMul(sdpa, V); // (L, D)
            }
        
            return Tensor.Concat(null, SDPA); // (B, L, D)
        }

        public Tensor Forward(Tensor input)
        {           
            if (input.Size(-1) != W_Q.Size(0))
            {
                throw new ShapeException($"Input features ({input.Size(-1)}) does not match with the Dense Layer features_num ({W_Q.Size(0)}).");
            }
            if (input.Rank != 3 && input.Rank != 2)
            {
                throw new ShapeException($"Input must be of shape (B, L, H) or (L, H) for unabatches input, and input received has shape ({input.Shape.ToCommaSeparatedString()}).");
            }

            InputCache = Tensor.Identity(input);

            bool isBatched = input.Rank == 3;
            int batch_size = isBatched ? input.Size(0) : 1;

            Tensor[] batch_elem = isBatched ? input.Split(0, 1) : new Tensor[] { input };
            Tensor[] SDPA = new Tensor[batch_size];

            for (int i = 0; i < batch_size; i++)
            {
                Tensor x = isBatched ? batch_elem[i].Squeeze(1) : batch_elem[i];
                Tensor Q = Tensor.MatMul(x, W_Q); // (L, H) * (H, D) = (L, D)
                Tensor K = Tensor.MatMul(x, W_K); // (L, D)
                Tensor V = Tensor.MatMul(x, W_V); // (L, D)
                Tensor sdpa = Tensor.MatMul(Q, K.Transpose(0, 1)); //(L, D) * (D, L) = (L, L)
                sdpa /= MathF.Sqrt(d);
                SoftmaxCache.Push(new Softmax());
                sdpa = SoftmaxCache.Peek().Forward(sdpa);
                SDPA[i] = Tensor.MatMul(sdpa, V); // (L, D)
            }

            return Tensor.Concat(null, SDPA);
        }
        public Tensor Backward(Tensor dfx_dx)
        {
            throw new NotImplementedException();
        }

        public object Clone()
        {
            var att = new Attention();
            att.d = this.d; 
            att.W_Q = (Tensor)this.W_Q.Clone();
            att.W_K = (Tensor)this.W_K.Clone();
            att.W_V = (Tensor)this.W_V.Clone();
            att.W_Q_grad = (Tensor)this.W_Q_grad.Clone();
            att.W_K_grad = (Tensor)this.W_K_grad.Clone();
            att.W_V_grad = (Tensor)this.W_V_grad.Clone();
            att.SoftmaxCache = new Stack<Softmax>(this.SoftmaxCache.Select(x => x.Clone() as Softmax));
            return att;
        }



        public void SetDevice(Device device)
        {
            return;
        }
        public int ParametersCount()
        {
            return W_Q.Count() + W_K.Count() + W_V.Count();
        }
        public Parameter[] Parameters()
        {
            if (W_Q_grad == null)
                OnAfterDeserialize();

            var q = new Parameter(W_Q, W_Q_grad);
            var k = new Parameter(W_K, W_K_grad);
            var v = new Parameter(W_V, W_V_grad);

            return new Parameter[] { q , k , v };
        }                          
        public virtual void OnBeforeSerialize()
        {

        }
        public virtual void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (W_Q.Shape == null)
                return;

            if (W_Q.Shape.Length == 0)
                return;

            // do not check if gamma is != null...
            this.W_Q_grad = Tensor.Zeros(W_Q.Shape);
            this.W_K_grad = Tensor.Zeros(W_K.Shape);
            this.W_V_grad = Tensor.Zeros(W_V.Shape);
        }
    }

}

