using System;
using Unity.VisualScripting;
using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using DeepUnity.Activations;

namespace DeepUnity.Modules
{
    /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention
    /// https://arxiv.org/pdf/1706.03762.pdf
    /// https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    /// https://www.youtube.com/watch?v=aw3H-wPuRcw
    /// https://machinelearningmastery.com/the-attention-mechanism-from-scratch/
    /// A. Liang Transformer March 21, 2023 Transformer Attention Derivative - https://say-hello2y.github.io/2022-09-07/attention-gradient
    /// 
    /// Previous are for multihead, better the simple one here:)
    /// https://iq.opengenus.org/scaled-dot-product-attention/

    /// <summary>
    /// <b>Applies a Scaled Dot-Product Attention between the elements of the input.</b> <br></br>
    /// Input: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
    /// Output: <b>(B, L, D)</b> or <b>(L, D)</b> for unbatched input. <br></br>
    /// where B = batch_size, L = sequence_length, H = num_features and D = embed_dim.<br></br>
    /// <b>Placed after the non-linear activation function.</b> <br></br>
    /// </summary>
    [Serializable]
    public class Attention : ILearnable, IModule
    {
        [SerializeField] public Device Device { get; set; } = Device.CPU;

        [SerializeField] private int d;
        [SerializeField] private bool mask;

        [SerializeField] private Tensor W_Q;
        [SerializeField] private Tensor W_K;
        [SerializeField] private Tensor W_V;

        [NonSerialized] private Tensor W_Q_grad;
        [NonSerialized] private Tensor W_K_grad;
        [NonSerialized] private Tensor W_V_grad;

        private Stack<Tensor> Q { get; set; }
        private Stack<Tensor> K { get; set; }
        private Stack<Tensor> V { get; set; }
        private Stack<Softmax> SoftmaxCache { get; set; }
        private Stack<Tensor> PostSoftmaxCache { get; set; }
        private Stack<Tensor> InputCache { get; set; }

        /// <summary>
        /// <b>Applies a Self Scaled Dot-Product Attention over the input.</b> <br></br>
        /// Input: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
        /// Output: <b>(B, L, D)</b> or <b>(L, D)</b> for unbatched input. <br></br>
        /// where B = batch_size, L = sequence_length, H = input_size and D = embed_dim.<br></br>
        /// <b>Placed after the non-linear activation function.</b> <br></br>
        /// </summary>
        /// <param name="input_size">Input features size.</param>
        /// <param name="mask">Use attention mask for future tokens.</param>
        /// <param name="embed_dim">Dimension of the attention mechanism.</param>
        public Attention(int input_size, int embed_dim, bool mask = false, InitType weight_init = InitType.LeCun_Uniform, Device device = Device.CPU)
        {
            // H and d have the same dimension
            d = embed_dim;
            this.mask = mask;
            this.Device = device;
            int H = input_size;

            int fanIn = H;
            int fanOut = embed_dim;

            W_Q = Parameter.Create(new int[] { H, d }, fanIn, fanOut, weight_init);
            W_K = Parameter.Create(new int[]{H, d}, fanIn, fanOut, weight_init);
            W_V = Parameter.Create(new int[] { H, d }, fanIn, fanOut, weight_init);
            W_Q_grad = Tensor.Zeros(W_Q.Shape);
            W_K_grad = Tensor.Zeros(W_K.Shape);
            W_V_grad = Tensor.Zeros(W_V.Shape);

            Q = new();
            K = new();
            V = new();
            SoftmaxCache = new();
            PostSoftmaxCache = new();
            InputCache = new();
        }
        private Attention() { }

        public Tensor Predict(Tensor input)
        {
            if (input.Size(-1) != W_Q.Size(0))
            {
                throw new ShapeException($"Input features ({input.Size(-1)}) does not match with the Dense Layer features_num ({W_Q.Size(0)}).");
            }
            if (input.Rank != 3 && input.Rank != 2)
            {
                throw new ShapeException($"Input must be of shape (B, L, H) or (L, H) for unbatched input, and input received has shape ({input.Shape.ToCommaSeparatedString()}).");
            }

            bool isBatched = input.Rank == 3;
            int batch_size = isBatched ? input.Size(0) : 1;

            Tensor[] batch_elem = isBatched ? input.Split(0, 1) : new Tensor[] { input };
            Tensor[] SDPA = new Tensor[batch_size];
            Q.Clear(); K.Clear(); V.Clear(); SoftmaxCache.Clear(); PostSoftmaxCache.Clear(); /// on simple predict there is no need for continuous forward caching.. so that's it
            for (int i = 0; i < batch_size; i++)
            {
                Tensor x = isBatched ? batch_elem[i].Squeeze(0) : batch_elem[i];
                InputCache.Push(x);
                Q.Push(Tensor.MatMul(x, W_Q, Device)); // (L, H) * (H, D) = (L, D)
                K.Push(Tensor.MatMul(x, W_K, Device)); // (L, D)
                V.Push(Tensor.MatMul(x, W_V, Device)); // (L, D)
                Tensor sdpa = Tensor.MatMul(Q.Peek(), K.Peek().Transpose(0, 1), Device); //(L, D) * (D, L) = (L, L)
                sdpa /= MathF.Sqrt(d); // (L, L)
                if (mask) Mask(sdpa);
                SoftmaxCache.Push(new Softmax());
                sdpa = SoftmaxCache.Peek().Forward(sdpa); // (L, L)
                PostSoftmaxCache.Push(sdpa); // no need for clone because concat makes a copy
                SDPA[i] = Tensor.MatMul(sdpa, V.Peek(), Device); // (L, D)
            }

            if (!isBatched)
                return Tensor.Concat(0, SDPA);

            return Tensor.Concat(null, SDPA);
        }
        public Tensor Forward(Tensor input)
        {
            return Predict(input);
        }
        public Tensor Backward(Tensor dLdy)
        {
            bool isBatched = dLdy.Rank == 3;
            int batch_size = isBatched ? dLdy.Size(0) : 1;

            Tensor[] batch_elem = isBatched ? dLdy.Split(0, 1) : new Tensor[] { dLdy };
            Tensor[] input_grad = new Tensor[batch_size];

            if(batch_size != SoftmaxCache.Count)
                throw new ArgumentException("Loss batch size is not matching the input batch size");
            
            for (int i = batch_size - 1; i >= 0; i--)
            {
                Tensor lossGrad = isBatched ? batch_elem[i].Squeeze(0) : batch_elem[i];
                Tensor vGrad = Tensor.MatMul(PostSoftmaxCache.Pop(), lossGrad, Device);    // V = (L, D), dLDY = (L, D), PSM = (L, L)
                Tensor QK_T_grad = SoftmaxCache.Pop().Backward(Tensor.MatMul(V.Pop(), lossGrad.Transpose(0, 1), Device)) / MathF.Sqrt(d); // (L, L)
                Tensor qGrad = Tensor.MatMul(QK_T_grad, K.Pop(), Device); // QKT = (L, L), k = (L, D), Q = (L, D)
                Tensor kGrad = Tensor.MatMul(QK_T_grad, Q.Pop(), Device);

                Tensor x = InputCache.Pop(); // x = (L, H), vGrad = (L, D)
                Tensor xT = x.Transpose(0, 1);
                W_V_grad += Tensor.MatMul(xT, vGrad, Device) / batch_size;
                W_Q_grad += Tensor.MatMul(xT, qGrad, Device) / batch_size;
                W_K_grad += Tensor.MatMul(xT, kGrad, Device) / batch_size;

                input_grad[i] = Tensor.Zeros(x.Size(-2), x.Size(-1));
                input_grad[i] += Tensor.MatMul(vGrad, W_V.Transpose(0, 1), Device); // (L, D) * (H, D)
                input_grad[i] += Tensor.MatMul(kGrad, W_K.Transpose(0, 1), Device);
                input_grad[i] += Tensor.MatMul(qGrad, W_Q.Transpose(0, 1), Device);
            }

            if (!isBatched)
                return Tensor.Concat(0, input_grad);

            return Tensor.Concat(null, input_grad);
        }

        public object Clone()
        {
            var att = new Attention();

            att.d = d;
            att.Device = Device;
            att.W_Q = (Tensor)W_Q.Clone();
            att.W_K = (Tensor)W_K.Clone();
            att.W_V = (Tensor)W_V.Clone();
            att.W_Q_grad = (Tensor)W_Q_grad.Clone();
            att.W_K_grad = (Tensor)W_K_grad.Clone();
            att.W_V_grad = (Tensor)W_V_grad.Clone();
            att.Q = new Stack<Tensor>(Q.Select(x => x.Clone() as Tensor));
            att.K = new Stack<Tensor>(K.Select(x => x.Clone() as Tensor));
            att.V = new Stack<Tensor>(V.Select(x => x.Clone() as Tensor));
            att.SoftmaxCache = new Stack<Softmax>(SoftmaxCache.Select(x => x.Clone() as Softmax));
            att.InputCache = new Stack<Tensor>(InputCache.Select(x => x.Clone() as Tensor));
            att.PostSoftmaxCache = new Stack<Tensor>(PostSoftmaxCache.Select(x => x.Clone() as Tensor));

            return att;
        }
        public Parameter[] Parameters()
        {
            if (W_Q_grad == null)
                OnAfterDeserialize();

            var q = new Parameter(W_Q, W_Q_grad);
            var k = new Parameter(W_K, W_K_grad);
            var v = new Parameter(W_V, W_V_grad);

            return new Parameter[] { q, k, v };
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
            W_Q_grad = Tensor.Zeros(W_Q.Shape);
            W_K_grad = Tensor.Zeros(W_K.Shape);
            W_V_grad = Tensor.Zeros(W_V.Shape);


            Q = new();
            K = new();
            V = new();
            SoftmaxCache = new();
            PostSoftmaxCache = new();
            InputCache = new();
        }

        /// <summary>
        /// Masks the sdpa by setting the upper triangular elements as - infinity.
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        private void Mask(Tensor sdpa)
        {
            // sdpa is (L, L)
            int L = sdpa.Size(-1);
            for (int i = 0; i < L; i++)
            {
                for (int j = 0; j < L; j++)
                {
                    if (j > i)
                        sdpa[i, j] = float.MinValue;
                }
            }
        }
    }

    // It seems like the old implementation is faster for larger batchsizes > 32.
    [Serializable]
    public class AttentionV2 : ILearnable, IModule
    {
        [SerializeField] public Device Device
        {
            get => W_Q.Device; set
            {
                W_Q.Device = value;
                W_K.Device = value;
                W_V.Device = value;
            }
        }

        [SerializeField] private int d;
        [SerializeField] private bool mask;

        [SerializeField] private Dense W_Q;
        [SerializeField] private Dense W_K;
        [SerializeField] private Dense W_V;
        [SerializeField] private Softmax softmax;

        private Tensor InputCache { get; set; }
        private Tensor PostSoftmaxCache { get; set; }
        private Tensor QCache { get; set; }
        private Tensor KCache { get; set; }
        private Tensor VCache { get; set; }

        public AttentionV2(int input_size, int embed_dim, bool mask = false, InitType weight_init = InitType.LeCun_Uniform, Device device = Device.CPU)
        {
            // H and d have the same dimension
            d = embed_dim;
            this.mask = mask;

            W_Q = new Dense(input_size, embed_dim, bias: false, weight_init, device: device);
            W_K = new Dense(input_size, embed_dim, bias: false, weight_init, device: device);
            W_V = new Dense(input_size, embed_dim, bias: false, weight_init, device: device);
            softmax = new Softmax();
        }
        private AttentionV2() { }


        public Tensor Predict(Tensor input)
        {
            Tensor Q = W_Q.Predict(input);
            Tensor K = W_K.Predict(input);
            Tensor V = W_V.Predict(input);

            var sdpa = Tensor.BatchedMatMul(Q, K.Transpose(-1, -2), device: Device) / Mathf.Sqrt(d);
            if (mask) Mask(sdpa);
            sdpa = softmax.Predict(sdpa);
            return Tensor.BatchedMatMul(sdpa, V, device: Device);
        }

        public Tensor Forward(Tensor input)
        {
            InputCache = input.Clone() as Tensor;
            QCache = W_Q.Forward(input);
            KCache = W_K.Forward(input);
            VCache = W_V.Forward(input);

            var sdpa = Tensor.BatchedMatMul(QCache, KCache.Transpose(-1, -2), device: Device) / Mathf.Sqrt(d);
            if (mask) Mask(sdpa);
            sdpa = softmax.Forward(sdpa);
            PostSoftmaxCache = sdpa;
            return Tensor.BatchedMatMul(sdpa, VCache, device: Device);
        }


        public Tensor Backward(Tensor dLdY)
        {
            if (dLdY.Rank != InputCache.Rank)
                throw new ArgumentException($"Loss rank {dLdY.Rank} not equal to input rank {InputCache.Rank}");

            Tensor vGrad = Tensor.BatchedMatMul(PostSoftmaxCache, dLdY, Device);
            Tensor xGrad = W_V.Backward(vGrad);

            Tensor qkTGrad = softmax.Backward(Tensor.BatchedMatMul(VCache, dLdY.Transpose(-1, -2), Device)) / MathF.Sqrt(d);
            Tensor qGrad = Tensor.BatchedMatMul(qkTGrad, KCache, Device);
            Tensor kGrad = Tensor.BatchedMatMul(qkTGrad, QCache, Device);

            xGrad += W_Q.Backward(qGrad);
            xGrad += W_K.Backward(kGrad);

            return xGrad;
        }


        public object Clone()
        {
            var att = new AttentionV2();
            att.d = this.d;
            att.mask = this.mask;
            att.Device = this.Device;
            att.W_Q = this.W_Q.Clone() as Dense;
            att.W_K = this.W_K.Clone() as Dense;
            att.W_V = this.W_V.Clone() as Dense;
            att.softmax = this.softmax.Clone() as Softmax;
            return att;
        }
        public Parameter[] Parameters()
        {
            if (W_Q == null)
                OnAfterDeserialize();

            var list = new List<Parameter>();
            list.AddRange(W_Q.Parameters());
            list.AddRange(W_K.Parameters());
            list.AddRange(W_V.Parameters());
            return list.ToArray();
        }

        public virtual void OnBeforeSerialize()
        {

        }
        public virtual void OnAfterDeserialize()
        {
            W_K.OnAfterDeserialize();
            W_V.OnAfterDeserialize();
            W_Q.OnAfterDeserialize();
        }
        private void Mask(Tensor sdpa)
        {
            // sdpa is (L, L)
            int L = sdpa.Size(-1);

            if (sdpa.Rank == 3)
            {
                int batch_size = sdpa.Size(0);
                for (int b = 0; b < batch_size; b++)
                {
                    for (int i = 0; i < L; i++)
                    {
                        for (int j = 0; j < L; j++)
                        {
                            if (j > i)
                                sdpa[b, i, j] = float.MinValue;
                        }
                    }
                }
            }
            else
                for (int i = 0; i < L; i++)
                {
                    for (int j = 0; j < L; j++)
                    {
                        if (j > i)
                            sdpa[i, j] = float.MinValue;
                    }
                }
        }
    }
}

