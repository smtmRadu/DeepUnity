using System;
using Unity.VisualScripting.Antlr3.Runtime;
using UnityEngine;
using UnityEngine.Windows;


/// This module is not going to be released due to high computational requirements....
/// 



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
    /// Input: <b>(B, T, H)</b> or <b>(T, H)</b> for unbatched input. <br></br>
    /// Output: <b>(B, T, H)</b> or <b>(T, H)</b> for unbatched input. <br></br>
    /// where B = batch_size, T = sequence_length / channels, H = num_features.<br></br>
    /// <b>Placed after the non-linear activation function.</b> <br></br>
    /// <br></br>
    /// <br></br>
    /// <em>TIPS: <br></br>
    ///     For images (C, H, W), the shape must be flattened into (T, H*), where T = C and H* = H · W. </em><br></br>
    /// </summary>
    public class Attention : ILearnable, IModule
    {
        // Caches used for each operation in order to compute the gradients
        private Tensor InputCache { get; set; }
        private Tensor[] QueryCache { get; set; }
        private Tensor[] KeyCache { get; set; }   
        private Tensor[] ValueCache { get; set; }
        private Tensor[] HeadCache { get; set; }

        [SerializeField] private Tensor W_Q;
        [SerializeField] private Tensor W_K;
        [SerializeField] private Tensor W_V;
        [SerializeField] private Tensor W_O;
        [NonSerialized] private Tensor W_Q_grad;
        [NonSerialized] private Tensor W_K_grad;
        [NonSerialized] private Tensor W_V_grad;
        [NonSerialized] private Tensor W_O_grad;


        [SerializeField] private float scale;
        [SerializeField] private int d;

        private Softmax softmax;

        /// <summary>
        /// <b>Applies a Scaled Dot-Product Attention over the input.</b> <br></br>
        /// Input: <b>(B, T, H)</b> or <b>(T, H)</b> for unbatched input. <br></br>
        /// Output: <b>(B, T, H)</b> or <b>(T, H)</b> for unbatched input. <br></br>
        /// where B = batch_size, T = sequence_length / channels, H = num_features.<br></br>
        /// <b>Placed after the non-linear activation function.</b> <br></br>
        /// <br></br>
        /// <br></br>
        /// <em>TIPS: <br></br>
        ///     For images (C, H, W), the shape must be flattened into (T, H*), where T = C and H* = H · W. </em><br></br>
        /// </summary>
        /// <param name="input_shape">(T, H)</param>
        /// <param name="embed_dim">Total dimension of the model.</param>
        /// <param name="scale">Apply scale in self dot-product attention.</param>
        public Attention((int, int) input_shape, int embed_dim, float scale = 1f)
        {

            throw new NotSupportedException("Attention layer is not implemented yet.");
            // remember that tr() is trace of the matrix *
            // this.scale = scale;
            // this.d = embed_dim;
            // 
            // // For the sake of simplicity d_q, d_k and d_v are the same. 
            // // This will produce the same output shape as the input.
            // int T = input_shape.Item1;
            // int H = input_shape.Item2;    
            // W_Q = Tensor.RandomNormal((0f, 0.2f), H, d);
            // W_K = Tensor.RandomNormal((0f, 0.2f), H, d);
            // W_V = Tensor.RandomNormal((0f, 0.2f), H, d);
            // W_O = Tensor.RandomNormal((0f, 0.2f), d, H);
            // 
            // softmax = new Softmax();
        }
        /// <summary>
        /// Input: <b>(B, T, H)</b> or <b>(T, H)</b> for unbatched input. <br></br>
        /// Output: <b>(B, T, H)</b> or <b>(T, H)</b> for unbatched input. <br></br>
        /// where B = batch_size, T = sequence_length / channels, H = num_features.<br></br>
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Tensor Predict(Tensor input)
        {
            // softmax(Q * KT / Sqrt(D)) * V [Scaled Dot-Product Attention, page 4]
            // input shape (B, T, H)
            int batch_size = input.Rank == 3 ? input.Size(0) : 1;

            Tensor[] batches = batch_size == 1 ? new Tensor[] { input } : input.Split(0, 1);
            Tensor[] sdp = new Tensor[batch_size];

            for (int i = 0; i < batches.Length; i++)
            {
                Tensor slice = batches[i];

                Tensor Q = Tensor.MatMulGPU(slice, W_Q); // (T, H) * (H, D) = (T, D)
                Tensor K = Tensor.MatMulGPU(slice, W_K); // (T, D)
                Tensor V = Tensor.MatMulGPU(slice, W_V); // (T, D)


                Tensor scaled_dot_product = Tensor.MatMulGPU(Q, K.Transpose(0, 1)); // (T, T)
                scaled_dot_product *= scale;
                scaled_dot_product /= MathF.Sqrt(d);
                scaled_dot_product = softmax.Forward(scaled_dot_product);
                scaled_dot_product = Tensor.MatMulGPU(scaled_dot_product, V); // (T, T) * (T, D) = (T, D)

                scaled_dot_product = Tensor.MatMulGPU(scaled_dot_product, W_O); // reshape the SDP to obtain the same output's shape as input's.
                sdp[i] = scaled_dot_product;
            }
            
            return input + Tensor.Concat(null, sdp);
        }
        /// <summary>
        /// Input: <b>(B, T, H)</b> or <b>(T, H)</b> for unbatched input. <br></br>
        /// Output: <b>(B, T, H)</b> or <b>(T, H)</b> for unbatched input. <br></br>
        /// where B = batch_size, T = sequence_length / channels, H = num_features.<br></br>
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Tensor Forward(Tensor input)
        {
           
            InputCache = Tensor.Identity(input);
            int batch_size = input.Rank == 3 ? input.Size(0) : 1;

            Tensor[] batches = batch_size == 1 ? new Tensor[] { input } : input.Split(0, 1);
            Tensor[] sdp = new Tensor[batch_size];

            QueryCache = new Tensor[batch_size];
            KeyCache = new Tensor[batch_size];
            ValueCache = new Tensor[batch_size];
            HeadCache = new Tensor[batch_size];

            for (int i = 0; i < batches.Length; i++)
            {
                Tensor slice = batches[i];
                Tensor Q = Tensor.MatMulGPU(slice, W_Q); // (T, H) * (H, D) = (T, D)
                Tensor K = Tensor.MatMulGPU(slice, W_K); // (T, D)
                Tensor V = Tensor.MatMulGPU(slice, W_V); // (T, D)
                QueryCache[i] = Tensor.Identity(Q);
                KeyCache[i] = Tensor.Identity(K);
                ValueCache[i] = Tensor.Identity(V);

                Tensor scaled_dot_product = Tensor.MatMulGPU(Q, K.Transpose(0, 1)); // (T, T)
                scaled_dot_product *= scale;
                scaled_dot_product /= MathF.Sqrt(d);
                scaled_dot_product = softmax.Forward(scaled_dot_product);
                scaled_dot_product = Tensor.MatMulGPU(scaled_dot_product, V); // (T, T) * (T, D) = (T, D)

                HeadCache[i] = Tensor.Identity(scaled_dot_product);
                scaled_dot_product = Tensor.MatMulGPU(scaled_dot_product, W_O); // reshape the SDP to obtain the same output's shape as input's.
                sdp[i] = scaled_dot_product;
            }

            return input + Tensor.Concat(null, sdp);
        }
        public Tensor Backward(Tensor dfx_dx)
        {
            float gamma = 1f / MathF.Sqrt(d);
            int batch_size = dfx_dx.Rank == 3 ? dfx_dx.Size(0) : 1;

            Tensor[] batches = batch_size == 1 ? new Tensor[] { dfx_dx } : dfx_dx.Split(0, 1);
            Tensor[] dSDP = new Tensor[batch_size];

            for (int i = 0; i < batches.Length; i++)
            {
                Tensor I = GetIMatrix(); // I s only on diagonal and edges
                Tensor P = GetPMatrix(); // P is identity matrix of size n x n
                Tensor A = GetAMatrix(i); // A is a bit more complicated
                Tensor S = GetSMatrix(i); // S is also more complicated
                Tensor V_prime = Tensor.MatMulGPU(ValueCache[i], W_V);

                // df(x) / dA
                Tensor dLossdA = Tensor.MatMulGPU(dfx_dx, W_O.Transpose(0, 1));
                dLossdA = Tensor.MatMulGPU(dLossdA, V_prime.Transpose(0, 1));
                dLossdA *= S;
                Tensor part2 = Tensor.MatMulGPU(dfx_dx, W_O.Transpose(0, 1));
                part2 = Tensor.MatMulGPU(part2, V_prime.Transpose(0, 1));
                part2 *= S;
                part2 *= GetIpsilonMatrix(Tensor.MatMulGPU(A.Exp(), I));
                part2 = Tensor.MatMulGPU(part2, I.Transpose(0, 1));
                part2 *= A.Exp();
                dLossdA -= part2;

                // df(X) / dWV
                Tensor dLossdWV = Tensor.MatMulGPU(A.Transpose(0, 1), dfx_dx);
                dLossdWV = Tensor.MatMulGPU(dLossdWV, W_O.Transpose(0, 1));
                W_V_grad += dLossdWV / batch_size;

                // df(X) / dWQ
                Tensor dLossdWQ = gamma * Tensor.MatMulGPU(QueryCache[i].Transpose(0, 1), dLossdA);
                dLossdWQ = Tensor.MatMulGPU(dLossdWQ, P.Transpose(0, 1));
                dLossdWQ = Tensor.MatMulGPU(dLossdWQ, KeyCache[i]);
                dLossdWQ = Tensor.MatMulGPU(dLossdWQ, W_K);
                W_Q_grad += dLossdWQ / batch_size;

                // df(X) / dWK
                Tensor dLossdWK = gamma * Tensor.MatMulGPU(KeyCache[i].Transpose(0, 1), P);
                dLossdWK = Tensor.MatMulGPU(dLossdWK, dLossdA.Transpose(0, 1));
                dLossdWK = Tensor.MatMulGPU(dLossdWK, QueryCache[i]);
                dLossdWK = Tensor.MatMulGPU(dLossdWK, W_Q);
                W_K_grad += dLossdWK / batch_size;

                // df(X)/ dWO
                Tensor dLossdWO = HeadCache[i].Transpose(0, 1) * dfx_dx;
                W_O_grad += dLossdWO / batch_size;
            }

                

            // now let's differentiate the scaled dot product.
            // return Tensor.Cat(null, dSDP);
            return null;
        }

        private  Tensor GetIpsilonMatrix(Tensor X)
        {
            return Tensor.Ones(X.Shape) / X;
        }

        private Tensor GetIMatrix()
        {
            int n = W_O.Size(-1); // num_features
            var mat = Tensor.Zeros(n, n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if(i == 0 || j == 0 || (i == n - 1) || (j == n - 1) || i == j)
                    { mat[i, j] = 1; }
                }
            }
            return mat;
        }
        private Tensor GetPMatrix()
        {
            int n = W_O.Size(-1); // num_features
            var mat = Tensor.Zeros(n, n);
            for (int i = 0; i < n; i++)
            {
                mat[i, i] = 1;
            }
            return mat;
        }
        private Tensor GetAMatrix(int batch_index)
        {
            Tensor A = Tensor.MatMulGPU(QueryCache[batch_index], W_Q);
            A = Tensor.MatMulGPU(A, W_K.Transpose(0, 1));
            A = Tensor.MatMulGPU(A, KeyCache[batch_index].Transpose(0, 1));
            A /= MathF.Sqrt(d);
            A = new Softmax().Forward(A);
            A = Tensor.MatMulGPU(A, ValueCache[batch_index]);
            return A;
        }
        private Tensor GetSMatrix(int batch_index)
        {
            Tensor S = Tensor.MatMulGPU(QueryCache[batch_index], W_Q);
            S = Tensor.MatMulGPU(S, W_K.Transpose(0, 1));
            S = Tensor.MatMulGPU(S, KeyCache[batch_index].Transpose(0, 1));
            S /= MathF.Sqrt(d);
            S = new Softmax().Forward(S);
            return S;
        }

        public object Clone()
        {
            var att = new Attention((1, 1), this.d, this.scale);
            att.W_Q = (Tensor)this.W_Q.Clone();
            att.W_K = (Tensor)this.W_K.Clone();
            att.W_V = (Tensor)this.W_V.Clone();
            att.W_O = (Tensor)this.W_O.Clone();
            att.W_Q_grad = (Tensor)this.W_Q_grad.Clone();
            att.W_K_grad = (Tensor)this.W_K_grad.Clone();
            att.W_V_grad = (Tensor)this.W_V_grad.Clone();
            att.W_O_grad = (Tensor)this.W_O_grad.Clone();
            att.softmax = (Softmax)this.softmax.Clone();
            return att;

        }



        public void SetDevice(Device device)
        {
            return;
        }
        public int ParametersCount()
        {
            return W_Q.Count() + W_K.Count() + W_V.Count() + W_O.Count();
        }
        public Parameter[] Parameters()
        {
            if (W_Q_grad == null)
                OnAfterDeserialize();

            var q = new Parameter(W_Q, W_Q_grad);
            var k = new Parameter(W_K, W_K_grad);
            var v = new Parameter(W_V, W_V_grad);
            var o = new Parameter(W_O, W_O_grad);

            return new Parameter[] { q , k , v , o  };
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
            this.W_O_grad = Tensor.Zeros(W_O.Shape);
        }
    }

}

