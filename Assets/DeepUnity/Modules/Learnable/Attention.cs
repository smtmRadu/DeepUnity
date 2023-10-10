using System;
using UnityEngine;


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
    public class Attention : Learnable, IModule, ISelfOptimizable
    {
        // Caches used for each operation in order to compute the gradients
        private Tensor InputCache { get; set; }
        private Tensor QueryCache { get; set; }
        private Tensor KeyCache { get; set; }   
        private Tensor ValueCache { get; set; }
        private Tensor HeadCache { get; set; }

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
        public Attention((int, int) input_shape, int embed_dim, float scale = 1f) : 
            base(Device.CPU, InitType.HE_Normal, InitType.HE_Normal, new int[] {1}, new int[] {1}, 1, 1)
        {

            throw new NotSupportedException("Attention layer is not implemented yet.");

            this.scale = scale;
            this.d = embed_dim;

            // For the sake of simplicity d_q, d_k and d_v are the same. 
            // This will produce the same output shape as the input.
            int T = input_shape.Item1;
            int H = input_shape.Item2;    
            W_Q = Tensor.RandomNormal((0f, 0.3f), H, d);
            W_K = Tensor.RandomNormal((0f, 0.3f), H, d);
            W_V = Tensor.RandomNormal((0f, 0.3f), H, d);
            W_O = Tensor.RandomNormal((0f, 0.3f), d, H);

            softmax = new Softmax();
        }
        public Tensor Predict(Tensor input)
        {
            // softmax(Q * KT / Sqrt(D)) * V [Scaled Dot-Product Attention, page 4]
            // input shape (B, T, H)

            Tensor Q = Tensor.MatMulGPU(input, W_Q); // (T, H) * (H, D) = (T, D)
            Tensor K = Tensor.MatMulGPU(input, W_K); // (T, D)
            Tensor V = Tensor.MatMulGPU(input, W_V); // (T, D)

            
            Tensor scaled_dot_product = Tensor.MatMulGPU(Q, K.Transpose(0, 1)); // (T, T)
            scaled_dot_product *= scale;
            scaled_dot_product /= MathF.Sqrt(d);
            scaled_dot_product = softmax.Forward(scaled_dot_product);
            scaled_dot_product = Tensor.MatMulGPU(scaled_dot_product, V); // (T, T) * (T, D) = (T, D)

            scaled_dot_product = Tensor.MatMulGPU(scaled_dot_product, W_O); // reshape the SDP to obtain the same output's shape as input's.
            return input + scaled_dot_product;
        }

        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);

            Tensor Q = Tensor.MatMulGPU(input, W_Q); // (T, H) * (H, D) = (T, D)
            Tensor K = Tensor.MatMulGPU(input, W_K); // (T, D)
            Tensor V = Tensor.MatMulGPU(input, W_V); // (T, D)
            QueryCache = Tensor.Identity(Q);
            KeyCache = Tensor.Identity(K);
            ValueCache = Tensor.Identity(V);

            Tensor scaled_dot_product = Tensor.MatMulGPU(Q, K.Transpose(0, 1)); // (T, T)
            scaled_dot_product *= scale;
            scaled_dot_product /= MathF.Sqrt(d);
            scaled_dot_product = softmax.Forward(scaled_dot_product);
            scaled_dot_product = Tensor.MatMulGPU(scaled_dot_product, V); // (T, T) * (T, D) = (T, D)

            HeadCache = Tensor.Identity(scaled_dot_product);
            scaled_dot_product = Tensor.MatMulGPU(scaled_dot_product, W_O); // reshape the SDP to obtain the same output's shape as input's.
            return input + scaled_dot_product;
        }
        public Tensor Backward(Tensor loss)
        {
            float gamma = 1f / MathF.Sqrt(d);

            Tensor I = GetIMatrix(); // I s only on diagonal and edges
            Tensor P = GetPMatrix(); // P is identity matrix of size n x n
            Tensor A = GetAMatrix(); // A is a bit more complicated
            Tensor S = GetSMatrix(); // S is also more complicated
            Tensor V_prime = Tensor.MatMulGPU(ValueCache, W_V);

            // df(X) / dX
            Tensor dLossdInput = null;  ///  Where is d f(X) / d X ?

            // df(x) / dA
            Tensor dLossdA = Tensor.MatMulGPU(dLossdInput, W_O.Transpose(0, 1));
            dLossdA = Tensor.MatMulGPU(dLossdA, V_prime.Transpose(0, 1));
            dLossdA *= S;
            Tensor part2 = Tensor.MatMulGPU(dLossdInput, W_O.Transpose(0, 1));
            part2 = Tensor.MatMulGPU(part2, V_prime.Transpose(0, 1));
            part2 *= S;
            part2 *= GetIpsilonMatrix(Tensor.MatMulGPU(A.Exp(), I));
            part2 = Tensor.MatMulGPU(part2, I.Transpose(0, 1));
            part2 *= A.Exp();
            dLossdA -= part2;

            // df(X) / dWV
            Tensor dLossdWV = Tensor.MatMulGPU(A.Transpose(0, 1), dLossdInput);
            dLossdWV = Tensor.MatMulGPU(dLossdWV, W_O.Transpose(0, 1));
            W_V_grad = dLossdWV;

            // df(X) / dWQ
            Tensor dLossdWQ = gamma * Tensor.MatMulGPU(QueryCache.Transpose(0, 1), dLossdA);
            dLossdWQ = Tensor.MatMulGPU(dLossdWQ, P.Transpose(0, 1));
            dLossdWQ = Tensor.MatMulGPU(dLossdWQ, KeyCache);
            dLossdWQ = Tensor.MatMulGPU(dLossdWQ, W_K);
            W_Q_grad = dLossdWQ;

            // df(X) / dWK
            Tensor dLossdWK = gamma * Tensor.MatMulGPU(KeyCache.Transpose(0, 1), P);
            dLossdWK = Tensor.MatMulGPU(dLossdWK, dLossdA.Transpose(0, 1));
            dLossdWK = Tensor.MatMulGPU(dLossdWK, QueryCache);
            dLossdWK = Tensor.MatMulGPU(dLossdWK, W_Q);
            W_K_grad = dLossdWK;

            // df(X)/ dWO
            Tensor dLossdWO = HeadCache.Transpose(0, 1) * dLossdInput;
            W_O_grad = dLossdWO;

            return dLossdInput;
        }

        public void SelfOptimise(float lr)
        {
            // optimise value

            W_Q -= lr * W_Q_grad;
            W_O -= lr * W_O_grad;
            W_K -= lr * W_K_grad;
            W_V -= lr * W_V_grad;
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
        private Tensor GetAMatrix()
        {
            Tensor A = Tensor.MatMulGPU(QueryCache, W_Q);
            A = Tensor.MatMulGPU(A, W_K.Transpose(0, 1));
            A = Tensor.MatMulGPU(A, KeyCache.Transpose(0, 1));
            A /= MathF.Sqrt(d);
            A = new Softmax().Forward(A);
            A = Tensor.MatMulGPU(A, ValueCache);
            return A;
        }

        private Tensor GetSMatrix()
        {
            Tensor S = Tensor.MatMulGPU(QueryCache, W_Q);
            S = Tensor.MatMulGPU(S, W_K.Transpose(0, 1));
            S = Tensor.MatMulGPU(S, KeyCache.Transpose(0, 1));
            S /= MathF.Sqrt(d);
            S = new Softmax().Forward(S);
            return S;
        }
    }

}

