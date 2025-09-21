using System;
using System.Threading.Tasks;
using DeepUnity.Modules;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Activations
{
    // https://www.youtube.com/watch?v=09c7bkxpv9I

    /// <summary>
    /// <b>Applies the Softmax function over the last input's dimension H (axis: -1).</b> <br></br>
    /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input <br></br>
    /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input <br></br>
    /// where * = any shape and H = features_num
    /// </summary>
    [Serializable]
    public sealed class Softmax : IModule, IActivation
    {
        [SerializeField] private float temperature = 1f;
        private Tensor OutputCache { get; set; }

        /// <summary>
        /// <b>Applies the Softmax function over the last input's dimension H (axis: -1).</b> <br></br>
        /// Input: <b>(B, H)</b>, <b>(H)</b> or  <b>(B, L, H)</b>, <b>(L, H)</b> for sequential input <br></br>
        /// Output: <b>(B, H)</b>, <b>(H)</b> or  <b>(B, L, H)</b>, <b>(L, H)</b> for sequential input <br></br>
        /// where * = any shape and H = features_num
        /// </summary>
        /// <param name="online">When true, computes softmax with a rolling algorithm.</param>
        public Softmax(float temperature = 1f) 
        {
            if (temperature <= 0f)
                throw new ArgumentException("Temperature cannot be less or equal with 0");

            this.temperature = temperature;
        }

       
        public Tensor Predict(Tensor input)
        {
            int iRank = input.Rank;
            if (iRank == 0)
                throw new ShapeException($"Softmax input must be of shape (H), (B, H), (L, H) or (B, L, H) (received ({input.Shape.ToCommaSeparatedString()})).");

                Tensor y = input.Clone() as Tensor;

                int rank = input.Rank;
                int H = input.Size(-1);

            if (rank == 1)
            {
                float m = float.NegativeInfinity;
                float l = 0f;

                for (int h = 0; h < H; h++)
                {
                    float x = input[h] / temperature;
                    float mNew = Mathf.Max(m, x);
                    l = l * MathF.Exp(m - mNew) + MathF.Exp(x - mNew);
                    m = mNew;
                }
                for (int h = 0; h < H; h++)
                {
                    float x = input[h] / temperature;
                    y[h] = MathF.Exp(x - m) / l;
                }
                return y;
            }
            else if (rank == 2)
            {
                int N = input.Size(0);
                for (int n = 0; n < N; n++)
                {
                    float m = float.NegativeInfinity;
                    float l = 0f;

                    for (int h = 0; h < H; h++)
                    {
                        float x = input[n, h] / temperature;
                        float mNew = MathF.Max(m, x);
                        l = l * MathF.Exp(m - mNew) + MathF.Exp(x - mNew);
                        m = mNew;
                    }
                    for (int h = 0; h < H; h++)
                    {
                        float x = input[n, h] / temperature;
                        y[n, h] = MathF.Exp(x - m) / l;
                    }
                }
                return y;
            }
            else if (rank == 3)
            {
                int A = input.Size(0);
                int B = input.Size(1);
                for (int a = 0; a < A; a++)
                {
                    for (int b = 0; b < B; b++)
                    {
                        float m = float.NegativeInfinity;
                        float l = 0f;

                        for (int h = 0; h < H; h++)
                        {
                            float x = input[a, b, h] / temperature;
                            float mNew = Mathf.Max(m, x);
                            l = l * MathF.Exp(m - mNew) + Mathf.Exp(x - mNew);
                            m = mNew;
                        }
                        for (int h = 0; h < H; h++)
                        {
                            float x = input[a, b, h] / temperature;
                            y[a, b, h] = MathF.Exp(x - m) / l;
                        }
                    }
                }
                return y;
            }
            else
            {
                throw new ArgumentException($"Softmax doesn't allow rank=4 input. (received shape: {input.Shape.ToCommaSeparatedString()})");
            }
            

            // Safe softmax
            // Tensor z = input / temperature;
            // Tensor max_ = z.Max(-1, keepDim: true).Expand(-1, z.Size(-1));
            // Tensor exp = Tensor.Exp(z - max_);
            // Tensor exp_sum = Tensor.Sum(exp, -1, true);
            // exp_sum = Tensor.Expand(exp_sum, -1, exp.Size(-1));
            // return exp / exp_sum;


            // Classic softmax
            // Tensor exp = Tensor.Exp(input / temperature);
            // Tensor exp_sum = Tensor.Sum(exp, -1, true);
            // exp_sum = Tensor.Expand(exp_sum, -1, exp.Size(-1));
            // Tensor y = exp / exp_sum;
            // return y;
     
        }
        public Tensor Forward(Tensor input)
        {
            Tensor y = Predict(input);
            OutputCache = y.Clone() as Tensor;
            return y;
        }
        public Tensor Backward(Tensor dLdY)
        {
            if (OutputCache.Rank != dLdY.Rank)
                throw new ArgumentException($"Input received in Softmax is Rank {OutputCache.Rank} and the output gradient is Rank {dLdY.Rank}.");

            return RecursiveLoRBackward(dLdY, OutputCache);
        }

        private Tensor RecursiveLoRBackward(Tensor dLdY, Tensor outputCache) // LoR means Low Rank decomp
        {
            if (dLdY.Rank == 3)
            {           
                Tensor[] batchElems_sm = outputCache.Split(0, 1);
                Tensor[] batchElems_loss = dLdY.Split(0, 1);
                Tensor[] batchElems_inputGrad = new Tensor[batchElems_sm.Length];

                Parallel.For(0, batchElems_loss.Length, i =>
                {
                    batchElems_inputGrad[i] = RecursiveLoRBackward(batchElems_loss[i].Squeeze(0), batchElems_sm[i].Squeeze(0));
                });
                
                return Tensor.Concat(null, batchElems_inputGrad);
            }
            if (dLdY.Rank == 2)
            {
                Tensor[] seqElems_sm = outputCache.Split(0, 1);
                Tensor[] seqElems_loss = dLdY.Split(0, 1);
                Tensor[] seqElems_inputGrad = new Tensor[seqElems_sm.Length];

                Parallel.For(0, seqElems_loss.Length, i =>
                {
                    seqElems_inputGrad[i] = RecursiveLoRBackward(seqElems_loss[i].Squeeze(0), seqElems_sm[i].Squeeze(0));
                });

                return Tensor.Concat(null, seqElems_inputGrad);
            }
            else if (dLdY.Rank == 1)
            {
                // slow method
                if(false)
                {
                    int H = outputCache.Size(-1);

                    Tensor jacobian_softmax = Tensor.Zeros(H, H);

                    for (int j = 0; j < H; j++)
                    {
                        for (int i = 0; i < H; i++)
                        {
                            float kdelta = i == j ? 1 : 0;
                            jacobian_softmax[j, i] += outputCache[i] * (kdelta - outputCache[j]);
                        }
                    }

                    return Tensor.MatMul(dLdY, jacobian_softmax);
                }
                // fast method
                else
                {
                    Tensor dot = Tensor.Dot(dLdY, outputCache);
                    Tensor tmp = dLdY - dot[0];
                    return tmp * outputCache;
                }
            }
            else
                throw new ArgumentException("Backward not allowed for Rank4 input.");
            
        }

        public object Clone() => new Softmax(temperature);
    }

}
