using System;
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
    public class Softmax : IModule, IActivation
    {
        [SerializeField] private float temperature = 1f;
        /// <summary>
        /// <b>Applies the Softmax function over the last input's dimension H (axis: -1).</b> <br></br>
        /// Input: <b>(B, H)</b>, <b>(H)</b> or  <b>(B, L, H)</b>, <b>(L, H)</b> for sequential input <br></br>
        /// Output: <b>(B, H)</b>, <b>(H)</b> or  <b>(B, L, H)</b>, <b>(L, H)</b> for sequential input <br></br>
        /// where * = any shape and H = features_num
        /// </summary>
        public Softmax(float temperature = 1f) 
        {
            if (temperature <= 0f)
                throw new ArgumentException("Temperature cannot be less or equal than 1");

            this.temperature = temperature;
        }

        private Tensor OutputCache { get; set; }
        public Tensor Predict(Tensor input)
        {
            int iRank = input.Rank;
            if (iRank == 0 || iRank == 4)
                throw new ShapeException($"Softmax input must be of shape (H), (B, H), (L, H) or (B, L, H) (received ({input.Shape.ToCommaSeparatedString()})).");

            // softmax(x[i]) = e^x[i] / sum{j:1->H}(e^x[j]])
            Tensor exp = Tensor.Exp(input / temperature);
            Tensor exp_sum = Tensor.Sum(exp, -1, true);
            exp_sum = Tensor.Expand(exp_sum, -1, exp.Size(-1));
            Tensor y = exp / exp_sum;
            return y;
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

        private Tensor RecursiveLoRBackward(Tensor dLdY, Tensor outputCache)
        {
            if (dLdY.Rank == 3)
            {           
                Tensor[] batchElems_sm = outputCache.Split(0, 1);
                Tensor[] batchElems_loss = dLdY.Split(0, 1);
                Tensor[] batchElems_inputGrad = new Tensor[batchElems_sm.Length];
                for (int i = 0; i < batchElems_loss.Length; i++)
                    batchElems_inputGrad[i] = RecursiveLoRBackward(batchElems_loss[i].Squeeze(0), batchElems_sm[i].Squeeze(0));
                return Tensor.Concat(null, batchElems_inputGrad);
            }
            if (dLdY.Rank == 2)
            {
                Tensor[] seqElems_sm = outputCache.Split(0, 1);
                Tensor[] seqElems_loss = dLdY.Split(0, 1);
                Tensor[] seqElems_inputGrad = new Tensor[seqElems_sm.Length];
                for (int i = 0; i < seqElems_loss.Length; i++)
                    seqElems_inputGrad[i] = RecursiveLoRBackward(seqElems_loss[i].Squeeze(0), seqElems_sm[i].Squeeze(0));
                return Tensor.Concat(null, seqElems_inputGrad);
            }
            else // Case one vector
            {
                int H = outputCache.Size(-1);

                Tensor jacobian_softmax = Tensor.Zeros(H, H);

                for (int j = 0; j < H; j++)
                {
                    for (int i = 0; i < H; i++)
                    {
                        float delta = i == j ? 1 : 0;
                        jacobian_softmax[j, i] += outputCache[i] * (delta - outputCache[j]);
                    }
                }

                return Tensor.MatMul(dLdY, jacobian_softmax);
            }
            
        }

        public object Clone() => new Softmax(temperature);
    }

}
