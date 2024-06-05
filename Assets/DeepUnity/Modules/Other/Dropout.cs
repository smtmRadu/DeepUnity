using System;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// <b>Placed after the non-linear activation function.</b> <br></br>
    /// Input: <b>(B, *)</b> or <b>(*)</b> for unbatched input. <br></br>
    /// Output: <b>(B, *)</b> or <b>(*)</b> for unbatched input. <br></br>
    /// where B = batch_size and * = input_shape.
    /// </summary>
    [Serializable]
    public class Dropout : IModule
    {
        [SerializeField] private bool inPlace = false;
        [SerializeField] private float dropout = 0.499999777646258f;
        private Tensor OutputCache { get; set; }

        /// <summary>
        /// <b>Placed after the non-linear activation function.</b> <br></br>
        /// Input: <b>(B, *)</b> or <b>(*)</b> for unbatched input. <br></br>
        /// Output: <b>(B, *)</b> or <b>(*)</b> for unbatched input. <br></br>
        /// where B = batch_size and * = input_shape.<br></br>
        /// <br></br>
        /// <em>The output shape is the same with the input shape.</em>
        /// </summary>
        /// <param name="dropout"> Low value: weak dropout | High value: strong dropout</param>
        public Dropout(float dropout = 0.5f, bool in_place = false)
        {
            if (dropout < Utils.EPSILON || dropout > 1f - Utils.EPSILON)
                throw new ArgumentException("Dropout value must be in range (0,1) when creating a Dropout layer module.");

            this.inPlace = in_place;
            this.dropout = dropout;
        }

        public Tensor Predict(Tensor input)
        {
            if (inPlace == true)
                return input;

            return input.Clone() as Tensor;
        }
        public Tensor Forward(Tensor input)
        {
            float scale = 1f / (1f - dropout);
            if(inPlace)
            {
                for (int i = 0; i < input.Count(); i++)
                {
                    input[i] = Utils.Random.Bernoulli(dropout) ? 0f : input[i] * scale;
                }
                OutputCache = input.Clone() as Tensor;
                return input;
            }
            else
            {
                OutputCache = input.Select(x => x = Utils.Random.Bernoulli(dropout) ? 0f : x * scale);
                return OutputCache.Clone() as Tensor;
            }
            
        }
        public Tensor Backward(Tensor loss)
        {
            return loss.Zip(OutputCache, (l, i) => i != 0f ? l : 0f);
        }

        public object Clone() => new Dropout(dropout, inPlace);
    }

}