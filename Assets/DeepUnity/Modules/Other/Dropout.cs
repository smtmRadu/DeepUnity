using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Dropout : IModule, IModuleRNN
    {
        [SerializeField] private float dropout;
        public Tensor InputCache { get; set; }

        /// <summary>
        /// <b>Placed after the non-linear activation function.</b>
        /// </summary>
        /// <param name="dropout"> Low value: weak dropout | High value: strong dropout</param>
        public Dropout(float dropout = 0.5f)
        {
            if(dropout < Utils.EPSILON || dropout > 1f - Utils.EPSILON)
            {
                throw new ArgumentException("Dropout value must be in range (0,1) when creating a Dropout layer module.");
            }
            this.dropout = dropout;
        }

        public Tensor Predict(Tensor input) => input;
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);
            
            return input.Select(x => x = Utils.Random.Bernoulli(dropout) ? 0f : x);

        }
        public Tensor Backward(Tensor loss)
        {
            return loss.Zip(InputCache, (l, i) => i != 0f ? l : 0f);
        }
    }

}