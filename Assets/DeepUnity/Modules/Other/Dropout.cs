using System;

namespace DeepUnity
{
    /// <summary>
    /// Always placed after a non-linear activation function.
    /// </summary>
    [Serializable]
    public class Dropout : IModule
    {
        public float dropout;
        public Tensor InputCache { get; set; }


        public Dropout(float dropout = 0.5f) => this.dropout = dropout;

        public Tensor Predict(Tensor input) => input;
        public Tensor Forward(Tensor input)
        {
            input.ForEach(x => Utils.Random.Bernoulli(dropout) ? 0f : x);
            InputCache = Tensor.Identity(input);
            return input;

        }
        public Tensor Backward(Tensor loss)
        {
            return loss.Zip(InputCache, (l, i) => i != 0f ? l : 0f);
        }
    }

}