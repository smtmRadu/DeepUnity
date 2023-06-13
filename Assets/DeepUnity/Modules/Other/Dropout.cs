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
        public NDArray InputCache { get; set; }


        public Dropout(float dropout = 0.5f) => this.dropout = dropout;

        public NDArray Predict(NDArray input) => input;
        public NDArray Forward(NDArray input)
        {
            input.ForEach(x => Utils.Random.Bernoulli(dropout)? 0f : x);
            InputCache = NDArray.Identity(input);
            return input;

        }
        public NDArray Backward(NDArray loss)
        {
            return loss.Zip(InputCache, (l, i) => i != 0f ? l : 0f);
        }
    }

}
