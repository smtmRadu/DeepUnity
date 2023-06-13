using DeepUnity;
using System;
using UnityEngine;

namespace kbRadu
{
    [Serializable]
    public class LayerNorm: IModule, IParameters
    {
        // Only resource i could find so far...
        // https://www.youtube.com/watch?v=eyPZ9Mrhri4
        [SerializeField] public float epsilon;

        // Learnable parameters
        [SerializeField] public NDArray runningMean;
        [SerializeField] public NDArray runningVar;

        [SerializeField] public NDArray gamma;
        [SerializeField] public NDArray beta;


        public LayerNorm(int hid_units, float eps = 1e-5f)
        {
            this.epsilon = eps;

            this.gamma = NDArray.Ones(hid_units);
            this.beta = NDArray.Zeros(hid_units);
        }
        public NDArray Predict(NDArray input)
        {
            int batch = input.Shape[1];

            // input [features, batch]

            NDArray mu = NDArray.Mean(input, axis: 0);
            NDArray std = NDArray.Std(input, axis: 0);

            input = (input - NDArray.Expand(mu, axis: 0, times: batch))
                    / NDArray.Expand(std, axis: 0, times : batch);

            return input;
        }

        public NDArray Forward(NDArray input)
        {
            return null;
        }
        public NDArray Backward(NDArray loss)
        {
            return null;
        }


        public void ZeroGrad()
        {

        }
        public void ClipGradValue(float clip_value)
        {

        }
        public void ClipGradNorm(float norm_value)
        {

        }
        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {

        }

    }
}

