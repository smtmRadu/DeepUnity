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
        [SerializeField] public Tensor runningMean;
        [SerializeField] public Tensor runningVar;

        [SerializeField] public Tensor gamma;
        [SerializeField] public Tensor beta;


        public LayerNorm(int hid_units, float eps = 1e-5f)
        {
            this.epsilon = eps;

            this.gamma = Tensor.Ones(hid_units);
            this.beta = Tensor.Zeros(hid_units);
        }
        public Tensor Predict(Tensor input)
        {
            int batch = input.Shape.width;

            // input [features, batch]

            Tensor mu = Tensor.Mean(input, 0);
            Tensor std = Tensor.Std(input, 0);


            input = (input - Tensor.Expand(mu, 0, batch))
                    / Tensor.Expand(std, 0, batch);

            return input;
        }

        public Tensor Forward(Tensor input)
        {
            return null;
        }
        public Tensor Backward(Tensor loss)
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

