using System;

namespace DeepUnity
{
    public sealed class Adam : IOptimizer
    {
        private int timestep;
        private float stepsize;
        private float beta1;
        private float beta2;
        private float weightDecay;

        public Adam(float learningRate = 0.001f, float beta1  = 0.9f, float beta2 = 0.999f, float weightDecay = 1e-5f)
        {
            this.timestep = 0;
            this.stepsize = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.weightDecay = weightDecay;
        }

        public void Step(Dense[] layers)
        {
            timestep++;

            float decay = 1f - weightDecay * stepsize;
            int channels = layers[0].InputCache.FullShape[1];
            float alpha = stepsize / channels;

            foreach (var L in layers)
            {
                Tensor<float> mHat;
                Tensor<float> vHat;

                // Update weights

                    // Update biased first momentum estimate
                    L.mWeights = beta1 * L.mWeights + (1f - beta1) * L.gWeights;

                    // Update biased second raw momentum estimate
                    L.vWeights = beta2 * L.vWeights + (1f - beta2) * (L.gWeights ^ 2f);

                    // Compute bias-corrected first momentum estimate
                    mHat = L.mWeights / (1 - MathF.Pow(beta1, timestep));

                    // Compute bias-corrected second raw momentum estimate
                    vHat = L.vWeights / (1 - MathF.Pow(beta2, timestep));

                    // Update parameters
                    L.Weights = L.Weights * decay - alpha * mHat / (vHat ^ 0.5f + Utils.EPSILON);

                // Update biases

                     // Update biased first momentum estimate
                     L.mBiases = beta1 * L.mBiases + (1f - beta1) * L.gBiases;

                     // Update biased second raw momentum estimate
                     L.vBiases = beta2 * L.vBiases + (1f - beta2) * (L.gBiases ^ 2f);

                     // Compute bias-corrected first momentum estimate
                     mHat = L.mBiases / (1 - MathF.Pow(beta1, timestep));

                     // Compute bias-corrected second raw momentum estimate
                     vHat = L.vBiases / (1 - MathF.Pow(beta2, timestep));

                     // Update parameters 
                     L.Biases = L.Biases * decay - alpha * mHat / (vHat ^ 0.5f + Utils.EPSILON);



                // Reset gradients
                L.gWeights.ForEach(x => 0f);
                L.gBiases.ForEach(x => 0f);
            }

        }
    }

}
