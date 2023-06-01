using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Adam : IOptimizer
    {
        [SerializeField] private int timestep;
        [SerializeField] private float alpha;
        [SerializeField] private float beta1;
        [SerializeField] private float beta2;
        [SerializeField] private float weightDecay;

        
        public Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 1e-5f)
        {
            this.timestep = 0;
            this.alpha = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.weightDecay = weightDecay;
        }

        public void Step(Dense[] layers)
        {
            timestep++;

            System.Threading.Tasks.Parallel.ForEach(layers, L =>
            {
                Tensor mHat;
                Tensor vHat;

                // Update biased first momentum estimate
                L.mWeights = beta1 * L.mWeights + (1f - beta1) * L.gWeights;

                // Update biased second raw momentum estimate
                L.vWeights = beta2 * L.vWeights + (1f - beta2) * Tensor.Pow(L.gWeights, 2f);

                // Compute bias-corrected first momentum estimate
                mHat = L.mWeights / (1f - MathF.Pow(beta1, timestep));

                // Compute bias-corrected second raw momentum estimate
                vHat = L.vWeights / (1f - MathF.Pow(beta2, timestep));

                // Update parameters
                L.Weights = L.Weights * (1f - weightDecay) - alpha * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);




                // Update biased first momentum estimate
                L.mBiases = beta1 * L.mBiases + (1f - beta1) * L.gBiases;

                // Update biased second raw momentum estimate
                L.vBiases = beta2 * L.vBiases + (1f - beta2) * Tensor.Pow(L.gBiases, 2f);

                // Compute bias-corrected first momentum estimate
                mHat = L.mBiases / (1f - MathF.Pow(beta1, timestep));

                // Compute bias-corrected second raw momentum estimate
                vHat = L.vBiases / (1f - MathF.Pow(beta2, timestep));

                // Update parameters 
                L.Biases = L.Biases - alpha * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);


                // Reset gradients
                L.gWeights.ForEach(x => 0f);
                L.gBiases.ForEach(x => 0f);
            });

        }
    }

}