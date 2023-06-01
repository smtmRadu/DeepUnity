using System;
using UnityEngine;
namespace DeepUnity
{
    [Serializable]
    public sealed class SGD : IOptimizer
    {
        [SerializeField] private float learningRate;
        [SerializeField] private float momentum;
        [SerializeField] private float weightDecay;

        public SGD(float learningRate = 0.01f, float momentum = 0.9f, float weightDecay = 1e-5f)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
            this.weightDecay = weightDecay;
        }

        public void Step(Dense[] layers)
        {
            int channels = layers[0].InputCache.Shape[1];

            System.Threading.Tasks.Parallel.ForEach(layers, L =>
            {
                L.mWeights = L.mWeights * momentum - L.gWeights * learningRate;
                L.mBiases = L.mBiases * momentum - L.gBiases * learningRate;

                L.Weights = L.Weights * (1f - weightDecay) + L.mWeights;
                L.Biases = L.Biases + L.mBiases;

                // Reset gradients
                L.gWeights.ForEach(x => 0f);
                L.gBiases.ForEach(x => 0f);
            });
        }
    }

}