namespace DeepUnity
{
    public sealed class SGD : IOptimizer
    {
        private float learningRate;
        private float momentum;
        private float weightDecay;

        public SGD(float learningRate = 0.01f, float momentum = 0.9f, float weightDecay = 0.00001f)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
            this.weightDecay = weightDecay;
        }

        public void Step(Dense[] layers)
        {
            float decay = 1f - weightDecay * learningRate;
            int channels = layers[0].InputCache.FullShape[1];

            foreach (var L in layers)
            {
                L.mWeights = L.mWeights * momentum - L.gWeights * learningRate;
                L.Weights = L.Weights * decay + L.mWeights;

                L.mBiases = L.mBiases * momentum - L.gBiases * learningRate;
                L.Biases = L.Biases + L.mBiases;

                // Reset gradients
                L.gWeights.ForEach(x => 0f);
                L.gBiases.ForEach(x => 0f);
            }

        }
    }

}