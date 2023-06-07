using UnityEngine;

namespace DeepUnity
{
    // https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html
    [System.Serializable]
    public class Adadelta : IOptimizer
    {
        [SerializeField] private float learningRate;
        [SerializeField] private float rho;
        [SerializeField] private float weightDecay;

        public Adadelta(float learningRate = 1.0f, float rho = 0.9f, float weightDecay = 0f)
        {
            this.learningRate = learningRate;
            this.weightDecay = weightDecay;
            this.rho = rho;
        }

        public void Step(Dense[] layers)
        {

            System.Threading.Tasks.Parallel.ForEach(layers, L =>
            {
                if(weightDecay != 0f)
                    L.g_W = L.g_W + weightDecay * L.t_W;
                
                L.v_W = L.v_W * rho + Tensor.Pow(L.g_W, 2f) * (1f - rho);
                L.v_B = L.v_B * rho + Tensor.Pow(L.g_B, 2f) * (1f - rho);

                // In Adadelta, i use v for square avg and m for accumulate variables
                var dxWeights = Tensor.Sqrt(L.m_W + Utils.EPSILON) / Tensor.Sqrt(L.v_W + Utils.EPSILON) * L.g_W;
                var dxBiases = Tensor.Sqrt(L.m_B + Utils.EPSILON) / Tensor.Sqrt(L.v_B + Utils.EPSILON) * L.g_B;

                L.m_W = L.m_W * rho + Tensor.Pow(dxWeights, 2f) * (1f - rho);
                L.m_B = L.m_B * rho + Tensor.Pow(dxBiases, 2f) * (1f - rho);

                L.t_W = L.t_W - learningRate * dxWeights;
                L.t_B = L.t_B - learningRate * dxBiases;

            });
        }
    }
}

