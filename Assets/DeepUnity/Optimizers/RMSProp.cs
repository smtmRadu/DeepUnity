using System;
using UnityEngine;

namespace DeepUnity
{
    // basic RMS prop alg https://medium.com/analytics-vidhya/a-complete-guide-to-adam-and-rmsprop-optimizer-75f4502d83be
    // pytorch algorithm https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    [Serializable]
    public sealed class RMSProp : Optimizer
    {
        [SerializeField] private float alpha; //smoothing constant
        [SerializeField] private float momentum;
        [SerializeField] private bool centered;

        // square average 
        [NonSerialized] public Tensor[] vGamma;
        [NonSerialized] public Tensor[] vBeta;

        // buffer   
        [NonSerialized] public Tensor[] buffGamma;
        [NonSerialized] public Tensor[] buffBeta;

        // avg grad
        [NonSerialized] public Tensor[] gaveGamma;
        [NonSerialized] public Tensor[] gaveBeta;

        public RMSProp(Learnable[] parameters, float lr = 0.01f, float alpha = 0.99f, float momentum = 0.9f, float weightDecay = 0f, bool centered = false) : base(parameters,lr, weightDecay)
        {
            this.alpha = alpha;
            this.momentum = momentum;
            this.centered = centered;

            vGamma = new Tensor[parameters.Length];
            vBeta = new Tensor[parameters.Length];


            buffGamma = new Tensor[parameters.Length];
            buffBeta = new Tensor[parameters.Length];

          
            gaveGamma = new Tensor[parameters.Length];
            gaveBeta = new Tensor[parameters.Length];
            

            for (int i = 0; i < parameters.Length; i++)
            {
                Learnable P = parameters[i];

                buffGamma[i] = Tensor.Zeros(P.gamma.Shape);
                buffBeta[i] = Tensor.Zeros(P.beta.Shape);

                vGamma[i] = Tensor.Zeros(P.gamma.Shape);
                vBeta[i] = Tensor.Zeros(P.beta.Shape);

                if (centered)
                {
                    gaveGamma[i] = Tensor.Zeros(P.gamma.Shape);
                    gaveBeta[i] = Tensor.Zeros(P.beta.Shape);
                }
               
            }
        }

        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                // Vanilla RMSProp
                // Learnable P = parameters[i];
                //
                // v_dGamma[i] = alpha * v_dGamma[i] + (1f - alpha) * Tensor.Pow(P.gradGamma, 2);
                // v_dBeta[i] = alpha * v_dBeta[i] + (1f - alpha) * Tensor.Pow(P.gradBeta, 2);
                //
                // P.gamma = P.gamma - learningRate * P.gradGamma / (Tensor.Sqrt(v_dGamma[i]) + Utils.EPSILON);
                // P.beta = P.beta - learningRate * P.gradBeta / (Tensor.Sqrt(v_dBeta[i]) + Utils.EPSILON);

                if (parameters[i] is Learnable L)
                {
                    if (weightDecay != 0)
                        L.gammaGrad = L.gammaGrad + weightDecay * L.gamma;


                    vGamma[i] = alpha * vGamma[i] + (1f - alpha) * Tensor.Pow(L.gammaGrad, 2f);
                    vBeta[i] = alpha * vBeta[i] + (1f - alpha) * Tensor.Pow(L.betaGrad, 2f);
                
                    var vBarGamma = Tensor.Identity(vGamma[i]);
                    var vBarBeta = Tensor.Identity(vBeta[i]);
                
                    if (centered)
                    {
                        gaveGamma[i] = gaveGamma[i] * alpha + (1f - alpha) * L.gammaGrad;
                        gaveBeta[i] = gaveBeta[i] * alpha + (1f - alpha) * L.betaGrad;
                
                        vBarGamma = vBarGamma - Tensor.Pow(gaveGamma[i], 2f);
                        vBarBeta = vBarBeta - Tensor.Pow(gaveBeta[i], 2f);
                    }
                
                    if (momentum > 0f)
                    {
                        buffGamma[i] = momentum * buffGamma[i] + L.gammaGrad / (Tensor.Sqrt(vBarGamma) + Utils.EPSILON);
                        buffBeta[i] = momentum * buffBeta[i] + L.betaGrad / (Tensor.Sqrt(vBarBeta) + Utils.EPSILON);
                
                        L.gamma = L.gamma - learningRate * buffGamma[i];
                        L.beta = L.beta - learningRate * buffBeta[i];
                    }
                    else
                    {
                        L.gamma = L.gamma - learningRate * L.gammaGrad / (Tensor.Sqrt(vBarGamma) + Utils.EPSILON);
                        L.beta = L.beta - learningRate * L.betaGrad / (Tensor.Sqrt(vBarBeta) + Utils.EPSILON);
                    }
                }
                if (parameters[i] is ISelfOptimizable S)
                    S.SelfOptimise(learningRate);
            });
        }
    }

}