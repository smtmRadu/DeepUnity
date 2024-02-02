using UnityEngine;
using System;
using System.Linq;

namespace DeepUnity
{
    [Serializable]
    public class PReLU : ILearnable, IModule, IActivation
    {
        private Tensor InputCache { get; set; }

        [SerializeField] private Tensor alpha;
        [NonSerialized] private Tensor alphaGrad;

        /// <summary>
        /// Parametric ReLU activation. Do not use weight decay for better performance.
        /// </summary>
        /// <param name="init_value">The initial value of the learnable parameter.</param>
        public PReLU(float init_value = 0.25f)
        {
            alpha = Tensor.Constant(init_value);
            alphaGrad = Tensor.Zeros(1);
        }

        public Tensor Predict(Tensor input)
        {
            return input.Select(x => MathF.Max(0, x) + alpha[0] * MathF.Min(0, x));
        }

        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);
            return Predict(input);
        }

        public Tensor Backward(Tensor loss)
        {
            // dLoss/dTheta = x if x < 0 and 0 otherwise
            float dLda = loss.Select(x => x >= 0f ? x : 0).ToArray().Average();
            alphaGrad[0] = dLda;
            return loss * InputCache.Select(x => x >= 0f ? 1f : alpha[0]);
        }

      

        public void SetDevice(Device device)
        {
            return;
        }
        public int ParametersCount()
        {
            return alpha.Count();
        }

        public Parameter[] Parameters()
        {
            if (alphaGrad == null)
                OnAfterDeserialize();

            return new Parameter[] { new Parameter(alpha, alphaGrad)};
        }

        public void OnBeforeSAerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            if (alpha == null)
                return;

            if (alpha.Shape.Length == 0)
                return;

            this.alphaGrad = Tensor.Zeros(alpha.Shape);
        }
        public object Clone() => new PReLU(alpha[0]);
    }

}


