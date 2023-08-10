using UnityEngine;
using System;
namespace DeepUnity
{
    /// <summary>
    /// Base class for all <see cref="IModule"/> that have learnable parameters.
    /// </summary>
    public abstract class Learnable : ISerializationCallbackReceiver
    {
        [SerializeField] public Device device;

        // Base weights and biases tensors.
        [SerializeField] public Tensor gamma;
        [SerializeField] public Tensor beta;
        [NonSerialized] public Tensor gammaGrad;
        [NonSerialized] public Tensor betaGrad;

        protected Learnable(Device device, InitType gamma_initializer, InitType beta_initializer, int[] gammaShape, int[] betaShape, int fan_in, int fan_out)
        {
            // Here will be no checkings the parameters are ok, only in each class constructors respectively.
            this.device = device;

            // Setting the grads to 0
            gammaGrad = Tensor.Zeros(gammaShape);
            betaGrad = Tensor.Zeros(betaShape);

            switch (gamma_initializer)
            {               
                case InitType.HE_Normal:
                    float sigmaHE = MathF.Sqrt(2f / fan_in);
                    gamma = Tensor.RandomNormal((0, sigmaHE), gammaShape);
                    break;
                case InitType.HE_Uniform:
                    float bound = MathF.Sqrt(6f / fan_in);
                    gamma = Tensor.RandomRange((-bound, bound), gammaShape);
                    break;
                case InitType.Glorot_Normal:
                    float sigmaXA = MathF.Sqrt(2f / (fan_in + fan_out));
                    gamma = Tensor.RandomNormal((0, sigmaXA), gammaShape);
                    break;
                case InitType.Glorot_Uniform:
                    float limit = MathF.Sqrt(6f / (fan_in + fan_out));
                    gamma = Tensor.RandomRange((-limit, limit), gammaShape);
                    break;
                case InitType.LeCun_Uniform:
                    float sqrtK = MathF.Sqrt(1f / fan_in);
                    gamma = Tensor.RandomRange((-sqrtK, sqrtK), gammaShape);
                    break;
                case InitType.LeCun_Normal:
                    float sigmaLC = MathF.Sqrt(3f / fan_in);
                    gamma = Tensor.RandomNormal((0, sigmaLC), gammaShape);
                    break;
                case InitType.Random_Normal:
                    gamma = Tensor.RandomNormal(gammaShape);
                    break;
                case InitType.Random_Uniform:
                    gamma = Tensor.RandomRange((-1f, 1f), gammaShape);
                    break;
                case InitType.Ones:
                    gamma = Tensor.Ones(gammaShape);
                    break;
                case InitType.Zeros:
                    gamma = Tensor.Zeros(gammaShape);
                    break;
                default:
                    throw new NotImplementedException("Unhandled initialization type!");
            }

            switch(beta_initializer)
            {
              
                case InitType.HE_Normal:
                    float sigmaHE = MathF.Sqrt(2f / fan_in);
                    beta = Tensor.RandomNormal((0, sigmaHE), betaShape);
                    break;
                case InitType.HE_Uniform:
                    float bound = MathF.Sqrt(6f / fan_in);
                    beta = Tensor.RandomRange((-bound, bound), betaShape);
                    break;
                case InitType.Glorot_Normal:
                    float sigmaXA = MathF.Sqrt(2f / (fan_in + fan_out));
                    beta = Tensor.RandomNormal((0, sigmaXA), betaShape);
                    break;
                case InitType.Glorot_Uniform:
                    float limit = MathF.Sqrt(6f / (fan_in + fan_out));
                    beta = Tensor.RandomRange((-limit, limit), betaShape);
                    break;
                case InitType.LeCun_Uniform:
                    float sqrtK = MathF.Sqrt(1f / fan_in);
                    beta = Tensor.RandomRange((-sqrtK, sqrtK), betaShape);
                    break;
                case InitType.LeCun_Normal:
                    float sigmaLC = MathF.Sqrt(3f / fan_in);
                    beta = Tensor.RandomNormal((0, sigmaLC), betaShape);
                    break;
                case InitType.Random_Normal:
                    beta = Tensor.RandomNormal(betaShape);
                    break;
                case InitType.Random_Uniform:
                    beta = Tensor.RandomRange((-1f, 1f), betaShape);
                    break;
                case InitType.Ones:
                    beta = Tensor.Ones(betaShape);
                    break;
                case InitType.Zeros:
                    beta = Tensor.Zeros(betaShape); 
                    break;
                default:
                    throw new NotImplementedException("Unhandled initialization type!");
            }
        }
      
        /// <summary>
        /// Set all gradients value to <b>0</b>.
        /// </summary>
        public virtual void ZeroGrad()
        {
            gammaGrad = Tensor.Zeros(gammaGrad.Shape);
            betaGrad = Tensor.Zeros(betaGrad.Shape);
        }
        /// <summary>
        /// Clips the gradients of the parameters in range (-<paramref name="clip_value"/>, <paramref name="clip_value"/>)
        /// </summary>
        /// <param name="clip_value"></param>
        public virtual void ClipGradValue(float clip_value)
        {
            Tensor.Clip(gammaGrad, -clip_value, clip_value);
            Tensor.Clip(betaGrad, -clip_value, clip_value);
        }
        /// <summary>
        /// ClipGrad by Norm applied locally only for this <see cref="Learnable"/> module
        /// </summary>
        /// <param name="max_norm"></param>
        public virtual void ClipGradNorm(float max_norm)
        {
            Tensor normG = Tensor.Norm(gammaGrad, NormType.ManhattanL1);

            if (normG[0] > max_norm)
            {
                float scale = max_norm / normG[0];
                gammaGrad *= scale;
            }

            Tensor normB = Tensor.Norm(betaGrad, NormType.ManhattanL1);

            if (normB[0] > max_norm)
            {
                float scale = max_norm / normB[0];
                betaGrad *= scale;
            }
        }
        /// <summary>
        /// Returns the number of all learnable parameters of this <see cref="Learnable"/> module.
        /// </summary>
        public virtual int ParametersCount() => gamma.Count() + beta.Count();
        public virtual void OnBeforeSerialize()
        {

        }
        public virtual void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (gamma.Shape == null)
                return;

            if (gamma.Shape.Length == 0)
                return;

            // do not check if gamma is != null...
            this.gammaGrad = Tensor.Zeros(gamma.Shape); 
            this.betaGrad = Tensor.Zeros(beta.Shape);
         
        }
    }
}

