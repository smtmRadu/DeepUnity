using UnityEngine;
using System;
namespace DeepUnity
{
    public abstract class Learnable : ISerializationCallbackReceiver
    {
        [SerializeField] public Device device;

        [SerializeField] public Tensor gamma;
        [SerializeField] public Tensor beta;
        [NonSerialized] public Tensor gammaGrad;
        [NonSerialized] public Tensor betaGrad;

        public Learnable(Device device) => this.device = device;

        public void ZeroGrad()
        {
            if (this is RNNCell R)
            {
                R.weightIHGrad = Tensor.Zeros(R.weightIHGrad.Shape);
                R.weightHHGrad = Tensor.Zeros(R.weightHHGrad.Shape);
                R.biasIHGrad = Tensor.Zeros(R.biasIHGrad.Shape);
                R.biasHHGrad = Tensor.Zeros(R.biasHHGrad.Shape);
            }
            else
            {
                gammaGrad = Tensor.Zeros(gammaGrad.Shape);
                betaGrad = Tensor.Zeros(betaGrad.Shape);
            }

                   
        }
        public void ClipGradValue(float clip_value)
        {
            if (this is RNNCell R)
            {
                Tensor.Clip(R.weightIHGrad, -clip_value, clip_value);
                Tensor.Clip(R.weightHHGrad, -clip_value, clip_value);
                Tensor.Clip(R.biasIHGrad, -clip_value, clip_value);
                Tensor.Clip(R.biasHHGrad, -clip_value, clip_value);
            }
            else
            {
                Tensor.Clip(gammaGrad, -clip_value, clip_value);
                Tensor.Clip(betaGrad, -clip_value, clip_value);
            }
           
        }
        public void ClipGradNorm(float max_norm)
        {
            if (this is RNNCell R)
            {
                throw new NotImplementedException("RNNCell ClipGradNorm not implemented yet");
            }
            else
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
            
            
            
        }
        public virtual void OnBeforeSerialize()
        {

        }
        public virtual void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            try
            {
                var dev = device;

                // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.
                var x = gamma.Shape;
                if (x == null || x.Length == 0)
                    throw new Exception("Is not even important...");
            
            }
            catch
            {
                return;
            }

            // do not check if gamma is != null...
            this.gammaGrad = Tensor.Zeros(gamma.Shape); 
            this.betaGrad = Tensor.Zeros(beta.Shape);
         
        }
    }
}

