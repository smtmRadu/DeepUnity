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

        // [SerializeField] protected TensorGPU gammaGPU;
        // [SerializeField] protected TensorGPU betaGPU;
        // [SerializeField] protected TensorGPU gammaGradGPU;
        // [SerializeField] protected TensorGPU betaGradGPU;
        public Learnable(Device device) => this.device = device;

        public void ZeroGrad()
        {
            gammaGrad = Tensor.Zeros(gammaGrad.Shape);
            betaGrad = Tensor.Zeros(betaGrad.Shape);
            
            // else
            // {
            //     gammaGradGPU = TensorGPU.Zeros(gammaGradGPU.Shape);
            //     betaGradGPU = TensorGPU.Zeros(betaGradGPU.Shape);
            // }          
        }
        public void ClipGradValue(float clip_value)
        {
  
            Tensor.Clip(gammaGrad, -clip_value, clip_value);
            Tensor.Clip(betaGrad, -clip_value, clip_value);
            
            // else
            // {
            //     TensorGPU.Clip(gammaGradGPU, -clip_value, clip_value);
            //     TensorGPU.Clip(betaGradGPU, -clip_value, clip_value);
            // }
        }
        public void ClipGradNorm(float max_norm)
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
            // else
            // {
            //     TensorGPU normG = TensorGPU.Norm(gammaGradGPU, NormType.ManhattanL1);
            // 
            //     if (normG[0] > max_norm)
            //     {
            //         float scale = max_norm / normG[0];
            //         gammaGradGPU *= scale;
            //     }
            // 
            // 
            //     TensorGPU normB = TensorGPU.Norm(betaGradGPU, NormType.ManhattanL1);
            // 
            //     if (normB[0] > max_norm)
            //     {
            //         float scale = max_norm / normB[0];
            //         betaGradGPU *= scale;
            //     }
            // }
            
        }
        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            try
            {
                var dev = device;

                // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.
                var x = gamma.Shape;
                if (x == null || x.Length == 0)
                    throw new System.Exception("Is not even important...");
            
                // else
                // {
                //     // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.
                //     var x = gammaGPU.Shape;
                //     if (x == null || x.Length == 0)
                //         throw new System.Exception("Is not even important...");
                // }
                
            }
            catch
            {
                return;
            }


            this.gammaGrad = Tensor.Zeros(gamma.Shape);
            this.betaGrad = Tensor.Zeros(beta.Shape);

            // else
            // {
            //     this.gammaGradGPU = TensorGPU.Zeros(gammaGPU.Shape);
            //     this.betaGradGPU = TensorGPU.Zeros(betaGPU.Shape);
            // }
         
        }
    }
}

