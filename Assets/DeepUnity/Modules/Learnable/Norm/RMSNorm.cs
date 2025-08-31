using System;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
    // https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
    /// <summary>
    /// <b>Applies root mean square normalization over the last dimension (H) of the input:</b> x/<b>RMS</b>(x) • γ, where <b>RMS</b>(x) = √(1/n • ∑x² + ε) <br />
    /// <b>Placed before the non-linear activation function. </b>    <br />
    /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
    /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
    /// where B = batch_size and H = in_features.<br />
    /// </summary>
    [Serializable]
    public class RMSNorm : IModule, ILearnable
    {
        [SerializeField] public Device Device { get; set; } = Device.CPU;
        [SerializeField] public bool RequiresGrad { get; set; } = true;

        [SerializeField] private float epsilon = 1e-6f;
        [SerializeField] private bool affine = true;
        [SerializeField] public Tensor gamma;
        [SerializeField] private Tensor gammaGrad;

        private Tensor xHat { get; set; }
        private Tensor ms_x { get; set; }

        /// <summary>
        /// <b>Applies root mean square normalization over the last dimension (H) of the input:</b> x/<b>RMS</b>(x) • γ, where <b>RMS</b>(x) = √(ε + 1/n • ∑x²) <br />
        /// <b>Placed before the non-linear activation function. </b>    <br />
        /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// where B = batch_size and H = in_features.<br />
        /// </summary>
        public RMSNorm(int num_features, float eps = 1e-6f, bool elementwise_affine=true)
        {
            this.epsilon  = eps; 
            this.affine = elementwise_affine;

            if(elementwise_affine)
            {
                gamma = Tensor.Ones(num_features);
                gammaGrad = null;  //Tensor.Zeros(num_features);
            }
        }
        private RMSNorm() { }

        public object Clone()
        {
            RMSNorm rmsnorm = new RMSNorm();
            rmsnorm.epsilon = this.epsilon;  
            rmsnorm.affine = this.affine;
            if(affine)
            {
                rmsnorm.gamma = this.gamma.Clone() as Tensor;
                if(gammaGrad != null)
                    rmsnorm.gammaGrad = this.gammaGrad.Clone() as Tensor;
            }
           
            return rmsnorm;
        }


        public Tensor Predict(Tensor input)
        {
            if (input.Rank > 2)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) received is invalid for RMSNorm. Make sure is of shape (B, H) or (H).");

            bool isBatched = input.Rank == 2;

            ms_x = input.Square().Mean(-1, keepDim: true).Expand(-1, input.Size(-1));
            xHat = input * Tensor.RSqrt(ms_x + epsilon);

            if (!affine)
                return xHat;

            Tensor expanded_gamma = isBatched? gamma.Unsqueeze(0).Expand(0, input.Size(0)) : gamma;
            return xHat * expanded_gamma;

        }
        public Tensor Forward(Tensor input)
        {
            return Predict(input);
        }


        public Tensor Backward(Tensor dLdY)
        {
            bool isBatched = dLdY.Rank == 2;
            int feature_size = dLdY.Size(-1);
            int m = isBatched ? dLdY.Size(0) : 1;

            if (gammaGrad == null)
                gammaGrad = Tensor.Zeros(gamma.Shape);

            if(!affine)
            {
                Tensor dLdX_ = dLdY * (epsilon + ms_x * (1f - 1f / feature_size)) / (ms_x + epsilon).Pow(1.5f);
                return dLdX_;
            }
            
    
            Tensor expanded_gamma = affine ? (isBatched ? gamma.Unsqueeze(0).Expand(0, m) : gamma) : Tensor.Ones(dLdY.Size(-1));
            Tensor dLdGamma = dLdY * xHat;
            Tensor.CopyTo(gammaGrad + (isBatched ? dLdGamma.Mean(0) : dLdGamma), gammaGrad);
            Tensor dLdX = dLdY * expanded_gamma * (epsilon + ms_x * (1f - 1f / feature_size)) / (ms_x + epsilon).Pow(1.5f);
            return dLdX;
            

            
        }
        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (gamma.Shape == null)
                return;

            // do not check if gamma is != null...
            gammaGrad = Tensor.Zeros(gamma.Shape);

        }

        public Parameter[] Parameters()
        {
            if (gammaGrad == null)
                gammaGrad = Tensor.Zeros(gammaGrad.Shape);

            if (gammaGrad == null)
                OnAfterDeserialize();

            return new Parameter[] { new Parameter(gamma, gammaGrad)};
        }

    }

}
