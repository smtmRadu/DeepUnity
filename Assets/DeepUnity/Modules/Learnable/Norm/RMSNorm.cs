using System;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
    // https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
    /// <summary>
    /// <b>Placed before the non-linear activation function. </b>    <br />
    /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
    /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
    /// where  B = batch_size and H = in_features.<br />
    /// <b>Applies root mean square normalization over the last dimension (H) of the input.</b> 
    /// </summary>
    [Serializable]
    public class RMSNorm : IModule
    {
        [SerializeField] public Device Device { get; set; } = Device.CPU;
        [SerializeField] public bool RequiresGrad { get; set; } = true;

        [SerializeField] private float epsilon = 1e-8f;
        [SerializeField] private bool affine = true;
        [SerializeField] private Tensor gamma;
        [SerializeField] private Tensor gammaGrad;

        private Tensor xHat { get; set; }
        private Tensor rms_x { get; set; }

        /// <summary>
        /// Applies x/(√rms(x) + e) * γ. <br />
        /// <b>Placed before the non-linear activation function. </b>    <br />
        /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// where  B = batch_size and H = in_features.<br />
        /// <b>Applies root mean square normalization over the last dimension (H) of the input.</b> 
        /// </summary>
        /// <param name="affine">Train gamma and beta parameters (elementwise-affine).</param>
        public RMSNorm(int num_features, float eps = 1e-10f, bool elementwise_affine=true)
        {
            this.epsilon  = eps; 
            this.affine = elementwise_affine;

            if(elementwise_affine)
            {
                gamma = Tensor.Ones(num_features);
                gammaGrad = Tensor.Zeros(num_features);
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
                rmsnorm.gammaGrad = this.gammaGrad.Clone() as Tensor;
            }
           
            return rmsnorm;
        }


        public Tensor Predict(Tensor input)
        {
            if (input.Rank > 2)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) received is invalid for RMSNorm. Make sure is of shape (B, H) or (H).");

            bool isBatched = input.Rank == 2;

            rms_x = input.Square().Mean(-1, keepDim: true).Expand(-1, input.Size(-1));
            xHat = input / Tensor.Sqrt(rms_x + epsilon);

            if (!affine)
                return xHat;

            Tensor expanded_gamma = isBatched ? gamma.Unsqueeze(0).Expand(0, input.Size(0)) : gamma;
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

            if(!affine)
            {
                Tensor dLdX_ = dLdY * (epsilon + rms_x * (1f - 1f / feature_size)) / (rms_x + epsilon).Pow(1.5f);
                return dLdX_;
            }
            
    
            Tensor expanded_gamma = affine ? (isBatched ? gamma.Unsqueeze(0).Expand(0, m) : gamma) : Tensor.Ones(dLdY.Size(-1));
            Tensor dLdGamma = dLdY * xHat;
            Tensor.CopyTo(gammaGrad + (isBatched ? dLdGamma.Mean(0) : dLdGamma), gammaGrad);
            Tensor dLdX = dLdY * expanded_gamma * (epsilon + rms_x * (1f - 1f / feature_size)) / (rms_x + epsilon).Pow(1.5f);
            return dLdX;
            

            
        }

    }

}
