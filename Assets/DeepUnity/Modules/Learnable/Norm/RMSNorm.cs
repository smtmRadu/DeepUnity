using System;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
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
       
        [SerializeField] private float epsilon = 1e-8f;

        private Tensor xHat { get; set; }
        private Tensor rmsNorm { get; set; }

        /// <summary>
        /// <b>Placed before the non-linear activation function. </b>    <br />
        /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// where  B = batch_size and H = in_features.<br />
        /// <b>Applies root mean square normalization over the last dimension (H) of the input.</b> 
        /// </summary>
        /// <param name="affine">Train gamma and beta parameters (elementwise-affine).</param>
        public RMSNorm(float eps = 1e-8f)
        {
            this.epsilon  = eps;     
        }
        private RMSNorm() { }

        public object Clone()
        {
            RMSNorm rmsnorm = new RMSNorm();
            rmsnorm.epsilon = this.epsilon;      
            return rmsnorm;
        }


        public Tensor Predict(Tensor input)
        {
            if (input.Rank > 2)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) received is invalid for RMSNorm. Make sure is of shape (B, H) or (H).");
      
            rmsNorm = (input.Square().Mean(-1, keepDim: true) + epsilon).Sqrt().Expand(-1, input.Size(-1));
            xHat = input / rmsNorm;
            return xHat;
        }
        public Tensor Forward(Tensor input)
        {
            return Predict(input);
        }
        public Tensor Backward(Tensor dLdY)
        {
            return dLdY / rmsNorm;
        }
    }

}
