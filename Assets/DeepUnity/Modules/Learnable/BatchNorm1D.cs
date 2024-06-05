using System;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// <b>Placed before or after the non-linear activation function.</b>    <br />
    /// Input: <b>(B, H)</b> on training and <b>(B, H)</b> or <b>(H)</b> on inference.<br></br>
    /// Output: <b>(B, H)</b> on training and <b>(B, H)</b> or <b>(H)</b> on inference.<br></br>
    /// where B = batch_size and H = num_features. <br></br>
    /// <b>Applies batch normalization over the first dimension (B) of the input.</b> 
    /// <br></br>
    /// <br></br>
    /// <em>TIPS:<br></br>
    ///     Shuffle training dataset. <br></br>
    ///     Increase learning rate. <br></br>
    ///     Remove dropout. <br></br>
    ///     Reduce L2 penalty. <br></br>
    ///     Accelerate learning rate decay. <br></br>
    /// </em>
    /// </summary>
    [Serializable]
    public class BatchNorm1D : ILearnable, IModule
    {
        // https://arxiv.org/pdf/1502.03167.pdf
        [SerializeField] public Device Device { get; set; } = Device.CPU;
        [SerializeField] public bool RequiresGrad { get; set; } = true;
        private Tensor xCentered { get; set; }
        private Tensor std { get; set; }
        private Tensor xHat { get; set; }

        [SerializeField] private int num_features;
        [SerializeField] private float momentum;
        [SerializeField] private float epsilon;

        // Learnable parameters
        [SerializeField] private Tensor runningMean;
        [SerializeField] private Tensor runningVar;

        // Affine parameters
        [SerializeField] private Tensor gamma;
        [SerializeField] private Tensor beta;
        [NonSerialized] private Tensor gammaGrad;
        [NonSerialized] private Tensor betaGrad;


        /// <summary>
        /// <b>Placed before or after the non-linear activation function.</b>    <br />
        /// Input: <b>(B, H)</b> on training and <b>(B, H)</b> or <b>(H)</b> on inference.<br></br>
        /// Output: <b>(B, H)</b> on training and <b>(B, H)</b> or <b>(H)</b> on inference.<br></br>
        /// where B = batch_size and H = num_features. <br></br>
        /// <b>Applies batch normalization over the first dimension (B) of the input.</b> 
        /// <br></br>
        /// <br></br>
        /// <em>TIPS:<br></br>
        ///     Shuffle training dataset. <br></br>
        ///     Increase learning rate. <br></br>
        ///     Remove dropout. <br></br>
        ///     Reduce L2 penalty. <br></br>
        ///     Accelerate learning rate decay. <br></br>
        /// </em>
        /// 
        /// </summary>
        /// <param name="num_features">Input's last axis dimension (H).</param>
        /// <param name="momentum">Small batch size (0.9 - 0.99), Big batch size (0.6 - 0.85). Best momentum value is <b>m</b> where <b>m = batch.size / dataset.size</b></param>
        /// <param name="affine">Do you use gamma and beta affine parameters?</param>       
        public BatchNorm1D(int num_features, float eps = 1e-5f, float momentum = 0.9f, bool affine = true)
        {
            if (num_features < 1)
                throw new ArgumentException($"BatchNorm layer cannot have num_features < 1. (Received arg: {num_features})");

            this.num_features = num_features;
            this.momentum = momentum;
            this.epsilon = eps;

            if(affine)
            {
                gamma = Tensor.Ones(num_features);
                beta = Tensor.Zeros(num_features);
                gammaGrad = Tensor.Zeros(num_features);
                betaGrad = Tensor.Zeros(num_features);
            }
            

            runningVar = Tensor.Ones(num_features);
            runningMean = Tensor.Zeros(num_features);
        }

        public Tensor Predict(Tensor input)
        {
            if (input.Rank > 2)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) must be (B, H) or (H).");


            if (input.Size(-1) != num_features)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) last dimension is not equal to num_features ({num_features})");


            bool isBatched = input.Rank == 2;

            if (isBatched)
            {
                int batch_size = input.Size(0);
                var e_mean = Tensor.Expand(Tensor.Unsqueeze(runningMean, 0), 0, batch_size);
                var e_std = Tensor.Expand(Tensor.Unsqueeze((runningVar + epsilon).Sqrt(), 0), 0, batch_size);
                var input_centered = (input - e_mean) / e_std;
                
                if (gamma == null) // not affine
                    return input_centered;

                var e_gamma = Tensor.Expand(Tensor.Unsqueeze(gamma, 0), 0, batch_size);
                var e_beta = Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);              
                var output = e_gamma * input_centered + e_beta;

                return output;
            }
            else
            {
                var input_centered = (input - runningMean) / ((runningVar + epsilon).Sqrt());

                if(gamma == null) // not affine
                    return input_centered;

                var output = gamma * input_centered + beta;
                return output;
            }

        }
        public Tensor Forward(Tensor input)
        {
            if (input.Size(-1) != num_features)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) last dimension is not equal to num_features ({num_features})");

            if (input.Rank != 2 || input.Size(-2) < 2)
                throw new InputException($"On training, input shape must be (B, H) exclusively, where B > 1 (received input shape: {input.Shape.ToCommaSeparatedString()}). For inference, use Predict method instead.");

            int batch_size = input.Size(0);

            // When training (only on mini-batch training), we cache the values for backprop also
            var mean = Tensor.Mean(input, 0, keepDim: true); // mini-batch means      [1, features_mean]
            var variance_biased = Tensor.Var(input, 0, correction: 0, keepDim: true); // zero-corrected mini-batch variances  [1, features_mean]
            var variance_unbiased = input.Var(0, correction: 1); // corrected variance

            // normalize and cache
            std = (variance_biased + epsilon).Sqrt().Expand(0, batch_size);
            xCentered = input - Tensor.Expand(mean, 0, batch_size);
            xHat = xCentered / std;


            // compute running mean and var     
            runningMean = runningMean * momentum + mean.Squeeze(0) * (1f - momentum);
            runningVar = runningVar * momentum + variance_unbiased * (1f - momentum);

            // no affine
            if (gamma == null)
                return xHat;

            // affine ON
            return Tensor.Expand(Tensor.Unsqueeze(gamma, 0), 0, batch_size) * xHat + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
        }
        public Tensor Backward(Tensor dLdY)
        {
            int m = dLdY.Size(0);

            // differentiation on https://arxiv.org/pdf/1502.03167.pdf page 4

            var dLdxHat = gamma == null ? dLdY : dLdY * gamma.Unsqueeze(0).Expand(0, m); // [batch, outs]

            var dLdVarB = Tensor.Mean(
                         dLdxHat * xCentered * (-1f / 2f) * (std.Square() + epsilon).Pow(-3f / 2f),
                         axis: 0,
                         keepDim: true).Expand(0, m);
            var dLdMuB = Tensor.Mean(
                         dLdxHat * -1f / std + dLdVarB * -2f * xCentered / m,
                         axis: 0,
                         keepDim: true).Expand(0, m);

            var dLdX = dLdxHat / std + dLdVarB * 2f * xCentered / m + dLdMuB * (1f / m);

            if(RequiresGrad && gamma != null)
            {
                var dLdGamma = Tensor.Mean(dLdY * xHat, 0);
                var dLdBeta = Tensor.Mean(dLdY, 0);

                Tensor.CopyTo(gammaGrad + dLdGamma, gammaGrad);
                Tensor.CopyTo(betaGrad + dLdBeta, betaGrad);

            }

            return dLdX;
        }

        public object Clone()
        {
            if(gamma != null)
            {
                BatchNorm1D bnclone = new BatchNorm1D(num_features, epsilon, momentum);
                bnclone.Device = Device;
                bnclone.RequiresGrad = RequiresGrad;
                bnclone.gamma = (Tensor)gamma.Clone();
                bnclone.beta = (Tensor)beta.Clone();
                bnclone.gammaGrad = (Tensor)gammaGrad.Clone();
                bnclone.betaGrad = (Tensor)betaGrad.Clone();
                bnclone.runningMean = (Tensor)runningMean.Clone();
                bnclone.runningVar = (Tensor)runningVar.Clone();
                return bnclone;
            }
            else
            {
                BatchNorm1D bnclone = new BatchNorm1D(1, epsilon, momentum, false);
                bnclone.Device = Device;
                bnclone.RequiresGrad = RequiresGrad;
                bnclone.runningMean = (Tensor)runningMean.Clone();
                bnclone.runningVar = (Tensor)runningVar.Clone();
                return bnclone;
            }
           
        }
        public Parameter[] Parameters()
        {
            if (gammaGrad == null)
                OnAfterDeserialize();

            var g = new Parameter(gamma, gammaGrad);
            var b = new Parameter(beta, betaGrad);

            return new Parameter[] { g, b };
        }

        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.
            if (gamma == null)
                return;

            if (gamma.Shape == null)
                return;

            if (gamma.Shape.Length == 0)
                return;

            // do not check if gamma is != null...
            gammaGrad = Tensor.Zeros(gamma.Shape);
            betaGrad = Tensor.Zeros(beta.Shape);

        }

        // TIPS for improvement
        // increase learn rate
        // remove dropout and reduce L2 penalty (BN regularizes the network)
        // accelerate lr decay (on StepLR)
        // shuffle dataset more

        /*   Improving BN networks (placed before activation)
             Increase learning rate. In a batch-normalized model,
          we have been able to achieve a training speedup from
          higher learning rates, with no ill side effects (Sec. 3.3).
             Remove Dropout. As described in Sec. 3.4, Batch Normalization fulfills some of the same goals as Dropout. Removing Dropout from Modified BN-Inception speeds up
          training, without increasing overfitting.
            Reduce the L2 weight regularization. While in Inception an L2 loss on the model parameters controls overfitting, in Modified BN-Inception the weight of this loss is
          reduced by a factor of 5. We find that this improves the
          accuracy on the held-out validation data.
            Accelerate the learning rate decay. In training Inception, learning rate was decayed exponentially. Because
          our network trains faster than Inception, we lower the
          learning rate 6 times faster.
            Remove Local Response Normalization While Inception and other networks (Srivastava et al., 2014) benefit
          from it, we found that with Batch Normalization it is not
          necessary.
            Shuffle training examples more thoroughly. We enabled
          within-shard shuffling of the training data, which prevents
          the same examples from always appearing in a mini-batch
          together. This led to about 1% improvements in the validation accuracy, which is consistent with the view of
          Batch Normalization as a regularizer (Sec. 3.4): the randomization inherent in our method should be most beneficial when it affects an example differently each time it is
          seen.
            Reduce the photometric distortions. Because batchnormalized networks train faster and observe each training example fewer times, we let the trainer focus on more
          “real” images by distorting them less.
            */
    }
}