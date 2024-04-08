using System;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// <b>Placed before or after the non-linear activation function.</b>    <br />
    /// Input: <b>(B, C, H, W)</b> on training and <b>(B, C, H, W)</b> or <b>(C, H, W)</b> on inference.<br></br>
    /// Output: <b>(B, C, H, W)</b> on training and <b>(B, C, H, W)</b> or <b>(C, H, W)</b> on inference.<br></br>
    /// where B = batch_size, C = channel_size, H = height and W = width. <br></br>
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
    public class BatchNorm2D : ILearnable, IModule
    {
        // https://arxiv.org/pdf/1502.03167.pdf
        [SerializeField] public Device Device { get; set; } = Device.CPU;

        private Tensor xCentered { get; set; }
        private Tensor std { get; set; }
        private Tensor xHat { get; set; }

        [SerializeField] private int num_features;
        [SerializeField] private float momentum;

        // Learnable parameters
        [SerializeField] private Tensor runningMean;
        [SerializeField] private Tensor runningVar;

        [SerializeField] private Tensor gamma;
        [SerializeField] private Tensor beta;
        [NonSerialized] private Tensor gammaGrad;
        [NonSerialized] private Tensor betaGrad;


        /// <summary>
        /// <b>Placed before or after the non-linear activation function.</b>    <br />
        /// Input: <b>(B, C, H, W)</b> on training and <b>(B, C, H, W)</b> or <b>(C, H, W)</b> on inference.<br></br>
        /// Output: <b>(B, C, H, W)</b> on training and <b>(B, C, H, W)</b> or <b>(C, H, W)</b> on inference.<br></br>
        /// where B = batch_size, C = channel_size, H = height and W = width. <br></br>
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
        /// <param name="num_channels">The number of input's channels (C).</param>
        /// <param name="momentum">Small batch size (0.9 - 0.99), Big batch size (0.6 - 0.85). Best momentum value is <b>m</b> where <b>m = batch.size / dataset.size</b></param>
        public BatchNorm2D(int num_channels, float momentum = 0.9f)
        {
            if (num_channels < 1)
                throw new ArgumentException($"BatchNorm2D layer cannot have num_channels < 1. (received: {num_channels})");

            this.num_features = num_channels;
            this.momentum = momentum;

            gamma = Tensor.Ones(num_channels);
            beta = Tensor.Zeros(num_channels);
            gammaGrad = Tensor.Zeros(num_channels);
            betaGrad = Tensor.Zeros(num_channels);

            runningVar = Tensor.Ones(num_channels);
            runningMean = Tensor.Zeros(num_channels);
        }

        public Tensor Predict(Tensor input)
        {
            if (input.Rank < 3)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) must be (B, C, H, W) or (C, H, W).");


            if (input.Size(-3) != num_features)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) channels is not equal to num_features ({num_features})");


            bool isBatched = input.Rank == 4;
            int height = input.Size(-2);
            int width = input.Size(-1);  
            if (isBatched)
            {
                int batch_size = input.Size(0);
                Tensor expanded_mean = Tensor.Zeros(batch_size, num_features, height, width);
                Tensor expanded_std = Tensor.Zeros(batch_size, num_features, height, width);
                Tensor expanded_gamma = Tensor.Zeros(batch_size, num_features, height, width);
                Tensor expanded_beta = Tensor.Zeros(batch_size, num_features, height, width);

                // Maybe parallel will be removed out from here
                Parallel.For(0, batch_size, b =>
                {
                    for (int c = 0; c < num_features; c++)
                    {
                        for (int h = 0; h < height; h++)
                        {
                            for (int w = 0; w < width; w++)
                            {
                                expanded_mean[b, c, h, w] = runningMean[c];
                                expanded_std[b, c, h, w] = MathF.Sqrt(runningVar[c] + Utils.EPSILON);
                                expanded_gamma[b, c, h, w] = gamma[c];
                                expanded_beta[b, c, h, w] = beta[c];
                            }
                        }
                    }
                });
                var input_centered = (input - expanded_mean) / expanded_std;
                var output = expanded_gamma * input_centered + expanded_beta;

                return output;
            }
            else
            {
                Tensor expanded_mean = Tensor.Zeros(num_features, height, width);
                Tensor expanded_std = Tensor.Zeros(num_features, height, width);
                Tensor expanded_gamma = Tensor.Zeros(num_features, height, width);
                Tensor expanded_beta = Tensor.Zeros(num_features, height, width);

                for (int c = 0; c < num_features; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            expanded_mean[c, h, w] = runningMean[c];
                            expanded_std[c, h, w] = MathF.Sqrt(runningVar[c] + Utils.EPSILON);
                            expanded_gamma[c, h, w] = gamma[c];
                            expanded_beta[c, h, w] = beta[c];
                        }
                    }
                }

                var input_centered = (input - expanded_mean) / expanded_std;
                var output = expanded_gamma * input_centered + expanded_beta;

                return output;
            }

        }
        public Tensor Forward(Tensor input)
        {
            if (input.Size(-3) != num_features)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) channels is not equal to num_features ({num_features})");

            if (input.Rank != 4)
                throw new InputException($"On training, input shape must be (B, C, H, W) exclusively, where B > 1 (received input shape: {input.Shape.ToCommaSeparatedString()}). For inference, use Predict method instead.");

            if (input.Size(0) <= 1)
                throw new InputException($"On training, the batch size must be greater than 1 (received input shape: {input.Shape.ToCommaSeparatedString()}). For inference, use Predict method instead.");

            int batch_size = input.Size(0);
            int height = input.Size(-2);
            int width = input.Size(-1);

            // When training (only on mini-batch training), we cache the values for backprop also
            var mean = input.Mean(0,true).Mean(2, true).Mean(3, true); // mini-batch means      [1, channels, 1, 1]
            var variance_biased = input.Var(0, 0, keepDim: true).Var(2, 0, keepDim: true).Var(3, 0, keepDim: true); // mini-batch variances  [1, channels, 1, 1]

            Tensor expanded_mean = Tensor.Zeros(batch_size, num_features, height, width);
            Tensor expanded_std = Tensor.Zeros(batch_size, num_features, height, width);
            for (int b = 0; b < batch_size; b++)
            {
                for (int c = 0; c < num_features; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            expanded_mean[b, c, h, w] = mean[0, c, 0, 0];
                            expanded_std[b, c, h, w] = MathF.Sqrt(variance_biased[0, c, 0, 0] + Utils.EPSILON);
                        }
                    }
                }
            }
            // normalize and cache
            xCentered = input - expanded_mean;
            std = expanded_std;
            xHat = xCentered / expanded_std;

            // compute running mean and var
            var variance_unbiased = input.Var(0, 1).Var(2, 1).Var(3, 1);
            runningMean = runningMean * momentum + mean.Reshape(num_features) * (1f - momentum);
            runningVar = runningVar * momentum + variance_unbiased * (1f - momentum);

            // scale and shift then expand 'em
            Tensor expanded_gamma = Tensor.Zeros(batch_size, num_features, height, width);
            Tensor expanded_beta = Tensor.Zeros(batch_size, num_features, height, width);
            for (int b = 0; b < batch_size; b++)
            {
                for (int c = 0; c < num_features; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            expanded_gamma[b, c, h, w] = gamma[c];
                            expanded_beta[b, c, h, w] = beta[c];
                        }
                    }
                }
            }

            Tensor y = expanded_gamma * xHat + expanded_beta;
            return y;
        }
        public Tensor Backward(Tensor dLdY)
        {
            int m = dLdY.Size(0);

            // differentiation on https://arxiv.org/pdf/1502.03167.pdf page 4
            Tensor expanded_gamma = Tensor.Zeros(m, num_features, dLdY.Size(-2), dLdY.Size(-1));

            var dLdxHat = dLdY * expanded_gamma; // [batch, C, H, W]

            var dLdVarB = Tensor.Sum(
                         dLdxHat * xCentered * (-1f / 2f) * (std.Square() + Utils.EPSILON).Pow(-3f / 2f),
                         axis: 0,
                         keepDim: true).Expand(0, m);
            var dLdMuB = Tensor.Sum(
                         dLdxHat * -1f / std + dLdVarB * -2f * xCentered / m,
                         axis: 0,
                         keepDim: true).Expand(0, m);

            var dLdX = dLdxHat * 1f / std + dLdVarB * 2f * xCentered / m + dLdMuB * (1f / m);


            var dLdGamma = Tensor.Sum(dLdY * xHat, 0).Mean(-1).Mean(-1);
            var dLdBeta = Tensor.Sum(dLdY, 0).Mean(-1).Mean(-1);

            Tensor.CopyTo(gammaGrad + dLdGamma, gammaGrad);
            Tensor.CopyTo(betaGrad + dLdBeta, betaGrad);

            return dLdX;
        }

        public object Clone()
        {
            BatchNorm2D bnclone = new BatchNorm2D(num_features, momentum);
            bnclone.gamma = (Tensor)gamma.Clone();
            bnclone.beta = (Tensor)beta.Clone();
            bnclone.gammaGrad = (Tensor)gammaGrad.Clone();
            bnclone.betaGrad = (Tensor)betaGrad.Clone();
            bnclone.runningMean = (Tensor)runningMean.Clone();
            bnclone.runningVar = (Tensor)runningVar.Clone();
            return bnclone;
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