using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class BatchNorm : Learnable, IModule
    {
        // https://arxiv.org/pdf/1502.03167.pdf

        private Tensor xCentered { get; set; }
        private Tensor std { get; set; }
        private Tensor xHat { get; set; }
       

        [SerializeField] private float momentum;

        // Learnable parameters
        [SerializeField] private Tensor runningMean;
        [SerializeField] private Tensor runningVar;


        /// <summary>
        /// <b>Placed after or before the non-linear activation function.</b>    <br />
        /// </summary>
        /// <param name="momentum">Small batch size (0.9 - 0.99), Big batch size (0.6 - 0.85). Best momentum value is <b>m</b> where <b>m = batch.size / dataset.size</b></param>
        public BatchNorm(int num_features, float momentum = 0.9f)
        {
            this.momentum = momentum;

            gamma = Tensor.Ones(num_features);
            beta = Tensor.Zeros(num_features);

            gradGamma = Tensor.Zeros(num_features);
            gradBeta = Tensor.Zeros(num_features);

            runningVar = Tensor.Ones(num_features);
            runningMean = Tensor.Zeros(num_features);          
        }

        public Tensor Predict(Tensor input)
        {
            int batch_size = input.Height;
            var e_mean = Tensor.Expand(runningMean, Dim.height, batch_size);
            var e_var = Tensor.Expand(runningVar, Dim.height, batch_size);
            var e_gamma = Tensor.Expand(gamma, Dim.height, batch_size);
            var e_beta = Tensor.Expand(beta, Dim.height, batch_size);

            var input_centered = (input - e_mean) / Tensor.Sqrt(e_var + Utils.EPSILON);
            var output = e_gamma * input_centered + e_beta;

            return output;
        }
        public Tensor Forward(Tensor input)
        {
            int batch_size = input.Height;

            // When training (only on mini-batch training), we cache the values for backprop also
            var mu_B = Tensor.Mean(input, Dim.height); // mini-batch means      [batch, 1]
            var var_B = Tensor.Var(input, Dim.height); // mini-batch variances  [batch, 1]

            // input [batch, features]  - muB or varB [features] -> need expand on axis 0 by batch

            // normalize and cache
            xCentered = input - Tensor.Expand(mu_B, Dim.height, batch_size);
            std = Tensor.Expand(Tensor.Sqrt(var_B + Utils.EPSILON), Dim.height, batch_size);
            xHat = xCentered / std;

            // scale and shift
            var yB = Tensor.Expand(gamma, Dim.height, batch_size) * xHat + Tensor.Expand(beta, Dim.height, batch_size);

            

            // compute running mean and var
            runningMean = runningMean * momentum + Tensor.Squeeze(mu_B * (1f - momentum));
            runningVar = runningVar * momentum + Tensor.Squeeze(var_B * (1f - momentum));

            return yB;

        }
        public Tensor Backward(Tensor dLdY)
        {
            int m = dLdY.Height;

            // paper algorithm https://arxiv.org/pdf/1502.03167.pdf

            var dLdxHat = dLdY * Tensor.Expand(gamma, Dim.height, m); // [batch, outs]

            var dLdVarB = Tensor.Mean(dLdxHat * xCentered * (-1f / 2f) *
                         Tensor.Pow(std + Utils.EPSILON, -3f / 2f), Dim.height, true);

            var dLdMuB = Tensor.Mean(
                         dLdxHat * -1f / (std + Utils.EPSILON) +
                         dLdVarB * -2f * xCentered / m, 
                         Dim.height, true);

            var dLdX = dLdxHat * 1f / Tensor.Sqrt(std + Utils.EPSILON) +
                       dLdVarB * 2f * xCentered / m +
                       dLdMuB * (1f / m);


            var dLdGamma = Tensor.Mean(dLdY * xHat, Dim.height);
            var dLdBeta = Tensor.Mean(dLdY, Dim.height);

            dLdGamma = Tensor.Squeeze(dLdGamma);
            dLdBeta = Tensor.Squeeze(dLdBeta);

            gradGamma += dLdGamma;
            gradBeta += dLdBeta;

            return dLdX;
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