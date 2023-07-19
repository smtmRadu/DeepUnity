using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class BatchNorm1D : Learnable, IModule
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
        /// <b>Placed before or after the non-linear activation function.</b>    <br />
        /// Input: (batch, features)
        /// Output: (batch, features)
        /// </summary>
        /// <param name="momentum">Small batch size (0.9 - 0.99), Big batch size (0.6 - 0.85). Best momentum value is <b>m</b> where <b>m = batch.size / dataset.size</b></param>
        public BatchNorm1D(int num_features, float momentum = 0.9f) : base(Device.CPU)
        {
            this.momentum = momentum;

            gamma = Tensor.Ones(num_features);
            beta = Tensor.Zeros(num_features);

            gammaGrad = Tensor.Zeros(num_features);
            betaGrad = Tensor.Zeros(num_features);

            runningVar = Tensor.Ones(num_features);
            runningMean = Tensor.Zeros(num_features);          
        }

        public Tensor Predict(Tensor input)
        {
            bool isBatched = input.Rank == 2;

            if(isBatched)
            {
                int batch_size = input.Size(0);
                var e_mean = Tensor.Expand(Tensor.Unsqueeze(runningMean, 0), 0, batch_size);
                var e_var = Tensor.Expand(Tensor.Unsqueeze(runningVar, 0), 0, batch_size);
                var e_gamma = Tensor.Expand(Tensor.Unsqueeze(gamma, 0), 0, batch_size);
                var e_beta = Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);

                var input_centered = (input - e_mean) / Tensor.Sqrt(e_var + Utils.EPSILON);
                var output = e_gamma * input_centered + e_beta;

                return output;
            }
            else
            {
                var input_centered = (input - runningMean) / Tensor.Sqrt(runningVar + Utils.EPSILON);
                var output = gamma * input_centered + beta;

                return output;
            }
            
        }
        public Tensor Forward(Tensor input)
        {
            if (input.Rank != 2)
                throw new ArgumentException("Models having BatchNorm layers must be trained using batched input (batch, features)");

            int batch_size = input.Size(0);

            // When training (only on mini-batch training), we cache the values for backprop also
            var mu_B = Tensor.Mean(input, 0); // mini-batch means      [features_mean]
            var var_B = Tensor.Var(input, 0); // mini-batch variances  [features_mean]

            // input [batch, features]  - muB or varB [features] -> need expand on axis 0 by batch

            // normalize and cache
            mu_B.Unsqueeze(0);
            xCentered = input - Tensor.Expand(mu_B, 0, batch_size);
            std = Tensor.Sqrt(var_B + Utils.EPSILON).Unsqueeze(0);
            std = Tensor.Expand(std, 0, batch_size);
            xHat = xCentered / std;

            // scale and shift
            var yB = Tensor.Expand(Tensor.Unsqueeze(gamma, 0), 0, batch_size) * xHat + 
                     Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);



            // compute running mean and var
            mu_B.Squeeze(0);
            runningMean = runningMean * momentum + mu_B * (1f - momentum);
            runningVar = runningVar * momentum + var_B * (1f - momentum);

            return yB;

        }
        public Tensor Backward(Tensor dLdY)
        {
            int m = dLdY.Size(0);

            // paper algorithm https://arxiv.org/pdf/1502.03167.pdf

            var dLdxHat = dLdY * Tensor.Expand(Tensor.Unsqueeze(gamma, 0), 0, m); // [batch, outs]

            var dLdVarB = Tensor.Mean(
                         dLdxHat * xCentered * (-1f / 2f) * Tensor.Pow(std + Utils.EPSILON, -3f / 2f),
                         axis: 0, 
                         keepDim: true).
                         Expand(0, m);

            var dLdMuB = Tensor.Mean(
                         dLdxHat * -1f / (std + Utils.EPSILON) + dLdVarB * -2f * xCentered / m, 
                         axis: 0, 
                         keepDim: true).
                         Expand(0, m);

            var dLdX = dLdxHat * 1f / Tensor.Sqrt(std + Utils.EPSILON) +
                       dLdVarB * 2f * xCentered / m +
                       dLdMuB * (1f / m);


            var dLdGamma = Tensor.Mean(dLdY * xHat, 0);
            var dLdBeta = Tensor.Mean(dLdY, 0);

            gammaGrad += dLdGamma;
            betaGrad += dLdBeta;

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