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
       

        [SerializeField] public float momentum;
        [SerializeField] public float epsilon;

        // Learnable parameters
        [SerializeField] private Tensor runningMean;
        [SerializeField] private Tensor runningVar;


        /// <summary>
        /// </summary>
        /// <param name="momentum">Small batch size (0.9 - 0.99), Big batch size (0.6 - 0.85). Best momentum value is 'm' where m * dataset_size = batch_size</param>
        public BatchNorm(int num_features, float momentum = 0.9f, float eps = 1e-5f)
        {
            this.momentum = momentum;
            this.epsilon = eps;

            gamma = Tensor.Ones(num_features);
            beta = Tensor.Zeros(num_features);

            gradGamma = Tensor.Zeros(num_features);
            gradBeta = Tensor.Zeros(num_features);

            runningVar = Tensor.Ones(num_features);
            runningMean = Tensor.Zeros(num_features);

           
        }

        public Tensor Predict(Tensor input)
        {
            var input_centered = (input - runningMean) / Tensor.Sqrt(runningVar + epsilon);
            var output = gamma * input_centered + beta;

            return output;
        }
        public Tensor Forward(Tensor input)
        {
            int batch = input.Shape.Height;

            // When training (only on mini-batch training), we cache the values for backprop also
            var mu_B = Tensor.Mean(input, TDim.height); // mini-batch means      [batch, 1]
            var var_B = Tensor.Var(input, TDim.height); // mini-batch variances  [batch, 1]

            // input [batch, features]  - muB or varB [features] -> need expand on axis 0 by batch

            // normalize and cache
            xCentered = input - Tensor.Expand(mu_B, TDim.height, batch);
            std = Tensor.Expand(Tensor.Sqrt(var_B + epsilon), TDim.height, batch);
            xHat = xCentered / std;

            // scale and shift
            var yB = Tensor.Expand(gamma, TDim.height, batch) * xHat + Tensor.Expand(beta, TDim.height, batch);

            

            // compute running mean and var
            runningMean = runningMean * momentum + mu_B * (1f - momentum);
            runningVar = runningVar * momentum + var_B * (1f - momentum);

            return yB;

        }
        public Tensor Backward(Tensor dLdY)
        {
            int m = dLdY.Shape.Height;

            // paper algorithm https://arxiv.org/pdf/1502.03167.pdf

            var dLdxHat = dLdY * Tensor.Expand(gamma, TDim.height, m); // [batch, outs]

            var dLdVarB = Tensor.Mean(dLdxHat * xCentered * (-1f / 2f) *
                         Tensor.Pow(std + epsilon, -3f / 2f), TDim.height, true);

            var dLdMuB = Tensor.Mean(
                         dLdxHat * -1f / (std + epsilon) +
                         dLdVarB * -2f * xCentered / m, 
                         TDim.height, true);

            var dLdX = dLdxHat * 1f / Tensor.Sqrt(std + epsilon) +
                       dLdVarB * 2f * xCentered / m +
                       dLdMuB * (1f / m);


            var dLdGamma = Tensor.Mean(dLdY * xHat, TDim.height);
            var dLdBeta = Tensor.Mean(dLdY, TDim.height);

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