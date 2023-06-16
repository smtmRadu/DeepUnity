using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class BatchNorm : IModule, IParameters
    {
        private Tensor x_minus_mu_cache { get; set; }
        private Tensor x_hat_Cache { get; set; }
        private Tensor std_cache { get; set; }


        [SerializeField] public float momentum;
        [SerializeField] public float epsilon;

        // Learnable parameters
        [SerializeField] public Tensor runningMean;
        [SerializeField] public Tensor runningVar;

        [SerializeField] public Tensor gamma;
        [SerializeField] public Tensor beta;


        [NonSerialized] public Tensor grad_Gamma;
        [NonSerialized] public Tensor grad_Beta;




        public BatchNorm(int num_features, float momentum = 0.1f, float eps = 1e-5f)
        {
            gamma = Tensor.Ones(num_features);
            beta = Tensor.Zeros(num_features);

            runningVar = Tensor.Ones(num_features);
            runningMean = Tensor.Zeros(num_features);


            grad_Gamma = Tensor.Zeros(num_features);
            grad_Beta = Tensor.Zeros(num_features);

            this.momentum = momentum;
            this.epsilon = eps;
        }

        public Tensor Predict(Tensor input)
        {
            var input_centered = (input - runningMean) / Tensor.Sqrt(runningVar + epsilon);
            var output = gamma * input_centered + beta;

            return output;
        }
        public Tensor Forward(Tensor input)
        {
            int batch = input.Shape.height;

            // When training (only on mini-batch training), we cache the values for backprop also
            var mu_B = Tensor.Mean(input, 0); // mini-batch means      [batch]
            var var_B = Tensor.Var(input, 0); // mini-batch variances  [batch]

            // input [batch, features]  - muB or varB [features] -> need expand on axis 0 by batch

            // normalize
            var x_minus_mu = input - Tensor.Expand(mu_B, -1, batch);
            var sqrt_var = Tensor.Expand(Tensor.Sqrt(var_B + epsilon), -1, batch);
            var x_hat = x_minus_mu / sqrt_var;

            // scale and shift
            var yB = Tensor.Expand(gamma, -1, batch) * x_hat + Tensor.Expand(beta, -1, batch);

            // Cache everything
            x_minus_mu_cache = x_minus_mu;
            x_hat_Cache = x_hat;
            std_cache = sqrt_var;

            // compute running mean and var
            runningMean = runningMean * momentum + mu_B * (1f - momentum);
            runningVar = runningVar * momentum + var_B * (1f - momentum);

            return yB;

        }
        public Tensor Backward(Tensor dLdY)
        {
            int m = dLdY.Shape.height;

            // paper algorithm https://arxiv.org/pdf/1502.03167.pdf

            var dLdxHat = dLdY * Tensor.Expand(gamma, -1, m); 

            var dLdVarB = Tensor.Mean(dLdxHat * x_minus_mu_cache * (-1f / 2f) *
                         Tensor.Pow(std_cache + epsilon, -3f / 2f), 0);

            var dLdMuB = Tensor.Mean(dLdxHat * -1f / std_cache + epsilon +
                        Tensor.Expand(dLdVarB, -1, m) * -2f * x_minus_mu_cache / m, 0);

            var dLdX = dLdxHat * 1f / Tensor.Sqrt(std_cache + epsilon) +
                       Tensor.Expand(dLdVarB, -1, m) * 2f * x_minus_mu_cache / m +
                       Tensor.Expand(dLdMuB, -1, m) * (1f / m);


            var dLdGamma = Tensor.Mean(dLdY * x_hat_Cache, 0);
            var dLdBeta = Tensor.Mean(dLdY, 0);

            grad_Gamma += dLdGamma;
            grad_Beta += dLdBeta;

            return dLdX;
        }

        public void ZeroGrad()
        {
            grad_Gamma.ForEach(x => 0f);
            grad_Beta.ForEach(x => 0f);
        }
        public void ClipGradValue(float clip_value)
        {
            Tensor.Clip(grad_Gamma, -clip_value, clip_value);
            Tensor.Clip(grad_Beta, -clip_value, clip_value);
        }
        public void ClipGradNorm(float max_norm)
        {
            Tensor normG = Tensor.Norm(grad_Gamma, NormType.ManhattanL1);

            if (normG[0] > max_norm)
            {
                float scale = max_norm / normG[0];
                grad_Gamma *= scale;
            }


            Tensor normB = Tensor.Norm(grad_Beta, NormType.ManhattanL1);

            if (normB[0] > max_norm)
            {
                float scale = max_norm / normB[0];
                grad_Beta *= scale;
            }

        }

        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization, and one on them is called when weights.shape.length == 0.
            if (gamma.Shape == null || gamma.Shape.width == 0)
                return;

            int num_features = gamma.Shape.width;

            this.grad_Gamma = Tensor.Zeros(num_features);
            this.grad_Beta = Tensor.Zeros(num_features);
        }
        /*   Improving BN networks (placed before 
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