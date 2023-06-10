using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class BatchNorm : IModule, IParameters
    {
        private Tensor x_norm_Cache { get; set; }
        private Tensor x_hat_Cache { get; set; }
        private Tensor std_Cache { get; set; }


        [SerializeField] public float momentum;
        [SerializeField] public float epsilon;

        // Learnable parameters
        [SerializeField] public Tensor gamma;
        [SerializeField] public Tensor beta;

        [NonSerialized] public Tensor grad_Gamma;
        [NonSerialized] public Tensor grad_Beta;

        [SerializeField] public Tensor runningMean;
        [SerializeField] public Tensor runningVar;

        
        public BatchNorm(int num_features, float momentum = 0.1f, float eps = 1e-5f)
        {
            gamma = Tensor.Ones(num_features);
            beta = Tensor.Zeros(num_features);

            runningMean = Tensor.Zeros(num_features);
            runningVar = Tensor.Zeros(num_features);

            grad_Gamma = Tensor.Zeros(num_features);
            grad_Beta = Tensor.Zeros(num_features);

            this.momentum = momentum;
            this.epsilon = eps;
        }
        
        public Tensor Forward(Tensor input)
        {
            bool train = input.Shape[1] > 1 ? true : false;

            if(train)
            {
                int features = input.Shape[0];
                // When training (only on mini-batch training), we cache the values for backprop also
                var mu_B = Tensor.Mean(input, 0); // mini-batch means [1, batch]
                var var_B = Tensor.Var(input, 0); // mini-batch variances [1, batch]

                // input [features, batch]  - mu_B [1, batch] -> need expand on axis 0
                var x_norm = input - Tensor.Expand(mu_B, 0, features);
                var std = Tensor.Sqrt(Tensor.Expand(var_B, 0, features) + epsilon);
                var x_hat = x_norm / std;
                var y = gamma * x_hat + beta;

                // Cache everything
                x_norm_Cache = x_norm;
                x_hat_Cache = x_hat;
                std_Cache = std;

                // compute running mean and var
                runningMean = runningMean * momentum + mu_B * (1f - momentum);
                runningVar = runningVar * momentum + var_B * (1f - momentum);

                return y;
            }
            else
            {
                // When using the network, we just normalize the input and forward it to the next module

                var xNorm = (input - runningMean) / Tensor.Sqrt(runningVar + epsilon);
                var y = gamma * xNorm + beta;

                return y;
            }           
        }
        public Tensor Backward(Tensor loss)
        {
            int m = loss.Shape[1];

            // algorithm from https://towardsdatascience.com/implementing-batch-normalization-in-python-a044b0369567
            // var dGamma = Tensor.Sum(loss * x_norm_Cache, 0);
            // var dBeta = Tensor.Sum(loss, 0);
            // 
            // var dx_norm = loss * gamma;
            // var dx_centered = dx_norm / std_Cache;
            // 
            // var dMean = -(Tensor.Sum(dx_centered, 0) +
            //               2f / m *
            //               Tensor.Sum(x_hat_Cache, 0));
            // var dStd = Tensor.Sum(dx_norm * x_hat_Cache * -Tensor.Pow(std_Cache, -2f), 1);
            // var dVar = dStd / 2 / std_Cache;
            // 
            // var dx = dx_centered + (dMean + dVar * 2 * x_hat_Cache) / m;

            //return dx;

            // paper algorithm https://arxiv.org/pdf/1502.03167.pdf

            var dLdxHat = loss * gamma;

            var dLdVar = Tensor.Sum(dLdxHat * x_norm_Cache * (-1f / 2f) * Tensor.Pow(std_Cache + epsilon, -3f / 2f), 0);
            var dLdMu = Tensor.Sum(dLdxHat * -1f / Tensor.Sum(std_Cache + epsilon, 0), 0) +
                        dLdVar * Tensor.Sum(-2f * x_norm_Cache, 0) / m;

            var dLdx = dLdxHat * 1f / (Tensor.Sqrt(std_Cache + epsilon)) +
                       dLdVar * 2f * (x_norm_Cache) / m +
                       dLdMu * (1f / m);

            
            var dLdGamma = Tensor.Sum(loss * x_hat_Cache, 0);
            var dLdBeta = Tensor.Sum(loss, 0);

            grad_Gamma += dLdGamma / m;
            grad_Beta += dLdBeta / m;

            // returning the derivative of the loss wrt inputs.
            return dLdx;
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

        /*   Improving BN networks
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