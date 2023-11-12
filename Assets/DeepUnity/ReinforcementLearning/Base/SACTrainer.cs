namespace DeepUnity
{
    // https://medium.com/intro-to-artificial-intelligence/soft-actor-critic-reinforcement-learning-algorithm-1934a2c3087f 
    // https://spinningup.openai.com/en/latest/algorithms/sac.html

    public class SACTrainer : DeepUnityTrainer
    {

        private void Train()
        {
            for (int epoch_index = 0; epoch_index < hp.numEpoch; epoch_index++)
            {
                // Sample a batch of transitions
                Tensor[] states = new Tensor[hp.batchSize];
                Tensor[] advantages = new Tensor[hp.batchSize];

            }
        }
        private void UpdateValueNetwork(Tensor states)
        {
            // Update Value function Jv(φ) = 1/2 * ((Vφ(s) - E[Q(s|~a) - logπθ(~a|s)])^2
            Tensor actions;
            Tensor probs;
            model.ContinuousPredict(states, false, out actions, out probs);

            Tensor q1_values = model.Q1Forward(states, actions);
            Tensor q2_values = model.Q2Forward(states, actions);
            Tensor critic_values = Tensor.Minimum(q1_values, q2_values);

            Tensor value_targets = critic_values - probs.Log();
            Tensor values = model.vNetwork.Forward(states);
            Loss ValueLoss = Loss.MSE(values, value_targets);

            model.vOptimizer.ZeroGrad();
            model.vNetwork.Backward(ValueLoss.Derivative * 0.5f);
            model.vOptimizer.ClipGradNorm(hp.gradClipNorm);
            model.vOptimizer.Step();
        }
        private void UpdateQNetworks(Tensor states, Tensor actions, Tensor q_targets)
        {
            // Update Q functions  /// Jq(θ)=E[1/2 * (Qθ(s|~a) - Qhat(s|~a))^2] where Qhat = r(s|a) + γE[V(st+1)]            
            Tensor q1_values = model.Q1Forward(states, actions);
            Tensor q2_values = model.Q2Forward(states, actions);

            Loss q1Loss = Loss.MSE(q1_values, q_targets);
            Loss q2Loss = Loss.MSE(q2_values, q_targets);

            model.q1Optimizer.ZeroGrad();
            model.q2Optimizer.ZeroGrad();
            model.q1Network.Backward(q1Loss.Derivative * 0.5f);
            model.q2Network.Backward(q2Loss.Derivative * 0.5f);
            model.q1Optimizer.ClipGradNorm(hp.gradClipNorm);
            model.q2Optimizer.ClipGradNorm(hp.gradClipNorm);
            model.q1Optimizer.Step();
            model.q2Optimizer.Step();
        }


        private void UpdatePolicyNetwork(Tensor states)
        {
            Tensor actions;
            model.ContinuousPredict(states, true, out actions, out _);
            Tensor mu, sigma;
            model.ContinuousForward(states, out mu, out sigma);
            Tensor log_probs = Tensor.LogProbability(actions, mu, sigma);

            Tensor q1_values = model.Q1Forward(states, actions);
            Tensor q2_values = model.Q2Forward(states, actions);
            Tensor critic_values = Tensor.Minimum(q1_values, q2_values);

            // This is the actor loss, we need to calculate the derivative of the loss func wrt mu and sigma
            Tensor actor_loss = log_probs - critic_values;



            Tensor dL_dMu = null;
            

            model.muOptimizer.ZeroGrad();
            model.muNetwork.Backward(dL_dMu);
            model.muOptimizer.ClipGradNorm(hp.gradClipNorm);
            model.muOptimizer.Step();


            Tensor dL_dSigma = null;


            model.sigmaOptimizer.ZeroGrad();
            model.sigmaNetwork.Backward(dL_dSigma);
            model.sigmaOptimizer.ClipGradNorm(hp.gradClipNorm);
            model.sigmaOptimizer.Step();


        }


    }
}



