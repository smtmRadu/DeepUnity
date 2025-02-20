using System;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// https://openreview.net/pdf?id=r1etN1rtPB
    /// https://gymnasium.farama.org/v0.29.0/_modules/gymnasium/wrappers/normalize/
    /// </summary>
    [Serializable]
    public class RewardsNormalizer
    {
        [SerializeField] private float gamma = 0.99f;
        [SerializeField] private RunningNormalizer returnsNormalizer = new RunningNormalizer(1, 1e-8f); // this is actually the one that must be serialized and hold info

        private Tensor returns = Tensor.Zeros(1, 1); // this holds along the episodes, no matter the sesion
        private int last_recorded_timestep = -1; // At the beggining of each training session (even if is a continuation of a previous one) this starts like this.
        public RewardsNormalizer(float gamma = 0.99f, float eps = 1e-8f)
        { 
            this.gamma = gamma;
            this.returns = Tensor.Zeros(1);
            this.returnsNormalizer = new RunningNormalizer(1, eps); 
        }

        /// <param name="reward">Reward of agent `i`.</param>
        /// <param name="done">Done of agent `i`.</param>
        /// <param name="ag_env_idx">Index `i` of agent in Trainer pool.</param>
        /// <param name="shared_timestep">Index of global shared step `t` of all agents in the training.</param>
        /// <returns></returns>
        public float Normalize(float reward, float done, int ag_env_idx, int shared_timestep)
        {
            // Adapt to the training dimensionality.
            if(DeepUnityTrainer.Instance.parallelAgents.Count > 1)
            {
                returns = Tensor.Zeros(DeepUnityTrainer.Instance.parallelAgents.Count, 1);
            }
            
            if(last_recorded_timestep != shared_timestep)
            {
                if (last_recorded_timestep != -1) // Do not update when the training just started.
                    returnsNormalizer.Update(returns);

                last_recorded_timestep = shared_timestep;
            }

            returns[ag_env_idx] = returns[ag_env_idx] * gamma * (1f - done) + reward;

            if (returnsNormalizer.Step == 0) // safe if step is 0, it just returns identity
                return reward;
            else
                return reward / (returnsNormalizer.M2[0] / (returnsNormalizer.Step - 1) + returnsNormalizer.Epsilon);

        }

    }
}

