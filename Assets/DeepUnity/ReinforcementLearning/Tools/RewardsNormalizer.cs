using System;
using Unity.VisualScripting;
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
        [SerializeField] public float gamma = 0.99f;
        [SerializeField] private RunningNormalizer returnsNormalizer = new RunningNormalizer(1, 1e-8f); // this is actually the one that must be serialized and hold info

        // Without NonSerialized, it seems like Unity Serializes them. So keep them like this.
        [NonSerialized] private Tensor returns = null; // This must reset at every training session.
        [NonSerialized] private int last_recorded_timestep = -1; // At the beggining of each training session (even if is a continuation of a previous one) this starts like this.
        
        public RewardsNormalizer(float gamma = 0.99f, float eps = 1e-8f)
        { 
            this.gamma = gamma;
            this.returns = null;
            this.returnsNormalizer = new RunningNormalizer(1, eps); 
        }

        /// <param name="reward">Reward of agent `i`.</param>
        /// <param name="done">Done of agent `i`.</param>
        /// <param name="ag_env_idx">Index `i` of agent in Trainer pool.</param>
        /// <param name="shared_fixed_frame_count">How many fixed frames passed from the beggining of the training. This is common timestep for all agents.</param>
        /// <returns></returns>
        public float Normalize(float reward, float done, int ag_env_idx, int shared_fixed_frame_count)
        {
            if (returns == null)
            {
                returns = Tensor.Zeros(DeepUnityTrainer.Instance.parallelAgents.Count, 1);
            }
            
            if(last_recorded_timestep != shared_fixed_frame_count)
            {
                if (last_recorded_timestep != -1) // Do not update when the training just started.
                    returnsNormalizer.Update(returns);

                last_recorded_timestep = shared_fixed_frame_count;
            }

            returns[ag_env_idx] = returns[ag_env_idx] * gamma * (1f - done) + reward;

            if (returnsNormalizer.Step == 0) // safe if step is 0, it just returns identity
                return reward;
            else
                return reward / MathF.Sqrt(returnsNormalizer.M2[0] / (returnsNormalizer.Step - 1) + returnsNormalizer.Epsilon);

        }

    }
}

