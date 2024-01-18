using System;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// https://openreview.net/pdf?id=r1etN1rtPB check appendix A 2
    /// </summary>
    [Serializable]
    public class RewardsNormalizer
    {
        [SerializeField] private int step;
        [SerializeField] private float R;
        [SerializeField] private float mean;
        [SerializeField] private float m2;
        [SerializeField] readonly float discount = 0.99f;
        [SerializeField] readonly float clip = 5f;
        public RewardsNormalizer(float gamma = 0.99f, float clip = 5f)
        {
            R = 0f;
            step = 0;
            mean = 0f;
            m2 = 0f;
            this.discount = gamma;  
            this.clip = clip;
        }


        public float ScaleReward(float r_t)
        {
            R = discount * R + r_t;

            Update(R); // In paper might be a mistake since it updates with R
            float std = MathF.Sqrt(m2 / (step - 1) + 1e-10f);

            return Math.Clamp(r_t / std, -clip, clip);
        }
        private void Update(float x)
        {
            step++;
            float delta1 = x - mean;
            mean += delta1 / step;
            float delta2 = x - mean;
            m2 += delta1 * delta2;
        }
    }
}

