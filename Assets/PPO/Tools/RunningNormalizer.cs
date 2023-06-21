using DeepUnity;
using System;
using UnityEngine;

namespace kbRadu
{
    [Serializable]
    public class RunningNormalizer
    {
        [SerializeField] readonly float MIN_LIMIT;
        [SerializeField] readonly float MAX_LIMIT;

        [SerializeField] Tensor min;
        [SerializeField] Tensor max;
 	    public RunningNormalizer(int size, float min = -1f, float max = 1f)
        {
            MIN_LIMIT = min;
            MAX_LIMIT = max;

            this.min = Tensor.Fill(min, size);
            this.min = Tensor.Fill(max, size);
        }
        private void Update(Tensor tuples)
        {
            
        }
        private Tensor Normalize(Tensor tuples)
        {
            return null;
        }
    }
}

