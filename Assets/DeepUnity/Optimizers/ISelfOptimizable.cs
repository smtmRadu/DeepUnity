using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// An interface for Learnable modules that have more tensor parameters than gamma and beta.
    /// </summary>
    public interface ISelfOptimizable
    {
        public void SelfOptimise(float lr);
    }
}


