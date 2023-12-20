using System;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// https://developers.google.com/machine-learning/data-prep/transform/normalization
    /// </summary>
    public interface INormalizer
    {
        /// <summary>
        /// Normalizes the input value.
        /// </summary>
        /// <param name="input">Tensor (batch_size) or (batch_size, dimension) for batched input.</param>
        /// <returns></returns>
        public Tensor Normalize(Tensor input);
        /// <summary>
        /// Updates the data population.
        /// </summary>
        /// <param name="input">Tensor (batch_size) or (batch_size, dimension) for batched input.</param>
        public void Update(Tensor input);
    }
}



