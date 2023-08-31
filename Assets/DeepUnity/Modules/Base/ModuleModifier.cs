namespace DeepUnity
{
    /// <summary>
    /// A class for modification of <see cref="IModule"/> layers.
    /// </summary>
    public static class ModuleModifier
    {
        /// <summary>
        /// Modifies a <see cref="Dense"/> module inputs and outputs to new values. 
        /// If there are new parameters after the modification, <paramref name="gamma_init"/> and <paramref name="beta_init"/> will be used for their initialization.
        /// </summary>
        /// <returns>The modified <see cref="Dense"/> module.</returns>
        public static Dense Modify(Dense oldDense, int new_in_features, int new_out_features, InitType gamma_init = InitType.Glorot_Uniform, InitType beta_init = InitType.Zeros)
        {
            Dense newDense = new Dense(new_in_features, new_out_features, gamma_init, beta_init, oldDense.device);

            // Copy-paste the old weights and biases over the new module;

            for (int h = 0; h < oldDense.gamma.Size(0) && h < newDense.gamma.Size(0); h++)
            {
                for (int w = 0; w < oldDense.gamma.Size(1) && w < newDense.gamma.Size(1); w++)
                {
                    newDense.gamma[h,w] = oldDense.gamma[h,w];
                }
            }

            for (int w = 0; w < oldDense.beta.Size(0) && w < newDense.beta.Size(0); w++)
            {
                newDense.beta[w] = oldDense.beta[w];
            }

            return newDense;
        }

    }
}


