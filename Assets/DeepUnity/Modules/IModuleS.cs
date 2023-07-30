namespace DeepUnity
{
    /// <summary>
    /// This modules are receiving sequencial input.
    /// </summary>
    public interface IModuleS
    {
        public Tensor Backward(Tensor loss);
    }

    [System.Serializable]
    public class IModuleSWrapper
    {
        public string name;

        public RNNCell rnncell;
        public Dropout dropout;
        public LayerNorm layernorm;

        private IModuleSWrapper(IModuleS module)
        {
            name = module.GetType().Name;

            if (module is RNNCell rnncellModule)
            {
                rnncell = rnncellModule;
            }
            else if (module is Dropout dropoutModule)
            {
                dropout = dropoutModule;
            }
            else if (module is LayerNorm layerNormModule)
            {
                layernorm = layerNormModule;
            }
            else
                throw new System.Exception("Unhandled rnn module type while wrapping.");
        }

        public static IModuleSWrapper Wrap(IModuleS module)
        {
            return new IModuleSWrapper(module);
        }
        public static IModuleS Unwrap(IModuleSWrapper moduleWrapper)
        {
            IModuleS module = null;

            if (typeof(RNNCell).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.rnncell;
            }
            else if (typeof(LayerNorm).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.layernorm;
            }
            else if(typeof(Dropout).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.dropout;
            }

            return module;
        }
    }
}

