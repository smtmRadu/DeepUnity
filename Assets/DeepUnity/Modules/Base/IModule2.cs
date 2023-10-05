namespace DeepUnity
{
    /// <summary>
    /// This modules are receiving sequencial input.
    /// </summary>
    public interface IModule2
    {
        public Tensor Backward(Tensor loss);
    }

    [System.Serializable]
    public class IModule2Wrapper
    {
        public string name;

        public RNNCell rnncell;
        public Dropout dropout;
        public LayerNorm layernorm;

        private IModule2Wrapper(IModule2 module)
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

        public static IModule2Wrapper Wrap(IModule2 module)
        {
            return new IModule2Wrapper(module);
        }
        public static IModule2 Unwrap(IModule2Wrapper moduleWrapper)
        {
            IModule2 module = null;

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

