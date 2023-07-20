using System;

namespace DeepUnity
{
    public interface IModuleRNN
    {
        public Tensor Backward(Tensor loss);
    }

    [Serializable]
    public class IModuleRNNWrapper
    {
        public string name;

        public RNNCell rnncell;
        public Dropout dropout;
        public LayerNorm layernorm;

        private IModuleRNNWrapper(IModuleRNN module)
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
                throw new Exception("Unhandled rnn module type while wrapping.");
        }

        public static IModuleRNNWrapper Wrap(IModuleRNN module)
        {
            return new IModuleRNNWrapper(module);
        }
        public static IModuleRNN Unwrap(IModuleRNNWrapper moduleWrapper)
        {
            IModuleRNN module = null;

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

