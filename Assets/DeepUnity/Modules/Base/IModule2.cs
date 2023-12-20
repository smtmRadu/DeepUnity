namespace DeepUnity
{
    /// <summary>
    /// This modules are receiving sequencial input.
    /// </summary>
    public interface IModule2
    {
        public Tensor Predict(Tensor input, Tensor hidden);
        public Tensor Forward(Tensor input, Tensor hidden);
        public Tensor Backward(Tensor loss);
        public object Clone();
    }

    [System.Serializable]
    public class IModule2Wrapper
    {
        public string name;

        public RNNCell rnncell;


        private IModule2Wrapper(IModule2 module)
        {
            name = module.GetType().Name;

            if (module is RNNCell rnncellModule)
            {
                rnncell = rnncellModule;
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

            return module;
        }
    }
}

