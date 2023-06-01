using System;
using UnityEngine;

namespace DeepUnity
{
    public interface IModule
    {
        public Tensor InputCache { get; set; }
        public Tensor Forward(Tensor input);
        public Tensor Backward(Tensor loss);
    }

    [Serializable]
    public class ModuleWrapper
    {
        public string name;
        [Space]
        public Dense dense;
        public ReLU relu;
        //...

        public ModuleWrapper(IModule module)
        {
            name = module.GetType().Name;
            if (module is Dense denseModule)
            {
                dense = denseModule;
            }
            else if (module is ReLU reluModule)
            {
                relu = reluModule;
            }
            else
                throw new Exception("Unhandled module type on wrapping.");
        }
        public static IModule Get(ModuleWrapper moduleWrapper)
        {
            IModule module = null;

            if (typeof(Dense).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.dense;
            }
            else if (typeof(ReLU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.relu;
            }
            else
                throw new Exception("Unhandled module type on unwrapping.");

            return module;
        }
    }

}
