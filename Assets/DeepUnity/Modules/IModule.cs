using kbRadu;
using System;
using UnityEngine;

namespace DeepUnity
{
    public interface IModule
    {
        public NDArray Predict(NDArray input);
        public NDArray Forward(NDArray input);
        public NDArray Backward(NDArray loss);
       
    }

    [Serializable]
    public class ModuleWrapper
    {
        public string name;

        [Header("Value save modules")]
        public Dense dense;                  
        public BatchNorm batchnorm;
        public LayerNorm layernorm;
        public Dropout dropout;

        // Activation modules
        public Linear linear;
        public ReLU relu;
        public TanH tanh;
        public SoftMax softmax;
        

        private ModuleWrapper(IModule module)
        {
            name = module.GetType().Name;

            if (module is Dense denseModule)
            {
                dense = denseModule;
            }
            else if(module is BatchNorm batchnormModule)
            {
                batchnorm = batchnormModule;
            }
            else if(module is LayerNorm layernormModule)
            {
                layernorm = layernormModule;
            }
            else if (module is ReLU reluModule)
            {
                relu = reluModule;
            }
            else if(module is TanH tanhModule)
            {
                tanh = tanhModule;
            }
            else if(module is Linear linearModule)
            {
                linear = linearModule;
            }
            else if(module is  Dropout dropoutModule)
            {
                dropout = dropoutModule;
            }
            else if(module is SoftMax softmaxModule)
            {
                softmax = softmaxModule;
            }
            else
               throw new Exception("Unhandled module type on wrapping.");
        }

        public static ModuleWrapper Wrap(IModule module)
        {
            return new ModuleWrapper(module);
        }
        public static IModule Unwrap(ModuleWrapper moduleWrapper)
        {
            IModule module = null;

            if (typeof(Dense).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.dense;
            }
            else if (typeof(BatchNorm).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.batchnorm;
            }
            else if (typeof(LayerNorm).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.layernorm;
            }
            else if (typeof(ReLU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.relu;
            }
            else if(typeof(TanH).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.tanh;
            }
            else if(typeof(Linear).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.linear;
            }
            else if(typeof(Dropout).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.dropout;
            }
            else if(typeof(SoftMax).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.softmax;
            }
            else
                throw new Exception("Unhandled module type on unwrapping.");

            return module;
        }
    }

}
