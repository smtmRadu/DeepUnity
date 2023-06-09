using System;
using Unity.VisualScripting;
using Unity.VisualScripting.ReorderableList;
using UnityEngine;

namespace DeepUnity
{
    public interface IModule
    {
        public Tensor Forward(Tensor input);
        public Tensor Backward(Tensor loss);
    }

    [Serializable]
    public class ModuleWrapper
    {
        public string name;

        [Space]
        public Dense dense;          
        public Dropout dropout;

        public Linear linear;
        public ReLU relu;
        public TanH tanh;
        public SoftMax softmax;
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
