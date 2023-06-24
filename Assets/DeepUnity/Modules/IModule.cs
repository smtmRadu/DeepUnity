using kbRadu;
using System;
using UnityEngine;

namespace DeepUnity
{
    public interface IModule
    {
        public Tensor Predict(Tensor input);
        public Tensor Forward(Tensor input);
        public Tensor Backward(Tensor loss);
       
    }

    [Serializable]
    public class ModuleWrapper
    {
        public string name;

        [Header("These modules have save-able fields.")]

        // Parameter modules
        public Dense dense;                  
        public BatchNorm batchnorm;    
        public Dropout dropout;
        public LayerNorm layernorm;
        public Conv2D conv2d;

        // Activation modules
        public Linear linear;
        public ReLU relu;
        public TanH tanh;
        public SoftMax softmax;
        public LeakyReLU leakyrelu;
        public Sigmoid sigmoid;
        public SoftPlus softplus;
        public Mish mish;
        public ELU elu;
        public Threshold threshold;
        
        
        

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
            else if(module is LeakyReLU leakyreluModule)
            {
                leakyrelu = leakyreluModule;
            }
            else if(module is Mish mishModule)
            {
                mish = mishModule;
            }
            else if(module is Sigmoid sigmoidModule)
            {
                sigmoid = sigmoidModule;
            }
            else if (module is SoftPlus softplusModule)
            {
                softplus = softplusModule;
            }
            else if(module is Conv2D conv2dModule)
            {
                conv2d = conv2dModule;
            }
            else if(module is ELU eluModule)
            {
                elu = eluModule;
            }
            else if(module is Threshold thresholdModule)
            {
                threshold = thresholdModule;
            }
            else
                throw new Exception("Unhandled module type while wrapping.");
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
            else if(typeof(LeakyReLU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.leakyrelu;
            }
            else if (typeof(Sigmoid).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.sigmoid;
            }
            else if (typeof(Mish).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.mish;
            }
            else if (typeof(SoftPlus).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.softplus;
            }
            else if(typeof(Conv2D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.conv2d;
            }
            else if (typeof(ELU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.elu;
            }
            else if (typeof(Threshold).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.threshold;
            }
            else
                throw new Exception("Unhandled module type while unwrapping.");

            return module;
        }
    }

}
