using System;
using UnityEngine;

namespace DeepUnity
{
    public interface IModule
    {
        public Tensor Predict(Tensor input);
        public Tensor Forward(Tensor input);
        public Tensor Backward(Tensor loss);
        public object Clone();
    }

  /*  public interface IModuleGPU
    {
        public TensorGPU Predict(TensorGPU input);
        public TensorGPU Forward(TensorGPU input);
        public TensorGPU Backward(TensorGPU loss);
        public object Clone();
    }*/


    [Serializable]
    public class IModuleWrapper
    {
        public string name;

        [Header("These modules have save-able properties.")]

        // Learnable modules
        public Dense dense;                  
        public BatchNorm batchnorm1d;         
        public LayerNorm layernorm;
        public Conv2D conv2d;
        public PReLU prelu;

        // Other modules
        public Dropout dropout;
        public Flatten flatten;
        public Reshape reshape;
        public AvgPool2D avgPool2d;
        public MaxPool2D maxPool2d;

        // Activations modules
        public ReLU relu;
        public Tanh tanh;
        public Softmax softmax;
        public LeakyReLU leakyrelu;
        public Sigmoid sigmoid;
        public Softplus softplus;
        public Mish mish;
        public ELU elu;
        public Threshold threshold;
        public HardTanh hardtanh;
        public Exp exp;
        public GELU gelu;
        
        
        

        private IModuleWrapper(IModule module)
        {
            name = module.GetType().Name;

            if (module is Dense denseModule)
            {
                dense = denseModule;
            }
            else if (module is BatchNorm batchnormModule)
            {
                batchnorm1d = batchnormModule;
            }
            else if (module is LayerNorm layernormModule)
            {
                layernorm = layernormModule;
            }
            else if (module is ReLU reluModule)
            {
                relu = reluModule;
            }
            else if (module is Tanh tanhModule)
            {
                tanh = tanhModule;
            }
            else if (module is Dropout dropoutModule)
            {
                dropout = dropoutModule;
            }
            else if (module is Softmax softmaxModule)
            {
                softmax = softmaxModule;
            }
            else if (module is LeakyReLU leakyreluModule)
            {
                leakyrelu = leakyreluModule;
            }
            else if (module is Mish mishModule)
            {
                mish = mishModule;
            }
            else if (module is Sigmoid sigmoidModule)
            {
                sigmoid = sigmoidModule;
            }
            else if (module is Softplus softplusModule)
            {
                softplus = softplusModule;
            }
            else if (module is Conv2D conv2dModule)
            {
                conv2d = conv2dModule;
            }
            else if (module is ELU eluModule)
            {
                elu = eluModule;
            }
            else if (module is Threshold thresholdModule)
            {
                threshold = thresholdModule;
            }
            else if (module is Reshape reshapeModule)
            {
                reshape = reshapeModule;
            }
            else if (module is Flatten flattenModule)
            {
                flatten = flattenModule;
            }
            else if (module is MaxPool2D maxpool2dModule)
            {
                maxPool2d = maxpool2dModule;
            }
            else if(module is AvgPool2D avgpool2dModule)
            {
                avgPool2d = avgpool2dModule;
            }
            else if (module is HardTanh hardtanhModule)
            {
                hardtanh = hardtanhModule;
            }
            else if (module is Exp expModule)
            {
                exp = expModule;
            }
            else if (module is GELU geluModule)
            {
                gelu = geluModule;
            }
            else if (module is PReLU preluModule)
            {
                prelu = preluModule;
            }
            else
                throw new Exception("Unhandled module type while wrapping.");
        }

        public static IModuleWrapper Wrap(IModule module)
        {
            return new IModuleWrapper(module);
        }
        public static IModule Unwrap(IModuleWrapper moduleWrapper)
        {
            IModule module = null;

            if (typeof(Dense).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.dense;
            }
            else if (typeof(BatchNorm).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.batchnorm1d;
            }
            else if (typeof(LayerNorm).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.layernorm;
            }
            else if (typeof(ReLU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.relu;
            }
            else if(typeof(Tanh).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.tanh;
            }
            else if(typeof(Dropout).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.dropout;
            }
            else if(typeof(Softmax).Name.Equals(moduleWrapper.name))
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
            else if (typeof(Softplus).Name.Equals(moduleWrapper.name))
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
            else if (typeof(Reshape).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.reshape;
            }
            else if (typeof(Flatten).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.flatten;
            }
            else if (typeof(MaxPool2D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.maxPool2d;
            }
            else if (typeof(AvgPool2D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.avgPool2d;
            }
            else if (typeof(HardTanh).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.hardtanh;
            }
            else if (typeof(Exp).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.exp;
            }
            else if (typeof(GELU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.gelu;
            }
            else if (typeof(PReLU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.prelu;
            }
            else
                throw new Exception("Unhandled module type while unwrapping.");

            return module;
        }
    }

}
