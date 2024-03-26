using DeepUnity.Activations;
using System;

namespace DeepUnity.Modules
{
    /// <summary>
    /// Whenever a new Serializable imodule is created,
    /// write the codde for the Wrapper (declare var, complete wrap and unwrap methods)
    /// I made this because Unity JsonUtility doesn't support inheritance, and i don't wanna use Newtonsoft.Json to keep the framework as simple to integrate as possible
    /// </summary>
    public interface IModule
    {
        public Tensor Predict(Tensor input);
        public Tensor Forward(Tensor input);
        public Tensor Backward(Tensor loss);
        public object Clone();
    }
    [Serializable]
    public class IModuleWrapper
    {
        public string name;

        // Learnable modules
        public Dense dense = null;
        public BatchNorm batchnorm1d = null;
        public LayerNorm layernorm = null;
        public Conv2D conv2d = null;
        public PReLU prelu = null;
        public RNNCell rnncell = null;
        public Attention attention = null;
        public MultiheadAttention multiheadattention = null;
        public LazyDense lazydense = null;
        public LazyBatchNorm lazybatchnorm1d = null;

        // Other modules
        public Dropout dropout = null;
        public Flatten flatten = null;
        public Reshape reshape = null;
        public AvgPool2D avgpool2d = null;
        public MaxPool2D maxpool2d = null;
        public MaxPool1D maxpool1d = null;

        // Activations modules
        public ReLU relu = null;
        public Tanh tanh = null;
        public Softmax softmax = null;
        public LeakyReLU leakyrelu = null;
        public Sigmoid sigmoid = null;
        public Softplus softplus = null;
        public Mish mish = null;
        public ELU elu = null;
        public Threshold threshold = null;
        public HardTanh hardtanh = null;
        public Exponential exponential = null;
        public GELU gelu = null;
        public Sparsemax sparsemax = null;
        public LogSoftmax logsoftmax = null;
        public RReLU rrelu = null;
        public SELU selu = null;

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
                maxpool2d = maxpool2dModule;
            }
            else if(module is AvgPool2D avgpool2dModule)
            {
                avgpool2d = avgpool2dModule;
            }
            else if (module is HardTanh hardtanhModule)
            {
                hardtanh = hardtanhModule;
            }
            else if (module is Exponential expModule)
            {
                exponential = expModule;
            }
            else if (module is GELU geluModule)
            {
                gelu = geluModule;
            }
            else if (module is PReLU preluModule)
            {
                prelu = preluModule;
            }
            else if (module is RNNCell rnncellModule)
            {
                rnncell = rnncellModule;
            }
            else if (module is Attention attentionModule)
            {
                attention = attentionModule;
            }
            else if(module is Sparsemax sparsemaxModule)
            {
                sparsemax = sparsemaxModule;
            }
            else if(module is MultiheadAttention multiheadattentionModule)
            {
                multiheadattention = multiheadattentionModule;
            }
            else if(module is LogSoftmax logsoftmaxModule)
            {
                logsoftmax = logsoftmaxModule;
            }
            else if (module is RReLU rreluModule)
            {
                rrelu = rreluModule;
            }
            else if (module is LazyDense lazydenseModule)
            {
                lazydense = lazydenseModule;
            }
            else if (module is LazyBatchNorm lazybatchnorm1dModule)
            {
                lazybatchnorm1d = lazybatchnorm1dModule;
            }
            else if (module is MaxPool1D maxpool1dModule)
            {
                maxpool1d = maxpool1dModule;
            }
            else if (module is SELU seluModule)
            {
                selu = seluModule;
            }
            else
                throw new Exception($"Unhandled module type while wrapping ({module.GetType().Name}).");
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
                module = moduleWrapper.maxpool2d;
            }
            else if (typeof(AvgPool2D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.avgpool2d;
            }
            else if (typeof(HardTanh).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.hardtanh;
            }
            else if (typeof(Exponential).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.exponential;
            }
            else if (typeof(GELU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.gelu;
            }
            else if (typeof(PReLU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.prelu;
            }
            else if (typeof(RNNCell).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.rnncell;
            }
            else if (typeof(Attention).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.attention;
            }
            else if (typeof(Sparsemax).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.sparsemax;
            }
            else if(typeof(MultiheadAttention).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.multiheadattention;
            }
            else if (typeof(LogSoftmax).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.logsoftmax;
            }
            else if (typeof(RReLU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.rrelu;
            }
            else if (typeof(LazyDense).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.lazydense;
            }
            else if (typeof(LazyBatchNorm).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.lazybatchnorm1d;
            }
            else if(typeof(MaxPool1D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.maxpool1d;
            }
            else if (typeof(SELU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.selu;
            }
            else
                throw new Exception($"Unhandled module type while unwrapping ({moduleWrapper.name}).");

            return module;
        }
    }

}
