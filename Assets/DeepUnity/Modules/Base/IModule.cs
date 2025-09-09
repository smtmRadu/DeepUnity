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

        // Learnable
        public Dense dense = null;      
        public DenseGPU densegpu = null;
        public RNNCell rnncell = null;   
        public GroupedQueryAttention groupedqueryattention = null;
        public LayerNorm layernorm = null;
        public RMSNorm rmsnorm = null;
        public BatchNorm batchnorm = null;
        public Embedding embedding = null;
        public GatedLinearUnit gatedlinearunit = null;

        // Lazy 
        public LazyConv2D lazyconv2d = null;
        public LazyDense lazydense = null;
        public LazyBatchNorm1D lazybatchnorm1d = null;
        public LazyBatchNorm2D lazybatchnorm2d = null;

        // Non-Learnable
        public Dropout dropout = null;      
        public Flatten flatten = null;
        public Reshape reshape = null;
        public LastSequence1DElementModule lastsequence1delementmodule = null;
        public ResidualConnection.Fork residualconnectionfork = null;
        public ResidualConnection.Join residualconnectionjoin = null;
        public Squeeze squeeze = null;
        public Unsqueeze unsqueeze = null;
        public Permute permute = null;
        public RotaryPositionalEmbeddings rotarypositionalembeddings = null;

        // 2D
        public BatchNorm2D batchnorm2d = null;        
        public Conv2D conv2d = null;
        public AvgPool2D avgpool2d = null;
        public MaxPool2D maxpool2d = null;    
        public ZeroPad2D zeropad2d = null;
        public MirrorPad2D mirrorpad2d = null;
        public Dropout2D dropout2d = null;

        // 1D   
        public AvgPool1D avgpool1d = null;
        public MaxPool1D maxpool1d = null;     
        public ZeroPad1D zeropad1d = null;
        public MirrorPad1D mirrorpad1d = null;
      
        // Activations
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
        public PReLU prelu = null;
        public Sparsemax sparsemax = null;
        public LogSoftmax logsoftmax = null;
        public RReLU rrelu = null;
        public SELU selu = null;
        public SiLU silu = null;
        public ReLU6 relu6 = null;
        public Rish rish = null;



        public static IModuleWrapper Wrap(IModule module)
        {
            return new IModuleWrapper(module);
        }
        private IModuleWrapper(IModule module)
        {
            name = module.GetType().Name;

            if (module is Dense denseModule)
            {
                dense = denseModule;
            }
            else if (module is BatchNorm batchnorm1dModule)
            {
                batchnorm = batchnorm1dModule;
            }
            else if (module is LayerNorm layernorm1dModule)
            {
                layernorm = layernorm1dModule;
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
            else if(module is Sparsemax sparsemaxModule)
            {
                sparsemax = sparsemaxModule;
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
            else if (module is LazyBatchNorm1D lazybatchnorm1dModule)
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
            else if (module is RMSNorm rmsnorm1dModule)
            {
                rmsnorm = rmsnorm1dModule;
            }
            else if (module is ZeroPad2D zeropad2DModule)
            {
                zeropad2d = zeropad2DModule;
            }
            else if (module is ResidualConnection.Fork resconforkModule)
            {
                residualconnectionfork = resconforkModule;
            }
            else if (module is ResidualConnection.Join resconjoinModule)
            {
                residualconnectionjoin = resconjoinModule;
            }
            else if (module is MirrorPad2D mirrorpad2dModule)
            {
                mirrorpad2d = mirrorpad2dModule;
            }
            else if (module is ZeroPad1D zeropad1dModule)
            {
                zeropad1d = zeropad1dModule;
            }
            else if (module is MirrorPad1D mirrorpad1dModule)
            {
                mirrorpad1d = mirrorpad1dModule;
            }
            else if (module is AvgPool1D avgpool1dModule)
            {
                avgpool1d = avgpool1dModule;
            }
            else if (module is BatchNorm2D batchnorm2dModule)
            {
                batchnorm2d = batchnorm2dModule;
            }
            else if (module is GroupedQueryAttention multiheadattentionModule)
            {
                groupedqueryattention = multiheadattentionModule;
            }
            else if(module is LastSequence1DElementModule lastSequenceElementModule)
            {
                lastsequence1delementmodule = lastSequenceElementModule;
            }
            else if(module is DenseGPU denseGPUModule)
            {
                densegpu = denseGPUModule;
            }
            else if (module is Squeeze squeezeModule)
            {
                squeeze = squeezeModule;
            }
            else if (module is Unsqueeze unsqueezeModule)
            {
                unsqueeze = unsqueezeModule;
            }
            else if (module is Permute permuteModule)
            {
                permute = permuteModule;
            }
            else if (module is SiLU siluModule)
            {
                silu = siluModule;
            }
            else if (module is Dropout2D dropout2dModule)
            {
                dropout2d = dropout2dModule;
            }
            else if (module is ReLU6 relu6Module)
            {
                relu6 = relu6Module;
            }
            else if (module is LazyConv2D lazyconv2dModule)
            {
                lazyconv2d = lazyconv2dModule;
            }
            else if (module is LazyBatchNorm2D lazybatchnorm2dModule)
            {
                lazybatchnorm2d = lazybatchnorm2dModule;
            }
            else if (module is Rish rishModule)
            {
                rish = rishModule;
            }
            else if(module is Embedding embeddingModule)
            {
                embedding = embeddingModule;
            }
            else if(module is RotaryPositionalEmbeddings ropeModule)
            {
                rotarypositionalembeddings = ropeModule;
            }
            else if(module is GatedLinearUnit gluModule)
            {
                gatedlinearunit = gluModule;
            }
            else
                throw new Exception($"Unhandled module type while wrapping ({module.GetType().Name}).");
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
            else if (typeof(Sparsemax).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.sparsemax;
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
            else if (typeof(LazyBatchNorm1D).Name.Equals(moduleWrapper.name))
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
            else if (typeof(RMSNorm).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.rmsnorm;
            }
            else if (typeof(ZeroPad2D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.zeropad2d;
            }
            else if (typeof(ResidualConnection.Fork).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.residualconnectionfork;
            }
            else if (typeof(ResidualConnection.Join).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.residualconnectionjoin;
            }
            else if (typeof(MirrorPad2D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.mirrorpad2d;
            }
            else if (typeof(ZeroPad1D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.zeropad1d;
            }
            else if (typeof(MirrorPad1D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.mirrorpad1d;
            }
            else if (typeof(AvgPool1D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.avgpool1d;
            }
            else if (typeof(BatchNorm2D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.batchnorm2d;
            }
            else if (typeof(GroupedQueryAttention).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.groupedqueryattention;
            }
            else if (typeof(LastSequence1DElementModule).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.lastsequence1delementmodule;
            }
            else if (typeof(DenseGPU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.densegpu;
            }
            else if (typeof(Squeeze).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.squeeze;
            }
            else if (typeof(Unsqueeze).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.unsqueeze;
            }
            else if (typeof(Permute).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.permute;
            }
            else if (typeof(SiLU).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.silu;
            }
            else if (typeof(Dropout2D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.dropout2d;
            }
            else if (typeof(ReLU6).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.relu6;
            }
            else if (typeof(LazyConv2D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.lazyconv2d;
            }
            else if (typeof(LazyBatchNorm2D).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.lazybatchnorm2d;
            }
            else if (typeof(Rish).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.rish;
            }
            else if(typeof(Embedding).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.embedding;
            }
            else if(typeof(RotaryPositionalEmbeddings).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.rotarypositionalembeddings;
            }
            else if(typeof(GatedLinearUnit).Name.Equals(moduleWrapper.name))
            {
                module = moduleWrapper.gatedlinearunit;
            }
            else
                throw new Exception($"Unhandled module type while unwrapping ({moduleWrapper.name}).");

            return module;
        }
    }

}
