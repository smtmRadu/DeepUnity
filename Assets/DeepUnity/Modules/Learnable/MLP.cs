using System;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// A fast MLP module used inside models for higher performance and larger networks.
    /// All hidden layers have the same activation, last layer has linear/identity activation.
    /// Self optimizable, uses Adam optimizer internally, but does not affect the current optimizer choosed. The initialization is HE for ReLU and Glorot for Tanh.
    /// /Maybe adding dropout for the future.
    /// </summary>
    [Serializable]
    public class MLP : Learnable, IModule, ISelfOptimizable
    {
        private static int AllocateKernel()
        {
            return kernel_allocator++;
        }
        private static int kernel_allocator = 0;

        [SerializeField] private Matrix2D[] weights;
        [SerializeField] private Matrix2D[] biases;
        [SerializeField] private NonLinearity activation;

        private ComputeBuffer[] weights_cbuff;
        private ComputeBuffer[] biases_cbuff;

        private int allocated_kernel;
        private ComputeShader mlpCS;

        /// <summary>
        /// A fast multilayer-perceptron that lives inside GPU. Last layer has identity activation.
        /// </summary>
        /// <param name="in_channels"></param>
        /// <param name="out_channels"></param>
        /// <param name="num_layers"> The number of hidden layers in the network. Default 2. Network default shape [in_c, 128 128, out_c]</param>
        /// <param name="hidden_size">The number of units of the hidden layers. Default 128. Network default shape [in_c, 128 128, out_c]</param>
        /// <param name="hiddenActivation">Depending of the activation, the network receives different initialization automatically.</param>
        public MLP(
            int in_channels, 
            int out_channels,
            int num_layers = 2, 
            int hidden_size = 128, 
            NonLinearity hiddenActivation = NonLinearity.ReLU)
            : base (Device.GPU, InitType.Zeros, InitType.Zeros, new int[1], new int[1], 1, 1)
        {

            if (num_layers > 5 || num_layers < 1)
            {
                throw new ArgumentException();
            }

            this.activation = hiddenActivation;
            mlpCS = DeepUnityMeta.MLPCS;
            this.allocated_kernel = AllocateKernel();

            // Initialize weights and biases
            weights = new Matrix2D[num_layers + 1];
            biases = new Matrix2D[num_layers + 1];
            for (int i = 0; i < num_layers + 1; i++)
            {
               
                int in_feat;
                int out_feat;
                InitType init = activation == NonLinearity.ReLU ? InitType.HE_Uniform : InitType.Glorot_Uniform;

                if (i == 0)
                {
                    in_feat = in_channels;
                    out_feat = hidden_size;
                }
                if(i == num_layers)
                {
                    in_feat = hidden_size;
                    out_feat = out_channels;
                }
                else
                {
                    in_feat = hidden_size;
                    out_feat = hidden_size;
                }

                Dense dense = new Dense(in_feat, out_feat, init, init);
                weights[i] = new Matrix2D(dense.gamma.ToArray(), dense.gamma.Size(-2), dense.gamma.Size(-1));
                biases[i] = new Matrix2D(dense.beta.ToArray(), 1, dense.beta.Size(-1));
            }

            InitializeOnGPU();
        }
        private void InitializeOnGPU()
        {
            weights_cbuff = new ComputeBuffer[weights.Length];
            biases_cbuff =new ComputeBuffer[biases.Length];

            for (int i = 0; i < weights.Length; i++)
            {
                weights_cbuff[i] = new ComputeBuffer(1, weights[i].GetByteSize());
                mlpCS.SetBuffer(allocated_kernel, $"weights{i}", weights_cbuff[i]);

                biases_cbuff[i] = new ComputeBuffer(1, biases[i].GetByteSize());
                mlpCS.SetBuffer(allocated_kernel, $"biases{i}", biases_cbuff[i]);
            }

            ComputeBuffer activ_cb = new ComputeBuffer(1, sizeof(int));
            activ_cb.SetData(new int[] { (int)this.activation });
            mlpCS.SetBuffer(allocated_kernel, "activation", activ_cb);
        }


        public Tensor Predict(Tensor input)
        {
            Matrix2D in_mat = new Matrix2D(input.ToArray(), input.Size(-2), input.Size(-1));
            ComputeBuffer input_computeBuffer = new ComputeBuffer(1, in_mat.GetByteSize());
            mlpCS.SetBuffer(allocated_kernel, "input", input_computeBuffer);
            for (int i = 0; i < weights.Length; i++)
            {

            }

            // retrieve the result directly from the input....
            throw new NotImplementedException();
        }
        public Tensor Forward(Tensor input)
        {
            throw new NotImplementedException();
        }
        public Tensor Backward(Tensor loss)
        {
            throw new NotImplementedException();
        }

        public void SelfOptimise(float lr)
        {

        }

        public override void OnBeforeSerialize()
        {
            // Retrieve the data from the gpu to gpu
            for (int i = 0; i < weights.Length; i++)
            {
                Matrix2D[] retriever = new Matrix2D[1];
                weights_cbuff[i].GetData(retriever);
                weights[i] = retriever[0];

                biases_cbuff[i].GetData(retriever);
                biases[i] = retriever[0];
            }
        }
        public override void OnAfterDeserialize()
        {
            mlpCS = DeepUnityMeta.MLPCS;

            AllocateKernel();
            InitializeOnGPU();
        }

        [Serializable]
        struct Matrix2D
        {
            [SerializeField] int width;
            [SerializeField] int height;
            [SerializeField] float[] data; 
            public Matrix2D(float[] dat, int w, int h)
            {
                this.data = dat;
                this.width = w;
                this.height = h;
            }

            public int GetByteSize() => sizeof(int) * 2 + sizeof(float) * data.Length;
        }
    }

}