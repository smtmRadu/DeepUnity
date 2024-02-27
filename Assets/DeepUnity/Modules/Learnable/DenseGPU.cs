using System;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// A deep neural network living inside GPU that can be used as a layer in Models for high level computing.
    /// </summary>
    [Serializable]
    public class DenseGPU : ISerializationCallbackReceiver, IModule, ILearnable
    {
        private TensorGPU InputCache { get; set; }

        [SerializeField] TensorGPU weights;
        [SerializeField] TensorGPU biases;

        [SerializeField] TensorGPU weightsGrad;
        [SerializeField] TensorGPU biasesGrad;


        public DenseGPU(int in_features, int out_features, InitType gamma_init = InitType.LeCun_Uniform, InitType beta_init = InitType.LeCun_Uniform)
        {
            if (in_features < 1)
                throw new ArgumentException("In_features cannot be less than 1.");
            if (out_features < 1)
                throw new ArgumentException("Out_features cannot be less than 1.");

            throw new NotImplementedException();
            // If i remember correctly the matmul is not correct 
            weights = Initializer.CreateParameterGPU(new int[] { out_features, in_features }, in_features, out_features, gamma_init);
            biases = Initializer.CreateParameterGPU(new int[] { out_features }, in_features, out_features, beta_init);
            weightsGrad = TensorGPU.Zeros(weights.Shape);
            biasesGrad = TensorGPU.Zeros(biases.Shape);
        }
        public Tensor Predict(Tensor input)
        {
            if (input.Size(-1) != weights.Size(-1))
                throw new ShapeException($"Input features ({input.Size(-1)}) does not match with the DNNGPU Layer features_num ({weights.Size(-1)}).");

            bool isBatched = input.Rank == 2;
            int batch_size = isBatched ? input.Size(-2) : 1;

            TensorGPU inputGPU = TensorGPU.Identity(input);
            TensorGPU outputGPU = TensorGPU.Zeros(batch_size, biases.Size(-1));
            ComputeShader cs = DeepUnityMeta.DenseCS;
            cs.SetBuffer(0, "input", inputGPU.data);
            cs.SetBuffer(0, "gamma", weights.data);
            cs.SetBuffer(0, "beta", biases.data);
            cs.SetBuffer(0, "output", outputGPU.data);

            Tensor y = Tensor.Identity(outputGPU);
            inputGPU.Dispose();
            outputGPU.Dispose();
            return y;
        }

        public Tensor Forward(Tensor input)
        {
            if (InputCache != null)
                InputCache.Dispose();
            InputCache = TensorGPU.Identity(input);

            return Predict(input);
        }


        public Tensor Backward(Tensor loss)
        {
            bool isBatched = loss.Rank == 2;
            int batch_size = isBatched ? loss.Size(-2) : 1;
            int H_in = weights.Size(-1);
            int H_out = biases.Size(-1);


            ComputeShader cs = DeepUnityMeta.DenseCS;

            TensorGPU lossGPU = TensorGPU.Identity(loss);
            cs.SetBuffer(1, "loss", lossGPU.data);
            cs.SetBuffer(1, "input", InputCache.data);
            cs.SetBuffer(1, "gamma_grad", weightsGrad.data);
            cs.SetBuffer(1, "beta_grad", biasesGrad.data);
            cs.SetInt("batch_size", batch_size);
            cs.SetInt("in_features", H_in);
            cs.SetInt("out_features", H_out);

            cs.Dispatch(1,
                   (H_in + 31) / 32,
                   (H_out + 31) / 32,
                   1);

            TensorGPU inputGradGPU = TensorGPU.MatMul(lossGPU, weights);
            Tensor inputGrad = Tensor.Identity(inputGradGPU);
            inputGradGPU.Dispose();
            lossGPU.Dispose();
            return inputGrad;
        }

        public object Clone()
        {
            var dense = new DenseGPU(1, 1);
            dense.weights = (TensorGPU)this.weights.Clone();
            dense.biases = (TensorGPU)this.biases.Clone();
            dense.weightsGrad = (TensorGPU)this.weightsGrad.Clone();
            dense.biasesGrad = (TensorGPU)this.biasesGrad.Clone();
            return dense;
        }

        public Parameter[] Parameters()
        {
            if (weightsGrad == null)
                OnAfterDeserialize();

            var w = new Parameter(weights, weightsGrad);
            var b = new Parameter(biases, biasesGrad);

            return new Parameter[] { w, b };
        }

        public int ParametersCount()
        {
            return weights.Count() + biases.Count();
        }

        public void SetDevice(Device device)
        {
            return;
        }

        public void OnBeforeSerialize()
        {

        }
        public virtual void OnAfterDeserialize()
        {
            if(weights.Shape == null)
                return;
            

            if (weights.Shape.Length == 0)
                return;

            this.weightsGrad = TensorGPU.Zeros(weights.Shape);
            this.biasesGrad = TensorGPU.Zeros(biases.Shape);
        }
    }
}