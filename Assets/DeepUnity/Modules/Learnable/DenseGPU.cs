/*using System;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// A deep neural network living inside GPU that can be used as a layer in Models for high level computing.
    /// </summary>
    [Serializable]
    public class DenseGPU : ISerializationCallbackReceiver, IModule, ILearnable
    {
        private Tensor InputCache { get; set; }

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


            TensorGPU output;
            if (isBatched)
            {
                var unsqueezedBiasesGPU = TensorGPU.Unsqueeze(biases, 0);
                var expandedBiasesGPU = TensorGPU.Expand(unsqueezedBiasesGPU, 0, batch_size);

                output = matmul + expandedBiasesGPU;

                unsqueezedBiasesGPU.Dispose();
                expandedBiasesGPU.Dispose();
            }
            else
            {
                output = matmul + biases;
            }

            weightsT.Dispose();
            matmul.Dispose();

            return output;

        }

        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);

            return Predict(input);
        }


        public TensorGPU Backward(TensorGPU loss)
        {
            throw new NotImplementedException();
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
}*/