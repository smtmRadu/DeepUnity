using UnityEngine;
using System;

namespace DeepUnity
{
    public enum InitType
    {
        Default,
        HE,
        Xavier,
        Normal,
        Uniform,
    }
    public enum Device
    {
        CPU,
        GPU
    }
    
    [Serializable]
    public class Dense : IModule, IParameters, ISerializationCallbackReceiver
    {
        private Tensor Input_Cache { get; set; }
        [SerializeField] public Device device;
        
        // Parameters (theta)
        [SerializeField] public Tensor weights;
        [SerializeField] public Tensor biases;

        // Gradients (g)
        [NonSerialized] public Tensor grad_Weights;
        [NonSerialized] public Tensor grad_Biases;
       

        public Dense(int in_features, int out_features, InitType init = InitType.Default, Device device = Device.CPU)
        {
            this.device = device;

            this.weights = Tensor.Zeros(out_features, in_features);
            this.biases = Tensor.Zeros(out_features);

            this.grad_Weights = Tensor.Zeros(out_features, in_features);
            this.grad_Biases = Tensor.Zeros(out_features);

            switch (init)
            {
                case InitType.Default:
                    float sqrtK = MathF.Sqrt(1f / in_features);
                    weights.ForEach(x => Utils.Random.Range(-sqrtK, sqrtK));
                    biases.ForEach(x => Utils.Random.Range(-sqrtK, sqrtK));
                    break;
                case InitType.HE:
                    float sigmaHE = MathF.Sqrt(2f / weights.Shape[1]); //fanIn
                    weights.ForEach(x => Utils.Random.Gaussian(0f, sigmaHE));
                    break;
                case InitType.Xavier:
                    float sigmaXA = MathF.Sqrt(2f / (weights.Shape[0] + weights.Shape[1])); // fanIn + fanOut
                    weights.ForEach(x => Utils.Random.Gaussian(0f, sigmaXA));
                    break;
                case InitType.Normal:
                    weights.ForEach(x => Utils.Random.Gaussian());
                    break;
                case InitType.Uniform:
                    weights.ForEach(x => Utils.Random.Value * 2f - 1f); // [-1, 1]
                    break;
                default:
                    throw new Exception("Unhandled initialization type!");
            }
        }

        public Tensor Forward(Tensor input) 
        {
            // for faster improvement on GPU, set for forward only on CPU!
            // it seems like forward is always faster with CPU rather than GPU for matrices < 1024 size. Maybe on large scales it must be changed again on GPU.
            Input_Cache = Tensor.Identity(input);
            return Tensor.MatMul(weights, input, device) + Tensor.Expand(biases, axis: 1, times: input.Shape[1]);
        }
        public Tensor Backward(Tensor loss)
        {
            var transposedInput = Tensor.Transpose(Input_Cache, 0, 1);
            Tensor gradW = Tensor.MatMul(loss, transposedInput, device);
            Tensor gradB = Tensor.MatMul(loss, Tensor.Ones(transposedInput.Shape), device);

            // Average the gradients
            float batch = loss.Shape[1];

            // Update the gradients
            grad_Weights += gradW / batch;
            grad_Biases += gradB / batch;

            // Backpropagate the loss
            // Tensor dLossdActivation = Tensor.MatMul(Tensor.MatTranspose(t_W), loss, MatMulCS);
            // return dLossdActivation;

            // A bit faster back with double tranposition on loss (may work better on large dense)
            Tensor dLossActivation = Tensor.MatMul(Tensor.Transpose(loss, 0, 1), weights, device);
            return Tensor.Transpose(dLossActivation, 0, 1);
        }


        public void ZeroGrad()
        {
            grad_Weights.ForEach(x => 0f);
            grad_Biases.ForEach(x => 0f);
        }
        public void ClipGradValue(float clip_value)
        {
            Tensor.Clip(grad_Weights, -clip_value, clip_value);
            Tensor.Clip(grad_Biases, -clip_value, clip_value);
        }
        public void ClipGradNorm(float max_norm)
        {
            Tensor norm = Tensor.Norm(grad_Weights, NormType.ManhattanL1) + Tensor.Norm(grad_Biases, NormType.ManhattanL1);

            if (norm[0] > max_norm)
            {
                float scale = max_norm / norm[0];
                grad_Weights *= scale;
                grad_Biases *= scale;
            }
        }

        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization, and one on them is called when weights.shape.length == 0.
            if (weights.Shape.Length == 0)
                return;

            int outputs = weights.Shape[0];
            int inputs = weights.Shape[1];

            this.grad_Weights = Tensor.Zeros(outputs, inputs);
            this.grad_Biases = Tensor.Zeros(outputs);
        }
    }

}