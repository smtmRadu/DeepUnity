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
    public class Dense : IModule, ISerializationCallbackReceiver
    {
        private Tensor Input_Cache { get; set; }
        [SerializeField] public Device device;
        
        // Parameters (theta)
        [SerializeField] public Tensor param_W;
        [SerializeField] public Tensor param_B;

        // Gradients (g)
        [NonSerialized] public Tensor grad_W;
        [NonSerialized] public Tensor grad_B;
       

        public Dense(int in_features, int out_features, InitType init = InitType.Default, Device device = Device.CPU)
        {
            this.device = device;

            this.param_W = Tensor.Zeros(out_features, in_features);
            this.param_B = Tensor.Zeros(out_features);

            this.grad_W = Tensor.Zeros(out_features, in_features);
            this.grad_B = Tensor.Zeros(out_features);

            switch (init)
            {
                case InitType.Default:
                    float sqrtK = MathF.Sqrt(1f / in_features);
                    param_W.ForEach(x => Utils.Random.Range(-sqrtK, sqrtK));
                    param_B.ForEach(x => Utils.Random.Range(-sqrtK, sqrtK));
                    break;
                case InitType.HE:
                    float sigmaHE = MathF.Sqrt(2f / param_W.Shape[1]); //fanIn
                    param_W.ForEach(x => Utils.Random.Gaussian(0f, sigmaHE));
                    break;
                case InitType.Xavier:
                    float sigmaXA = MathF.Sqrt(2f / (param_W.Shape[0] + param_W.Shape[1])); // fanIn + fanOut
                    param_W.ForEach(x => Utils.Random.Gaussian(0f, sigmaXA));
                    break;
                case InitType.Normal:
                    param_W.ForEach(x => Utils.Random.Gaussian());
                    break;
                case InitType.Uniform:
                    param_W.ForEach(x => Utils.Random.Value * 2f - 1f); // [-1, 1]
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
            return Tensor.MatMul(param_W, input, device) + Tensor.Expand(param_B, axis: 1, times: input.Shape[1]);
        }
        public Tensor Backward(Tensor loss)
        {
            var transposedInput = Tensor.TransposeMat(Input_Cache);
            Tensor gradW = Tensor.MatMul(loss, transposedInput, device);
            Tensor gradB = Tensor.MatMul(loss, Tensor.Ones(transposedInput.Shape), device);

            // Average the gradients
            float batch = loss.Shape[1];

            // Update the gradients
            grad_W += gradW / batch;
            grad_B += gradB / batch;

            // Backpropagate the loss
            // Tensor dLossdActivation = Tensor.MatMul(Tensor.MatTranspose(t_W), loss, MatMulCS);
            // return dLossdActivation;

            // A bit faster back with double tranposition on loss (may work better on large dense)
            Tensor dLossActivation = Tensor.MatMul(Tensor.TransposeMat(loss), param_W, device);
            return Tensor.TransposeMat(dLossActivation);
        }

        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization, and one on them is called when weights.shape.length == 0.
            if (param_W.Shape.Length == 0)
                return;

            int outputs = param_W.Shape[0];
            int inputs = param_W.Shape[1];

            this.grad_W = Tensor.Zeros(outputs, inputs);
            this.grad_B = Tensor.Zeros(outputs);
        }
    }

}