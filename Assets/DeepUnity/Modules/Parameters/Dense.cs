using UnityEngine;
using System;

namespace DeepUnity
{


    [Serializable]
    public class Dense : IModule, IParameters
    {
        private Tensor Input_Cache { get; set; }

        // Parameters (theta)
        [SerializeField] public Tensor weights;
        [SerializeField] public Tensor biases;

        // Gradients (g)
        [NonSerialized] public Tensor grad_Weights;
        [NonSerialized] public Tensor grad_Biases;


        public Dense(int in_features, int out_features, InitType init = InitType.Default)
        {
            this.weights = Tensor.Zeros(in_features, out_features);
            this.biases = Tensor.Zeros(out_features);

            this.grad_Weights = Tensor.Zeros(in_features, out_features);
            this.grad_Biases = Tensor.Zeros(out_features);

            switch (init)
            {
                case InitType.Default:
                    float sqrtK = MathF.Sqrt(1f / in_features);
                    weights.ForEach(x => Utils.Random.Range(-sqrtK, sqrtK));
                    biases.ForEach(x => Utils.Random.Range(-sqrtK, sqrtK));
                    break;
                case InitType.HE:
                    float sigmaHE = MathF.Sqrt(2f / weights.Shape.height); //fanIn
                    weights.ForEach(x => Utils.Random.Gaussian(0f, sigmaHE));
                    break;
                case InitType.Xavier:
                    float sigmaXA = MathF.Sqrt(2f / (weights.Shape.width + weights.Shape.height)); // fanIn + fanOut
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

        public Tensor Predict(Tensor input)
        {
            int batch = input.Shape.height;
            return Tensor.MatMul(input, weights) + Tensor.Expand(biases, -1, batch);

        }
        public Tensor Forward(Tensor input)
        {
            // for faster improvement on GPU, set for forward only on CPU!
            // it seems like forward is always faster with CPU rather than GPU for matrices < 1024 size. Maybe on large scales it must be changed again on GPU.
            Input_Cache = Tensor.Identity(input);
            int batch_size = input.Shape.height;
            return Tensor.MatMul(input, weights) + Tensor.Expand(biases, -1, batch_size);
        }
        public Tensor Backward(Tensor loss)
        {
            int batch_size = loss.Shape.height;
            var transposedInput = Tensor.MatTranspose(Input_Cache);

            Tensor gradW = Tensor.MatMul(transposedInput, loss);
            Tensor gradB = Tensor.MatMul(Tensor.Ones(1, batch_size), loss);


            // Update the gradients
            grad_Weights += gradW / batch_size;
            grad_Biases += gradB / batch_size;

            // Backpropagate the loss
            // Tensor dLossdActivation = Tensor.MatMul(Tensor.MatTranspose(t_W), loss, MatMulCS);
            // return dLossdActivation;

            // A bit faster back with double tranposition on loss (may work better on large dense)
            Tensor dLossActivation = Tensor.MatMul(weights, Tensor.MatTranspose(loss));
            return Tensor.MatTranspose(dLossActivation);
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
            // Stil here a problemo..
            // This function is actually having 2 workers on serialization, and one on them is called when weights.shape.length == 0.
            if (weights.Shape == null || weights.Shape.width == 0)
                return;

            int outputs = weights.Shape.height;
            int inputs = weights.Shape.width;

            this.grad_Weights = Tensor.Zeros(inputs, outputs);
            this.grad_Biases = Tensor.Zeros(outputs);
        }
    }

}