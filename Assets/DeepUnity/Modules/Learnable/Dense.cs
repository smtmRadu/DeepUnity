using System;

namespace DeepUnity
{
    [Serializable]
    public class Dense : Learnable, IModule
    {
        private Tensor Input_Cache { get; set; }

        public Dense(int in_features, int out_features, InitType init = InitType.Default)
        {
            switch (init)
            {
                case InitType.Default:
                    float sqrtK = MathF.Sqrt(1f / in_features);
                    gamma = Tensor.RandomRange((-sqrtK, sqrtK), in_features, out_features);
                    beta = Tensor.RandomRange((-sqrtK, sqrtK), out_features);
                    break;
                case InitType.HE:
                    float sigmaHE = MathF.Sqrt(2f / gamma.Height); //fanIn
                    gamma = Tensor.RandomNormal((0, sigmaHE), in_features, out_features);
                    beta = Tensor.Zeros(out_features);
                    break;
                case InitType.Xavier:
                    float sigmaXA = MathF.Sqrt(2f / (gamma.Width + gamma.Height)); // fanIn + fanOut
                    gamma = Tensor.RandomNormal((0, sigmaXA), in_features, out_features);
                    beta = Tensor.Zeros(out_features);
                    break;
                case InitType.Normal:
                    gamma = Tensor.RandomNormal((0f, 1f), in_features, out_features);
                    beta = Tensor.Zeros(out_features);
                    break;
                case InitType.Uniform:
                    gamma = Tensor.RandomRange((-1, 1), in_features, out_features);
                    beta = Tensor.Zeros(out_features);
                    break;
                default:
                    throw new Exception("Unhandled initialization type!");
            }

            this.gradGamma = Tensor.Zeros(in_features, out_features);
            this.gradBeta = Tensor.Zeros(out_features);
        }

        public Tensor Predict(Tensor input)
        {
            int batch_size = input.Height;
            return Tensor.MatMul(input, gamma) + Tensor.Expand(beta, Dim.height, batch_size);
        }
        public Tensor Forward(Tensor input)
        {
            Input_Cache = Tensor.Identity(input);
            int batch_size = input.Height;
            return Tensor.MatMul(input, gamma) + Tensor.Expand(beta, Dim.height, batch_size);
        }
        public Tensor Backward(Tensor loss)
        {
            int batch_size = loss.Height;
            var transposedInput = Tensor.Transpose(Input_Cache, Dim.width, Dim.height);

            Tensor gradW = Tensor.MatMul(transposedInput, loss);
            Tensor gradB = Tensor.MatMul(Tensor.Ones(1, batch_size), loss);

            // Update the gradients
            gradGamma += gradW / batch_size;
            gradBeta += gradB / batch_size;

            // Backpropagate the loss
            Tensor dLossActivation = Tensor.MatMul(gamma, Tensor.Transpose(loss, Dim.width, Dim.height));
            return Tensor.Transpose(dLossActivation, Dim.width, Dim.height);
        }

    }

}