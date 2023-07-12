using System;

namespace DeepUnity
{
    [Serializable]
    public class Dense : Learnable, IModule
    {
        private Tensor InputCache { get; set; }

        public Dense(int in_features, int out_features, InitType init = InitType.Default)
        {
            switch (init)
            {
                case InitType.Default:
                    float sqrtK = MathF.Sqrt(1f / in_features);
                    gamma = Tensor.RandomRange((-sqrtK, sqrtK), out_features, in_features);
                    beta = Tensor.RandomRange((-sqrtK, sqrtK), out_features);
                    break;
                case InitType.HE:
                    float sigmaHE = MathF.Sqrt(2f / in_features); //fanIn
                    gamma = Tensor.RandomNormal((0, sigmaHE), out_features, in_features);
                    beta = Tensor.Zeros(out_features);
                    break;
                case InitType.Xavier:
                    float sigmaXA = MathF.Sqrt(2f / (in_features + out_features)); // fanIn + fanOut
                    gamma = Tensor.RandomNormal((0, sigmaXA), out_features, in_features);
                    beta = Tensor.Zeros(out_features);
                    break;
                case InitType.Normal:
                    gamma = Tensor.RandomNormal((0f, 1f), out_features, in_features);
                    beta = Tensor.Zeros(out_features);
                    break;
                case InitType.Uniform:
                    gamma = Tensor.RandomRange((-1, 1), out_features, in_features);
                    beta = Tensor.Zeros(out_features);
                    break;
                default:
                    throw new Exception("Unhandled initialization type!");
            }

            this.gradGamma = Tensor.Zeros(out_features, in_features);
            this.gradBeta = Tensor.Zeros(out_features);
        }

        public Tensor Predict(Tensor input)
        {
            // input = (B, IN)
            // gamma = (OUT, IN)

            // out = (B, OUT)
            // input = (batch_size, in_features)
            // gamma = (out_features, in_features)
            // out = (out_features, batch_size)
            int batch_size = input.Height;
            return Tensor.MatMul(input, Tensor.Transpose(gamma, Dim.width, Dim.height)) + Tensor.Expand(beta, Dim.height, batch_size);
        }
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);

            int batch_size = input.Height;
            return Tensor.MatMul(input, Tensor.Transpose(gamma, Dim.width, Dim.height)) + Tensor.Expand(beta, Dim.height, batch_size);
        }
        public Tensor Backward(Tensor loss)
        {
            // input = (B, IN)
            // loss = (B, OUT)
            // tloss = (OUT, B)

            //gradGamma(OUT, IN)
            int batch_size = loss.Height;
            var transposedLoss = Tensor.Transpose(loss, Dim.width, Dim.height);

            Tensor gradW = Tensor.MatMul(transposedLoss, InputCache);
            Tensor gradB = Tensor.MatMul(transposedLoss, Tensor.Ones(batch_size));

            // Update the gradients
            gradGamma += gradW / batch_size; // (out, in)
            gradBeta += gradB / batch_size; // (out)
            
            // Backpropagate the loss (batch_size, in)
            return Tensor.MatMul(loss, gamma);
        }

    }

}