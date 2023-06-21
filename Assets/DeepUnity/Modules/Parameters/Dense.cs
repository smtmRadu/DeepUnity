using System;

namespace DeepUnity
{


    [Serializable]
    public class Dense : Learnable, IModule
    {
        private Tensor Input_Cache { get; set; }

        public Dense(int in_features, int out_features, InitType init = InitType.Default)
        {
            this.gamma = Tensor.Zeros(in_features, out_features);
            this.beta = Tensor.Zeros(out_features);

            this.gradGamma = Tensor.Zeros(in_features, out_features);
            this.gradBeta = Tensor.Zeros(out_features);

            switch (init)
            {
                case InitType.Default:
                    float sqrtK = MathF.Sqrt(1f / in_features);
                    gamma.ForEach(x => Utils.Random.Range(-sqrtK, sqrtK));
                    beta.ForEach(x => Utils.Random.Range(-sqrtK, sqrtK));
                    break;
                case InitType.HE:
                    float sigmaHE = MathF.Sqrt(2f / gamma.Shape.height); //fanIn
                    gamma.ForEach(x => Utils.Random.Gaussian(0f, sigmaHE));
                    break;
                case InitType.Xavier:
                    float sigmaXA = MathF.Sqrt(2f / (gamma.Shape.width + gamma.Shape.height)); // fanIn + fanOut
                    gamma.ForEach(x => Utils.Random.Gaussian(0f, sigmaXA));
                    break;
                case InitType.Normal:
                    gamma.ForEach(x => Utils.Random.Gaussian());
                    break;
                case InitType.Uniform:
                    gamma.ForEach(x => Utils.Random.Value * 2f - 1f); // [-1, 1]
                    break;
                default:
                    throw new Exception("Unhandled initialization type!");
            }
        }

        public Tensor Predict(Tensor input)
        {
            int batch = input.Shape.height;
            return Tensor.MatMul(input, gamma) + Tensor.Expand(beta, TDim.height, batch);

        }
        public Tensor Forward(Tensor input)
        {
            // for faster improvement on GPU, set for forward only on CPU!
            // it seems like forward is always faster with CPU rather than GPU for matrices < 1024 size. Maybe on large scales it must be changed again on GPU.
            Input_Cache = Tensor.Identity(input);
            int batch_size = input.Shape.height;
            return Tensor.MatMul(input, gamma) + Tensor.Expand(beta,TDim.height, batch_size);
        }
        public Tensor Backward(Tensor loss)
        {
            int batch_size = loss.Shape.height;
            var transposedInput = Tensor.Transpose(Input_Cache, TDim.width, TDim.height);

            Tensor gradW = Tensor.MatMul(transposedInput, loss);
            Tensor gradB = Tensor.MatMul(Tensor.Ones(1, batch_size), loss);


            // Update the gradients
            gradGamma += gradW / batch_size;
            gradBeta += gradB / batch_size;

            // Backpropagate the loss
            Tensor dLossActivation = Tensor.MatMul(gamma, Tensor.Transpose(loss, TDim.width, TDim.height));
            return Tensor.Transpose(dLossActivation, TDim.width, TDim.height);
        }

    }

}