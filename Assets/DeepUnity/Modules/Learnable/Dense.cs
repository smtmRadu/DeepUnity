using System;

namespace DeepUnity
{
    [Serializable]
    public class Dense : Learnable, IModule
    {
        private Tensor InputCache { get; set; }
        private Device device;

        public Dense(int in_features, int out_features, InitType init = InitType.Default, Device device = Device.CPU)
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
            this.device = device;
        }

        public Tensor Predict(Tensor input)
        {
            // input = (B, IN)
            // gamma = (OUT, IN)

            // out = (B, OUT)
            // input = (batch_size, in_features)
            // gamma = (out_features, in_features)
            // out = (out_features, batch_size)

            bool isBatched = input.Rank == 2;

            if(isBatched)
            {
                int batch_size = input.Size(-2);

                if (device == Device.CPU)
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
                else
                    return Tensor.MatMulGPU(input, Tensor.Transpose(gamma, 0, 1)) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
            }
            else
            {
                if (device == Device.CPU)
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + beta;
                else
                    return Tensor.MatMulGPU(input, Tensor.Transpose(gamma, 0, 1)) + beta;
            }
            
        }
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);

            bool isBatched = input.Rank == 2;

            if (isBatched)
            {
                int batch_size = input.Size(-2);

                if (device == Device.CPU)
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
                else
                    return Tensor.MatMulGPU(input, Tensor.Transpose(gamma, 0, 1)) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
            }
            else
            {
                if (device == Device.CPU)
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + Tensor.Unsqueeze(beta, 0);
                else
                    return Tensor.MatMulGPU(input, Tensor.Transpose(gamma, 0, 1)) + Tensor.Unsqueeze(beta, 0);
            }
        }
        public Tensor Backward(Tensor loss)
        {
            // input = (B, IN)
            // loss = (B, OUT)
            // tloss = (OUT, B)

            //gradGamma(OUT, IN)
            bool isBatched = loss.Rank == 2;
            int batch_size = isBatched ? loss.Size(-2) : 1;
            var transposedLoss = Tensor.Transpose(loss, 0, 1);


            Tensor gradW = device == Device.CPU ? 
                Tensor.MatMul(transposedLoss, InputCache) : 
                Tensor.MatMulGPU(transposedLoss, InputCache);

            Tensor gradB = device == Device.CPU ? 
                Tensor.MatMul(transposedLoss, Tensor.Ones(batch_size)) : 
                Tensor.MatMulGPU(transposedLoss, Tensor.Ones(batch_size));

            // Update the gradients
            gradGamma += gradW / batch_size; // (out, in)
            gradBeta += gradB / batch_size; // (out)
            
            // Backpropagate the loss (batch_size, in)
            return device == Device.CPU ? 
                Tensor.MatMul(loss, gamma) : 
                Tensor.MatMulGPU(loss, gamma);
        }

    }

}