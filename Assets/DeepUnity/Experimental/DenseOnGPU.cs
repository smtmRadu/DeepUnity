/*using System;
using UnityEngine;

// Extremly slow option, a dense where parameters are living in the GPU
namespace DeepUnity
{
    [Serializable]
    public class DenseOnGPU : Learnable, IModule
    {
        private Tensor InputCache { get; set; }
        private TensorGPU InputCacheGPU { get; set; }


        public DenseOnGPU(int in_features, int out_features, InitType init = InitType.Default, Device device = Device.CPU) : base(device)
        {
            if (device == Device.CPU)
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

                gammaGrad = Tensor.Zeros(out_features, in_features);
                betaGrad = Tensor.Zeros(out_features);
            }
            else
            {
                switch (init)
                {
                    case InitType.Default:
                        float sqrtK = MathF.Sqrt(1f / in_features);
                        gammaGPU = TensorGPU.RandomRange((-sqrtK, sqrtK), out_features, in_features);
                        betaGPU = TensorGPU.RandomRange((-sqrtK, sqrtK), out_features);
                        break;
                    case InitType.HE:
                        float sigmaHE = MathF.Sqrt(2f / in_features); //fanIn
                        gammaGPU = TensorGPU.RandomNormal((0, sigmaHE), out_features, in_features);
                        betaGPU = TensorGPU.Zeros(out_features);
                        break;
                    case InitType.Xavier:
                        float sigmaXA = MathF.Sqrt(2f / (in_features + out_features)); // fanIn + fanOut
                        gammaGPU = TensorGPU.RandomNormal((0, sigmaXA), out_features, in_features);
                        betaGPU = TensorGPU.Zeros(out_features);
                        break;
                    case InitType.Normal:
                        gammaGPU = TensorGPU.RandomNormal((0f, 1f), out_features, in_features);
                        betaGPU = TensorGPU.Zeros(out_features);
                        break;
                    case InitType.Uniform:
                        gammaGPU = TensorGPU.RandomRange((-1, 1), out_features, in_features);
                        betaGPU = TensorGPU.Zeros(out_features);
                        break;
                    default:
                        throw new Exception("Unhandled initialization type!");
                }

                gammaGradGPU = TensorGPU.Zeros(out_features, in_features);
                betaGradGPU = TensorGPU.Zeros(out_features);
            }

        }


        public Tensor Predict(Tensor input)
        {
            bool isBatched = input.Rank == 2;
            int batch_size = input.Size(-2);

            if (device == Device.CPU)
            {
                if (isBatched)
                {
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
                }
                else
                {
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + beta;
                }
            }
            else
            {
                if (isBatched)
                {
                    TensorGPU output = TensorGPU.MatMul(TensorGPU.Identity(input), TensorGPU.Transpose(gammaGPU, 0, 1))
                                     + TensorGPU.Expand(TensorGPU.Unsqueeze(betaGPU, 0), 0, batch_size);
                    return Tensor.Identity(output);
                }
                else
                {
                    TensorGPU output = TensorGPU.MatMul(TensorGPU.Identity(input), TensorGPU.Transpose(gammaGPU, 0, 1)) + betaGPU;
                    return Tensor.Identity(output);
                }

            }
        }
        public Tensor Forward(Tensor input)
        {
            bool isBatched = input.Rank == 2;
            int batch_size = input.Size(-2);

            if (device == Device.CPU)
            {
                InputCache = Tensor.Identity(input);
                if (isBatched)
                {
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
                }
                else
                {
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + beta;
                }

            }
            else
            {
                InputCacheGPU = TensorGPU.Identity(input);
                if (isBatched)
                {
                    TensorGPU output = TensorGPU.MatMul(InputCacheGPU, TensorGPU.Transpose(gammaGPU, 0, 1))
                                     + TensorGPU.Expand(TensorGPU.Unsqueeze(betaGPU, 0), 0, batch_size);
                    return Tensor.Identity(output);
                }
                else
                {
                    TensorGPU output = TensorGPU.MatMul(InputCacheGPU, TensorGPU.Transpose(gammaGPU, 0, 1)) + betaGPU;
                    return Tensor.Identity(output);
                }

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


            if (device == Device.CPU)
            {
                var transposedLoss = Tensor.Transpose(loss, 0, 1);

                Tensor gradW = Tensor.MatMul(transposedLoss, InputCache);
                Tensor gradB = Tensor.MatMul(transposedLoss, Tensor.Ones(batch_size));

                // Update the gradients
                gammaGrad += gradW / batch_size; // (out, in)
                betaGrad += gradB / batch_size; // (out)

                return Tensor.MatMul(loss, gamma);
            }
            else
            {
                TensorGPU lossGPU = TensorGPU.Identity(loss);
                TensorGPU transposedLossGPU = TensorGPU.Transpose(lossGPU, 0, 1);
                TensorGPU gradW = TensorGPU.MatMul(transposedLossGPU, InputCacheGPU);
                TensorGPU gradB = TensorGPU.MatMul(transposedLossGPU, TensorGPU.Ones(batch_size));

                // Update the gradients
                gammaGradGPU += gradW / batch_size; // (out, in)
                betaGradGPU += gradB / batch_size; // (out)

                gammaGPU -= 0.01f * gammaGradGPU;
                betaGPU -= 0.01f * betaGradGPU;

                TensorGPU back = TensorGPU.MatMul(lossGPU, gammaGPU);
                return Tensor.Identity(back);
            }
        }
    }

}*/