using System;
using UnityEngine;
namespace DeepUnity
{
    [Serializable]
    public class Dense : Learnable, IModule
    {
        private Tensor InputCache { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="in_features"></param>
        /// <param name="out_features"></param>
        /// <param name="init"></param>
        /// <param name="device">MatMul operation running on CPU or GPU.</param>
        /// <exception cref="Exception"></exception>
        public Dense(int in_features, int out_features, InitType init = InitType.Default, Device device = Device.CPU) : base(device)
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
                case InitType.Debug:
                    gamma = Tensor.Fill(1.5f, out_features, in_features);
                    beta = Tensor.Fill(2f, out_features);
                    break;
                default:
                    throw new Exception("Unhandled initialization type!");
            }

            gammaGrad = Tensor.Zeros(out_features, in_features);
            betaGrad = Tensor.Zeros(out_features);
        }
        public Tensor Predict(Tensor input)
        {
            // input = (B, IN)
            // gamma = (OUT, IN)

            // out = (B, OUT)
            // input = (batch_size, in_features)
            // gamma = (out_features, in_features)
            // out = (out_features, batch_size)
            int batch_size = input.Rank == 2 ? input.Size(-2) : 1;

            if (device == Device.CPU)
            {
                if (batch_size == 1)
                {
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + beta;
                }
                else
                {
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
                }
            }
            else
            {
                Tensor output = Tensor.Zeros(batch_size, beta.Size(-1));

                ComputeShader cs = DeepUnityMeta.DenseCS;

                ComputeBuffer inputBuffer = new ComputeBuffer(input.Count(), 4);
                inputBuffer.SetData(input.Data);
                cs.SetBuffer(0, "input", inputBuffer);

                ComputeBuffer tranposedGammaBuffer = new ComputeBuffer(gamma.Count(), 4);
                tranposedGammaBuffer.SetData(Tensor.Transpose(gamma, 0, 1).Data);
                cs.SetBuffer(0, "transposed_gamma", tranposedGammaBuffer);

                ComputeBuffer betaBuffer = new ComputeBuffer(beta.Count(), 4);
                betaBuffer.SetData(beta.Data);
                cs.SetBuffer(0, "beta", betaBuffer);

                ComputeBuffer outputBuffer = new ComputeBuffer(output.Count(), 4);
                outputBuffer.SetData(output.Data);
                cs.SetBuffer(0, "output", outputBuffer);

                cs.SetInt("batch_size", batch_size);
                cs.SetInt("in_features", gamma.Size(-1));
                cs.SetInt("out_features", beta.Size(-1));
                cs.SetInt("gamma_rank", gamma.Rank);
                cs.SetInt("input_rank", input.Rank);

                cs.Dispatch(0,
                    (output.Size(-1) + 32 - 1) / 32,
                    (output.Size(-2) + 32 - 1) / 32,
                    1);

                outputBuffer.GetData(output.Data);


                inputBuffer.Release();
                tranposedGammaBuffer.Release();
                betaBuffer.Release();
                outputBuffer.Release();

                // output.Squeeze(); do not squeeze
                return output.Squeeze(-2);
            }
        }
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);

            int batch_size = input.Rank == 2 ? input.Size(-2) : 1;

            if (device == Device.CPU)
            {
                if (batch_size == 1)
                {
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + beta;
                }
                else
                {
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
                }
            }
            else
            {
                Tensor output = Tensor.Zeros(batch_size, beta.Size(-1));

                ComputeShader cs = DeepUnityMeta.DenseCS;

                ComputeBuffer inputBuffer = new ComputeBuffer(input.Count(), 4);
                inputBuffer.SetData(input.Data);
                cs.SetBuffer(0, "input", inputBuffer);

                ComputeBuffer tranposedGammaBuffer = new ComputeBuffer(gamma.Count(), 4);
                tranposedGammaBuffer.SetData(Tensor.Transpose(gamma, 0, 1).Data);
                cs.SetBuffer(0, "transposed_gamma", tranposedGammaBuffer);

                ComputeBuffer betaBuffer = new ComputeBuffer(beta.Count(), 4);
                betaBuffer.SetData(beta.Data);
                cs.SetBuffer(0, "beta", betaBuffer);

                ComputeBuffer outputBuffer = new ComputeBuffer(output.Count(), 4);
                outputBuffer.SetData(output.Data);
                cs.SetBuffer(0, "output", outputBuffer);

                cs.SetInt("batch_size", batch_size);
                cs.SetInt("in_features", gamma.Size(-1));
                cs.SetInt("out_features", beta.Size(-1));
                cs.SetInt("gamma_rank", gamma.Rank);
                cs.SetInt("input_rank", input.Rank);

                cs.Dispatch(0,
                    (output.Size(-1) + 31) / 32,
                    (output.Size(-2) + 31) / 32,
                    1);

                outputBuffer.GetData(output.Data);
                

                inputBuffer.Release();
                tranposedGammaBuffer.Release();
                betaBuffer.Release();
                outputBuffer.Release();

                // output.Squeeze(); do not squeeze
                return output.Squeeze(-2);
            }
        }
        public Tensor Backward(Tensor loss)
        {
            // input = (B, IN)
            // loss = (B, OUT)
            // tloss = (OUT, B)

            //gradGamma(OUT, IN)
            int batch_size = loss.Rank == 2 ? loss.Size(-2) : 1;
            if(batch_size == 1)
            {
                loss.Unsqueeze(0);
                InputCache.Unsqueeze(0);
            }
            Tensor transposedLoss = Tensor.Transpose(loss, 0, 1);

            if(device == Device.CPU)
            {
                Tensor gradW = Tensor.MatMul(transposedLoss, InputCache);
                Tensor gradB = Tensor.MatMul(transposedLoss, Tensor.Ones(batch_size));

                // Update the gradients
                gammaGrad += gradW / batch_size; // (out, in)
                betaGrad += gradB / batch_size; // (out)

                // Backpropagate the loss (batch_size, in)
                return Tensor.MatMul(loss, gamma).Squeeze(-2);
            }
            else
            {
                // dLoss w.r.t input
                Tensor output = Tensor.Zeros(batch_size, gamma.Size(-1));
                ComputeShader cs = DeepUnityMeta.DenseCS;

                ComputeBuffer transposedLossBuffer = new ComputeBuffer(transposedLoss.Count(), 4);
                transposedLossBuffer.SetData(transposedLoss.Data);
                cs.SetBuffer(1, "transposed_loss", transposedLossBuffer);

                ComputeBuffer inputCacheBuffer = new ComputeBuffer(InputCache.Count(), 4);
                inputCacheBuffer.SetData(InputCache.Data);
                cs.SetBuffer(1, "input", inputCacheBuffer);

                ComputeBuffer gammaGradBuffer = new ComputeBuffer(gammaGrad.Count(), 4);
                gammaGradBuffer.SetData(gammaGrad.Data);
                cs.SetBuffer(1, "gamma_grad", gammaGradBuffer);

                ComputeBuffer betaGradBuffer = new ComputeBuffer(betaGrad.Count(), 4);
                betaGradBuffer.SetData(betaGrad.Data);
                cs.SetBuffer(1, "beta_grad", betaGradBuffer);

                cs.SetInt("batch_size", batch_size);
                cs.SetInt("in_features", gamma.Size(-1));
                cs.SetInt("out_features", beta.Size(-1));
                cs.SetInt("input_rank", InputCache.Rank);

                cs.Dispatch(1,
                    (gammaGrad.Size(-1) + 31) / 32,
                    (gammaGrad.Size(-2) + 31) / 32,
                    1);

                gammaGradBuffer.GetData(gammaGrad.Data);
                betaGradBuffer.GetData(betaGrad.Data);

                transposedLossBuffer.Release();
                inputCacheBuffer.Release();
                gammaGradBuffer.Release();
                betaGradBuffer.Release();

                // Backpropagate the loss (batch_size, in)
                return Tensor.MatMulGPU(loss, gamma).Squeeze(-2);
            }
           
        }
    }

}