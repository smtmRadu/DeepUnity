using System;
using UnityEngine;
namespace DeepUnity
{
    [Serializable]
    public class Dense : Learnable, IModule
    {
        private Tensor InputCache { get; set; }

        /// <summary>
        /// Input: (B, H_in) or (H_in) for unbatched input.<br></br>
        /// Output: (B, H_out) or (H_out) for unbatched input.<br></br>
        /// where B = batch_size, H_in = in_features and H_out = out_features.
        /// </summary>
        /// <param name="in_features"></param>
        /// <param name="out_features"></param>
        /// <param name="init"></param>
        /// <param name="device">If Dense module is a middle layer, larger than (64, 64), it is recommended to run it on GPU. </param>
        public Dense(int in_features, int out_features, InitType init = InitType.Default, Device device = Device.CPU) : base(device)
        {
            if (in_features < 1)
                throw new ArgumentException("In_features cannot be less than 1.");
            if (out_features < 1)
                throw new ArgumentException("Out_features cannot be less than 1.");

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
                    throw new NotImplementedException("Unhandled initialization type!");
            }

            gammaGrad = Tensor.Zeros(out_features, in_features);
            betaGrad = Tensor.Zeros(out_features);
        }
        public Tensor Predict(Tensor input)
        {
            if (input.Size(-1) != gamma.Size(-1))
                throw new ShapeException($"Input ({input.Size(-1)}) shape in Dense layer must be ({gamma.Size(-1)}).");

            // input = (B, IN)
            // gamma = (OUT, IN)

            // out = (B, OUT)
            // input = (batch_size, in_features)
            // gamma = (out_features, in_features)
            // out = (out_features, batch_size)
            int batch_size = input.Rank == 2 ? input.Size(-2) : 1;

            if (device == Device.CPU)
            {
                if (input.Rank == 1)
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + beta;
                else
                    return Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);   
            }
            else
            {
                
                ComputeShader cs = DeepUnityMeta.DenseCS;

                ComputeBuffer inputBuffer = new ComputeBuffer(input.Count(), 4);
                inputBuffer.SetData(input.ToArray());
                cs.SetBuffer(0, "input", inputBuffer);

                ComputeBuffer tranposedGammaBuffer = new ComputeBuffer(gamma.Count(), 4);
                tranposedGammaBuffer.SetData(Tensor.Transpose(gamma, 0, 1).ToArray());
                cs.SetBuffer(0, "transposed_gamma", tranposedGammaBuffer);

                ComputeBuffer betaBuffer = new ComputeBuffer(beta.Count(), 4);
                betaBuffer.SetData(beta.ToArray());
                cs.SetBuffer(0, "beta", betaBuffer);

                ComputeBuffer outputBuffer = new ComputeBuffer(batch_size * beta.Size(-1), 4);
                // outputBuffer.SetData(zero_values); // we do not need this because the values are set (not added) to the rw structrured buffer.
                cs.SetBuffer(0, "output", outputBuffer);

                cs.SetInt("batch_size", batch_size);
                cs.SetInt("in_features", gamma.Size(-1));
                cs.SetInt("out_features", beta.Size(-1));
                cs.SetInt("gamma_rank", gamma.Rank);
                cs.SetInt("input_rank", input.Rank);

                cs.Dispatch(0,
                    (beta.Size(-1) + 32 - 1) / 32,
                    (batch_size + 32 - 1) / 32,
                    1);

                Tensor result = Tensor.Constant(outputBuffer);

                inputBuffer.Release();
                tranposedGammaBuffer.Release();
                betaBuffer.Release();
                outputBuffer.Release();

                return result.Reshape(input.Shape);
            }
        }
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);

            return Predict(input);           
        }
        public Tensor Backward(Tensor loss)
        {
            // input = (B, IN)
            // loss = (B, OUT)
            // tloss = (OUT, B)

            //gradGamma(OUT, IN)
            bool isBatched = loss.Rank == 2;
            int batch_size = isBatched ? loss.Size(-2) : 1;

            if (!isBatched)
            {
                loss = loss.Unsqueeze(0);
                InputCache = InputCache.Unsqueeze(0);
            }

            Tensor transposedLoss = Tensor.Transpose(loss, 0, 1);
            if (device == Device.CPU)
            {
                // compute the gradients
                gammaGrad += Tensor.MatMul(transposedLoss, InputCache) / batch_size;
                betaGrad += Tensor.Mean(transposedLoss, axis: 1);

            }
            else
            {
                // dLoss w.r.t input
                ComputeShader cs = DeepUnityMeta.DenseCS;

                ComputeBuffer transposedLossBuffer = new ComputeBuffer(transposedLoss.Count(), 4);
                transposedLossBuffer.SetData(transposedLoss.ToArray());
                cs.SetBuffer(1, "transposed_loss", transposedLossBuffer);

                ComputeBuffer inputCacheBuffer = new ComputeBuffer(InputCache.Count(), 4);
                inputCacheBuffer.SetData(InputCache.ToArray());
                cs.SetBuffer(1, "input", inputCacheBuffer);

                ComputeBuffer gammaGradBuffer = new ComputeBuffer(gammaGrad.Count(), 4);
                gammaGradBuffer.SetData(gammaGrad.ToArray());
                cs.SetBuffer(1, "gamma_grad", gammaGradBuffer);

                ComputeBuffer betaGradBuffer = new ComputeBuffer(betaGrad.Count(), 4);
                betaGradBuffer.SetData(betaGrad.ToArray());
                cs.SetBuffer(1, "beta_grad", betaGradBuffer);

                cs.SetInt("batch_size", batch_size);
                cs.SetInt("in_features", gamma.Size(-1));
                cs.SetInt("out_features", beta.Size(-1));
                cs.SetInt("input_rank", InputCache.Rank);

                cs.Dispatch(1,
                    (gammaGrad.Size(-1) + 31) / 32,
                    (gammaGrad.Size(-2) + 31) / 32,
                    1);

                gammaGrad = Tensor.Constant(gammaGradBuffer).Reshape(gammaGrad.Shape);
                betaGrad = Tensor.Constant(betaGradBuffer).Reshape(betaGrad.Shape);

                transposedLossBuffer.Release();
                inputCacheBuffer.Release();
                gammaGradBuffer.Release();
                betaGradBuffer.Release();

                
            }

            // Backpropagate the loss (batch_size, in)
            if (isBatched)
                return Tensor.MatMulGPU(loss, gamma);
            else
                return Tensor.MatMulGPU(loss, gamma).Squeeze(0);

        }
    }

}