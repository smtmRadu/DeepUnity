using System;
using UnityEngine;
namespace DeepUnity
{
    // https://www.youtube.com/watch?v=tMjdQLylyGI&t=602s
    // https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf (251 - 253)
    /// <summary>
    /// Input: <b>(B, H_in)</b> or <b>(H_in)</b> for unbatched input.<br></br>
    /// Output: <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input.<br></br>
    /// where B = batch_size, H_in = in_features and H_out = out_features.
    /// </summary>
    [Serializable]
    public class Dense : ILearnable, IModule
    {
        private Tensor InputCache { get; set; }

        [SerializeField] private Device device;
        [SerializeField] private Tensor weights;
        [SerializeField] private Tensor biases;
        [NonSerialized] private Tensor weigthsGrad;
        [NonSerialized] private Tensor biasesGrad;

        /// <summary>
        /// Input: <b>(B, H_in)</b> or <b>(H_in)</b> for unbatched input.<br></br>
        /// Output: <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input.<br></br>
        /// where B = batch_size, H_in = in_features and H_out = out_features.
        /// </summary>
        /// <param name="in_features">Input's last dimension value (H_in).</param>
        /// <param name="out_features">Output's last dimension value (H_out).</param>
        /// <param name="gamma_init">Initializer used for weights.</param>
        /// <param name="beta_init">Initializer used for biases.</param>
        /// <param name="device">Computation device used. Recommended <see cref="Device.GPU"/> for large <see cref="Dense"/> layers with <b>in_features</b> &amp; <b>out_features > 64</b>.</param>
        public Dense(int in_features, int out_features, InitType gamma_init = InitType.LeCun_Uniform, InitType beta_init = InitType.LeCun_Uniform, Device device = Device.CPU)
        {
            if (in_features < 1)
                throw new ArgumentException("In_features cannot be less than 1.");
            if (out_features < 1)
                throw new ArgumentException("Out_features cannot be less than 1.");

            this.device = device;
            weights = Initializer.InitializeParameter(new int[] {out_features, in_features}, in_features, out_features, gamma_init);
            biases = Initializer.InitializeParameter(new int[] { out_features}, in_features, out_features, gamma_init);
            weigthsGrad = Tensor.Zeros(weights.Shape);
            biasesGrad = Tensor.Zeros(biases.Shape);
        }
        public Tensor Predict(Tensor input)
        {
            if (input.Size(-1) != weights.Size(-1))
                throw new ShapeException($"Input features ({input.Size(-1)}) does not match with the Dense Layer features_num ({weights.Size(-1)}).");

            bool isBatched = input.Rank == 2;
            int batch_size = isBatched ? input.Size(-2) : 1;

            if (device == Device.CPU)
            {
                if (isBatched)
                    return Tensor.MatMul(input, Tensor.Transpose(weights, 0, 1)) + Tensor.Expand(Tensor.Unsqueeze(biases, 0), 0, batch_size);
                else
                    return Tensor.MatMul(input, Tensor.Transpose(weights, 0, 1)) + biases;
            }
            else
            {

                ComputeShader cs = DeepUnityMeta.DenseCS;

                ComputeBuffer inputBuffer = new ComputeBuffer(input.Count(), 4);
                inputBuffer.SetData(input.ToArray());
                cs.SetBuffer(0, "input", inputBuffer);

                ComputeBuffer tranposedGammaBuffer = new ComputeBuffer(weights.Count(), 4);
                tranposedGammaBuffer.SetData(Tensor.Transpose(weights, 0, 1).ToArray());
                cs.SetBuffer(0, "transposed_gamma", tranposedGammaBuffer);

                ComputeBuffer betaBuffer = new ComputeBuffer(biases.Count(), 4);
                betaBuffer.SetData(biases.ToArray());
                cs.SetBuffer(0, "beta", betaBuffer);

                ComputeBuffer outputBuffer = new ComputeBuffer(batch_size * biases.Size(-1), 4);
                // outputBuffer.SetData(zero_values); // we do not need this because the values are set (not added) to the rw structrured buffer.
                cs.SetBuffer(0, "output", outputBuffer);

                cs.SetInt("batch_size", batch_size);
                cs.SetInt("in_features", weights.Size(-1));
                cs.SetInt("out_features", biases.Size(-1));
                cs.SetInt("gamma_rank", weights.Rank);
                cs.SetInt("input_rank", input.Rank);

                cs.Dispatch(0,
                    (biases.Size(-1) + 31) / 32,
                    (batch_size + 31) / 32,
                    1);

                Tensor result = Tensor.Constant(outputBuffer);

                inputBuffer.Release();
                tranposedGammaBuffer.Release();
                betaBuffer.Release();
                outputBuffer.Release();

                if (isBatched)
                    return result.Reshape(batch_size, biases.Size(-1));
                else
                    return result.Reshape(biases.Size(-1));
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
                weigthsGrad.AssignAs(weigthsGrad + Tensor.MatMul(transposedLoss, InputCache) / batch_size);
                biasesGrad.AssignAs(biasesGrad + Tensor.Mean(transposedLoss, axis: 1));

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

                ComputeBuffer gammaGradBuffer = new ComputeBuffer(weigthsGrad.Count(), 4);
                gammaGradBuffer.SetData(weigthsGrad.ToArray());
                cs.SetBuffer(1, "gamma_grad", gammaGradBuffer);

                ComputeBuffer betaGradBuffer = new ComputeBuffer(biasesGrad.Count(), 4);
                betaGradBuffer.SetData(biasesGrad.ToArray());
                cs.SetBuffer(1, "beta_grad", betaGradBuffer);

                cs.SetInt("batch_size", batch_size);
                cs.SetInt("in_features", weights.Size(-1));
                cs.SetInt("out_features", biases.Size(-1));
                cs.SetInt("input_rank", InputCache.Rank);

                cs.Dispatch(1,
                    (weigthsGrad.Size(-1) + 31) / 32,
                    (weigthsGrad.Size(-2) + 31) / 32,
                    1);

                weigthsGrad.AssignAs(weigthsGrad + Tensor.Constant(gammaGradBuffer).Reshape(weigthsGrad.Shape));
                biasesGrad.AssignAs(biasesGrad + Tensor.Constant(betaGradBuffer).Reshape(biasesGrad.Shape));

                transposedLossBuffer.Release();
                inputCacheBuffer.Release();
                gammaGradBuffer.Release();
                betaGradBuffer.Release();
            }


            // Backpropagate the loss (batch_size, in)
            if (isBatched)
                return Tensor.MatMulGPU(loss, weights);
            else
                return Tensor.MatMulGPU(loss, weights).Squeeze(0);

        }
        
        


        public void SetDevice(Device device) { this.device = device; }
        public int ParametersCount()
        {
            return weights.Count() + biases.Count();
        }
        public Tensor[] Parameters()
        {
            return new Tensor[] { weights, biases };
        }
        
        public Tensor[] Gradients()
        {
            if (weigthsGrad == null)
                OnAfterDeserialize();
            return new Tensor[] { weigthsGrad, biasesGrad };
        }
        public object Clone()
        {
            var dense = new Dense(1, 1, device: this.device);
            dense.weights = (Tensor)this.weights.Clone();
            dense.biases = (Tensor)this.biases.Clone();
            return dense;
        }


        public virtual void OnBeforeSerialize()
        {

        }
        
        public virtual void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (weights.Shape == null)
                return;

            if (weights.Shape.Length == 0)
                return;

            // do not check if gamma is != null...
            this.weigthsGrad = Tensor.Zeros(weights.Shape);
            this.biasesGrad = Tensor.Zeros(biases.Shape);

        }
    }

}