using System;
using UnityEngine;
using System.Threading.Tasks;
namespace DeepUnity.Layers
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
        [NonSerialized] private Tensor weightsGrad;
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
            weights = Initializer.CreateParameter(new int[] { out_features, in_features }, in_features, out_features, gamma_init);
            biases = Initializer.CreateParameter(new int[] { out_features }, in_features, out_features, beta_init);
            weightsGrad = Tensor.Zeros(weights.Shape);
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
                return Linear(input, weights, biases, isBatched, batch_size); // faster inference x10 with this...
            }
            else
            {
                int H_in = weights.Size(-1);
                int H_out = biases.Size(-1);
                ComputeShader cs = DeepUnityMeta.DenseCS;

                ComputeBuffer inputBuffer = new ComputeBuffer(input.Count(), 4);
                inputBuffer.SetData(input.ToArray());
                cs.SetBuffer(0, "input", inputBuffer);

                ComputeBuffer weightsBuffer = new ComputeBuffer(weights.Count(), 4);
                weightsBuffer.SetData(weights.ToArray());
                cs.SetBuffer(0, "gamma", weightsBuffer);

                ComputeBuffer biasesBuffer = new ComputeBuffer(biases.Count(), 4);
                biasesBuffer.SetData(biases.ToArray());
                cs.SetBuffer(0, "beta", biasesBuffer);

                ComputeBuffer outputBuffer = new ComputeBuffer(batch_size * H_out, 4);
                // outputBuffer.SetData(zero_values); // we do not need this because the values are set (not added) to the rw structrured buffer.
                cs.SetBuffer(0, "output", outputBuffer);

                cs.SetInt("batch_size", batch_size);
                cs.SetInt("in_features", H_in);
                cs.SetInt("out_features", H_out);

                cs.Dispatch(0,
                    (H_out + 31) / 32,
                    (batch_size + 31) / 32,
                    1);

                Tensor result = isBatched ?
                    Tensor.Constant(outputBuffer, batch_size, H_out) :
                    Tensor.Constant(outputBuffer, H_out);

                inputBuffer.Release();
                weightsBuffer.Release();
                biasesBuffer.Release();
                outputBuffer.Release();

                return result;
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

            if (device == Device.CPU)
            {
                // Benchmark : 0.93s avg (This method was replaced due to performance benchmark)
                // Tensor transposedLoss = isBatched ?
                //         Tensor.Transpose(loss, 0, 1) :
                //         Tensor.Transpose(loss.Unsqueeze(0), 0, 1);
                // 
                // Tensor.CopyTo(weightsGrad + Tensor.MatMul(transposedLoss, InputCache) / batch_size, weightsGrad);
                // Tensor.CopyTo(biasesGrad + Tensor.Mean(loss, axis: 0), biasesGrad);


                // Benchmark : 0.72s avg
                Tensor weights_grad;
                Tensor biases_grad;
                ComputeGradients(InputCache, loss, isBatched, batch_size, out weights_grad, out biases_grad);

                Tensor.CopyTo(weightsGrad + weights_grad, weightsGrad);
                Tensor.CopyTo(biasesGrad + biases_grad, biasesGrad);
            }
            else
            {
                int H_in = weights.Size(-1);
                int H_out = biases.Size(-1);

                // dLoss w.r.t theta
                ComputeShader cs = DeepUnityMeta.DenseCS;

                ComputeBuffer lossBuffer = new ComputeBuffer(loss.Count(), 4);
                lossBuffer.SetData(loss.ToArray());
                cs.SetBuffer(1, "loss", lossBuffer);

                ComputeBuffer inputCacheBuffer = new ComputeBuffer(InputCache.Count(), 4);
                inputCacheBuffer.SetData(InputCache.ToArray());
                cs.SetBuffer(1, "input", inputCacheBuffer);

                ComputeBuffer weightsGradBuffer = new ComputeBuffer(weightsGrad.Count(), 4);
                weightsGradBuffer.SetData(weightsGrad.ToArray());
                cs.SetBuffer(1, "gamma_grad", weightsGradBuffer);

                ComputeBuffer biasesGradBuffer = new ComputeBuffer(biasesGrad.Count(), 4);
                biasesGradBuffer.SetData(biasesGrad.ToArray());
                cs.SetBuffer(1, "beta_grad", biasesGradBuffer);

                cs.SetInt("batch_size", batch_size);
                cs.SetInt("in_features", H_in);
                cs.SetInt("out_features", H_out);

                cs.Dispatch(1,
                    (H_in + 31) / 32,
                    (H_out + 31) / 32,
                    1);

                Tensor.CopyTo(Tensor.Constant(weightsGradBuffer, weightsGrad.Shape), weightsGrad);
                Tensor.CopyTo(Tensor.Constant(biasesGradBuffer, biases.Shape), biasesGrad);

                lossBuffer.Release();
                inputCacheBuffer.Release();
                weightsGradBuffer.Release();
                biasesGradBuffer.Release();

                
            }

            return Tensor.MatMul(loss, weights, device);
        }

        /// <summary>
        /// Implemented only to make the inference even more efficient rather than using Matmul..
        /// </summary>
        /// <param name="x"></param>
        /// <param name="weights"></param>
        /// <param name="biases"></param>
        /// <returns></returns>
        private Tensor Linear(Tensor x, Tensor weights, Tensor biases, bool isBatched, int B_size)
        {
            // x = (B, H_in) or (H_in)
            // W = (H_out, H_in)
            // B = (H_out)
            int H_out = weights.Size(-2);
            int H_in = weights.Size(-1);

            Tensor y = isBatched ? Tensor.Zeros(B_size, H_out) : Tensor.Zeros(H_out);

            //   (B, H_in) * (H_in, H_out)
            //  (n, m) * (m, p) = (n, p)
            if (isBatched)
            {
                Parallel.For(0, B_size, b =>
                {
                    for (int hout = 0; hout < H_out; hout++)
                    {
                        float sum = 0f;
                        for (int hin = 0; hin < H_in; hin++)
                        {
                            sum += x[b, hin] * weights[hout, hin];
                        }
                        y[b, hout] = sum + biases[hout];
                    }
                });
            }
            else
            {
                //Tests show that even only on 64 hid units it might perform better without multithread, so keep it like this indefinetely..
                Parallel.For(0, H_out, hout =>
                {
                    float sum = 0f;
                    for (int hin = 0; hin < H_in; hin++)
                    {
                        sum += x[hin] * weights[hout, hin];
                    }
                    y[hout] = sum + biases[hout];
                });
            }

            return y;
        }
        private void ComputeGradients(Tensor x, Tensor loss, bool isBatched, int B_size, out Tensor weights_grad, out Tensor biases_grad)
        {
            int H_out = weights.Size(-2);
            int H_in = weights.Size(-1);

            Tensor wg = Tensor.Zeros(H_out, H_in);
            Tensor bg = Tensor.Zeros(H_out);


            // lossT * input = (H_out, B) * (B, H_in)
            if (isBatched)
            {
                Parallel.For(0, H_in, hin =>
                {
                    for (int hout = 0; hout < H_out; hout++)
                    {
                        float mm = 0f;

                        for (int b = 0; b < B_size; b++)
                            mm += loss[b, hout] * x[b, hin];

                        wg[hout, hin] = mm / B_size;
                    }

                    if (hin == 0)
                    {
                        for (int hout = 0; hout < H_out; hout++)
                        {
                            float loss_mean = 0f;
                            for (int b = 0; b < B_size; b++)
                            {
                                loss_mean += loss[b, hout];
                            }
                            bg[hout] = loss_mean / B_size;
                        }
                    }

                });
            }
            else
            {
                Parallel.For(0, H_out, hout =>
                {
                    for (int hin = 0; hin < H_in; hin++)
                    {
                        wg[hout, hin] = x[hin] * loss[hout];
                    }
                    bg[hout] = loss[hout];
                });
            }

            weights_grad = wg;
            biases_grad = bg;
        }
        public void SetDevice(Device device) { this.device = device; }
        public int ParametersCount()
        {
            return weights.Count() + biases.Count();
        }
        public Parameter[] Parameters()
        {
            if (weightsGrad == null)
                OnAfterDeserialize();

            var w = new Parameter(weights, weightsGrad);
            var b = new Parameter(biases, biasesGrad);

            return new Parameter[] { w, b };
        }

        public object Clone()
        {
            var dense = new Dense(1, 1, device: device);
            dense.weights = (Tensor)weights.Clone();
            dense.biases = (Tensor)biases.Clone();
            dense.weightsGrad = (Tensor)weightsGrad.Clone();
            dense.biasesGrad = (Tensor)biasesGrad.Clone();
            return dense;
        }


        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (weights.Shape == null)
                return;

            if (weights.Shape.Length == 0)
                return;

            // do not check if gamma is != null...
            weightsGrad = Tensor.Zeros(weights.Shape);
            biasesGrad = Tensor.Zeros(biases.Shape);

        }
    }

}