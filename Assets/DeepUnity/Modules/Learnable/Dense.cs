using System;
using UnityEngine;
using System.Threading.Tasks;
using Unity.VisualScripting;

namespace DeepUnity.Modules
{
    // DO NOT TRY TO UNDERSTAND Dense implementation because was over-optimized (cause is the most used layer) and the code became unreadable
    // also the gradients are initialized on the fly only when needed because this layer is used only for inference sometimes.
    // https://www.youtube.com/watch?v=tMjdQLylyGI&t=602s
    // https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf (251 - 253)
    /// <summary>
    /// Input: <b>(B, H_in)</b>, <b>(H_in)</b> or <b>(B, L, H_in)</b>, <b>(L, H_in)</b> for sequential input.<br></br>
    /// Output: <b>(B, H_out)</b>, <b>(H_out)</b> or <b>(B, L, H_out)</b>, <b>(L, H_out)</b> for sequential input.<br></br>
    /// where B = batch_size, L = sequence_length, H_in = in_features and H_out = out_features.
    /// </summary>
    [Serializable]
    public class Dense : ILearnable, IModule
    {
        public Device Device { get; set; } = Device.CPU;
        public bool RequiresGrad { get; set; } = true;
        private Tensor InputCache { get; set; }

        [SerializeField] private bool bias = true;
        [SerializeField] public Tensor weights;
        [SerializeField] public Tensor biases;
        [NonSerialized]  private Tensor weightsGrad;
        [NonSerialized]  private Tensor biasesGrad;


        /// <summary>
        /// Input: <b>(B, H_in)</b>, <b>(H_in)</b> or <b>(B, L, H_in)</b>, <b>(L, H_in)</b> for sequential input.<br></br>
        /// Output: <b>(B, H_out)</b>, <b>(H_out)</b> or <b>(B, L, H_out)</b>, <b>(L, H_out)</b> for sequential input.<br></br>
        /// where B = batch_size, L = sequence_length, H_in = in_features and H_out = out_features.
        /// </summary>
        /// <param name="in_features">Input's last dimension value (H_in).</param>
        /// <param name="out_features">Output's last dimension value (H_out).</param>
        /// <param name="weight_init">Initializer used for weights.</param>
        /// <param name="bias_init">Initializer used for biases.</param>
        /// <param name="device">Computation device used. Recommended <see cref="Device.GPU"/> for large <see cref="Dense"/> layers with <b>in_features</b> &amp; <b>out_features > 64</b>.</param>
        public Dense(int in_features, int out_features, bool bias = true, InitType weight_init = InitType.LeCun_Uniform, InitType bias_init = InitType.LeCun_Uniform, Device device = default)
        {
            if (in_features < 1)
                throw new ArgumentException("In_features cannot be less than 1.");
            if (out_features < 1)
                throw new ArgumentException("Out_features cannot be less than 1.");

            this.bias = bias;
            this.Device = device;
            weights = Parameter.Create(new int[] { out_features, in_features }, in_features, out_features, weight_init);
            weightsGrad = null; //Tensor.Zeros(weights.Shape);

            if(bias)
            {
                biases = Parameter.Create(new int[] { out_features }, in_features, out_features, bias_init);
                biasesGrad = null; // Tensor.Zeros(biases.Shape);
            }
           
        }
        private Dense() { }
        public Tensor Predict(Tensor input)
        {
            if (input.Rank > 3)
                throw new ArgumentException($"Input must have the shape as (H_in), (B, H_in), (L, H_in) or (B, L, H_in), and the received input is ({input.Shape.ToCommaSeparatedString()})");

            if (input.Size(-1) != weights.Size(-1))
                throw new ShapeException($"Input features ({input.Size(-1)}) does not match with the Dense Layer features_num ({weights.Size(-1)}).");


            if (input.Rank == 3)
            {
                Tensor y = Tensor.BatchedMatMul(input, weights.Transpose(0, 1).Unsqueeze(0).Expand(0, input.Size(0)), Device);

                if (bias)
                    y += ExpandedBiases(y.Shape);

                return y;              
            }

           
            bool isBatched = input.Rank == 2;
            int batch_size = isBatched ? input.Size(-2) : 1;

            if (Device == Device.CPU)
            {
                return Linear(input, weights, biases, isBatched, batch_size); // faster inference x10 with this... instead of using MatMul and Transpose methods..
            }
            else
            {
                int H_in = weights.Size(-1);
                int H_out = weights.Size(-2);
                ComputeShader cs = DeepUnityMeta.DenseCS;

                ComputeBuffer inputBuffer = new ComputeBuffer(input.Count(), 4);
                inputBuffer.SetData(input.ToArray());
                cs.SetBuffer(0, "input", inputBuffer);

                ComputeBuffer weightsBuffer = new ComputeBuffer(weights.Count(), 4);
                weightsBuffer.SetData(weights.ToArray());
                cs.SetBuffer(0, "gamma", weightsBuffer);

                ComputeBuffer biasesBuffer = null;
                if (bias)
                {
                    biasesBuffer = new ComputeBuffer(bias ? biases.Count() : H_out, 4);
                    biasesBuffer.SetData(bias ? biases.ToArray() : new float[H_out]);
                    cs.SetBuffer(0, "beta", biasesBuffer);
                }
                

                ComputeBuffer outputBuffer = new ComputeBuffer(batch_size * H_out, 4);
                // outputBuffer.SetData(zero_values); // we do not need this because the values are set (not added) to the rw structrured buffer.
                cs.SetBuffer(0, "output", outputBuffer);

                cs.SetInt("batch_size", batch_size);
                cs.SetInt("in_features", H_in);
                cs.SetInt("out_features", H_out);
                cs.SetBool("use_bias", bias);

                cs.Dispatch(0,
                    (H_out + 31) / 32,
                    (batch_size + 31) / 32,
                    1);

                Tensor result = isBatched ?
                    Tensor.Constant(outputBuffer, batch_size, H_out) :
                    Tensor.Constant(outputBuffer, H_out);

                inputBuffer.Release();
                weightsBuffer.Release();
                biasesBuffer?.Release();
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
            if (loss.Size(-1) != weights.Size(0))
                throw new ArgumentException($"Hidden features of the loss ({loss.Size(-1)}) doesn't correspond to the hidden features returned by the dense layer ({weights.Size(0)}).");

            if (weightsGrad == null)
            {
                weightsGrad = Tensor.Zeros(weights.Shape);
                if (bias)
                    biasesGrad = Tensor.Zeros(biases.Shape);
            }

            if (loss.Rank == 3)
            {
                if(RequiresGrad)
                {
                    Tensor wsGrad = Tensor.BatchedMatMul(loss.Transpose(1, 2), InputCache);
                    wsGrad /= loss.Size(0); // divide by batch size
                    wsGrad = wsGrad.Sum(0);
                    Tensor.CopyTo(wsGrad, weightsGrad);

                    if (bias)
                    {
                        Tensor bGrad = loss.Mean(0).Sum(0);
                        Tensor.CopyTo(bGrad, biasesGrad);
                    }

                }

                Tensor inputGrad = Tensor.BatchedMatMul(loss, weights.Unsqueeze(0).Expand(0, loss.Size(0)));
                return inputGrad;
            }
            // input = (B, IN)
            // loss = (B, OUT)
            // tloss = (OUT, B)

            //gradGamma(OUT, IN)
            bool isBatched = loss.Rank == 2;
            int batch_size = isBatched ? loss.Size(-2) : 1;

            if(RequiresGrad)
            {
                if (Device == Device.CPU)
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
                    if (bias)
                        Tensor.CopyTo(biasesGrad + biases_grad, biasesGrad);
                }
                else
                {
                    int H_in = weights.Size(-1);
                    int H_out = weights.Size(-2);

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

                    ComputeBuffer biasesGradBuffer = null;
                    if (bias)
                    {
                        biasesGradBuffer = new ComputeBuffer(bias ? biasesGrad.Count() : H_out, 4);
                        biasesGradBuffer.SetData(bias ? biasesGrad.ToArray() : new float[H_out]);
                        cs.SetBuffer(1, "beta_grad", biasesGradBuffer);
                    }
                    

                    cs.SetInt("batch_size", batch_size);
                    cs.SetInt("in_features", H_in);
                    cs.SetInt("out_features", H_out);

                    cs.Dispatch(1,
                        (H_in + 31) / 32,
                        (H_out + 31) / 32,
                        1);

                    Tensor.CopyTo(Tensor.Constant(weightsGradBuffer, weightsGrad.Shape), weightsGrad);
                    if (bias)
                        Tensor.CopyTo(Tensor.Constant(biasesGradBuffer, biases.Shape), biasesGrad);

                    lossBuffer.Release();
                    inputCacheBuffer.Release();
                    weightsGradBuffer.Release();
                    biasesGradBuffer?.Release();
                }

            }

            return Tensor.MatMul(loss, weights, Device);
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
                Parallel.For(0, B_size, (Action<int>)(b =>
                {
                    for (int hout = 0; hout < H_out; hout++)
                    {
                        float sum = this.bias ? biases[hout] : 0f;
                        for (int hin = 0; hin < H_in; hin++)
                        {
                            sum += x[b, hin] * weights[hout, hin];
                        }
                        y[b, hout] = sum;
                    }
                }));
            }
            else
            {
                //Tests show that even only on 64 hid units it might perform better without multithread, so keep it like this indefinetely..
                Parallel.For(0, H_out, (Action<int>)(hout =>
                {
                    float sum = this.bias ? biases[hout] : 0f;
                    for (int hin = 0; hin < H_in; hin++)
                    {
                        sum += x[hin] * weights[hout, hin];
                    }
                    y[hout] = sum;
                }));
            }

            return y;
        }
        private void ComputeGradients(Tensor x, Tensor loss, bool isBatched, int B_size, out Tensor weights_grad, out Tensor biases_grad)
        {
            int H_out = weights.Size(-2);
            int H_in = weights.Size(-1);

            Tensor wg = Tensor.Zeros(H_out, H_in);
            Tensor bg = bias ? Tensor.Zeros(H_out) : null;

            // lossT * input = (H_out, B) * (B, H_in)
            if (isBatched)
            {
                Parallel.For(0, H_in, (Action<int>)(hin =>
                {
                    for (int hout = 0; hout < H_out; hout++)
                    {
                        float mm = 0f;

                        for (int b = 0; b < B_size; b++)
                            mm += loss[b, hout] * x[b, hin];

                        wg[hout, hin] = mm / B_size;
                    }

                    if (hin == 0 && this.bias)  //// here check for bias usage..............
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

                }));
            }
            else
            {
                Parallel.For(0, H_out, (Action<int>)(hout =>
                {
                    for (int hin = 0; hin < H_in; hin++)
                    {
                        wg[hout, hin] = x[hin] * loss[hout];
                    }
                    if(this.bias)
                        bg[hout] = loss[hout];
                }));
            }

            weights_grad = wg;
            biases_grad = bg;
        }
        private Tensor ExpandedBiases(int[] forShape)
        {
            if (forShape.Length == 1)
                return biases;

            var expBiases = Tensor.Zeros(forShape);

            if (forShape.Length == 3)
                for (int i = 0; i < forShape[0]; i++)
                {
                    for (int j = 0; j < forShape[1]; j++)
                    {
                        for (int k = 0; k < forShape[2]; k++)
                        {
                            expBiases[i, j, k] = biases[k];
                        }
                    }
                }
            else if (forShape.Length == 2)
                for (int j = 0; j < forShape[1]; j++)
                {
                    for (int k = 0; k < forShape[2]; k++)
                    {
                        expBiases[j, k] = biases[k];
                    }
                }

            return expBiases;
        }


        public Parameter[] Parameters()
        {
            if (weightsGrad == null)
            {
                weightsGrad = Tensor.Zeros(weights.Shape);
                if (bias)
                    biasesGrad = Tensor.Zeros(biases.Shape);
            }

            return bias ? 
                new Parameter[] { new Parameter(weights, weightsGrad), new Parameter(biases, biasesGrad) } : 
                new Parameter[] { new Parameter(weights, weightsGrad) };
        }
        public object Clone()
        {
            var dense = new Dense();
            dense.Device = Device;
            dense.RequiresGrad = RequiresGrad;
            dense.bias = bias;



            dense.weights = (Tensor)weights.Clone();     
            if(weightsGrad != null)
                dense.weightsGrad = (Tensor)weightsGrad.Clone();

            if(bias)
            {
                dense.biases = (Tensor)biases.Clone();
                if (biasesGrad != null) 
                    dense.biasesGrad = (Tensor)biasesGrad.Clone();
            }
            
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

            if(bias)
                biasesGrad = Tensor.Zeros(biases.Shape);

        }
    }

}