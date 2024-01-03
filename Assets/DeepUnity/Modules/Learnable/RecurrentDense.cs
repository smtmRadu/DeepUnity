using UnityEngine;
using System;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEditor.Search;
using System.Linq;
using System.Threading.Tasks;
namespace DeepUnity
{
    // The implementation is adapted for this framework so it works for our models
    // https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html#torch.nn.RNNCell
    // Most of the cases we do not care about the rest of the hidden states, only the last one is important.
    [Serializable]
    public class RecurrentDense : ILearnable, IModule
    {
        [SerializeField] private Device device;
        [SerializeField] private Tensor weights;
        [SerializeField] private Tensor biases;
        [SerializeField] private Tensor r_weights;
        [SerializeField] private Tensor r_biases;

        [NonSerialized] private Tensor weightsGrad;
        [NonSerialized] private Tensor biasesGrad;
        [NonSerialized] private Tensor r_weightsGrad;
        [NonSerialized] private Tensor r_biasesGrad;

        [NonSerialized] private Stack<Tensor> InputCache;
        [NonSerialized] private Stack<Tensor> HiddenCache;
        [NonSerialized] private Stack<Tanh> Activations;


        /// <summary>
        /// <b>Usually needs a Reshape layer behind.</b> <br></br>
        /// <b>Outputs the last hidden state, so do not put 2 Recurrent Dense in a row.</b> <br></br>
        /// <b>It uses Tanh non-linearity.</b> <br></br>
        /// <br></br>
        /// Input:  <b>(B, L, H_in)</b> or <b>(L, H_in)</b> for unbatched input.<br></br>
        /// Output:  <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input. <br></br>
        /// <br></br>
        /// where B = batch_size, L = sequence_length, H_in = in_features, H_out = out_features.
        /// </summary>
        /// <param name="in_features"></param>
        /// <param name="out_features"></param>
        /// <param name="device"></param>
        public RecurrentDense(int in_features, int out_features, Device device = Device.CPU)
        {
            this.device = device;
            InputCache = new Stack<Tensor>();
            HiddenCache = new Stack<Tensor>();
            Activations = new Stack<Tanh>();

            float range = MathF.Sqrt(1f / out_features);
            weights = Tensor.RandomRange((-range, range), out_features, in_features);
            biases = Tensor.RandomRange((-range, range), out_features);
            r_weights = Tensor.RandomRange((-range, range), out_features, out_features);
            r_biases = Tensor.RandomRange((-range, range), out_features);
            weightsGrad = Tensor.Zeros(weights.Shape);
            biasesGrad = Tensor.Zeros(biases.Shape);
            r_weightsGrad = Tensor.Zeros(r_weights.Shape);
            r_biasesGrad = Tensor.Zeros(r_biases.Shape);
        }
        private RecurrentDense() { }


        public Tensor Predict(Tensor input)
        {
            // input is (B, L, H)
            // or (L, H)
            if(input.Size(-1) != weights.Size(-1))
            {
                throw new ShapeException($"Input features ({input.Size(-1)}) does not match with the Dense Layer features_num ({weights.Size(-1)}).");

            }
            if (input.Rank != 3 && input.Rank != 2)
            {
                throw new ShapeException($"Input must be of shape (B, L, H) or (L, H) for unabatches input, and input received has shape ({input.Shape.ToCommaSeparatedString()}).");
            }
            bool isBatched = input.Rank == 3;
            int batch_size = isBatched ? input.Size(0) : 1;

            Tensor expandedBiases = batch_size == 1 ? biases : Tensor.Expand(Tensor.Unsqueeze(biases, 0), 0, batch_size);
            Tensor expandedRecurrentBiases = batch_size == 1 ? r_biases : Tensor.Expand(Tensor.Unsqueeze(r_biases, 0), 0, batch_size);

            Tensor[] sequences = isBatched ? input.Split(1, 1) : input.Split(0, 1);
            Tensor h = isBatched ? Tensor.Zeros(batch_size, biases.Size(-1)) : Tensor.Zeros(biases.Size(-1));

            Tanh tanh = new Tanh();
            foreach (var elem in sequences)
            {
                // x has Shape (B, 1, H) or (1, H) => need to be (B, H) or (H)
                Tensor x = isBatched ? elem.Squeeze(1) : elem.Squeeze(0);
              
                Tensor linear;
                if(device == Device.CPU)
                    linear = Linear(x, weights, biases, isBatched, batch_size) + Linear(h, r_weights, r_biases, isBatched, batch_size);
                else
                    linear = Tensor.MatMulGPU(x, Tensor.Transpose(weights, 0, 1)) + expandedBiases +
                        Tensor.MatMulGPU(h, Tensor.Transpose(r_weights, 0, 1)) + expandedRecurrentBiases;

                h = tanh.Predict(linear);
            }

            return h;
        }
        public Tensor Forward(Tensor input)
        {
            // input is (B, L, H)
            // or (L, H)
            if (input.Size(-1) != weights.Size(-1))
            {
                throw new ShapeException($"Input features ({input.Size(-1)}) does not match with the Dense Layer input features_num ({weights.Size(-1)}).");

            }
            if (input.Rank != 3 && input.Rank != 2)
            {
                throw new ShapeException($"Input must be of shape (B, L, H) or (L, H) for unabatches input, and input received has shape ({input.Shape.ToCommaSeparatedString()}).");
            }
            bool isBatched = input.Rank == 3;
            int batch_size = isBatched ? input.Size(0) : 1;

            Tensor expandedBiases = batch_size == 1 ? biases : Tensor.Expand(Tensor.Unsqueeze(biases, 0), 0, batch_size);
            Tensor expandedRecurrentBiases = batch_size == 1 ? r_biases : Tensor.Expand(Tensor.Unsqueeze(r_biases, 0), 0, batch_size);

            Tensor[] sequences = isBatched ? input.Split(1, 1) : input.Split(0, 1);
            Tensor h = isBatched ? Tensor.Zeros(batch_size, biases.Size(-1)) : Tensor.Zeros(biases.Size(-1));

           
            foreach (var elem in sequences)
            {
                Activations.Push(new Tanh());

                // x has Shape (B, 1, H) or (1, H) => need to be (B, H) or (H)
                Tensor x = isBatched ? elem.Squeeze(1) : elem.Squeeze(0);
                
                Tensor linear;
                if (device == Device.CPU)
                    linear = Linear(x, weights, biases, isBatched, batch_size) + Linear(h, r_weights, r_biases, isBatched, batch_size);
                else
                    linear = Tensor.MatMulGPU(x, Tensor.Transpose(weights, 0, 1)) + expandedBiases +
                        Tensor.MatMulGPU(h, Tensor.Transpose(r_weights, 0, 1)) + expandedRecurrentBiases;

                h = Activations.Peek().Forward(linear);
                InputCache.Push(x.Clone() as Tensor);
                HiddenCache.Push(h.Clone() as Tensor);
            }

            return h;
         
        }
        public Tensor Backward(Tensor dLdH)
        {
            bool isBatched = dLdH.Rank == 2;
            int batch_size = isBatched ? dLdH.Size(-2) : 1;
     
            // We go in reverse for each element
            while(InputCache.Count > 0)
            {
                Tensor dLdLinear = Activations.Peek().Backward(dLdH);

                if (device == Device.CPU)
                {
                    Tensor weights_grad;
                    Tensor biases_grad;
                    Tensor r_weights_grad;
                    Tensor r_biases_grad;
                    int hin = weights.Size(-1);
                    int hout = biases.Size(-1);
                    ComputeGradients(InputCache.Peek(), dLdLinear, isBatched, batch_size, hin, hout, out weights_grad, out biases_grad);
                    ComputeGradients(HiddenCache.Peek(), dLdLinear, isBatched, batch_size, hout, hout, out r_weights_grad, out r_biases_grad);

                    Tensor.CopyTo(weightsGrad + weights_grad, weightsGrad);
                    Tensor.CopyTo(biasesGrad + biases_grad, biasesGrad);
                    Tensor.CopyTo(r_weightsGrad + r_weights_grad, r_weightsGrad);
                    Tensor.CopyTo(r_biasesGrad + r_biases_grad, r_biasesGrad);

                    InputCache.Pop();
                    HiddenCache.Pop();
                    Activations.Pop();

                    if (InputCache.Count == 0) //dLdx
                        return Tensor.MatMul(dLdLinear, weights);
                    else
                        dLdH = Tensor.MatMul(dLdLinear, r_weights);
                }
                else
                {
                    Tensor transposed_dLdLinear = isBatched ?
                                                Tensor.Transpose(dLdLinear, 0, 1) :
                                                Tensor.Transpose(dLdLinear.Unsqueeze(0), 0, 1);

                    Tensor.CopyTo(weightsGrad + Tensor.MatMulGPU(transposed_dLdLinear, InputCache.Peek()) / batch_size, weightsGrad);
                    Tensor.CopyTo(biasesGrad + Tensor.Mean(dLdLinear, axis: 0), biasesGrad);
                    Tensor.CopyTo(r_weightsGrad + Tensor.MatMulGPU(transposed_dLdLinear, HiddenCache.Peek()) / batch_size, r_weightsGrad);
                    Tensor.CopyTo(r_biasesGrad + Tensor.Mean(dLdLinear, axis: 0), r_biasesGrad);

                    InputCache.Pop();
                    HiddenCache.Pop();
                    Activations.Pop();

                    if (InputCache.Count == 0)//dLdx
                        return Tensor.MatMulGPU(dLdLinear, weights);
                    else
                        dLdH = Tensor.MatMulGPU(dLdLinear, r_weights);
                }      
            }
            throw new Exception("There is no way it will reach this point");
        }



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

        private void ComputeGradients(Tensor x, Tensor loss, bool isBatched, int B_size, int H_in, int H_out, out Tensor weights_grad, out Tensor biases_grad)
        {
            Tensor wg = Tensor.Zeros(H_out, H_in);
            Tensor bg = Tensor.Zeros(H_out);

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





        public int ParametersCount()
        {
            return weights.Count() + biases.Count() + r_weights.Count() + r_biases.Count();
        }
        public Parameter[] Parameters()
        {
            if (weightsGrad == null)
                OnAfterDeserialize();

            var w = new Parameter(weights, weightsGrad);
            var b = new Parameter(biases, biasesGrad);
            var rw = new Parameter(r_weights, r_weightsGrad);
            var rb = new Parameter(r_biases, r_biasesGrad);

            return new Parameter[] { w, b, rw, rb };
        }

        public void SetDevice(Device device) { this.device = device; }

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

            InputCache = new Stack<Tensor>();
            HiddenCache = new Stack<Tensor>();
            Activations = new Stack<Tanh>();

            this.weightsGrad = Tensor.Zeros(weights.Shape);
            this.biasesGrad = Tensor.Zeros(biases.Shape);

            r_weightsGrad = Tensor.Zeros(r_weights.Shape);  
            r_biasesGrad = Tensor.Zeros(r_biases.Shape);    

        }


        public object Clone()
        {
            var rnncell = new RecurrentDense();
            rnncell.InputCache = new Stack<Tensor>();
            rnncell.HiddenCache = new Stack<Tensor>();
            rnncell.Activations = new Stack<Tanh>();
            rnncell.weights = (Tensor)this.weights.Clone();
            rnncell.biases = (Tensor)this.biases.Clone();
            rnncell.r_weights = (Tensor)this.r_weights.Clone();
            rnncell.r_biases = (Tensor)this.r_biases.Clone();
            rnncell.weightsGrad = (Tensor)this.weightsGrad.Clone();
            rnncell.biasesGrad = (Tensor)this.biasesGrad.Clone();
            rnncell.r_weightsGrad = (Tensor)this.r_weightsGrad.Clone();
            rnncell.r_biasesGrad = (Tensor)this.r_biasesGrad.Clone();
            return rnncell;
        }
    }

}


