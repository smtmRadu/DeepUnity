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


    /// <summary>
    /// <b>Usually needs a Reshape layer behind.</b> <br></br>
    /// <br></br>
    /// Input:  <b>(B, L, H_in)</b> or <b>(L, H_in)</b> for unbatched input.<br></br>
    /// Output:  <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input when <paramref name="on_forward"/> == <see cref="HiddenStates.ReturnLast"/> or  <br></br>
    ///  <b>(B, L, H_out)</b> or <b>(L, H_out)</b> for unbatched input when <paramref name="on_forward"/> == <see cref="HiddenStates.ReturnAll"/> <br></br>
    ///  <br></br>
    /// where B = batch_size, L = sequence_length, H_in = input_size, H_out = hidden_size.
    /// </summary>
    /// https://mmuratarat.github.io/2019-02-07/bptt-of-rnn
    [Serializable]
    public class RNNCell : ILearnable, IModule
    {
        [SerializeField] private NonLinearity nonlinearity;
        [SerializeField] private HiddenStates onReturn;
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
        [NonSerialized] private Stack<Activation> ActivationCache;


        /// <summary>
        /// <b>Usually needs a Reshape layer behind.</b> <br></br>
        /// <br></br>
        /// Input:  <b>(B, L, H_in)</b> or <b>(L, H_in)</b> for unbatched input.<br></br>
        /// Output:  <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input when <paramref name="on_forward"/> == <see cref="HiddenStates.ReturnLast"/> or  <br></br>
        ///  <b>(B, L, H_out)</b> or <b>(L, H_out)</b> for unbatched input when <paramref name="on_forward"/> == <see cref="HiddenStates.ReturnAll"/> <br></br>
        ///  <br></br>
        /// where B = batch_size, L = sequence_length, H_in = input_size, H_out = hidden_size.
        /// </summary>
        /// <param name="input_size"></param>
        /// <param name="hidden_size"></param>
        /// <param name="on_forward">Either return the last or all hidden states.</param>
        /// <param name="activation">Non-linear activation used in the layer.</param>
        public RNNCell(int input_size, int hidden_size, HiddenStates on_forward = HiddenStates.ReturnLast, NonLinearity activation = NonLinearity.Tanh)
        {
            InputCache = new();
            HiddenCache = new();
            ActivationCache = new();

            float range = MathF.Sqrt(1f / hidden_size);
            weights = Tensor.RandomRange((-range, range), hidden_size, input_size);
            biases = Tensor.RandomRange((-range, range), hidden_size);
            r_weights = Tensor.RandomRange((-range, range), hidden_size, hidden_size);
            r_biases = Tensor.RandomRange((-range, range), hidden_size);
            weightsGrad = Tensor.Zeros(weights.Shape);
            biasesGrad = Tensor.Zeros(biases.Shape);
            r_weightsGrad = Tensor.Zeros(r_weights.Shape);
            r_biasesGrad = Tensor.Zeros(r_biases.Shape);
            this.nonlinearity = activation;
            this.onReturn = on_forward;
        }
        private RNNCell() { }


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

           
            Tensor[] sequences = isBatched ? input.Split(1, 1) : input.Split(0, 1);
            Tensor h = isBatched ? Tensor.Zeros(batch_size, biases.Size(-1)) : Tensor.Zeros(biases.Size(-1));


            Activation activation = nonlinearity == NonLinearity.Tanh ? new Tanh() : new ReLU();
            Tensor[] hiddenStates = new Tensor[sequences.Length];
            for (int i = 0; i < sequences.Length; i++)
            {
                // x has Shape (B, 1, H) or (1, H) => need to be (B, H) or (H)
                Tensor x = sequences[i].Squeeze(isBatched ? 1 : 0);

                Tensor l = Linear(x, weights, biases, isBatched, batch_size) + Linear(h, r_weights, r_biases, isBatched, batch_size);

                h = activation.Predict(l);
                if(onReturn == HiddenStates.ReturnAll)
                    hiddenStates[i] = h.Unsqueeze(1); // (B, 1, H)
            }

            if (onReturn == HiddenStates.ReturnLast)
                return h;
            else
                return Tensor.Concat(1, hiddenStates);
                     
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

           
            Tensor[] sequences = isBatched ? input.Split(1, 1) : input.Split(0, 1);
            Tensor h = isBatched ? Tensor.Zeros(batch_size, biases.Size(-1)) : Tensor.Zeros(biases.Size(-1));
            HiddenCache.Push(h); // add h_0
            Tensor[] hiddenStates = new Tensor[sequences.Length];
            for (int i = 0; i < sequences.Length; i++)
            {
                // x has Shape (B, 1, H) or (1, H) => need to be (B, H) or (H)
                Tensor x = sequences[i].Squeeze(isBatched ? 1 : 0);
                InputCache.Push(x);

                Tensor l = Linear(x, weights, biases, isBatched, batch_size) + Linear(h, r_weights, r_biases, isBatched, batch_size);
                
                ActivationCache.Push(nonlinearity == NonLinearity.Tanh ? new Tanh() : new ReLU());
                h = ActivationCache.Peek().Forward(l);
                HiddenCache.Push(h.Clone() as Tensor);

                if (onReturn == HiddenStates.ReturnAll)
                    hiddenStates[i] = h.Unsqueeze(1); // (B, 1, H)
            }
            if (onReturn == HiddenStates.ReturnLast)
                return h;
            else
                return Tensor.Concat(1, hiddenStates);

        }
        public Tensor Backward(Tensor dLdY)
        {
            bool isBatched = dLdY.Rank == 3;
            int batch_size = isBatched ? dLdY.Size(0) : 1;
            int sequence_length = InputCache.Count;
            int hin = weights.Size(-1);
            int hout = biases.Size(-1);

            Tensor[] dLdH;
            if (onReturn == HiddenStates.ReturnLast) // place dLdY at the last position in the array
            {
                dLdH =  new Tensor[sequence_length];
                for (int i = 0; i < sequence_length - 1; i++)
                {
                    dLdH[i] = Tensor.Zeros(dLdY.Shape);
                }
                dLdH[sequence_length - 1] = dLdY;
            } 
            else
            {
                dLdH = dLdY.Split(1, 1);
                for (int i = 0; i < sequence_length; i++)
                {
                    dLdH[i] = dLdH[i].Squeeze(1);
                }
            }
                

            Tensor[] inputGrad = new Tensor[sequence_length];

            HiddenCache.Pop(); // pop h_n because we don t need it (it contained 0 -> n)

            while (InputCache.Count > 0)
            {
                Tensor dLdLinear = ActivationCache.Pop().Backward(dLdH[InputCache.Count - 1]);

                Tensor weights_grad;
                Tensor biases_grad;
                Tensor r_weights_grad;
                Tensor r_biases_grad;      
                ComputeGradients(InputCache.Pop(), dLdLinear, isBatched, batch_size, hin, hout, out weights_grad, out biases_grad);
                ComputeGradients(HiddenCache.Pop(), dLdLinear, isBatched, batch_size, hout, hout, out r_weights_grad, out r_biases_grad);

                Tensor.CopyTo(weightsGrad + weights_grad, weightsGrad);
                Tensor.CopyTo(biasesGrad + biases_grad, biasesGrad);
                Tensor.CopyTo(r_weightsGrad + r_weights_grad, r_weightsGrad);
                Tensor.CopyTo(r_biasesGrad + r_biases_grad, r_biasesGrad);

                inputGrad[InputCache.Count] = Tensor.MatMul(dLdLinear, weights).Unsqueeze(1);
                dLdH[InputCache.Count] += Tensor.MatMul(dLdLinear, r_weights);
            }
            return Tensor.Concat(1, inputGrad);
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

        public void SetDevice(Device device) { return; }

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

            InputCache = new();
            HiddenCache = new();
            ActivationCache = new();

            this.weightsGrad = Tensor.Zeros(weights.Shape);
            this.biasesGrad = Tensor.Zeros(biases.Shape);

            r_weightsGrad = Tensor.Zeros(r_weights.Shape);  
            r_biasesGrad = Tensor.Zeros(r_biases.Shape);    

        }


        public object Clone()
        {
            var rnncell = new RNNCell();
            rnncell.InputCache = new();
            rnncell.HiddenCache = new();
            rnncell.ActivationCache = new();
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


