using System;
using System.Collections.Generic;
using Unity.VisualScripting.Antlr3.Runtime;
using Unity.VisualScripting;
using UnityEngine;
using System.Threading.Tasks;
using System.Linq;

namespace DeepUnity
{
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
    [Serializable]
    public class LSTMCell : ILearnable, IModule
    {
        [SerializeField] private Tensor weights_ii;
        [SerializeField] private Tensor biases_ii;
        [SerializeField] private Tensor weights_hi;
        [SerializeField] private Tensor biases_hi;

        [SerializeField] private Tensor weights_if;
        [SerializeField] private Tensor biases_if;
        [SerializeField] private Tensor weights_hf;
        [SerializeField] private Tensor biases_hf;

        [SerializeField] private Tensor weights_ig;
        [SerializeField] private Tensor biases_ig;
        [SerializeField] private Tensor weights_hg;
        [SerializeField] private Tensor biases_hg;

        [SerializeField] private Tensor weights_io;
        [SerializeField] private Tensor biases_io;
        [SerializeField] private Tensor weights_ho;
        [SerializeField] private Tensor biases_ho;


        [NonSerialized] private Tensor weights_iiGrad;
        [NonSerialized] private Tensor biases_iiGrad;
        [NonSerialized] private Tensor weights_hiGrad;
        [NonSerialized] private Tensor biases_hiGrad;
      
        [NonSerialized] private Tensor weights_ifGrad;
        [NonSerialized] private Tensor biases_ifGrad;
        [NonSerialized] private Tensor weights_hfGrad;
        [NonSerialized] private Tensor biases_hfGrad;
      
        [NonSerialized] private Tensor weights_igGrad;
        [NonSerialized] private Tensor biases_igGrad;
        [NonSerialized] private Tensor weights_hgGrad;
        [NonSerialized] private Tensor biases_hgGrad;
     
        [NonSerialized] private Tensor weights_ioGrad;
        [NonSerialized] private Tensor biases_ioGrad;
        [NonSerialized] private Tensor weights_hoGrad;
        [NonSerialized] private Tensor biases_hoGrad;



        [NonSerialized] private Stack<Tensor> InputCache;
        [NonSerialized] private Stack<Tensor> CellCache;
        [NonSerialized] private Stack<Tensor> HiddenCache;
        [NonSerialized] private Stack<Activation> Activations;

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
        public LSTMCell(int in_features, int out_features)
        {
            InputCache = new Stack<Tensor>();
            CellCache = new Stack<Tensor>();
            HiddenCache = new Stack<Tensor>();
            Activations = new Stack<Activation>();

            float range = MathF.Sqrt(1f / out_features);

            weights_ii = Tensor.RandomRange((-range, range), out_features, in_features);
            biases_ii = Tensor.RandomRange((-range, range), out_features);
            weights_hi = Tensor.RandomRange((-range, range), out_features, out_features);
            biases_hi = Tensor.RandomRange((-range, range), out_features);
            weights_iiGrad = Tensor.Zeros(weights_ii.Shape);
            biases_iiGrad = Tensor.Zeros(biases_ii.Shape);
            weights_hiGrad = Tensor.Zeros(weights_hi.Shape);
            biases_hiGrad = Tensor.Zeros(biases_hi.Shape);

            weights_if = Tensor.RandomRange((-range, range), out_features, in_features);
            biases_if = Tensor.RandomRange((-range, range), out_features);
            weights_hf = Tensor.RandomRange((-range, range), out_features, out_features);
            biases_hf = Tensor.RandomRange((-range, range), out_features);
            weights_ifGrad = Tensor.Zeros(weights_if.Shape);
            biases_ifGrad = Tensor.Zeros(biases_if.Shape);
            weights_hfGrad = Tensor.Zeros(weights_hf.Shape);
            biases_hfGrad = Tensor.Zeros(biases_hf.Shape);

            weights_ig = Tensor.RandomRange((-range, range), out_features, in_features);
            biases_ig = Tensor.RandomRange((-range, range), out_features);
            weights_hg = Tensor.RandomRange((-range, range), out_features, out_features);
            biases_hg = Tensor.RandomRange((-range, range), out_features);
            weights_igGrad = Tensor.Zeros(weights_ig.Shape);
            biases_igGrad = Tensor.Zeros(biases_ig.Shape);
            weights_hgGrad = Tensor.Zeros(weights_hg.Shape);
            biases_hgGrad = Tensor.Zeros(biases_hg.Shape);

            weights_io = Tensor.RandomRange((-range, range), out_features, in_features);
            biases_io = Tensor.RandomRange((-range, range), out_features);
            weights_ho = Tensor.RandomRange((-range, range), out_features, out_features);
            biases_ho = Tensor.RandomRange((-range, range), out_features);
            weights_ioGrad = Tensor.Zeros(weights_io.Shape);
            biases_ioGrad = Tensor.Zeros(biases_io.Shape);
            weights_hoGrad = Tensor.Zeros(weights_ho.Shape);
            biases_hoGrad = Tensor.Zeros(biases_ho.Shape);
        }
        private LSTMCell() { }
        public Tensor Predict(Tensor input)
        {
            if (input.Size(-1) != weights_ii.Size(-1))
            {
                throw new ShapeException($"Input features ({input.Size(-1)}) does not match with the Dense Layer input features_num ({weights_ii.Size(-1)}).");

            }
            if (input.Rank != 3 && input.Rank != 2)
            {
                throw new ShapeException($"Input must be of shape (B, L, H) or (L, H) for unabatches input, and input received has shape ({input.Shape.ToCommaSeparatedString()}).");
            }
            bool isBatched = input.Rank == 3;
            int batch_size = isBatched ? input.Size(0) : 1;

            Tensor[] sequences = isBatched ? input.Split(1, 1) : input.Split(0, 1);
            Tensor h = isBatched ? Tensor.Zeros(batch_size, biases_ii.Size(-1)) : Tensor.Zeros(biases_ii.Size(-1));
            Tensor c = isBatched ? Tensor.Zeros(batch_size, biases_ii.Size(-1)) : Tensor.Zeros(biases_ii.Size(-1));


            foreach (var elem in sequences)
            {
                // x has Shape (B, 1, H) or (1, H) => need to be (B, H) or (H)
                Tensor x = isBatched ? elem.Squeeze(1) : elem.Squeeze(0);

                Tensor i = new Sigmoid().Predict(Linear(x, weights_ii, biases_ii, isBatched, batch_size) + Linear(h, weights_hi, biases_hi, isBatched, batch_size));
                Tensor f = new Sigmoid().Predict(Linear(x, weights_if, biases_if, isBatched, batch_size) + Linear(h, weights_hf, biases_hf, isBatched, batch_size));
                Tensor g = new Tanh().Predict(Linear(x, weights_ig, biases_ig, isBatched, batch_size) + Linear(h, weights_hg, biases_hg, isBatched, batch_size));
                Tensor o = new Sigmoid().Predict(Linear(x, weights_io, biases_io, isBatched, batch_size) + Linear(h, weights_ho, biases_ho, isBatched, batch_size));
                c = f * c + i * g;
                h = o * new Tanh().Predict(c);
            }

            return h;
        }
        public Tensor Forward(Tensor input)
        {
            if (input.Size(-1) != weights_ii.Size(-1))
            {
                throw new ShapeException($"Input features ({input.Size(-1)}) does not match with the Dense Layer input features_num ({weights_ii.Size(-1)}).");

            }
            if (input.Rank != 3 && input.Rank != 2)
            {
                throw new ShapeException($"Input must be of shape (B, L, H) or (L, H) for unabatches input, and input received has shape ({input.Shape.ToCommaSeparatedString()}).");
            }
            bool isBatched = input.Rank == 3;
            int batch_size = isBatched ? input.Size(0) : 1;

            Tensor[] sequences = isBatched ? input.Split(1, 1) : input.Split(0, 1);
            Tensor h = isBatched ? Tensor.Zeros(batch_size, biases_ii.Size(-1)) : Tensor.Zeros(biases_ii.Size(-1));
            Tensor c = isBatched ? Tensor.Zeros(batch_size, biases_ii.Size(-1)) : Tensor.Zeros(biases_ii.Size(-1));


            foreach (var elem in sequences)
            {
                // x has Shape (B, 1, H) or (1, H) => need to be (B, H) or (H)
                Tensor x = isBatched ? elem.Squeeze(1) : elem.Squeeze(0);

                Activations.Push(new Sigmoid());
                Tensor i = Activations.Peek().Forward(Linear(x, weights_ii, biases_ii, isBatched, batch_size) + Linear(h, weights_hi, biases_hi, isBatched, batch_size));
                Activations.Push(new Sigmoid());
                Tensor f = Activations.Peek().Forward(Linear(x, weights_if, biases_if, isBatched, batch_size) + Linear(h, weights_hf, biases_hf, isBatched, batch_size));
                Activations.Push(new Tanh());
                Tensor g = Activations.Peek().Forward(Linear(x, weights_ig, biases_ig, isBatched, batch_size) + Linear(h, weights_hg, biases_hg, isBatched, batch_size));
                Activations.Push(new Sigmoid());
                Tensor o = Activations.Peek().Forward(Linear(x, weights_io, biases_io, isBatched, batch_size) + Linear(h, weights_ho, biases_ho, isBatched, batch_size));
                c = f * c + i * g;
                Activations.Push(new Tanh());
                h = o * Activations.Peek().Forward(c);

                InputCache.Push(x.Clone() as Tensor);
                CellCache.Push(c.Clone() as Tensor);
                HiddenCache.Push(h.Clone() as Tensor);
            }

            return h;
        }
        public Tensor Backward(Tensor dLdHPrime)
        {
            throw new NotImplementedException();
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



        public void SetDevice(Device device) { return; }

        public int ParametersCount()
        {
            int count = weights_ii.Count() + biases_ii.Count() + weights_hi.Count() + biases_hi.Count() +
                        weights_if.Count() + biases_if.Count() + weights_hf.Count() + biases_hf.Count() +
                        weights_ig.Count() + biases_ig.Count() + weights_hg.Count() + biases_hg.Count() +
                        weights_io.Count() + biases_io.Count() + weights_ho.Count() + biases_ho.Count();

            return count;
        }
        public Parameter[] Parameters()
        {
            if (weights_iiGrad == null)
                OnAfterDeserialize();

            var w_ii = new Parameter(weights_ii, weights_iiGrad);
            var b_ii = new Parameter(biases_ii, biases_iiGrad);
            var w_hi = new Parameter(weights_hi, weights_hiGrad);
            var b_hi = new Parameter(biases_hi, biases_hiGrad);

            var w_if = new Parameter(weights_if, weights_ifGrad);
            var b_if = new Parameter(biases_if, biases_ifGrad);
            var w_hf = new Parameter(weights_hf, weights_hfGrad);
            var b_hf = new Parameter(biases_hf, biases_hfGrad);

            var w_ig = new Parameter(weights_ig, weights_igGrad);
            var b_ig = new Parameter(biases_ig, biases_igGrad);
            var w_hg = new Parameter(weights_hg, weights_hgGrad);
            var b_hg = new Parameter(biases_hg, biases_hgGrad);

            var w_io = new Parameter(weights_io, weights_ioGrad);
            var b_io = new Parameter(biases_io, biases_ioGrad);
            var w_ho = new Parameter(weights_ho, weights_hoGrad);
            var b_ho = new Parameter(biases_ho, biases_hoGrad);

            return new Parameter[] { w_ii, b_ii, w_hi, b_hi, w_if, b_if, w_hf, b_hf, w_ig, b_ig, w_hg, b_hg, w_io, b_io, w_ho, b_ho };
        }

        public void OnBeforeSerialize()
        {
            // Add any necessary logic before serialization
        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (weights_ii.Shape == null)
                return;

            if (weights_ii.Shape.Length == 0)
                return;

            InputCache = new Stack<Tensor>();
            HiddenCache = new Stack<Tensor>();
            CellCache = new Stack<Tensor>();
            Activations = new Stack<Activation>();

            this.weights_iiGrad = Tensor.Zeros(weights_ii.Shape);
            this.biases_iiGrad = Tensor.Zeros(biases_ii.Shape);

            this.weights_hiGrad = Tensor.Zeros(weights_hi.Shape);
            this.biases_hiGrad = Tensor.Zeros(biases_hi.Shape);

            this.weights_ifGrad = Tensor.Zeros(weights_if.Shape);
            this.biases_ifGrad = Tensor.Zeros(biases_if.Shape);

            this.weights_hfGrad = Tensor.Zeros(weights_hf.Shape);
            this.biases_hfGrad = Tensor.Zeros(biases_hf.Shape);

            this.weights_igGrad = Tensor.Zeros(weights_ig.Shape);
            this.biases_igGrad = Tensor.Zeros(biases_ig.Shape);

            this.weights_hgGrad = Tensor.Zeros(weights_hg.Shape);
            this.biases_hgGrad = Tensor.Zeros(biases_hg.Shape);

            this.weights_ioGrad = Tensor.Zeros(weights_io.Shape);
            this.biases_ioGrad = Tensor.Zeros(biases_io.Shape);

            this.weights_hoGrad = Tensor.Zeros(weights_ho.Shape);
            this.biases_hoGrad = Tensor.Zeros(biases_ho.Shape);
        }

        public object Clone()
        {
            var lstmCell = new LSTMCell();
            lstmCell.InputCache = new Stack<Tensor>(InputCache.Select(t => (Tensor)t.Clone()));
            lstmCell.HiddenCache = new Stack<Tensor>(HiddenCache.Select(t => (Tensor)t.Clone()));
            lstmCell.CellCache = new Stack<Tensor>(CellCache.Select(t => (Tensor)t.Clone()));
            lstmCell.Activations = new Stack<Activation>(Activations.Select(a => (Activation)a.Clone()));

            lstmCell.weights_ii = (Tensor)this.weights_ii.Clone();
            lstmCell.biases_ii = (Tensor)this.biases_ii.Clone();
            lstmCell.weights_hi = (Tensor)this.weights_hi.Clone();
            lstmCell.biases_hi = (Tensor)this.biases_hi.Clone();

            lstmCell.weights_if = (Tensor)this.weights_if.Clone();
            lstmCell.biases_if = (Tensor)this.biases_if.Clone();
            lstmCell.weights_hf = (Tensor)this.weights_hf.Clone();
            lstmCell.biases_hf = (Tensor)this.biases_hf.Clone();

            lstmCell.weights_ig = (Tensor)this.weights_ig.Clone();
            lstmCell.biases_ig = (Tensor)this.biases_ig.Clone();
            lstmCell.weights_hg = (Tensor)this.weights_hg.Clone();
            lstmCell.biases_hg = (Tensor)this.biases_hg.Clone();

            lstmCell.weights_io = (Tensor)this.weights_io.Clone();
            lstmCell.biases_io = (Tensor)this.biases_io.Clone();
            lstmCell.weights_ho = (Tensor)this.weights_ho.Clone();
            lstmCell.biases_ho = (Tensor)this.biases_ho.Clone();

            lstmCell.weights_iiGrad = (Tensor)this.weights_iiGrad.Clone();
            lstmCell.biases_iiGrad = (Tensor)this.biases_iiGrad.Clone();
            lstmCell.weights_hiGrad = (Tensor)this.weights_hiGrad.Clone();
            lstmCell.biases_hiGrad = (Tensor)this.biases_hiGrad.Clone();

            lstmCell.weights_ifGrad = (Tensor)this.weights_ifGrad.Clone();
            lstmCell.biases_ifGrad = (Tensor)this.biases_ifGrad.Clone();
            lstmCell.weights_hfGrad = (Tensor)this.weights_hfGrad.Clone();
            lstmCell.biases_hfGrad = (Tensor)this.biases_hfGrad.Clone();

            lstmCell.weights_igGrad = (Tensor)this.weights_igGrad.Clone();
            lstmCell.biases_igGrad = (Tensor)this.biases_igGrad.Clone();
            lstmCell.weights_hgGrad = (Tensor)this.weights_hgGrad.Clone();
            lstmCell.biases_hgGrad = (Tensor)this.biases_hgGrad.Clone();

            lstmCell.weights_ioGrad = (Tensor)this.weights_ioGrad.Clone();
            lstmCell.biases_ioGrad = (Tensor)this.biases_ioGrad.Clone();
            lstmCell.weights_hoGrad = (Tensor)this.weights_hoGrad.Clone();
            lstmCell.biases_hoGrad = (Tensor)this.biases_hoGrad.Clone();

            return lstmCell;
        }

    }

}


