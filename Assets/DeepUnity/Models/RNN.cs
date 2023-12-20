using UnityEditor;
using UnityEngine;
using System;
using System.Linq;
using System.Collections.Generic;
using Unity.VisualScripting;
using System.Text;
using System.Collections;

namespace DeepUnity
{
    /// <summary>
    /// RNN models require low learning rate on training due to divergence. Also gradient clipping norm helps with this.
    /// </summary>
    [Serializable]
    public class RNN : Model<RNN, (Tensor, Tensor)>, ISerializationCallbackReceiver
    {
        [NonSerialized] private RNNCell[] rnn_cells;
        [SerializeField] private IModule2Wrapper[] serialized_rnn_cells;
        [SerializeField, ReadOnly] private NonLinearity nonlinearity;

        /// <summary>
        /// 
        /// Inputs: (input, h_0). <br></br>
        /// input:  <b>(B, L, H_in)</b> or <b>(L, H_in)</b> for unbatched input.<br></br>
        /// h_0:    <b>(num_layers, B, H_in)</b> or <b>(num_layers, H_in)</b> for unbatched input. <br></br>
        /// 
        /// <br></br>
        /// Outputs: (output, h_n). <br></br>
        /// output:  <b>(B, L, H_out)</b> or <b>(L, H_in)</b> for unbatched input. <br></br>
        /// h_n: <b>(num_layers, B, H_out)</b> or <b>(num_layers, H_out)</b> for unbatched input. <br></br>
        /// 
        /// <br></br>
        /// where B = batch_size, L = sequence_length, H_in = input_size, H_out = hidden_size.
        /// </summary>
        public RNN(int input_size, int hidden_size, int num_layers = 2, NonLinearity nonlinearity = NonLinearity.Tanh)
        {
            this.nonlinearity = nonlinearity;
            if (num_layers < 1)
            {
                throw new ArgumentException($"An RNN must have at least one layer, not {num_layers}.");
            }

            List<RNNCell> moduleList = new() { new RNNCell(input_size, hidden_size, nonlinearity) };

            for (int i = 1; i < num_layers; i++)          
                moduleList.Add(new RNNCell(hidden_size, hidden_size, nonlinearity));
            

            rnn_cells = moduleList.ToArray();         
        }


        public override (Tensor, Tensor) Predict((Tensor, Tensor) input_h0)
        {
            Tensor input = input_h0.Item1;
            Tensor h_0 = input_h0.Item2;

            Tensor input_clone = Tensor.Identity(input);
            Tensor h_0_clone = Tensor.Identity(h_0);

            if (input_clone.Rank != h_0_clone.Rank)
                throw new Exception($"Input ({input_clone.Shape.ToCommaSeparatedString()}) or H_0({h_0_clone.Shape.ToCommaSeparatedString()}) must have the same rank.");

            if (h_0_clone.Size(0) != rnn_cells.Count(x => x is RNNCell))
                throw new Exception($"H_0 must have the first dimension equal to num_layers ({rnn_cells.Count(x => x is RNNCell)})");


            bool isBatched = input.Rank == 3;

            // Split input into sequence of length L and h_0 per layers           
            Tensor[] input_sequence = null; // (B, Hin)[]
            Tensor[] h_0_per_layers = null; // (B, Hout)[]
            if (isBatched)
            {
                input_sequence = Tensor.Split(input_clone, -2, 1);
                h_0_per_layers = Tensor.Split(h_0_clone, -3, 1);
            }
            else
            {
                input_sequence = Tensor.Split(input_clone, -2, 1);
                h_0_per_layers = Tensor.Split(h_0_clone, -2, 1);

            }

            // Parse sequencially through each module ---------------- done
            // input_sequence[i] (B, H_in)  or (H_in)

            int rnncell_index = 0;
            foreach (var cell in rnn_cells)
            {
                for (int t = 0; t < input_sequence.Length; t++)
                {
                    h_0_per_layers[rnncell_index] = cell.Forward(input_sequence[t], h_0_per_layers[rnncell_index]);
                    input_sequence[t] = Tensor.Identity(h_0_per_layers[rnncell_index]);

                }
                rnncell_index++;
            }
            // Test if RNN cell is returning the shape of the outputs well...... 



            // input_sequence (B, H_in)
            // Join into output and h_n ----------------done
            Tensor h_n = Tensor.Concat(null, h_0_per_layers);
            Tensor output = null;

            for (int i = 0; i < input_sequence.Length; i++)
            {
                input_sequence[i] = input_sequence[i].Unsqueeze(1);
            }
            output = Tensor.Concat(1, input_sequence);

            return (output, h_n);
        }
        /// <summary>
        /// input:  <b>(L, H_in)</b> for unbatched input, <b>(L, B, H_in)</b> when batch_first = false or <b>(B, L, H_in)</b> when batch_first = true. <br></br>
        /// h_0:    <b>(num_layers, H_out)</b> for unbatched input, or <b>(num_layers, B, H_out)</b>. <br></br>
        /// </summary>
        /// <param name="input"></param>
        /// <param name="h_0"></param>
        /// <returns>
        /// output: <b>(L, H_out)</b> for unbatched input, or <b>(L, B, H_out)</b> when batch_first = false or <b>(B, L, H_out)</b> when batch_first = true. <br></br>
        /// h_n: <b>(num_layers, H_out)</b> for unbatched input or <b>(num_layers, B, H_out)</b>. <br></br>
        /// </returns>
        public override (Tensor, Tensor) Forward((Tensor, Tensor) input_h0)
        {
            return Predict(input_h0);
        }

        /// <summary>
        /// loss w.r.t outputs: <b>(L, H_out)</b> for unbatched input, or <b>(L, B, H_out)</b> when batch_first = false or <b>(B, L, H_out)</b> when batch_first = true. <br></br>
        /// </summary>
        /// <param name="lossDerivative"></param>
        public override (Tensor, Tensor) Backward((Tensor, Tensor) lossDerivative_none)
        {
            Tensor lossDerivative = lossDerivative_none.Item1;

            Tensor loss_clone = Tensor.Identity(lossDerivative);
            // loss (L, B, H_out)/(B, L, H_Out) or (L, H_out)
            bool isBatched = lossDerivative.Rank == 3;

            // Split the loss into an array of sequences
            Tensor[] loss_sequence = null;
            if(isBatched) // (L, H_out)           
            {           
                loss_sequence = Tensor.Split(loss_clone, -2, 1);
                for (int i = 0; i < loss_sequence.Length; i++)
                {
                    loss_sequence[i] = loss_sequence[i].Squeeze(-2);
                }           
            }
            else
            {
                loss_sequence = Tensor.Split(loss_clone,-2, 1);
                for (int i = 0; i < loss_sequence.Length; i++)
                {
                    loss_sequence[i] = loss_sequence[i].Squeeze(-2);
                }
            }
           


            // Backpropagate each sequence
            for (int t = loss_sequence.Length - 1; t >= 0; t--)
            {
                for (int m = rnn_cells.Length - 1; m >= 0; m--)
                {
                    //Debug.Log(loss_sequence[t]);
                    loss_sequence[t] = rnn_cells[m].Backward(loss_sequence[t]);
                }
            }

            return (Tensor.Concat(null, loss_sequence), null);
        }

        public override string Summary()
        {
            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.AppendLine($"Name: {name}");
            stringBuilder.AppendLine($"Type: {GetType()}");
            stringBuilder.AppendLine($"Layers : {rnn_cells.Length}");
            foreach (var module in rnn_cells)
            {
                stringBuilder.AppendLine($"         {module.GetType().Name}");
            }
            stringBuilder.AppendLine($"Parameters: {rnn_cells.Where(x => x is ILearnable).Select(x => (ILearnable)x).Sum(x => x.ParametersCount())}");
            return stringBuilder.ToString();
        }



        /// <summary>
        /// Get all <see cref="Parameter"/>s of this model.
        /// </summary>
        /// <returns></returns>
        public override Parameter[] Parameters()
        {
            List<Parameter> param = new();
            foreach (var item in rnn_cells.OfType<ILearnable>())
            {
                param.AddRange(item.Parameters());
            }
            return param.ToArray();
        }

        public void OnBeforeSerialize()
        {
            serialized_rnn_cells = rnn_cells.Select(x => IModule2Wrapper.Wrap(x)).ToArray();
        }
        public void OnAfterDeserialize()
        {
            rnn_cells = serialized_rnn_cells.Select(x => (RNNCell) IModule2Wrapper.Unwrap(x)).ToArray();
        }

        public override object Clone()
        {
            var rnn_clone = new RNN(2, 2, 1, this.nonlinearity);
            rnn_clone.rnn_cells = this.rnn_cells.Select(x => (RNNCell) x.Clone()).ToArray();
            return rnn_clone;
        }
    }
}

