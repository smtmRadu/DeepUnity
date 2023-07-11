using UnityEngine;

namespace DeepUnity
{
    [SerializeField]
    public class RNN : IModule
    {
        private Tensor[] InputCache;

        [SerializeField] private Tensor[] weight_lh_l;
        [SerializeField] private Tensor[] weight_hh_l;
        [SerializeField] private Tensor[] bias_lh_l;
        [SerializeField] private Tensor[] bias_hh_l;

        [SerializeField] private RNNnonLinearity nonLinearity;
        [SerializeField] private bool batchFirst;
        [SerializeField] private float dropout;


        /// <summary>
        /// Inputs: input, h_0 <br></br>
        /// <b>input</b>:
        /// <b>(seq, input_size)</b> for unbatched input, 
        /// <b>(seq, batch_size, input_size)</b> when batch_first = false, or <b>(batch, seq, feature)</b> when batch_first = true.<br></br>
        /// <b>h_0</b>:
        /// <b>(num_layers, input_size)</b> for unbatched input, 
        /// or <b>(num_layers, batch_size, input_size)</b>  containing the initial hidden state for the input sequence batch. Defaults to zeros if not provided.<br></br>
        /// <br></br>
        /// Outputs: output, h_n (last output)<br></br>
        /// <b>output</b>: 
        /// <b>(seq, input_size)</b> for unbatched input,<b>(seq, batch_size, hidden_size)</b> when batch_first = false,
        /// or <b>(batch_size, seq, hidden_size)</b> when batch_first = true.<br></br>
        /// <b>h_n</b>:
        /// <b>(num_layers, hidden_size)</b> for unbatched input, or <b>(num_layers, batch_size, hidden_size)</b> containing the final hidden state for each element in the batch.
        /// 
        /// </summary>
        /// <param name="input_size"></param>
        /// <param name="hidden_size"></param>
        /// <param name="num_layers"></param>
        /// <param name="dropout"></param>
        public RNN(int input_size, int hidden_size, int num_layers = 2, RNNnonLinearity nonlinearity = RNNnonLinearity.TanH, bool batch_first = false, float dropout = 0f)
        {
            InputCache = new Tensor[num_layers];

            weight_lh_l = new Tensor[num_layers];
            weight_hh_l = new Tensor[num_layers];
            bias_lh_l = new Tensor[num_layers];
            bias_hh_l = new Tensor[num_layers];

            var range = (-1f / hidden_size, 1f / hidden_size);
            for (int k = 0; k < num_layers; k++)
            {
                weight_lh_l[k] = Tensor.RandomRange(range, hidden_size, input_size);
                weight_hh_l[k] = Tensor.RandomRange(range, hidden_size, hidden_size);
                bias_lh_l[k] = Tensor.RandomRange(range, hidden_size);
                bias_hh_l[k] = Tensor.RandomRange(range, hidden_size);
            }

            this.nonLinearity = nonlinearity;
            this.batchFirst = batch_first;
            this.dropout = dropout;
        }

        public Tensor Predict(Tensor input)
        {
            return null;
        }
        public Tensor Forward(Tensor input)
        {
            return null;
        }
        public Tensor Backward(Tensor loss)
        {
            return null;
        }

    }
}

