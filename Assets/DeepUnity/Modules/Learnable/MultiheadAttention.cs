using System;
using System.Linq;
using UnityEngine;

namespace DeepUnity.Layers
{
    /// <summary>
    /// <b>Applied a Self Scaled Dot-Product Attention with multiple heads.</b> <br></br>
    /// Input: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
    /// Output: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
    /// where B = batch_size, L = sequence_length and H = num_features<br></br>
    /// <b>Placed after the non-linear activation function.</b> <br></br>
    /// </summary>
    /// <param name="embed_dim"></param>
    /// <param name="heads_num"></param>
    [SerializeField]
    public class MultiheadAttention : ILearnable, IModule
    {
        [SerializeField] private Device device;
        [SerializeField] private Attention[] attention_heads;
        [SerializeField] private Dense W_O;

        /// <summary>
        /// <b>Applied a Self Scaled Dot-Product Attention with multiple heads.</b> <br></br>
        /// Input: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
        /// Output: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
        /// where B = batch_size, L = sequence_length and H = num_features<br></br>
        /// <b>Placed after the non-linear activation function.</b> <br></br>
        /// </summary>
        /// <param name="embed_dim">Must divisible by heads_num</param>
        /// <exception cref="ArgumentException">Embedding dimension must be divisible by heads number</exception>
        public MultiheadAttention(int embed_dim, int heads_num, Device device = Device.CPU)
        {
            this.device = device;
            if (embed_dim % heads_num != 0)
            {
                throw new ArgumentException("Embedding dimension must be divisible by heads number");
            }
            attention_heads = new Attention[heads_num];
            for (int i = 0; i < heads_num; i++)
            {
                attention_heads[i] = new Attention(embed_dim, embed_dim / heads_num);
            }
            W_O = new Dense(embed_dim, embed_dim, device: device);
        }
        private MultiheadAttention() { }

        public Tensor Predict(Tensor input)
        {
            Tensor[] outputs = new Tensor[attention_heads.Length];
            for (int i = 0; i < attention_heads.Length; i++)
            {
                outputs[i] = attention_heads[i].Predict(input);
            }
            Tensor output_conc = Tensor.Concat(-1, outputs); // B, L, E
            Tensor[] sequence = Tensor.Split(output_conc, -2, 1);
            Tensor[] final_output = sequence.Select(x => W_O.Predict(x.Squeeze(-2))).Select(x => x.Unsqueeze(-1)).ToArray();
            return Tensor.Concat(-2, final_output);
        }

        public Tensor Forward(Tensor input)
        {
            Tensor[] outputs = new Tensor[attention_heads.Length];
            for (int i = 0; i < attention_heads.Length; i++)
            {
                outputs[i] = attention_heads[i].Forward(input);
            }
            Tensor output_conc = Tensor.Concat(-1, outputs); // B, L, E
            Tensor[] sequence = Tensor.Split(output_conc, -2, 1);
            Tensor[] final_output = sequence.Select(x => W_O.Forward(x.Squeeze(-2))).Select(x => x.Unsqueeze(-1)).ToArray();
            return Tensor.Concat(-2, final_output);
        }

        public Tensor Backward(Tensor loss)
        {
            Tensor[] sequence = loss.Split(-2, 1);
            Tensor[] mhatt_loss = sequence.Select(x => W_O.Backward(x.Squeeze(-2))).Select(x => x.Unsqueeze(-1)).ToArray();
            Tensor mhatt_grad = Tensor.Concat(-2, mhatt_loss);
            Tensor[] mhat_grad_splitted = mhatt_grad.Split(-1, mhatt_grad.Size(-1) / attention_heads.Length);

            var input_grad = attention_heads[0].Backward(mhat_grad_splitted[0]);
            for (int i = 1; i < attention_heads.Length; i++)
            {
                input_grad += attention_heads[i].Backward(mhat_grad_splitted[i]);
            }
            return input_grad;
        }


        public void SetDevice(Device device)
        {
            this.device = device;
            for (int i = 0; i < attention_heads.Length; i++)
            {
                attention_heads[i].SetDevice(device);
            }
            W_O.SetDevice(device);
        }
        public int ParametersCount()
        {
            int paramst = W_O.ParametersCount();
            paramst += attention_heads.Sum(x => x.ParametersCount());
            return paramst;
        }
        public Parameter[] Parameters()
        {
            var attpar = attention_heads.Select(x => x.Parameters());
            var woparr = W_O.Parameters();
            var all_p = attpar.SelectMany(x => x).ToList();

            return woparr.Concat(all_p).ToArray();
        }


        public virtual void OnBeforeSerialize()
        {

        }

        public virtual void OnAfterDeserialize()
        {
            W_O.OnAfterDeserialize();

            for (int i = 0; i < attention_heads.Length; i++)
            {
                attention_heads[i].OnAfterDeserialize();
            }
        }

        public object Clone()
        {
            var matt = new MultiheadAttention();
            matt.attention_heads = attention_heads.Select(x => x.Clone() as Attention).ToArray();
            matt.device = device;
            matt.W_O = W_O.Clone() as Dense;
            return matt;
        }
    }



}


