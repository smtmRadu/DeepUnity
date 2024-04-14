using System;
using System.Linq;
using UnityEngine;
using DeepUnity.Activations;
using System.Collections.Generic;

namespace DeepUnity.Modules
{
    // I HAVE TO MAKE THIS SERIALIZABLE IN THE FUTURE (just remove attention heads and make them from scratch pfah..)
    /// <summary>
    /// <b>Applied a Self Scaled Dot-Product Attention with multiple heads.</b> <br></br>
    /// Input: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
    /// Output: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
    /// where B = batch_size, L = sequence_length and H = num_features<br></br>
    /// <b>Placed after the non-linear activation function.</b> <br></br>
    /// </summary>
    /// <param name="embed_dim"></param>
    /// <param name="heads_num"></param>
    [Serializable]
    public class MultiheadAttention : ILearnable, IModule
    {
        [SerializeField] public Device Device
        { 
            get => W_O.Device; 
            set
                {
                    for (int i = 0; i < attention_heads.Length; i++)
                    {
                        attention_heads[i].Device = value;
                    }
                    W_O.Device = value;
                } 
        }
        [SerializeField] private Attention[] attention_heads;
        [SerializeField] private Dense W_O;

        /// <summary>
        /// <b>Applied a Self Scaled Dot-Product Attention with multiple heads.</b> <br></br>
        /// Input: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
        /// Output: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
        /// where B = batch_size, L = sequence_length and H = num_features<br></br>
        /// <b>Placed after the non-linear activation function.</b> <br></br>
        /// </summary>
        /// <param name="embed_dim">Must be divisible by heads_num.</param>
        /// <param name="heads_num">Must divide embed_dim exactly.</param>
        /// <param name="mask">Causal attention.</param>
        /// <exception cref="ArgumentException">Embedding dimension must be divisible by heads number</exception>
        public MultiheadAttention(int embed_dim, int heads_num, bool mask = false, Device device = Device.CPU)
        {
            if (embed_dim % heads_num != 0)
            {
                throw new ArgumentException("Embedding dimension must be divisible by heads number");
            }
            attention_heads = new Attention[heads_num];
            for (int i = 0; i < heads_num; i++)
            {
                attention_heads[i] = new Attention(embed_dim, embed_dim / heads_num, mask, device: device);
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
            return W_O.Predict(output_conc);
        }

        public Tensor Forward(Tensor input)
        {
            Tensor[] outputs = new Tensor[attention_heads.Length];
            for (int i = 0; i < attention_heads.Length; i++)
            {
                outputs[i] = attention_heads[i].Forward(input);
            }
            Tensor output_conc = Tensor.Concat(-1, outputs); // B, L, E
            return W_O.Forward(output_conc);
        }

        public Tensor Backward(Tensor loss)
        {
            Tensor mhatt_grad = W_O.Backward(loss);
            Tensor[] mhat_grad_splitted = mhatt_grad.Split(-1, mhatt_grad.Size(-1) / attention_heads.Length);

            var input_grad = attention_heads[0].Backward(mhat_grad_splitted[0]);
            for (int i = 1; i < attention_heads.Length; i++)
            {
                input_grad += attention_heads[i].Backward(mhat_grad_splitted[i]);
            }
            return input_grad;
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
            matt.Device = Device;
            matt.W_O = W_O.Clone() as Dense;
            return matt;
        }
    }
}


