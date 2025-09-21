using DeepUnity.Modules;
using System;
using System.Linq;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    // https://docs.pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html
    // https://arxiv.org/abs/2104.09864
    public class RotaryPositionalEmbeddings : IModule, ICloneable
    {
        [SerializeField] public int embedding_dim;   
        [SerializeField] public int max_seq_len;
        [SerializeField] public int theta = 10_000;

        public enum RoPEType
        {
            Interleaved,
            SplitHalf
        }

        public Tensor cosCachePredict;
        public Tensor sinCachePredict;

        /// <summary>
        /// Input: <b>(B, L, num_heads, head_dim)</b> or <b>(L, L, num_heads, head_dim)</b>.<br></br>
        /// Output: <b>(B, L, L, num_heads, head_dim)</b> or <b>(L, L, num_heads, head_dim)</b>.<br></br>
        /// where B = batch_size, L = sequence_length
        /// </summary>
        /// <param name="dim">embedding_dim // num_heads</param>
        /// <param name="max_seq_len">Maximum expected sequence length for the model, if exceeded the cached freqs will be recomputed</param>
        /// <param name="theta"></param>
        public RotaryPositionalEmbeddings(int dim, int max_seq_len = 4096, int theta = 10_000)
        {
            if (dim % 2 != 0)
                throw new ArgumentException("head_dim for RoPE must be even.");

            this.embedding_dim = dim;
            this.max_seq_len = max_seq_len;
            this.theta = theta;

            int half_dim = dim / 2;
            cosCachePredict = Tensor.Zeros(max_seq_len, half_dim);
            sinCachePredict = Tensor.Zeros(max_seq_len, half_dim);

            Parallel.For(0, max_seq_len, pos =>
            {
                for (int i = 0; i < half_dim; i++)
                {
                    float freq = 1.0f / MathF.Pow(theta, (2.0f * i) / dim);
                    float angle = pos * freq;
                    cosCachePredict[pos, i] = MathF.Cos(angle);
                    sinCachePredict[pos, i] = MathF.Sin(angle);
                }

            });
        }

        public Tensor ApplyRotaryEmbeddings(Tensor x, int[] input_pos = null, RoPEType type = RoPEType.SplitHalf)
        {

            int rank = x.Rank;
            bool is_batched = x.Rank == 4;

            int head_dim = x.Size(-1); // head_dim
            if (head_dim != embedding_dim)
                throw new ShapeException($"RoPE last dim={head_dim} but constructed with embedding_dim={embedding_dim}");

            int seq_len = x.Size(-3);
            if (input_pos is null)
                input_pos = Enumerable.Range(0, seq_len).ToArray();

            if (seq_len != input_pos.Length)
                throw new ShapeException($"Number of input positions is not matching the number of vectors to rotate: {x.Shape.ToCommaSeparatedString()}, input_pos {input_pos.ToCommaSeparatedString()})");

            

            if (input_pos.Max() > max_seq_len)
                throw new ShapeException($"Input max_seq_len should be {max_seq_len} (received input_pos: {input_pos.ToCommaSeparatedString()})");

            int batch_size = is_batched ? x.Size(-4) : 1;
            int num_heads = x.Size(-2);
            Tensor y = Tensor.Zeros(x.Shape);

            //Debug.Log(input_pos.ToCommaSeparatedString());

            if(type == RoPEType.SplitHalf)
            {
                int half_dim = head_dim / 2;

                Parallel.For(0, seq_len, l =>
                {
                    for (int b = 0; b < batch_size; b++)
                    {
                        for (int h = 0; h < num_heads; h++)
                        {
                            for (int i = 0; i < half_dim; i++)
                            {
                                float cos_ = cosCachePredict[input_pos[l], i];
                                float sin_ = sinCachePredict[input_pos[l], i];

                                float x1 = x[b, l, h, i];              
                                float x2 = x[b, l, h, i + half_dim];   

                                y[b, l, h, i] = x1 * cos_ - x2 * sin_;
                                y[b, l, h, i + half_dim] = x2 * cos_ + x1 * sin_;
                            }
                        }
                    }
                });
            }
            else if(type == RoPEType.Interleaved)
            {
                Parallel.For(0, seq_len, l =>
                {
                    for (int b = 0; b < batch_size; b++)
                    {
                        for (int h = 0; h < num_heads; h++)
                        {
                            for (int i = 0; i < head_dim / 2; i++)
                            {
                                float cos_ = cosCachePredict[input_pos[l], i];
                                float sin_ = sinCachePredict[input_pos[l], i];

                                float x_even_ = x[b, l, h, 2 * i];
                                float x_odd__ = x[b, l, h, 2 * i + 1];

                                y[b, l, h, 2 * i] = x_even_ * cos_ - x_odd__ * sin_;
                                y[b, l, h, 2 * i + 1] = x_odd__ * cos_ + x_even_ * sin_;
                            }
                        }
                    }
                });
            }
                

            return y;
        }

        public Tensor Predict(Tensor x)
        {
            throw new Exception("Use ApplyRotaryEmbeddings function instead");
        }

        public Tensor Forward(Tensor x)
        {
            throw new Exception("This module was implemented only for inference.");
        }

        public Tensor Backward(Tensor dLdY)
        {
            throw new Exception("This module was implemented only for inference.");
        }

        public object Clone() => new RotaryPositionalEmbeddings(embedding_dim, max_seq_len, theta);




    }
}
