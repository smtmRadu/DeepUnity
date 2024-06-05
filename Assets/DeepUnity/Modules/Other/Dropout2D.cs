using System;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// <b>Placed after the non-linear activation function.</b> <br></br>
    /// Input: <b>(B, C, H, W)</b> or <b>(C, H, W)</b> for unbatched input. <br></br>
    /// Output: <b>(B, C, H, W)</b> or <b>(C, H, W)</b> for unbatched input. <br></br>
    /// where B = batch_size, C = channels, H = height and W = width.<br></br>
    /// <br></br>
    /// <em>The output shape is the same with the input shape.</em>
    /// </summary>
    [Serializable]
    public class Dropout2D : IModule
    {
        [SerializeField] private bool inPlace = false;
        [SerializeField] private float dropout = 0.499999777646258f;
        private Tensor OutputCache { get; set; }

        /// <summary>
        /// <b>Placed after the non-linear activation function.</b> <br></br>
        /// Input: <b>(B, C, H, W)</b> or <b>(C, H, W)</b> for unbatched input. <br></br>
        /// Output: <b>(B, C, H, W)</b> or <b>(C, H, W)</b> for unbatched input. <br></br>
        /// where B = batch_size, C = channels, H = height and W = width.<br></br>
        /// <br></br>
        /// <em>The output shape is the same with the input shape.</em>
        /// </summary>
        /// <param name="dropout"> Low value: weak dropout | High value: strong dropout</param>
        public Dropout2D(float dropout = 0.5f, bool in_place = false)
        {
            if (dropout < Utils.EPSILON || dropout > 1f - Utils.EPSILON)
                throw new ArgumentException("Dropout value must be in range (0,1) when creating a Dropout layer module.");

            this.inPlace = in_place;
            this.dropout = dropout;
        }

        public Tensor Predict(Tensor input)
        {
            if (inPlace == true)
                return input;

            return input.Clone() as Tensor;
        }
        public Tensor Forward(Tensor input)
        {
            if (input.Rank < 3)
                throw new ArgumentException($"Input must be of shape (B, C, H, W) or (C, H, W) (received: {input.Shape.ToCommaSeparatedString()}).");

            int batch_size = input.Rank == 4 ? input.Size(0) : 1;
            int channels = input.Size(-3);
            int height = input.Size(-2);
            int width = input.Size(-1);
            float scale = 1f / (1f - dropout);
            if (inPlace)
            {
                Parallel.For(0, batch_size, b =>
                {
                    bool do_we_drop = Utils.Random.Bernoulli(dropout);

                    for (int c = 0; c < channels; c++)
                    {
                        for (int h = 0; h < height; h++)
                        {
                            for (int w = 0; w < width; w++)
                            {
                                input[b, c, h, w] = do_we_drop ? 0f : input[b, c, h, w] * scale;
                            }
                        }
                    }
                });
                OutputCache = input.Clone() as Tensor;
                return input;
            }
            else
            {
                OutputCache = Tensor.Zeros(input.Shape);
                Parallel.For(0, batch_size, b =>
                {
                    bool do_we_drop = Utils.Random.Bernoulli(dropout);

                    for (int c = 0; c < channels; c++)
                    {
                        for (int h = 0; h < height; h++)
                        {
                            for (int w = 0; w < width; w++)
                            {
                                OutputCache[b, c, h, w] = do_we_drop ? 0f : input[b, c, h, w] * scale;
                            }
                        }
                    }
                });
                return OutputCache.Clone() as Tensor;
            }

        }
        public Tensor Backward(Tensor loss)
        {
            return loss.Zip(OutputCache, (l, i) => i != 0f ? l : 0f);
        }

        public object Clone() => new Dropout2D(dropout, inPlace);
    }

}