using UnityEngine;
using System;
namespace DeepUnity
{
    [Serializable]
    public class Conv2D : IModule, IParameters
    {
        // https://www.youtube.com/watch?v=Lakz2MoHy6o
        private Tensor Input_Cache { get; set; }


        // input shape = [batch, Ichannels, Iheight, Iwidth]
        // output_shape = [batch, Ochannels, Iheight - K + 1, Iwidth - K + 1] 


        [SerializeField] Tensor kernels; // [Ochannels, Ichannels, K, K]
        [SerializeField] Tensor biases; // [Ochannels, 1, K, K]

        Tensor grad_Kernels;
        Tensor grad_Biases;

        public Conv2D(int in_channels, int out_channels, int kernel_size, int padding = 0)
        {
            kernels = Tensor.RandomNormal(out_channels, in_channels, kernel_size, kernel_size);
            biases = Tensor.RandomNormal(out_channels, 1, kernel_size, kernel_size);

            grad_Kernels = Tensor.Zeros(out_channels, in_channels, kernel_size, kernel_size);
            grad_Biases = Tensor.Zeros(out_channels, 1, kernel_size, kernel_size);
        }
        public Tensor Predict(Tensor input)
        {
            return null;
        }
        public Tensor Forward(Tensor input)
        {
            int batch = input.Shape.batch;
            int output_channels = biases.Shape.batch;
            int output_height = biases.Shape.height;
            int output_width = biases.Shape.width;
            Tensor output = Tensor.Zeros(batch, output_channels, output_height, output_width);


            for (int b = 0; b < batch; b++)
            {
                for (int oc = 0; oc < output_channels; oc++)
                {
                    for (int ic = 0; ic < input.Shape.channels; ic++)
                    {
                        // And here we apply corr2D
                        // Y = B + E(1..k kernels)(X star K)
                        // I * K (Convolution) == I star rot180(K) (Correlation)
                        



                    }
                }
            }

            return output;
        }
        public Tensor Backward(Tensor loss)
        {
            return null;
        }

        public void ZeroGrad()
        {
            grad_Kernels.ForEach(x => 0f);
            grad_Biases.ForEach(x => 0f);
        }
        public void ClipGradValue(float clip_value)
        {
            return;
        }
        public void ClipGradNorm(float max_norm)
        {
            return;
        }
        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization, and one on them is called when weights.shape.length == 0.
            if (kernels.Shape == null || kernels.Shape.width == 0)
                return;

            int out_channels = kernels.Shape.batch;
            int in_channels = kernels.Shape.channels;
            int kernel_size = kernels.Shape.width;


            grad_Kernels = Tensor.Zeros(out_channels, in_channels, kernel_size, kernel_size);
            grad_Biases = Tensor.Zeros(out_channels, 1, kernel_size, kernel_size);
        }

    }
}

