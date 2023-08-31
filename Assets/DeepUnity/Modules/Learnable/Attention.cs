using System;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// TO BE IMPLEMENTED IN THE FUTURE...
    /// Multihead(Q,K,V) = Cat(head1,head2...headh)* W_O <br></br>
    /// Attention(Q,K,V) = softmax(Q * K_T / sqrt(dk)) * V
    /// where dk = embedded dim.
    /// 
    /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention
    /// https://arxiv.org/pdf/1706.03762.pdf
    /// https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    /// </summary>
    public class Attention : Learnable, IModuleS, ISelfOptimizable
    {

        [SerializeField] private bool useScale;
        /// Gamma is Query
        /// Beta is Key
        [SerializeField] private Tensor value;
        [NonSerialized] private Tensor valueGrad;


        ///
        /// <summary>
        /// 
        /// </summary>
        /// <param name="embedded_dim"></param>
        /// <param name="num_heads"></param>
        public Attention(bool use_scale = false) : base(Device.CPU, InitType.HE_Normal, InitType.HE_Normal, new int[1], new int[1], 1, 1)
        {
            this.useScale = use_scale;
        }

        public Tensor Forward1(Tensor input)
        {
            Tensor result = Tensor.MatMulGPU(gamma, beta);
            
            if(useScale)
            {

            }
            // Apply mask optional
            result = new Softmax().Forward(result);
            result = Tensor.MatMulGPU(result, value);
            return result;
        }
        public Tensor Backward(Tensor loss)
        {
            return null;
        }

        public (Tensor, Tensor) Forward(Tensor input)
        {
            return (null, null);
        }
        public void SelfOptimise(float lr)
        {
            // optimise value
        }
    }

}

