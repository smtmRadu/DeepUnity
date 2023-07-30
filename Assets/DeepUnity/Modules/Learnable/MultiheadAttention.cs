/*using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// TO BE IMPLEMENTED IN THE FUTURE...
    /// Multihead(Q,K,V) = Cat(head1,head2...headh)* W_O <br></br>
    /// Attention(Q,K,V) = softmax(Q * K_T / sqrt(dk)) * V
    /// where dk = embedded dim.
    /// </summary>
    public class MultiheadAttention : Learnable, IModuleS
    {
        [SerializeField] private int embedded_dim;
        [SerializeField] private int num_heads;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="embedded_dim"></param>
        /// <param name="num_heads"></param>
        public MultiheadAttention(int embedded_dim, int num_heads) : base(Device.CPU)
        {
            throw new System.NotImplementedException();
        }

        public Tensor Backward(Tensor loss)
        {
            return null;
        }

        public (Tensor, Tensor) Forward(Tensor input)
        {
            return null;
        }
    }

}
*/

