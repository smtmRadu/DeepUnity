using System;
using System.Linq;
using System.Threading.Tasks;

namespace DeepUnity.Optimizers
{
    public abstract partial class Optimizer
    {
        private Lazy<Tensor[]> normWeights; // original weight tensors
        private Lazy<Tensor[]> denormWeights; // weight clones
        private Lazy<Tensor[]> u;
        
        /// <summary>
        /// Applies spectral normalization <b>(Miyato et al., 2018)</b> for all parameter tensors with Rank 2 (weight matrices).<br></br> 
        /// Before SGD step, use <see cref="Optimizer.SpectralDenorm()"/>. <br></br> 
        /// If <paramref name="iterations"/>=0, spectral norm does not apply (inactive). <br></br>
        /// <br></br>
        /// Usage example: <br></br><br></br>
        /// optim.<b><i>SpectralNorm()</i></b> <br></br>
        /// net.<b>Forward()</b> <br></br>
        /// net.<b>Backward(loss.Grad)</b> <br></br>
        /// optim.<b><i>SpectralDenorm()</i></b> <br></br>
        /// optim.<b>Step</b>()
        /// </summary>
        /// <param name="iterations">Number of power iterations to calculate spectral norm.</param>
        public void SpectralNorm(int iterations = 1, float eps = 1E-12F)
        {
            if (iterations == 0)
                return;      
            if (iterations < 0)
                throw new ArgumentException($"Iterations cannot be less than 1 (received {iterations}).");
            if (epsilon <= 0)
                throw new ArgumentException($"Epsilon cannot be less or equal to 0 (received eps={eps})");
            
            // Lazy init
            if (u == null)
            {
                Tensor[] weight_m = parameters.Where(x => x.param?.Rank == 2).Select(x => x.param).ToArray();

                normWeights = new Lazy<Tensor[]>(() => weight_m);
                u = new Lazy<Tensor[]>(() => new Tensor[weight_m.Length]);

                for (int i = 0; i < normWeights.Value.Length; i++)
                {
                    u.Value[i] = Tensor.RandomNormal(normWeights.Value[i].Size(0));
                }
            }

            denormWeights = new Lazy<Tensor[]>(() => normWeights.Value.Select(x =>(Tensor)x.Clone()).ToArray());

            Parallel.For(0, normWeights.Value.Length, l =>
            {
                Tensor W = normWeights.Value[l];
                Tensor v = null;

                for (int __i = 0; __i < iterations; __i++)
                {
                    Tensor WT = W.Transpose(0, 1);
                    Tensor WT_u = Tensor.MatMul(WT, u.Value[l]);
                    float WT_u_norm = Tensor.Norm(WT_u, NormType.EuclideanL2, eps)[0];
                    v = WT_u / WT_u_norm;

                    Tensor W_v = Tensor.MatMul(W, v);
                    float W_v_norm = Tensor.Norm(W_v)[0];
                    Tensor.CopyTo(W_v / W_v_norm, u.Value[l]);
                }

                Tensor uT = u.Value[l].Unsqueeze(0);
                Tensor uTW = Tensor.MatMul(uT, W);
                float sigma = Tensor.MatMul(uTW, v)[0];

                Tensor.CopyTo(W / sigma, W);
            });          
        }

        /// <summary>
        /// Applies old weight values to the Rank 2 parameters that were previously spectral normalized (denormalization). Applied after forward and backward pass, before SGD (or it's variations) step.
        /// </summary>
        public void SpectralDenorm()
        {
            if (denormWeights == null)
                return;

            for (int i = 0; i < normWeights.Value.Length; i++)
            {
                Tensor.CopyTo(denormWeights.Value[i], normWeights.Value[i]);
            }

            denormWeights = null;
        }
    
    
    
    }

}


