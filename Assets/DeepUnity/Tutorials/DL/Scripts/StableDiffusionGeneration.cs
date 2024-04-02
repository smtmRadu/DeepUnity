using DeepUnity;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class StableDiffusionGeneration : MonoBehaviour
    {
        public int timesteps = 50;
        Tensor betas;
        Tensor alphas;
        Tensor alphasBar;

        // Actually fuck of with this because we do not have the power to create a strong UNET

        private void Start()
        {
            betas = GenerateBetas(50);
            alphas = -betas + 1;
            alphasBar = Tensor.CumProd(alphas, 0);
            
            // I'm not sure but should i scale the images in range (-1, 1)?
        }

        private Tensor GenerateBetas(int timesteps, float start = 0.0001f, float end = 0.02f) // as in stable diffusion paper these are good ranges for linear betas, i should also check cosine from openai variant
        {
            return Tensor.LinSpace(start, end, timesteps);
        }

        private Tensor ForwardDiffusion(Tensor original_image, int timestep) // Q function
        {
            // reparametrization trick used here

            Tensor eps = Tensor.RandomNormal(original_image.Shape);
            Tensor q = Mathf.Sqrt(alphasBar[timestep]) * original_image + Mathf.Sqrt(1 - alphasBar[timestep]) * eps;
            return q;
        }
    }

}


