using System;
using System.Threading.Tasks;
using DeepUnity.Modules;
using UnityEngine;

namespace DeepUnity.Activations
{
    /// <summary>
    /// GELU_Activation_Function_in_Deep_Learning_A_Compre.pdf
    /// Minhyeok Lee
    /// 1School of Electrical and Electronics Engineering, Chung-Ang University, Seoul 06974, Korea
    /// mlee @cau.ac.kr
    /// </summary>
    [Serializable]
    public sealed class GELU : IModule, IActivation
    {
        [SerializeField] private bool inPlace = false;
        private const float sqrt_2overPI = 0.79788456080286535587989211986876f;
        private Tensor InputCache { get; set; }


        /// <summary>
        /// <b>Applies the GELU activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape. 
        /// </summary>
        /// <param name="in_place">Modifies the input tensor in place.</param>
        public GELU(bool in_place = false)
        {
            this.inPlace = in_place;
        }
       
        public Tensor Predict(Tensor x)
        {
            if(inPlace)
            {
                Parallel.For(0, x.Count(), i =>
                {
                   float tanh_ = Utils.Hyperbolics.Tanh(sqrt_2overPI * (x[i] + 0.044715f * MathF.Pow(x[i],3f)));
                    x[i] = 0.5f * x[i] * (1f + tanh_);
                });
                return x;
            }

            Tensor output = Tensor.Zeros(x.Shape);
            Parallel.For(0, x.Count(), i =>
            {
                float tanh_ = Utils.Hyperbolics.Tanh(sqrt_2overPI * (x[i] + 0.044715f * MathF.Pow(x[i], 3f)));
                output[i] = 0.5f * x[i] * (1f + tanh_);
            });
            return output;
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }

        public Tensor Backward(Tensor dLdY)
        {
            Tensor inputGrad = Tensor.Zeros(dLdY.Shape);
            Parallel.For(0, InputCache.Count(), i =>
            {
                float _elem = sqrt_2overPI * (InputCache[i] + 0.044715f * MathF.Pow(InputCache[i], 3f));
                float sech_ = Utils.Hyperbolics.Sech(_elem);
                float tanh_ = Utils.Hyperbolics.Tanh(_elem);
                float dgelu = 0.5f * (1f + tanh_) + 0.5f * InputCache[i] * sech_ * sech_;
                inputGrad[i] = dLdY[i] * dgelu;
            });
            return inputGrad;
        }

        public object Clone() => new GELU(in_place:inPlace);
    }
}




