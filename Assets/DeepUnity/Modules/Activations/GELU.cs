using System;
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
        private static float sqrt2OverPI = 0.79788456080286535587989211986876f;
        private Tensor InputCache { get; set; }


        /// <summary>
        /// <b>Applies the GELU activation function using a negative slope of <paramref name="alpha"/>. </b><br></br>
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
                float tanh_;
                for (int i = 0; i < x.Count(); i++)
                {
                    tanh_ = Utils.Hyperbolics.Tanh(sqrt2OverPI * (x[i] + 0.044715f * x[i] * x[i] * x[i]));
                    x[i] = 0.5f * x[i] * (1f + tanh_);
                }
                return x;
            }
            return x.Select(x =>
            {
                float tanh_ = Utils.Hyperbolics.Tanh(sqrt2OverPI * (x + 0.044715f * x * x * x));
                float gelu = 0.5f * x * (1f + tanh_);
                return gelu;
            });
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * InputCache.Select(x =>
            {
                float _elem = sqrt2OverPI * (x + 0.044715f * x * x * x);
                float sech_ = Utils.Hyperbolics.Sech(_elem);
                float tanh_ = Utils.Hyperbolics.Tanh(_elem);
                float dgelu = 0.5f * (1f + tanh_) + 0.5f * x * sech_ * sech_;
                return dgelu;
            });
        }

        public object Clone() => new GELU(in_place:inPlace);
    }
}




