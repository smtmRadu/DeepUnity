using System;
using DeepUnity.Modules;

namespace DeepUnity.Activations
{
    /// <summary>
    /// GELU_Activation_Function_in_Deep_Learning_A_Compre.pdf
    /// Minhyeok Lee
    /// 1School of Electrical and Electronics Engineering, Chung-Ang University, Seoul 06974, Korea
    /// mlee @cau.ac.kr
    /// </summary>
    [Serializable]
    public class GELU : IModule, IActivation
    {
        private static float sqrt2OverPI = 0.79788456080286535587989211986876f;


        protected Tensor InputCache { get; set; }
        public Tensor Predict(Tensor x)
        {
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

        public object Clone() => new GELU();
    }
}




