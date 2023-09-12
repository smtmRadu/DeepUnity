using System;
/// <summary>
/// GELU_Activation_Function_in_Deep_Learning_A_Compre.pdf
/// Minhyeok Lee
/// 1School of Electrical and Electronics Engineering, Chung-Ang University, Seoul 06974, Korea
/// mlee @cau.ac.kr
/// </summary>
namespace DeepUnity
{
    
    [Serializable]
    public class GELU : Activation
    {
        private static float sqrt2OverPI = 0.79788456080286535587989211986876f;
        protected override Tensor Activate(Tensor x)
        {
            return x.Select(x =>
            {
                float gelu = 0.5f * x * (1f + Utils.Hyperbolics.Tanh(sqrt2OverPI * (x + 0.044715f * x * x * x)));
                return gelu;
            });
        }
        protected override Tensor Derivative(Tensor x)
        {
            return x.Select(X =>
            {
                float sech = Utils.Hyperbolics.Sech(sqrt2OverPI * (X + 0.044715f * X * X * X));
                float dgelu = 0.5f * (1f + Utils.Hyperbolics.Tanh(sqrt2OverPI * (X + 0.044715f * X * X * X))) +
                              0.5f * X * sech * sech;
                return dgelu;
            });
        }
    }
}




