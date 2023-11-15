using System;

namespace DeepUnity
{
    public static class Initializer
    {
        public static Tensor InitializeParameter(int[] gammaShape, int fan_in, int fan_out, InitType initializer)
        {
            switch (initializer)
            {
                case InitType.HE_Normal:
                    float sigmaHE = MathF.Sqrt(2f / fan_in);
                    return Tensor.RandomNormal((0, sigmaHE), gammaShape);
                    break;
                case InitType.HE_Uniform:
                    float bound = MathF.Sqrt(6f / fan_in);
                    return Tensor.RandomRange((-bound, bound), gammaShape);
                    break;
                case InitType.Glorot_Normal:
                    float sigmaXA = MathF.Sqrt(2f / (fan_in + fan_out));
                    return Tensor.RandomNormal((0, sigmaXA), gammaShape);
                    break;
                case InitType.Glorot_Uniform:
                    float limit = MathF.Sqrt(6f / (fan_in + fan_out));
                    return Tensor.RandomRange((-limit, limit), gammaShape);
                    break;
                case InitType.LeCun_Uniform:
                    float sqrtK = MathF.Sqrt(1f / fan_in);
                    return Tensor.RandomRange((-sqrtK, sqrtK), gammaShape);
                    break;
                case InitType.LeCun_Normal:
                    float sigmaLC = MathF.Sqrt(3f / fan_in);
                    return Tensor.RandomNormal((0, sigmaLC), gammaShape);
                    break;
                case InitType.Random_Normal:
                    return Tensor.RandomNormal(gammaShape);
                    break;
                case InitType.Random_Uniform:
                    return Tensor.RandomRange((-1f, 1f), gammaShape);
                    break;
                case InitType.Ones:
                    return Tensor.Ones(gammaShape);
                    break;
                case InitType.Zeros:
                    return Tensor.Zeros(gammaShape);
                    break;
                default:
                    throw new NotImplementedException("Unhandled initialization type!");
            }
        }
    }


}

