using System;

namespace DeepUnity
{
    public static class Initializer
    {
        public static Tensor CreateParameter(int[] shape, int fan_in, int fan_out, InitType initializer)
        {
            switch (initializer)
            {
                case InitType.HE_Normal:
                    float sigmaHE = MathF.Sqrt(2f / fan_in);
                    return Tensor.RandomNormal((0, sigmaHE), shape);
                   
                case InitType.HE_Uniform:
                    float bound = MathF.Sqrt(6f / fan_in);
                    return Tensor.RandomRange((-bound, bound), shape);
                    
                case InitType.Glorot_Normal:
                    float sigmaXA = MathF.Sqrt(2f / (fan_in + fan_out));
                    return Tensor.RandomNormal((0, sigmaXA), shape);
                    
                case InitType.Glorot_Uniform:
                    float limit = MathF.Sqrt(6f / (fan_in + fan_out));
                    return Tensor.RandomRange((-limit, limit), shape);
                    
                case InitType.LeCun_Uniform:
                    float sqrtK = MathF.Sqrt(1f / fan_in);
                    return Tensor.RandomRange((-sqrtK, sqrtK), shape);
                    
                case InitType.LeCun_Normal:
                    float sigmaLC = MathF.Sqrt(3f / fan_in);
                    return Tensor.RandomNormal((0, sigmaLC), shape);
                   
                case InitType.Random_Normal:
                    return Tensor.RandomNormal(shape);
                    
                case InitType.Random_Uniform:
                    return Tensor.RandomRange((-1f, 1f), shape);
                    
                case InitType.Ones:
                    return Tensor.Ones(shape);
                   
                case InitType.Zeros:
                    return Tensor.Zeros(shape);
                   
                default:
                    throw new NotImplementedException("Unhandled initialization type!");
            }
        }


        public static TensorGPU CreateParameterGPU(int[] shape, int fan_in, int fan_out, InitType initializer)
        {
            switch (initializer)
            {
                case InitType.HE_Normal:
                    float sigmaHE = MathF.Sqrt(2f / fan_in);
                    return TensorGPU.RandomNormal((0, sigmaHE), shape);

                case InitType.HE_Uniform:
                    float bound = MathF.Sqrt(6f / fan_in);
                    return TensorGPU.RandomRange((-bound, bound), shape);

                case InitType.Glorot_Normal:
                    float sigmaXA = MathF.Sqrt(2f / (fan_in + fan_out));
                    return TensorGPU.RandomNormal((0, sigmaXA), shape);

                case InitType.Glorot_Uniform:
                    float limit = MathF.Sqrt(6f / (fan_in + fan_out));
                    return TensorGPU.RandomRange((-limit, limit), shape);

                case InitType.LeCun_Uniform:
                    float sqrtK = MathF.Sqrt(1f / fan_in);
                    return TensorGPU.RandomRange((-sqrtK, sqrtK), shape);

                case InitType.LeCun_Normal:
                    float sigmaLC = MathF.Sqrt(3f / fan_in);
                    return TensorGPU.RandomNormal((0, sigmaLC), shape);

                case InitType.Random_Normal:
                    return TensorGPU.RandomNormal(shape);

                case InitType.Random_Uniform:
                    return TensorGPU.RandomRange((-1f, 1f), shape);

                case InitType.Ones:
                    return TensorGPU.Ones(shape);

                case InitType.Zeros:
                    return TensorGPU.Zeros(shape);

                default:
                    throw new NotImplementedException("Unhandled initialization type!");
            }
        }
    }


}

