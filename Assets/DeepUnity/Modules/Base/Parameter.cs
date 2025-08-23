using System;
using UnityEngine;

// Note, in practice, it seems like the uniform distribution is better than normal distribution for this
namespace DeepUnity.Modules
{
    /// <summary>
    /// The parameter of a learnable layer. 
    /// It's defined by <b>theta</b> and it's gradient <b>g</b>.
    /// </summary>
    public class Parameter : ICloneable
    {
        public Tensor param;
        public Tensor g;

        public TensorGPU paramGPU;
        public TensorGPU gGPU;

        public Device Device => paramGPU != null ? Device.GPU : Device.CPU;

        public Parameter(Tensor param, Tensor grad)
        {
            this.param = param;
            this.g = grad;
        }
        public Parameter(TensorGPU param, TensorGPU grad)
        {
            this.paramGPU = param;
            this.gGPU = grad;
        }

        /// <summary>
        /// Initializes a parameter tensor given <paramref name="fan_in"/>, <paramref name="fan_out"/> and the <paramref name="initializer"/> type.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="fan_in"></param>
        /// <param name="fan_out"></param>
        /// <param name="initializer"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="NotImplementedException"></exception>
        public static Tensor Create(int[] shape, int fan_in, int fan_out, InitType initializer)
        {
            switch (initializer)
            {
                case InitType.Zeros:
                    return Tensor.Zeros(shape);

                case InitType.Kaiming_Normal:
                    float sigmaHE = MathF.Sqrt(2f / fan_in);
                    return Tensor.RandomNormal((0, sigmaHE), shape);

                case InitType.Kaiming_Uniform:
                    float bound = MathF.Sqrt(6f / fan_in);
                    return Tensor.RandomRange((-bound, bound), shape);
           
                case InitType.Xavier_Normal:
                    float sigmaXA = MathF.Sqrt(2f / (fan_in + fan_out));
                    return Tensor.RandomNormal((0, sigmaXA), shape);
                
                case InitType.Xavier_Uniform:
                    float limit = MathF.Sqrt(6f / (fan_in + fan_out));
                    return Tensor.RandomRange((-limit, limit), shape);

                case InitType.LeCun_Normal:
                    float sigmaLC = MathF.Sqrt(1f / fan_in);
                    return Tensor.RandomNormal((0, sigmaLC), shape);

                case InitType.LeCun_Uniform:
                    float sqrtK = MathF.Sqrt(3f / fan_in);
                    return Tensor.RandomRange((-sqrtK, sqrtK), shape);
                

                case InitType.Normal:
                    return Tensor.RandomNormal(shape);
                case InitType.Normal0_1:
                    return Tensor.RandomNormal((0, 0.1f), shape);
                case InitType.Normal0_01:
                    return Tensor.RandomNormal((0, 0.01f), shape);
                case InitType.Normal0_001:
                    return Tensor.RandomNormal((0, 0.001f), shape);

                case InitType.Uniform:
                    return Tensor.RandomRange((-1f, 1f), shape);
                case InitType.Uniform0_1:
                    return Tensor.RandomRange((-0.1f, 0.1f), shape);
                case InitType.Uniform0_01:
                    return Tensor.RandomRange((-0.01f, 0.01f), shape);
                case InitType.Uniform0_001:
                    return Tensor.RandomRange((-0.001f, 0.001f), shape);

                case InitType.Ones:
                    return Tensor.Ones(shape);

                case InitType.Orthogonal:
                    if (shape.Length != 2) throw new ArgumentException("Orthogonal initialization can be used only for 2 dimensional parameter tensors.");
                    /// A = QR
                    Tensor A = Tensor.RandomNormal(shape);
                    Tensor Q = Tensor.Zeros(A.Shape);
                    Tensor R = Tensor.Zeros(A.Size(-1), A.Size(-1));
                    Tensor[] a_s = Tensor.Split(A, 1, 1);
                    Tensor[] q_s = Tensor.Split(Q, 1, 1);
                    for (int j = 0; j < A.Size(-1); j++)
                    {
                        Tensor v = a_s[j].Clone() as Tensor;
                        for (int i = 0; i < j; i++)
                        {
                            R[i, j] = (q_s[i] * a_s[j]).Sum(-2)[0];
                            v -= R[i, j] * q_s[i];
                        }
                        R[j, j] = Tensor.Norm(v)[0];
                        q_s[j] = v / R[j, j];
                    }
                    return Q;

                default:
                    throw new NotImplementedException("Unhandled initialization type!");
            }
        }
        public static TensorGPU CreateOnGPU(int[] shape, int fan_in, int fan_out, InitType initializer)
        {
            switch (initializer)
            {
                case InitType.Kaiming_Normal:
                    float sigmaHE = MathF.Sqrt(2f / fan_in);
                    return TensorGPU.RandomNormal((0, sigmaHE), shape);
                    
                case InitType.Kaiming_Uniform:
                    float bound = MathF.Sqrt(6f / fan_in);
                    return TensorGPU.RandomRange((-bound, bound), shape);

                case InitType.Xavier_Normal:
                    float sigmaXA = MathF.Sqrt(2f / (fan_in + fan_out));
                    return TensorGPU.RandomNormal((0, sigmaXA), shape);

                case InitType.Xavier_Uniform:
                    float limit = MathF.Sqrt(6f / (fan_in + fan_out));
                    return TensorGPU.RandomRange((-limit, limit), shape);


                case InitType.LeCun_Normal:
                    float sigmaLC = MathF.Sqrt(1f / fan_in);
                    return TensorGPU.RandomNormal((0, sigmaLC), shape);

                case InitType.LeCun_Uniform:
                    float sqrtK = MathF.Sqrt(3f / fan_in);
                    return TensorGPU.RandomRange((-sqrtK, sqrtK), shape);

              

                case InitType.Normal:
                    return TensorGPU.RandomNormal(shape);
                case InitType.Normal0_1:
                    return TensorGPU.RandomNormal((0, 0.1f), shape);
                case InitType.Normal0_01:
                    return TensorGPU.RandomNormal((0, 0.01f), shape);
                case InitType.Normal0_001:
                    return TensorGPU.RandomNormal((0, 0.001f), shape);

                case InitType.Uniform:
                    return TensorGPU.RandomRange((-1f, 1f), shape);
                case InitType.Uniform0_1:
                    return TensorGPU.RandomRange((-0.1f, 0.1f), shape);
                case InitType.Uniform0_01:
                    return TensorGPU.RandomRange((-0.01f, 0.01f), shape);
                case InitType.Uniform0_001:
                    return TensorGPU.RandomRange((-0.001f, 0.001f), shape);

                case InitType.Ones:
                    return TensorGPU.Ones(shape);

                case InitType.Zeros:
                    return TensorGPU.Zeros(shape);

                default:
                    throw new NotImplementedException("Unhandled initialization type!");
            }
        }

        public object Clone()
        {
            if(Device == Device.CPU)
                return new Parameter(param.Clone() as Tensor, g.Clone() as Tensor);

            return new Parameter(paramGPU.Clone() as Tensor, gGPU.Clone() as Tensor);

        }
    }
}



