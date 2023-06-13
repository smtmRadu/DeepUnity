using UnityEngine;

namespace DeepUnity
{
    public readonly partial struct Kernels
    {
        public readonly partial struct Mean
        {
            internal static NDArray AsTensor
            {
                get => NDArray.Constant(AsMatrix);
            }
            internal static float[,] AsMatrix
            {
                get => new float[3, 3]
                {
                    {0.11111111f, 0.11111111f, 0.11111111f },
                    {0.11111111f, 0.11111111f, 0.11111111f },
                    {0.11111111f, 0.11111111f, 0.11111111f }
                };
            }
            internal static float[] AsArray
            {
                get => new float[9]
                {
                0.11111111f,
                0.11111111f,
                0.11111111f,
                0.11111111f,
                0.11111111f,
                0.11111111f,
                0.11111111f,
                0.11111111f,
                0.11111111f
                };
            }

        }
    }
}