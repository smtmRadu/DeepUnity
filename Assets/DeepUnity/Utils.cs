using System;
namespace DeepUnity
{
    public static class Utils
    {
        public static readonly float EPSILON = 1e-8f;
        private static System.Random RNG = new System.Random(DateTime.Now.Millisecond);

        public static float LogDensity(float x, float mu, float sigma)
        {
            //* https://stats.stackexchange.com/questions/404191/what-is-the-log-of-the-pdf-for-a-normal-distribution *//
            float frac = (x - mu) / sigma;
            float elem1 = MathF.Log(sigma);
            float elem2 = 0.5f * MathF.Log(2.0f * MathF.PI);
            float elem3 = 0.5f * frac * frac;
            return -elem1 - elem2 - elem3;
        }
        public static float Density(float x, float mu, float sigma)
        {
            float part1 = 1.0f / (sigma * MathF.Sqrt(2 * MathF.PI));
            float std = (x - mu) / sigma;
            float part2 = -0.5f * std * std;
            return part1 * MathF.Exp(part2);
        }




        public static class Random
        {        
            public static float Value { get { lock (RNG) return (float) RNG.NextDouble(); } }
            public static float Gaussian(float mean, float stddev, out float entropy)
            {
                float x1 = 1.0f - Value;
                float x2 = 1.0f - Value;

                entropy = MathF.Sqrt(-2.0f * MathF.Log(x1)) * MathF.Cos(2.0f * MathF.PI * x2);
                return entropy * stddev + mean;
            }
        }
    }

}
