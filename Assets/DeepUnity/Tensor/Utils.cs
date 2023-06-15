using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace DeepUnity
{
    public static class Utils
    {
        public static readonly float EPSILON = 1e-8f;
        private static System.Random RNG = new System.Random(DateTime.Now.Millisecond);

        public static T[] Shuffle<T>(T[] collection)
        {
            lock (RNG)
            {
                for (int i = collection.Length - 1; i > 0; i--)
                {
                    int r = RNG.Next(i + 1);
                    T temp = collection[i];
                    collection[i] = collection[r];
                    collection[r] = temp;
                }
            }

            return collection;
        }
        public static bool IsValueIn<T>(T value, IEnumerable<T> collection)
        {
            foreach (var item in collection)
            {
                if (value.Equals(item)) return true;
            }
            return false;
        }
        public static void DebugInFile(string text, bool append = true)
        {
            string desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            string filePath = Path.Combine(desktopPath, "UnityDebug.txt");

            using (StreamWriter sw = new StreamWriter(filePath, append))
            {
                sw.Write(text);
            }
        }
        public static void Swap<T>(ref T obj1, ref T obj2)
        {
            T temp = obj1;
            obj1 = obj2;
            obj2 = temp;
        }
        public static int ArgMax<T>(T[] values) where T : struct, IConvertible
        {
            int index = -1;
            double max = double.MinValue;
            for (int i = 0; i < values.Length; i++)
                if ((dynamic)values[i] > max)
                {
                    max = (dynamic)values[i];
                    index = i;
                }
            return index;
        }
        public static T[] FlatOf<T>(T[,] matrix)
        {
            int w = matrix.GetLength(0);
            int h = matrix.GetLength(1);

            T[] flat = new T[w * h];
            int ind = 0;
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    flat[ind++] = matrix[j, i];
                }
            }

            return flat;
        }
        public static T[,] MatrixOf<T>(T[] flat, int w, int h)
        {
            T[,] matrix = new T[w, h];
            int ind = 0;
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    matrix[j, i] = flat[ind++];
                }
            }
            return matrix;
        }
        public static string StringOf(IEnumerable collection, string tag = null)
        {
            StringBuilder sb = new StringBuilder();
            if (tag != null)
                sb.Append($"[{tag}]");

            sb.Append("[");
            foreach (var item in collection)
            {
                sb.Append(item.ToString());
                sb.Append(", ");
            }
            sb.Remove(sb.Length - 2, 2);
            sb.Append("]");
            return sb.ToString();
        }
        public static string HexOf(float r, float g, float b)
        {
            int ri = (int)(r * 255.0f);
            int gi = (int)(g * 255.0f);
            int bi = (int)(b * 255.0f);

            return string.Format("#{0:X2}{1:X2}{2:X2}", ri, gi, bi);
        }
        public static string HexOf(int r, int g, int b)
        {
            return string.Format("#{0:X2}{1:X2}{2:X2}", r, g, b);
        }

        public static class Numerics
        {
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


        }
        public static class Random
        {        
            /// <summary>
            /// Returns a value in range [0, 1)
            /// </summary>
            public static float Value { get { lock (RNG) return (float) RNG.NextDouble(); } }
            public static float Range(float min, float max) => Value * (max - min) + min;
            public static bool Bernoulli(float p = 0.5f) => Value < p;
            public static float Gaussian(float mean = 0f, float stddev = 1f)
            {
                float x1 = 1.0f - Value;
                float x2 = 1.0f - Value;

                var entropy = MathF.Sqrt(-2.0f * MathF.Log(x1)) * MathF.Cos(2.0f * MathF.PI * x2);
                return entropy * stddev + mean;
            }
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
