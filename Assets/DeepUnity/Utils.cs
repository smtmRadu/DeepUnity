using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    public static class Utils
    {
        public const float EPSILON = 1e-8f;
        private static System.Random RNG = new System.Random(DateTime.Now.Millisecond);

        public static void Shuffle<T>(T[] arrayToShuffle)
        {
            lock (RNG)
            {
                for (int i = arrayToShuffle.Length - 1; i > 0; i--)
                {
                    int r = RNG.Next(i + 1);
                    T temp = arrayToShuffle[i];
                    arrayToShuffle[i] = arrayToShuffle[r];
                    arrayToShuffle[r] = temp;
                }
            }
        }
        public static void Shuffle<T>(List<T> listToShuffle)
        {
            lock (RNG)
            {
                for (int i = listToShuffle.Count - 1; i > 0; i--)
                {
                    int r = RNG.Next(i + 1);
                    T temp = listToShuffle[i];
                    listToShuffle[i] = listToShuffle[r];
                    listToShuffle[r] = temp;
                }
            }
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
                if (Convert.ToDouble(values[i]) > max)
                {
                    max = Convert.ToDouble(values[i]);
                    index = i;
                }
            return index;
        }
        public static T[] MatrixToVector<T>(T[,] matrix)
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
        public static T[,] VectorToMatrix<T>(T[] vector, int w, int h)
        {
            T[,] matrix = new T[w, h];
            int ind = 0;
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    matrix[j, i] = vector[ind++];
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
        public static Array GetRange(Array arr, int start, int count)
        {
            if (arr == null)
                throw new ArgumentNullException(nameof(arr));

            if (start < 0 || start >= arr.Length)
                throw new ArgumentOutOfRangeException(nameof(start));

            if (count < 0 || start + count > arr.Length)
                throw new ArgumentOutOfRangeException(nameof(count));

            Array result = Array.CreateInstance(arr.GetType().GetElementType(), count);
            Array.Copy(arr, start, result, 0, count);
            return result;
        }
        public static List<T[]> Split<T>(IEnumerable<T> collection, int split_size)
        {
            List<T[]> slices = new();
            int length = collection.Count();

            int index = 0;
            while(index < length)
            {
                int sliceSize = Math.Min(split_size, length - index);
                T[] slice = new T[sliceSize];
                for (int i = 0; i < sliceSize; i++)
                {
                    slice[i] = collection.ElementAt(index++);
                }

                slices.Add(slice);
            }

            return slices;
        }
        public static Texture2D TensorToTexture(Tensor tensor)
        {
            if (tensor.Shape.Length != 3)
                throw new ShapeException($"Cannot convert tensor ({tensor.Shape}) to a Texture2D because the shape must be (C, H, W)");

            int width = tensor.Size(2);
            int height = tensor.Size(1);
            int channels = tensor.Size(0);

            Texture2D tex = new Texture2D(width, height);

            Color[] colors = new Color[width * height];

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    if (channels == 1)
                    {
                        colors[h * width + w] = new Color(tensor[0, h, w], tensor[0, h, w], tensor[0, h, w]);
                    }
                    else if(channels == 2)
                    {
                        colors[h * width + w] = new Color(tensor[0, h, w], tensor[0, h, w], tensor[0, h, w], tensor[1, h, w]);
                    }
                    else if (channels == 3)
                    {
                        colors[h * width + w] = new Color(tensor[0, h, w], tensor[1, h, w], tensor[2, h, w]);
                    }
                    else if (channels == 4)
                    {
                        colors[h * width + w] = new Color(tensor[0, h, w], tensor[1, h, w], tensor[2, h, w], tensor[3, h, w]);
                    }
                }
            }

            tex.SetPixels(colors);
            tex.Apply();

            return tex;
        }

        public static class ImageProcessing
        {
            /// <summary>
            /// Scales the texture rezolution.
            /// </summary>
            /// <param name="texture"></param>
            /// <param name="scale"></param>
            /// <returns></returns>
            public static Texture2D Resize(Texture2D texture, float scale)
            {
                int width = Mathf.FloorToInt(texture.width * scale);
                int height = Mathf.FloorToInt(texture.height * scale);
                Texture2D scaledTexture = new Texture2D(width, height);




                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        float x0 = x / scale;
                        float y0 = y / scale;

                        Color pixel = texture.GetPixelBilinear(x0 / texture.width, y0 / texture.height);
                        scaledTexture.SetPixel(x, y, pixel);
                    }
                }

                scaledTexture.Apply();
                scaledTexture.filterMode = texture.filterMode;
                return scaledTexture;
            }
            public static Texture2D Rotate(Texture2D texture, float angle)
            {
                int width = texture.width;
                int height = texture.height;
                Texture2D rotatedTexture = new Texture2D(width, height);

                Vector2 pivot = new Vector2(width * 0.5f, height * 0.5f); // Center pivot point

                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        Color pixel = texture.GetPixel(x, y);
                        float radianAngle = angle * Mathf.Deg2Rad;
                        float cos = Mathf.Cos(radianAngle);
                        float sin = Mathf.Sin(radianAngle);

                        // Calculate coordinates relative to the pivot
                        float xOffset = x - pivot.x;
                        float yOffset = y - pivot.y;

                        // Apply rotation around the pivot
                        float x0 = xOffset * cos - yOffset * sin + pivot.x;
                        float y0 = xOffset * sin + yOffset * cos + pivot.y;

                        x0 = Mathf.Clamp(x0, 0, width - 1);
                        y0 = Mathf.Clamp(y0, 0, height - 1);

                        rotatedTexture.SetPixel(x, y, texture.GetPixel(Mathf.FloorToInt(x0), Mathf.FloorToInt(y0)));
                    }
                }

                rotatedTexture.Apply();
                rotatedTexture.filterMode = texture.filterMode;
                return rotatedTexture;
            }
            public static Texture2D Zoom(Texture2D texture, float zoomFactor)
            {
                if(zoomFactor <= 0f)
                {
                    throw new ArgumentException($"Zoom Factor (received value: {zoomFactor}) cannot be equal or less than 0");
                }
                int width = texture.width;
                int height = texture.height;
                Texture2D zoomedTexture = new Texture2D(width, height);

                zoomedTexture.filterMode = texture.filterMode;

                float centerX = width / 2f;
                float centerY = height / 2f;

                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        float offsetX = (x - centerX) / zoomFactor;
                        float offsetY = (y - centerY) / zoomFactor;

                        int originalX = Mathf.RoundToInt(centerX + offsetX);
                        int originalY = Mathf.RoundToInt(centerY + offsetY);

                        originalX = Mathf.Clamp(originalX, 0, width - 1);
                        originalY = Mathf.Clamp(originalY, 0, height - 1);

                        Color pixel = texture.GetPixel(originalX, originalY);
                        zoomedTexture.SetPixel(x, y, pixel);
                    }
                }

                zoomedTexture.Apply();
                return zoomedTexture;
            }
            public static Texture2D Offset(Texture2D texture, float x, float y)
            {
                int width = texture.width;
                int height = texture.height;
                Texture2D offsetTexture = new Texture2D(width, height);

                for (int destX = 0; destX < width; destX++)
                {
                    for (int destY = 0; destY < height; destY++)
                    {
                        int srcX = Mathf.Clamp(destX - Mathf.FloorToInt(x), 0, width - 1);
                        int srcY = Mathf.Clamp(destY - Mathf.FloorToInt(y), 0, height - 1);
                        offsetTexture.SetPixel(destX, destY, texture.GetPixel(srcX, srcY));
                    }
                }

                offsetTexture.Apply();
                offsetTexture.filterMode = texture.filterMode;
                return offsetTexture;
            }
            public static Texture2D Noise(Texture2D texture, float noise_prob, float noise_size)
            {
                int width = texture.width;
                int height = texture.height;
                Texture2D noisyTexture = new Texture2D(width, height);

                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        Color pixel = texture.GetPixel(x, y);
                        if (UnityEngine.Random.value < noise_prob)
                        {
                            float intensity = UnityEngine.Random.Range(-noise_size, noise_size);
                            Color noise = new Color(intensity, intensity, intensity);
                            pixel += noise;
                            pixel = Color.Lerp(pixel, Color.black, Mathf.Abs(intensity));
                        }
                        noisyTexture.SetPixel(x, y, pixel);
                    }
                }

                noisyTexture.Apply();
                noisyTexture.filterMode = texture.filterMode;
                return noisyTexture;
            }
            public static Texture2D Mask(Texture2D texture, Texture2D mask)
            {
                int width = texture.width;
                int height = texture.height;
                Texture2D maskedTexture = new Texture2D(width, height);

                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        Color texturePixel = texture.GetPixel(x, y);
                        Color maskPixel = mask.GetPixel(x % mask.width, y % mask.height);
                        Color maskedColor = new Color(texturePixel.r * maskPixel.r, texturePixel.g * maskPixel.g, texturePixel.b * maskPixel.b, texturePixel.a * maskPixel.a);
                        maskedTexture.SetPixel(x, y, maskedColor);
                    }
                }

                maskedTexture.Apply();
                maskedTexture.filterMode = texture.filterMode;
                return maskedTexture;
            }
            public static Texture2D Blur(Texture2D texture, int kernel_size)
            {
                if (kernel_size % 2 == 0 || kernel_size <= 0)
                {
                    throw new ArgumentException("Kernel size must be an odd positive number.");
                }

                int width = texture.width;
                int height = texture.height;
                Color[] originalPixels = texture.GetPixels();
                Color[] blurredPixels = new Color[width * height];

                int halfKernel = kernel_size / 2;

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        float r = 0f, g = 0f, b = 0f, a = 0f;

                        for (int ky = -halfKernel; ky <= halfKernel; ky++)
                        {
                            for (int kx = -halfKernel; kx <= halfKernel; kx++)
                            {
                                int offsetX = x + kx;
                                int offsetY = y + ky;

                                if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height)
                                {
                                    Color pixel = originalPixels[offsetY * width + offsetX];
                                    r += pixel.r;
                                    g += pixel.g;
                                    b += pixel.b;
                                    a += pixel.a;
                                }
                            }
                        }

                        int numPixelsInKernel = kernel_size * kernel_size;
                        blurredPixels[y * width + x] = new Color(r / numPixelsInKernel, g / numPixelsInKernel, b / numPixelsInKernel, a / numPixelsInKernel);
                    }
                }

                Texture2D blurredTexture = new Texture2D(width, height);
                blurredTexture.SetPixels(blurredPixels);
                blurredTexture.Apply();
                blurredTexture.filterMode = texture.filterMode;

                return blurredTexture;
            }
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
            public static float Range(float minInclusive, float maxExclusive) => Value * (maxExclusive - minInclusive) + minInclusive;
            public static int Range(int minInclusive, int maxExclusive) => (int) (Value * (maxExclusive - minInclusive) + minInclusive);
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
