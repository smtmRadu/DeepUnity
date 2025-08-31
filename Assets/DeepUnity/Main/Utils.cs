using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;


namespace DeepUnity
{
    public static class Utils
    {
        /// <summary>
        /// A very small number for numerical stability. (1E-8)
        /// </summary>
        public const float EPSILON = 1e-8f;
        private static System.Random RNG = new System.Random(DateTime.Now.Millisecond);

        /// <summary>
        /// Checks whether or not we are on the current thread.
        /// </summary>
        /// <returns></returns>
        public static bool IsMainThread()
        {
            return Thread.CurrentThread.ManagedThreadId == 1;
        }
        /// <summary>
        /// Get the path to the machine's Desktop.
        /// </summary>
        /// <returns></returns>
        public static string GetDesktopPath()
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
        }
        /// <summary>
        /// Shuffles the elements of the given array.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="arrayToShuffle"></param>
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
        /// <summary>
        /// Shuffles the elements of the given list.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="listToShuffle"></param>
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
        /// <summary>
        /// Shuffles the elements of the given arrays using the same randomizer.
        /// </summary>
        /// <exception cref="ArgumentException"></exception>
        public static void PairwiseShuffle<T>(T[] array1, T[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentException("The two lists must have the same length");
            lock (RNG)
            {
                for (int i = array1.Length - 1; i > 0; i--)
                {
                    int r = RNG.Next(i + 1);
                    T temp = array1[i];
                    array1[i] = array1[r];
                    array1[r] = temp;

                    T temp2 = array2[i];
                    array2[i] = array2[r];
                    array2[r] = temp2;
                }
            }
        }
        /// <summary>
        /// Shuffles the elements of the given lists using the same randomizer.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list1"></param>
        /// <param name="list2"></param>
        /// <exception cref="ArgumentException"></exception>
        public static void PairwiseShuffle<T>(List<T> list1, List<T> list2)
        {
            if (list1.Count != list2.Count)
                throw new ArgumentException("The two lists must have the same length");

            lock (RNG)
            {
                for (int i = list1.Count - 1; i > 0; i--)
                {
                    int r = RNG.Next(i + 1);

                    T temp = list1[i];
                    list1[i] = list1[r];
                    list1[r] = temp;

                    T temp2 = list2[i];
                    list2[i] = list2[r];
                    list2[r] = temp2;
                }
            }
        }
        /// <summary>
        /// Debug some info in a unique file created on desktop automatically named 'DeepUnity-Debug.txt'.
        /// </summary>
        /// <param name="text"></param>
        /// <param name="append"></param>
        public static void DebugInFile(string text, bool append = true)
        {
            string desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            string filePath = Path.Combine(desktopPath, "DeepUnity-Debug.txt");

            using (StreamWriter sw = new StreamWriter(filePath, append))
            {
                sw.Write(text);
            }
        }
        /// <summary>
        /// Swap the values of two objects. Can be used on lists or so.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="obj1"></param>
        /// <param name="obj2"></param>
        public static void Swap<T>(ref T obj1, ref T obj2)
        {
            T temp = obj1;
            obj1 = obj2;
            obj2 = temp;
        }
        /// <summary>
        /// Get the index of the maximum element in the array. Note that the <see cref="T"/> must be convertible to double.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="values"></param>
        /// <returns></returns>
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
        /// <summary>
        /// Transforms a matrix into a vector.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="matrix"></param>
        /// <returns></returns>
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
        /// <summary>
        /// Transforms a vector into a matrix of size <see cref="w"/> * <see cref="h"/>.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="vector"></param>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <returns></returns>
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
        /// <summary>
        /// Transforms a collection into a string, optionally with a given tag at the beginning.
        /// </summary>
        /// <param name="collection"></param>
        /// <param name="tag"></param>
        /// <returns></returns>
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
        /// <summary>
        /// Returns the hex value of the given r, g & b values in range [0, 1].
        /// </summary>
        public static string HexOf(float r, float g, float b)
        {
            int ri = (int)(r * 255.0f);
            int gi = (int)(g * 255.0f);
            int bi = (int)(b * 255.0f);

            return string.Format("#{0:X2}{1:X2}{2:X2}", ri, gi, bi);
        }
        /// <summary>
        /// Returns the hex value of the given r, g & b values in range [0, 255].
        /// </summary>
        public static string HexOf(int r, int g, int b)
        {
            return string.Format("#{0:X2}{1:X2}{2:X2}", r, g, b);
        }
        /// <summary>
        /// Get a number of values from a certain range in the givven collection.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="collection"></param>
        /// <param name="start"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public static IEnumerable<T> GetRange<T>(IEnumerable<T> collection, int start, int count)
        {
            if (collection == null)
                throw new ArgumentNullException(nameof(collection));

            if (start < 0 || start >= collection.Count())
                throw new ArgumentOutOfRangeException(nameof(start));

            if (count < 0 || start + count > collection.Count())
                throw new ArgumentOutOfRangeException(nameof(count));

            return collection.Skip(start).Take(count);
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
        /// <summary>
        /// Input: Tensor <b>(C, H, W)</b> <br></br>
        /// Output: <b>Color[]</b> pixels used to load on Texture2D's.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns>Pixels array</returns>
        /// <exception cref="ShapeException"></exception>
        public static Color[] TensorToColorArray(Tensor tensor)
        {
            if (tensor.Shape.Length != 3)
                throw new ShapeException($"Cannot convert tensor ({tensor.Shape.ToCommaSeparatedString()}) to a Texture2D because the shape must be (C, H, W)");

            int width = tensor.Size(2);
            int height = tensor.Size(1);
            int channels = tensor.Size(0);

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

            return colors;
        }
        /// <summary>
        /// Clamps the value in the given interval.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static float Clip(float value, float min, float max) 
        {
            return Math.Clamp(value, min, max);
        }
        public static float DiracDelta(float value)
        {
            return value == 0 ? float.MaxValue : 0;
        }
        public static float KroneckerDelta(float a, float b)
        {
            return a == b ? 1 : 0;
        }
        public static float Sigmoid(float value)
        {
            return 1.0f / (1.0f + MathF.Exp(-value));
        }

        public static float[] ReadWeights(string path, int numWeights)
        {
            float[] weights = new float[numWeights];
            byte[] bytes = File.ReadAllBytes(path); // read file at once
            Buffer.BlockCopy(bytes, 0, weights, 0, bytes.Length);

            return weights;
        }

        /// <summary>
        /// A class that contains all hyberbolic functions
        /// </summary>
        public static class Hyperbolics
        {
            public static float Sinh(float x)
            {
                return (1f - MathF.Exp(-2f * x)) / (2f * MathF.Exp(-x));
            }
            public static float Cosh(float x)
            {
                return (1f + MathF.Exp(-2f * x)) / (2f * MathF.Exp(-x));
            }
            
            public static float Csch(float x)
            {
                return 2f * MathF.Exp(x) / (MathF.Exp(2f * x) - 1f);
            }
            public static float Sech(float x)
            {
                return 2f * MathF.Exp(x) / (MathF.Exp(2f * x) + 1f);
            }

            public static float Tanh(float x)
            {
                float e2x = MathF.Exp(2f * x);
                return (e2x - 1f) / (e2x + 1f);
            }
            public static float Coth(float x)
            {
                float e2x = MathF.Exp(2f * x);
                return (e2x + 1f) / (e2x - 1f);
            }
        }

        /// <summary>
        /// An easy way to modify images. You can input batch images as well.
        /// </summary>
        public static class Vision
        {
            public static Tensor Resize(Tensor image, float scale)
            {
                int width  = (int)MathF.Floor(image.Size(-1) * scale);
                int height = (int)MathF.Floor(image.Size(-2) * scale);
                int channels = image.Rank >= 3 ? image.Size(-3) : 1;
                int batch_size = image.Rank == 4 ? image.Size(-4) : 1;

                int[] newShape = image.Shape;
                newShape[newShape.Length - 1] = width;
                newShape[newShape.Length - 2] = height;
                Tensor scaled = Tensor.Zeros(newShape);

                Parallel.For(0, batch_size, b =>
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            for (int x = 0; x < width; x++)
                            {
                                float x0 = x / scale;
                                float y0 = y / scale;

                                float pix = image[b, c, (int)y0 / image.Size(-2), (int)x0 / image.Size(-1)];
                                scaled[b, c, y, x] = pix;
                            }
                        }
                    }
                });

                return scaled;
            }
            public static Tensor Rotate(Tensor image, float angle)
            {
                int width = image.Size(-1);
                int height = image.Size(-2);
                int channels = image.Rank >= 3 ? image.Size(-3) : 1;
                int batch_size = image.Rank == 4 ? image.Size(-4) : 1;

                Tensor rotated = Tensor.Zeros(image.Shape);

                Vector2 pivot = new Vector2(width * 0.5f, height * 0.5f); // Center pivot point

                Parallel.For(0, batch_size, b =>
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            for (int x = 0; x < width; x++)
                            {
                                float value = image[b, c, y, x];
                                float radianAngle = angle * Mathf.Deg2Rad;
                                float cos = MathF.Cos(radianAngle);
                                float sin = MathF.Sin(radianAngle);

                                // Calculate coordinates relative to the pivot
                                float xOffset = x - pivot.x;
                                float yOffset = y - pivot.y;

                                // Apply rotation around the pivot
                                float x0 = xOffset * cos - yOffset * sin + pivot.x;
                                float y0 = xOffset * sin + yOffset * cos + pivot.y;

                                x0 = Math.Clamp(x0, 0, width - 1);
                                y0 = Math.Clamp(y0, 0, height - 1);

                                rotated[b, c, y, x] = image[b, c, (int)MathF.Round(y0), (int)MathF.Round(x0)];
                            }
                        }
                    }
                });

                return rotated;
            }
            public static Tensor Zoom(Tensor image, float zoomFactor)
            {
                if (zoomFactor <= 0f)
                {
                    throw new ArgumentException($"Zoom Factor (received value: {zoomFactor}) cannot be equal or less than 0");
                }

                int width = image.Size(-1);
                int height = image.Size(-2);
                int channels = image.Rank >= 3 ? image.Size(-3) : 1;
                int batch_size = image.Rank == 4 ? image.Size(-4) : 1;

                Tensor zoomed = Tensor.Zeros(image.Shape);

                float centerX = width / 2f;
                float centerY = height / 2f;

                Parallel.For(0, batch_size, b =>
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            for (int x = 0; x < width; x++)
                            {
                                float offsetX = (x - centerX) / zoomFactor;
                                float offsetY = (y - centerY) / zoomFactor;

                                int originalX = (int)MathF.Round(centerX + offsetX);
                                int originalY = (int)MathF.Round(centerY + offsetY);

                                originalX = Math.Clamp(originalX, 0, width - 1);
                                originalY = Math.Clamp(originalY, 0, height - 1);

                                zoomed[b, c, y, x] = image[b, c, originalY, originalX];
                            }
                        }
                    }
                });

                return zoomed;
            }
            public static Tensor Offset(Tensor image, float x, float y)
            {
                int width = image.Size(-1);
                int height = image.Size(-2);
                int channels = image.Rank >= 3 ? image.Size(-3) : 1;
                int batch_size = image.Rank == 4 ? image.Size(-4) : 1;

                Tensor offsetImage = Tensor.Zeros(image.Shape);

                Parallel.For(0, batch_size, b =>
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int destX = 0; destX < width; destX++)
                        {
                            for (int destY = 0; destY < height; destY++)
                            {
                                int srcX = (int)Math.Clamp(destX - (int)MathF.Floor(x), 0f, width - 1f);
                                int srcY = (int)Math.Clamp(destY - (int)MathF.Floor(y), 0f, height - 1f);
                                offsetImage[b, c, destY, destX] = image[b, c, srcY, srcX];
                            }
                        }
                    }
                });

                return offsetImage;
            }
            public static Tensor Noise(Tensor image, float noise_prob, float noise_size)
            {
                int width = image.Size(-1);
                int height = image.Size(-2);
                int channels = image.Rank >= 3 ? image.Size(-3) : 1;
                int batch_size = image.Rank == 4 ? image.Size(-4) : 1;

                Tensor noisyImage = Tensor.Zeros(image.Shape);

                Parallel.For(0, batch_size, b =>
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            for (int x = 0; x < width; x++)
                            {
                                float pixel = image[b, c, y, x];
                                if (Random.Value < noise_prob)
                                {
                                    float intensity = Random.Range(-noise_size, noise_size);
                                    pixel += intensity;
                                    pixel = Math.Clamp(pixel, 0f, 1f); 
                                }
                                noisyImage[b, c, y, x] = pixel;
                            }
                        }
                    }
                });

                return noisyImage;
            }
            public static Tensor Mask(Tensor image, Tensor mask)
            {
                int width = image.Size(-1);
                int height = image.Size(-2);
                int channels = image.Rank >= 3 ? image.Size(-3) : 1;
                int batch_size = image.Rank == 4 ? image.Size(-4) : 1;
                int maskWidth = mask.Shape[2];
                int maskHeight = mask.Shape[1];

                Tensor maskedImage = Tensor.Zeros(image.Shape);

                Parallel.For(0, batch_size, b =>
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            for (int y = 0; y < height; y++)
                            {
                                float texturePixel = image[b, c, y, x];
                                float maskPixel = mask[b, c, y % maskHeight, x % maskWidth];
                                maskedImage[b, c, y, x] = texturePixel * maskPixel;
                            }
                        }
                    }
                });

                return maskedImage;
            }
            public static Tensor Blur(Tensor image, int kernel_size)
            {
                if (kernel_size % 2 == 0 || kernel_size <= 0)
                {
                    throw new ArgumentException("Kernel size must be an odd positive number.");
                }

                int width = image.Size(-1);
                int height = image.Size(-2);
                int channels = image.Rank >= 3 ? image.Size(-3) : 1;
                int batch_size = image.Rank == 4 ? image.Size(-4) : 1;

                Tensor blurredImage = Tensor.Zeros(image.Shape);

                int halfKernel = kernel_size / 2;

                Parallel.For(0, batch_size, ba =>
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            for (int x = 0; x < width; x++)
                            {
                                float v = 0f;

                                for (int ky = -halfKernel; ky <= halfKernel; ky++)
                                {
                                    for (int kx = -halfKernel; kx <= halfKernel; kx++)
                                    {
                                        int offsetX = x + kx;
                                        int offsetY = y + ky;

                                        if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height)
                                        {
                                            float pixel = image[ba, c, offsetY, offsetX];
                                            v += pixel;
                                        }
                                    }
                                }

                                int numPixelsInKernel = kernel_size * kernel_size;
                                blurredImage[ba, c, y, x] = v;
                            }
                        }
                    }
                });

                return blurredImage;
            }
        }

        /// <summary>
/// A thread-safe way to sample random numbers.
/// </summary>
        public static class Random
        {        
            // -- On tests, sampling numbers threadsafely is just 2 times less efficient than checking if we are on the main thread --//
            /// <summary>
            /// Set the seed of the RNG, including UnityEngine's generator. If not set, the seed is taken from the current time in ms.
            /// </summary>
            public static int Seed
            {
                set
                {
                    RNG = new System.Random(value);
                    UnityEngine.Random.InitState(value);
                }
            }
            /// <summary>
            /// Returns a float value in range [0, 1] thread-safely (range is inclusive).
            /// </summary>
            public static float Value 
            {
                // https://stackoverflow.com/questions/66680283/how-to-generate-a-random-double-number-in-the-inclusive-0-1-range
                get
                {
                    lock (RNG)
                    {
                        double d = 0;
                        int i = 0;

                        do
                        {
                            d = RNG.NextDouble();
                            i = RNG.Next(2);
                        } while (i == 1 && d > 0);

                        return (float)(d + i);

                        //return (float)RNG.NextDouble();
                    }
                } 
            }        
            /// <summary>
            /// Returns a float value in range [0, 1] non-safely (range is inclusive)
            /// </summary>
            public static float ValueUnsafe
            {
                get
                {
                    lock (RNG)
                    {
                        double d = 0;
                        int i = 0;

                        do
                        {
                            d = RNG.NextDouble();
                            i = RNG.Next(2);
                        } while (i == 1 && d > 0);

                        return (float)(d + i);

                        //return (float)RNG.NextDouble();
                    }
                }
            }
            /// <summary>
            /// Returns a float value in range [0, 1) thread-safely (range is exclusive)
            /// </summary>
            public static float ValueExclusive
            {
                get
                {
                    lock(RNG)
                        return (float)RNG.NextDouble();
                }
            }
            /// <summary>
            /// Returns a float value in range [0, 1) non-safely (range is exclusive)
            /// </summary>
            public static float ValueExclusiveUnsafe
            {
                get=> (float) RNG.NextDouble();
            }
            public static Vector2 OnUnitCircle
            {
                get
                {
                    float angle = Range(0f, 2f * MathF.PI - float.MinValue);
                    return new Vector2(MathF.Cos(angle), MathF.Sin(angle));
                }
            }
            public static Vector2 InUnitCircle
            {
                get
                {
                    float t = 2f * MathF.PI * Value - float.MinValue;
                    float u = Value + Value;
                    float r = u > 1 ? 2 - u : u;
                    return new Vector2(r * MathF.Cos(t), r * MathF.Sin(t));
                }
            }
            public static Vector3 OnUnitSphere
            {
                get
                {
                    float theta = Range(0f, 2f * MathF.PI - float.MinValue); 
                    float phi = Range(0f, MathF.PI - float.MinValue); 

                    float x = MathF.Sin(phi) * MathF.Cos(theta);
                    float y = MathF.Sin(phi) * MathF.Sin(theta);
                    float z = MathF.Cos(phi);

                    return new Vector3(x, y, z);
                }
            }
            public static Vector3 InUnitSphere
            {
                get
                {
                    while (true)
                    {
                        float x = Range(-1f, 1f);
                        float y = Range(-1f, 1f);
                        float z = Range(-1f, 1f);

                        if (x * x + y * y + z * z <= 1)
                        {
                            return new Vector3(x, y, z);
                        }
                    }
                }
            }

            /// <summary>
            /// Returns a float value in range [<paramref name="minInclusive"/>, <paramref name="maxInclusive"/>] thread-safely. (range is inclusive)
            /// </summary>
            /// <param name="minInclusive"></param>
            /// <param name="maxInclusive"></param>
            /// <returns></returns>
            public static float Range(float minInclusive, float maxInclusive) => Value * (maxInclusive - minInclusive) + minInclusive;
            /// <summary>
            /// Returns an integer value in range [<paramref name="minInclusive"/>, <paramref name="maxExclusive"/>) thread-safely. (range in exclusive on the right handside)
            /// </summary>
            /// <param name="minInclusive"></param>
            /// <param name="maxExclusive"></param>
            /// <returns></returns>
            public static int Range(int minInclusive, int maxExclusive)
            {
                lock (RNG)
                {
                    return (int)(RNG.NextDouble() * (maxExclusive - minInclusive) + minInclusive);
                }
            }
            /// <summary>
            /// Samples a random element thread-safely in the collection given the probs. If probs is null, uniform distribution of probabilities is applied. 
            /// </summary>
            public static T Sample<T>(in IEnumerable<T> collection,in IEnumerable<float> probs = null)
            {
                if(probs != null && collection.Count() != probs.Count())
                {
                    throw new ArgumentException("Collection must have the same length as probs.");
                }
                if(collection == null || collection.Count() == 0)
                {
                    throw new ArgumentException("Collection is empty.");
                }

                if(probs == null)
                {
                    float surpass = 0f;
                    float random = Random.Value;
                    foreach (var item in collection)
                    {
                        surpass += 1f / collection.Count();
                        if(surpass >= random)
                        {
                            return item;
                        }
                    }
                }
                else
                {
                   
                    float surpass = 0f;
                    float random = Random.Value;
                    for (int i = 0; i < collection.Count(); i++)
                    {
                        surpass += probs.ElementAt(i);
                        if (surpass >= random)
                            return collection.ElementAt(i);
                    }
                }

                throw new Exception($"Probs must always sum 1. (received {probs.Sum()})");
            }
            /// <summary>
            /// Samples multiple elements thread-safely in the given collection. 
            /// </summary>
            /// <typeparam name="T"></typeparam>
            /// <param name="no_samples"></param>
            /// <param name="collection"></param>
            /// <param name="replacement"></param>
            /// <returns></returns>
            /// <exception cref="ArgumentException"></exception>
            public static T[] Sample<T>(int no_samples, in IEnumerable<T> collection, bool replacement = false)
            {
                if (no_samples <= 0)
                {
                    throw new ArgumentException("Number of samples must be greater than zero.");
                }

                if (collection == null || collection.Count() == 0)
                {
                    throw new ArgumentException("Collection is empty.");
                }

                if (!replacement && no_samples > collection.Count())
                {
                    throw new ArgumentException($"Number of samples {no_samples} cannot exceed the size of the collection {collection.Count()} when replacement is not allowed.");
                }

                List<T> samples = new List<T>();
                HashSet<int> sampledIndices = new HashSet<int>();

                for (int sampleCount = 0; sampleCount < no_samples; sampleCount++)
                {
                    int index;
                    do
                    {
                        index = Random.Range(0, collection.Count());
                    } while (!replacement && sampledIndices.Contains(index));

                    sampledIndices.Add(index);
                    samples.Add(collection.ElementAt(index));
                }

                return samples.ToArray();
            }
            public static bool Bernoulli(float p = 0.5f) => Value < p;
            /// <summary>
            /// Returns a sample thread-safely from normal distribution.
            /// </summary>
            /// <param name="mean"></param>
            /// <param name="stddev"></param>
            /// <returns></returns>
            public static float Normal(float mean = 0f, float stddev = 1f, bool threadsafe = true)
            {
                if(threadsafe)
                {
                    // x1 must be > 0 to avoid log(0)
                    float x1;
                    lock (RNG)
                        x1 = (float)(1.0 - RNG.NextDouble());
                    float x2 = Value; // here is a bit strange, the algorithm doesn't says that x2 [0,1] or (0,1] or (0,1) or [0, 1)

                    var entropy = MathF.Sqrt(-2.0f * MathF.Log(x1)) * MathF.Cos(2.0f * MathF.PI * x2);
                    return entropy * stddev + mean;
                }
                else
                {
                    // x1 must be > 0 to avoid log(0) !!
                    float x1 = (float)(1.0f - RNG.NextDouble());
                    float x2 = (float)RNG.NextDouble(); // i let it in [0, 1) to be faster

                    var entropy = MathF.Sqrt(-2.0f * MathF.Log(x1)) * MathF.Cos(2.0f * MathF.PI * x2);
                    return entropy * stddev + mean;
                }
                
            }
        }
    }
}

