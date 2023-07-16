using System;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Tensor : IEquatable<Tensor>, IEquatable<TensorGPU>
    {
        [SerializeField] private float[] data;
        [SerializeField] private int[] shape;

        public int Rank
        {
            get
            {
                if (shape.Length > 1)
                    return shape.Length;
                else if (shape[0] > 1)
                    return 1;
                else
                    return 0;
            }
        }
        public int[] Shape
        {
            get => shape.ToArray();
            
        }
        private int Width
        {
            get
            {
                return shape.Last();
            }
        }
        private int Height
        {
            get
            {
                if (shape.Length < 2)
                    return 1;
                else
                    return shape[shape.Length - 2];
            }
        }
        private int Channels
        {
            get
            {
                if (shape.Length < 3)
                    return 1;
                else
                    return shape[shape.Length - 3];
            }
        }
        private int Batch
        {
            get
            {
                if (shape.Length < 4)
                    return 1;
                else
                    return shape[shape.Length - 4];
            }
        }
        public float this[int w]
        {
            get => data[w];
            private set => data[w] = value;
        }
        public float this[int h, int w]
        {
            get => data[h * Width + w];
            private set => data[h * Width + w] = value;
        }
        public float this[int c, int h, int w]
        {
            get => data[c * Height * Width + h * Width + w];
            private set => data[c * Height * Width * h * Width + w] = value;
        }
        public float this[int n, int c, int h, int w]
        {
            get => data[n * Channels * Height * Width + c * Height * Width + h * Width + w];
            private set => data[n * Channels * Height * Width + c * Height * Width + h * Width + w] = value;

        }

        #region Create
        private Tensor(params int[] shape)
        {
            if (shape == null)
                throw new ArgumentException("Tensor cannot be instantiated with null ");
            if (shape.Length == 0)
                throw new ArgumentException("Tensor cannot be instantiated with a shape of length 0");
            if (shape.Length > 4)
                throw new ArgumentException("Tensor cannot be instantiated with more than 4 dimensions.");
            if (shape.Any(x => x < 1))
                throw new ArgumentException("Tensor cannot be instantiated with a dimension < 1.");

            int size = 1;
            foreach (var item in shape)
            {
                size *= item;
            }       

            if (size > 16_777_216) // hardcoded like this because 4096x4096 max allowed matrix, on 8192 it crashes
                throw new NotSupportedException("Tensor dimensions is too large on initialization (cannot surpass 16,777,216 units).");

            this.shape = shape.ToArray();
            data = new float[size];
        }
        public static Tensor Reshape(Tensor tensor, params int[] newShape)
        {
            int count = 1;
            foreach (var item in newShape)
            {
                count *= item;
            }
            if (count != tensor.Count())
                throw new ArgumentException("The new shape must provide the same capacity of the tensor when reshaping it.");

            // if new shape is broader than the original shape
            Tensor result = new Tensor(newShape);
            Array.Copy(tensor.data, result.data, tensor.data.Length);
            return result;
        }
        public static Tensor Identity(Tensor other)
        {
            Tensor clone = new(other.shape);              
            Array.Copy(other.data, clone.data, other.data.Length);
            return clone;

        }
        public static Tensor Identity(TensorGPU other)
        {
            Tensor clone = new(other.Shape);
            Array.Copy(other.ToArray(), clone.data, other.Count());
            return clone;
        }
        public static Tensor Arange(float start, float end, float step)
        {
            if (step == 0)
                throw new ArgumentException("On Arange, step must be non-zero.");

            int count = (int)MathF.Ceiling((end - start) / step);

            Tensor result = new(count);


            float val = start;
            for (int i = 0; i < count; i++)
            {
                result.data[i] = val;

                if (start < end)
                    val += step;
                else
                    val -= step;
            }

            return result;
        }
        public static Tensor Constant(float scalar)
        {
            Tensor t = new(1);
            t.data[0] = scalar;
            return t;
        }
        public static Tensor Constant(float[] vector)
        {
            int width = vector.GetLength(0);

            Tensor t = new(width);
            for (int i = 0; i < width; i++)
            {
                t[i] = vector[i];
            }

            return t;

        }
        public static Tensor Constant(float[,] matrix)
        {
            int height = matrix.GetLength(0);
            int width = matrix.GetLength(1);

            Tensor t = new(height, width);
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    t[i, j] = matrix[i, j];
                }

            }


            return t;
        }
        public static Tensor Constant(float[,,] cube)
        {
            int width = cube.GetLength(2);
            int height = cube.GetLength(1);
            int depth = cube.GetLength(0);

            Tensor t = new(depth, height, width);
            for (int z = 0; z < depth; z++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        t[z, y, x] = cube[z, y, z];
                    }
                }
            }

            return t;
        }
        public static Tensor Constant(float[,,,] tesseract)
        {
            int width = tesseract.GetLength(3);
            int height = tesseract.GetLength(2);
            int depth = tesseract.GetLength(1);
            int time = tesseract.GetLength(0);

            Tensor t = new(time, depth, height, width);
            for (int w = 0; w < time; w++)
            {
                for (int z = 0; z < depth; z++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            t[w, z, y, x] = tesseract[w, z, y, x];
                        }
                    }
                }
            }

            return t;
        }
        public static Tensor Zeros(params int[] shape)
        {
            return new(shape);
        }
        public static Tensor Ones(params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = 1f;
            }
            return t;
        }
        public static Tensor Fill(float value, params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = value;
            }
            return t;
        }
        public static Tensor Random01(params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = Utils.Random.Value;
            }
            return t;
        }
        public static Tensor RandomNormal((float, float) mean_sd, params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = Utils.Random.Gaussian(mean_sd.Item1, mean_sd.Item2);
            }
            return t;
        }
        public static Tensor RandomRange((float, float) min_max, params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = Utils.Random.Range(min_max.Item1, min_max.Item2);
            }
            return t;
        }

        #endregion


        #region Operator overloading element-wise (+, -, *, /)
        public static Tensor operator +(Tensor tensor)
        {
            Tensor result = new(tensor.shape);
            for (int i = 0; i < tensor.data.Length; i++)
            {
                result.data[i] = tensor.data[i];
            }

            return result;
        }
        public static Tensor operator -(Tensor tensor)
        {
            Tensor result = new(tensor.shape);
            for (int i = 0; i < tensor.data.Length; i++)
            {
                result.data[i] = -tensor.data[i];
            }

            return result;
        }
        public static Tensor operator +(Tensor left, float right)
        {
            Tensor result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] += right;
            }

            return result;
        }
        public static Tensor operator -(Tensor left, float right)
        {
            Tensor result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] -= right;
            }

            return result;
        }
        public static Tensor operator *(Tensor left, float right)
        {
            Tensor result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] *= right;
            }

            return result;
        }
        public static Tensor operator /(Tensor left, float right)
        {
            Tensor result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] /= right;
            }

            return result;
        }
        public static Tensor operator +(float left, Tensor right) => right + left;
        public static Tensor operator *(float left, Tensor right) => right * left;
        public static Tensor operator +(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left[{left.shape.ToCommaSeparatedString()}] and right[{right.shape.ToCommaSeparatedString()}] tensors must have similar shape for Element-wise addition (+).");

            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] + right.data[i];
            }


            return result;
        }
        public static Tensor operator -(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left[{left.shape.ToCommaSeparatedString()}] and right[{right.shape.ToCommaSeparatedString()}] tensors must have similar shape for Element-wise subtraction (-).");

            Tensor result = new(left.shape);
            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] - right.data[i];
            }

            return result;
        }
        public static Tensor operator *(Tensor left, Tensor right)
        {
           if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left[{left.shape.ToCommaSeparatedString()}] and right[{right.shape.ToCommaSeparatedString()}] tensors must have similar shape for Element-wise multiplication (*).");

            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] * right.data[i];
            }


            return result;
        }
        public static Tensor operator /(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left[{left.shape.ToCommaSeparatedString()}] and right[{right.shape.ToCommaSeparatedString()}] tensors must have similar shape for Element-wise division (/).");

            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] / right.data[i];
            }

            return result;
        }

        #endregion


        #region Special

        /// <summary>
        /// left <b>(j, 1, n, m)</b> * right <b>(k, m, p)</b> => out <b>(j, k, n, p)</b>
        /// </summary>
        public static Tensor MatMul(Tensor left, Tensor right)
        {
            int left_rank = left.Rank;
            int right_rank = right.Rank;

            if (left_rank == 1 && right_rank == 1)
                return left * right;
                       
            if(left_rank == 1 && left.Width != right.Height)
                throw new ArgumentException($"Tensor must have compatible shapes for matrix multiplication (Left[{left.Shape.ToCommaSeparatedString()}] doesn't match Right[{right.Shape.ToCommaSeparatedString()}]).");

            if(right_rank == 1 && left.Width != right.Width)
                throw new ArgumentException($"Tensor must have compatible shapes for matrix multiplication (Left[{left.Shape.ToCommaSeparatedString()}] doesn't match Right[{right.Shape.ToCommaSeparatedString()}]).");

            if (left_rank > 1 && right_rank > 1 && left.Width != right.Height)
                throw new ArgumentException($"Tensor must have compatible shapes for matrix multiplication (Left[{left.Shape.ToCommaSeparatedString()}] doesn't match Right[{right.Shape.ToCommaSeparatedString()}]).");


            int N = left.Height;
            int M = left.Width;
            int P = right.Width;
            int K = right.Channels;
            int J = left.Batch;

            Tensor result;
            if (left_rank == 1)
                result = new(CreateShape(left.Rank, J, K, 1, P));
            else if (right_rank == 1)
                result = new(CreateShape(left.Rank, J, K, 1, N));
            else
                result = new(CreateShape(left.Rank, J, K, N, P));

 
            if(right_rank == 1)
            {
                for (int j = 0; j < J; j++)
                {
                    for (int k = 0; k < K; k++)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            float sum = 0f;
                            for (int m = 0; m < M; m++)
                            {
                                float l = left[j, 0, n, m];
                                float r = right[k, 0, m];
                                sum += l * r;
                            }
                            result[j, k, 0, n] = sum;
                        }
                    }
                }
            }
            else if(left_rank == 1)
            {
                for (int j = 0; j < J; j++)
                {
                    for (int k = 0; k < K; k++)
                    {
                        for (int p = 0; p < P; p++)
                        {
                            float sum = 0f;
                            for (int m = 0; m < M; m++)
                            {
                                float l = left[j, 0, 0, m];
                                float r = right[k, m, p];
                                sum += l * r;
                            }
                            result[j, k, 0, p] = sum;
                        }
                    }
                }
            }
            else
            {
                // starting with 64x64 matmul, GPU based multiplication becomes better
                // case non-batched and non-channeled matmul
                if (J == 1 && K == 1)
                    Parallel.For(0, N, n =>
                    {
                         for (int p = 0; p < P; p++)
                         {
                             float sum = 0f;
                             for (int m = 0; m < M; m++)
                             {
                                 sum += left[n, m] * right[m, p];
                             }
                             result[n, p] = sum;
                         }
                            
                    });
                else // base matmul operation
                    Parallel.For(0, N, n =>
                    {
                        for (int j = 0; j < J; j++)
                        {
                            for (int k = 0; k < K; k++)
                            {
                                for (int p = 0; p < P; p++)
                                {
                                    float sum = 0f;
                                    for (int m = 0; m < M; m++)
                                    {
                                        sum += left[j, 0, n, m] * right[k, m, p];
                                    }
                                    result[j, k, n, p] = sum;
                                }
                            }
                        }
                    });
            }


            // The result keeps the smallest rank
            LinkedList<int> resultShape = new LinkedList<int>();


            if (right.Rank > 1)
                resultShape.AddFirst(P);

            if (left.Rank > 1)
                resultShape.AddFirst(N);
                       
            if(J > 1)
            {
                resultShape.AddFirst(K);
                resultShape.AddFirst(J);
               
            }
            else if(K > 1)
            {
                resultShape.AddFirst(K);           
            }
            
            result.shape = resultShape.ToArray();
            return result;
        }
        /// <summary>
        /// [Deprecated] Matrix multiplication but on GPU. Efficient for matrices > 64x64 
        /// left <b>(j, 1, n, m)</b> * right <b>(k, m, p)</b> => out <b>(j, k, n, p)</b>
        /// </summary>
        public static Tensor MatMulGPU(Tensor left, Tensor right)
        {
            int left_rank = left.Rank;
            int right_rank = right.Rank;

            if (left_rank == 1 && right_rank == 1)
                return left * right;

            if (left_rank == 1 && left.Width != right.Height)
                throw new ArgumentException($"Tensor must have compatible shapes for matrix multiplication (Left[{left.Shape.ToCommaSeparatedString()}] doesn't match Right[{right.Shape.ToCommaSeparatedString()}]).");

            if (right_rank == 1 && left.Width != right.Width)
                throw new ArgumentException($"Tensor must have compatible shapes for matrix multiplication (Left[{left.Shape.ToCommaSeparatedString()}] doesn't match Right[{right.Shape.ToCommaSeparatedString()}]).");

            if (left_rank > 1 && right_rank > 1 && left.Width != right.Height)
                throw new ArgumentException($"Tensor must have compatible shapes for matrix multiplication (Left[{left.Shape.ToCommaSeparatedString()}] doesn't match Right[{right.Shape.ToCommaSeparatedString()}]).");


            int N = left.Height;
            int M = left.Width;
            int P = right.Width;
            int K = right.Channels;
            int J = left.Batch;

            Tensor result;
            if (left_rank == 1)
                result = new(CreateShape(left.Rank, J, K, 1, P));
            else if (right_rank == 1)
                result = new(CreateShape(left.Rank, J, K, 1, N));
            else
                result = new(CreateShape(left.Rank, J, K, N, P));


            ComputeShader cs = DeepUnityMeta.TensorCS;

            ComputeBuffer leftData = new(left.data.Length, 4);
            ComputeBuffer rightData = new(right.data.Length, 4);
            ComputeBuffer resultData = new(J * K * N * P, 4);

            leftData.SetData(left.data);
            rightData.SetData(right.data);

            int kernel = cs.FindKernel("MatMul");

            cs.SetBuffer(kernel, "data1", leftData);
            cs.SetBuffer(kernel, "data2", rightData);
            cs.SetBuffer(kernel, "result", resultData);

            cs.SetInt("w1", left.Width);
            cs.SetInt("h1", left.Height);
            cs.SetInt("c1", left.Channels);
            cs.SetInt("b1", left.Batch);
            cs.SetInt("r1", left.Rank);

            cs.SetInt("w2", right.Width);
            cs.SetInt("h2", right.Height);
            cs.SetInt("c2", right.Channels);
            cs.SetInt("b2", right.Batch);
            cs.SetInt("r2", right.Rank);

            if(left_rank == 1)
            {
                cs.SetInt("wr", P);
                cs.SetInt("hr", 1);
            }
            else if(right_rank == 1)
            {
                cs.SetInt("wr", N);
                cs.SetInt("hr", 1);
            }
            else
            {
                cs.SetInt("wr", P);
                cs.SetInt("hr", N);          
            }
            cs.SetInt("cr", K);
            cs.SetInt("br", J);
            cs.SetInt("rr", result.Rank);


            cs.Dispatch(kernel,
                  (P + 7) / 8,
                  (N + 7) / 8,
                  (K + 7) / 8);

            resultData.GetData(result.data);

            leftData.Release();
            rightData.Release();
            resultData.Release();


            // The result keeps the smallest rank

            LinkedList<int> resultShape = new LinkedList<int>();


            if (right.Rank > 1)
                resultShape.AddFirst(P);

            if (left.Rank > 1)
                resultShape.AddFirst(N);

            if (J > 1)
            {
                resultShape.AddFirst(K);
                resultShape.AddFirst(J);

            }
            else if (K > 1)
            {
                resultShape.AddFirst(K);
            }

            result.shape = resultShape.ToArray();
            return result;
        }
        /// <summary>
        /// Pad(tensor(b, k, n, m), padding: p) => tensor(b, k, n + p * 2, m + p * 2) 
        /// </summary>
        /// <returns></returns>
        public static Tensor MatPad(Tensor tensor, int padding, PaddingType paddingMode)
        {
            if (padding == 0)
                return tensor;

            int w = tensor.Width + 2;
            int h = tensor.Height + 2;
            int b = tensor.Channels;
            int n = tensor.Batch;
            Tensor result = new(CreateShape(tensor.Rank, n, b, h, w));

            for (int l = 0; l < tensor.Batch; l++)
            {
                for (int k = 0; k < tensor.Channels; k++)
                {
                    for (int j = 0; j < tensor.Height; j++)
                    {
                        for (int i = 0; i < tensor.Width; i++)
                        {
                            result[l, k, j + 1, i + 1] = tensor[k, j, i];
                        }
                    }
                }
            }


            if (paddingMode == PaddingType.Mirror)
            {

                for (int l = 0; l < tensor.Batch; l++)
                {
                    for (int k = 0; k < tensor.Channels; k++)
                    {
                        result[l, k, 0, 0] = result[l, k, 1, 1];
                        result[l, k, h - 1, 0] = result[l, k, h - 2, 1];
                        result[l, k, 0, w - 1] = result[l, k, 1, w - 2];
                        result[l, k, h - 1, w - 1] = result[l, k, h - 2, w - 2];

                        for (int i = 0; i < w - 1; i++)
                        {
                            result[l, k, 0, i] = result[l, k, 1, i];
                            result[l, k, h - 1, i] = result[l, k, h - 2, i];
                        }

                        for (int j = 0; j < h - 1; j++)
                        {
                            result[l, k, j, 0] = result[l, k, j, 1];
                            result[l, k, j, w - 1] = result[l, k, j, w - 2];
                        }
                    }
                }

            }


            return MatPad(result, padding - 1, paddingMode);
        }
        /// <summary>
        /// Perform cross-correlation. <br />
        /// The output has the same number of channels as the kernels.
        /// </summary>
        /// <param name="inputs"> (batch, input_channels, height, width]</param>
        /// <param name="kernels">(output_channels, input_channels, height, width]</param>
        public static Tensor Correlate2D(Tensor inputs, Tensor kernels, CorrelationMode correlationType)
        {
            Tensor output = null;

            // Output shape : [batch, kern.batch, *W, *H] 

            int outputChannels = kernels.Batch;
            int inputChannels = kernels.Channels;

            int batchSize = inputs.Batch;
            int inputHeight = inputs.Height;
            int inputWidth = inputs.Width;
            int kernelHeight = kernels.Height;
            int kernelWidth = kernels.Width;

            if (correlationType == CorrelationMode.Valid)
            {


                int outputHeight = inputHeight - kernelHeight + 1;
                int outputWidth = inputWidth - kernelWidth + 1;


                output = Zeros(batchSize, outputChannels, outputHeight, outputWidth);

                Parallel.For(0, batchSize, b =>
                {
                    for (int oc = 0; oc < outputChannels; oc++)
                    {
                        for (int ic = 0; ic < inputChannels; ic++)
                        {
                            for (int h = 0; h < outputHeight; h++)
                            {
                                for (int w = 0; w < outputWidth; w++)
                                {
                                    float sum = 0f;

                                    for (int j = 0; j < kernelHeight; j++)
                                    {
                                        for (int i = 0; i < kernelWidth; i++)
                                        {
                                            sum += inputs[b, ic, h + j, w + i] * kernels[oc, ic, j, i];
                                        }
                                    }

                                    output[b, oc, h, w] = sum;
                                }
                            }
                        }
                    }
                });
            }
            else if (correlationType == CorrelationMode.Full)
            {
                int outputHeight = inputHeight + kernelHeight - 1;
                int outputWidth = inputWidth + kernelWidth - 1;

                output = Zeros(batchSize, outputChannels, outputHeight, outputWidth);

                Parallel.For(0, batchSize, b =>
                {
                    for (int oc = 0; oc < outputChannels; oc++)
                    {
                        for (int ic = 0; ic < inputChannels; ic++)
                        {
                            for (int h = 0; h < outputHeight; h++)
                            {
                                for (int w = 0; w < outputWidth; w++)
                                {
                                    float sum = 0f;

                                    for (int j = 0; j < kernelHeight; j++)
                                    {
                                        for (int i = 0; i < kernelWidth; i++)
                                        {
                                            int inputRow = h - j;
                                            int inputCol = w - i;

                                            if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth)
                                            {
                                                sum += inputs[b, ic, inputRow, inputCol] * kernels[oc, ic, j, i];
                                            }
                                        }
                                    }

                                    output[b, oc, h, w] = sum;
                                }
                            }
                        }
                    }
                });

            }
            else if (correlationType == CorrelationMode.Same)
            {
                int outputHeight = inputHeight;
                int outputWidth = inputWidth;

                int paddingHeight = (kernelHeight - 1) / 2;
                int paddingWidth = (kernelWidth - 1) / 2;

                output = Zeros(batchSize, outputChannels, outputHeight, outputWidth);

                Parallel.For(0, batchSize, b =>
                {
                    for (int oc = 0; oc < outputChannels; oc++)
                    {
                        for (int ic = 0; ic < inputChannels; ic++)
                        {
                            for (int h = 0; h < outputHeight; h++)
                            {
                                for (int w = 0; w < outputWidth; w++)
                                {
                                    float sum = 0f;

                                    for (int j = 0; j < kernelHeight; j++)
                                    {
                                        for (int i = 0; i < kernelWidth; i++)
                                        {
                                            int inputRow = h + j - paddingHeight;
                                            int inputCol = w + i - paddingWidth;

                                            if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth)
                                            {
                                                sum += inputs[b, ic, inputRow, inputCol] * kernels[oc, ic, j, i];
                                            }
                                        }
                                    }

                                    output[b, oc, h, w] = sum;
                                }
                            }
                        }
                    }
                });
            }

            return output;
        }
        /// <summary>
        /// Perform convolution between a batch of channeled input images and a channeled kernel. <br />
        /// Tensor channels and kernel channels MUST match. Kernel is rotated by 180d, then the method returns corr2D between tensor input and rotated kernel. <br />
        /// </summary>
        /// <param name="inputs"> (batch, channels, height, width]</param>
        /// <param name="kernels">(channels, height, width]</param>
        /// <returns></returns>
        public static Tensor Convolve2D(Tensor inputs, Tensor kernels, CorrelationMode correlationType)
        {
            // Rotate kernel by 180d
            int height = kernels.Height;
            int width = kernels.Width;
            Tensor rot180Kernel = new(kernels.shape);

            for (int c = 0; c < kernels.Channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        rot180Kernel[c, height - h - 1, width - w - 1] = kernels[c, h, w];
                    }
                }
            }

            return Correlate2D(inputs, rot180Kernel, correlationType);
        }

        #endregion Special


        #region Static operations
        public static int Size(Tensor tensor, int axis)
        {
            HandleAxis(tensor, ref axis);
            return tensor.shape[axis];
        }
        /// <summary>
        /// If axis is null, removes all dimensions equal to 1.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static Tensor Squeeze(Tensor tensor, int? axis = null)
        {
            if (axis == null)
            {
                // Removes all axis with value 1
                LinkedList<int> squeezedShape = new LinkedList<int>();

                if (tensor.Width > 1)
                    squeezedShape.AddFirst(tensor.Width);

                if (tensor.Height > 1)
                    squeezedShape.AddFirst(tensor.Height);

                if (tensor.Channels > 1)
                    squeezedShape.AddFirst(tensor.Channels);

                if (tensor.Batch > 1)
                    squeezedShape.AddFirst(tensor.Batch);

                if (squeezedShape.Count == 0)
                    squeezedShape.AddFirst(tensor.Width);

                Tensor result = new(squeezedShape.ToArray());
                Array.Copy(tensor.data, result.data, tensor.data.Length);
                return result;
            }
            else
            {
                int ax = axis.Value;
                HandleAxis(tensor, ref ax);

                // if axis is not 1, tensor remains unchanged
                if (tensor.shape[ax] != 1)
                    return Identity(tensor);

                // Esle remove that axis
                List<int> newShape = tensor.shape.ToList();
                newShape.RemoveAt(ax);

                Tensor result = new(newShape.ToArray());
                Array.Copy(tensor.data, result.data, tensor.data.Length);
                return result;
            }

        }
        public static Tensor Unsqueeze(Tensor tensor, int axis)
        {
            HandleAxis(tensor, ref axis);


            List<int> unsqueezedShape = tensor.shape.ToList();
            unsqueezedShape.Insert(axis, 1);
            Tensor result = new(unsqueezedShape.ToArray());
            Array.Copy(tensor.data, result.data, tensor.data.Length);
            return result;
        }     
        public static Tensor Flatten(Tensor tensor, int startAxis = 0, int endAxis = -1)
        {
            if (startAxis > endAxis)
                throw new Exception($"Start axis ({startAxis}) must be greater or equal to the end axis ({endAxis}) when flattening.");

            HandleAxis(tensor, ref startAxis);
            HandleAxis(tensor, ref endAxis);

            List<int> newShape = new();
            
            for (int i = 0; i < startAxis; i++)
            {
                newShape.Add(tensor.shape[i]);
            }
            int mergedDim = 1;
            for (int i = startAxis; i <= endAxis; i++)
            {
                mergedDim *= tensor.shape[i];
            }
            newShape.Add(mergedDim);
            for (int i = endAxis + 1; i < tensor.shape.Length; i++)
            {
                newShape.Add(tensor.shape[i]);
            }

            Tensor result = new Tensor(newShape.ToArray());
            Array.Copy(tensor.data, result.data, tensor.data.Length);
            return result;
        }
        /// <summary>
        /// If axis is null, the tensors are joined along the next dimension. <br></br>
        /// Example: <br></br>
        /// Join(0, {(2,3),(2,3),(2,3),(2,3)}) => (8,3) <br></br>
        /// Join(1, {(2,3),(2,3),(2,3),(2,3)}) => (2,12) <br></br>
        /// Join(null, {(2,3),(2,3),(2,3),(2,3)}) => (4,2,3) <br></br>
        /// </summary>
        /// <param name="axis"></param>
        /// <param name="tensors"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor Join(int? axis, params Tensor[] tensors)
        {            

            if (tensors.Length == 1)
                return Identity(tensors[0]);

            for (int i = 1; i < tensors.Length; i++)
            {
                if (!tensors[i - 1].shape.SequenceEqual(tensors[i].shape))
                {
                    throw new ArgumentException($"Tensors must have the same shape in order to be joined ([{tensors[i - 1].shape.ToCommaSeparatedString()}] != [{tensors[i].shape.ToCommaSeparatedString()}])");
                }
            }

            int no_slices = tensors.Length;
            Tensor slice = tensors[0];
            int[] shapex = null;
            Dim dim;

            int ax = axis.Value;
            HandleAxis(tensors[0], ref ax);

            if (axis == null)
            {
                int axisIndex = (int)AxisToDim(tensors[0], 0) - 1;

                if (axisIndex < 0)
                    throw new ArgumentException("Cannot join tensors along the fifth dimension because the limit of a tensor shape is 4.");
                dim = (Dim) axisIndex;// if axis is null, we join the tensors on the extra dimension
                
            }
            else
                dim = AxisToDim(tensors[0], ax);
            switch (dim)
            {
                case Dim.width:
                    shapex = CreateShape(slice.Rank, slice.Batch, slice.Channels, slice.Height, slice.Width * no_slices);
                    break;
                case Dim.height:
                    shapex = CreateShape(slice.Rank, slice.Batch, slice.Channels, slice.Height * no_slices, slice.Width);
                    break;
                case Dim.channel:
                    shapex = CreateShape(slice.Rank, slice.Batch, slice.Channels * no_slices, slice.Height, slice.Width);
                    break;
                case Dim.batch:
                    shapex = CreateShape(slice.Rank, slice.Batch * no_slices, slice.Channels, slice.Height, slice.Width);
                    break;
            }
            Tensor result = new(shapex);

            for (int s = 0; s < no_slices; s++)
            {
                for (int l = 0; l < slice.Batch; l++)
                {
                    for (int k = 0; k < slice.Channels; k++)
                    {
                        for (int j = 0; j < slice.Height; j++)
                        {
                            for (int i = 0; i < slice.Width; i++)
                            {
                                switch (dim)
                                {
                                    case Dim.width:
                                        result[l, k, j, s * slice.Width + i] = tensors[s][l, k, j, i];
                                        break;
                                    case Dim.height:
                                        result[l, k, s * slice.Height + j, i] = tensors[s][l, k, j, i];
                                        break;
                                    case Dim.channel:
                                        result[l, s * slice.Channels + k, j, i] = tensors[s][l, k, j, i];
                                        break;
                                    case Dim.batch:
                                        result[s * slice.Batch + l, k, j, i] = tensors[s][l, k, j, i];
                                        break;
                                }

                            }
                        }
                    }
                }
            }


            return result;
        }
        public static Tensor Expand(Tensor tensor, int axis, int times)
        {
            if (times < 1)
                throw new ArgumentException("When expanding a tensor, times cannot be < 1");

            if (times == 1)
                return Identity(tensor);

            HandleAxis(tensor, ref axis);

            Dim dim = AxisToDim(tensor, axis);

            int[] shapex = null;
            switch (dim)
            {
                case Dim.width:
                    shapex = CreateShape(tensor.Rank, tensor.Batch, tensor.Channels, tensor.Height, tensor.Width * times);
                    break;
                case Dim.height:
                    shapex = CreateShape(tensor.Rank, tensor.Batch, tensor.Channels, tensor.Height * times, tensor.Width);
                    break;
                case Dim.channel:
                    shapex = CreateShape(tensor.Rank, tensor.Batch, tensor.Channels * times, tensor.Height, tensor.Width);
                    break;
                case Dim.batch:
                    shapex = CreateShape(tensor.Rank, tensor.Batch * times, tensor.Channels, tensor.Height, tensor.Width);
                    break;

            }
            Tensor result = new(shapex);

            for (int t = 0; t < times; t++)
            {
                for (int l = 0; l < tensor.Batch; l++)
                {
                    for (int k = 0; k < tensor.Channels; k++)
                    {
                        for (int j = 0; j < tensor.Height; j++)
                        {
                            for (int i = 0; i < tensor.Width; i++)
                            {
                                switch (dim)
                                {
                                    case Dim.width:
                                        result[l, k, j, t * tensor.Width + i] = tensor[l, k, j, i];
                                        break;
                                    case Dim.height:
                                        result[l, k, t * tensor.Height + j, i] = tensor[l, k, j, i];
                                        break;
                                    case Dim.channel:
                                        result[l, t * tensor.Channels + k, j, i] = tensor[l, k, j, i];
                                        break;
                                    case Dim.batch:
                                        result[t * tensor.Batch + l, k, j, i] = tensor[l, k, j, i];
                                        break;
                                }

                            }
                        }
                    }
                }

            }

            return result;
        }
        public static Tensor Transpose(Tensor tensor, int axis0, int axis1)
        {
            HandleAxis(tensor, ref axis0);
            HandleAxis(tensor, ref axis1);

            if (axis0 == axis1)
                return Identity(tensor);


            axis0 = (int)AxisToDim(tensor, axis0);
            axis1 = (int)AxisToDim(tensor, axis1);
            int[] permutation = new int[] { tensor.Batch, tensor.Channels, tensor.Height, tensor.Width };


            var temp = permutation[axis0];
            permutation[axis0] = permutation[axis1];
            permutation[axis1] = temp;

            Tensor result = new(CreateShape(tensor.Rank, permutation[0], permutation[1], permutation[2], permutation[3]));


            for (int l = 0; l < tensor.Batch; l++)
            {
                for (int k = 0; k < tensor.Channels; k++)
                {
                    for (int j = 0; j < tensor.Height; j++)
                    {
                        for (int i = 0; i < tensor.Width; i++)
                        {
                            int[] transposedIndices = new int[] { l, k, j, i };

                            temp = transposedIndices[axis0];
                            transposedIndices[axis0] = transposedIndices[axis1];
                            transposedIndices[axis1] = temp;

                            result[transposedIndices[0], transposedIndices[1], transposedIndices[2], transposedIndices[3]] = tensor[l, k, j, i];
                        }
                    }
                }
            }
            return result;
        }
        public static Tensor[] Split(Tensor tensor, int axis, int split_size)
        {
            HandleAxis(tensor, ref axis);

            Dim dim = AxisToDim(tensor, axis);
            int[] stackShape = new int[] { tensor.Batch, tensor.Channels, tensor.Height, tensor.Width };
            List<Tensor> slices = new();

            int dimLength = stackShape[(int)dim];
            int dimPos = 0;
            while (dimPos < dimLength)
            {
                int dimCopySize = Math.Min(split_size, dimLength - dimPos);
                int[] sliceShape = stackShape.ToArray();
                sliceShape[(int)dim] = dimCopySize;
                Tensor slice = new(CreateShape(tensor.Rank, sliceShape[0], sliceShape[1], sliceShape[2], sliceShape[3]));

                for (int l = 0; l < slice.Batch; l++)
                {
                    for (int k = 0; k < slice.Channels; k++)
                    {
                        for (int j = 0; j < slice.Height; j++)
                        {
                            for (int i = 0; i < slice.Width; i++)
                            {
                                switch (dim)
                                {
                                    case Dim.width:
                                        slice[l, k, j, i] = tensor[l, k, j, dimPos + i];
                                        break;
                                    case Dim.height:
                                        slice[l, k, j, i] = tensor[l, k, j + dimPos, i];
                                        break;
                                    case Dim.channel:
                                        slice[l, k, j, i] = tensor[l, k + dimPos, j, i];
                                        break;
                                    case Dim.batch:
                                        slice[l, k, j, i] = tensor[l + dimPos, k, j, i];
                                        break;
                                }

                            }
                        }
                    }
                }

                slices.Add(slice);
                dimPos += split_size;
            }

            return slices.ToArray();
        }
        public static Tensor Shuffle(Tensor tensor, int axis)
        {
            HandleAxis(tensor, ref axis);
            Tensor[] slices = Split(tensor, axis, 1);
            slices = Utils.Shuffle(slices).ToArray();
            return Join(axis, slices);
        }
        public static Tensor Sum(Tensor tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dimIndex = AxisToDim(tensor, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = keepDim == true ? newShape[axis] : 1;
            Tensor result = new(newShape);

            if (dimIndex == Dim.width)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int j = 0; j < height; j++)
                        {
                            float sum = 0f;
                            for (int i = 0; i < width; i++)
                            {
                                sum += tensor[l, k, j, i];
                            }

                            for (int i = 0; i < newShape[axis]; i++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == Dim.height)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float sum = 0f;
                            for (int j = 0; j < height; j++)
                            {
                                sum += tensor[l, k, j, i];
                            }

                            for (int j = 0; j < newShape[axis]; j++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == Dim.channel)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float sum = 0f;

                            for (int k = 0; k < channels; k++)
                            {
                                sum += tensor[l, k, j, i];
                            }

                            for (int k = 0; k < newShape[axis]; k++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == Dim.batch)
            {
                for (int k = 0; k < channels; k++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float sum = 0f;
                            for (int l = 0; l < batch; l++)
                            {
                                sum += tensor[l, k, j, i];
                            }
                            for (int l = 0; l < newShape[axis]; l++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }

            result.Squeeze();
            return result;
        }
        public static Tensor Mean(Tensor tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dim = AxisToDim(tensor, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = keepDim == true ? newShape[axis] : 1;
            Tensor result = new(newShape);


            if (dim == Dim.width)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int j = 0; j < height; j++)
                        {
                            float sum = 0f;
                            for (int i = 0; i < width; i++)
                            {
                                sum += tensor[l, k, j, i];
                            }
                            sum /= width;
                            for (int i = 0; i < newShape[axis]; i++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.height)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float sum = 0f;
                            for (int j = 0; j < height; j++)
                            {
                                sum += tensor[l, k, j, i];
                            }
                            sum /= height;

                            for (int j = 0; j < newShape[axis]; j++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.channel)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float sum = 0f;

                            for (int k = 0; k < channels; k++)
                            {
                                sum += tensor[l, k, j, i];
                            }
                            sum /= channels;
                            for (int k = 0; k < newShape[axis]; k++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.batch)
            {
                for (int k = 0; k < channels; k++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float sum = 0f;
                            for (int l = 0; l < batch; l++)
                            {
                                sum += tensor[l, k, j, i];
                            }
                            sum /= batch;
                            for (int l = 0; l < newShape[axis]; l++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }

            result.Squeeze();
            return result;
        }
        public static Tensor Var(Tensor tensor, int axis, int correction = 1, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dimIndex = AxisToDim(tensor, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = keepDim == true ? newShape[axis] : 1;
            Tensor result = new(newShape);

            if (dimIndex == Dim.width)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int j = 0; j < height; j++)
                        {
                            float sum = 0f;
                            float sumSqr = 0f;
                            for (int i = 0; i < width; i++)
                            {
                                float value = tensor[l, k, j, i];
                                sum += value;
                                sumSqr += value * value;
                            }
                            float vr = (sumSqr - (sum * sum) / width) / (width - correction);

                            for (int i = 0; i < newShape[axis]; i++)
                            {
                                result[l, k, j, i] = vr;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == Dim.height)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float sum = 0f;
                            float sumSqr = 0f;
                            for (int j = 0; j < height; j++)
                            {
                                float value = tensor[l, k, j, i];
                                sum += value;
                                sumSqr += value * value;
                            }
                            float vr = (sumSqr - (sum * sum) / height) / (height - correction);

                            for (int j = 0; j < newShape[axis]; j++)
                            {
                                result[l, k, j, i] = vr;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == Dim.channel)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float sum = 0f;
                            float sumSqr = 0f;
                            for (int k = 0; k < channels; k++)
                            {
                                float value = tensor[l, k, j, i];
                                sum += value;
                                sumSqr += value * value;
                            }

                            float vr = (sumSqr - (sum * sum) / channels) / (channels - correction);

                            for (int k = 0; k < newShape[axis]; k++)
                            {
                                result[l, k, j, i] = vr;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == Dim.batch)
            {
                for (int k = 0; k < channels; k++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float sum = 0f;
                            float sumSqr = 0f;
                            for (int l = 0; l < batch; l++)
                            {
                                float value = tensor[l, k, j, i];
                                sum += value;
                                sumSqr += value * value;
                            }
                            float vr = (sumSqr - (sum * sum) / batch) / (batch - correction);

                            for (int l = 0; l < newShape[axis]; l++)
                            {
                                result[l, k, j, i] = vr;
                            }
                        }
                    }
                }
            }

            result.Squeeze();
            return result;
        }
        public static Tensor Std(Tensor tensor, int axis, int correction = 1, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);
            return Sqrt(Var(tensor, axis, correction, keepDim));
        }
        public static Tensor Min(Tensor tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dim = AxisToDim(tensor, axis);


            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = keepDim == true ? newShape[axis] : 1;
            Tensor result = new(newShape);

            if (dim == Dim.width)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int j = 0; j < height; j++)
                        {
                            float min = float.MaxValue;
                            for (int i = 0; i < width; i++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }

                            for (int i = 0; i < newShape[axis]; i++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.height)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float min = float.MaxValue;
                            for (int j = 0; j < height; j++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }

                            for (int j = 0; j < newShape[axis]; j++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.channel)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float min = float.MaxValue;

                            for (int k = 0; k < channels; k++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }

                            for (int k = 0; k < newShape[axis]; k++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.batch)
            {
                for (int k = 0; k < channels; k++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float min = float.MaxValue;
                            for (int l = 0; l < batch; l++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }
                            for (int l = 0; l < newShape[axis]; l++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }

            return result.Squeeze();
        }
        public static Tensor Max(Tensor tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dim = AxisToDim(tensor, axis);


            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = keepDim == true ? newShape[axis] : 1;
            Tensor result = new(newShape);

            if (dim == Dim.width)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int j = 0; j < height; j++)
                        {
                            float m = float.MinValue;
                            for (int i = 0; i < width; i++)
                            {
                                m = MathF.Max(m, tensor[l, k, j, i]);
                            }

                            for (int i = 0; i < newShape[axis]; i++)
                            {
                                result[l, k, j, i] = m;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.height)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float m = float.MinValue;
                            for (int j = 0; j < height; j++)
                            {
                                m = MathF.Max(m, tensor[l, k, j, i]);
                            }

                            for (int j = 0; j < newShape[axis]; j++)
                            {
                                result[l, k, j, i] = m;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.channel)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float m = float.MinValue;

                            for (int k = 0; k < channels; k++)
                            {
                                m = MathF.Max(m, tensor[l, k, j, i]);
                            }

                            for (int k = 0; k < newShape[axis]; k++)
                            {
                                result[l, k, j, i] = m;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.batch)
            {
                for (int k = 0; k < channels; k++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            float m = float.MinValue;
                            for (int l = 0; l < batch; l++)
                            {
                                m = MathF.Max(m, tensor[l, k, j, i]);
                            }
                            for (int l = 0; l < newShape[axis]; l++)
                            {
                                result[l, k, j, i] = m;
                            }
                        }
                    }
                }
            }
            return result.Squeeze();

        }
        public static Tensor Pow(Tensor tensor, float power)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Pow(tensor.data[i], power);
            }

            return result;
        }
        public static Tensor Sqrt(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Sqrt(tensor.data[i]);
            }

            return result;
        }
        public static Tensor Exp(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Exp(tensor.data[i]);
            }

            return result;
        }
        public static Tensor Log(Tensor tensor, float @base = MathF.E)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Log(tensor.data[i], @base);
            }

            return result;
        }
        public static Tensor Abs(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Abs(tensor.data[i]);
            }

            return result;
        }
        public static Tensor Sin(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Sin(tensor.data[i]);
            }

            return result;
        }
        public static Tensor Cos(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Cos(tensor.data[i]);
            }

            return result;
        }
        /// <summary>
        /// Clips each element in the tensor in [min,max] range.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static Tensor Clip(Tensor tensor, float min, float max)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = Math.Clamp(tensor.data[i], min, max);
            }

            return result;
        }
        /// <summary>
        /// Computed the norm of the tensor.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="normType"></param>
        /// <returns><see cref="Tensor"/> (1)</returns>
        /// <exception cref="Exception"></exception>
        public static Tensor Norm(Tensor tensor, NormType normType = NormType.EuclideanL2)
        {
            switch (normType)
            {
                case NormType.NonZeroL0:
                    int nonzeros = tensor.Count(x => x != 0);
                    return Constant(nonzeros);
                case NormType.ManhattanL1:
                    float absSum = tensor.data.Sum(x => MathF.Abs(x));
                    return Constant(absSum);
                case NormType.EuclideanL2:
                    float sqrSum = tensor.data.Sum(x => x * x);
                    return Constant(MathF.Sqrt(sqrSum));
                case NormType.MaxLInf:
                    float maxAbs = tensor.data.Max(x => Math.Abs(x));
                    return Constant(maxAbs);
                default:
                    throw new Exception("Unhandled norm type.");
            }
        }



        /// <summary>
        /// Computes the Element-Wise minimum.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor Minimum(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left[{left.shape.ToCommaSeparatedString()}] and right[{right.shape.ToCommaSeparatedString()}] tensors must have different shape for Min operation.");


            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Min(left.data[i], right.data[i]);
            }

            return result;
        }
        /// <summary>
        /// Computes the Element-Wise maximum.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor Maximum(Tensor left, Tensor right)
        {
           if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left[{left.shape.ToCommaSeparatedString()}] and right[{right.shape.ToCommaSeparatedString()}] tensors must have different shape for Max operation.");



            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Max(left.data[i], right.data[i]);
            }

            return result;
        }
          
        /// <summary>
        /// Computes the element-wise log density,
        /// </summary>
        /// <param name="x"></param>
        /// <param name="mu"></param>
        /// <param name="sigma"></param>
        /// <returns></returns>
        public static Tensor LogDensity(Tensor x, Tensor mu, Tensor sigma)
        {
            var frac = (x - mu) / sigma;
            var elem1 = Log(sigma);
            var elem2 = 0.5f * MathF.Log(2.0f * MathF.PI);
            var elem3 = 0.5f * frac * frac;
            return -elem1 - elem2 - elem3;
        }
        /// <summary>
        /// Computes the element-wise density,
        /// </summary>
        /// <param name="x"></param>
        /// <param name="mu"></param>
        /// <param name="sigma"></param>
        /// <returns></returns>
        public static Tensor Density(Tensor x, Tensor mu, Tensor sigma)
        {
            Tensor p1 = (sigma * MathF.Sqrt(2f * MathF.PI)) * 0.5f;
            Tensor std = (x - mu) / sigma;
            Tensor p2 = -0.5f * std * std;
            return p1 * Tensor.Exp(p2);
        }
        /// <summary>
        /// Computes the element-wise Kullback-Leibler divergence. Measures the distance between two data distributions showing how different the two distributions are from each other.
        /// </summary>
        /// <param name="mu1"></param>
        /// <param name="sig1"></param>
        /// <param name="mu2"></param>
        /// <param name="sig2"></param>
        /// <returns></returns>
        public static Tensor KLDivergence(Tensor mu1, Tensor sig1, Tensor mu2, Tensor sig2)
        {
            var var1 = sig1 * sig1;
            var var2 = sig2 * sig2;

            return Tensor.Log((sig2 / (sig1 + Utils.EPSILON)) + Utils.EPSILON) +
                (var1 + (mu1 - mu2) * (mu1 - mu2)) / (2f * var2) - 0.5f;
        }

        #endregion Statics


        #region Instance opeations
        public int Size(int axis)
        {
            HandleAxis(this, ref axis);
            return shape[axis];
        }
        public Tensor Reshape(params int[] newShape)
        {
            int count = 1;
            foreach (var item in newShape)
            {
                count *= item;
            }

            if (count != Count())
                throw new ArgumentException("The new shape must provide the same capacity of the tensor when reshaping it.");

            this.shape = newShape;
            return this;
        }
        /// <summary>
        /// If axis is null, removes all dimensions equal to 1.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public Tensor Squeeze(int? axis = null)
        {
            if (axis == null)
            {
                // Removes all axis with value 1
                LinkedList<int> squeezedShape = new LinkedList<int>();

                if (Width > 1)
                    squeezedShape.AddFirst(Width);

                if (Height > 1)
                    squeezedShape.AddFirst(Height);

                if (Channels > 1)
                    squeezedShape.AddFirst(Channels);

                if (Batch > 1)
                    squeezedShape.AddFirst(Batch);

                if (squeezedShape.Count == 0)
                    squeezedShape.AddFirst(Width);

                this.shape = squeezedShape.ToArray();
            }
            else
            {
                int ax = axis.Value;
                HandleAxis(this, ref ax);
                // if axis is not 1, tensor remains unchanged
                if (shape[ax] != 1)
                    return this;

                // Esle remove that axis
                List<int> newShape = shape.ToList();
                newShape.RemoveAt(ax);

                this.shape = newShape.ToArray();
            }

            return this;
        }
        public Tensor Unsqueeze(int axis)
        {
            HandleAxis(this, ref axis);


            List<int> unsqueezedShape = shape.ToList();
            unsqueezedShape.Insert(axis, 1);
            this.shape = unsqueezedShape.ToArray();

            return this;
        }
        public Tensor Flatten(int startAxis = 0, int endAxis = -1)
        {
            if (startAxis > endAxis)
                throw new Exception($"Start axis ({startAxis}) must be greater or equal to the end axis ({endAxis}) when flattening.");

            HandleAxis(this, ref startAxis);
            HandleAxis(this, ref endAxis);

           

            List<int> newShape = new();

            for (int i = 0; i < startAxis; i++)
            {
                newShape.Add(shape[i]);
            }
            int mergedDim = 1;
            for (int i = startAxis; i <= endAxis; i++)
            {
                mergedDim *= shape[i];
            }
            newShape.Add(mergedDim);
            for (int i = endAxis + 1; i <shape.Length; i++)
            {
                newShape.Add(shape[i]);
            }

            this.shape = newShape.ToArray();
            return this;
        }
        public Tensor Expand(int axis, int times)
        {
            if (times < 1)
                throw new ArgumentException("When expanding a tensor, times cannot be < 1");

            if (times == 1)
                return this;

            HandleAxis(this, ref axis);

            Dim dim = AxisToDim(this, axis);

            int[] shapex = null;
            switch (dim)
            {
                case Dim.width:
                    shapex = CreateShape(Rank, Batch, Channels, Height, Width * times);
                    break;
                case Dim.height:
                    shapex = CreateShape(Rank, Batch, Channels, Height * times, Width);
                    break;
                case Dim.channel:
                    shapex = CreateShape(Rank, Batch, Channels * times, Height, Width);
                    break;
                case Dim.batch:
                    shapex = CreateShape(Rank, Batch * times, Channels, Height, Width);
                    break;
            }

            Tensor result = new Tensor(shapex);

            for (int t = 0; t < times; t++)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int k = 0; k < Channels; k++)
                    {
                        for (int j = 0; j < Height; j++)
                        {
                            for (int i = 0; i < Width; i++)
                            {
                                switch (dim)
                                {
                                    case Dim.width:
                                        result[l, k, j, t * Width + i] = this[l, k, j, i];
                                        break;
                                    case Dim.height:
                                        result[l, k, t * Height + j, i] = this[l, k, j, i];
                                        break;
                                    case Dim.channel:
                                        result[l, t * Channels + k, j, i] = this[l, k, j, i];
                                        break;
                                    case Dim.batch:
                                        result[t * Batch + l, k, j, i] = this[l, k, j, i];
                                        break;
                                }
                            }
                        }
                    }
                }
            }

            return result;
        }
        public Tensor Transpose(int axis0, int axis1)
        {
            HandleAxis(this, ref axis0);
            HandleAxis(this, ref axis1);
            if (axis0 == axis1)
                return this;

            int axis0Dim = (int)AxisToDim(this, axis0);
            int axis1Dim = (int)AxisToDim(this, axis1);

            int[] permutation = new int[] { Batch, Channels, Height, Width };
            var temp = permutation[axis0Dim];
            permutation[axis0Dim] = permutation[axis1Dim];
            permutation[axis1Dim] = temp;

            Tensor result = new Tensor(CreateShape(Rank, permutation[0], permutation[1], permutation[2], permutation[3]));

            for (int l = 0; l < Batch; l++)
            {
                for (int k = 0; k < Channels; k++)
                {
                    for (int j = 0; j < Height; j++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            int[] transposedIndices = new int[] { l, k, j, i };
                            temp = transposedIndices[axis0Dim];
                            transposedIndices[axis0Dim] = transposedIndices[axis1Dim];
                            transposedIndices[axis1Dim] = temp;

                            result[transposedIndices[0], transposedIndices[1], transposedIndices[2], transposedIndices[3]] = this[l, k, j, i];
                        }
                    }
                }
            }

            return result;
        }
        public Tensor[] Split(int axis, int split_size)
        {
            HandleAxis(this, ref axis);
            Dim dim = AxisToDim(this, axis);
            int[] stackShape = new int[] { Batch, Channels, Height, Width };
            List<Tensor> slices = new List<Tensor>();

            int dimLength = stackShape[(int)dim];
            int dimPos = 0;
            while (dimPos < dimLength)
            {
                int dimCopySize = Math.Min(split_size, dimLength - dimPos);
                int[] sliceShape = stackShape.ToArray();
                sliceShape[(int)dim] = dimCopySize;
                Tensor slice = new Tensor(CreateShape(Rank, sliceShape[0], sliceShape[1], sliceShape[2], sliceShape[3]));

                for (int l = 0; l < slice.Batch; l++)
                {
                    for (int k = 0; k < slice.Channels; k++)
                    {
                        for (int j = 0; j < slice.Height; j++)
                        {
                            for (int i = 0; i < slice.Width; i++)
                            {
                                switch (dim)
                                {
                                    case Dim.width:
                                        slice[l, k, j, i] = this[l, k, j, dimPos + i];
                                        break;
                                    case Dim.height:
                                        slice[l, k, j, i] = this[l, k, j + dimPos, i];
                                        break;
                                    case Dim.channel:
                                        slice[l, k, j, i] = this[l, k + dimPos, j, i];
                                        break;
                                    case Dim.batch:
                                        slice[l, k, j, i] = this[l + dimPos, k, j, i];
                                        break;
                                }
                            }
                        }
                    }
                }

                slices.Add(slice);
                dimPos += split_size;
            }

            return slices.ToArray();
        }
        public Tensor Shuffle(int axis)
        {
            HandleAxis(this, ref axis);
            Tensor[] slices = Split(axis, 1);
            slices = Utils.Shuffle(slices).ToArray();
            return Join(axis, slices);
        }
        public Tensor Sum(int axis, bool keepDim = false)
        {
            HandleAxis(this, ref axis);
            Dim dim = AxisToDim(this, axis);

            int[] newShape = shape.ToArray();
            newShape[axis] = keepDim ? newShape[axis] : 1;
            Tensor result = new Tensor(newShape);

            if (dim == Dim.width)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int k = 0; k < Channels; k++)
                    {
                        for (int j = 0; j < Height; j++)
                        {
                            float sum = 0f;
                            for (int i = 0; i < Width; i++)
                            {
                                sum += this[l, k, j, i];
                            }

                            for (int i = 0; i < newShape[axis]; i++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.height)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int k = 0; k < Channels; k++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float sum = 0f;
                            for (int j = 0; j < Height; j++)
                            {
                                sum += this[l, k, j, i];
                            }

                            for (int j = 0; j < newShape[axis]; j++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.channel)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int j = 0; j < Height; j++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float sum = 0f;

                            for (int k = 0; k < Channels; k++)
                            {
                                sum += this[l, k, j, i];
                            }

                            for (int k = 0; k < newShape[axis]; k++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.batch)
            {
                for (int k = 0; k < Channels; k++)
                {
                    for (int j = 0; j < Height; j++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float sum = 0f;
                            for (int l = 0; l < Batch; l++)
                            {
                                sum += this[l, k, j, i];
                            }
                            for (int l = 0; l < newShape[axis]; l++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }

            result.Squeeze();
            return result;
        }
        public Tensor Mean(int axis, bool keepDim = false)
        {
            HandleAxis(this, ref axis);
            Dim dim = AxisToDim(this, axis);

            int[] newShape = shape.ToArray();
            newShape[axis] = keepDim ? newShape[axis] : 1;
            Tensor result = new Tensor(newShape);

            if (dim == Dim.width)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int k = 0; k < Channels; k++)
                    {
                        for (int j = 0; j < Height; j++)
                        {
                            float sum = 0f;
                            for (int i = 0; i < Width; i++)
                            {
                                sum += this[l, k, j, i];
                            }
                            sum /= Width;
                            for (int i = 0; i < newShape[axis]; i++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.height)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int k = 0; k < Channels; k++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float sum = 0f;
                            for (int j = 0; j < Height; j++)
                            {
                                sum += this[l, k, j, i];
                            }
                            sum /= Height;

                            for (int j = 0; j < newShape[axis]; j++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.channel)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int j = 0; j < Height; j++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float sum = 0f;

                            for (int k = 0; k < Channels; k++)
                            {
                                sum += this[l, k, j, i];
                            }
                            sum /= Channels;
                            for (int k = 0; k < newShape[axis]; k++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.batch)
            {
                for (int k = 0; k < Channels; k++)
                {
                    for (int j = 0; j < Height; j++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float sum = 0f;
                            for (int l = 0; l < Batch; l++)
                            {
                                sum += this[l, k, j, i];
                            }
                            sum /= Batch;
                            for (int l = 0; l < newShape[axis]; l++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }

            result.Squeeze();
            return result;
        }
        public Tensor Var(int axis, int correction = 1, bool keepDim = false)
        {
            HandleAxis(this, ref axis);
            Dim dim = AxisToDim(this, axis);

            int[] newShape = shape.ToArray();
            newShape[axis] = keepDim ? newShape[axis] : 1;
            Tensor result = new Tensor(newShape);

            if (dim == Dim.width)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int k = 0; k < Channels; k++)
                    {
                        for (int j = 0; j < Height; j++)
                        {
                            float sum = 0f;
                            float sumSqr = 0f;
                            for (int i = 0; i < Width; i++)
                            {
                                float value = this[l, k, j, i];
                                sum += value;
                                sumSqr += value * value;
                            }
                            float vr = (sumSqr - (sum * sum) / Width) / (Width - correction);

                            for (int i = 0; i < newShape[axis]; i++)
                            {
                                result[l, k, j, i] = vr;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.height)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int k = 0; k < Channels; k++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float sum = 0f;
                            float sumSqr = 0f;
                            for (int j = 0; j < Height; j++)
                            {
                                float value = this[l, k, j, i];
                                sum += value;
                                sumSqr += value * value;
                            }
                            float vr = (sumSqr - (sum * sum) / Height) / (Height - correction);

                            for (int j = 0; j < newShape[axis]; j++)
                            {
                                result[l, k, j, i] = vr;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.channel)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int j = 0; j < Height; j++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float sum = 0f;
                            float sumSqr = 0f;
                            for (int k = 0; k < Channels; k++)
                            {
                                float value = this[l, k, j, i];
                                sum += value;
                                sumSqr += value * value;
                            }

                            float vr = (sumSqr - (sum * sum) / Channels) / (Channels - correction);

                            for (int k = 0; k < newShape[axis]; k++)
                            {
                                result[l, k, j, i] = vr;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.batch)
            {
                for (int k = 0; k < Channels; k++)
                {
                    for (int j = 0; j < Height; j++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float sum = 0f;
                            float sumSqr = 0f;
                            for (int l = 0; l < Batch; l++)
                            {
                                float value = this[l, k, j, i];
                                sum += value;
                                sumSqr += value * value;
                            }
                            float vr = (sumSqr - (sum * sum) / Batch) / (Batch - correction);

                            for (int l = 0; l < newShape[axis]; l++)
                            {
                                result[l, k, j, i] = vr;
                            }
                        }
                    }
                }
            }

            result.Squeeze();
            return result;
        }
        public Tensor Std(int axis, int correction = 1, bool keepDim = false)
        {
            HandleAxis(this, ref axis);
            return Sqrt(Var(axis, correction, keepDim));
        }
        public Tensor Min(int axis, bool keepDim = false)
        {
            HandleAxis(this, ref axis);

            Dim dim = AxisToDim(this, axis);


            int[] newShape = shape.ToArray();
            newShape[axis] = keepDim ? newShape[axis] : 1;
            Tensor result = new Tensor(newShape);

            if (dim == Dim.width)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int k = 0; k < Channels; k++)
                    {
                        for (int j = 0; j < Height; j++)
                        {
                            float min = float.MaxValue;
                            for (int i = 0; i < Width; i++)
                            {
                                min = MathF.Min(min, this[l, k, j, i]);
                            }

                            for (int i = 0; i < result.Width; i++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.height)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int k = 0; k < Channels; k++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float min = float.MaxValue;
                            for (int j = 0; j < Height; j++)
                            {
                                min = MathF.Min(min, this[l, k, j, i]);
                            }

                            for (int j = 0; j < result.Height; j++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.channel)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int j = 0; j < Height; j++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float min = float.MaxValue;
                            for (int k = 0; k < Channels; k++)
                            {
                                min = MathF.Min(min, this[l, k, j, i]);
                            }

                            for (int k = 0; k < result.Channels; k++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.batch)
            {
                for (int k = 0; k < Channels; k++)
                {
                    for (int j = 0; j < Height; j++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float min = float.MaxValue;
                            for (int l = 0; l < Batch; l++)
                            {
                                min = MathF.Min(min, this[l, k, j, i]);
                            }

                            for (int l = 0; l < result.Batch; l++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }

            result.Squeeze();
            return result;
        }
        public Tensor Max(int axis, bool keepDim = false)
        {
            HandleAxis(this, ref axis);

            Dim dim = AxisToDim(this, axis);

            int[] newShape = shape.ToArray();
            newShape[axis] = keepDim ? newShape[axis] : 1;
            Tensor result = new Tensor(newShape);

            if (dim == Dim.width)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int k = 0; k < Channels; k++)
                    {
                        for (int j = 0; j < Height; j++)
                        {
                            float max = float.MinValue;
                            for (int i = 0; i < Width; i++)
                            {
                                max = MathF.Max(max, this[l, k, j, i]);
                            }

                            for (int i = 0; i < result.Width; i++)
                            {
                                result[l, k, j, i] = max;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.height)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int k = 0; k < Channels; k++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float max = float.MinValue;
                            for (int j = 0; j < Height; j++)
                            {
                                max = MathF.Max(max, this[l, k, j, i]);
                            }

                            for (int j = 0; j < result.Height; j++)
                            {
                                result[l, k, j, i] = max;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.channel)
            {
                for (int l = 0; l < Batch; l++)
                {
                    for (int j = 0; j < Height; j++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float max = float.MinValue;
                            for (int k = 0; k < Channels; k++)
                            {
                                max = MathF.Max(max, this[l, k, j, i]);
                            }

                            for (int k = 0; k < result.Channels; k++)
                            {
                                result[l, k, j, i] = max;
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.batch)
            {
                for (int k = 0; k < Channels; k++)
                {
                    for (int j = 0; j < Height; j++)
                    {
                        for (int i = 0; i < Width; i++)
                        {
                            float max = float.MinValue;
                            for (int l = 0; l < Batch; l++)
                            {
                                max = MathF.Max(max, this[l, k, j, i]);
                            }

                            for (int l = 0; l < result.Batch; l++)
                            {
                                result[l, k, j, i] = max;
                            }
                        }
                    }
                }
            }

            result.Squeeze();
            return result;
        }
        public Tensor Pow(float power)
        {

            for (int i = 0; i < data.Length; i++)
            {
                data[i] = MathF.Pow(data[i], power);
            }

            return this;
        }
        public Tensor Sqrt()
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = MathF.Sqrt(data[i]);
            }

            return this;
        }
        public Tensor Exp()
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = MathF.Exp(data[i]);
            }

            return this;
        }
        public Tensor Log(float @base = MathF.E)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = MathF.Log(data[i], @base);
            }

            return this;
        }
        public Tensor Abs()
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = MathF.Abs(data[i]);
            }

            return this;
        }
        public Tensor Sin()
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = MathF.Sin(data[i]);
            }

            return this;
        }
        public Tensor Cos()
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = MathF.Cos(data[i]);
            }

            return this;
        }
        public Tensor Clip(float min, float max)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = Math.Clamp(data[i], min, max);
            }

            return this;
        }
        public Tensor Norm(NormType normType = NormType.EuclideanL2)
        {
            switch (normType)
            {
                case NormType.NonZeroL0:
                    int nonzeros = data.Count(x => x != 0);
                    return Constant(nonzeros);
                case NormType.ManhattanL1:
                    float absSum = data.Sum(x => MathF.Abs(x));
                    return Constant(absSum);
                case NormType.EuclideanL2:
                    float sqrSum = data.Sum(x => x * x);
                    return Constant(MathF.Sqrt(sqrSum));
                case NormType.MaxLInf:
                    float maxAbs = data.Max(x => MathF.Abs(x));
                    return Constant(maxAbs);
                default:
                    throw new Exception("Unhandled norm type.");
            }
        }

        #endregion Instance


        public void ForEach(Action<float> action)
        {
            for (int i = 0; i < data.Length; i++)
            {
                action(data[i]);
            }
        }
        public Tensor Select(Func<float, float> selector)
        {
            Tensor result = new(shape);

            for (int i = 0; i < data.Length; i++)
            {
                result.data[i] = selector(data[i]);
            }

            return result;
        }
        public Tensor Zip(Tensor second, Func<float, float, float> resultSelector)
        {       
            Tensor result = new(shape);

            for (int i = 0; i < data.Length; i++)
            {
                result.data[i] = resultSelector(data[i], second.data[i]);
            }

            return result;
        }
        public int Count(Func<float, bool> predicate = null)
        {
            if (predicate == null)
                return data.Count();
            else
                return data.Count(predicate);
        }
        public float Min(Func<float, float> selector = null)
        {
            if (selector == null)
                return data.Min();
            else
                return data.Min(selector);
        }
        public float Max(Func<float, float> selector = null)
        {
            if (selector == null)
                return data.Max();
            else
                return data.Max(selector);
        }  
        public float[] ToArray()
        {
            return data.ToArray();
        }
        public bool Equals(Tensor other)
        {
            if (!shape.SequenceEqual(other.shape))
                return false;

            if (!data.SequenceEqual(other.data))
                return false;
            
            return true;
        }
        public bool Equals(TensorGPU other)
        {
            if(!shape.SequenceEqual(other.Shape))
                return false;

            if(!data.SequenceEqual(other.ToArray()))
                return false;

            return true;
        }
        public override string ToString()
        {
            int rank = Rank;

            StringBuilder sb = new();

            sb.Append($"Tensor[{Shape.ToCommaSeparatedString()}]");

            sb.Append("\n[");
            for (int l = 0; l < Batch; l++)
            {
                if (l > 0)
                {
                    sb.Append("\n\n\n");
                    for (int indent = 0; indent < rank - 3; indent++)
                    {
                        sb.Append(" ");
                    }
                }
                if (rank > 3)
                    sb.Append("[");

                for (int k = 0; k < Channels; k++)
                {
                    if (k > 0)
                    {
                        sb.Append("\n\n");
                        for (int indent = 0; indent < rank - 2; indent++)
                        {
                            sb.Append(" ");
                        }
                    }
                    if (rank > 2)
                        sb.Append("[");

                    for (int j = 0; j < Height; j++)
                    {
                        if (j > 0 && rank > 1)
                        {
                            sb.Append("\n");
                            for (int indent = 0; indent < rank - 1; indent++)
                            {
                                sb.Append(" ");
                            }
                        }
                        if (rank > 1)
                            sb.Append("[");

                        for (int i = 0; i < Width; i++)
                        {
                            if (i > 0)
                                sb.Append(", ");

                            sb.Append(this[l, k, j, i].ToString("0.00000"));
                        }

                        if (rank > 1)
                            sb.Append("]");
                    }

                    if (rank > 2)
                        sb.Append("]");
                }

                if (rank > 3)
                    sb.Append("]");
            }

            sb.Append("]");

            return sb.ToString();
        }
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }


        // inside use
        /// <summary>
        /// Checks if the axis is out of range or not. Also transforms a negative axis to a positive one.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        private static void HandleAxis(Tensor tensor, ref int axis)
        {
            int rank = tensor.Rank;


            if(rank == 0)
            {
                if (axis != 0 && axis != -1)
                    throw new ArgumentOutOfRangeException($"Invalid axis value ({axis}) for a tensor with rank ({tensor.Rank})");

                axis = 0;
            }
            else
            {
                if (axis >= rank)
                    throw new ArgumentOutOfRangeException($"Invalid axis value ({axis}) for a tensor with rank ({tensor.Rank})");

                if (axis < 0)
                   axis = rank + axis;
            }     
        }
        private static Dim AxisToDim(Tensor t, int axis)
        {
            // Returns the index in the full shape array of the axis. ([0,1,2,3])
            // Used only for methods along the axis, that uses full shape call.
            int rank = t.Rank;

            if (axis > rank)
                throw new ArgumentException($"Cannot use axis {axis} for a tensor of rank {rank}.");

            // check for rank 0
            if (rank == 0 && (axis == 0 || axis == -1))
                return (Dim) 3;

            // check for negative axis as well
            if (axis >= 0)
                return (Dim) 4 - rank + axis;
            else
                return (Dim) 4 + axis;

        }
        private static int[] CreateShape(int rank, int b, int c, int h, int w)
        {
            /// Auto corrects the rank.
            int correctedRank;
            if (b > 1)
                correctedRank = 4;
            else if (c > 1)
                correctedRank = 3;
            else if(h > 1)
                correctedRank = 2;
            else if(w > 1)
                correctedRank = 1;
            else 
                correctedRank = 0;

            rank = Math.Max(rank, correctedRank);

            if (rank < 2)
                return new int[] { w };
            else if (rank < 3)
                return new int[] { h, w };
            else if (rank < 4)
                return new int[] { c, h, w };
            else
                return new int[] { b, c, h, w };
        }

    }
}