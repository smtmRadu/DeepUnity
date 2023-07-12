using System;
using System.Collections;
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
    public sealed partial class Tensor : IEquatable<Tensor>, IEquatable<TensorGPU>
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
        public int Width
        {
            get
            {
                return shape.Last();
            }
        }
        public int Height
        {
            get
            {
                if (shape.Length < 2)
                    return 1;
                else
                    return shape[shape.Length - 2];
            }
        }
        public int Channels
        {
            get
            {
                if (shape.Length < 3)
                    return 1;
                else
                    return shape[shape.Length - 3];
            }
        }
        public int Batch
        {
            get
            {
                if (shape.Length < 4)
                    return 1;
                else
                    return shape[shape.Length - 4];
            }
        }
        public int[] Shape
        {
            get => shape.ToArray();
            
        }
        public int Size(int axis)
        {
            if (axis >= 0)
                return shape[axis];
            else
                return shape[shape.Length + axis];
        }

        public float this[int w]
        {
            get => data[w];
            private set => data[w] = value;
        }
        public float this[int h, int w]
        {
            get => data[w * Height + h];
            private set => data[w * Height + h] = value;
        }
        public float this[int c, int h, int w]
        {
            get => data[c * Height * Width + w * Height + h];
            private set => data[c * Height * Width + w * Height + h] = value;
        }
        public float this[int n, int c, int h, int w]
        {
            get => data[n * Channels * Height * Width + c * Height * Width + w * Height + h];
            private set => data[n * Channels * Height * Width + c * Height * Width + w * Height + h] = value;

        }
  

        #region Create Tensor
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

            Tensor result = new Tensor(newShape);

            int batch = result.Batch;
            int channels = result.Channels;
            int height = result.Height;
            int width = result.Width;


            // Reshape the data inside
            int index = 0;
            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            result[b, c, h, w] = tensor.data[index++];
                        }
                    }
                }
            }

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


        #region Operator overloading (+, -, *, /)
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


        #region Custom Operations

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
        /// Matrix multiplication but on GPU. 
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

        #endregion


        #region On Dimension Operations
        public static Tensor Transpose(Tensor tensor, Dim dim0, Dim dim1)
        {
            if (dim0 == dim1)
                return tensor;

            int[] swappedShape = null;

            if (dim0 == Dim.width)
            {
                if (dim1 == Dim.height)
                    swappedShape = CreateShape(tensor.Rank, tensor.Batch, tensor.Channels, tensor.Width, tensor.Height);
                else if (dim1 == Dim.channel)
                    swappedShape = CreateShape(tensor.Rank, tensor.Batch, tensor.Width, tensor.Height, tensor.Channels);
                else if (dim1 == Dim.batch)
                    swappedShape = CreateShape(tensor.Rank, tensor.Width, tensor.Channels, tensor.Height, tensor.Batch);
            }
            else if (dim0 == Dim.height)
            {
                if (dim1 == Dim.width)
                    swappedShape = CreateShape(tensor.Rank, tensor.Batch, tensor.Channels, tensor.Height, tensor.Width);
                else if (dim1 == Dim.channel)
                    swappedShape = CreateShape(tensor.Rank, tensor.Batch, tensor.Height, tensor.Width, tensor.Channels);
                else if (dim1 == Dim.batch)
                    swappedShape = CreateShape(tensor.Rank, tensor.Height, tensor.Batch, tensor.Width, tensor.Channels);
            }
            else if (dim0 == Dim.channel)
            {
                if (dim1 == Dim.width)
                    swappedShape = CreateShape(tensor.Rank, tensor.Batch, tensor.Height, tensor.Channels, tensor.Width);
                else if (dim1 == Dim.height)
                    swappedShape = CreateShape(tensor.Rank, tensor.Batch, tensor.Channels, tensor.Height, tensor.Width);
                else if (dim1 == Dim.batch)
                    swappedShape = CreateShape(tensor.Rank, tensor.Channels, tensor.Batch, tensor.Height, tensor.Width);
            }
            else if (dim0 == Dim.batch)
            {
                if (dim1 == Dim.width)
                    swappedShape = CreateShape(tensor.Rank, tensor.Channels, tensor.Batch, tensor.Height, tensor.Width);
                else if (dim1 == Dim.height)
                    swappedShape = CreateShape(tensor.Rank, tensor.Height, tensor.Batch, tensor.Channels, tensor.Width);
                else if (dim1 == Dim.channel)
                    swappedShape = CreateShape(tensor.Rank, tensor.Batch, tensor.Height, tensor.Channels, tensor.Width);
            }

            Tensor result = new(swappedShape);

            for (int l = 0; l < result.Batch; l++)
            {
                for (int k = 0; k < result.Channels; k++)
                {
                    for (int j = 0; j < result.Height; j++)
                    {
                        for (int i = 0; i < result.Width; i++)
                        {
                            if (dim0 == Dim.width && dim1 == Dim.height)
                                result[l, k, j, i] = tensor[l, k, i, j];
                            else if (dim0 == Dim.width && dim1 == Dim.channel)
                                result[l, j, i, k] = tensor[l, k, i, j];
                            else if (dim0 == Dim.width && dim1 == Dim.batch)
                                result[i, k, j, l] = tensor[l, k, i, j];

                            else if (dim0 == Dim.height && dim1 == Dim.width)
                                result[l, k, i, j] = tensor[l, k, j, i];
                            else if (dim0 == Dim.height && dim1 == Dim.channel)
                                result[l, j, k, i] = tensor[l, k, i, j];
                            else if (dim0 == Dim.height && dim1 == Dim.batch)
                                result[i, k, j, l] = tensor[l, k, i, j];

                            else if (dim0 == Dim.channel && dim1 == Dim.width)
                                result[l, k, i, j] = tensor[l, k, j, i];
                            else if (dim0 == Dim.channel && dim1 == Dim.height)
                                result[l, j, k, i] = tensor[l, k, i, j];
                            else if (dim0 == Dim.channel && dim1 == Dim.batch)
                                result[i, j, k, l] = tensor[l, k, i, j];

                            else if (dim0 == Dim.batch && dim1 == Dim.width)
                                result[i, k, j, l] = tensor[l, k, j, i];
                            else if (dim0 == Dim.batch && dim1 == Dim.height)
                                result[j, k, i, l] = tensor[l, k, i, j];
                            else if (dim0 == Dim.batch && dim1 == Dim.channel)
                                result[i, j, k, l] = tensor[l, k, i, j];
                            else
                                throw new ArgumentException("Something went wrong bro.");

                        }
                    }
                }
            }

            return result;
        }
        public static Tensor Var(Tensor tensor, Dim dim, int correction = 1, bool keepDim = false)
        {
            Tensor result = null;
            int[] fullshape = tensor.FullShape();

            if (!keepDim)
            {
                if (dim == Dim.width)
                {
                    result = new(CreateShape(tensor.Rank, fullshape[0], fullshape[1], fullshape[2], 1));
                    for (int l = 0; l < fullshape[0]; l++)
                    {
                        for (int k = 0; k < fullshape[1]; k++)
                        {
                            for (int j = 0; j < fullshape[2]; j++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int i = 0; i < fullshape[3]; i++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }
                                result[l, k, j, 0] = (sumSqr - (sum * sum) / fullshape[3]) / (fullshape[3] - correction);
                            }
                        }
                    }
                }
                else if (dim == Dim.height)
                {
                    result = new(CreateShape(tensor.Rank, fullshape[0], fullshape[1], 1, fullshape[3]));
                    for (int l = 0; l < fullshape[0]; l++)
                    {
                        for (int k = 0; k < fullshape[1]; k++)
                        {
                            for (int i = 0; i < fullshape[3]; i++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int j = 0; j < fullshape[2]; j++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }
                                result[l, k, 0, i] = (sumSqr - (sum * sum) / fullshape[2]) / (fullshape[2] - correction);
                            }
                        }
                    }
                }
                else if (dim == Dim.channel)
                {
                    result = new(CreateShape(tensor.Rank, fullshape[0], 1, fullshape[2], fullshape[3]));
                    for (int l = 0; l < fullshape[0]; l++)
                    {
                        for (int j = 0; j < fullshape[2]; j++)
                        {
                            for (int i = 0; i < fullshape[3]; i++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int k = 0; k < fullshape[1]; k++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }

                                result[l, 0, j, i] = (sumSqr - (sum * sum) / fullshape[1]) / (fullshape[1] - correction);
                            }
                        }
                    }
                }
                else if (dim == Dim.batch)
                {
                    result = new(CreateShape(tensor.Rank, 1, fullshape[1], fullshape[2], fullshape[3]));
                    for (int k = 0; k < fullshape[1]; k++)
                    {
                        for (int j = 0; j < fullshape[2]; j++)
                        {
                            for (int i = 0; i < fullshape[3]; i++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int l = 0; l < fullshape[0]; l++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }
                                result[0, k, j, i] = (sumSqr - (sum * sum) / fullshape[0]) / (fullshape[0] - correction);
                            }
                        }
                    }
                }

            }
            else
            {
                result = new(tensor.shape);
                if (dim == Dim.width)
                {
                    for (int l = 0; l < fullshape[0]; l++)
                    {
                        for (int k = 0; k < fullshape[1]; k++)
                        {
                            for (int j = 0; j < fullshape[2]; j++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int i = 0; i < fullshape[3]; i++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }

                                float res = (sumSqr - (sum * sum) / fullshape[3]) / (fullshape[3] - correction);
                                for (int i = 0; i < fullshape[3]; i++)
                                {
                                    result[l, k, j, i] = res;
                                }

                            }
                        }
                    }
                }
                else if (dim == Dim.height)
                {
                    for (int l = 0; l < fullshape[0]; l++)
                    {
                        for (int k = 0; k < fullshape[1]; k++)
                        {
                            for (int i = 0; i < fullshape[3]; i++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int j = 0; j < fullshape[2]; j++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }
                                float res = (sumSqr - (sum * sum) / fullshape[2]) / (fullshape[2] - correction);
                                for (int j = 0; j < fullshape[2]; j++)
                                {
                                    result[l, k, j, i] = res;
                                }
                            }
                        }
                    }
                }
                else if (dim == Dim.channel)
                {
                    for (int l = 0; l < fullshape[0]; l++)
                    {
                        for (int j = 0; j < fullshape[2]; j++)
                        {
                            for (int i = 0; i < fullshape[3]; i++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int k = 0; k < fullshape[1]; k++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }

                                float res = (sumSqr - (sum * sum) / fullshape[1]) / (fullshape[1] - correction);
                                for (int k = 0; k < fullshape[1]; k++)
                                {
                                    result[l, k, j, i] = res;
                                }
                            }
                        }
                    }
                }
                else if (dim == Dim.batch)
                {
                    for (int k = 0; k < fullshape[1]; k++)
                    {
                        for (int j = 0; j < fullshape[2]; j++)
                        {
                            for (int i = 0; i < fullshape[3]; i++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int l = 0; l < fullshape[0]; l++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }
                                float res = (sumSqr - (sum * sum) / fullshape[0]) / (fullshape[0] - correction);
                                for (int l = 0; l < fullshape[0]; l++)
                                {
                                    result[l, k, j, i] = res;
                                }
                            }
                        }
                    }
                }
            }
            return result;
        }
        public static Tensor Std(Tensor tensor, Dim dim, int correction = 1, bool keepDim = false)
        {
            return Sqrt(Var(tensor, dim, correction, keepDim));
        }
        public static Tensor Mean(Tensor tensor, Dim dim, bool keepDim = false)
        {
            Tensor result = null;
            int[] shape = tensor.FullShape();

            if (!keepDim)
            {
                if (dim == Dim.width)
                {
                    result = new(CreateShape(tensor.Rank, shape[0], shape[1], shape[2], 1));
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int k = 0; k < shape[1]; k++)
                        {
                            for (int j = 0; j < shape[2]; j++)
                            {
                                float sum = 0f;
                                for (int i = 0; i < shape[3]; i++)
                                {
                                    sum += tensor[l, k, j, i];
                                }
                                result[l, k, j, 0] = sum / shape[3];
                            }
                        }
                    }
                }
                else if (dim == Dim.height)
                {
                    result = new(CreateShape(tensor.Rank, shape[0], shape[1], 1, shape[3]));
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int k = 0; k < shape[1]; k++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                for (int j = 0; j < shape[2]; j++)
                                {
                                    sum += tensor[l, k, j, i];
                                }
                                result[l, k, 0, i] = sum / shape[2];
                            }
                        }
                    }
                }
                else if (dim == Dim.channel)
                {
                    result = new(CreateShape(tensor.Rank, shape[0], 1, shape[2], shape[3]));
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;

                                for (int k = 0; k < shape[1]; k++)
                                {
                                    sum += tensor[l, k, j, i];
                                }

                                result[l, 0, j, i] = sum / shape[1];
                            }
                        }
                    }
                }
                else if (dim == Dim.batch)
                {
                    result = new(CreateShape(tensor.Rank, 1, shape[1], shape[2], shape[3]));
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                for (int l = 0; l < shape[0]; l++)
                                {
                                    sum += tensor[l, k, j, i];
                                }
                                result[0, k, j, i] = sum / shape[0];
                            }
                        }
                    }
                }
            }
            else
            {
                result = new(tensor.shape);
                if (dim == Dim.width)
                {

                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int k = 0; k < shape[1]; k++)
                        {
                            for (int j = 0; j < shape[2]; j++)
                            {
                                float sum = 0f;
                                for (int i = 0; i < shape[3]; i++)
                                {
                                    sum += tensor[l, k, j, i];
                                }
                                sum /= shape[3];

                                for (int i = 0; i < shape[3]; i++)
                                {
                                    result[l, k, j, i] = sum;
                                }
                            }
                        }
                    }
                }
                else if (dim == Dim.height)
                {
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int k = 0; k < shape[1]; k++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                for (int j = 0; j < shape[2]; j++)
                                {
                                    sum += tensor[l, k, j, i];
                                }
                                sum /= shape[2];

                                for (int j = 0; j < shape[2]; j++)
                                {
                                    result[l, k, j, i] = sum;
                                }
                            }
                        }
                    }
                }
                else if (dim == Dim.channel)
                {
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;

                                for (int k = 0; k < shape[1]; k++)
                                {
                                    sum += tensor[l, k, j, i];
                                }

                                sum /= shape[1];

                                for (int k = 0; k < shape[1]; k++)
                                {
                                    result[l, k, j, i] = sum;
                                }
                            }
                        }
                    }
                }
                else if (dim == Dim.batch)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                for (int l = 0; l < shape[0]; l++)
                                {
                                    sum += tensor[l, k, j, i];
                                }
                                sum /= shape[0];

                                for (int l = 0; l < shape[0]; l++)
                                {
                                    result[l, k, j, i] = sum;
                                }
                            }
                        }
                    }
                }
            }

            return result;
        }
        public static Tensor Sum(Tensor tensor, Dim dim, bool keepDim = false)
        {
            Tensor result = null;
            int[] shape = tensor.FullShape();

            if (!keepDim)
            {
                if (dim == Dim.width)
                {
                    result = new(CreateShape(tensor.Rank, shape[0], shape[1], shape[2], 1));
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int k = 0; k < shape[1]; k++)
                        {
                            for (int j = 0; j < shape[2]; j++)
                            {
                                float sum = 0f;
                                for (int i = 0; i < shape[3]; i++)
                                {
                                    sum += tensor[l, k, j, i];
                                }
                                result[l, k, j, 0] = sum;
                            }
                        }
                    }
                }
                else if (dim == Dim.height)
                {
                    result = new(CreateShape(tensor.Rank, shape[0], shape[1], 1, shape[3]));
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int k = 0; k < shape[1]; k++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                for (int j = 0; j < shape[2]; j++)
                                {
                                    sum += tensor[l, k, j, i];
                                }
                                result[l, k, 0, i] = sum;
                            }
                        }
                    }
                }
                else if (dim == Dim.channel)
                {
                    result = new(CreateShape(tensor.Rank, shape[0], 1, shape[2], shape[3]));
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;

                                for (int k = 0; k < shape[1]; k++)
                                {
                                    sum += tensor[l, k, j, i];
                                }

                                result[l, 0, j, i] = sum;
                            }
                        }
                    }
                }
                else if (dim == Dim.batch)
                {
                    result = new(CreateShape(tensor.Rank, 1, shape[1], shape[2], shape[3]));
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                for (int l = 0; l < shape[0]; l++)
                                {
                                    sum += tensor[l, k, j, i];
                                }
                                result[0, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else
            {
                result = new(tensor.shape);
                if (dim == Dim.width)
                {

                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int k = 0; k < shape[1]; k++)
                        {
                            for (int j = 0; j < shape[2]; j++)
                            {
                                float sum = 0f;
                                for (int i = 0; i < shape[3]; i++)
                                {
                                    sum += tensor[l, k, j, i];
                                }

                                for (int i = 0; i < shape[3]; i++)
                                {
                                    result[l, k, j, i] = sum;
                                }

                            }
                        }
                    }
                }
                else if (dim == Dim.height)
                {

                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int k = 0; k < shape[1]; k++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                for (int j = 0; j < shape[2]; j++)
                                {
                                    sum += tensor[l, k, j, i];
                                }

                                for (int j = 0; j < shape[2]; j++)
                                {
                                    result[l, k, j, i] = sum;
                                }
                            }
                        }
                    }
                }
                else if (dim == Dim.channel)
                {

                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;

                                for (int k = 0; k < shape[1]; k++)
                                {
                                    sum += tensor[l, k, j, i];
                                }

                                for (int k = 0; i < shape[1]; k++)
                                {
                                    result[l, k, j, i] = sum;
                                }
                            }
                        }
                    }
                }
                else if (dim == Dim.batch)
                {

                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                for (int l = 0; l < shape[0]; l++)
                                {
                                    sum += tensor[l, k, j, i];
                                }

                                for (int l = 0; l < shape[0]; l++)
                                {
                                    result[l, k, j, i] = sum;
                                }
                            }
                        }
                    }
                }
            }


            return result;
        }
        public static Tensor Shuffle(Tensor tensor, Dim dim)
        {
            Tensor[] slices = Split(tensor, dim, 1);
            slices = Utils.Shuffle(slices).ToArray();
            return Join(dim, slices);
        }
        /// <summary>
        /// Expands the tensor along the specified dimension. If times = 1, no changes are applied to the tensor.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="dim"></param>
        /// <param name="times"></param>
        /// <returns></returns>
        public static Tensor Expand(Tensor tensor, Dim dim, int times)
        {
            if (times < 1)
                throw new ArgumentException("When expanding a tensor, times cannot be < 1");

            if (times == 1)
                return Identity(tensor);


            int[] shapex = null;
            switch (dim)
            {
                case Dim.width:
                    shapex = CreateShape(tensor.Rank, tensor.Batch, tensor.Channels, tensor.Height, times);
                    break;
                case Dim.height:
                    shapex = CreateShape(tensor.Rank, tensor.Batch, tensor.Channels, times, tensor.Width);
                    break;
                case Dim.channel:
                    shapex = CreateShape(tensor.Rank, tensor.Batch, times, tensor.Height, tensor.Width);
                    break;
                case Dim.batch:
                    shapex = CreateShape(tensor.Rank, times, tensor.Channels, tensor.Height, tensor.Width);
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
        public static Tensor Join(Dim dim, params Tensor[] tensors)
        {
            if (tensors == null || tensors.Length == 0)
                throw new ArgumentException("Tensor used for joining are not defined.");

            if (tensors.Length == 1)
                return Identity(tensors[0]);

            int no_slices = tensors.Length;
            Tensor slice = tensors[0];
            int[] shapex = null;
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
        /// <summary>
        /// The selected dimension is divided into smaller chunks. <br />
        /// If the dimension is not a multiple of split_size, the last batch will remain incompletely.<br />
        /// Example: Split(tensor(m, n, p), dim: height, split_size: 1) => tensor(m, 1, p) x n tensors. 
        /// </summary>
        public static Tensor[] Split(Tensor tensor, Dim dim, int split_size)
        {
            List<Tensor> slices = new();

            int dimLength = -1;
            if (dim == Dim.width)
                dimLength = tensor.Width;
            else if (dim == Dim.height)
                dimLength = tensor.Height;
            else if (dim == Dim.channel)
                dimLength = tensor.Channels;
            else if (dim == Dim.batch)
                dimLength = tensor.Batch;


            int dimPos = 0;
            while (dimPos < dimLength)
            {
                int dimCopySize = Math.Min(split_size, dimLength - dimPos);

                int[] shapex = null;
                switch (dim)
                {
                    case Dim.width:
                        shapex = CreateShape(tensor.Rank, tensor.Batch, tensor.Channels, tensor.Height, dimCopySize);
                        break;
                    case Dim.height:
                        shapex = CreateShape(tensor.Rank, tensor.Batch, tensor.Channels, dimCopySize, tensor.Width);
                        break;
                    case Dim.channel:
                        shapex = CreateShape(tensor.Rank, tensor.Batch, dimCopySize, tensor.Height, tensor.Width);
                        break;
                    case Dim.batch:
                        shapex = CreateShape(tensor.Rank, dimCopySize, tensor.Channels, tensor.Height, tensor.Width);
                        break;

                }
                Tensor slice = new(shapex);

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
        public static Tensor Min(Tensor tensor, Dim dim, bool keepDim = false)
        {
            Tensor result = null;
            int[] fullshape = tensor.FullShape();

            if (keepDim)
                result = new(tensor.shape);
            else
            {
                if (dim == Dim.width) result = new(CreateShape(tensor.Rank, fullshape[0], fullshape[1], fullshape[2], 1));
                else if (dim == Dim.height) result = new(CreateShape(tensor.Rank, fullshape[0], fullshape[1], 1, fullshape[3]));
                else if (dim == Dim.channel) result = new(CreateShape(tensor.Rank, fullshape[0], 1, fullshape[2], fullshape[3]));
                else if (dim == Dim.batch) result = new(CreateShape(tensor.Rank, 1, fullshape[1], fullshape[2], fullshape[3]));
            }

            if (dim == Dim.width)
            {

                for (int l = 0; l < fullshape[0]; l++)
                {
                    for (int k = 0; k < fullshape[1]; k++)
                    {
                        for (int j = 0; j < fullshape[2]; j++)
                        {
                            float min = float.MaxValue;
                            for (int i = 0; i < fullshape[3]; i++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
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

                for (int l = 0; l < fullshape[0]; l++)
                {
                    for (int k = 0; k < fullshape[1]; k++)
                    {
                        for (int i = 0; i < fullshape[3]; i++)
                        {
                            float min = float.MaxValue;
                            for (int j = 0; j < fullshape[2]; j++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
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

                for (int l = 0; l < fullshape[0]; l++)
                {
                    for (int j = 0; j < fullshape[2]; j++)
                    {
                        for (int i = 0; i < fullshape[3]; i++)
                        {
                            float min = float.MaxValue;

                            for (int k = 0; k < fullshape[1]; k++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
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
                for (int k = 0; k < fullshape[1]; k++)
                {
                    for (int j = 0; j < fullshape[2]; j++)
                    {
                        for (int i = 0; i < fullshape[3]; i++)
                        {
                            float min = float.MaxValue;
                            for (int l = 0; l < fullshape[0]; l++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }
                            for (int l = 0; l < result.Batch; l++)
                            {
                                result[l, k, j, i] = min;
                            }

                        }
                    }
                }
            }

            return result;
        }
        public static Tensor Max(Tensor tensor, Dim dim, bool keepDim = false)
        {
            Tensor result = null;
            int[] fullshape = tensor.FullShape();

            if (keepDim)
                result = new(tensor.shape);
            else
            {
                if (dim == Dim.width) result = new(CreateShape(tensor.Rank, fullshape[0], fullshape[1], fullshape[2], 1));
                else if (dim == Dim.height) result = new(CreateShape(tensor.Rank, fullshape[0], fullshape[1], 1, fullshape[3]));
                else if (dim == Dim.channel) result = new(CreateShape(tensor.Rank, fullshape[0], 1, fullshape[2], fullshape[3]));
                else if (dim == Dim.batch) result = new(CreateShape(tensor.Rank, 1, fullshape[1], fullshape[2], fullshape[3]));
            }

            if (dim == Dim.width)
            {

                for (int l = 0; l < fullshape[0]; l++)
                {
                    for (int k = 0; k < fullshape[1]; k++)
                    {
                        for (int j = 0; j < fullshape[2]; j++)
                        {
                            float max = float.MinValue;
                            for (int i = 0; i < fullshape[3]; i++)
                            {
                                max = MathF.Max(max, tensor[l, k, j, i]);
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

                for (int l = 0; l < fullshape[0]; l++)
                {
                    for (int k = 0; k < fullshape[1]; k++)
                    {
                        for (int i = 0; i < fullshape[3]; i++)
                        {
                            float max = float.MinValue;
                            for (int j = 0; j < fullshape[2]; j++)
                            {
                                max = MathF.Max(max, tensor[l, k, j, i]);
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

                for (int l = 0; l < fullshape[0]; l++)
                {
                    for (int j = 0; j < fullshape[2]; j++)
                    {
                        for (int i = 0; i < fullshape[3]; i++)
                        {
                            float max = float.MinValue;

                            for (int k = 0; k < fullshape[1]; k++)
                            {
                                max = MathF.Max(max, tensor[l, k, j, i]);
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
                for (int k = 0; k < fullshape[1]; k++)
                {
                    for (int j = 0; j < fullshape[2]; j++)
                    {
                        for (int i = 0; i < fullshape[3]; i++)
                        {
                            float max = float.MinValue;
                            for (int l = 0; l < fullshape[0]; l++)
                            {
                                max = MathF.Max(max, tensor[l, k, j, i]);
                            }
                            for (int l = 0; l < result.Batch; l++)
                            {
                                result[l, k, j, i] = max;
                            }

                        }
                    }
                }
            }

            return result;
        }

        #endregion On Dimensions Operations


        #region On Axis Operation
        public static Tensor Unsqueeze(Tensor tensor, int? axis = null)
        {
            if (axis == null)
            {
                Tensor result = Identity(tensor);
                result.shape = CreateShape(tensor.Rank + 1, tensor.Batch, tensor.Channels, tensor.Height, tensor.Width);
                return result;
            }
            else
            {
                throw new NotImplementedException();
            }
        }
        public static Tensor Squeeze(Tensor tensor, int? axis = null)
        {
            if (axis == null)
            {
                LinkedList<int> squeezedShape = new LinkedList<int>();

                squeezedShape.AddFirst(tensor.Width);

                if (tensor.Height > 1)
                    squeezedShape.AddFirst(tensor.Height);

                if (tensor.Channels > 1)
                    squeezedShape.AddFirst(tensor.Channels);

                if (tensor.Batch > 1)
                    squeezedShape.AddFirst(tensor.Batch);

                Tensor result = new(squeezedShape.ToArray());
                Array.Copy(tensor.data, result.data, tensor.data.Length);
                return result;
            }
            else
            {
                throw new NotImplementedException();
            }

        }
        public static Tensor Transpose(Tensor tensor, int axis0, int axis1)
        {
            if (axis0 < 0 || axis0 >= tensor.Rank || axis1 < 0 || axis1 >= tensor.Rank)
                throw new ArgumentException("The specified axes are out of range for this tensor's rank.");


            if (axis0 == axis1)
                return Identity(tensor);


            axis0 = AxisToDim(tensor.Rank, axis0);
            axis1 = AxisToDim(tensor.Rank, axis1);
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
            int rank = tensor.Rank;
            int axisIndex = AxisToDim(rank, axis);
            int[] stackShape = tensor.FullShape();
            List<Tensor> slices = new();

            int dimLength = stackShape[axisIndex];
            int dimPos = 0;
            while (dimPos < dimLength)
            {
                int dimCopySize = Math.Min(split_size, dimLength - dimPos);
                int[] sliceShape = stackShape.ToArray();
                sliceShape[axisIndex] = dimCopySize;
                Tensor slice = new(CreateShape(tensor.Rank, sliceShape[0], sliceShape[1], sliceShape[2], sliceShape[3]));

                for (int l = 0; l < slice.Batch; l++)
                {
                    for (int k = 0; k < slice.Channels; k++)
                    {
                        for (int j = 0; j < slice.Height; j++)
                        {
                            for (int i = 0; i < slice.Width; i++)
                            {
                                switch (axisIndex)
                                {
                                    case 3:
                                        slice[l, k, j, i] = tensor[l, k, j, dimPos + i];
                                        break;
                                    case 2:
                                        slice[l, k, j, i] = tensor[l, k, j + dimPos, i];
                                        break;
                                    case 1:
                                        slice[l, k, j, i] = tensor[l, k + dimPos, j, i];
                                        break;
                                    case 0:
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
        public static Tensor Join(int axis, params Tensor[] tensors)
        {
            if (tensors == null || tensors.Length == 0)
                throw new ArgumentException("Tensor used for joining are not defined.");

            int rank = tensors[0].Rank;
            int axisIndex = AxisToDim(rank, axis);
            int no_slices = tensors.Length;



            Tensor slice = tensors[0];

            int[] result_shape = slice.FullShape();
            result_shape[axisIndex] *= no_slices;

            Tensor result = new(CreateShape(rank + 1, result_shape[0], result_shape[1], result_shape[2], result_shape[3]));

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
                                switch (axisIndex)
                                {
                                    case 3:
                                        result[l, k, j, s * slice.Width + i] = tensors[s][l, k, j, i];
                                        break;
                                    case 2:
                                        result[l, k, s * slice.Height + j, i] = tensors[s][l, k, j, i];
                                        break;
                                    case 1:
                                        result[l, s * slice.Channels + k, j, i] = tensors[s][l, k, j, i];
                                        break;
                                    case 0:
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
            int rank = tensor.Rank;
            int axisIndex = AxisToDim(rank, axis);
            int[] fullshape = tensor.FullShape();
            fullshape[axisIndex] *= times;

            Tensor result = new(CreateShape(tensor.Rank, fullshape[0], fullshape[1], fullshape[2], fullshape[3]));

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
                                switch (axisIndex)
                                {
                                    case 3:
                                        result[l, k, j, t * tensor.Width + i] = tensor[l, k, j, i];
                                        break;
                                    case 2:
                                        result[l, k, t * tensor.Height + j, i] = tensor[l, k, j, i];
                                        break;
                                    case 1:
                                        result[l, t * tensor.Channels + k, j, i] = tensor[l, k, j, i];
                                        break;
                                    case 0:
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
        public static Tensor Shuffle(Tensor tensor, int axis)
        {
            Tensor[] slices = Split(tensor, axis, 1);
            slices = Utils.Shuffle(slices).ToArray();
            return Join(axis, slices);
        }

        public static Tensor Sum(Tensor tensor, int axis, bool keepDim = false)
        {
            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            int dimIndex = AxisToDim(tensor.Rank, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = keepDim == true ? newShape[axis] : 1;
            Tensor result = new(newShape);

            if (dimIndex == 3)
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

                            for (int i = 0; i < width; i++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 2)
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

                            for (int j = 0; j < height; j++)
                            {
                                tensor[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 1)
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

                            for (int k = 0; k < channels; k++)
                            {
                                tensor[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 0)
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
                            for (int l = 0; l < batch; l++)
                            {
                                tensor[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }

            return result;
        }
        public static Tensor Mean(Tensor tensor, int axis, bool keepDim = false)
        {
            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            int dimIndex = AxisToDim(tensor.Rank, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = keepDim == true ? newShape[axis] : 1;
            Tensor result = new(newShape);


            if (dimIndex == 3)
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
                            for (int i = 0; i < width; i++)
                            {
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 2)
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

                            for (int j = 0; j < height; j++)
                            {
                                tensor[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 1)
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
                            for (int k = 0; k < channels; k++)
                            {
                                tensor[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 0)
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
                            for (int l = 0; l < batch; l++)
                            {
                                tensor[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }

            return result;
        }
        public static Tensor Var(Tensor tensor, int axis, int correction = 1, bool keepDim = false)
        {
            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            int dimIndex = AxisToDim(tensor.Rank, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = keepDim == true ? newShape[axis] : 1;
            Tensor result = new(newShape);

            if (dimIndex == 3)
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

                            for (int i = 0; i < width; i++)
                            {
                                result[l, k, j, i] = vr;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 2)
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

                            for (int j = 0; j < height; j++)
                            {
                                tensor[l, k, j, i] = vr;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 1)
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

                            for (int k = 0; k < channels; k++)
                            {
                                tensor[l, k, j, i] = vr;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 0)
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

                            for (int l = 0; l < batch; l++)
                            {
                                tensor[l, k, j, i] = vr;
                            }
                        }
                    }
                }
            }

            return result;
        }
        public static Tensor Std(Tensor tensor, int axis, int correction = 1, bool keepDim = false)
        {
            return Sqrt(Var(tensor, axis, correction, keepDim));
        }
        public static Tensor Min(Tensor tensor, int axis, bool keepDim = false)
        {
            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            int dimIndex = AxisToDim(tensor.Rank, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = keepDim == true ? newShape[axis] : 1;
            Tensor result = new(newShape);

            if (dimIndex == 3)
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

                            for (int i = 0; i < result.Width; i++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 2)
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

                            for (int j = 0; j < result.Height; j++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 1)
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

                            for (int k = 0; k < result.Channels; k++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 0)
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
                            for (int l = 0; l < result.Batch; l++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }

            return result;

        }
        public static Tensor Max(Tensor tensor, int axis, bool keepDim = false)
        {
            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            int dimIndex = AxisToDim(tensor.Rank, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = keepDim == true ? newShape[axis] : 1;
            Tensor result = new(newShape);

            if (dimIndex == 3)
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

                            for (int i = 0; i < result.Width; i++)
                            {
                                result[l, k, j, i] = m;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 2)
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

                            for (int j = 0; j < result.Height; j++)
                            {
                                result[l, k, j, i] = m;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 1)
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

                            for (int k = 0; k < result.Channels; k++)
                            {
                                result[l, k, j, i] = m;
                            }
                        }
                    }
                }
            }
            else if (dimIndex == 0)
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
                            for (int l = 0; l < result.Batch; l++)
                            {
                                result[l, k, j, i] = m;
                            }
                        }
                    }
                }
            }

            return result;

        }

        #endregion On Axis Operations


        #region Math Operations
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
        /// Computed the element-wise log density,
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
        /// Computed the element-wise density,
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
        /// Computed the element-wise Kullback-Leibler divergence.
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

        #endregion Math operations




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
        private static int DimToAxis(Tensor t, Dim dim)
        {
            int rank = t.Rank;

            if(rank == 0)
            {
                if (dim == Dim.width)
                    return 0;
            }
            if(rank == 1)
            {
                if (dim == Dim.width)
                    return 0;
            }
            if(rank == 2)
            {
                if (dim == Dim.width)
                    return 1;

                if (dim == Dim.height)
                    return 0;
            }
            if(rank == 3)
            {
                if (dim == Dim.width)
                    return 2;

                if (dim == Dim.height)
                    return 1;

                if (dim == Dim.channel)
                    return 0;
            }
            if (rank == 4)
            {
                if (dim == Dim.width)
                    return 3;

                if (dim == Dim.height)
                    return 2;

                if (dim == Dim.channel)
                    return 1;

                if (dim == Dim.batch)
                    return 0;
            }

            throw new ArgumentException("Cannot convert Dim to Axis");
        }
        private static int AxisToDim(int rank, int axis)
        {
            // Returns the index in the full shape array of the axis.
            // Used only for methods along the axis, that uses full shape call.

            if (axis > rank)
                throw new ArgumentException($"Cannot use axis {axis} for a tensor of rank {rank}.");


            if (rank == 0 && (axis == 0 || axis == -1))
                return 3;

            if (axis >= 0)
                return 4 - rank + axis;
            else
                return 4 + axis;

        }
        private int[] FullShape()
        {
            return new int[] { Batch, Channels, Height, Width };
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