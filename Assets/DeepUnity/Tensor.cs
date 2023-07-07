using System;
using System.Linq;
using System.Text;
using UnityEngine;
using System.Threading.Tasks;
using System.Collections.Generic;
using Unity.VisualScripting;

namespace DeepUnity
{
    /// <summary>
    /// Mutable. 4D.
    /// [batch, channels, height, width]
    /// </summary>
    [Serializable]
    public sealed partial class Tensor : IEquatable<Tensor>
    {
        [SerializeField] private TShape shape;
        [SerializeField] public float[] data;

        public static string PrintFormat = "0.00000";
        public int Rank => shape.Rank;


        public TShape Shape => shape;
        private int Size(int axis)
        {
            return shape.ToArray()[GetAxisIndex(Rank, axis)];
        }
        public int Size(TDim dim)
        {
            switch (dim)
            {
                case TDim.width: return shape.Width;
                case TDim.height: return shape.Height;
                case TDim.channel: return shape.Channels;
                case TDim.batch: return shape.Batch;
                default: throw new Exception("Unhandled TDim type");
            }
        }

        public float this[int w]
        {
            get => data[w];
            set => data[w] = value;
        }
        public float this[int h, int w]
        {
            get => data[w * shape.Height + h];
            set => data[w * shape.Height + h] = value;
        }
        public float this[int c, int h, int w]
        {
            get => data[c * shape.Height * shape.Width + w * shape.Height + h];
            set => data[c * shape.Height * shape.Width + w * shape.Height + h] = value;
        }
        public float this[int n, int c, int h, int w]
        {
            get => data[n * shape.Channels * shape.Height * shape.Width + c * shape.Height * shape.Width + w * shape.Height + h];
            set => data[n * shape.Channels * shape.Height * shape.Width + c * shape.Height * shape.Width + w * shape.Height + h] = value;

        }


        #region Create Tensor

        private Tensor(params int[] shortShape)
        {
            if (shortShape == null || shortShape.Length == 0)
                throw new ArgumentException("Tensor cannot be instantiated with null shape.");
            if (shortShape.Length > 4)
                throw new ArgumentException("Tensor cannot be instantiated with more than 4 dimensions.");

            int width = 1;
            int height = 1;
            int channels = 1;
            int batch = 1;
            if (shortShape.Length == 1)
            {
                width = shortShape[0];
            }
            else if (shortShape.Length == 2)
            {
                width = shortShape[1];
                height = shortShape[0];
            }
            else if (shortShape.Length == 3)
            {
                width = shortShape[2];
                height = shortShape[1];
                channels = shortShape[0];
            }
            else if (shortShape.Length == 4)
            {
                width = shortShape[3];
                height = shortShape[2];
                channels = shortShape[1];
                batch = shortShape[0];
            }

            int size = batch * channels * height * width;

            if (size > 16_777_216) // hardcoded like this because 4096x4096 max allowed matrix, on 8192 it crashes
                throw new NotSupportedException("Tensor dimensions is too large on initialization (cannot surpass 16,777,216 units).");

            shape = new TShape(batch, channels, height, width);
            data = new float[size];
        }
        private Tensor(TShape tshape)
        {
            this.shape = new TShape(tshape.Batch, tshape.Channels, tshape.Height, tshape.Width);

            int size = tshape.Batch * tshape.Channels * tshape.Height * tshape.Width;

            if (size > 16_777_216) // hardcoded like this because 4096x4096 max allowed matrix, on 8192 it crashes
                throw new NotSupportedException("Tensor dimensions is too large on initialization (cannot surpass 16,777,216 units).");

            data = new float[size];
        }
        public static Tensor Identity(Tensor other)
        {
            Tensor clone = new(other.shape);
            Array.Copy(other.data, clone.data, other.data.Length);
            return clone;
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
            int width = matrix.GetLength(0);
            int height = matrix.GetLength(1);

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
            int width = cube.GetLength(0);
            int height = cube.GetLength(1);
            int depth = cube.GetLength(2);

            Tensor t = new(depth, height, width);
            for (int z = 0; z < depth; z++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        t[z, y, x] = cube[x, y, z];
                    }
                }
            }

            return t;
        }
        public static Tensor Constant(float[,,,] tesseract)
        {
            int width = tesseract.GetLength(0);
            int height = tesseract.GetLength(1);
            int depth = tesseract.GetLength(2);
            int time = tesseract.GetLength(3);

            Tensor t = new(time, depth, height, width);
            for (int w = 0; w < time; w++)
            {
                for (int z = 0; z < depth; z++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            t[w, z, y, x] = tesseract[x, y, z, w];
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
        public static Tensor Fill(float value, params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = value;
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
            if (!left.shape.Equals(right.shape))
                throw new OperationCanceledException($"Left{left.Shape} and right{right.Shape} tensors must have different shape for Element-wise addition (+).");

            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] + right.data[i];
            }


            return result;
        }
        public static Tensor operator -(Tensor left, Tensor right)
        {
            if (!left.shape.Equals(right.shape))
                throw new OperationCanceledException($"Left{left.Shape} and right{right.Shape} tensors must have different shape for Element-wise subtraction (-).");

            Tensor result = new(left.shape);
            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] - right.data[i];
            }

            return result;
        }
        public static Tensor operator *(Tensor left, Tensor right)
        {
            if (!left.shape.Equals(right.shape))
                throw new OperationCanceledException($"Left{left.Shape} and right{right.Shape} tensors must have different shape for Element-wise multiplication (*).");

            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] * right.data[i];
            }


            return result;
        }
        public static Tensor operator /(Tensor left, Tensor right)
        {
            if (!left.shape.Equals(right.shape))
                throw new OperationCanceledException($"Left{left.Shape} and right{right.Shape} tensors must have different shape for Element-wise division (/).");

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
        /// left (k, n, m) * right (k, m, p) = result (k, n, p)
        /// </summary>
        public static Tensor MatMul(Tensor left, Tensor right)
        {
            /* N x M dot M x P => N x P
             */
            int w1 = left.shape.Height;
            int h1 = left.shape.Width;
            int w2 = right.shape.Height;
            int h2 = right.shape.Width;
            int b1 = left.shape.Channels;
            int b2 = right.shape.Channels;

            if (h1 != w2)
                throw new ArgumentException("Tensor must have compatible shapes for matrix multiplication (height of left tensor is not matching the width of the right tensor).");

            if (b1 != b2)
                throw new ArgumentException("Tensors must have similar number of channels for channeled matrix multiplication.");

            Tensor result = new(new TShape(1, b1, w1, h2));

            if (DeepUnityMeta.device == Device.CPU)
            {
                if (b1 == 1)
                {
                    Parallel.For(0, w1, m =>
                    {
                        for (int r = 0; r < h2; r++)
                        {
                            float sum = 0f;
                            for (int k = 0; k < h1; k++)
                            {
                                sum += left[m, k] * right[k, r];
                            }
                            result[m, r] = sum;

                        }

                    });
                }
                else
                {
                    Parallel.For(0, b1, b =>
                    {
                        for (int m = 0; m < w1; m++)
                        {
                            for (int r = 0; r < h2; r++)
                            {
                                float sum = 0f;
                                for (int k = 0; k < h1; k++)
                                {
                                    sum += left[b, m, k] * right[b, k, r];
                                }
                                result[b, m, r] = sum;

                            }
                        }
                    });
                }

            }
            else
            {
                ComputeShader CS = DeepUnityMeta.MatMulCS;

                ComputeBuffer leftBuffer = new(left.data.Length, 4);
                ComputeBuffer rightBuffer = new(right.data.Length, 4);
                ComputeBuffer resultBuffer = new(b1 * h2 * w1, 4);

                leftBuffer.SetData(left.data);
                rightBuffer.SetData(right.data);


                CS.SetBuffer(0, "leftArr", leftBuffer);
                CS.SetBuffer(0, "rightArr", rightBuffer);
                CS.SetBuffer(0, "resultArr", resultBuffer);
                CS.SetInt("w1", w1);
                CS.SetInt("h1w2", h1);
                CS.SetInt("h2", h2);

                CS.Dispatch(0,
                           (w1 + DeepUnityMeta.numthreads[0] - 1) / DeepUnityMeta.numthreads[0],
                           (h2 + DeepUnityMeta.numthreads[1] - 1) / DeepUnityMeta.numthreads[1],
                           (b1 + DeepUnityMeta.numthreads[2] - 1) / DeepUnityMeta.numthreads[2]);

                resultBuffer.GetData(result.data);

                leftBuffer.Release();
                rightBuffer.Release();
                resultBuffer.Release();
            }

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

            int w = tensor.shape.Width + 2;
            int h = tensor.shape.Height + 2;
            int b = tensor.shape.Channels;
            int n = tensor.shape.Batch;
            Tensor result = new(n, b, h, w);

            for (int l = 0; l < tensor.shape.Batch; l++)
            {
                for (int k = 0; k < tensor.shape.Channels; k++)
                {
                    for (int j = 0; j < tensor.shape.Height; j++)
                    {
                        for (int i = 0; i < tensor.shape.Width; i++)
                        {
                            result[l, k, j + 1, i + 1] = tensor[k, j, i];
                        }
                    }
                }
            }


            if (paddingMode == PaddingType.Mirror)
            {

                for (int l = 0; l < tensor.shape.Batch; l++)
                {
                    for (int k = 0; k < tensor.shape.Channels; k++)
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

            int outputChannels = kernels.Size(TDim.batch);
            int inputChannels = kernels.Size(TDim.channel);

            int batchSize = inputs.Size(TDim.batch);
            int inputHeight = inputs.Size(TDim.height);
            int inputWidth = inputs.Size(TDim.width);
            int kernelHeight = kernels.Size(TDim.height);
            int kernelWidth = kernels.Size(TDim.width);

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
            int height = kernels.Size(TDim.height);
            int width = kernels.Size(TDim.width);
            Tensor rot180Kernel = new(kernels.shape);

            for (int c = 0; c < kernels.Size(TDim.channel); c++)
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

        public static Tensor Transpose(Tensor tensor, TDim dim0, TDim dim1)
        {
            if (dim0 == dim1)
                return tensor;

            TShape swappedShape = null;

            if (dim0 == TDim.width)
            {
                if (dim1 == TDim.height)
                    swappedShape = new TShape(tensor.shape.Batch, tensor.shape.Channels, tensor.shape.Width, tensor.shape.Height);
                else if (dim1 == TDim.channel)
                    swappedShape = new TShape(tensor.shape.Batch, tensor.shape.Width, tensor.shape.Height, tensor.shape.Channels);
                else if (dim1 == TDim.batch)
                    swappedShape = new TShape(tensor.shape.Width, tensor.shape.Channels, tensor.shape.Height, tensor.shape.Batch);
            }
            else if (dim0 == TDim.height)
            {
                if (dim1 == TDim.width)
                    swappedShape = new TShape(tensor.shape.Batch, tensor.shape.Channels, tensor.shape.Height, tensor.shape.Width);
                else if (dim1 == TDim.channel)
                    swappedShape = new TShape(tensor.shape.Batch, tensor.shape.Height, tensor.shape.Width, tensor.shape.Channels);
                else if (dim1 == TDim.batch)
                    swappedShape = new TShape(tensor.shape.Height, tensor.shape.Batch, tensor.shape.Width, tensor.shape.Channels);
            }
            else if (dim0 == TDim.channel)
            {
                if (dim1 == TDim.width)
                    swappedShape = new TShape(tensor.shape.Batch, tensor.shape.Height, tensor.shape.Channels, tensor.shape.Width);
                else if (dim1 == TDim.height)
                    swappedShape = new TShape(tensor.shape.Batch, tensor.shape.Channels, tensor.shape.Height, tensor.shape.Width);
                else if (dim1 == TDim.batch)
                    swappedShape = new TShape(tensor.shape.Channels, tensor.shape.Batch, tensor.shape.Height, tensor.shape.Width);
            }
            else if (dim0 == TDim.batch)
            {
                if (dim1 == TDim.width)
                    swappedShape = new TShape(tensor.shape.Channels, tensor.shape.Batch, tensor.shape.Height, tensor.shape.Width);
                else if (dim1 == TDim.height)
                    swappedShape = new TShape(tensor.shape.Height, tensor.shape.Batch, tensor.shape.Channels, tensor.shape.Width);
                else if (dim1 == TDim.channel)
                    swappedShape = new TShape(tensor.shape.Batch, tensor.shape.Height, tensor.shape.Channels, tensor.shape.Width);
            }

            Tensor result = new(swappedShape);

            for (int l = 0; l < swappedShape.Batch; l++)
            {
                for (int k = 0; k < swappedShape.Channels; k++)
                {
                    for (int j = 0; j < swappedShape.Height; j++)
                    {
                        for (int i = 0; i < swappedShape.Width; i++)
                        {
                            if (dim0 == TDim.width && dim1 == TDim.height)
                                result[l, k, j, i] = tensor[l, k, i, j];
                            else if (dim0 == TDim.width && dim1 == TDim.channel)
                                result[l, j, i, k] = tensor[l, k, i, j];
                            else if (dim0 == TDim.width && dim1 == TDim.batch)
                                result[i, k, j, l] = tensor[l, k, i, j];

                            else if (dim0 == TDim.height && dim1 == TDim.width)
                                result[l, k, i, j] = tensor[l, k, j, i];
                            else if (dim0 == TDim.height && dim1 == TDim.channel)
                                result[l, j, k, i] = tensor[l, k, i, j];
                            else if (dim0 == TDim.height && dim1 == TDim.batch)
                                result[i, k, j, l] = tensor[l, k, i, j];

                            else if (dim0 == TDim.channel && dim1 == TDim.width)
                                result[l, k, i, j] = tensor[l, k, j, i];
                            else if (dim0 == TDim.channel && dim1 == TDim.height)
                                result[l, j, k, i] = tensor[l, k, i, j];
                            else if (dim0 == TDim.channel && dim1 == TDim.batch)
                                result[i, j, k, l] = tensor[l, k, i, j];

                            else if (dim0 == TDim.batch && dim1 == TDim.width)
                                result[i, k, j, l] = tensor[l, k, j, i];
                            else if (dim0 == TDim.batch && dim1 == TDim.height)
                                result[j, k, i, l] = tensor[l, k, i, j];
                            else if (dim0 == TDim.batch && dim1 == TDim.channel)
                                result[i, j, k, l] = tensor[l, k, i, j];
                            else
                                throw new ArgumentException("Something went wrong bro.");

                        }
                    }
                }
            }

            return result;
        }
        public static Tensor Var(Tensor tensor, TDim dim, int correction = 1, bool keepDim = false)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();

            if (!keepDim)
            {
                if (dim == TDim.width)
                {
                    result = new(shape[0], shape[1], shape[2], 1);
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int k = 0; k < shape[1]; k++)
                        {
                            for (int j = 0; j < shape[2]; j++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int i = 0; i < shape[3]; i++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }
                                result[l, k, j, 0] = (sumSqr - (sum * sum) / shape[3]) / (shape[3] - correction);
                            }
                        }
                    }
                }
                else if (dim == TDim.height)
                {
                    result = new(shape[0], shape[1], 1, shape[3]);
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int k = 0; k < shape[1]; k++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int j = 0; j < shape[2]; j++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }
                                result[l, k, 0, i] = (sumSqr - (sum * sum) / shape[2]) / (shape[2] - correction);
                            }
                        }
                    }
                }
                else if (dim == TDim.channel)
                {
                    result = new(shape[0], 1, shape[2], shape[3]);
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int k = 0; k < shape[1]; k++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }

                                result[l, 0, j, i] = (sumSqr - (sum * sum) / shape[1]) / (shape[1] - correction);
                            }
                        }
                    }
                }
                else if (dim == TDim.batch)
                {
                    result = new(1, shape[1], shape[2], shape[3]);
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int l = 0; l < shape[0]; l++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }
                                result[0, k, j, i] = (sumSqr - (sum * sum) / shape[0]) / (shape[0] - correction);
                            }
                        }
                    }
                }

            }
            else
            {
                result = new(tensor.shape);
                if (dim == TDim.width)
                {
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int k = 0; k < shape[1]; k++)
                        {
                            for (int j = 0; j < shape[2]; j++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int i = 0; i < shape[3]; i++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }

                                float res = (sumSqr - (sum * sum) / shape[3]) / (shape[3] - correction);
                                for (int i = 0; i < shape[3]; i++)
                                {
                                    result[l, k, j, i] = res;
                                }

                            }
                        }
                    }
                }
                else if (dim == TDim.height)
                {
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int k = 0; k < shape[1]; k++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int j = 0; j < shape[2]; j++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }
                                float res = (sumSqr - (sum * sum) / shape[2]) / (shape[2] - correction);
                                for (int j = 0; j < shape[2]; j++)
                                {
                                    result[l, k, j, i] = res;
                                }
                            }
                        }
                    }
                }
                else if (dim == TDim.channel)
                {
                    for (int l = 0; l < shape[0]; l++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int k = 0; k < shape[1]; k++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }

                                float res = (sumSqr - (sum * sum) / shape[1]) / (shape[1] - correction);
                                for (int k = 0; k < shape[1]; k++)
                                {
                                    result[l, k, j, i] = res;
                                }
                            }
                        }
                    }
                }
                else if (dim == TDim.batch)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float sum = 0f;
                                float sumSqr = 0f;
                                for (int l = 0; l < shape[0]; l++)
                                {
                                    float value = tensor[l, k, j, i];
                                    sum += value;
                                    sumSqr += value * value;
                                }
                                float res = (sumSqr - (sum * sum) / shape[0]) / (shape[0] - correction);
                                for (int l = 0; l < shape[0]; l++)
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
        public static Tensor Std(Tensor tensor, TDim dim, int correction = 1, bool keepDim = false)
        {
            return Sqrt(Var(tensor, dim, correction, keepDim));
        }
        public static Tensor Mean(Tensor tensor, TDim dim, bool keepDim = false)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();

            if (!keepDim)
            {
                if (dim == TDim.width)
                {
                    result = new(shape[0], shape[1], shape[2], 1);
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
                else if (dim == TDim.height)
                {
                    result = new(shape[0], shape[1], 1, shape[3]);
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
                else if (dim == TDim.channel)
                {
                    result = new(shape[0], 1, shape[2], shape[3]);
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
                else if (dim == TDim.batch)
                {
                    result = new(1, shape[1], shape[2], shape[3]);
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
                if (dim == TDim.width)
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
                else if (dim == TDim.height)
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
                else if (dim == TDim.channel)
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
                else if (dim == TDim.batch)
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
        public static Tensor Sum(Tensor tensor, TDim dim, bool keepDim = false)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();

            if (!keepDim)
            {
                if (dim == TDim.width)
                {
                    result = new(shape[0], shape[1], shape[2], 1);
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
                else if (dim == TDim.height)
                {
                    result = new(shape[0], shape[1], 1, shape[3]);
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
                else if (dim == TDim.channel)
                {
                    result = new(shape[0], 1, shape[2], shape[3]);
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
                else if (dim == TDim.batch)
                {
                    result = new(1, shape[1], shape[2], shape[3]);
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
                if (dim == TDim.width)
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
                else if (dim == TDim.height)
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
                else if (dim == TDim.channel)
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
                else if (dim == TDim.batch)
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
        public static Tensor Shuffle(Tensor tensor, TDim dim)
        {
            Tensor[] slices = Split(tensor, dim, 1);
            slices = Utils.Shuffle(slices).ToArray();
            return Join(dim, slices);
        }
        public static Tensor Expand(Tensor tensor, TDim dim, int times)
        {
            if (times == 1)
                return Identity(tensor);

            TShape shape = null;
            switch (dim)
            {
                case TDim.width:
                    shape = new TShape(tensor.shape.Batch, tensor.shape.Channels, tensor.shape.Height, times);
                    break;
                case TDim.height:
                    shape = new TShape(tensor.shape.Batch, tensor.shape.Channels, times, tensor.shape.Width);
                    break;
                case TDim.channel:
                    shape = new TShape(tensor.shape.Batch, times, tensor.shape.Height, tensor.shape.Width);
                    break;
                case TDim.batch:
                    shape = new TShape(times, tensor.shape.Channels, tensor.shape.Height, tensor.shape.Width);
                    break;

            }
            Tensor result = new(shape);

            for (int t = 0; t < times; t++)
            {
                for (int l = 0; l < tensor.shape.Batch; l++)
                {
                    for (int k = 0; k < tensor.shape.Channels; k++)
                    {
                        for (int j = 0; j < tensor.shape.Height; j++)
                        {
                            for (int i = 0; i < tensor.shape.Width; i++)
                            {
                                switch (dim)
                                {
                                    case TDim.width:
                                        result[l, k, j, t * tensor.shape.Width + i] = tensor[l, k, j, i];
                                        break;
                                    case TDim.height:
                                        result[l, k, t * tensor.shape.Height + j, i] = tensor[l, k, j, i];
                                        break;
                                    case TDim.channel:
                                        result[l, t * tensor.shape.Channels + k, j, i] = tensor[l, k, j, i];
                                        break;
                                    case TDim.batch:
                                        result[t * tensor.shape.Batch + l, k, j, i] = tensor[l, k, j, i];
                                        break;
                                }

                            }
                        }
                    }
                }

            }


            return result;
        }
        public static Tensor Join(TDim dim, params Tensor[] tensors)
        {
            if (tensors == null || tensors.Length == 0)
                throw new ArgumentException("Tensor used for joining are not defined.");

            if (tensors.Length == 1)
                return Identity(tensors[0]);

            int no_slices = tensors.Length;
            Tensor slice = tensors[0];
            TShape shape = null;
            switch (dim)
            {
                case TDim.width:
                    shape = new TShape(slice.shape.Batch, slice.shape.Channels, slice.shape.Height, slice.shape.Width * no_slices);
                    break;
                case TDim.height:
                    shape = new TShape(slice.shape.Batch, slice.shape.Channels, slice.shape.Height * no_slices, slice.shape.Width);
                    break;
                case TDim.channel:
                    shape = new TShape(slice.shape.Batch, slice.shape.Channels * no_slices, slice.shape.Height, slice.shape.Width);
                    break;
                case TDim.batch:
                    shape = new TShape(slice.shape.Batch * no_slices, slice.shape.Channels, slice.shape.Height, slice.shape.Width);
                    break;
            }
            Tensor result = new(shape);

            for (int s = 0; s < no_slices; s++)
            {
                for (int l = 0; l < slice.shape.Batch; l++)
                {
                    for (int k = 0; k < slice.shape.Channels; k++)
                    {
                        for (int j = 0; j < slice.shape.Height; j++)
                        {
                            for (int i = 0; i < slice.shape.Width; i++)
                            {
                                switch (dim)
                                {
                                    case TDim.width:
                                        result[l, k, j, s * slice.shape.Width + i] = tensors[s][l, k, j, i];
                                        break;
                                    case TDim.height:
                                        result[l, k, s * slice.shape.Height + j, i] = tensors[s][l, k, j, i];
                                        break;
                                    case TDim.channel:
                                        result[l, s * slice.shape.Channels + k, j, i] = tensors[s][l, k, j, i];
                                        break;
                                    case TDim.batch:
                                        result[s * slice.shape.Batch + l, k, j, i] = tensors[s][l, k, j, i];
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
        public static Tensor[] Split(Tensor tensor, TDim dim, int split_size)
        {
            List<Tensor> slices = new();

            int dimLength = tensor.Size(dim);
            int dimPos = 0;
            while (dimPos < dimLength)
            {
                int dimCopySize = Math.Min(split_size, dimLength - dimPos);

                TShape shape = null;
                switch (dim)
                {
                    case TDim.width:
                        shape = new TShape(tensor.shape.Batch, tensor.shape.Channels, tensor.shape.Height, dimCopySize);
                        break;
                    case TDim.height:
                        shape = new TShape(tensor.shape.Batch, tensor.shape.Channels, dimCopySize, tensor.shape.Width);
                        break;
                    case TDim.channel:
                        shape = new TShape(tensor.shape.Batch, dimCopySize, tensor.shape.Height, tensor.shape.Width);
                        break;
                    case TDim.batch:
                        shape = new TShape(dimCopySize, tensor.shape.Channels, tensor.shape.Height, tensor.shape.Width);
                        break;

                }
                Tensor slice = new(shape);

                for (int l = 0; l < slice.shape.Batch; l++)
                {
                    for (int k = 0; k < slice.shape.Channels; k++)
                    {
                        for (int j = 0; j < slice.shape.Height; j++)
                        {
                            for (int i = 0; i < slice.shape.Width; i++)
                            {
                                switch (dim)
                                {
                                    case TDim.width:
                                        slice[l, k, j, i] = tensor[l, k, j, dimPos + i];
                                        break;
                                    case TDim.height:
                                        slice[l, k, j, i] = tensor[l, k, j + dimPos, i];
                                        break;
                                    case TDim.channel:
                                        slice[l, k, j, i] = tensor[l, k + dimPos, j, i];
                                        break;
                                    case TDim.batch:
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
        public static Tensor Min(Tensor tensor, TDim dim, bool keepDim = false)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();

            if (keepDim)
                result = new(tensor.shape);
            else
            {
                if (dim == TDim.width) result = new(shape[0], shape[1], shape[2], 1);
                else if (dim == TDim.height) result = new(shape[0], shape[1], 1, shape[3]);
                else if (dim == TDim.channel) result = new(shape[0], 1, shape[2], shape[3]);
                else if (dim == TDim.batch) result = new(1, shape[1], shape[2], shape[3]);
            }

            if (dim == TDim.width)
            {

                for (int l = 0; l < shape[0]; l++)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            float min = float.MaxValue;
                            for (int i = 0; i < shape[3]; i++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }

                            for (int i = 0; i < result.shape.Width; i++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }
            else if (dim == TDim.height)
            {

                for (int l = 0; l < shape[0]; l++)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float min = float.MaxValue;
                            for (int j = 0; j < shape[2]; j++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }

                            for (int j = 0; j < result.shape.Height; j++)
                            {
                                result[l, k, j, i] = min;
                            }

                        }
                    }
                }
            }
            else if (dim == TDim.channel)
            {

                for (int l = 0; l < shape[0]; l++)
                {
                    for (int j = 0; j < shape[2]; j++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float min = float.MaxValue;

                            for (int k = 0; k < shape[1]; k++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }

                            for (int k = 0; k < result.shape.Channels; k++)
                            {
                                result[l, k, j, i] = min;
                            }


                        }
                    }
                }
            }
            else if (dim == TDim.batch)
            {
                for (int k = 0; k < shape[1]; k++)
                {
                    for (int j = 0; j < shape[2]; j++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float min = float.MaxValue;
                            for (int l = 0; l < shape[0]; l++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }
                            for (int l = 0; l < result.shape.Batch; l++)
                            {
                                result[l, k, j, i] = min;
                            }

                        }
                    }
                }
            }

            return result;
        }
        public static Tensor Max(Tensor tensor, TDim dim, bool keepDim = false)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();

            if (keepDim)
                result = new(tensor.shape);
            else
            {
                if (dim == TDim.width) result = new(shape[0], shape[1], shape[2], 1);
                else if (dim == TDim.height) result = new(shape[0], shape[1], 1, shape[3]);
                else if (dim == TDim.channel) result = new(shape[0], 1, shape[2], shape[3]);
                else if (dim == TDim.batch) result = new(1, shape[1], shape[2], shape[3]);
            }

            if (dim == TDim.width)
            {

                for (int l = 0; l < shape[0]; l++)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            float max = float.MinValue;
                            for (int i = 0; i < shape[3]; i++)
                            {
                                max = MathF.Max(max, tensor[l, k, j, i]);
                            }

                            for (int i = 0; i < result.shape.Width; i++)
                            {
                                result[l, k, j, i] = max;
                            }
                        }
                    }
                }
            }
            else if (dim == TDim.height)
            {

                for (int l = 0; l < shape[0]; l++)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float max = float.MinValue;
                            for (int j = 0; j < shape[2]; j++)
                            {
                                max = MathF.Max(max, tensor[l, k, j, i]);
                            }

                            for (int j = 0; j < result.shape.Height; j++)
                            {
                                result[l, k, j, i] = max;
                            }

                        }
                    }
                }
            }
            else if (dim == TDim.channel)
            {

                for (int l = 0; l < shape[0]; l++)
                {
                    for (int j = 0; j < shape[2]; j++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float max = float.MinValue;

                            for (int k = 0; k < shape[1]; k++)
                            {
                                max = MathF.Max(max, tensor[l, k, j, i]);
                            }

                            for (int k = 0; k < result.shape.Channels; k++)
                            {
                                result[l, k, j, i] = max;
                            }


                        }
                    }
                }
            }
            else if (dim == TDim.batch)
            {
                for (int k = 0; k < shape[1]; k++)
                {
                    for (int j = 0; j < shape[2]; j++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float max = float.MinValue;
                            for (int l = 0; l < shape[0]; l++)
                            {
                                max = MathF.Max(max, tensor[l, k, j, i]);
                            }
                            for (int l = 0; l < result.shape.Batch; l++)
                            {
                                result[l, k, j, i] = max;
                            }

                        }
                    }
                }
            }

            return result;
        }

        #endregion


        #region On Axis Operation

        private static Tensor Transpose(Tensor tensor, int axis0, int axis1)
        {
            if (axis0 < 0 || axis0 >= tensor.Rank || axis1 < 0 || axis1 >= tensor.Rank)
                throw new ArgumentException("The specified axes are out of range for this tensor's rank.");


            if (axis0 == axis1)
                return Identity(tensor);


            axis0 = GetAxisIndex(tensor.Rank, axis0);
            axis1 = GetAxisIndex(tensor.Rank, axis1);
            int[] permutation = new int[] { tensor.shape.Batch, tensor.shape.Channels, tensor.shape.Height, tensor.shape.Width };


            var temp = permutation[axis0];
            permutation[axis0] = permutation[axis1];
            permutation[axis1] = temp;

            Tensor result = new(permutation);


            for (int l = 0; l < tensor.shape.Batch; l++)
            {
                for (int k = 0; k < tensor.shape.Channels; k++)
                {
                    for (int j = 0; j < tensor.shape.Height; j++)
                    {
                        for (int i = 0; i < tensor.shape.Width; i++)
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
        private static Tensor[] Split(Tensor tensor, int axis, int split_size)
        {
            int rank = tensor.Rank;
            int axisIndex = GetAxisIndex(rank, axis);
            int[] stackShape = tensor.shape.ToArray();
            List<Tensor> slices = new();

            int dimLength = stackShape[axisIndex];
            int dimPos = 0;
            while (dimPos < dimLength)
            {
                int dimCopySize = Math.Min(split_size, dimLength - dimPos);
                int[] sliceShape = stackShape.ToArray();
                sliceShape[axisIndex] = dimCopySize;
                Tensor slice = new(sliceShape);

                for (int l = 0; l < slice.shape.Batch; l++)
                {
                    for (int k = 0; k < slice.shape.Channels; k++)
                    {
                        for (int j = 0; j < slice.shape.Height; j++)
                        {
                            for (int i = 0; i < slice.shape.Width; i++)
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
        private static Tensor Join(int axis, params Tensor[] tensors)
        {
            if (tensors == null || tensors.Length == 0)
                throw new ArgumentException("Tensor used for joining are not defined.");

            int rank = tensors[0].Rank;
            int axisIndex = GetAxisIndex(rank, axis);
            int no_slices = tensors.Length;



            Tensor slice = tensors[0];

            int[] result_shape = slice.shape.ToArray();
            result_shape[axisIndex] *= no_slices;

            Tensor result = new(result_shape);

            for (int s = 0; s < no_slices; s++)
            {
                for (int l = 0; l < slice.shape.Batch; l++)
                {
                    for (int k = 0; k < slice.shape.Channels; k++)
                    {
                        for (int j = 0; j < slice.shape.Height; j++)
                        {
                            for (int i = 0; i < slice.shape.Width; i++)
                            {
                                switch (axisIndex)
                                {
                                    case 3:
                                        result[l, k, j, s * slice.shape.Width + i] = tensors[s][l, k, j, i];
                                        break;
                                    case 2:
                                        result[l, k, s * slice.shape.Height + j, i] = tensors[s][l, k, j, i];
                                        break;
                                    case 1:
                                        result[l, s * slice.shape.Channels + k, j, i] = tensors[s][l, k, j, i];
                                        break;
                                    case 0:
                                        result[s * slice.shape.Batch + l, k, j, i] = tensors[s][l, k, j, i];
                                        break;
                                }

                            }
                        }
                    }
                }
            }


            return result;
        }
        private static Tensor Expand(Tensor tensor, int axis, int times)
        {
            int rank = tensor.Rank;
            int axisIndex = GetAxisIndex(rank, axis);
            int[] shape = tensor.shape.ToArray();
            shape[axisIndex] *= times;

            Tensor result = new(shape);

            for (int t = 0; t < times; t++)
            {
                for (int l = 0; l < tensor.shape.Batch; l++)
                {
                    for (int k = 0; k < tensor.shape.Channels; k++)
                    {
                        for (int j = 0; j < tensor.shape.Height; j++)
                        {
                            for (int i = 0; i < tensor.shape.Width; i++)
                            {
                                switch (axisIndex)
                                {
                                    case 3:
                                        result[l, k, j, t * tensor.shape.Width + i] = tensor[l, k, j, i];
                                        break;
                                    case 2:
                                        result[l, k, t * tensor.shape.Height + j, i] = tensor[l, k, j, i];
                                        break;
                                    case 1:
                                        result[l, t * tensor.shape.Channels + k, j, i] = tensor[l, k, j, i];
                                        break;
                                    case 0:
                                        result[t * tensor.shape.Batch + l, k, j, i] = tensor[l, k, j, i];
                                        break;
                                }

                            }
                        }
                    }
                }

            }


            return result;
        }
        private static Tensor Shuffle(Tensor tensor, int axis)
        {
            Tensor[] slices = Split(tensor, axis, 1);
            slices = Utils.Shuffle(slices).ToArray();
            return Join(axis, slices);
        }
        private static Tensor Sum(Tensor tensor, int axis)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();
            int axisIndex = GetAxisIndex(tensor.Rank, axis);

            if (axisIndex == 3)
            {
                result = new(shape[0], shape[1], shape[2], 1);
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
            else if (axisIndex == 2)
            {
                result = new(shape[0], shape[1], 1, shape[3]);
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
            else if (axisIndex == 1)
            {
                result = new(shape[0], 1, shape[2], shape[3]);
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
            else if (axisIndex == 0)
            {
                result = new(1, shape[1], shape[2], shape[3]);
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

            return result;
        }
        private static Tensor Mean(Tensor tensor, int axis)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();
            int axisIndex = GetAxisIndex(tensor.Rank, axis);

            if (axisIndex == 3)
            {
                result = new(shape[0], shape[1], shape[2], 1);
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
            else if (axisIndex == 2)
            {
                result = new(shape[0], shape[1], 1, shape[3]);
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
            else if (axisIndex == 1)
            {
                result = new(shape[0], 1, shape[2], shape[3]);
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
            else if (axisIndex == 0)
            {
                result = new(1, shape[1], shape[2], shape[3]);
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

            return result;
        }
        private static Tensor Var(Tensor tensor, int axis, int correction = 1)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();
            int axisIndex = GetAxisIndex(tensor.Rank, axis);

            if (axisIndex == 3)
            {
                result = new(shape[0], shape[1], shape[2], 1);
                for (int l = 0; l < shape[0]; l++)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            float sum = 0f;
                            float sumSqr = 0f;
                            for (int i = 0; i < shape[3]; i++)
                            {
                                float value = tensor[l, k, j, i];
                                sum += value;
                                sumSqr += value * value;
                            }
                            result[l, k, j, 0] = (sumSqr - (sum * sum) / shape[3]) / (shape[3] - correction);
                        }
                    }
                }
            }
            else if (axisIndex == 2)
            {
                result = new(shape[0], shape[1], 1, shape[3]);
                for (int l = 0; l < shape[0]; l++)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float sum = 0f;
                            float sumSqr = 0f;
                            for (int j = 0; j < shape[2]; j++)
                            {
                                float value = tensor[l, k, j, i];
                                sum += value;
                                sumSqr += value * value;
                            }
                            result[l, k, 0, i] = (sumSqr - (sum * sum) / shape[2]) / (shape[2] - correction);
                        }
                    }
                }
            }
            else if (axisIndex == 1)
            {
                result = new(shape[0], 1, shape[2], shape[3]);
                for (int l = 0; l < shape[0]; l++)
                {
                    for (int j = 0; j < shape[2]; j++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float sum = 0f;
                            float sumSqr = 0f;
                            for (int k = 0; k < shape[1]; k++)
                            {
                                float value = tensor[l, k, j, i];
                                sum += value;
                                sumSqr += value * value;
                            }

                            result[l, 0, j, i] = (sumSqr - (sum * sum) / shape[1]) / (shape[1] - correction);
                        }
                    }
                }
            }
            else if (axisIndex == 0)
            {
                result = new(1, shape[1], shape[2], shape[3]);
                for (int k = 0; k < shape[1]; k++)
                {
                    for (int j = 0; j < shape[2]; j++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float sum = 0f;
                            float sumSqr = 0f;
                            for (int l = 0; l < shape[0]; l++)
                            {
                                float value = tensor[l, k, j, i];
                                sum += value;
                                sumSqr += value * value;
                            }
                            result[0, k, j, i] = (sumSqr - (sum * sum) / shape[0]) / (shape[0] - correction);
                        }
                    }
                }
            }

            return result;
        }
        private static Tensor Std(Tensor tensor, int axis, int correction = 1)
        {
            return Sqrt(Var(tensor, axis, correction));
        }
        private static Tensor Min(Tensor tensor, int axis, bool keepDim = false)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();
            int axisIndex = GetAxisIndex(tensor.Rank, axis);

            if (keepDim)
                result = new(tensor.shape);
            else
            {
                if (axisIndex == 3) result = new(shape[0], shape[1], shape[2], 1);
                else if (axisIndex == 2) result = new(shape[0], shape[1], 1, shape[3]);
                else if (axisIndex == 1) result = new(shape[0], 1, shape[2], shape[3]);
                else if (axisIndex == 0) result = new(1, shape[1], shape[2], shape[3]);
            }

            if (axisIndex == 3)
            {

                for (int l = 0; l < shape[0]; l++)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            float min = float.MaxValue;
                            for (int i = 0; i < shape[3]; i++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }

                            for (int i = 0; i < result.shape.Width; i++)
                            {
                                result[l, k, j, i] = min;
                            }
                        }
                    }
                }
            }
            else if (axisIndex == 2)
            {

                for (int l = 0; l < shape[0]; l++)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float min = float.MaxValue;
                            for (int j = 0; j < shape[2]; j++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }

                            for (int j = 0; j < result.shape.Height; j++)
                            {
                                result[l, k, j, i] = min;
                            }

                        }
                    }
                }
            }
            else if (axisIndex == 1)
            {

                for (int l = 0; l < shape[0]; l++)
                {
                    for (int j = 0; j < shape[2]; j++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float min = float.MaxValue;

                            for (int k = 0; k < shape[1]; k++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }

                            for (int k = 0; k < result.shape.Channels; k++)
                            {
                                result[l, k, j, i] = min;
                            }


                        }
                    }
                }
            }
            else if (axisIndex == 0)
            {
                for (int k = 0; k < shape[1]; k++)
                {
                    for (int j = 0; j < shape[2]; j++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float min = float.MaxValue;
                            for (int l = 0; l < shape[0]; l++)
                            {
                                min = MathF.Min(min, tensor[l, k, j, i]);
                            }
                            for (int l = 0; l < result.shape.Batch; l++)
                            {
                                result[l, k, j, i] = min;
                            }

                        }
                    }
                }
            }

            return result;

        }
        private static Tensor Max(Tensor tensor, int axis, bool keepDim = false)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();
            int axisIndex = GetAxisIndex(tensor.Rank, axis);

            if (keepDim)
                result = new(tensor.shape);
            else
            {
                if (axisIndex == 3) result = new(shape[0], shape[1], shape[2], 1);
                else if (axisIndex == 2) result = new(shape[0], shape[1], 1, shape[3]);
                else if (axisIndex == 1) result = new(shape[0], 1, shape[2], shape[3]);
                else if (axisIndex == 0) result = new(1, shape[1], shape[2], shape[3]);
            }

            if (axisIndex == 3)
            {

                for (int l = 0; l < shape[0]; l++)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int j = 0; j < shape[2]; j++)
                        {
                            float max = float.MinValue;
                            for (int i = 0; i < shape[3]; i++)
                            {
                                max = MathF.Max(max, tensor[l, k, j, i]);
                            }

                            for (int i = 0; i < result.shape.Width; i++)
                            {
                                result[l, k, j, i] = max;
                            }
                        }
                    }
                }
            }
            else if (axisIndex == 2)
            {

                for (int l = 0; l < shape[0]; l++)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float max = float.MinValue;
                            for (int j = 0; j < shape[2]; j++)
                            {
                                max = MathF.Max(max, tensor[l, k, j, i]);
                            }

                            for (int j = 0; j < result.shape.Height; j++)
                            {
                                result[l, k, j, i] = max;
                            }

                        }
                    }
                }
            }
            else if (axisIndex == 1)
            {

                for (int l = 0; l < shape[0]; l++)
                {
                    for (int j = 0; j < shape[2]; j++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float max = float.MinValue;

                            for (int k = 0; k < shape[1]; k++)
                            {
                                max = MathF.Max(max, tensor[l, k, j, i]);
                            }

                            for (int k = 0; k < result.shape.Channels; k++)
                            {
                                result[l, k, j, i] = max;
                            }


                        }
                    }
                }
            }
            else if (axisIndex == 0)
            {
                for (int k = 0; k < shape[1]; k++)
                {
                    for (int j = 0; j < shape[2]; j++)
                    {
                        for (int i = 0; i < shape[3]; i++)
                        {
                            float max = float.MinValue;
                            for (int l = 0; l < shape[0]; l++)
                            {
                                max = MathF.Max(max, tensor[l, k, j, i]);
                            }
                            for (int l = 0; l < result.shape.Batch; l++)
                            {
                                result[l, k, j, i] = max;
                            }

                        }
                    }
                }
            }

            return result;
        }

        #endregion


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
        /// Computes the element-wise minimum
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor Minimum(Tensor left, Tensor right)
        {
            if (!left.shape.Equals(right.shape))
                throw new ArgumentException($"Left{left.Shape} and right{right.Shape} tensors must have different shape for Min operation.");


            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Min(left.data[i], right.data[i]);
            }

            return result;
        }
        /// <summary>
        /// Computes the element-wise maximum.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor Maximum(Tensor left, Tensor right)
        {
            if (!left.shape.Equals(right.shape))
                throw new ArgumentException($"Left{left.Shape} and right{right.Shape} tensors must have different shape for Max operation.");



            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Max(left.data[i], right.data[i]);
            }

            return result;
        }
        public static Tensor Clip(Tensor tensor, float min, float max)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = Math.Clamp(tensor.data[i], min, max);
            }

            return result;
        }
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
        /// Samples random normal distribution elements element-wise.
        /// </summary>
        /// <param name="mu"></param>
        /// <param name="sigma"></param>
        /// <param name="entropies"></param>
        /// <returns></returns>
        public static Tensor RandomGaussian(Tensor mu, Tensor sigma, out Tensor entropies)
        {
            Tensor x = new(mu.shape);
            entropies = new(mu.shape);

            for (int i = 0; i < x.data.Length; i++)
            {
                x.data[i] = Utils.Random.Gaussian(mu.data[i], sigma.data[i], out entropies.data[i]);
            }

            return x;
        }
        public static Tensor LogDensity(Tensor x, Tensor mu, Tensor sigma)
        {
            var frac = (x - mu) / sigma;
            var elem1 = Log(sigma);
            var elem2 = 0.5f * MathF.Log(2.0f * MathF.PI);
            var elem3 = 0.5f * frac * frac;
            return -elem1 - elem2 - elem3;
        }
        public static Tensor Density(Tensor x, Tensor mu, Tensor sigma)
        {
            Tensor p1 = (sigma * MathF.Sqrt(2f * MathF.PI)) * 0.5f;
            Tensor std = (x - mu) / sigma;
            Tensor p2 = -0.5f * std * std;
            return p1 * Tensor.Exp(p2);
        }
        public static Tensor KLDivergence(Tensor mu1, Tensor sig1, Tensor mu2, Tensor sig2)
        {
            var var1 = sig1 * sig1;
            var var2 = sig2 * sig2;

            return Tensor.Log((sig2 / (sig1 + Utils.EPSILON)) + Utils.EPSILON) +
                (var1 + (mu1 - mu2) * (mu1 - mu2)) / (2f * var2) - 0.5f;
        }
        public static bool HasNaN(Tensor tensor)
        {
            for (int i = 0; i < tensor.data.Length; i++)
            {
                if (tensor.data[i] == float.NaN)
                    return true;
            }
            return false;
        }

        #endregion

        public static Tensor Reshape(Tensor tensor, params int[] newShape)
        {
            throw new NotImplementedException();
        }
        public void ForEach(Func<float, float> function, bool multithreaded = false)
        {
            if (!multithreaded)
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = function(data[i]);
                }
            else
                Parallel.For(0, data.Length, DeepUnityMeta.threadLimit8,
                    i => data[i] = function(data[i]));
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
        /// <summary>
        /// Counts the total number of elements in the Tensor [that matches the selector function].
        /// </summary>
        public int Count(Func<float, bool> selector = null)
        {
            if (selector == null)
                return data.Length;

            int count = 0;

            for (int i = 0; i < data.Length; i++)
            {
                count += selector(data[i]) ? 1 : 0;
            }


            return count;
        }
        /// <summary>
        /// Returns a copy of the Tensor data.
        /// </summary>
        public float[] ToArray() => data.ToArray();
        /// <summary>
        /// Creates a Tensor using a copy of the values parsed as argument.
        /// </summary>
        public static Tensor FromArray(float[] data, params int[] shape)
        {
            int prod = 1;
            foreach (var item in shape)
            {
                prod *= item;
            }

            if (prod != data.Length)
                throw new System.FormatException("Tensor shape does not matches the size of the data.");


            var t = Zeros(shape);
            t.data = data.ToArray();
            return t;
        }



        public bool Equals(Tensor other)
        {
            if (!shape.Equals(other.shape))
                return false;

            for (int i = 0; i < data.Length; i++)
                if (!data[i].Equals(other.data[i]))
                    return false;

            return true;
        }
        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
                return false;

            return Equals(obj as Tensor);
        }
        public override string ToString()
        {
            int rank = Rank;

            StringBuilder sb = new();

            sb.Append("Tensor");
            sb.Append(Shape);

            sb.Append("\n[");
            for (int l = 0; l < shape.Batch; l++)
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

                for (int k = 0; k < shape.Channels; k++)
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

                    for (int j = 0; j < shape.Height; j++)
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

                        for (int i = 0; i < shape.Width; i++)
                        {
                            if (i > 0)
                                sb.Append(", ");

                            sb.Append(this[l, k, j, i].ToString(PrintFormat));
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
        private static int GetAxisIndex(int rank, int axis)
        {
            if (axis > rank)
                throw new ArgumentException($"Cannot use axis {axis} for a tensor of rank {rank}.");

            if (rank == 0)
                return 3 + axis;

            int index = 4 - rank + axis;

            if (index < 0)
                throw new ArgumentException($"You cannot call axis {axis} because the limit of Tensor dimensions is 4.");

            return index;
        }

    }
    [Serializable]
    public class TShape
    {
        [SerializeField] private int _batch;
        [SerializeField] private int _channels;
        [SerializeField] private int _height;
        [SerializeField] private int _width;

        public int Batch => _batch;
        public int Channels => _channels;
        public int Height => _height;
        public int Width => _width;

        public int Rank
        {
            get
            {
                if (_batch == 1)
                {
                    if (_channels == 1)
                    {
                        if (_height == 1)
                        {
                            if (_width == 1)
                            {
                                return 0;
                            }
                            else return 1;
                        }
                        else return 2;
                    }
                    else return 3;
                }
                else return 4;
            }
        }
        public TShape(int batch, int channels, int height, int width)
        {
            _batch = batch;
            _channels = channels;
            _height = height;
            _width = width;
        }
        /// <summary>
        /// Returns the full shape in the form int[] { _batch, _channels, _height, _width }.
        /// </summary>
        /// <returns></returns>
        internal int[] ToArray() => new int[] { _batch, _channels, _height, _width };
        public bool Equals(TShape other)
        {
            if (Batch != other.Batch) return false;
            if (Channels != other.Channels) return false;
            if (Height != other.Height) return false;
            if (Width != other.Width) return false;
            return true;
        }

        public override string ToString()
        {

            StringBuilder sb = new();
            sb.Append('[');
            int rank = Rank;
            if (rank == 0 || rank == 1)
            {
                sb.Append(_width);
            }
            else if (rank == 2)
            {
                sb.Append(_height);
                sb.Append(", ");
                sb.Append(_width);
            }
            else if (rank == 3)
            {
                sb.Append(_channels);
                sb.Append(", ");
                sb.Append(_height);
                sb.Append(", ");
                sb.Append(_width);
            }
            else if (rank == 4)
            {
                sb.Append(_batch);
                sb.Append(", ");
                sb.Append(_channels);
                sb.Append(", ");
                sb.Append(_height);
                sb.Append(", ");
                sb.Append(_width);
            }

            sb.Append(']');

            return sb.ToString();
        }
    }

}