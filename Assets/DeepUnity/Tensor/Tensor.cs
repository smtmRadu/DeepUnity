using System;
using System.Linq;
using System.Text;
using UnityEngine;
using System.Threading.Tasks;
using System.Collections.Generic;
using kbRadu;

namespace DeepUnity
{
    [Serializable]
    public class Tensor : IEquatable<Tensor>
    {
        [SerializeField] private TShape shape;
        [SerializeField] private float[] data;
           
        public int Rank
        {
            get
            {
                if (shape.ndim == 1)
                {
                    if (shape.batch == 1)
                    {
                        if (shape.height == 1)
                        {
                            if (shape.width == 1)
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
        public TShape Shape => shape;
        public string ShapeToString
        {
            get
            {
                StringBuilder sb = new StringBuilder();
                sb.Append('[');
                int rank = Rank;
                if (rank == 0 || rank == 1)
                {
                    sb.Append(shape.width);
                }
                else if (rank == 2)
                {
                    sb.Append(shape.height);
                    sb.Append(", ");
                    sb.Append(shape.width);
                }
                else if (rank == 3)
                {
                    sb.Append(shape.batch);
                    sb.Append(", ");
                    sb.Append(shape.height);
                    sb.Append(", ");
                    sb.Append(shape.width);
                }
                else if (rank == 4)
                {
                    sb.Append(shape.ndim);
                    sb.Append(", ");
                    sb.Append(shape.batch);
                    sb.Append(", ");
                    sb.Append(shape.height);
                    sb.Append(", ");
                    sb.Append(shape.width);
                }

                sb.Append(']');

                return sb.ToString();
            }
        }
        public override string ToString()
        {
            int rank = Rank;
            string format = "0.000000";

            StringBuilder sb = new StringBuilder();

            sb.Append("Tensor");
            sb.Append(ShapeToString);

            sb.Append("\n[");
            for (int l = 0; l < shape.ndim; l++)
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

                for (int k = 0; k < shape.batch; k++)
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

                    for (int j = 0; j < shape.height; j++)
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

                        for (int i = 0; i < shape.width; i++)
                        {
                            if (i > 0)
                                sb.Append(", ");

                            sb.Append(this[l, k, j, i].ToString(format));
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
        public float this[int width]
        {
            get => data[width];
            set => data[width] = value;
        }
        public float this[int height, int width]
        {
            get => data[width * shape.height + height];
            set => data[width * shape.height + height] = value;
        }
        public float this[int batch, int height, int width]
        {
            get => data[batch * shape.height * shape.width + width * shape.height + height];
            set => data[batch * shape.height * shape.width + width * shape.height + height] = value;
        }
        public float this[int ndim, int batch, int height, int width]
        {
            get => data[ndim * shape.batch * shape.height * shape.width + batch * shape.height * shape.width + width * shape.height + height];
            set => data[ndim * shape.batch * shape.height * shape.width + batch * shape.height * shape.width + width * shape.height + height] = value;

        }


        private Tensor(params int[] shortShape)
        {
            if (shortShape == null || shortShape.Length == 0)
                throw new Exception("Tensor cannot be instantiated with null shape.");
            if (shortShape.Length > 4)
                throw new Exception("Tensor cannot be instantiated with more than 4 dimensions.");
            
            int width = 1;
            int height = 1;
            int batch = 1;
            int ndim = 1;
            if(shortShape.Length == 1)
            {
                width = shortShape[0];
            }
            else if(shortShape.Length == 2)
            {
                width = shortShape[1];
                height = shortShape[0];
            }
            else if (shortShape.Length == 3)
            {
                width = shortShape[2];
                height = shortShape[1];
                batch = shortShape[0];
            }
            else if (shortShape.Length == 4)
            {
                width = shortShape[3];
                height = shortShape[2];
                batch = shortShape[1];
                ndim = shortShape[0];
            }

            int size = ndim * batch * height * width;

            if (size > 16_777_216) // hardcoded like this because 4096x4096 max allowed matrix, on 8192 it crashes
                throw new Exception("Tensor dimensions is too large on initialization (cannot surpass 16,777,216 units).");

            shape = new TShape(ndim, batch, height, width);
            data = new float[size];
        }
        private Tensor(TShape tshape)
        {
            this.shape = new TShape(tshape.ndim, tshape.batch, tshape.height, tshape.width);
            
            int size = tshape.ndim * tshape.batch * tshape.height * tshape.width;

            if (size > 16_777_216) // hardcoded like this because 4096x4096 max allowed matrix, on 8192 it crashes
                throw new Exception("Tensor dimensions is too large on initialization (cannot surpass 16,777,216 units).");

            data = new float[size];
        }
        public static Tensor Identity(Tensor other)
        {
            Tensor clone = new Tensor(other.shape);
            Array.Copy(other.data, clone.data, other.data.Length);
            return clone;
        }
        public static Tensor Constant(float scalar)
        {
            Tensor t = new Tensor(1);
            t.data[0] = scalar;
            return t;
        }
        public static Tensor Constant(float[] vector)
        {
            Tensor t = new Tensor(vector.GetLength(0));
            t.data = vector.ToArray();
            return t;
        }
        public static Tensor Constant(float[,] matrix)
        {
            Tensor t = new Tensor(matrix.GetLength(0), matrix.GetLength(1));
            t.data = matrix.Cast<float>().ToArray();
            return t;
        }
        public static Tensor Constant(float[,,] cube)
        {
            Tensor t = new Tensor(cube.GetLength(0), cube.GetLength(1), cube.GetLength(2));
            t.data = cube.Cast<float>().ToArray();
            return t;
        }
        public static Tensor Constant(float[,,,] tesseract)
        {
            Tensor t = new Tensor(tesseract.GetLength(0), tesseract.GetLength(1), tesseract.GetLength(2), tesseract.GetLength(3));
            t.data = tesseract.Cast<float>().ToArray();
            return t;
        }
        public static Tensor Zeros(params int[] shape)
        {
            return new Tensor(shape);
        }
        public static Tensor Ones(params int[] shape)
        {
            Tensor t = new Tensor(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = 1f;
            }
            return t;
        }
        public static Tensor Random01(params int[] shape)
        {
            Tensor t = new Tensor(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = Utils.Random.Value;
            }
            return t;
        }
        public static Tensor RandomNormal(params int[] shape)
        {
            Tensor t = new Tensor(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = Utils.Random.Gaussian();
            }
            return t;
        }
        public static Tensor RandomRange(float min, float max, params int[] shape)
        {
            Tensor t = new Tensor(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = Utils.Random.Range(min, max);
            }
            return t;
        }
        public static Tensor Fill(float value, params int[] shape)
        {
            Tensor t = new Tensor(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = value;
            }
            return t;
        }

        public static Tensor operator +(Tensor tensor)
        {
            Tensor result = new Tensor(tensor.shape);
            for (int i = 0; i < tensor.data.Length; i++)
            {
                result.data[i] = tensor.data[i];
            }

            return result;
        }
        public static Tensor operator -(Tensor tensor)
        {
            Tensor result = new Tensor(tensor.shape);
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
                throw new Exception($"Left{left.ShapeToString} and right{right.ShapeToString} tensors must have different shape for Element-wise addition (+).");

            Tensor result = new Tensor(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] + right.data[i];
            }


            return result;
        }
        public static Tensor operator -(Tensor left, Tensor right)
        {
            if (!left.shape.Equals(right.shape))
                throw new Exception($"Left{left.ShapeToString} and right{right.ShapeToString} tensors must have different shape for Element-wise subtraction (-).");

            Tensor result = new Tensor(left.shape);
            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] - right.data[i];
            }

            return result;
        }
        public static Tensor operator *(Tensor left, Tensor right)
        {
            if (!left.shape.Equals(right.shape))
                throw new Exception($"Left{left.ShapeToString} and right{right.ShapeToString} tensors must have different shape for Element-wise multiplication (*).");

            Tensor result = new Tensor(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] * right.data[i];
            }


            return result;
        }
        public static Tensor operator /(Tensor left, Tensor right)
        {
            if (!left.shape.Equals(right.shape))
                throw new Exception($"Left{left.ShapeToString} and right{right.ShapeToString} tensors must have different shape for Element-wise division (/).");

            Tensor result = new Tensor(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] / right.data[i];
            }

            return result;
        }


        /// <summary>
        /// left (k, n, m) * right (k, m, p) = result (k, n, p)
        /// </summary>
        public static Tensor MatMul(Tensor left, Tensor right)
        {
            /* N x M dot M x P => N x P
             */
            int w1 = left.shape.height;
            int h1 = left.shape.width;
            int w2 = right.shape.height;
            int h2 = right.shape.width;
            int b1 = left.shape.batch;
            int b2 = right.shape.batch;

            if (h1 != w2)
                throw new ArgumentException("Tensor must have compatible shapes for matrix multiplication (height of left ndarray is not matching the width of the right ndarray).");

            if (b1 != b2)
                throw new ArgumentException("Tensors must have similar number of batches for batched matrix multiplication.");
            
            Tensor result = new Tensor(b1, w1, h2);

            if (Settings.Device == Device.CPU)
            {
                if(b1 == 1)
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
                ComputeShader CS = Settings.MatMulCS;

                ComputeBuffer leftBuffer = new ComputeBuffer(left.data.Length, 4);
                ComputeBuffer rightBuffer = new ComputeBuffer(right.data.Length, 4);
                ComputeBuffer resultBuffer = new ComputeBuffer(b1 * h2 * w1, 4);

                leftBuffer.SetData(left.data);
                rightBuffer.SetData(right.data);

                
                CS.SetBuffer(0, "leftArr", leftBuffer);
                CS.SetBuffer(0, "rightArr", rightBuffer);
                CS.SetBuffer(0, "resultArr", resultBuffer);
                CS.SetInt("w1", w1);
                CS.SetInt("h1w2", h1);
                CS.SetInt("h2", h2);

                CS.Dispatch(0,
                           (w1 + Settings.numthreads[0] - 1) / Settings.numthreads[0],
                           (h2 + Settings.numthreads[1] - 1) / Settings.numthreads[1],
                           (b1 + Settings.numthreads[2] - 1) / Settings.numthreads[2]);

                resultBuffer.GetData(result.data);

                leftBuffer.Dispose();
                rightBuffer.Dispose();
                resultBuffer.Dispose();
            }

            return result;
        }
        public static Tensor MatPad(Tensor tensor, int padding, PaddingType paddingMode)
        {
            if (padding == 0)
                return tensor;

            int w = tensor.shape.width + 2;
            int h = tensor.shape.height + 2;
            int b = tensor.shape.batch;
            int n = tensor.shape.ndim;
            Tensor result = new Tensor(n, b, h, w);


            for (int k = 0; k < tensor.shape.batch; k++)
            {
                for (int j = 0; j < tensor.shape.height; j++)
                {
                    for (int i = 0; i < tensor.shape.width; i++)
                    {
                        result[k, j + 1, i + 1] = tensor[k, j, i];
                    }
                }
            }

            if(paddingMode == PaddingType.Mirror)
            {
                result[0, 0] = result[1, 1];
                result[h - 1, 0] = result[h - 2, 1];
                result[0, w - 1] = result[1, w - 2];
                result[h - 1, w - 1] = result[h - 2, w - 2];

                for (int i = 0; i < w - 1; i++)
                {
                    result[0, i] = result[1, i];
                    result[h - 1, i] = result[h - 2, i];
                }

                for (int j = 0; j < h - 1; j++)
                {
                    result[j, 0] = result[j, 1];
                    result[j, w - 1] = result[j, w - 2];
                }
            }


            return MatPad(result, padding - 1, paddingMode);
        }
        public static Tensor MatTranspose(Tensor tensor)
        {
            int w = tensor.shape.width;
            int h = tensor.shape.height;
            int b = tensor.shape.batch;

            Tensor result = new Tensor(b, w, h);

            for (int k = 0; k < b; k++)
            {
                for (int j = 0; j < h; j++)
                {
                    for (int i = 0; i < w; i++)
                    {
                        result[k, i, j] = tensor[k, j, i];
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// [Deprecated] 
        /// Splits axis in split_size batches.
        /// If Axis is not a multiple of split_size, the last batch will remain incompletely.
        /// </summary>-
        public static Tensor[] Split(Tensor tensor, int axis, int split_size)
        {
            int rank = tensor.Rank;
            int axisIndex = GetAxisIndex(rank, axis);
            int[] stackShape = tensor.shape.ToArray();
            List<Tensor> slices = new List<Tensor>();

            int dimLength = stackShape[axisIndex];
            int dimPos = 0;
            while(dimPos < dimLength)
            {
                int dimCopySize = Math.Min(split_size, dimLength - dimPos);
                int[] sliceShape = stackShape.ToArray();
                sliceShape[axisIndex] = dimCopySize;
                Tensor slice = new Tensor(sliceShape);

                for (int l = 0; l < slice.shape.ndim; l++)
                {
                    for (int k = 0; k < slice.shape.batch; k++)
                    {
                        for (int j = 0; j < slice.shape.height; j++)
                        {
                            for (int i = 0; i < slice.shape.width; i++)
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
        /// <summary>
        /// [Deprecated]
        /// Concatenates all tensors same dimension.
        /// Tensors dimension N is N * no_slices in the result tensor.
        /// On axis - 1, the tensors are batched.
        /// </summary>
        public static Tensor Join(int axis, params Tensor[] tensors)
        {
            if (tensors == null || tensors.Length == 0)
                throw new Exception("Tensor used for joining are not defined.");

            int rank = tensors[0].Rank;
            int axisIndex = GetAxisIndex(rank, axis);
            int no_slices = tensors.Length;

            

            Tensor slice = tensors[0];

            int[] result_shape = slice.shape.ToArray();
            result_shape[axisIndex] *= no_slices;

            Tensor result = new Tensor(result_shape);

            for (int s = 0; s < no_slices; s++)
            {
                for (int l = 0; l < slice.shape.ndim; l++)
                {
                    for (int k = 0; k < slice.shape.batch; k++)
                    {
                        for (int j = 0; j < slice.shape.height; j++)
                        {
                            for (int i = 0; i < slice.shape.width; i++)
                            {
                                switch (axisIndex)
                                {
                                    case 3:
                                        result[l, k, j, s * slice.shape.width + i] = tensors[s][l, k, j, i];
                                        break;
                                    case 2:
                                        result[l, k, s * slice.shape.height + j, i] = tensors[s][l, k, j, i];
                                        break;
                                    case 1:
                                        result[l, s * slice.shape.batch + k, j, i] = tensors[s][l, k, j, i];
                                        break;
                                    case 0:
                                        result[s * slice.shape.ndim + l, k, j, i] = tensors[s][l, k, j, i];
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
        /// [Deprecated]
        /// Expands a dimension N of value X to value X * times. 
        /// On axis -1, the tensor is duplicated in times batches.
        /// </summary>
        /// <returns></returns>
        public static Tensor Expand(Tensor tensor, int axis, int times)
        {
            int rank = tensor.Rank;
            int axisIndex = GetAxisIndex(rank, axis);
            int[] shape = tensor.shape.ToArray();
            shape[axisIndex] *= times;

            Tensor result = new Tensor(shape);

            for (int t = 0; t < times; t++)
            {
                for (int l = 0; l < tensor.shape.ndim; l++)
                {
                    for (int k = 0; k < tensor.shape.batch; k++)
                    {
                        for (int j = 0; j < tensor.shape.height; j++)
                        {
                            for (int i = 0; i < tensor.shape.width; i++)
                            {
                                switch(axisIndex)
                                {
                                    case 3:
                                        result[l, k, j, t * tensor.shape.width + i] = tensor[l, k, j, i];
                                        break;
                                    case 2:
                                        result[l, k, t * tensor.shape.height + j, i] = tensor[l, k, j, i];
                                        break;
                                    case 1:
                                        result[l, t * tensor.shape.batch + k, j, i] = tensor[l, k, j, i];
                                        break;
                                    case 0:
                                        result[t * tensor.shape.ndim + l, k, j, i] = tensor[l, k, j, i];
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
        /// [Deprecated]
        /// </summary>
        public static Tensor Shuffle(Tensor tensor, int axis)
        {
            Tensor[] slices = Split(tensor, axis, 1);
            slices = Utils.Shuffle(slices).ToArray();
            return Join(axis, slices);
        }
        /// <summary>
        /// [Deprecated]
        /// </summary>
        public static Tensor Sum(Tensor tensor, int axis)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();
            int axisIndex = GetAxisIndex(tensor.Rank, axis);

            if(axisIndex == 3)
            {
                result = new Tensor(shape[0], shape[1], shape[2], 1);
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
            else if(axisIndex == 2)
            {
                result = new Tensor(shape[0], shape[1], 1, shape[3]);
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
            else if(axisIndex == 1)
            {
                result = new Tensor(shape[0], 1, shape[2], shape[3]);
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
            else if(axisIndex == 0)
            {
                result = new Tensor(1, shape[1], shape[2], shape[3]);
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
        /// <summary>
        /// [Deprecated]
        /// </summary>
        public static Tensor Mean(Tensor tensor, int axis)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();
            int axisIndex = GetAxisIndex(tensor.Rank, axis);

            if (axisIndex == 3)
            {
                result = new Tensor(shape[0], shape[1], shape[2], 1);
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
                result = new Tensor(shape[0], shape[1], 1, shape[3]);
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
                result = new Tensor(shape[0], 1, shape[2], shape[3]);
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
                result = new Tensor(1, shape[1], shape[2], shape[3]);
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
        /// <summary>
        /// [Deprecated]
        /// </summary>
        public static Tensor Var(Tensor tensor, int axis, int correction = 1)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();
            int axisIndex = GetAxisIndex(tensor.Rank, axis);

            if (axisIndex == 3)
            {
                result = new Tensor(shape[0], shape[1], shape[2], 1);
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
                result = new Tensor(shape[0], shape[1], 1, shape[3]);
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
                result = new Tensor(shape[0], 1, shape[2], shape[3]);
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
                result = new Tensor(1, shape[1], shape[2], shape[3]);
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
        /// <summary>
        /// [Deprecated]
        /// </summary>
        public static Tensor Std(Tensor tensor, int axis, int correction = 1)
        {
            return Sqrt(Var(tensor, axis, correction));
        }

        // On Dimension
        public static Tensor Var(Tensor tensor, TDim dim, int correction = 1, bool keepDim = false)
        {
            Tensor result = null;
            int[] shape = tensor.shape.ToArray();

            if(!keepDim)
            {
                if (dim == TDim.width)
                {
                    result = new Tensor(shape[0], shape[1], shape[2], 1);
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
                    result = new Tensor(shape[0], shape[1], 1, shape[3]);
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
                else if (dim == TDim.batch)
                {
                    result = new Tensor(shape[0], 1, shape[2], shape[3]);
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
                else if (dim == TDim.ndim)
                {
                    result = new Tensor(1, shape[1], shape[2], shape[3]);
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
                result = new Tensor(tensor.shape);
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
                else if (dim == TDim.batch)
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
                else if (dim == TDim.ndim)
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

            if(!keepDim)
            {
                if (dim == TDim.width)
                {
                    result = new Tensor(shape[0], shape[1], shape[2], 1);
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
                    result = new Tensor(shape[0], shape[1], 1, shape[3]);
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
                else if (dim == TDim.batch)
                {
                    result = new Tensor(shape[0], 1, shape[2], shape[3]);
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
                else if (dim == TDim.ndim)
                {
                    result = new Tensor(1, shape[1], shape[2], shape[3]);
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
                result = new Tensor(tensor.shape);
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
                else if (dim == TDim.batch)
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
                else if (dim == TDim.ndim)
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

            if(!keepDim)
            {
                if (dim == TDim.width)
                {
                    result = new Tensor(shape[0], shape[1], shape[2], 1);
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
                    result = new Tensor(shape[0], shape[1], 1, shape[3]);
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
                else if (dim == TDim.batch)
                {
                    result = new Tensor(shape[0], 1, shape[2], shape[3]);
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
                else if (dim == TDim.ndim)
                {
                    result = new Tensor(1, shape[1], shape[2], shape[3]);
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
                result = new Tensor(tensor.shape);
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
                else if (dim == TDim.batch)
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
                else if (dim == TDim.ndim)
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
            TShape shape = null;
            switch(dim)
            {
                case TDim.width:
                    shape = new TShape(tensor.shape.ndim, tensor.shape.batch, tensor.shape.height, times);
                    break;
                case TDim.height:
                    shape = new TShape(tensor.shape.ndim, tensor.shape.batch, times, tensor.shape.width);
                    break;
                case TDim.batch:
                    shape = new TShape(tensor.shape.ndim, times, tensor.shape.height, tensor.shape.width);
                    break;
                case TDim.ndim:
                    shape = new TShape(times, tensor.shape.batch, tensor.shape.height, tensor.shape.width);
                    break;                

            }
            Tensor result = new Tensor(shape);

            for (int t = 0; t < times; t++)
            {
                for (int l = 0; l < tensor.shape.ndim; l++)
                {
                    for (int k = 0; k < tensor.shape.batch; k++)
                    {
                        for (int j = 0; j < tensor.shape.height; j++)
                        {
                            for (int i = 0; i < tensor.shape.width; i++)
                            {
                                switch (dim)
                                {
                                    case TDim.width:
                                        result[l, k, j, t * tensor.shape.width + i] = tensor[l, k, j, i];
                                        break;
                                    case TDim.height:
                                        result[l, k, t * tensor.shape.height + j, i] = tensor[l, k, j, i];
                                        break;
                                    case TDim.batch:
                                        result[l, t * tensor.shape.batch + k, j, i] = tensor[l, k, j, i];
                                        break;
                                    case TDim.ndim:
                                        result[t * tensor.shape.ndim + l, k, j, i] = tensor[l, k, j, i];
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
                throw new Exception("Tensor used for joining are not defined.");

            int no_slices = tensors.Length;


            Tensor slice = tensors[0];
            TShape shape = null;
            switch (dim)
            {
                case TDim.width:
                    shape = new TShape(slice.shape.ndim, slice.shape.batch, slice.shape.height, slice.shape.width * no_slices);
                    break;
                case TDim.height:
                    shape = new TShape(slice.shape.ndim, slice.shape.batch, slice.shape.height * no_slices, slice.shape.width);
                    break;
                case TDim.batch:
                    shape = new TShape(slice.shape.ndim, slice.shape.batch * no_slices, slice.shape.height, slice.shape.width);
                    break;
                case TDim.ndim:
                    shape = new TShape(slice.shape.ndim * no_slices, slice.shape.batch, slice.shape.height, slice.shape.width);
                    break;
            }
            Tensor result = new Tensor(shape);

            for (int s = 0; s < no_slices; s++)
            {
                for (int l = 0; l < slice.shape.ndim; l++)
                {
                    for (int k = 0; k < slice.shape.batch; k++)
                    {
                        for (int j = 0; j < slice.shape.height; j++)
                        {
                            for (int i = 0; i < slice.shape.width; i++)
                            {
                                switch (dim)
                                {
                                    case TDim.width:
                                        result[l, k, j, s * slice.shape.width + i] = tensors[s][l, k, j, i];
                                        break;
                                    case TDim.height:
                                        result[l, k, s * slice.shape.height + j, i] = tensors[s][l, k, j, i];
                                        break;
                                    case TDim.batch:
                                        result[l, s * slice.shape.batch + k, j, i] = tensors[s][l, k, j, i];
                                        break;
                                    case TDim.ndim:
                                        result[s * slice.shape.ndim + l, k, j, i] = tensors[s][l, k, j, i];
                                        break;
                                }

                            }
                        }
                    }
                }
            }


            return result;
        }
        public static Tensor[] Split(Tensor tensor, TDim dim, int split_size)
        {
            List<Tensor> slices = new List<Tensor>();

            int dimLength = tensor.shape.Get(dim);
            int dimPos = 0;
            while (dimPos < dimLength)
            {
                int dimCopySize = Math.Min(split_size, dimLength - dimPos);

                TShape shape = null;
                switch (dim)
                {
                    case TDim.width:
                        shape = new TShape(tensor.shape.ndim, tensor.shape.batch, tensor.shape.height, dimCopySize);
                        break;
                    case TDim.height:
                        shape = new TShape(tensor.shape.ndim, tensor.shape.batch, dimCopySize, tensor.shape.width);
                        break;
                    case TDim.batch:
                        shape = new TShape(tensor.shape.ndim, dimCopySize, tensor.shape.height, tensor.shape.width);
                        break;
                    case TDim.ndim:
                        shape = new TShape(dimCopySize, tensor.shape.batch, tensor.shape.height, tensor.shape.width);
                        break;

                }
                Tensor slice = new Tensor(shape);

                for (int l = 0; l < slice.shape.ndim; l++)
                {
                    for (int k = 0; k < slice.shape.batch; k++)
                    {
                        for (int j = 0; j < slice.shape.height; j++)
                        {
                            for (int i = 0; i < slice.shape.width; i++)
                            {
                                switch (dim)
                                {
                                    case TDim.width:
                                        slice[l, k, j, i] = tensor[l, k, j, dimPos + i];
                                        break;
                                    case TDim.height:
                                        slice[l, k, j, i] = tensor[l, k, j + dimPos, i];
                                        break;
                                    case TDim.batch:
                                        slice[l, k, j, i] = tensor[l, k + dimPos, j, i];
                                        break;
                                    case TDim.ndim:
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


        // Basic Math
        public static Tensor Pow(Tensor tensor, float power)
        {
            Tensor result = new Tensor(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Pow(tensor.data[i], power);
            }

            return result;
        }
        public static Tensor Sqrt(Tensor tensor)
        {
            Tensor result = new Tensor(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Sqrt(tensor.data[i]);
            }

            return result;
        }
        public static Tensor Exp(Tensor tensor)
        {
            Tensor result = new Tensor(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Exp(tensor.data[i]);
            }

            return result;
        }
        public static Tensor Abs(Tensor tensor)
        {
            Tensor result = new Tensor(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Abs(tensor.data[i]);
            }

            return result;
        }
        public static Tensor Sin(Tensor tensor)
        {
            Tensor result = new Tensor(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Sin(tensor.data[i]);
            }

            return result;
        }
        public static Tensor Cos(Tensor tensor)
        {
            Tensor result = new Tensor(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Cos(tensor.data[i]);
            }

            return result;
        }
        public static Tensor Min(Tensor left, Tensor right)
        {
            if (!left.shape.Equals(right.shape))
                throw new Exception($"Left{left.ShapeToString} and right{right.ShapeToString} tensors must have different shape for Min operation.");


            Tensor result = new Tensor(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Min(left.data[i], right.data[i]);
            }

            return result;
        }
        public static Tensor Max(Tensor left, Tensor right)
        {
            if (!left.shape.Equals(right.shape))
                throw new Exception($"Left{left.ShapeToString} and right{right.ShapeToString} tensors must have different shape for Max operation.");



            Tensor result = new Tensor(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Max(left.data[i], right.data[i]);
            }

            return result;
        }
        public static Tensor Clip(Tensor tensor, float min, float max)
        {
            Tensor result = new Tensor(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = Math.Clamp(tensor.data[i], min, max);
            }

            return result;
        }
        public static Tensor Norm(Tensor tensor, NormType normType = NormType.ManhattanL1)
        {
            switch(normType)
            {
                case NormType.ManhattanL1:
                    float abssum = tensor.data.Sum(x => MathF.Abs(x));
                    return Constant(abssum);
                case NormType.EuclideanL2:
                    float sum = tensor.data.Sum();
                    return Constant(MathF.Sqrt(sum));
                case NormType.Frobenius:
                    float sqrsum = tensor.data.Sum(x => x * x);
                    return Constant(MathF.Sqrt(sqrsum));
                default:
                    throw new Exception("Unhandled norm type.");
            }
        }
        /// <summary>
        /// Returns the minimum value in the array.
        /// </summary>
        /// <returns>A tensor with shape [1], or a tensor with the same shape if keepShape is true.</returns>
        public static Tensor MinValue(Tensor tensor, bool keepShape = false)
        {
            if (!keepShape)
                return Constant(tensor.data.Min());
            else return Fill(tensor.data.Min(), tensor.shape.ToArray());
        }
        /// <summary>
        /// Returns the maximum value in the array.
        /// </summary>
        /// <returns>A tensor with shape [1], or a tensor with the same shape if keepShape is true.</returns>
        public static Tensor MaxValue(Tensor tensor, bool keepShape = false)
        {
            if (!keepShape)
                return Constant(tensor.data.Max());
            else return Fill(tensor.data.Max(), tensor.shape.ToArray());
        }

        public void ForEach(Func<float, float> function)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = function(data[i]);
            }
        }
        public Tensor Select(Func<float, float> selector)
        {
            Tensor result = new Tensor(shape);

            for (int i = 0; i < data.Length; i++)
            {
                result.data[i] = selector(data[i]);
            }

            return result;
        }
        public Tensor Zip(Tensor second, Func<float, float, float> resultSelector)
        {
            Tensor result = new Tensor(shape);

            for (int i = 0; i < data.Length; i++)
            {
                result.data[i] = resultSelector(data[i], second.data[i]);
            }

            return result;
        }
        public Tensor Count(Func<float, bool> selector = null)
        {
            if (selector == null)
                return Constant(data.Length);

            int count = 0;

            for (int i = 0; i < data.Length; i++)
            {
                count += selector(data[i]) ? 1 : 0;
            }


            return Constant(count);
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
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        private static int GetAxisIndex(int rank, int axis)
        {
            if (axis > rank)
                throw new Exception($"Cannot use axis {axis} for a tensor of rank {rank}.");
           
            if (rank == 0)
                return 3 + axis;

            int index = 4 - rank + axis;

            if (index < 0)
                throw new Exception($"You cannot call axis {axis} because the limit of Tensor dimensions is 4.");

            return index;
        }
        

    }
    [Serializable]
    public class TShape
    {
        [SerializeField] private int _ndim;
        [SerializeField] private int _batch;
        [SerializeField] private int _height;
        [SerializeField] private int _width;

        public int ndim => _ndim;
        public int batch => _batch;
        public int height => _height;
        public int width => _width;


        public TShape(int ndim, int batch, int height, int width)
        {
            _ndim = ndim;
            _batch = batch;
            _height = height;
            _width = width;
        }
        internal int[] ToArray() => new int[] { _ndim, _batch, _height, _width };
        public bool Equals(TShape other)
        {
            if (ndim != other.ndim) return false;
            if (batch != other.batch) return false;
            if (height != other.height) return false;
            if (width != other.width) return false;
            return true;
        }
        public int Get(TDim dim)
        {
            switch (dim)
            {
                case TDim.width: return _width;
                case TDim.height: return _height;
                case TDim.batch: return _batch;
                case TDim.ndim: return _ndim;
                default: throw new Exception("Unhandled dim type");
            }
        }
    }
    public enum TDim
    {
        ndim,
        batch,
        height,
        width
    }
}

