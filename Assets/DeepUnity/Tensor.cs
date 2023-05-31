using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    public class Tensor : IEnumerable, ICloneable, IEquatable<Tensor>
    {
        private readonly static int[] numthreads = new int[] { 32, 32, 1 };

        public readonly int[] Shape;
        public readonly int Rank;
        private float[] Data;
        public float this[int x]
        {
            get => Data[x];
            set => Data[x] = value;
        }
        public float this[int x, int y]
        {
            get => Data[y * Shape[0] + x];
            set => Data[y * Shape[0] + x] = value;
        }
        public float this[int x, int y, int z]
        {
            get => Data[z * Shape[1] + y * Shape[0] + x];
            set => Data[z * Shape[1] + y * Shape[0] + x] = value;
        }
        public float this[int x, int y, int z, int w]
        {
            get => Data[w * Shape[2] + z * Shape[1] + y * Shape[0] + x];
            set => Data[w * Shape[2] + z * Shape[1] + y * Shape[0] + x] = value;
        }


        private Tensor(params int[] shape)
        {
            if (shape.Length > 4)
                throw new Exception("Tensor cannot be instantiated with more than 4 dimensions.");

            Shape = new int[] { 1, 1, 1, 1 };
            Rank = 0;

            if (shape.Length > 0)
            {
                Shape[0] = shape[0];
                Rank = 1;
            }
            if (shape.Length > 1)
            {
                Shape[1] = shape[1];
                Rank = 2;
            }
            if (shape.Length > 2)
            {
                Shape[2] = shape[2];
                Rank = 3;
            }
            if (shape.Length > 3)
            {
                Shape[3] = shape[3];
                Rank = 4;
            }
            Data = new float[Shape[0] * Shape[1] * Shape[2] * Shape[3]];
        }
        public static Tensor Constant(float scalar)
        {
            Tensor tensor = new Tensor();
            tensor.Data[0] = scalar;
            return tensor;
        }
        public static Tensor Constant(float[] vector)
        {
            Tensor tensor = new Tensor(vector.GetLength(0));
            var shape = tensor.Shape;
            for (int i = 0; i < shape[0]; i++)
            {
                tensor.Data[i] = vector[i];
            }
            return tensor;
        }
        public static Tensor Constant(float[,] matrix)
        {
            Tensor tensor = new Tensor(matrix.GetLength(0), matrix.GetLength(1));
            var shape = tensor.Shape;
            for (int j = 0; j < shape[1]; j++)
            {
                for (int i = 0; i < shape[0]; i++)
                {
                    
                    tensor[i, j] = matrix[i, j];
                    
                }
            }
            return tensor;
        }
        public static Tensor Constant(float[,,] cuboid)
        {
            Tensor tensor = new Tensor(cuboid.GetLength(0), cuboid.GetLength(1), cuboid.GetLength(2));
            var shape = tensor.Shape;


            for (int k = 0; k < shape[2]; k++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        tensor[i, j, k] = cuboid[i, j, k];
                    }
                }
            }

            return tensor;
        }
        public static Tensor Constant(float[,,,] tesseract)
        {
            Tensor tensor = new Tensor(tesseract.GetLength(0), tesseract.GetLength(1), tesseract.GetLength(2), tesseract.GetLength(3));
            var shape = tensor.Shape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            tensor[i, j, k, l] = tesseract[i, j, k, l];
                        }
                    }
                }
            }

            return tensor;
        }
        public static Tensor Zeros(params int[] shape) => new Tensor(shape);
        public static Tensor Ones(params int[] shape)
        {
            Tensor tensor = new Tensor(shape);

            for (int i = 0; i < tensor.Data.Length; i++)
            {
                tensor.Data[i] = 1;
            }

            return tensor;
        }
        public static Tensor Random(params int[] shape)
        {
            Tensor tensor = new Tensor(shape);

            for (int i = 0; i < tensor.Data.Length; i++)
            {
                tensor.Data[i] = Utils.Random.Value;
            }

            return tensor;
        }
        public static Tensor Normal(params int[] shape)
        {
            Tensor tensor = new Tensor(shape);

            for (int i = 0; i < tensor.Data.Length; i++)
            {
                tensor.Data[i] = Utils.Random.Gaussian(0f, 1f, out _);
            }

            return tensor;
        }
        public static Tensor Fill(float value, params int[] shape)
        {
            Tensor tensor = new Tensor(shape);

            for (int i = 0; i < tensor.Data.Length; i++)
            {
                tensor.Data[i] = value;
            }

            return tensor;
        }


        public static Tensor operator +(Tensor left, float right)
        {
            Tensor result = left.Clone() as Tensor;

            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] += right;
            }

            return result;
        }
        public static Tensor operator -(Tensor left, float right)
        {
            Tensor result = left.Clone() as Tensor;

            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] -= right;
            }

            return result;
        }
        public static Tensor operator *(Tensor left, float right)
        {
            Tensor result = left.Clone() as Tensor;

            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] *= right;
            }

            return result;
        }
        public static Tensor operator /(Tensor left, float right)
        {
            Tensor result = left.Clone() as Tensor;

            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] /= right;
            }

            return result;
        }
        public static Tensor operator +(float left, Tensor right) => right + left;
        public static Tensor operator *(float left, Tensor right) => right * left;
        public static Tensor operator +(Tensor left, Tensor right)
        {
            Tensor result = new Tensor(left.Shape);

            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] = left.Data[i] + right.Data[i];
            }

            return result;
        }
        public static Tensor operator -(Tensor left, Tensor right)
        {
            Tensor result = new Tensor(left.Shape);
            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] = left.Data[i] - right.Data[i];
            }
            return result;
        }
        public static Tensor operator *(Tensor left, Tensor right)
        {
            Tensor result = new Tensor(left.Shape);

            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] = left.Data[i] * right.Data[i];
            }

            return result;
        }
        public static Tensor operator /(Tensor left, Tensor right)
        {
            Tensor result = new Tensor(left.Shape);

            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] = left.Data[i] / right.Data[i];
            }

            return result;
        }

      
        public static Tensor MatMul(Tensor left, Tensor right, ComputeShader MatMulCS = null)
        {
            int w1 = left.Shape[0];
            int h1 = left.Shape[1];
            int w2 = right.Shape[0];
            int h2 = right.Shape[1];
            int batch = left.Shape[2];

            if (h1 != w2)
                throw new ArgumentException("Tensors must have compatible shapes for matrix multiplication (height of left tensor is not matching the width of the right tensor).");

            Tensor resultTensor = new Tensor(w1, h2, batch);


            if (MatMulCS == null)
            {
                System.Threading.Tasks.Parallel.For(0, w1, i =>
                {
                    for (int k = 0; k < batch; k++)
                    {
                        for (int j = 0; j < h2; j++)
                        {
                            float sum = 0.0f;

                            for (int l = 0; l < h1; l++)
                            {
                                sum += left[i, l, k] * right[l, j, k];
                            }

                            lock (resultTensor)
                                resultTensor[i, j, k] = sum;
                        }
                    }
                });
            }
            else
            {
                ComputeBuffer leftBuffer = new ComputeBuffer(left.Data.Length, sizeof(float));
                ComputeBuffer rightBuffer = new ComputeBuffer(right.Data.Length, sizeof(float));
                ComputeBuffer resultBuffer = new ComputeBuffer(w1 * h2 * batch, sizeof(float));
                leftBuffer.SetData(left.Data);
                rightBuffer.SetData(right.Data);


                MatMulCS.SetBuffer(0, "leftArr", leftBuffer);
                MatMulCS.SetBuffer(0, "rightArr", rightBuffer);
                MatMulCS.SetBuffer(0, "resultArr", resultBuffer);
                MatMulCS.SetInt("leftWidth", w1);
                MatMulCS.SetInt("leftHeight", h1);
                MatMulCS.SetInt("rightWidth", w2);
                MatMulCS.SetInt("rightHeight", h2);

                MatMulCS.Dispatch(0, 
                                 (w1 + numthreads[0] - 1) / numthreads[0], 
                                 (h2 + numthreads[1] - 1) / numthreads[1], 
                                 (batch + numthreads[2] - 1) / numthreads[2]);

                // Get result[]
                resultBuffer.GetData(resultTensor.Data);
                
                leftBuffer.Dispose();
                rightBuffer.Dispose();
                resultBuffer.Dispose();
            }

            return resultTensor;
        }
        public static Tensor MatTranspose(Tensor tensor)
        {
            var shape = tensor.Shape;
            Tensor result = new Tensor(shape[1], shape[0], shape[2]);

            
            for (int k = 0; k < shape[2]; k++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        result[j, i] = tensor[i, j];
                    }
                }
            }

            return result;
        }
        public static Tensor[] Slice(Tensor tensor, int axis)
        {
            if (axis < 0 || axis >= tensor.Shape.Length)
                throw new ArgumentException("Invalid axis.");

            int[] shape = tensor.Shape;
            int sliceSize = shape[axis];
            int[] slicedShape = shape.Clone() as int[];
            slicedShape[axis] = 1;

            Tensor[] slices = new Tensor[sliceSize];

            for (int i = 0; i < sliceSize; i++)
            {
                slices[i] = new Tensor(slicedShape);

                int[] indices = Enumerable.Repeat(0, shape.Length).ToArray();
                indices[axis] = i;

                for (int j = 0; j < shape[0]; j++)
                {
                    for (int k = 0; k < shape[1]; k++)
                    {
                        for (int l = 0; l < shape[2]; l++)
                        {
                            for (int m = 0; m < shape[3]; m++)
                            {
                                int[] slicedIndices = new int[] { j, k, l, m };
                                slicedIndices[axis] = 0;
                                slices[i][slicedIndices[0], slicedIndices[1], slicedIndices[2], slicedIndices[3]] = tensor[indices[0], indices[1], indices[2], indices[3]];
                            }
                        }
                    }
                }
            }

            return slices;
        }


        public static Tensor Pow(Tensor @base, float power)
        {
            Tensor result = new Tensor(@base.Shape);

            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] = MathF.Pow(@base.Data[i], power);
            }
            return result;
        }
        public static Tensor Sqrt(Tensor @base)
        {
            Tensor result = new Tensor(@base.Shape);

            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] = MathF.Sqrt(@base.Data[i]);
            }
            return result;
        }
        public static float Mean(Tensor tensor) => tensor.Data.Average();
        public static float StdDev(Tensor tensor) => MathF.Sqrt(Var(tensor));
        public static float Var(Tensor tensor)
        {
            float sum = 0;
            float sumSqr = 0;

            for (int i = 0; i < tensor.Data.Length; i++)
            {
                sum += tensor.Data[i];
                sumSqr += tensor.Data[i] * tensor.Data[i];
            }

            float mean = sum / tensor.Data.Length;
            float meanSquares = sumSqr / tensor.Data.Length;
            return meanSquares - (mean * mean);
        }

        public int Count(Func<float, bool> selector = null)
        {
            if (selector == null)
                return Shape[0] * Shape[1] * Shape[2] * Shape[3];

            int count = 0;

            for (int i = 0; i < Data.Length; i++)
            {
                count += selector(Data[i]) ? 1 : 0;
            }

            
            return count;
        }
        public float[] ToArray() => Data.ToArray();
        public float Sum(Func<float, float> selector = null) 
        {
            if (selector == null)
                return Data.Sum();

            float sum = 0.0f;

            for (int i = 0; i < Data.Length; i++)
            {
                sum += selector(Data[i]);
            }

            return sum;
        }
        public void ForEach(Func<float, float> action)
        {
            for (int i = 0; i < Data.Length; i++)
            {
                Data[i] = action(Data[i]);
            }
        }
        public Tensor Select(Func<float, float> selector)
        {
            Tensor result = new Tensor(Shape);

            for (int i = 0; i < Data.Length; i++)
            {
                result.Data[i] = selector(Data[i]);
            }

            return result;
        }
        public Tensor Zip(Tensor second, Func<float, float, float> resultSelector)
        {
            Tensor result = new Tensor(Shape);

            for (int i = 0; i < Data.Length; i++)
            {
                result.Data[i] = resultSelector(Data[i], second.Data[i]);
            }

            return result;
        }
        

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
        public IEnumerator<float> GetEnumerator()
        {
            for (int i = 0; i < Data.Length; i++)
            {
                yield return Data[i];
            }
        }
        public bool Equals(Tensor other)
        {
            if (!Shape.SequenceEqual(other.Shape))
                return false;

            for (int i = 0; i < Data.Length; i++)
                if (Data[i].Equals(other.Data[i]))
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
            StringBuilder sb = new StringBuilder();
            sb.Append("[");

            int[] shape = Shape;
            int rank = Rank;

            for (int l = 0; l < shape[3]; l++)
            {
                if (l > 0)
                    sb.Append(",\n\n\n");

                if (rank > 3)
                    sb.Append("[");

                for (int k = 0; k < shape[2]; k++)
                {
                    if (k > 0)
                        sb.Append(",\n\n");

                    if (rank > 2)
                        sb.Append("[");

                    for (int j = 0; j < shape[1]; j++)
                    {
                        if (j > 0 && rank > 1)
                            sb.Append(",\n");

                        if (rank > 1)
                            sb.Append("[");

                        for (int i = 0; i < shape[0]; i++)
                        {
                            if (i > 0)
                                sb.Append(", ");

                            sb.Append(this[i, j, k, l].ToString());
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
        public object Clone()
        {
            var clone = new Tensor(Shape);
            Array.Copy(Data, clone.Data, Data.Length);
            return clone;
        }

    }
}


// Matmul benchmark on generic Tensor<T>
/*
 * 
 * Matmul benchmark on base Tensor (float type)
 * 
 * (1024, 1024) * (1024, 1024)  X  1 Times
 * CPU: 03.28s | GPU: 00.12s
 * 
 * (32, 32) * (32, 32)          X  1K Times
 * CPU: 01.05s | GPU: 00.53s   
 * 
 * (8, 8) * (8, 8)              X  10K Times
 * CPU: 00.38s | GPU: 02.79
 * 
 * (4096, 4096) * (4096, 4096)  X 100 Times (uses 50-95% of GPU)
 * GPU: 01:01.63 (above 1 minute)
 * Conclusion: On SMALL tensors the CPU is faster. 
 *             On MEDIUM and LARGE tensor the GPU is faster.
 *///