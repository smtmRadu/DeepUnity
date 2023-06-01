using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Tensor : IEnumerable, ICloneable, IEquatable<Tensor>
    {
        private readonly static int[] numthreads = new int[] { 32, 32, 1 };
        [SerializeField] private int[] shape; 
        [SerializeField] private float[] data;
       
        // Create
        private Tensor(params int[] _shape)
        {
            if (_shape.Length > 4)
                throw new Exception("Tensor cannot be instantiated with more than 4 dimensions.");

            this.shape = new int[] { 1, 1, 1, 1 };

            if (_shape.Length > 0)
            {
                this.shape[0] = _shape[0];
            }
            if (_shape.Length > 1)
            {
                this.shape[1] = _shape[1];
            }
            if (_shape.Length > 2)
            {
                this.shape[2] = _shape[2];
            }
            if (_shape.Length > 3)
            {
                this.shape[3] = _shape[3];
            }
            data = new float[this.shape[0] * this.shape[1] * this.shape[2] * this.shape[3]];
        }
        public static Tensor Constant(float scalar)
        {
            Tensor tensor = new Tensor();
            tensor.data[0] = scalar;
            return tensor;
        }
        public static Tensor Constant(float[] vector)
        {
            Tensor tensor = new Tensor(vector.GetLength(0));
            var shape = tensor.shape;
            for (int i = 0; i < shape[0]; i++)
            {
                tensor.data[i] = vector[i];
            }
            return tensor;
        }
        public static Tensor Constant(float[,] matrix)
        {
            Tensor tensor = new Tensor(matrix.GetLength(0), matrix.GetLength(1));
            var shape = tensor.shape;
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
            var shape = tensor.shape;


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
            var shape = tensor.shape;

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

            for (int i = 0; i < tensor.data.Length; i++)
            {
                tensor.data[i] = 1;
            }

            return tensor;
        }
        public static Tensor Random(params int[] shape)
        {
            Tensor tensor = new Tensor(shape);

            for (int i = 0; i < tensor.data.Length; i++)
            {
                tensor.data[i] = Utils.Random.Value;
            }

            return tensor;
        }
        public static Tensor Normal(params int[] shape)
        {
            Tensor tensor = new Tensor(shape);

            for (int i = 0; i < tensor.data.Length; i++)
            {
                tensor.data[i] = Utils.Random.Gaussian(0f, 1f, out _);
            }

            return tensor;
        }
        public static Tensor Fill(float value, params int[] shape)
        {
            Tensor tensor = new Tensor(shape);

            for (int i = 0; i < tensor.data.Length; i++)
            {
                tensor.data[i] = value;
            }

            return tensor;
        }

        // Operator overloading
        public int Rank
        {
            get
            {
                for (int i = shape.Length - 1; i >= 0; i--)
                {
                    if (shape[i] > 1)
                        return i + 1;
                }
                return 0;
            }
        }
        public int[] Shape
        {
            get => shape;
        }
        public float this[int x]
        {
            get => data[x];
            set => data[x] = value;
        }
        public float this[int x, int y]
        {
            get => data[y * shape[0] + x];
            set => data[y * shape[0] + x] = value;
        }
        public float this[int x, int y, int z]
        {
            get => data[z * shape[1] * shape[0] + y * shape[0] + x];
            set => data[z * shape[1] * shape[0] + y * shape[0] + x] = value;
        }
        public float this[int x, int y, int z, int w]
        {
            get => data[w * shape[2] * shape[1] * shape[0] + z * shape[1] * shape[0] + y * shape[0] + x];
            set => data[w * shape[2] * shape[1] * shape[0] + z * shape[1] * shape[0] + y * shape[0] + x] = value;
        }
        public static Tensor operator +(Tensor left, float right)
        {
            Tensor result = left.Clone() as Tensor;

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] += right;
            }

            return result;
        }
        public static Tensor operator -(Tensor left, float right)
        {
            Tensor result = left.Clone() as Tensor;

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] -= right;
            }

            return result;
        }
        public static Tensor operator *(Tensor left, float right)
        {
            Tensor result = left.Clone() as Tensor;

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] *= right;
            }

            return result;
        }
        public static Tensor operator /(Tensor left, float right)
        {
            Tensor result = left.Clone() as Tensor;

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
            Tensor result = new Tensor(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] + right.data[i];
            }

            return result;
        }
        public static Tensor operator -(Tensor left, Tensor right)
        {
            Tensor result = new Tensor(left.shape);
            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] - right.data[i];
            }
            return result;
        }
        public static Tensor operator *(Tensor left, Tensor right)
        {
            Tensor result = new Tensor(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] * right.data[i];
            }

            return result;
        }
        public static Tensor operator /(Tensor left, Tensor right)
        {
            Tensor result = new Tensor(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] / right.data[i];
            }

            return result;
        }

        // Refactoring operations
        public static Tensor MatMul(Tensor left, Tensor right, ComputeShader MatMulCS = null)
        {
            int w1 = left.shape[0];
            int h1 = left.shape[1];
            int w2 = right.shape[0];
            int h2 = right.shape[1];
            int batch = left.shape[2];

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
                ComputeBuffer leftBuffer = new ComputeBuffer(left.data.Length, sizeof(float));
                ComputeBuffer rightBuffer = new ComputeBuffer(right.data.Length, sizeof(float));
                ComputeBuffer resultBuffer = new ComputeBuffer(w1 * h2 * batch, sizeof(float));
                leftBuffer.SetData(left.data);
                rightBuffer.SetData(right.data);


                MatMulCS.SetBuffer(0, "leftArr", leftBuffer);
                MatMulCS.SetBuffer(0, "rightArr", rightBuffer);
                MatMulCS.SetBuffer(0, "resultArr", resultBuffer);
                MatMulCS.SetInt("leftWidth", w1);
                MatMulCS.SetInt("leftHeightRightWidth", h1); // or w2 same thing
                MatMulCS.SetInt("rightHeight", h2);

                MatMulCS.Dispatch(0, 
                                 (w1 + numthreads[0] - 1) / numthreads[0], 
                                 (h2 + numthreads[1] - 1) / numthreads[1], 
                                 (batch + numthreads[2] - 1) / numthreads[2]);

                // Get result[]
                resultBuffer.GetData(resultTensor.data);
                
                leftBuffer.Dispose();
                rightBuffer.Dispose();
                resultBuffer.Dispose();
            }

            return resultTensor;
        }
        public static Tensor MatTranspose(Tensor tensor)
        {
            var shape = tensor.shape;
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
            if (axis < 0 || axis >= tensor.shape.Length)
                throw new ArgumentException("Invalid axis.");

            int[] shape = tensor.shape;
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

        // Math functions
        public static Tensor Pow(Tensor @base, float power)
        {
            Tensor result = new Tensor(@base.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Pow(@base.data[i], power);
            }
            return result;
        }
        public static Tensor Sqrt(Tensor @base)
        {
            Tensor result = new Tensor(@base.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Sqrt(@base.data[i]);
            }
            return result;
        }
        public static float Mean(Tensor tensor) => tensor.data.Average();
        public static float StdDev(Tensor tensor) => MathF.Sqrt(Var(tensor));
        public static float Var(Tensor tensor)
        {
            float sum = 0;
            float sumSqr = 0;

            for (int i = 0; i < tensor.data.Length; i++)
            {
                sum += tensor.data[i];
                sumSqr += tensor.data[i] * tensor.data[i];
            }

            float mean = sum / tensor.data.Length;
            float meanSquares = sumSqr / tensor.data.Length;
            return meanSquares - (mean * mean);
        }

        // LINQ
        public int Count(Func<float, bool> selector = null)
        {
            if (selector == null)
                return shape[0] * shape[1] * shape[2] * shape[3];

            int count = 0;

            for (int i = 0; i < data.Length; i++)
            {
                count += selector(data[i]) ? 1 : 0;
            }

            
            return count;
        }
        public float[] ToArray() => data.ToArray();
        public float Sum(Func<float, float> selector = null) 
        {
            if (selector == null)
                return data.Sum();

            float sum = 0.0f;

            for (int i = 0; i < data.Length; i++)
            {
                sum += selector(data[i]);
            }

            return sum;
        }
        public void ForEach(Func<float, float> action)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = action(data[i]);
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
        
        // System.Object/Collection
        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
        public IEnumerator<float> GetEnumerator()
        {
            for (int i = 0; i < data.Length; i++)
            {
                yield return data[i];
            }
        }
        public bool Equals(Tensor other)
        {
            if (!shape.SequenceEqual(other.shape))
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
            StringBuilder sb = new StringBuilder();
            sb.Append("[");

            int[] shape = this.shape;
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
        public string ShapeToString { get => "[" + string.Join(", ", Shape) + "]"; }
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
        public object Clone()
        {
            var clone = new Tensor(shape);
            Array.Copy(data, clone.data, data.Length);
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