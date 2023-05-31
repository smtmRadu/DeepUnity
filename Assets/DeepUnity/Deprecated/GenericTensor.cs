using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// [Deprecated]
    /// A generic version of Tensor class. Deprecated due lack of performance.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class GenericTensor<T> : IEnumerable, ICloneable where T : struct, IComparable<T>
    {
        private readonly static int[] numthreads = new int[] { 32, 32, 1 };

        public readonly int[] Shape;
        public readonly int Rank;
        private T[] data;
        public T this[int x]
        {
            get => data[x];
            set => data[x] = value;
        }
        public T this[int x, int y]
        {
            get => data[y * Shape[0] + x];
            set => data[y * Shape[0] + x] = value;
        }
        public T this[int x, int y, int z]
        {
            get => data[z * Shape[1] + y * Shape[0] + x];
            set => data[z * Shape[1] + y * Shape[0] + x] = value;
        }
        public T this[int x, int y, int z, int w]
        {
            get => data[w * Shape[2] + z * Shape[1] + y * Shape[0] + x];
            set => data[w * Shape[2] + z * Shape[1] + y * Shape[0] + x] = value;
        }

        
        private GenericTensor(params int[] shape)
        {
            if (shape.Length > 4)
                throw new Exception("Tensor cannot be instantiated with more than 4 dimensions.");

            Shape = new int[] { 1, 1, 1, 1};
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
            if(shape.Length > 3)
            {
                Shape[3] = shape[3];
                Rank = 4;
            }
                

            data = new T[Shape[0] * Shape[1] * Shape[2] * Shape[3]];
        }
        public static GenericTensor<T> Constant(T scalar)
        {
            GenericTensor<T> tensor = new GenericTensor<T>(1, 1, 1, 1);
            tensor[0] = scalar;
            return tensor;
        }
        public static GenericTensor<T> Constant(T[] vector)
        {
            GenericTensor<T> tensor = new GenericTensor<T>(vector.GetLength(0), 1, 1, 1);
            var shape = tensor.Shape;
            for (int i = 0; i < shape[0]; i++)
            {
                tensor[i] = vector[i];
            }
            return tensor;
        }
        public static GenericTensor<T> Constant(T[,] matrix)
        {
            GenericTensor<T> tensor = new GenericTensor<T>(matrix.GetLength(0), matrix.GetLength(1), 1, 1);
            var shape = tensor.Shape;
            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    tensor[i, j] = matrix[i, j];
                }
            }
            return tensor;
        }
        public static GenericTensor<T> Constant(T[,,] cuboid)
        {
            GenericTensor<T> tensor = new GenericTensor<T>(cuboid.GetLength(0), cuboid.GetLength(1), cuboid.GetLength(2), 1);
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
        public static GenericTensor<T> Constant(T[,,,] tesseract)
        {
            GenericTensor<T> tensor = new GenericTensor<T>(tesseract.GetLength(0), tesseract.GetLength(1), tesseract.GetLength(2), tesseract.GetLength(3));
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
        public static GenericTensor<T> Zeros(params int[] shape)
        {
            GenericTensor<T> tensor = new GenericTensor<T>(shape);
            shape = tensor.Shape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                           
                            tensor[i, j, k, l] = default;
                        }
                    }
                }
            }

            return tensor;
        }
        public static GenericTensor<T> Ones(params int[] shape)
        {
            GenericTensor<T> tensor = new GenericTensor<T>(shape);
            T one = (T)Convert.ChangeType(1f, typeof(T));
            shape = tensor.Shape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            tensor[i, j, k, l] = one;
                        }
                    }
                }
            }

            return tensor;
        }
        public static GenericTensor<T> Random(params int[] shape)
        {
            GenericTensor<T> tensor = new GenericTensor<T>(shape);

            shape = tensor.Shape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            tensor[i, j, k, l] = (T) Convert.ChangeType(Utils.Random.Value, typeof(T));
                        }
                    }
                }
            }

            return tensor;
        }
        public static GenericTensor<T> Normal(params int[] shape)
        {
            GenericTensor<T> tensor = new GenericTensor<T>(shape);

            for (int i = 0; i < tensor.data.Length; i++)
            {
                tensor.data[i] = (T)Convert.ChangeType(Utils.Random.Gaussian(0f, 1f, out _), typeof(T));
            }

            return tensor;
        }
        public static GenericTensor<T> Fill(T value, params int[] shape)
        {
            GenericTensor<T> tensor = new GenericTensor<T>(shape);
            shape = tensor.Shape;
            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            tensor[i, j, k, l] = value;
                        }
                    }
                }
            }

            return tensor;
        }


        public static GenericTensor<T> operator +(GenericTensor<T> left, T right)
        {
            var shape = left.Shape;
            GenericTensor<T> result = new GenericTensor<T>(shape[0], shape[1], shape[2], shape[3]);

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] += (dynamic) right;
                        }
                    }
                }
            }

            return result;
        }
        public static GenericTensor<T> operator -(GenericTensor<T> left, T right)
        {
            var shape = left.Shape;
            GenericTensor<T> result = new GenericTensor<T>(shape[0], shape[1], shape[2], shape[3]);

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] -= (dynamic)right;
                        }
                    }
                }
            }

            return result;
        }
        public static GenericTensor<T> operator *(GenericTensor<T> left, T right)
        {
            var shape = left.Shape;
            GenericTensor<T> result = new GenericTensor<T>(shape[0], shape[1], shape[2], shape[3]);

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] *= (dynamic)right;
                        }
                    }
                }
            }

            return result;
        }
        public static GenericTensor<T> operator /(GenericTensor<T> left, T right)
        {
            var shape = left.Shape;
            GenericTensor<T> result = new GenericTensor<T>(shape[0], shape[1], shape[2], shape[3]);

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] /= (dynamic)right;
                        }
                    }
                }
            }

            return result;
        }
        public static GenericTensor<T> operator +(T left, GenericTensor<T> right) => right + left;
        public static GenericTensor<T> operator *(T left, GenericTensor<T> right) => right * left;
        public static GenericTensor<T> operator +(GenericTensor<T> left, GenericTensor<T> right)
        {
            var shape = left.Shape;
            GenericTensor<T> result = new GenericTensor<T>(shape[0], shape[1], shape[2], shape[3]);

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] = (dynamic)left[i, j, k, l] + right[i, j, k, l];
                        }
                    }
                }
            }

            return result;
        }
        public static GenericTensor<T> operator -(GenericTensor<T> left, GenericTensor<T> right)
        {
            var shape = left.Shape;
            GenericTensor<T> result = new GenericTensor<T>(shape[0], shape[1], shape[2], shape[3]);
            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] = (dynamic)left[i, j, k, l] - right[i, j, k, l];
                        }
                    }
                }
            }
            return result;
        }
        public static GenericTensor<T> operator *(GenericTensor<T> left, GenericTensor<T> right)
        {
            var shape = left.Shape;
            GenericTensor<T> result = new GenericTensor<T>(shape[0], shape[1], shape[2], shape[3]);

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] = (dynamic)left[i, j, k, l] * right[i, j, k, l];
                        }
                    }
                }
            }

            return result;
        }
        public static GenericTensor<T> operator /(GenericTensor<T> left, GenericTensor<T> right)
        {
            var shape = left.Shape;
            GenericTensor<T> result = new GenericTensor<T>(shape[0], shape[1], shape[2], shape[3]);

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] = (dynamic)left[i, j, k, l] / right[i, j, k, l];
                        }
                    }
                }
            }

            return result;
        }

        public static GenericTensor<T> MatMul(GenericTensor<T> left, GenericTensor<T> right, ComputeShader matmulCS = null)
        {
            int[] shape1 = left.Shape;
            int[] shape2 = right.Shape;

            int w1 = shape1[0];
            int h1 = shape1[1];
            int w2 = shape2[0];
            int h2 = shape2[1];
            int batch = shape1[2];

            if (h1 != w2)
                throw new ArgumentException("Tensors must have compatible shapes for matrix multiplication (height of left tensor is not matching the width of the right tensor).");

            GenericTensor<T> resultTensor = new GenericTensor<T>(w1, h2, batch);

            
            if (matmulCS == null)
            {            
                System.Threading.Tasks.Parallel.For(0, w1, i =>
                {
                    for (int k = 0; k < batch; k++)
                    {
                        for (int j = 0; j < h2; j++)
                        {
                            dynamic sum = 0.0f;

                            for (int l = 0; l < h1; l++)
                            {
                                sum += (dynamic)left[i, l, k] * right[l, j, k];
                            }

                            lock (resultTensor)
                                resultTensor[i, j, k] = (T) sum;                      
                        }
                    }             
                });
            }
            else
            {
                // Setup CS
                int kernelIndex = matmulCS.FindKernel("MatMul");

                float[] leftArr = (dynamic)left.data;
                float[] rightArr = (dynamic)right.data;

                ComputeBuffer leftBuffer = new ComputeBuffer(leftArr.Length, sizeof(float));
                ComputeBuffer rightBuffer = new ComputeBuffer(rightArr.Length, sizeof(float));
                ComputeBuffer resultBuffer = new ComputeBuffer(w1 * h2 * batch, sizeof(float));
                leftBuffer.SetData(leftArr);
                rightBuffer.SetData(rightArr);
                

                matmulCS.SetBuffer(kernelIndex, "leftArr", leftBuffer);
                matmulCS.SetBuffer(kernelIndex, "rightArr", rightBuffer);
                matmulCS.SetBuffer(kernelIndex, "resultArr", resultBuffer);
                matmulCS.SetInt("leftWidth", w1);
                matmulCS.SetInt("leftHeight", h1);
                matmulCS.SetInt("rightWidth", w2);
                matmulCS.SetInt("rightHeight", h2);

                matmulCS.Dispatch(kernelIndex, (w1 + numthreads[0] - 1) / numthreads[0], (h2 + numthreads[1] - 1) / numthreads[1], (batch + numthreads[2] - 1) / numthreads[2]);

                // Get result[]
                float[] resultArr = new float[w1 * h2 * batch];
                resultBuffer.GetData(resultArr);
                leftBuffer.Dispose();
                rightBuffer.Dispose();
                resultBuffer.Dispose();

                // Convert result[] to Tensor<T>
                for (int i = 0; i < resultArr.Length; i++)
                {
                    resultTensor.data[i] = (dynamic)resultArr[i];
                }              
            }

            return resultTensor;
        }
        public static GenericTensor<T> MatTranspose(GenericTensor<T> tensor)
        {
            var shape = tensor.Shape;
            GenericTensor<T> result = new GenericTensor<T>(shape[1], shape[0], shape[2], shape[3]);

            for (int k = 0; k < shape[2]; k++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        result[j, i, k] = tensor[i, j, k];
                    }
                }
            }

            return result;
        }
        public static GenericTensor<T> MatReduceSum(GenericTensor<T> tensor)
        {
            var shape = tensor.Shape;
            GenericTensor<T> result = GenericTensor<T>.Zeros(shape[0]);

            for (int k = 0; k < shape[2]; k++)
            {
                for (int i = 0; i < shape[0]; i++)
                {
                    dynamic sum = 0.0;
                    for (int j = 0; j < shape[1]; j++)
                    {
                        sum += tensor[i, j, k];
                    }
                    result[i, 1, k] = sum;
                }
            }

            

            return result;
        }
        public static GenericTensor<T>[] Slice(GenericTensor<T> tensor, int axis)
        {
            if (axis < 0 || axis >= tensor.Shape.Length)
                throw new ArgumentException("Invalid axis.");

            int[] shape = tensor.Shape;
            int sliceSize = shape[axis];
            int[] slicedShape = shape.Clone() as int[];
            slicedShape[axis] = 1;

            GenericTensor<T>[] slices = new GenericTensor<T>[sliceSize];

            for (int i = 0; i < sliceSize; i++)
            {
                slices[i] = new GenericTensor<T>(slicedShape);

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

        public T Mean()
        {
            dynamic sum = 0;
            int count = 0;

            var shape = Shape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            sum += this[i, j, k, l];
                            count++;
                        }
                    }
                }
            }

            return (T)(sum / count);
        }
        public T StdDev()
        {
            float var = (float) Convert.ChangeType(Var(), typeof(float));
            float stddev = MathF.Sqrt(var);
            return (T) Convert.ChangeType(stddev, typeof(T));
        }
        public T Var()
        {
            dynamic sum = 0;
            dynamic sumSqr = 0;
            int count = 0;

            var shape = Shape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            sum += this[i, j, k, l];
                            sumSqr += (dynamic)this[i, j, k, l] * this[i, j, k, l];
                            count++;
                        }
                    }
                }
            }

            dynamic mean = sum / count;
            dynamic meanSqr = sumSqr / count;
            return (T)meanSqr - (mean * mean);
        }

        public int Count(Func<T, bool> selector = null)
        {
            if(selector == null)
                return Shape[0] * Shape[1] * Shape[2] * Shape[3];

            int count = 0;
            var shape = Shape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            count += Convert.ToInt32(selector(this[i, j, k, l]));

                        }   
                    }
                }
            }

            return count;
        }
        public T[] ToArray()
        {
            T[] arr = new T[Count()];

            var shape = Shape;
            int index = 0;


            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            arr[index++] = this[i, j, k, l];
                        }
                    }
                }
            }

            return arr;
        }
        public T Sum(Func<T, T> selector = null)
        {
            if (selector == null)
                selector = x => x;

            dynamic sum = 0.0;

            for (int i = 0; i < data.Length; i++)
            {
                sum += selector(data[i]);
            }

            return (T) sum;
        }
        public GenericTensor<T> Select(Func<T, T> selector)
        {
            var shape = Shape;
            GenericTensor<T> result = new GenericTensor<T>(shape[0], shape[1], shape[2], shape[3]);


            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] = selector(this[i, j, k, l]);
                        }
                    }
                }
            }

            return result;
        }
        public GenericTensor<T> Zip(GenericTensor<T> second, Func<T, T, T> resultSelector)
        {
            var shape = Shape;
            GenericTensor<T> result = new GenericTensor<T>(shape[0], shape[1], shape[2], shape[3]);

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] = resultSelector(this[i, j, k, l], second[i, j, k, l]);
                        }
                    }
                }
            }
            return result;
        }
        public void ForEach(Func<T,T> action)
        {
            var shape = Shape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            this[i, j, k, l] = action(this[i, j, k, l]);
                        }
                    }
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
        public IEnumerator<T> GetEnumerator()
        {
            var shape = Shape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            yield return this[i, j, k, l];
                        }
                    }
                }
            }
        }
        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
                return false;

            GenericTensor<T> other = (GenericTensor<T>)obj;

            if (!Shape.SequenceEqual(other.Shape))
                return false;

            for (int i = 0; i < data.Length; i++)
            {
                if (data[i].Equals(other.data[i]))
                    return false;
            }
            
            return true;
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
            var _shape = this.Shape;
            var clone = new GenericTensor<T>(_shape[0], _shape[1], _shape[2], _shape[3]);
            Array.Copy(data, clone.data, data.Length);
            return clone;
        }
        
    }
}


// Matmul benchmark
/* For matrices [100,1000] x [1000,500]
 * One matmul : CPU: 36.48s, GPU: 00.09s
 * 
 * For matrices [10,5] * [5,3]
 * 100 matmul : CPU: 00.04s  , GPU: 00.10s 
 * 
 * Benchmark conclusion
 * For batched matmul or large dimensions matrices it is **neccesary to use GPU device**.
 * For non-batched matmul and small sized matrices CPU matmul scores a better runtime (x2 times better), but still GPU based matmul is a viable option.
 *///