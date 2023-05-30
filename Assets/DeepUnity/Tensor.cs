using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    public class Tensor<T> : IEnumerable, ICloneable where T : struct
    {
        private static Vector3Int numthreads = new Vector3Int(32, 32, 1);
        private T[,,,] data;
        public T this[int x]
        {
            get => data[x, 0, 0, 0];
            set => data[x, 0, 0, 0] = value;
        }
        public T this[int x, int y]
        {
            get => data[x, y, 0, 0];
            set => data[x, y, 0, 0] = value;
        }
        public T this[int x, int y, int z]
        {
            get => data[x, y, z, 0];
            set => data[x, y, z, 0] = value;
        }
        public T this[int x, int y, int z, int w]
        {
            get => data[x, y, z, w];
            set => data[x, y, z, w] = value;
        }

        public int Rank
        {
            get
            {
                var shape = FullShape;
                for (int i = 3; i >= 0; i--)
                {
                    if (shape[i] > 1)
                        return i + 1;
                }
                return 0;
            }
        }
        public int[] Shape
        {
            get
            {
                int r = Rank;
                
                if (r == 0)
                    return new int[] { 1 };

                if (r == 1)
                    return new int[] { data.GetLength(0) };

                if (r == 2)
                    return new int[] { data.GetLength(0), data.GetLength(1) };

                if (r == 3)
                    return new int[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) };

                if (r == 4)
                    return new int[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) };

                throw new Exception("Rank too large?");
            }
        }         
        public int[] FullShape
        {
            get
            {
                return new int[] { data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3) };
            }
        }

        protected Tensor(params int[] shape)
        {
            if (shape.Length > 4)
                throw new Exception("Tensor cannot be instantiated with more than 4 dimensions.");

            int[] Tshape = Enumerable.Repeat(1, 4).ToArray();
            for (int i = 0; i < shape.Length; i++)
            {
                if (shape[i] < 1)
                    throw new Exception("Tensor cannot have a dimension size less than 1.");
                Tshape[i] = shape[i];
            }
            data = new T[Tshape[0], Tshape[1], Tshape[2], Tshape[3]];

        }
        public static Tensor<T> Constant(T scalar)
        {
            Tensor<T> tensor = new Tensor<T>(1, 1, 1, 1);
            tensor[0] = scalar;
            return tensor;
        }
        public static Tensor<T> Constant(T[] vector)
        {
            Tensor<T> tensor = new Tensor<T>(vector.GetLength(0), 1, 1, 1);
            var shape = tensor.FullShape;
            for (int i = 0; i < shape[0]; i++)
            {
                tensor[i] = vector[i];
            }
            return tensor;
        }
        public static Tensor<T> Constant(T[,] matrix)
        {
            Tensor<T> tensor = new Tensor<T>(matrix.GetLength(0), matrix.GetLength(1), 1, 1);
            var shape = tensor.FullShape;
            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    tensor[i, j] = matrix[i, j];
                }
            }
            return tensor;
        }
        public static Tensor<T> Constant(T[,,] cuboid)
        {
            Tensor<T> tensor = new Tensor<T>(cuboid.GetLength(0), cuboid.GetLength(1), cuboid.GetLength(2), 1);
            var shape = tensor.FullShape;


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
        public static Tensor<T> Constant(T[,,,] tesseract)
        {
            Tensor<T> tensor = new Tensor<T>(tesseract.GetLength(0), tesseract.GetLength(1), tesseract.GetLength(2), tesseract.GetLength(3));
            var shape = tensor.FullShape;

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
        public static Tensor<T> Zeros(params int[] shape)
        {
            Tensor<T> tensor = new Tensor<T>(shape);
            shape = tensor.FullShape;

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
        public static Tensor<T> Ones(params int[] shape)
        {
            Tensor<T> tensor = new Tensor<T>(shape);
            T one = (T)Convert.ChangeType(1, typeof(T));
            shape = tensor.FullShape;

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
        public static Tensor<T> Random(params int[] shape)
        {
            Tensor<T> tensor = new Tensor<T>(shape);
            T one = (T)Convert.ChangeType(1, typeof(T));
            shape = tensor.FullShape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            tensor[i, j, k, l] = (dynamic)Utils.Random.Value;
                        }
                    }
                }
            }

            return tensor;
        }
        public static Tensor<T> Normal(params int[] shape)
        {
            Tensor<T> tensor = new Tensor<T>(shape);
            T one = (T)Convert.ChangeType(1, typeof(T));
            shape = tensor.FullShape;


            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            tensor[i, j, k, l] = (dynamic)Utils.Random.Gaussian(0f, 1f, out _);
                        }
                    }
                }
            }

            return tensor;
        }
        public static Tensor<T> Fill(T value, params int[] shape)
        {
            Tensor<T> tensor = new Tensor<T>(shape);
            shape = tensor.FullShape;
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


        public static Tensor<T> operator +(Tensor<T> left, T right)
        {
            var shape = left.FullShape;
            Tensor<T> result = new Tensor<T>(shape[0], shape[1], shape[2], shape[3]);

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] += (dynamic)right;
                        }
                    }
                }
            }

            return result;
        }
        public static Tensor<T> operator -(Tensor<T> left, T right)
        {
            var shape = left.FullShape;
            Tensor<T> result = new Tensor<T>(shape[0], shape[1], shape[2], shape[3]);

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
        public static Tensor<T> operator *(Tensor<T> left, T right)
        {
            var shape = left.FullShape;
            Tensor<T> result = new Tensor<T>(shape[0], shape[1], shape[2], shape[3]);

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
        public static Tensor<T> operator /(Tensor<T> left, T right)
        {
            var shape = left.FullShape;
            Tensor<T> result = new Tensor<T>(shape[0], shape[1], shape[2], shape[3]);

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
        public static Tensor<T> operator ^(Tensor<T> @base, T power)
        {
            var shape = @base.FullShape;
            Tensor<T> result = new Tensor<T>(shape[0], shape[1], shape[2], shape[3]);

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] = (T)Math.Pow((dynamic)@base[i, j, k, l], (dynamic)power);
                        }
                    }
                }
            }

            return result;
        }
        public static Tensor<T> operator +(T left, Tensor<T> right) => right + left;
        public static Tensor<T> operator *(T left, Tensor<T> right) => right * left;
        public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right)
        {
            var shape = left.FullShape;
            Tensor<T> result = new Tensor<T>(shape[0], shape[1], shape[2], shape[3]);

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            result[i, j, k, l] = (dynamic)left[i, j, k, l] + (dynamic)right[i, j, k, l];
                        }
                    }
                }
            }

            return result;
        }
        public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right)
        {
            var shape = left.FullShape;
            Tensor<T> result = new Tensor<T>(shape[0], shape[1], shape[2], shape[3]);
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
        public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
        {
            var shape = left.FullShape;
            Tensor<T> result = new Tensor<T>(shape[0], shape[1], shape[2], shape[3]);

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
        public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right)
        {
            var shape = left.FullShape;
            Tensor<T> result = new Tensor<T>(shape[0], shape[1], shape[2], shape[3]);

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

        public static Tensor<T> MatMul(Tensor<T> left, Tensor<T> right, ComputeShader matmulCS = null)
        {
            int[] shape1 = left.FullShape;
            int[] shape2 = right.FullShape;

            int w1 = shape1[0];
            int h1 = shape1[1];
            int w2 = shape2[0];
            int h2 = shape2[1];
            int batch = shape1[2];

            if (h1 != w2)
                throw new ArgumentException("Tensors must have compatible shapes for matrix multiplication (height of left tensor is not matching the width of the right tensor).");

            Tensor<T> resultTensor = new Tensor<T>(w1, h2, batch);

            
            if (matmulCS == null)
            {            
                System.Threading.Tasks.Parallel.For(0, w1, i =>
                {
                    for (int k = 0; k < batch; k++)
                    {
                        for (int j = 0; j < h2; j++)
                        {
                            dynamic sum = 0.0;

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

                float[] leftArr = left.ToArray<float>();
                float[] rightArr = right.ToArray<float>();

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

                matmulCS.Dispatch(kernelIndex, (w1 + numthreads.x - 1) / numthreads.x, (h2 + numthreads.y - 1) / numthreads.y, (batch + numthreads.z - 1) / numthreads.z);

                // Get result[]
                float[] resultArr = new float[w1 * h2 * batch];
                resultBuffer.GetData(resultArr);
                leftBuffer.Dispose();
                rightBuffer.Dispose();
                resultBuffer.Dispose();

                // Convert result[] to Tensor<T>
                int index = 0;

                for (int k = 0; k < batch; k++)
                {
                    for (int j = 0; j < h2; j++)
                    {
                        for (int i = 0; i < w1; i++)
                        {
                            resultTensor[i, j, k] = (dynamic)resultArr[index++];
                        }
                    }
                }               
            }

            return resultTensor;
        }
        public static Tensor<T> MatTranspose(Tensor<T> tensor)
        {
            var shape = tensor.FullShape;
            Tensor<T> result = new Tensor<T>(shape[1], shape[0], shape[2], shape[3]);

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
        public static Tensor<T> MatReduceSum(Tensor<T> tensor)
        {
            var shape = tensor.FullShape;
            Tensor<T> result = Tensor<T>.Zeros(shape[0]);

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
        public static Tensor<T>[] Slice(Tensor<T> tensor, int axis)
        {
            if (axis < 0 || axis >= tensor.FullShape.Length)
                throw new ArgumentException("Invalid axis.");

            int[] shape = tensor.FullShape;
            int sliceSize = shape[axis];
            int[] slicedShape = shape.Clone() as int[];
            slicedShape[axis] = 1;

            Tensor<T>[] slices = new Tensor<T>[sliceSize];

            for (int i = 0; i < sliceSize; i++)
            {
                slices[i] = new Tensor<T>(slicedShape);

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

            var shape = FullShape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            sum += (dynamic)this[i, j, k, l];
                            count++;
                        }
                    }
                }
            }

            return (T)(sum / count);
        }
        public T StdDev()
        {
            return (T)Math.Sqrt((dynamic)Var());
        }
        public T Var()
        {
            dynamic sum = 0;
            dynamic sumSqr = 0;
            int count = 0;

            var shape = FullShape;

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
                return data.GetLength(0) * data.GetLength(1) * data.GetLength(2) * data.GetLength(3);

            int count = 0;
            var shape = FullShape;

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

            var shape = FullShape;
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
        public TResult[] ToArray<TResult>()
        {
            TResult[] arr = new TResult[Count()];

            var shape = FullShape;
            int index = 0;


            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            arr[index++] = (dynamic)this[i, j, k, l];
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


            var shape = FullShape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            sum += selector(this[i, j, k, l]);
                        }
                    }
                }
            }

            return (T) sum;
        }
        public Tensor<TResult> Select<TResult>(Func<T, TResult> selector) where TResult : struct
        {
            var shape = FullShape;
            Tensor<TResult> result = new Tensor<TResult>(shape[0], shape[1], shape[2], shape[3]);


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
        public Tensor<TResult> Zip<TSecond, TResult>(Tensor<TSecond> second, Func<T, TSecond, TResult> resultSelector) where TResult : struct where TSecond : struct
        {
            var shape = FullShape;
            Tensor<TResult> result = new Tensor<TResult>(shape[0], shape[1], shape[2], shape[3]);

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
            var shape = FullShape;

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
            var shape = FullShape;

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

            Tensor<T> other = (Tensor<T>)obj;

            if (!FullShape.SequenceEqual(other.FullShape))
                return false;


            for (int i = 0; i < data.GetLength(0); i++)
            {
                for (int j = 0; j < data.GetLength(1); j++)
                {
                    for (int k = 0; k < data.GetLength(2); k++)
                    {
                        for (int l = 0; l < data.GetLength(3); l++)
                        {
                            if (!this[i, j, k, l].Equals(other[i, j, k, l]))
                            {
                                return false;
                            }
                        }
                    }
                }
            }

            return true;
        }
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("[");

            int[] shape = FullShape;
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
            var _shape = this.FullShape;
            var clone = new Tensor<T>(_shape[0], _shape[1], _shape[2], _shape[3]);
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