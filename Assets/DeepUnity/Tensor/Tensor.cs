using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Tensor : IEnumerable, IEquatable<Tensor>
    {
        private readonly static ComputeShader MatMulCS;
        private readonly static int[] numthreads;

        [SerializeField] private int[] shape;
        [SerializeField] private float[] data;
        private Tape tape = null;

        // Create
        static Tensor()
        {
            numthreads = new int[] { 32, 32, 1 };

            string csguid = AssetDatabase.FindAssets("MatMulCS")[0];
            string cspath = AssetDatabase.GUIDToAssetPath(csguid);
            MatMulCS = AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
        }
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

            int size = shape[0] * shape[1] * shape[2] * shape[3];

            if (size > 16_777_216) // hardcoded like this because 4096x4096 max allowed matrix, on 8192 it crashes
                throw new Exception("Tensor dimensions is too large on initialization (cannot surpass 16,777,216 units).");

            data = new float[size];
        }
        public static Tensor Identity(Tensor other)
        {
            Tensor clone = new Tensor(other.shape);
            Array.Copy(other.data, clone.data, other.data.Length);
            Array.Copy(other.shape, clone.shape, other.shape.Length);
            return clone;
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
                tensor.data[i] = 1f;
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
        public string ShapeToString
        {
            get => "[" + string.Join(", ", Shape) + "]";
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
        public static Tensor MatMul(Tensor left, Tensor right, Device device)
        {
            int w1 = left.shape[0];
            int h1 = left.shape[1];
            int w2 = right.shape[0];
            int h2 = right.shape[1];
            int batch = left.shape[2];

            if (h1 != w2)
                throw new ArgumentException("Tensors must have compatible shapes for matrix multiplication (height of left tensor is not matching the width of the right tensor).");

            Tensor resultTensor = new Tensor(w1, h2, batch);


            if (device == Device.CPU)
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

                            resultTensor[i, j, k] = sum;

                        }
                    }
                });
            }
            else
            {
                ComputeBuffer leftBuffer = new ComputeBuffer(left.data.Length, 4);
                ComputeBuffer rightBuffer = new ComputeBuffer(right.data.Length, 4);
                ComputeBuffer resultBuffer = new ComputeBuffer(w1 * h2 * batch, 4);

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

                resultBuffer.GetData(resultTensor.data);

                leftBuffer.Dispose();
                rightBuffer.Dispose();
                resultBuffer.Dispose();
            }


            return resultTensor;
        }
        public static Tensor TransposeMat(Tensor tensor)
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
        public static Tensor JoinSclsToVec(Tensor[] scalars)
        {
            Tensor vector = Zeros(scalars.Length);
            for (int i = 0; i < scalars.Length; i++)
            {
                vector.data[i] = scalars[i].data[0];
            }
            return vector;

        }
        public static Tensor JoinVecsToMat(Tensor[] vectors)
        {
            Tensor matrix = new Tensor(vectors[0].shape[0], vectors.Length);

            int veclength = vectors[0].shape[0];
            int index = 0;
            for (int i = 0; i < vectors.Length; i++)
            {
                for (int j = 0; j < veclength; j++)
                {
                    matrix.data[index++] = vectors[i].data[j]; 
                }
            }

            return matrix;
        }
        public static Tensor JoinMatsToCube(Tensor[] matrices)
        {
            int matRows = matrices[0].shape[0];
            int matCols = matrices[0].shape[1];
            int cubeDepth = matrices.Length;

            Tensor cube = new Tensor(matRows, matCols, cubeDepth);

            int index = 0;
            for (int k = 0; k < cubeDepth; k++)
            {
                for (int i = 0; i < matRows; i++)
                {
                    for (int j = 0; j < matCols; j++)
                    {
                        cube.data[index++] = matrices[k].data[i * matCols + j];
                    }
                }
            }

            return cube;
        }
        public static Tensor ExpandScl(Tensor scalar, int times)
        {
            Tensor expandedScalar = Tensor.Zeros(times);
            float scalarValue = scalar.data[0];

            for (int i = 0; i < times; i++)
            {
                expandedScalar.data[i] = scalarValue;
            }

            return expandedScalar;
        }
        public static Tensor ExpandVec(Tensor vector, int times)
        {
            int vecLength = vector.shape[0];
            Tensor matrix = Zeros(vecLength, times);

            int index = 0;
            for (int k = 0; k < times; k++)
            {
                for (int i = 0; i < vecLength; i++)
                {
                    matrix.data[index++] = vector.data[i];
                }
            }

            return matrix;
        }
        public static Tensor ExpandMat(Tensor matrix, int times)
        {
            int matRows = matrix.shape[0];
            int matCols = matrix.shape[1];

            Tensor expandedMatrix = Tensor.Zeros(matRows, matCols * times);

            for (int i = 0; i < matRows; i++)
            {
                for (int j = 0; j < matCols; j++)
                {
                    float matrixValue = matrix.data[i * matCols + j];
                    int index = i * (matCols * times) + j;
                    for (int k = 0; k < times; k++)
                    {
                        expandedMatrix.data[index] = matrixValue;
                        index += matCols;
                    }
                }
            }

            return expandedMatrix;
        }
        public static Tensor[] SliceVec(Tensor vector)
        {
            Tensor[] scalars = new Tensor[vector.shape[0]];
            for (int i = 0; i < scalars.Length; i++)
            {
                scalars[i] = Tensor.Constant(vector.data[i]);
            }
            return scalars;
        }
        public static Tensor[] SliceMatrix(Tensor matrix)
        {
            Tensor[] vectors = new Tensor[matrix.shape[1]];

            for (int j = 0; j < matrix.shape[1]; j++)
            {
                Tensor vector = new Tensor(matrix.shape[0]);

                for (int i = 0; i < matrix.shape[0]; i++)
                {
                    vector.data[i] = matrix[i, j];
                }

                vectors[j] = vector;
            }
            return vectors;
        }
        public static Tensor[] SliceCube(Tensor cube)
        {
            Tensor[] matrices = new Tensor[cube.shape[2]];

            for (int k = 0; k < cube.shape[2]; k++)
            {
                Tensor matrix = new Tensor(cube.shape[0], cube.shape[1]);

                for (int j = 0; j < cube.shape[1]; j++)
                {
                    for (int i = 0; i < cube.shape[0]; i++)
                    {
                        matrix[i, j] = cube[i, j, k];
                    }
                }

                matrices[k] = matrix;
            }

            return matrices;
        }

        // Math functions
        public static Tensor Exp(Tensor tensor)
        {
            Tensor result = new Tensor(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Exp(tensor.data[i]);

            }

            return result;
        }
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
        public static Tensor Abs(Tensor tensor)
        {
            Tensor result = new Tensor(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Abs(tensor.data[i]);
            }

            return result;
        }
        public static Tensor Max(Tensor left, Tensor right)
        {
            Tensor result = new Tensor(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Max(left.data[i], right.data[i]);
            }

            return result;
        }
        public static Tensor Min(Tensor left, Tensor right)
        {
            Tensor result = new Tensor(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Min(left.data[i], right.data[i]);
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


        // LINQ (Not applied in autograd system)
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
        public void ForEach(Func<float, float> action, bool multithreaded = false)
        {
            if(multithreaded)
                System.Threading.Tasks.Parallel.For(0, data.Length, i =>
                {
                    data[i] = action(data[i]);
                });
            else
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
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        
    }
}