using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// [width, height, depth, n-dim]     <br />
    /// </summary>
    [Serializable]
    public class NDArray : IEnumerable, IEquatable<NDArray>
    {
        [SerializeField] protected int[] shape;
        [SerializeField] protected float[] data;


        // Create
        protected NDArray(params int[] short_shape)
        {
            if (short_shape.Length > 4)
                throw new Exception("NDArray cannot be instantiated with more than 4 dimensions.");

            this.shape = new int[] { 1, 1, 1, 1 };

            if (short_shape.Length > 0)
            {
                this.shape[0] = short_shape[0];
            }
            if (short_shape.Length > 1)
            {
                this.shape[1] = short_shape[1];
            }
            if (short_shape.Length > 2)
            {
                this.shape[2] = short_shape[2];
            }
            if (short_shape.Length > 3)
            {
                this.shape[3] = short_shape[3];
            }

            int size = shape[0] * shape[1] * shape[2] * shape[3];

            if (size > 16_777_216) // hardcoded like this because 4096x4096 max allowed matrix, on 8192 it crashes
                throw new Exception("NDArray dimensions is too large on initialization (cannot surpass 16,777,216 units).");

            data = new float[size];
        }
        public static NDArray Identity(NDArray other)
        {
            NDArray clone = new NDArray(other.shape);
            Array.Copy(other.data, clone.data, other.data.Length);
            return clone;
        }
        public static NDArray Constant(float scalar)
        {
            NDArray ndarray = new NDArray();
            ndarray.data[0] = scalar;
            return ndarray;
        }
        public static NDArray Constant(float[] vector)
        {
            NDArray ndarray = new NDArray(vector.GetLength(0));
            var shape = ndarray.shape;
            for (int i = 0; i < shape[0]; i++)
            {
                ndarray.data[i] = vector[i];
            }
            return ndarray;
        }
        public static NDArray Constant(float[,] matrix)
        {
            NDArray ndarray = new NDArray(matrix.GetLength(0), matrix.GetLength(1));
            var shape = ndarray.shape;
            for (int j = 0; j < shape[1]; j++)
            {
                for (int i = 0; i < shape[0]; i++)
                {

                    ndarray[i, j] = matrix[i, j];

                }
            }
            return ndarray;
        }
        public static NDArray Constant(float[,,] cuboid)
        {
            NDArray ndarray = new NDArray(cuboid.GetLength(0), cuboid.GetLength(1), cuboid.GetLength(2));
            var shape = ndarray.shape;


            for (int k = 0; k < shape[2]; k++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        ndarray[i, j, k] = cuboid[i, j, k];
                    }
                }
            }

            return ndarray;
        }
        public static NDArray Constant(float[,,,] tesseract)
        {
            NDArray ndarray = new NDArray(tesseract.GetLength(0), tesseract.GetLength(1), tesseract.GetLength(2), tesseract.GetLength(3));
            var shape = ndarray.shape;

            for (int l = 0; l < shape[3]; l++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            ndarray[i, j, k, l] = tesseract[i, j, k, l];
                        }
                    }
                }
            }

            return ndarray;
        }
        public static NDArray Zeros(params int[] shape) => new NDArray(shape);
        public static NDArray Ones(params int[] shape)
        {
            NDArray NDArray = new NDArray(shape);

            for (int i = 0; i < NDArray.data.Length; i++)
            {
                NDArray.data[i] = 1f;
            }

            return NDArray;
        }
        public static NDArray Random(params int[] shape)
        {
            NDArray ndarray = new NDArray(shape);

            for (int i = 0; i < ndarray.data.Length; i++)
            {
                ndarray.data[i] = Utils.Random.Value;
            }

            return ndarray;
        }
        public static NDArray RandomNormal(params int[] shape)
        {
            NDArray ndarray = new NDArray(shape);

            for (int i = 0; i < ndarray.data.Length; i++)
            {
                ndarray.data[i] = Utils.Random.Gaussian();
            }

            return ndarray;
        }
        public static NDArray Fill(float value, params int[] shape)
        {
            NDArray ndarray = new NDArray(shape);

            for (int i = 0; i < ndarray.data.Length; i++)
            {
                ndarray.data[i] = value;
            }

            return ndarray;
        }

        // Operators
        public static NDArray operator +(NDArray ndarray)
        {
            NDArray result = new NDArray(ndarray.shape);
            for (int i = 0; i < ndarray.data.Length; i++)
            {
                result.data[i] = ndarray.data[i];
            }

            return result;
        }
        public static NDArray operator -(NDArray ndarray)
        {
            NDArray result = new NDArray(ndarray.shape);
            for (int i = 0; i < ndarray.data.Length; i++)
            {
                result.data[i] = -ndarray.data[i];
            }

            return result;
        }
        public static NDArray operator +(NDArray left, float right)
        {
            NDArray result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] += right;
            }

            return result;
        }
        public static NDArray operator -(NDArray left, float right)
        {
            NDArray result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] -= right;
            }

            return result;
        }
        public static NDArray operator *(NDArray left, float right)
        {
            NDArray result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] *= right;
            }

            return result;
        }
        public static NDArray operator /(NDArray left, float right)
        {
            NDArray result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] /= right;
            }

            return result;
        }
        public static NDArray operator +(float left, NDArray right) => right + left;
        public static NDArray operator *(float left, NDArray right) => right * left;
        public static NDArray operator +(NDArray left, NDArray right)
        {
            NDArray result = new NDArray(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] + right.data[i];
            }


            return result;
        }
        public static NDArray operator -(NDArray left, NDArray right)
        {
            NDArray result = new NDArray(left.shape);
            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] - right.data[i];
            }

            return result;
        }
        public static NDArray operator *(NDArray left, NDArray right)
        {
            NDArray result = new NDArray(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] * right.data[i];
            }


            return result;
        }
        public static NDArray operator /(NDArray left, NDArray right)
        {
            NDArray result = new NDArray(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] / right.data[i];
            }

            return result;
        }

        // Operations
        public static NDArray MatMul(NDArray left, NDArray right)
        {
            int w1 = left.shape[0];
            int h1 = left.shape[1];
            int w2 = right.shape[0];
            int h2 = right.shape[1];
            int batch = left.shape[2];

            if (h1 != w2)
                throw new ArgumentException("NDArray must have compatible shapes for matrix multiplication (height of left ndarray is not matching the width of the right ndarray).");

            NDArray resultNDArray = new NDArray(w1, h2, batch);


            if (Settings.Device == Device.CPU)
            {
                Parallel.For(0, w1, i =>
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

                            resultNDArray[i, j, k] = sum;

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

                Settings.MatMulCS.SetBuffer(0, "leftArr", leftBuffer);
                Settings.MatMulCS.SetBuffer(0, "rightArr", rightBuffer);
                Settings.MatMulCS.SetBuffer(0, "resultArr", resultBuffer);
                Settings.MatMulCS.SetInt("leftWidth", w1);
                Settings.MatMulCS.SetInt("leftHeightRightWidth", h1); // or w2 same thing
                Settings.MatMulCS.SetInt("rightHeight", h2);

                Settings.MatMulCS.Dispatch(0,
                                 (w1 + Settings.numthreads[0] - 1) / Settings.numthreads[0],
                                 (h2 + Settings.numthreads[1] - 1) / Settings.numthreads[1],
                                 (batch + Settings.numthreads[2] - 1) / Settings.numthreads[2]);

                resultBuffer.GetData(resultNDArray.data);

                leftBuffer.Dispose();
                rightBuffer.Dispose();
                resultBuffer.Dispose();
            }


            return resultNDArray;
        }
        public static NDArray Transpose(NDArray ndarray, int axis0, int axis1)
        {
            TreatNegativeAxis(ref ndarray, ref axis0);
            TreatNegativeAxis(ref ndarray, ref axis1);

            // For mat transpose, axis0 = 0, axis1 = 1
            int[] shape = ndarray.shape;
            int[] transposedShape = shape.ToArray();

            // Swap the dimensions
            int dim0 = transposedShape[axis0];
            transposedShape[axis0] = transposedShape[axis1];
            transposedShape[axis1] = dim0;

            NDArray transposed = Zeros(transposedShape);

            int[] indices = new int[4];
            int newIndex0, newIndex1, newIndex2, newIndex3;

            for (indices[0] = 0; indices[0] < shape[0]; indices[0]++)
            {
                for (indices[1] = 0; indices[1] < shape[1]; indices[1]++)
                {
                    for (indices[2] = 0; indices[2] < shape[2]; indices[2]++)
                    {
                        for (indices[3] = 0; indices[3] < shape[3]; indices[3]++)
                        {
                            // Set the indices according to the transposition
                            newIndex0 = (axis0 == 0) ? indices[axis1] : indices[0];
                            newIndex1 = (axis1 == 1) ? indices[axis0] : indices[1];
                            newIndex2 = (axis0 == 2) ? indices[axis1] : indices[2];
                            newIndex3 = (axis1 == 3) ? indices[axis0] : indices[3];

                            // Assign the element to the transposed ndarray
                            transposed[newIndex0, newIndex1, newIndex2, newIndex3] = ndarray[indices[0], indices[1], indices[2], indices[3]];
                        }
                    }
                }
            }

            return transposed;
        }
        public static NDArray Join(int axis, params NDArray[] ndarrays)
        {
            if (ndarrays == null || ndarrays.Length == 0)
                throw new Exception("NDArray used for joining are not defined.");

            TreatNegativeAxis(ref ndarrays[0], ref axis);

            int[] sliceShape = ndarrays[0].shape;
            int[] joinedShape = sliceShape.ToArray();
            joinedShape[axis] = ndarrays.Length;

            NDArray joinedNDArray = Zeros(joinedShape);

            // tested on multithreaded.. works good when the ndarray shape is large enough, but singlethreaded is faster for small ndarrays
            //Parallel.For(0, ndarrays.Length, s =>
            for (int s = 0; s < ndarrays.Length; s++)
            {
                if (axis == 0)
                {
                    for (int i = 0; i < sliceShape[0]; i++)
                    {
                        for (int j = 0; j < sliceShape[1]; j++)
                        {
                            for (int k = 0; k < sliceShape[2]; k++)
                            {
                                for (int l = 0; l < sliceShape[3]; l++)
                                {
                                    joinedNDArray[s, j, k, l] = ndarrays[s][i, j, k, l];
                                }
                            }
                        }
                    }

                }
                else if (axis == 1)
                {
                    for (int i = 0; i < sliceShape[0]; i++)
                    {
                        for (int j = 0; j < sliceShape[1]; j++)
                        {
                            for (int k = 0; k < sliceShape[2]; k++)
                            {
                                for (int l = 0; l < sliceShape[3]; l++)
                                {
                                    joinedNDArray[i, s, k, l] = ndarrays[s][i, j, k, l];
                                }
                            }
                        }
                    }

                }
                else if (axis == 2)
                {

                    for (int i = 0; i < sliceShape[0]; i++)
                    {
                        for (int j = 0; j < sliceShape[1]; j++)
                        {
                            for (int k = 0; k < sliceShape[2]; k++)
                            {
                                for (int l = 0; l < sliceShape[3]; l++)
                                {
                                    joinedNDArray[i, j, s, l] = ndarrays[s][i, j, k, l];
                                }
                            }
                        }
                    }

                }
                else if (axis == 3)
                {
                    for (int i = 0; i < sliceShape[0]; i++)
                    {
                        for (int j = 0; j < sliceShape[1]; j++)
                        {
                            for (int k = 0; k < sliceShape[2]; k++)
                            {
                                for (int l = 0; l < sliceShape[3]; l++)
                                {
                                    joinedNDArray[i, j, k, s] = ndarrays[s][i, j, k, l];
                                }
                            }
                        }
                    }
                }
            }//);


            return joinedNDArray;

        }
        public static NDArray Expand(NDArray ndarray, int axis, int times)
        {
            TreatNegativeAxis(ref ndarray, ref axis);

            int[] shape = ndarray.shape;
            int[] expandedShape = shape.ToArray();
            expandedShape[axis] *= times;

            NDArray expandedNDArray = Zeros(expandedShape);

            if (axis == 0)
            {
                for (int t = 0; t < times; t++)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            for (int k = 0; k < shape[2]; k++)
                            {
                                for (int l = 0; l < shape[3]; l++)
                                {
                                    expandedNDArray[t * shape[0] + i, j, k, l] = ndarray[i, j, k, l];
                                }
                            }
                        }
                    }
                }
            }
            else if (axis == 1)
            {
                for (int t = 0; t < times; t++)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            for (int k = 0; k < shape[2]; k++)
                            {
                                for (int l = 0; l < shape[3]; l++)
                                {
                                    expandedNDArray[i, t * shape[1] + j, k, l] = ndarray[i, j, k, l];
                                }
                            }
                        }
                    }
                }
            }
            else if (axis == 2)
            {
                for (int t = 0; t < times; t++)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            for (int k = 0; k < shape[2]; k++)
                            {
                                for (int l = 0; l < shape[3]; l++)
                                {
                                    expandedNDArray[i, j, t * shape[2] + k, l] = ndarray[i, j, k, l];
                                }
                            }
                        }
                    }
                }
            }
            else if (axis == 3)
            {
                for (int t = 0; t < times; t++)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            for (int k = 0; k < shape[2]; k++)
                            {
                                for (int l = 0; l < shape[3]; l++)
                                {
                                    expandedNDArray[i, j, k, t * shape[3] + l] = ndarray[i, j, k, l];
                                }
                            }
                        }
                    }
                }
            }

            return expandedNDArray;
        }
        public static NDArray[] Split(NDArray ndarray, int axis, int split_size)
        {
            TreatNegativeAxis(ref ndarray, ref axis);

            int[] shape = ndarray.shape;
            int dimAxis = shape[axis];

            List<NDArray> slices = new List<NDArray>();

            int dimIndex = 0;
            while (dimIndex < dimAxis)
            {
                // If axis dim is a multiple of split_size, then each slice will have the exact same size
                // Otherwise, the last slice will have a smaller shape
                int dimCopySize = Math.Min(split_size, dimAxis - dimIndex);
                int[] sliceShape = shape.ToArray();
                sliceShape[axis] = dimCopySize;
                NDArray slice = Zeros(sliceShape);

                // Copy the data from the original ndarray to the slice ndarray based on the specified axis
                if (axis == 0)
                {
                    for (int i = 0; i < dimCopySize; i++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            for (int k = 0; k < shape[2]; k++)
                            {
                                for (int l = 0; l < shape[3]; l++)
                                {
                                    slice[i, j, k, l] = ndarray[dimIndex + i, j, k, l];
                                }
                            }
                        }
                    }
                }
                else if (axis == 1)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        for (int j = 0; j < dimCopySize; j++)
                        {
                            for (int k = 0; k < shape[2]; k++)
                            {
                                for (int l = 0; l < shape[3]; l++)
                                {
                                    slice[i, j, k, l] = ndarray[i, dimIndex + j, k, l];
                                }
                            }
                        }
                    }
                }
                else if (axis == 2)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            for (int k = 0; k < dimCopySize; k++)
                            {
                                for (int l = 0; l < shape[3]; l++)
                                {
                                    slice[i, j, k, l] = ndarray[i, j, dimIndex + k, l];
                                }
                            }
                        }
                    }
                }
                else if (axis == 3)
                {
                    for (int i = 0; i < shape[0]; i++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            for (int k = 0; k < shape[2]; k++)
                            {
                                for (int l = 0; l < dimCopySize; l++)
                                {
                                    slice[i, j, k, l] = ndarray[i, j, k, dimIndex + l];
                                }
                            }
                        }
                    }
                }

                slices.Add(slice);
                dimIndex += split_size;
            }

            return slices.ToArray();
        }
        public static NDArray Shuffle(NDArray ndarray, int axis)
        {
            TreatNegativeAxis(ref ndarray, ref axis);

            NDArray[] slices = Split(ndarray, axis, 1);
            slices = Utils.Shuffle(slices).ToArray();
            return Join(axis, slices);
        }
        public static NDArray Sum(NDArray ndarray, int axis)
        {
            TreatNegativeAxis(ref ndarray, ref axis);

            NDArray sumT = null;
            int[] shape = ndarray.shape;

            if (axis == 0)
            {
                sumT = Zeros(1, shape[1], shape[2], shape[3]);
                for (int l = 0; l < shape[3]; l++)
                {
                    for (int k = 0; k < shape[2]; k++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            float sum = 0f;
                            for (int i = 0; i < shape[0]; i++)
                            {
                                sum += ndarray[i, j, k, l];
                            }
                            sumT[0, j, k, l] = sum;
                        }
                    }
                }
            }
            else if (axis == 1)
            {
                sumT = Zeros(shape[0], 1, shape[2], shape[3]);
                for (int l = 0; l < shape[3]; l++)
                {
                    for (int k = 0; k < shape[2]; k++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            float sum = 0f;
                            for (int j = 0; j < shape[1]; j++)
                            {
                                sum += ndarray[i, j, k, l];
                            }
                            sumT[i, 0, k, l] = sum;
                        }
                    }
                }
            }
            else if (axis == 2)
            {
                sumT = Zeros(shape[0], shape[1], 1, shape[3]);
                for (int l = 0; l < shape[3]; l++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            float sum = 0f;
                            for (int k = 0; k < shape[2]; k++)
                            {
                                sum += ndarray[i, j, k, l];
                            }
                            sumT[i, j, 0, l] = sum;
                        }
                    }
                }
            }
            else if (axis == 3)
            {
                sumT = Zeros(shape[0], shape[1], shape[2], 1);
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            float sum = 0f;
                            for (int l = 0; l < shape[3]; l++)
                            {
                                sum += ndarray[i, j, k, l];
                            }
                            sumT[i, j, k, 0] = sum;
                        }
                    }
                }
            }

            return sumT;
        }
        public static NDArray Mean(NDArray ndarray, int axis)
        {
            TreatNegativeAxis(ref ndarray, ref axis);

            NDArray meanT = null;
            int[] shape = ndarray.shape;

            if (axis == 0)
            {
                meanT = Zeros(1, shape[1], shape[2], shape[3]);
                for (int l = 0; l < shape[3]; l++)
                {
                    for (int k = 0; k < shape[2]; k++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            float sum = 0f;
                            for (int i = 0; i < shape[0]; i++)
                            {
                                sum += ndarray[i, j, k, l];
                            }
                            meanT[0, j, k, l] = sum / shape[0];
                        }
                    }
                }
            }
            else if (axis == 1)
            {
                meanT = Zeros(shape[0], 1, shape[2], shape[3]);
                for (int l = 0; l < shape[3]; l++)
                {
                    for (int k = 0; k < shape[2]; k++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            float sum = 0f;
                            for (int j = 0; j < shape[1]; j++)
                            {
                                sum += ndarray[i, j, k, l];
                            }
                            meanT[i, 0, k, l] = sum / shape[1];
                        }
                    }
                }
            }
            else if (axis == 2)
            {
                meanT = Zeros(shape[0], shape[1], 1, shape[3]);
                for (int l = 0; l < shape[3]; l++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            float sum = 0f;
                            for (int k = 0; k < shape[2]; k++)
                            {
                                sum += ndarray[i, j, k, l];
                            }
                            meanT[i, j, 0, l] = sum / shape[2];
                        }
                    }
                }
            }
            else if (axis == 3)
            {
                meanT = Zeros(shape[0], shape[1], shape[2], 1);
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            float sum = 0f;
                            for (int l = 0; l < shape[3]; l++)
                            {
                                sum += ndarray[i, j, k, l];
                            }
                            meanT[i, j, k, 0] = sum / shape[3];
                        }
                    }
                }
            }
            else throw new Exception("Available axis for NDArray are 0, 1, 2 and 3.");

            return meanT;
        }
        public static NDArray Std(NDArray ndarray, int axis, int correction = 1)
        {
            return Sqrt(Var(ndarray, axis, correction));
        }
        public static NDArray Var(NDArray ndarray, int axis, int correction = 1)
        {
            TreatNegativeAxis(ref ndarray, ref axis);

            NDArray varT = null;
            int[] shape = ndarray.shape;

            if (axis == 0)
            {
                varT = Zeros(1, shape[1], shape[2], shape[3]);
                for (int l = 0; l < shape[3]; l++)
                {
                    for (int k = 0; k < shape[2]; k++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            float sum = 0f;
                            float sumSquared = 0f;
                            for (int i = 0; i < shape[0]; i++)
                            {
                                sum += ndarray[i, j, k, l];
                                sumSquared += ndarray[i, j, k, l] * ndarray[i, j, k, l];
                            }
                            varT[0, j, k, l] = (sumSquared - (sum * sum) / shape[0]) /
                                                    (shape[0] - correction);
                        }
                    }
                }
            }
            else if (axis == 1)
            {
                varT = Zeros(shape[0], 1, shape[2], shape[3]);
                for (int l = 0; l < shape[3]; l++)
                {
                    for (int k = 0; k < shape[2]; k++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            float sum = 0f;
                            float sumSquared = 0f;
                            for (int j = 0; j < shape[1]; j++)
                            {
                                sum += ndarray[i, j, k, l];
                                sumSquared += ndarray[i, j, k, l] * ndarray[i, j, k, l];
                            }
                            varT[i, 0, k, l] = (sumSquared - (sum * sum) / shape[1]) /
                                                    (shape[1] - correction);
                        }
                    }
                }
            }
            else if (axis == 2)
            {
                varT = Zeros(shape[0], shape[1], 1, shape[3]);
                for (int l = 0; l < shape[3]; l++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            float sum = 0f;
                            float sumSquared = 0f;
                            for (int k = 0; k < shape[2]; k++)
                            {
                                sum += ndarray[i, j, k, l];
                                sumSquared += ndarray[i, j, k, l] * ndarray[i, j, k, l];
                            }
                            varT[i, j, 0, l] = (sumSquared - (sum * sum) / shape[2]) /
                                                    (shape[2] - correction);
                        }
                    }
                }
            }
            else if (axis == 3)
            {
                varT = Zeros(shape[0], shape[1], shape[2], 1);
                for (int k = 0; k < shape[2]; k++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        for (int i = 0; i < shape[0]; i++)
                        {
                            float sum = 0f;
                            float sumSquared = 0f;
                            for (int l = 0; l < shape[3]; l++)
                            {
                                sum += ndarray[i, j, k, l];
                                sumSquared += ndarray[i, j, k, l] * ndarray[i, j, k, l];
                            }
                            varT[i, j, k, 0] = (sumSquared - (sum * sum) / shape[3]) /
                                                    (shape[3] - correction);
                        }
                    }
                }
            }
            else
            {
                throw new Exception("Available axis for ndarray are 0, 1, 2, and 3.");
            }

            return varT;
        }


        // Math operations
        public static NDArray Exp(NDArray ndarray)
        {
            NDArray result = new NDArray(ndarray.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Exp(ndarray.data[i]);

            }

            return result;
        }
        public static NDArray Pow(NDArray @base, float power)
        {
            NDArray result = new NDArray(@base.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Pow(@base.data[i], power);

            }

            return result;
        }
        public static NDArray Sqrt(NDArray @base)
        {
            NDArray result = new NDArray(@base.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Sqrt(@base.data[i]);
            }

            return result;
        }
        public static NDArray Abs(NDArray ndarray)
        {
            NDArray result = new NDArray(ndarray.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Abs(ndarray.data[i]);
            }

            return result;
        }
        public static NDArray Cos(NDArray ndarray)
        {
            NDArray result = new NDArray(ndarray.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Cos(ndarray.data[i]);

            }

            return result;
        }
        public static NDArray Sin(NDArray ndarray)
        {
            NDArray result = new NDArray(ndarray.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Sin(ndarray.data[i]);

            }

            return result;
        }
        public static NDArray Clip(NDArray ndarray, float min, float max)
        {
            NDArray result = new NDArray(ndarray.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = Math.Clamp(ndarray.data[i], min, max);
            }

            return result;
        }
        public static NDArray Minimum(NDArray ndarray1, NDArray ndarray2)
        {
            NDArray result = new NDArray(ndarray1.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Min(ndarray1.data[i], ndarray2.data[i]);
            }

            return result;
        }
        public static NDArray Maximum(NDArray ndarray1, NDArray ndarray2)
        {
            NDArray result = new NDArray(ndarray1.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Max(ndarray1.data[i], ndarray2.data[i]);
            }

            return result;
        }
        
        /// <summary>
        /// Returns the norm of all ndarray's values.
        /// </summary>
        /// <returns>NDArray with shape [1]</returns>
        public static NDArray Norm(NDArray ndarray, NormType normType = NormType.ManhattanL1)
        {
            switch (normType)
            {
                case NormType.ManhattanL1:
                    float abssum = ndarray.data.Sum(x => MathF.Abs(x));
                    return Constant(abssum);
                case NormType.EuclideanL2:
                    float sum = ndarray.data.Sum();
                    return Constant(MathF.Sqrt(sum));
                case NormType.Frobenius:
                    float sqrsum = ndarray.data.Sum(x => x * x);
                    return Constant(MathF.Sqrt(sqrsum));
                default:
                    throw new Exception("Unhandled norm type.");
            }
        }
        /// <summary>
        /// Returns the minimum element from all ndarray's values.
        /// </summary>
        /// <returns>NDArray with shape [1]</returns>
        public static NDArray Min(NDArray ndarray)
        {
            float min = ndarray.data.Min();
            return Constant(min);
        }
        /// <summary>
        /// Returns the maximum element from all ndarray's values.
        /// </summary>
        /// <returns>NDArray with shape [1]</returns>
        public static NDArray Max(NDArray ndarray)
        {
            float max = ndarray.data.Max();
            return Constant(max);
        }


        // LINQ (Not applied in autograd system)
        /// <summary>
        /// This ndarray values = function(this ndarray values).
        /// </summary>
        public void ForEach(Func<float, float> function, bool multithreaded = false)
        {
            if (multithreaded)
                Parallel.For(0, data.Length, i =>
                {
                    data[i] = function(data[i]);
                });
            else
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = function(data[i]);
                }
        }
        /// <summary>
        /// Returns a new NDArray where 'new ndarray values = selector(this ndarray values)'
        /// </summary>
        public NDArray Select(Func<float, float> selector, bool multithreaded = false)
        {
            NDArray result = new NDArray(shape);

            if (multithreaded)
                Parallel.For(0, data.Length, i =>
                {
                    result.data[i] = selector(data[i]);
                });
            else
                for (int i = 0; i < data.Length; i++)
                {
                    result.data[i] = selector(data[i]);
                }

            return result;
        }
        /// <summary>
        /// Returns a new NDArray where 'new ndarray values = resultSelector(1st ndarray values, 2nd ndarray values).
        /// </summary>
        public NDArray Zip(NDArray second, Func<float, float, float> resultSelector)
        {
            NDArray result = new NDArray(shape);

            for (int i = 0; i < data.Length; i++)
            {
                result.data[i] = resultSelector(data[i], second.data[i]);
            }

            return result;
        }
        public NDArray Count(Func<float, bool> selector = null)
        {
            if (selector == null)
                return Constant(shape[0] * shape[1] * shape[2] * shape[3]);

            int count = 0;

            for (int i = 0; i < data.Length; i++)
            {
                count += selector(data[i]) ? 1 : 0;
            }


            return Constant(count);
        }

        // System.Object/Collection
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
        public float[] Data
        {
            get => data;
            set => data = value;    
        }
        public string ShapeToString
        {
            get
            {
                StringBuilder sb = new StringBuilder();
                sb.Append('[');
                int rank = Rank;
                if (rank == 0 || rank == 1)
                {
                    sb.Append(shape[0]);
                }
                else if (rank == 2)
                {
                    sb.Append(shape[0]);
                    sb.Append(", ");
                    sb.Append(shape[1]);
                }
                else if (rank == 3)
                {
                    sb.Append(shape[0]);
                    sb.Append(", ");
                    sb.Append(shape[1]);
                    sb.Append(", ");
                    sb.Append(shape[2]);
                }
                else if (rank == 4)
                {
                    sb.Append(shape[0]);
                    sb.Append(", ");
                    sb.Append(shape[1]);
                    sb.Append(", ");
                    sb.Append(shape[2]);
                    sb.Append(", ");
                    sb.Append(shape[3]);
                }

                sb.Append(']');

                return sb.ToString();
            }
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
        public bool Equals(NDArray other)
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

            return Equals(obj as NDArray);
        }
        public override string ToString()
        {
            int rank = Rank;
            string format = "0.000000";

            StringBuilder sb = new StringBuilder();
            sb.Append("\n[");


            for (int l = 0; l < shape[3]; l++)
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

                for (int k = 0; k < shape[2]; k++)
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

                    for (int j = 0; j < shape[1]; j++)
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

                        for (int i = 0; i < shape[0]; i++)
                        {
                            if (i > 0)
                                sb.Append(", ");

                            sb.Append(this[i, j, k, l].ToString(format));
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
        private static void TreatNegativeAxis(ref NDArray ndarray, ref int axis)
        {
            if (axis >= 0)
                return;

            axis = ndarray.Rank + axis;
            ;
            if (axis < 0)
                throw new Exception("You are trying to access the last n-th dimension using negative axis, but the axis doesn't exist. Make sure you know the shape of your NDArray!");

        }  
    }
}