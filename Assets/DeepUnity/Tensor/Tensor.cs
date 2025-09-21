﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;
using System.Globalization;

namespace DeepUnity
{
    /// <summary>
    /// Any method like Reshape, Transpose etc. keeps the tensor continguous.
    /// </summary>
    [Serializable]
    public partial class Tensor : IEquatable<Tensor>, IEquatable<TensorGPU>, ICloneable
    {
        [ViewOnly, SerializeField] private float[] data;
        [ViewOnly, SerializeField] private int[] shape;

        public float[] Data => data;
        /// <summary>
        /// Read-Only
        /// </summary>
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
        /// <summary>
        /// Read-Only
        /// </summary>
        public int[] Shape
        {
            get => shape.ToArray();         
        }
        private int Width
        {
            get
            {
                return shape[shape.Length - 1];
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
            set => data[w] = value;
        }
        public float this[int h, int w]
        {
            get => data[h * Width + w];
            set => data[h * Width + w] = value;
        }
        public float this[int c, int h, int w]
        {
            get => data[c * Height * Width + h * Width + w];
            set => data[c * Height * Width + h * Width + w] = value;
        }
        public float this[int n, int c, int h, int w]
        {
            get => data[n * Channels * Height * Width + c * Height * Width + h * Width + w];
            set => data[n * Channels * Height * Width + c * Height * Width + h * Width + w] = value;

        }


        #region Create
        /// <summary>
        /// Default hidden <see cref="Tensor"/> constructor. Equivalent to <see cref="Zeros(int[])"/>.
        /// </summary>
        /// <param name="shape"></param>
        /// <exception cref="ShapeException"></exception>
        /// <exception cref="NotSupportedException"></exception>
        private Tensor(params int[] shape)
        {
            if (shape == null)
                throw new ShapeException($"Tensor cannot be instantiated with null shape, received shape ({shape.ToCommaSeparatedString()})");
            if (shape.Length == 0)
                throw new ShapeException($"Tensor cannot be instantiated with a shape of length 0, received shape ({shape.ToCommaSeparatedString()})");
            if (shape.Length > 4)
                throw new ShapeException($"Tensor cannot be instantiated with more than 4 dimensions, received shape ({shape.ToCommaSeparatedString()})");
            if (shape.Any(x => x < 1))
                throw new ShapeException($"Tensor cannot be instantiated with a dimension < 1, received shape ({shape.ToCommaSeparatedString()}).");

            int size = 1;
            foreach (var item in shape)
            {
                size *= item;
            }       

            if (size == int.MaxValue) // old 67_108_864
                throw new NotSupportedException($"Tensor dimensions is too large on initialization (cannot surpass {int.MaxValue} units, size attentded = {size}).");

            this.shape = shape.ToArray();
            data = new float[size];
        }
        /// <summary>
        /// Return a Tensor with the same shape and contents as <paramref name="other"/>. Better use Clone() instead.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public static Tensor Identity(Tensor other)
        {
            Tensor clone = new(other.shape);              
            Array.Copy(other.data, clone.data, other.data.Length);
            return clone;

        }
        /// <summary>
        /// Return a Tensor that lives in RAM with the values gained from a <see cref="TensorGPU"/>.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public static Tensor Identity(TensorGPU other)
        {
            Tensor clone = new(other.Shape);
            Array.Copy(other.ToArray(), clone.data, other.Count());
            return clone;
        }
        /// <summary>
        /// Returns a one-dimensional tensor of size <b>Ceil((end - start) / step)</b>, with the following elements:
        /// </br>[start, start + step, start + 2*step, ... ]
        /// </summary>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <param name="step"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor Arange(float start, float end, float step = 1)
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
        /// <summary>
        /// Creates a one-dimensional vector of shape (steps) whose values are evenly spaced from start to end, inclusive.
        /// </summary>
        /// <returns></returns>
        public static Tensor LinSpace(float start, float end, int steps)
        {
            if (steps <= 0)
                throw new ArgumentException("Steps must be more than 0");

            Tensor linspace = new(steps);
            float step_size = (end - start) / (steps -1);

            for (int i = 0; i < steps; i++)
            {
                linspace[i] = start + i * step_size;
            }

            return linspace;
        }
        /// <summary>
        /// Creates a one-dimensional vector of shape (steps) whose values are evenly spaced from start to end, inclusive. on a logarithmic scale with base <paramref name="base"/>.
        /// </summary>
        /// <returns></returns>
        public static Tensor LogSpace(float start, float end, int steps, float @base = 10f)
        {
            if (steps <= 0)
                throw new ArgumentException("Steps must be more than 0");

            Tensor logspace = LinSpace(MathF.Log(start, @base), MathF.Log(end, @base), steps);
            
            for (int i = 0; i < steps; i++)
            {
                logspace[i] = MathF.Pow(@base, logspace[i]);
            }

            return logspace;
        }
        /// <summary>
        /// Creates a Tensor using predefined data. 
        /// </summary>
        /// <param name="array"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static Tensor FromArray(float[] array, params int[] shape)
        {
            Tensor t = new(shape);
            Array.Copy(array, t.data, array.Length);
            return t;
        }
        /// <summary>
        /// Creates a predefined tensor of shape (1).
        /// </summary>
        public static Tensor Constant(float scalar)
        {
            Tensor t = new(1);
            t.data[0] = scalar;
            return t;
        }
        /// <summary>
        /// Creates a predefined one-dimensional tensor.
        /// </summary>
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
        /// <summary>
        /// Creates a predefined two-dimensional tensor.
        /// </summary>
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
        /// <summary>
        /// Creates a predefined cube-dimensional tensor.
        /// </summary>
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
                        t[z, y, x] = cube[z, y, x];
                    }
                }
            }

            return t;
        }
        /// <summary>
        /// Creates a predefined four-dimensional tensor.
        /// </summary>
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
        /// <summary>
        /// Output: (C, H, W) <br></br>
        /// where <br></br>
        /// C = <paramref name="shape"/>.Item1 (1 = Grayscale, 2 = Grayscale with Alpha, 3 = RGB, 4 = RGBA)<br></br>
        /// H = <paramref name="shape"/>.Item2  <br></br>
        /// W = <paramref name="shape"/>.Item3  <br></br>
        /// <em>Note: Tensors are displayed as the image mirrored on vertical axis when visualized as strings.</em>
        /// </summary>
        /// <param name="shape">the shape of the resulting tensor</param>
        /// <returns></returns>
        public static Tensor Constant(Color[] pixels, (int,int,int) shape)
        {
            int width = shape.Item3;
            int height = shape.Item2;
            int channels = shape.Item1;

            Tensor result = Zeros(channels, height, width);
           
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    if(channels == 1)
                    {
                        Color col = pixels[j * width + i];
                        result[0, j, i] = (col.r + col.g + col.b) / 3f;
                    }
                    else if(channels == 2)
                    {
                        Color col = pixels[j * width + i];
                        result[0, j, i] = (col.r + col.g + col.b) / 3f;
                        result[1, j, i] = col.a;
                    }
                    else if(channels == 3)
                    {
                        result[0, j, i] = pixels[j * width + i].r;
                        result[1, j, i] = pixels[j * width + i].g;
                        result[2, j, i] = pixels[j * width + i].b;

                    }
                    else if(channels == 4)
                    {

                        result[0, j, i] = pixels[j * width + i].r;
                        result[1, j, i] = pixels[j * width + i].g;
                        result[2, j, i] = pixels[j * width + i].b;
                        result[3, j, i] = pixels[j * width + i].a;
                    }                   
                }
            }
           
            return result;
        }
        /// <summary>
        /// <b>Extracts the data from a ComputeBuffer into a tensor.</b> <br></br>
        /// Output: (L) <br></br>
        /// where L = computeBuffer.count
        /// </summary>
        /// <param name="computeBuffer"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor Constant(ComputeBuffer computeBuffer, params int[] shape)
        {
            if (computeBuffer.stride != sizeof(float))
                throw new ArgumentException("ComputeBuffer stride must be equal to sizeof(float).");

            if (shape == null)
                throw new ArgumentException("Shape of the tensor was not defined. If unknown, use <b>computeBuffer.count</b>");
            Tensor result = Zeros(shape);
            computeBuffer.GetData(result.data);
                
            return result;
        }
        /// <summary>
        /// Creates a 1D tensor with the values from the given <paramref name="collection"/>.
        /// </summary>
        /// <param name="collection"></param>
        /// <returns></returns>
        public static Tensor Constant(IEnumerable<float> collection)
        {
            int length = collection.Count();
            Tensor outs = Tensor.Zeros(length);
            int index = 0;
            foreach (var item in collection)
            {
                outs[index++] = item;
            }
            return outs;
        }
        /// <summary>
        /// Returns a tensor filled with zeros with the given <paramref name="shape"/>.
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static Tensor Zeros(params int[] shape)
        {
            return new(shape);
        }
        /// <summary>
        /// Returns a tensor filled with ones with the given <paramref name="shape"/>.
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static Tensor Ones(params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = 1f;
            }
            return t;
        }
        /// <summary>
        ///  Returns a tensor filled with the value given as argument.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static Tensor Fill(float value, params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = value;
            }
            return t;
        }
        /// <summary>
        /// Returns a tensor of random real numbers drawn from separate uniform distributions whose bounds are [0, 1] (inclusive range).
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static Tensor Random01(params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = Utils.Random.ValueUnsafe;
            }
            return t;
        }
        /// <summary>
        /// Returns a tensor of random numbers drawn from separate normal distributions whose mean = 0 and standard-deviation = 1.
        /// </summary>
        /// <param name="mean_sd"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static Tensor RandomNormal(params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = Utils.Random.Normal(threadsafe: false);
            }
            return t;
        }
        /// <summary>
        /// Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
        /// </summary>
        /// <param name="mean_sd">mean and standard deviation</param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static Tensor RandomNormal((float, float) mean_sd, params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = Utils.Random.Normal(mean_sd.Item1, mean_sd.Item2, threadsafe:false);
            }
            return t;
        }
        /// <summary>
        /// Returns a tensor of random real numbers drawn from separate uniform distributions whose bounds are given as arguments. (Min inclusive, Max inclusive)
        /// </summary>
        /// <param name="min_max"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static Tensor RandomRange((float, float) min_max, params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = Utils.Random.Range(min_max.Item1, min_max.Item2);
            }
            return t;
        }
        /// <summary>
        /// Returns a tensor of random integers drawn from separate uniform distributions whose bounds are given as arguments. (Min inclusive, Max exclusive)
        /// </summary>
        /// <param name="min_max"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static Tensor RandomRangeInt((int, int) min_max, params int[] shape)
        {
            Tensor t = new(shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = Utils.Random.Range(min_max.Item1, min_max.Item2);
            }
            return t;
        }
        /// <summary>
        /// Returns a tensor with 0 and 1 elements drawn from a bernoulli distribution parametrized by the input tensor.
        /// </summary>
        /// <param name="probabilities"></param>
        /// <returns></returns>
        public static Tensor Bernoulli(Tensor probabilities)
        {
            Tensor t = new(probabilities.shape);
            for (int i = 0; i < t.data.Length; i++)
            {
                t.data[i] = Utils.Random.Bernoulli(probabilities.data[i]) == true ? 1f : 0f;
            }
            return t;
        }
        /// <summary>
        /// Creates a 2D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        /// <param name="n">The size of the square matrix</param>
        /// <returns></returns>
        public static Tensor Eye(int n)
        {
            Tensor t = new(n, n);
            for (int i = 0; i < n; i++)
            {
                t[i, i] = 1;
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
                result.data[i] = +tensor.data[i];
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

        /// <summary>
        /// Elements addition by <paramref name="right"/> value.
        /// </summary>
        public static Tensor operator +(Tensor left, float right)
        {
            Tensor result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] += right;
            }

            return result;
        }
        /// <summary>
        /// Elements subtraction by <paramref name="right"/> value.
        /// </summary>
        public static Tensor operator -(Tensor left, float right)
        {
            Tensor result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] -= right;
            }

            return result;
        }
        /// <summary>
        /// Elements multiplication by <paramref name="right"/> value.
        /// </summary>
        public static Tensor operator *(Tensor left, float right)
        {
            Tensor result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] *= right;
            }

            return result;
        }
        /// <summary>
        /// Elements division by <paramref name="right"/> value.
        /// </summary>
        public static Tensor operator /(Tensor left, float right)
        {
            Tensor result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] /= right;
            }

            return result;
        }
        

        /// <summary>
        /// Elements addition by <paramref name="left"/> value.
        /// </summary>
        public static Tensor operator +(float left, Tensor right) => right + left;
        /// <summary>
        /// Elements subtraction by <paramref name="right"/> value.
        /// </summary>
        public static Tensor operator -(float left, Tensor right)
        {
            Tensor result = Identity(right);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left - right.data[i];
            }

            return result;
        }
        /// <summary>
        /// Elements multiplication by <paramref name="left"/> value.
        /// </summary>
        public static Tensor operator *(float left, Tensor right) => right * left;
        /// <summary>
        /// Elements subtraction by <paramref name="right"/> value.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        public static Tensor operator /(float left, Tensor right)
        {
            Tensor result = Identity(right);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left / right.data[i];
            }

            return result;
        }


        /// <summary>
        /// Elements multiplication by <paramref name="left"/> value.
        /// </summary>
        /// <summary>
        /// Element-wise addition.
        /// </summary>
        public static Tensor operator +(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left({left.shape.ToCommaSeparatedString()}) and Right({right.shape.ToCommaSeparatedString()}) tensors must have similar shape for Element-wise addition (+).");

            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] + right.data[i];
            }


            return result;
        }
        /// <summary>
        /// Element-wise subtraction.
        /// </summary>
        public static Tensor operator -(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left({left.shape.ToCommaSeparatedString()}) and Right({right.shape.ToCommaSeparatedString()}) tensors must have similar shape for Element-wise subtraction (-).");

            Tensor result = new(left.shape);
            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] - right.data[i];
            }

            return result;
        }
        /// <summary>
        /// Element-wise (Hadamard) multiplication.
        /// </summary>
        public static Tensor operator *(Tensor left, Tensor right)
        {
           if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left({left.shape.ToCommaSeparatedString()}) and Right({right.shape.ToCommaSeparatedString()}) tensors must have similar shape for Element-wise multiplication (*).");

            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] * right.data[i];
            }


            return result;
        }
        /// <summary>
        /// Element-wise division.
        /// </summary>
        public static Tensor operator /(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left({left.shape.ToCommaSeparatedString()}) and Right({right.shape.ToCommaSeparatedString()}) tensors must have similar shape for Element-wise division (/).");

            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] / right.data[i];
            }

            return result;
        }
        /// <summary>
        /// Element-wise modulo operator.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        public static Tensor operator %(Tensor left, int right)
        {
            Tensor result = Identity(left);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] %= right;
            }

            return result;
        }
        /// <summary>
        /// Element-wise greater than.
        /// </summary>
        /// <param name="obj1"></param>
        /// <param name="obj2"></param>
        /// <returns></returns>
        public static Tensor operator >(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left({left.shape.ToCommaSeparatedString()}) and Right({right.shape.ToCommaSeparatedString()}) tensors must have similar shape for Element-wise comparison (>).");

            Tensor result = new(left.shape);
            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] > right.data[i] ? 1f : 0f;
            }

            return result;
        }
        /// <summary>
        /// Element-wise less than.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        /// <exception cref="OperationCanceledException"></exception>
        public static Tensor operator <(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left({left.shape.ToCommaSeparatedString()}) and Right({right.shape.ToCommaSeparatedString()}) tensors must have similar shape for Element-wise comparison (<).");

            Tensor result = new(left.shape);
            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] < right.data[i] ? 1f : 0f;
            }

            return result;
        }
        /// <summary>
        /// Element-wise greater or equal than.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        /// <exception cref="OperationCanceledException"></exception>
        public static Tensor operator >=(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left({left.shape.ToCommaSeparatedString()}) and Right({right.shape.ToCommaSeparatedString()}) tensors must have similar shape for Element-wise comparison (>=).");

            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] >= right.data[i] ? 1f : 0f;
            }

            return result;
        }
        /// <summary>
        /// Element-wise less or equal than.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        /// <exception cref="OperationCanceledException"></exception>
        public static Tensor operator <=(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left({left.shape.ToCommaSeparatedString()}) and Right({right.shape.ToCommaSeparatedString()}) tensors must have similar shape for Element-wise comparison (<=).");

            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = left.data[i] <= right.data[i] ? 1f : 0f;
            }

            return result;
        }
        #endregion


        #region Special
        /// <summary>
        /// Copies all data from <paramref name="fromTensor"/> and assignes them to <paramref name="toTensor"/>, along with the shape.
        /// </summary>
        /// <param name="fromTensor"></param>
        /// <param name="toTensor"></param>
        public static void CopyTo(Tensor fromTensor, Tensor toTensor)
        {
            toTensor.data = fromTensor.data.ToArray();
            toTensor.shape = fromTensor.shape.ToArray();
        }
        /// <summary>
        /// Computes the matrix product of two tensors. <br></br><br></br>
        /// Left: <b>(J, 1, N, M)</b> <br></br>
        /// Right: <b>(K, M, P)</b> <br></br>
        /// <br></br>
        /// <em>If device == GPU, the tensors are loaded on VRAM of the GPU, operations are done there, and the result is retrieved back to RAM.</em>
        /// </summary>
        /// <returns>Output: <b>(J, K, N, P)</b></returns>
        public static Tensor MatMul(Tensor left, Tensor right, Device device = Device.CPU)
        {
            if(device == Device.CPU)
            {
                int left_rank = left.Rank;
                int right_rank = right.Rank;

                if (left_rank == 1 && right_rank == 1)
                    throw new ArgumentException($"At least one of the tensors must have a shape > 1 for matrix multiplication");

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


                if (right_rank == 1)
                {
                    Parallel.For(0, N, n =>
                    {
                        for (int j = 0; j < J; j++)
                        {
                            for (int k = 0; k < K; k++)
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
                    });

                }
                else if (left_rank == 1)
                {
                    Parallel.For(0, P, p =>
                    {
                        for (int j = 0; j < J; j++)
                        {
                            for (int k = 0; k < K; k++)
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
                    });
                }
                else
                {
                    // note: starting with 64x64 matmul, GPU based multiplication becomes better
                    // case non-batched and non-channeled matmul
                    if (J == 1 && K == 1)
                    {
                        if (N > P)
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
                        else
                            Parallel.For(0, P, p =>
                            {
                                for (int n = 0; n < N; n++)
                                {
                                    float sum = 0f;
                                    for (int m = 0; m < M; m++)
                                    {
                                        sum += left[n, m] * right[m, p];
                                    }
                                    result[n, p] = sum;
                                }
                            });

                    }
                    else if (J > 1)
                        Parallel.For(0, J, j =>
                        {
                            for (int k = 0; k < K; k++)
                            {
                                for (int n = 0; n < N; n++)
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
                    else // if (K > 1) no batch case
                        Parallel.For(0, K, k =>
                        {
                            for (int n = 0; n < N; n++)
                            {
                                for (int p = 0; p < P; p++)
                                {
                                    float sum = 0f;
                                    for (int m = 0; m < M; m++)
                                    {
                                        sum += left[0, n, m] * right[k, m, p];
                                    }
                                    result[k, n, p] = sum;
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
            else
            {             
                int left_rank = left.Rank;
                int right_rank = right.Rank;

                // Special cases .. that will maybe be forbidden in the future
                if (left_rank == 1 && right_rank == 1)
                    throw new ArgumentException($"At least one of the tensors must have the Rank > 1 for matrix multiplication (both inputs have rank 1)");

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

                if (left_rank == 1)
                {
                    cs.SetInt("wr", P);
                    cs.SetInt("hr", 1);
                }
                else if (right_rank == 1)
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
        }
        /// <summary>
        /// Performs a batched matrix-matrix product. <br></br><br></br>
        /// Left: <b>(B, N, M)</b> or <b>(N, M)</b> for unbatched inputs<br></br>
        /// Right: <b>(B, M, P)</b> or <b>(M, P)</b> for unbatched inputs<br></br>
        /// <br></br>
        /// <em>If device == GPU, the tensors are loaded on VRAM of the GPU, operations are done there, and the result is retrieved back to RAM.</em>
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="device"></param>
        /// <returns>Output: <b>(B, N, P)</b> or <b>(N, P)</b> for unbatched inputs</returns>
        public static Tensor BatchedMatMul(Tensor left, Tensor right, Device device = Device.CPU)
        {
            if (left.Width != right.Height)
                throw new ArgumentException($"Tensors must have compatible shapes for batched matrix multiplication (Left[{left.Shape.ToCommaSeparatedString()}] doesn't match Right[{right.Shape.ToCommaSeparatedString()}]).");

            if (left.Channels != right.Channels)
                throw new ArgumentException($"Tensors must have compatible shapes for batched matrix multiplication (Left[{left.Shape.ToCommaSeparatedString()}] doesn't match Right[{right.Shape.ToCommaSeparatedString()}]).");

            if (left.Rank > 3 || right.Rank > 3)
                throw new ArgumentException($"Maximum allowed rank is 3 (received left={left.Rank}, right={right.Rank})");

            if (left.Rank < 2 || right.Rank < 2)
                throw new ArgumentException($"Minimum allowed rank is 2 (received left={left.Rank}, right={right.Rank})");

            int C = left.Channels;
            int N = left.Height;
            int M = left.Width;
            int P = right.Width;

            Tensor result = new Tensor(CreateShape(left.Rank, 1, C, N, P));

            if (device == Device.CPU)
            {               
                Parallel.For(0, C, c =>
                {
                    for (int n = 0; n < N; n++)
                    {
                        for (int p = 0; p < P; p++)
                        {
                            float sum = 0f;
                            for (int m = 0; m < M; m++)
                            {
                                sum += left[c, n, m] * right[c, m, p];
                            }
                            result[c, n, p] = sum;
                        }
                    }
                });
            }
            else
            {
                ComputeShader cs = DeepUnityMeta.TensorCS;

                ComputeBuffer leftData = new(left.data.Length, 4);
                ComputeBuffer rightData = new(right.data.Length, 4);
                ComputeBuffer resultData = new(C * N * P, 4);

                leftData.SetData(left.data);
                rightData.SetData(right.data);

                int kernel = cs.FindKernel("BatchedMatMul");

                cs.SetBuffer(kernel, "data1", leftData);
                cs.SetBuffer(kernel, "data2", rightData);
                cs.SetBuffer(kernel, "result", resultData);

                cs.SetInt("w1", M);
                cs.SetInt("h1", N);
                cs.SetInt("c1", C);

                cs.SetInt("w2", P);
                cs.SetInt("h2", M);
                cs.SetInt("c2", C);

                cs.SetInt("wr", P);
                cs.SetInt("hr", N);
                cs.SetInt("cr", C);

                // channels c are the batch here

                cs.Dispatch(kernel,
                      (C + 7) / 8,
                      (N + 7) / 8,
                      (P + 7) / 8);

                resultData.GetData(result.data);

                leftData.Release();
                rightData.Release();
                resultData.Release();
            }

            return result;
        }
        /// <summary>
        /// Computes the dot product of two 1D tensors.
        /// </summary>
        /// <param name="input1"></param>
        /// <param name="input2"></param>
        /// <returns>Tensor of shape <b>(1)</b></returns>
        public static Tensor Dot(Tensor input1, Tensor input2)
        {
            int leftRank = input1.Rank;
            int rightRank = input2.Rank;
            if (leftRank != 1)
                throw new ArgumentException($"First tensor in the dot product must have the rank equal to 1 (received {leftRank}");

            if (rightRank != 1)
                throw new ArgumentException($"First tensor in the dot product must have the rank equal to 1 (received {rightRank}");

            float result = 0f;
            for (int i = 0; i < input1.data.Length; i++)
            {
                result += input1.data[i] * input2.data[i];
            }

            return Constant(result);
        }
        /// <summary>
        /// Pads the last two dimensions of the given matrix. <br></br>
        /// Input: <b>(B, C, H, W)</b> or <b>(C, H, W) or <b>(H, W)</b></b><br></br>
        ///
        /// where B = batch_size, C = channels, H = height, W = width.
        /// </summary>
        /// <returns> Output: <b>(B, C, H + P * 2, W + P * 2)</b> <br></br>
        /// where P = padding</returns>
        public static Tensor MatPad(Tensor tensor, int padding, PaddingType paddingMode)
        {
            if (padding == 0)
                return Identity(tensor);

            if(paddingMode == PaddingType.Zeros)
            {
                int w = tensor.Width + padding * 2;
                int h = tensor.Height + padding * 2;
                int chans = tensor.Channels;
                int n = tensor.Batch;
                Tensor result = new(CreateShape(tensor.Rank, n, chans, h, w));
                if (n > 1)
                    Parallel.For(0, n, l =>
                    {
                        for (int k = 0; k < chans; k++)
                        {
                            for (int j = 0; j < tensor.Height; j++)
                            {
                                for (int i = 0; i < tensor.Width; i++)
                                {
                                    result[l, k, j + padding, i + padding] = tensor[k, j, i];
                                }
                            }
                        }
                    });
                else if (chans > 1)
                    Parallel.For(0, chans, k =>
                    {
                        for (int j = 0; j < tensor.Height; j++)
                        {
                            for (int i = 0; i < tensor.Width; i++)
                            {
                                result[k, j + padding, i + padding] = tensor[k, j, i];
                            }
                        }
                    });
                else
                    for (int j = 0; j < tensor.Height; j++)
                    {
                        for (int i = 0; i < tensor.Width; i++)
                        {
                            result[j + padding, i + padding] = tensor[j, i];
                        }
                    }
                return result;
            }   
            else if (paddingMode == PaddingType.Mirror)
            {
                 int w = tensor.Width + 2;
                 int h = tensor.Height + 2;
                 int channels = tensor.Channels;
                 int batch_size = tensor.Batch;
                 Tensor result = new(CreateShape(tensor.Rank, batch_size, channels, h, w));

                for (int l = 0; l < batch_size; l++)
                {
                    for (int k = 0; k < channels; k++)
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
                for (int l = 0; l < batch_size; l++)
                 {
                     for (int k = 0; k < channels; k++)
                     {
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

                        result[l, k, h - 1, 0] = result[l, k, h - 2, 1];
                        result[l, k, h - 1, w - 1] = result[l, k, h - 2, w - 2];

                    }
                }
                 
                 return MatPad(result, padding - 1, PaddingType.Mirror);
            }
            throw new ArgumentException("Unhandled padding mode");          
        }
        /// <summary>
        /// Pads the last dimension of the given vector. <br></br>
        /// Input: <b>(B, C, H, W)</b><br></br>
        ///
        /// where B = batch_size, C = channels, H = height, W = width.
        /// </summary>
        /// <returns> Output: <b>(B, C, H, W + P * 2)</b> <br></br>
        /// where P = padding</returns>
        public static Tensor VecPad(Tensor tensor, int padding, PaddingType paddingMode)
        {
            // Note that here actualy there is no Width, only height, so the input is technically (B, C, H). Though anyways, it doesn t matter (just for the fact the variables are named according to the Tensor default dim naming).
            if (padding == 0)
                return Identity(tensor);

            if (paddingMode == PaddingType.Zeros)
            {
                int widthy = tensor.Width + padding * 2;
                int heighty = tensor.Height;
                int chans = tensor.Channels;
                int batch_size = tensor.Batch;
                Tensor result = new(CreateShape(tensor.Rank, batch_size, chans, heighty, widthy));
                if (batch_size > 1)
                    Parallel.For(0, batch_size, l =>
                    {
                        for (int k = 0; k < chans; k++)
                        {
                            for (int j = 0; j < heighty; j++)
                            {
                                for (int i = 0; i < widthy; i++)
                                {
                                    result[l, k, j, i + padding] = tensor[k, j, i];
                                }
                            }
                        }
                    });
                else if (chans > 1)
                    Parallel.For(0, chans, k =>
                    {
                        for (int j = 0; j < heighty; j++)
                        {
                            for (int i = 0; i < widthy; i++)
                            {
                                result[k, j, i + padding] = tensor[k, j, i];
                            }
                        }
                    });
                else
                    for (int j = 0; j < heighty; j++)
                    {
                        for (int i = 0; i < widthy; i++)
                        {
                            result[j, i + padding] = tensor[j, i];
                        }
                    }
                return result;
            }
            else if (paddingMode == PaddingType.Mirror)
            {
                int widthy = tensor.Width;
                int heighty = tensor.Height;
                int channels = tensor.Channels;
                int batch_size = tensor.Batch;
                Tensor result = new(CreateShape(tensor.Rank, batch_size, channels, heighty, widthy + 2 * padding));
                for (int l = 0; l < batch_size; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int h = 0; h < heighty; h++)
                        {
                            for (int w = 0; w < widthy; w++)
                            {
                                result[l, k, h, w + padding] = tensor[l, k, h, w];                          
                            }

                            for (int i = 0; i < padding; i++)
                            {
                                result[l, k, h, i] = tensor[l, k, h, 0];
                                result[l, k, h, padding + widthy + i] = tensor[l, k, h, widthy - 1];
                            }
                        }
                    }
                }
                return result;
            }
            throw new ArgumentException("Unhandled padding mode");
        }
        /// <summary>
        /// Performs 2D cross-correlation. Inputs must have the same number of dimensions.
        /// </summary>
        /// <param name="input1"></param>
        /// <param name="input2"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor Correlate2D(Tensor input1, Tensor input2, CorrelationMode mode)
        {
            if (input1.Rank != input2.Rank && input1.Rank != 2)
                throw new ArgumentException($"Input1({input1.shape.ToCommaSeparatedString()}) and input2({input2.shape.ToCommaSeparatedString()}) must have rank 2.");

            Tensor result = null;

            int h1 = input1.Height;
            int w1 = input1.Width;
            int h2 = input2.Height; 
            int w2 = input2.Width;
            if(mode == CorrelationMode.Valid)
            {
                int output_height = h1 - h2 + 1;
                int output_width = w1 - w2 + 1;
                result = Zeros(output_height, output_width);

                for (int h = 0; h < output_height; h++)
                {
                    for (int w = 0; w < output_width; w++)
                    {
                        float sum = 0f;

                        for (int j = 0; j < h2; j++)
                        {
                            for (int i = 0; i < w2; i++)
                            {
                                sum += input1[h + j, w + i] * input2[j, i];
                            }
                        }

                        result[h, w] = sum;
                    }
                }
            }
            else if(mode == CorrelationMode.Full)
            {
                int outputHeight = h1 + h2 - 1;
                int outputWidth =  w1 + w2 - 1;
                result = Zeros(outputHeight, outputWidth);

                for (int h = 0; h < outputHeight; h++)
                {
                    for (int w = 0; w < outputWidth; w++)
                    {
                        float sum = 0f;

                        for (int j = 0; j < h2; j++)
                        {
                            for (int i = 0; i < w2; i++)
                            {
                                int inputRow = h - j;
                                int inputCol = w - i;

                                if (inputRow >= 0 && inputRow < h1 && inputCol >= 0 && inputCol < w1)
                                {
                                    sum += input1[inputRow, inputCol] * input2[j, i];
                                }
                            }
                        }

                        result[h, w] = sum;
                    }
                }

            }
            else if(mode == CorrelationMode.Same)
            {
                int outputHeight = h1;
                int outputWidth = w1 ;

                int paddingHeight = (h2 - 1) / 2;
                int paddingWidth = (w2 - 1) / 2;

                result = Zeros(outputHeight, outputWidth);

                for (int h = 0; h < outputHeight; h++)
                {
                    for (int w = 0; w < outputWidth; w++)
                    {
                        float sum = 0f;

                        for (int j = 0; j < h2; j++)
                        {
                            for (int i = 0; i < w2; i++)
                            {
                                int inputRow = h + j - paddingHeight;
                                int inputCol = w + i - paddingWidth;

                                if (inputRow >= 0 && inputRow < h1 && inputCol >= 0 && inputCol < w1)
                                {
                                    sum += input1[inputRow, inputCol] * input2[j, i];
                                }
                            }
                        }

                        result[h, w] = sum;
                    }
                }
            }
            return result;
        }
        /// <summary>
        /// Performs 2D cross-correlation but input2 matrix is rotated by 180 degrees. Inputs must have the same number of dimensions.
        /// </summary>
        /// <param name="input1"></param>
        /// <param name="input2"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor Convolve2D(Tensor input1, Tensor input2, CorrelationMode mode)
        {
            if (input1.Rank != input2.Rank && input1.Rank != 2)
                throw new ArgumentException($"Input1({input1.shape.ToCommaSeparatedString()}) and input2({input2.shape.ToCommaSeparatedString()}) must have rank 2.");

            // Rotate input2 by 180d
            int height = input2.Height;
            int width = input2.Width;
            Tensor input2_rot180d = new(input2.shape);

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    input2_rot180d[height - h - 1, width - w - 1] = input2[h, w];
                }
            }

            return Correlate2D(input1, input2_rot180d, mode);
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
                throw new OperationCanceledException($"Left({left.shape.ToCommaSeparatedString()}) and right({right.shape.ToCommaSeparatedString()}) tensors must have different shape for Min operation.");

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
                throw new OperationCanceledException($"Left({left.shape.ToCommaSeparatedString()}) and right({right.shape.ToCommaSeparatedString()}) tensors must have different shape for Max operation.");



            Tensor result = new(left.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Max(left.data[i], right.data[i]);
            }

            return result;
        }
        /// <summary>
        /// Computes the Element-Wise equality.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        /// <exception cref="OperationCanceledException"></exception>
        public static Tensor Eq(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left({left.shape.ToCommaSeparatedString()}) and Right({right.shape.ToCommaSeparatedString()}) tensors must have similar shape.");

            Tensor result = Zeros(left.shape);

            for (int i = 0; i < left.data.Length; i++)
            {
                if (left.data[i] == right.data[i])
                    result.data[i] = 1;
            }
            return result;
        }
        /// <summary>
        /// Computes the Element-Wise non-equality.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        /// <exception cref="OperationCanceledException"></exception>
        public static Tensor Ne(Tensor left, Tensor right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left({left.shape.ToCommaSeparatedString()}) and Right({right.shape.ToCommaSeparatedString()}) tensors must have similar shape.");

            Tensor result = Zeros(left.shape);

            for (int i = 0; i < left.data.Length; i++)
            {
                if (left.data[i] != right.data[i])
                    result.data[i] = 1;
            }
            return result;
        }
        /// <summary>
        /// Computes the element-wise log probability density/mass function w.r.t the received distribution 
        /// N(<paramref name="mu"/>, <paramref name="sigma"/>) at <paramref name="value"/>.  <br></br>
        /// https://stats.stackexchange.com/questions/404191/what-is-the-log-of-the-pdf-for-a-normal-distribution
        /// </summary>
        /// <param name="value"></param>
        /// <param name="mu"></param>
        /// <param name="sigma"></param>
        /// <returns></returns>
        public static Tensor LogProbability(Tensor value, Tensor mu, Tensor sigma)
        {
            var elem1 = Log(sigma);
            var elem2 = 0.5f * MathF.Log(2f * MathF.PI);
            var elem3 = 0.5f * ((value - mu) / sigma).Pow(2f);
            return -elem1 - elem2 - elem3;
        }
        /// <summary>
        /// Computes the element-wise probability density/mass function on the received distribution
        /// N(<paramref name="mu"/>, <paramref name="sigma"/>) at <paramref name="value"/>.  <br></br>
        /// </summary>
        /// <param name="value"></param>
        /// <param name="mu"></param>
        /// <param name="sigma"></param>
        /// <returns></returns>
        public static Tensor Probability(Tensor value, Tensor mu, Tensor sigma)
        {
            Tensor elem1 = sigma * MathF.Sqrt(2f * MathF.PI);
            Tensor elem2 = -0.5f * ((value - mu) / sigma).Pow(2f);
            return elem2.Exp() / elem1;
        }
        /// <summary>
        /// Filters out NaN values by replacing them with the specified argument.
        /// </summary>
        /// <param name="nan_replacement">The value that replaces the NaN</param>
        /// <returns></returns>
        public static Tensor FilterNaN(Tensor tensor, float nan_replacement = 0f)
        {
            return tensor.Select(x =>
            {
                if (float.IsNaN(x))
                    return nan_replacement;
                return x;
            });
        }
        /// <summary>
        /// Computes the Discrete Fast Fourier Transform (DFFT) of the given vector.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor FFT(Tensor tensor)
        {
            if (!(tensor.Rank == 1 || tensor.Rank == 0))
                throw new ArgumentException("FFT works only for tensors with rank 0 or 1.");

            int n = tensor.Size(-1);

            if (n == 1)
                return Identity(tensor);

            Tensor odd = Zeros(n / 2);
            Tensor even = Zeros(n / 2);
            for (int i = 0; i < n/2; i++)
            {
                odd[i] = tensor[i * 2];
                even[i] = tensor[i * 2 + 1];
            }

            odd = FFT(odd);
            even = FFT(even);

            Tensor y = Zeros(n);
            for (int i = 0; i < n/2; i++)
            {
                float t = MathF.Cos(-2 * MathF.PI * i / n) + MathF.Sin(-2 * MathF.PI * i / n);
                y[i] = even[i] + t * odd[i];
                y[i + n / 2] = even[i] - t * odd[i];
            }
            return y;
        }
        /// <summary>
        /// Compute the Discrete Inversed Fast Fourier Transform (DIFFT) of the given vector.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor IFFT(Tensor tensor)
        {
            if (!(tensor.Rank == 1 || tensor.Rank == 0))
                throw new ArgumentException("IFFT works only for tensors with rank 0 or 1.");

            int n = tensor.Size(-1);

            if (n == 1)
                return Identity(tensor);

            Tensor odd = Zeros(n / 2);
            Tensor even = Zeros(n / 2);
            for (int i = 0; i < n / 2; i++)
            {
                odd[i] = tensor[i * 2 + 1];
                even[i] = tensor[i * 2];
            }

            Tensor ifftEven = IFFT(even);
            Tensor ifftOdd = IFFT(odd);

            Tensor y = Zeros(n);
            for (int i = 0; i < n / 2; i++)
            {
                float t = MathF.Cos(2f * MathF.PI * i / n) + MathF.Sin(2f * MathF.PI * i / n);
                y[i] = (ifftEven[i] + t * ifftOdd[i]) / n;
                y[i + n / 2] = (ifftEven[i] - t * ifftOdd[i]) / n;
            }
            return y;
        }

        #endregion Special


        #region Static operations
        /// <summary>
        /// Returns the dimension of the tensor along the specified axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static int Size(Tensor tensor, int axis)
        {
            HandleAxis(tensor, ref axis);
            return tensor.shape[axis];
        }
        /// <summary>
        /// Changes the shape of the tensor. Π(<paramref name="tensor"/>.shape) must be equal to Π(<paramref name="newShape"/>).
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="newShape"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor Reshape(Tensor tensor, params int[] newShape)
        {
            int count = 1;
            foreach (var item in newShape)
            {
                count *= item;
            }
            if (count != tensor.data.Length)
                throw new ArgumentException($"The shape ({tensor.shape.ToCommaSeparatedString()}) cannot be reshaped to ({newShape.ToCommaSeparatedString()}).");

            // if new shape is broader than the original shape
            Tensor result = new Tensor(newShape);
            Array.Copy(tensor.data, result.data, tensor.data.Length);
            return result;
        }
        /// <summary>
        /// Permutes the dimensions of the tensor according to the specified order.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="axes">The desired ordering of dimensions.</param>
        /// <returns>A new tensor with permuted dimensions.</returns>
        /// <exception cref="ArgumentException">Thrown if the number of dimensions does not match or dims contains invalid indices.</exception>
        public static Tensor Permute(Tensor tensor, params int[] axes)
        {
            if (axes.Length != tensor.shape.Length)
                throw new ArgumentException($"Number of dimensions to permute ({axes.Length}) must match the number of tensor dimensions ({tensor.shape.Length}).");

            for (int i = 0; i < axes.Length; i++)
            {
                HandleAxis(tensor, ref axes[i]);
            }

            // Check if dims contains valid indices
            bool[] seen = new bool[tensor.shape.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                if (axes[i] < 0 || axes[i] >= tensor.shape.Length || seen[axes[i]])
                    throw new ArgumentException($"Invalid dimension index in permutation array: {axes[i]}");
                seen[axes[i]] = true;
            }

            // Calculate the new shape
            int[] newShape = new int[tensor.shape.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                newShape[i] = tensor.shape[axes[i]];
            }

            Tensor result = new Tensor(newShape);

            // put the elem in the result
            int[] indices = new int[tensor.Shape.Length];
            for (int i = 0; i < tensor.data.Length; i++)
            {
                int flatIndex = i;
                // Convert the flat index to multidimensional indices
                for (int dim = tensor.Shape.Length - 1; dim >= 0; dim--)
                {
                    indices[dim] = flatIndex % tensor.Shape[dim];
                    flatIndex /= tensor.Shape[dim];
                }
                // Rearrange elements based on permutation
                int[] permutedIndices = new int[tensor.Shape.Length];
                for (int dim = 0; dim < axes.Length; dim++)
                {
                    permutedIndices[dim] = indices[axes[dim]];
                }
                // Convert the permuted indices to flat index for the result tensor
                int resultIndex = 0;
                int multiplier = 1;
                for (int dim = tensor.Shape.Length - 1; dim >= 0; dim--)
                {
                    resultIndex += permutedIndices[dim] * multiplier;
                    multiplier *= result.Shape[dim];
                }
                result.data[resultIndex] = tensor.data[i];
            }

            return result;
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
                if(newShape.Count > 1)
                    newShape.RemoveAt(ax);

                Tensor result = new(newShape.ToArray());
                Array.Copy(tensor.data, result.data, tensor.data.Length);
                return result;
            }

        }
        /// <summary>
        /// Creates a new dimension, higher than the provided axis. <br></br>
        /// Example: <br></br>
        /// Unsqueeze(tensor: (8,16), axis: 0) => output tensor: (1, 8, 16)
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static Tensor Unsqueeze(Tensor tensor, int axis)
        {
            HandleAxis(tensor, ref axis);


            List<int> unsqueezedShape = tensor.shape.ToList();
            unsqueezedShape.Insert(axis, 1);
            Tensor result = new(unsqueezedShape.ToArray());
            Array.Copy(tensor.data, result.data, tensor.data.Length);
            return result;
        }     
        /// <summary>
        /// Combines multiple dimensions into a single dimension. <br></br>
        /// Example: <br></br>
        /// Flatten(tensor: (10,4,3), startAxis: 1, endAxis: -1) => output tensor: (10,12)
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="startAxis"></param>
        /// <param name="endAxis"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static Tensor Flatten(Tensor tensor, int startAxis = 0, int endAxis = -1)
        {
            int orig_start_axis = startAxis;
            int orig_end_axis = endAxis;
            HandleAxis(tensor, ref startAxis);
            HandleAxis(tensor, ref endAxis);

            if (startAxis > endAxis)
                throw new Exception($"Start axis ({orig_start_axis}) must be smaller or equal to the end axis ({orig_end_axis}) when flattening.");


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
        ///  <br></br>
        /// If <b>axis == null</b>, all tensors are stacked on a new dimension (unsqueezed on axis 0 then concatenated on axis 0) <br></br>
        /// <br></br>Examples: <br></br>
        /// Cat(axis: 0,    tensors: {(2,3),(2,3),(2,3),(2,3)}) => output (8,3) <br></br>
        /// Cat(axis: 1,    tensors: {(2,3),(2,5),(2,4),(2,3)}) => output (2,14) <br></br>
        /// Car(axis: 1,    tensors: {(2,3)} => output (2,3) <br></br>
        /// Cat(axis: null, tensors: {(2,3),(2,3),(2,3),(2,3)}) => output (4,2,3) <br></br>
        /// Cat(axis: null, tensors: {(2,3)} => output (1,2,3) <br></br>
        /// <br></br>
        /// <em>All tensors must have the same dimensions (except the concatenation axis).</em>
        /// </summary>
        public static Tensor Concat(int? axis, params Tensor[] tensors)
        {
            if (tensors == null || tensors.Length == 0)
                throw new ArgumentException("At least one tensor must be provided for concatenation");

            if (tensors.Length == 1)
            {
                if (axis == null)
                    return tensors[0].Unsqueeze(0);
                else
                    return Identity(tensors[0]);
            }

            // Handle the null axis case (stacking on new dimension)
            if (axis == null)
            {
                // First, unsqueeze all tensors on axis 0 to add a new dimension
                Tensor[] unsqueezedTensors = new Tensor[tensors.Length];
                for (int i = 0; i < tensors.Length; i++)
                {
                    unsqueezedTensors[i] = tensors[i].Unsqueeze(0);
                }

                // Then concatenate along axis 0 (the new dimension)
                return Concat(0, unsqueezedTensors);
            }

            // Normal concatenation case
            int concatAxis = axis.Value;
            HandleAxis(tensors[0], ref concatAxis);
            Dim dim = AxisToDim(tensors[0], concatAxis);

            // Validate tensor compatibility - all dimensions except concatenation axis must match
            for (int i = 1; i < tensors.Length; i++)
            {
                if (tensors[i] == null)
                {
                    throw new ArgumentException($"Tensors argument contains a null tensor on index {i}.");
                }

                if (tensors[i].shape.Length != tensors[0].shape.Length)
                {
                    throw new ArgumentException($"All tensors must have the same number of dimensions (received shapes: {tensors.Select(x => $"({x.shape.ToCommaSeparatedString()})").ToCommaSeparatedString()})");
                }

                // Check all dimensions except the concatenation axis
                for (int d = 0; d < tensors[0].shape.Length; d++)
                {
                    if (d != concatAxis && tensors[i].shape[d] != tensors[0].shape[d])
                    {
                        throw new ArgumentException($"Tensors must have the same shape except along the concatenation axis. " +
                            $"Tensor 0 shape: [{tensors[0].shape.ToCommaSeparatedString()}], " +
                            $"Tensor {i} shape: [{tensors[i].shape.ToCommaSeparatedString()}]");
                    }
                }
            }

            // Calculate the total size along the concatenation axis
            int totalConcatSize = 0;
            for (int i = 0; i < tensors.Length; i++)
            {
                totalConcatSize += tensors[i].shape[concatAxis];
            }

            // Create the result shape
            int[] shapex = null;
            Tensor firstTensor = tensors[0];

            switch (dim)
            {
                case Dim.width:
                    shapex = CreateShape(firstTensor.Rank, firstTensor.Batch, firstTensor.Channels, firstTensor.Height, totalConcatSize);
                    break;
                case Dim.height:
                    shapex = CreateShape(firstTensor.Rank, firstTensor.Batch, firstTensor.Channels, totalConcatSize, firstTensor.Width);
                    break;
                case Dim.channel:
                    shapex = CreateShape(firstTensor.Rank, firstTensor.Batch, totalConcatSize, firstTensor.Height, firstTensor.Width);
                    break;
                case Dim.batch:
                    shapex = CreateShape(firstTensor.Rank, totalConcatSize, firstTensor.Channels, firstTensor.Height, firstTensor.Width);
                    break;
            }

            Tensor result = new(shapex);

            // Keep track of the offset along the concatenation axis
            int offset = 0;

            for (int s = 0; s < tensors.Length; s++)
            {
                Tensor currentTensor = tensors[s];

                for (int l = 0; l < currentTensor.Batch; l++)
                {
                    for (int k = 0; k < currentTensor.Channels; k++)
                    {
                        for (int j = 0; j < currentTensor.Height; j++)
                        {
                            for (int i = 0; i < currentTensor.Width; i++)
                            {
                                switch (dim)
                                {
                                    case Dim.width:
                                        result[l, k, j, offset + i] = currentTensor[l, k, j, i];
                                        break;
                                    case Dim.height:
                                        result[l, k, offset + j, i] = currentTensor[l, k, j, i];
                                        break;
                                    case Dim.channel:
                                        result[l, offset + k, j, i] = currentTensor[l, k, j, i];
                                        break;
                                    case Dim.batch:
                                        result[offset + l, k, j, i] = currentTensor[l, k, j, i];
                                        break;
                                }
                            }
                        }
                    }
                }

                // Update offset for next tensor
                offset += currentTensor.shape[concatAxis];
            }

            return result;
        }
        /// <summary>
        /// Expands the tensor along the specified axis. <br></br>
        /// Example: <br></br>
        /// Expand(tensor: (10, <b>4</b>, 12), axis: 1, times: 4) => output tensor: (10, <b>20</b>, 12)
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="times"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
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
        /// <summary>
        /// Transposes the tensor along the two specified axis. <br></br>
        /// Input: (B, C, H, W) <br></br>
        /// Output: (N, C, W, H) <br></br>
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis0"></param>
        /// <param name="axis1"></param>
        /// <returns></returns>
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
        /// <summary>
        /// Transposes a tensor matrix (Rank == 2). Expects input to be of Rank <= 2. Tensors of rank 0 or 1 are returned with no modifications, tensors of rank 2 are equivalent to Transpose(tensor, 0, 1)
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor T(Tensor tensor)
        {
            if (tensor.Rank == 0 || tensor.Rank == 1)
                return Identity(tensor);

            if (tensor.Rank > 2)
                throw new ArgumentException($"Input tensor has rank {tensor.Rank}, and the allowed tensor must be of Rank 0, 1 or 2.");

            return Transpose(tensor, 0, 1);
        }
        /// <summary>
        /// Splits the tensor into multiple tensors along the specified axis. The resulting tensors are having the same rank as the main tensor. <br></br>
        /// Examples: <br></br>
        /// tensor = (4, 8, 8), axis = 0, split_size = 1 => [(1, 8, 8), (1, 8, 8), (1, 8, 8), (1, 8, 8)] <br></br>
        /// tensor = (6, 2, 10), axis = 1, split_size = 1 => [(6, 1, 10), (6, 1, 10)] (no squeezed involved)<br></br>
        /// tensor = (16, 5, 8), axis = 1, split_size = 3 => [(16, 3, 8), (16, 2, 8)] (last can remain partial)<br></br>
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="split_size"></param>
        /// <returns></returns>
        public static Tensor[] Split(Tensor tensor, int axis, int split_size)
        {
            HandleAxis(tensor, ref axis);
            Dim dim = AxisToDim(tensor, axis);
            int[] stackShape = new int[] { tensor.Batch, tensor.Channels, tensor.Height, tensor.Width };
            List<Tensor> slices = new();
            int dimLength = stackShape[(int)dim];
            int dimPos = 0;

            if (split_size < 1)
                throw new ArgumentException("Number of chunks must be positive.");
            if (split_size > dimLength)
                throw new ArgumentException("Number of chunks exceeds dimension length.");


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
        /// <summary>
        /// Attempts to split a tensor into <paramref name="num_chunks"/>.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="num_chunks"></param>
        /// <returns></returns>
        public static Tensor[] Chunk(Tensor tensor, int axis, int num_chunks)
        {
            HandleAxis(tensor, ref axis);
            Dim dim = AxisToDim(tensor, axis);
            int[] stackShape = new int[] { tensor.Batch, tensor.Channels, tensor.Height, tensor.Width };
            List<Tensor> chunks_ = new();
            int dimLength = stackShape[(int)dim];

            if (num_chunks <= 0)
                throw new ArgumentException("Number of chunks must be positive.");
            if (num_chunks > dimLength)
                throw new ArgumentException("Number of chunks exceeds dimension length.");

            int chunk_dim = (int)Math.Ceiling((double)dimLength / num_chunks);
            int start = 0;
            for (int i = 0; i < num_chunks; i++)
            {
                int current_split_size = (dimLength - start) > chunk_dim ? chunk_dim : dimLength - start;
                int[] chunkShape = stackShape.ToArray();
                chunkShape[(int)dim] = current_split_size;
                Tensor slice = new(CreateShape(tensor.Rank, chunkShape[0], chunkShape[1], chunkShape[2], chunkShape[3]));
                
                for (int l = 0; l < slice.Batch; l++)
                {
                    for (int k = 0; k < slice.Channels; k++)
                    {
                        for (int j = 0; j < slice.Height; j++)
                        {
                            for (int m = 0; m < slice.Width; m++)
                            {
                                switch (dim)
                                {
                                    case Dim.width:
                                        slice[l, k, j, m] = tensor[l, k, j, start + m];
                                        break;
                                    case Dim.height:
                                        slice[l, k, j, m] = tensor[l, k, start + j, m];
                                        break;
                                    case Dim.channel:
                                        slice[l, k, j, m] = tensor[l, start + k, j, m];
                                        break;
                                    case Dim.batch:
                                        slice[l, k, j, m] = tensor[start + l, k, j, m];
                                        break;
                                }
                            }
                        }
                    }
                }

                chunks_.Add(slice);
                start += current_split_size;
            }

            return chunks_.ToArray();
        }
        /// <summary>
        /// Shifts the elements by `<paramref name="shifts"/>` times on the specified axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="shifts"></param>
        /// <returns></returns>
        public static Tensor Roll(Tensor tensor, int axis, int shifts)
        {
            HandleAxis(tensor, ref axis);

            if (shifts == 0)
                return Identity(tensor);
           
            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dimIndex = AxisToDim(tensor, axis);

            Tensor result = new(tensor.shape);
            int newL, newK, newJ, newI;
            shifts = -shifts;
            for (int l = 0; l < batch; l++)
            {
                for (int k = 0; k < channels; k++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            newL = l;
                            newK = k;
                            newJ = j;
                            newI = i;

                            switch (dimIndex)
                            {
                                case Dim.width:
                                    newI = (i + shifts) % width;
                                    if (newI < 0) newI += width;
                                    break;
                                case Dim.height:
                                    newJ = (j + shifts) % height;
                                    if (newJ < 0) newJ += height;
                                    break;
                                case Dim.channel:
                                    newK = (k + shifts) % channels;
                                    if (newK < 0) newK += channels;
                                    break;
                                case Dim.batch:
                                    newL = (l + shifts) % batch;
                                    if (newL < 0) newL += batch;
                                    break;
                            }

                            result[l, k, j, i] = tensor[newL, newK, newJ, newI];
                        }
                    }
                }
            }
            

            return result;
        }
        /// <summary>
        /// Shuffles the elements along the specified axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static Tensor Shuffle(Tensor tensor, int axis)
        {
            HandleAxis(tensor, ref axis);
            Tensor[] slices = Split(tensor, axis, 1);
            Utils.Shuffle(slices);
            return Concat(axis, slices);
        }

        public static (Tensor, Tensor) PairwiseShuffle(Tensor tensor1, Tensor tensor2, int axis)
        {
            if (tensor1.shape.SequenceEqual(tensor2.shape))
                throw new ArgumentException($"The tensors must have similar shapes in order to be shuffled pairwise! (({tensor1.shape.ToCommaSeparatedString()}) and ({tensor2.shape.ToCommaSeparatedString()}))");


            HandleAxis(tensor1, ref axis);

           
            Tensor[] slices1 = Split(tensor1, axis, 1);
            Tensor[] slices2 = Split(tensor1, axis, 1);

            Utils.PairwiseShuffle(slices1, slices2);
            return (Concat(axis, slices1), Concat(axis, slices2));
        }
        /// <summary>
        /// Computes the sum along the speficied axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepDim">If false, squeezes the tensor on <paramref name="axis"/>.</param>
        /// <returns></returns>
        public static Tensor Sum(Tensor tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dimIndex = AxisToDim(tensor, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
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
                            result[l, k, j, 0] = sum;
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
                            result[l, k, 0, i] = sum;
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
                            result[l, 0, j, i] = sum;
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
                            result[0, k, j, i] = sum;
                        }
                    }
                }
            }
            if (!keepDim)
                Squeeze_(result, axis);
            return result;
        }
        /// <summary>
        /// Computes the product along the speficied axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepDim">If false, squeezes the tensor on <paramref name="axis"/>.</param>
        /// <returns></returns>
        public static Tensor Prod(Tensor tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dimIndex = AxisToDim(tensor, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1; // keepDim = true? newShape[axis] : 1
            Tensor result = new(newShape);

            if (dimIndex == Dim.width)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int j = 0; j < height; j++)
                        {
                            float prod = 1f;
                            for (int i = 0; i < width; i++)
                            {
                                prod *= tensor[l, k, j, i];
                            }
                            result[l, k, j, 0] = prod;
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
                            float prod = 1f;
                            for (int j = 0; j < height; j++)
                            {
                                prod *= tensor[l, k, j, i];
                            }
                            result[l, k, 0, i] = prod;
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
                            float prod = 1f;

                            for (int k = 0; k < channels; k++)
                            {
                                prod *= tensor[l, k, j, i];
                            }
                            result[l, 0, j, i] = prod;
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
                            float prod = 1f;
                            for (int l = 0; l < batch; l++)
                            {
                                prod *= tensor[l, k, j, i];
                            }
                            result[0, k, j, i] = prod;
                        }
                    }
                }
            }
            if (!keepDim)
                Squeeze_(result, axis);
            return result;
        }
        /// <summary>
        /// Computes the mean along the speficied axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepDim">If false, squeezes the tensor on <paramref name="axis"/>.</param>
        /// <returns></returns>
        public static Tensor Mean(Tensor tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dim = AxisToDim(tensor, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
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
                            result[l, k, j, 0] = sum;
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
                            result[l, k, 0, i] = sum;
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
                            result[l, 0, j, i] = sum;
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
                            result[0, k, j, i] = sum;
                        }
                    }
                }
            }
            if(!keepDim)
                Squeeze_(result, axis);
            return result;
        }
        /// <summary>
        /// Computes the variance along the speficied axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="correction">Bessel's correction</param>
        /// <param name="keepDim">If false, squeezes the tensor on <paramref name="axis"/>.</param>
        /// <returns></returns>
        public static Tensor Var(Tensor tensor, int axis, int correction = 1, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dimIndex = AxisToDim(tensor, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
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
                            result[l, k, j, 0] = vr;
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
                            result[l, k, 0, i] = vr;
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
                            result[l, 0, j, i] = vr;
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
                            result[0, k, j, i] = vr;
                        }
                    }
                }
            }

            if(!keepDim)
                Squeeze_(result, axis);
            return result;
        }
        /// <summary>
        /// Computes the standard deviation along the speficied axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="correction">Bessel's correction</param>
        /// <param name="keepDim">If false, squeezes the tensor on <paramref name="axis"/>.</param>
        /// <returns></returns>
        public static Tensor Std(Tensor tensor, int axis, int correction = 1, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);
            return Sqrt(Var(tensor, axis, correction, keepDim));
        }
        /// <summary>
        /// Computes the min element along the speficied axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepDim">If false, squeezes the tensor on <paramref name="axis"/>.</param>
        /// <returns></returns>
        public static Tensor Min(Tensor tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dim = AxisToDim(tensor, axis);


            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
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
                            result[l, k, j, 0] = min;
                            // for (int i = 0; i < newShape[axis]; i++)
                            // {
                            //     result[l, k, j, i] = min;
                            // }
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
                            result[l, k, 0, i] = min;
                            // for (int j = 0; j < newShape[axis]; j++)
                            // {
                            //     result[l, k, j, i] = min;
                            // }
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
                            result[l, 0, j, i] = min;
                            // for (int k = 0; k < newShape[axis]; k++)
                            // {
                            //     result[l, k, j, i] = min;
                            // }
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
                            result[0, k, j, i] = min;
                            // for (int l = 0; l < newShape[axis]; l++)
                            // {
                            //     result[l, k, j, i] = min;
                            // }
                        }
                    }
                }
            }

            if (!keepDim)
                Squeeze_(result, axis);
            return result;
        }
        /// <summary>
        /// Computes the max element along the speficied axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepDim">If false, squeezes the tensor on <paramref name="axis"/>.</param>
        /// <returns></returns>
        public static Tensor Max(Tensor tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dim = AxisToDim(tensor, axis);


            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
            Tensor result = Zeros(newShape);

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
                            result[l, k, j, 0] = m;
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
                            result[l, k, 0, i] = m;
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
                            result[l, 0, j, i] = m;
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
                            result[0, k, j, i] = m;
                        }
                    }
                }
            }

            if (!keepDim)
                Squeeze_(result, axis);
            return result;

        }
        /// <summary>
        /// Returns the indices of the maximum value of all elements in the input tensor along the specified axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepDim"></param>
        /// <returns></returns>
        public static Tensor ArgMax(Tensor tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dim = AxisToDim(tensor, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
            Tensor result = Zeros(newShape); // Result tensor will store the indices as integers

            if (dim == Dim.width)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int j = 0; j < height; j++)
                        {
                            int maxIndex = 0;
                            float maxValue = float.MinValue;
                            for (int i = 0; i < width; i++)
                            {
                                if (tensor[l, k, j, i] > maxValue)
                                {
                                    maxValue = tensor[l, k, j, i];
                                    maxIndex = i;
                                }
                            }
                            result[l, k, j, 0] = maxIndex;
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
                            int maxIndex = 0;
                            float maxValue = float.MinValue;
                            for (int j = 0; j < height; j++)
                            {
                                if (tensor[l, k, j, i] > maxValue)
                                {
                                    maxValue = tensor[l, k, j, i];
                                    maxIndex = j;
                                }
                            }
                            result[l, k, 0, i] = maxIndex;
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
                            int maxIndex = 0;
                            float maxValue = float.MinValue;
                            for (int k = 0; k < channels; k++)
                            {
                                if (tensor[l, k, j, i] > maxValue)
                                {
                                    maxValue = tensor[l, k, j, i];
                                    maxIndex = k;
                                }
                            }
                            result[l, 0, j, i] = maxIndex;
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
                            int maxIndex = 0;
                            float maxValue = float.MinValue;
                            for (int l = 0; l < batch; l++)
                            {
                                if (tensor[l, k, j, i] > maxValue)
                                {
                                    maxValue = tensor[l, k, j, i];
                                    maxIndex = l;
                                }
                            }
                            result[0, k, j, i] = maxIndex;
                        }
                    }
                }
            }

            if (!keepDim)
                Squeeze_(result, axis);
            return result;
        }
        /// <summary>
        /// Returns the indices of the minimum value of all elements in the input tensor along the specified axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepDim"></param>
        /// <returns></returns>
        public static Tensor ArgMin(Tensor tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dim = AxisToDim(tensor, axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
            Tensor result = Zeros(newShape); // Result tensor will store the indices as integers

            if (dim == Dim.width)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int j = 0; j < height; j++)
                        {
                            int minIndex = 0;
                            float minValue = float.MaxValue;
                            for (int i = 0; i < width; i++)
                            {
                                if (tensor[l, k, j, i] < minValue)
                                {
                                    minValue = tensor[l, k, j, i];
                                    minIndex = i;
                                }
                            }
                            result[l, k, j, 0] = minIndex;
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
                            int minIndex = 0;
                            float minValue = float.MaxValue;
                            for (int j = 0; j < height; j++)
                            {
                                if (tensor[l, k, j, i] < minValue)
                                {
                                    minValue = tensor[l, k, j, i];
                                    minIndex = j;
                                }
                            }
                            result[l, k, 0, i] = minIndex;
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
                            int minIndex = 0;
                            float minValue = float.MaxValue;
                            for (int k = 0; k < channels; k++)
                            {
                                if (tensor[l, k, j, i] < minValue)
                                {
                                    minValue = tensor[l, k, j, i];
                                    minIndex = k;
                                }
                            }
                            result[l, 0, j, i] = minIndex;
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
                            int minIndex = 0;
                            float minValue = float.MaxValue;
                            for (int l = 0; l < batch; l++)
                            {
                                if (tensor[l, k, j, i] < minValue)
                                {
                                    minValue = tensor[l, k, j, i];
                                    minIndex = l;
                                }
                            }
                            result[0, k, j, i] = minIndex;
                        }
                    }
                }
            }

            if (!keepDim)
                Squeeze_(result, axis);
            return result;
        }
        /// <summary>
        /// Sorts the tensors elements along the specified axis.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static Tensor Sort(Tensor tensor, int axis, bool ascending = true)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dim = AxisToDim(tensor, axis);

            Tensor result = Identity(tensor);

            if (dim == Dim.width)
            {
                // Sorting along the width dimension
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int j = 0; j < height; j++)
                        {
                            // Bubble sort along the width dimension
                            for (int i = 0; i < width - 1; i++)
                            {
                                for (int m = 0; m < width - i - 1; m++)
                                {
                                    bool swapCondition = ascending ? (result[l, k, j, m] > result[l, k, j, m + 1]) : (result[l, k, j, m] < result[l, k, j, m + 1]);

                                    if (swapCondition)
                                    {
                                        // Swap elements if they are out of order
                                        float temp = result[l, k, j, m];
                                        result[l, k, j, m] = result[l, k, j, m + 1];
                                        result[l, k, j, m + 1] = temp;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.height)
            {
                // Sorting along the height dimension
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            // Bubble sort along the height dimension
                            for (int j = 0; j < height - 1; j++)
                            {
                                for (int m = 0; m < height - j - 1; m++)
                                {
                                    bool swapCondition = ascending ? (result[l, k, m, i] > result[l, k, m + 1, i]) : (result[l, k, m, i] < result[l, k, m + 1, i]);

                                    if (swapCondition)
                                    {
                                        // Swap elements if they are out of order
                                        float temp = result[l, k, m, i];
                                        result[l, k, m, i] = result[l, k, m + 1, i];
                                        result[l, k, m + 1, i] = temp;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.channel)
            {
                // Sorting along the channel dimension
                for (int l = 0; l < batch; l++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            // Bubble sort along the channel dimension
                            for (int k = 0; k < channels - 1; k++)
                            {
                                for (int m = 0; m < channels - k - 1; m++)
                                {
                                    bool swapCondition = ascending ? (result[l, m, j, i] > result[l, m + 1, j, i]) : (result[l, m, j, i] < result[l, m + 1, j, i]);

                                    if (swapCondition)
                                    {
                                        // Swap elements if they are out of order
                                        float temp = result[l, m, j, i];
                                        result[l, m, j, i] = result[l, m + 1, j, i];
                                        result[l, m + 1, j, i] = temp;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else if (dim == Dim.batch)
            {
                // Sorting along the batch dimension
                for (int k = 0; k < channels; k++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            // Bubble sort along the batch dimension
                            for (int l = 0; l < batch - 1; l++)
                            {
                                for (int m = 0; m < batch - l - 1; m++)
                                {
                                    bool swapCondition = ascending ? (result[m, k, j, i] > result[m + 1, k, j, i]) : (result[m, k, j, i] < result[m + 1, k, j, i]);

                                    if (swapCondition)
                                    {
                                        // Swap elements if they are out of order
                                        float temp = result[m, k, j, i];
                                        result[m, k, j, i] = result[m + 1, k, j, i];
                                        result[m + 1, k, j, i] = temp;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return result;
        }
        /// <summary>
        /// Computes the cumulative sum along the specified axis for each element.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static Tensor CumSum(Tensor tensor, int axis)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dimIndex = AxisToDim(tensor, axis);

            Tensor result = Zeros(tensor.shape);

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
                                result[l, k, j, i] = sum;
                            }
                        }
                    }
                }
            }
            return result;
        }
        /// <summary>
        /// Computes the cumulative product along the specified axis for each element.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static Tensor CumProd(Tensor tensor, int axis)
        {
            HandleAxis(tensor, ref axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            Dim dimIndex = AxisToDim(tensor, axis);

            Tensor result = Zeros(tensor.shape);

            if (dimIndex == Dim.width)
            {
                for (int l = 0; l < batch; l++)
                {
                    for (int k = 0; k < channels; k++)
                    {
                        for (int j = 0; j < height; j++)
                        {
                            float prod = 1f;
                            for (int i = 0; i < width; i++)
                            {
                                prod *= tensor[l, k, j, i];
                                result[l, k, j, i] = prod;
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
                            float prod = 1f;
                            for (int j = 0; j < height; j++)
                            {
                                prod *= tensor[l, k, j, i];
                                result[l, k, j, i] = prod;
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
                            float prod = 1f;
                            for (int k = 0; k < channels; k++)
                            {
                                prod *= tensor[l, k, j, i];
                                result[l, k, j, i] = prod;
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
                            float prod = 1f;
                            for (int l = 0; l < batch; l++)
                            {
                                prod *= tensor[l, k, j, i];
                                result[l, k, j, i] = prod;
                            }
                        }
                    }
                }
            }
            return result;
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
        /// <summary>
        /// Returns the given tensor raised to the 2nd power.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor Square(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Pow(tensor.data[i], 2f);
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
        /// <summary>
        /// Returns the reciprocal of square root of x, which is 1 / sqrt(x)
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor RSqrt(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = 1f / MathF.Sqrt(tensor.data[i]);
            }

            return result;
        }
        /// <summary>
        /// Returns the reciprocal of the tensor elements, which is 1 / x.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor Reciprocal(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = 1f / tensor.data[i];
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
        public static Tensor ArcSin(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Asin(tensor.data[i]);
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
        public static Tensor ArcCos(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Acos(tensor.data[i]);
            }

            return result;
        }
        public static Tensor Tan(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Tan(tensor.data[i]);
            }

            return result;
        }
        public static Tensor ArcTan(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Atan(tensor.data[i]);
            }

            return result;
        }
        /// <summary>
        /// Returns the signs of each value in the tensor. <br></br>
        /// Example: t = [-1.2, 3.2, 0, 1]
        /// sign(t) = [-1, 1, 0, 1]
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor Sign(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Sign(tensor.data[i]);
            }

            return result;
        }
        /// <summary>
        /// Clamps all values of the tensor within the range (<paramref name="min"/>, <paramref name="max"/>)
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
        public static Tensor Ceil(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Ceiling(tensor.data[i]);
            }

            return result;
        }
        public static Tensor Floor(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Floor(tensor.data[i]);
            }

            return result;
        }
        /// <summary>
        /// Computes the element-wise logical NOT of the given input tensor. Zeros will become 1 and non-zeros will become 0.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor LogicalNot(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = tensor.data[i] == 0 ? 1 : 0;
            }

            return result;
        }
        /// <summary>
        /// Computes the tensor values passed through the ReLU function.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor ReLU(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Max(0f, tensor.data[i]);
            }

            return result;
        }
        /// <summary>
        /// Computes the tensor values passed through the hyperbolic tangent function.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor Tanh(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                float e2x = MathF.Exp(2f * tensor.data[i]);
                result.data[i] = (e2x - 1f) / (e2x + 1f);
            }

            return result;
        }
        /// <summary>
        /// Computes the tensor values passed through the hyperbolic secant function.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor Sech(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = 2f * MathF.Exp(tensor.data[i]) / (MathF.Exp(2f * tensor.data[i]) + 1f);
            }

            return result;
           
        }
        /// <summary>
        /// Computes the tensor values passed through the sigmoid activation function.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor Sigmoid(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = 1.0f /  (1 + MathF.Exp(-tensor.data[i]));
            }

            return result;
        }
        /// <summary>
        /// Computes the tensor values passed through the softplus activation function.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor Softplus(Tensor tensor)
        {
            Tensor result = new(tensor.shape);

            for (int i = 0; i < result.data.Length; i++)
            {
                result.data[i] = MathF.Log(1 + MathF.Exp(tensor.data[i]));
            }

            return result;
        }
        /// <summary>
        /// Computes the norm of the tensor, of any shape.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="norm"></param>
        /// <param name="eps">Value for stability when computing <see cref="NormType.EuclideanL2"/> norm.</param>
        /// <returns><see cref="Tensor"/> (1)</returns>
        /// <exception cref="Exception"></exception>
        public static Tensor Norm(Tensor tensor, NormType norm = NormType.EuclideanL2, float eps = 1e-12f)
        {
            switch (norm)
            {
                case NormType.NonZeroL0:
                    int nonzeros = tensor.Count(x => x != 0);
                    return Constant(nonzeros);
                case NormType.ManhattanL1:
                    float absSum = tensor.data.Sum(x => MathF.Abs(x));
                    return Constant(absSum);
                case NormType.EuclideanL2:
                    float sqrSum = tensor.data.Sum(x => x * x);
                    return Constant(MathF.Sqrt(sqrSum + eps));
                case NormType.MaxLInf:
                    float maxAbs = tensor.data.Max(x => Math.Abs(x));
                    return Constant(maxAbs);
                default:
                    throw new Exception("Unhandled norm type.");
            }
        }
        /// <summary>
        /// Computes the trace of a 2D tensor.
        /// </summary>
        /// <param name="axis"></param>
        /// <returns><see cref="Tensor"/> of shape (1)</returns>
        public static Tensor Trace(Tensor tensor)
        {
            if (!(tensor.Rank == 2 || tensor.Rank == 1))
                throw new ArgumentException($"Tensor must have rank 2 or 1 (received {tensor.Rank})");

            if (tensor.Width != tensor.Height)
                throw new ArgumentException($"Tensor received must be a square matrix (received shape ({tensor.Shape.ToCommaSeparatedString()}))");
            Tensor tr = Zeros(1);
            for (int i = 0; i < tensor.Width; i++)
            {
                tr[0] += tensor[i, i];
            }
            return tr;

        }
        /// <summary>
        /// Returns the lower triangular part of the matrix 2D tensor (allows also batched input), the other elements of the result tensor out are set to zero.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="diagonal"></param>
        /// <returns></returns>
        public static Tensor Tril(Tensor tensor, int diagonal = 0)
        {
            if (tensor.Rank < 2)
                throw new ArgumentException($"Tensor rank must be higher than 1 (received shape {tensor.shape.ToCommaSeparatedString()} and rank {tensor.Rank})");

            Tensor tril = Tensor.Zeros(tensor.shape);

            for (int b = 0; b < tril.Batch; b++)
            {
                for (int c = 0; c < tril.Channels; c++)
                {
                    for (int h = 0; h < tril.Height; h++)
                    {
                        for (int w = 0; w < tril.Width; w++)
                        {
                            if(h >= w - diagonal)
                                tril[b, c, h, w] = tensor[b, c, h, w];
                        }
                    }
                }
            }

            return tril;
        }
        /// <summary>
        /// Returns the upper triangular part of the matrix 2D tensor (allows also batched input), the other elements of the result tensor out are set to zero.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="diagonal"></param>
        /// <returns></returns>
        public static Tensor Triu(Tensor tensor, int diagonal = 0)
        {
            if (tensor.Rank < 2)
                throw new ArgumentException($"Tensor rank must be higher than 1 (received shape {tensor.shape.ToCommaSeparatedString()} and rank {tensor.Rank})");

            Tensor triu = Tensor.Zeros(tensor.shape);

            for (int b = 0; b < triu.Batch; b++)
            {
                for (int c = 0; c < triu.Channels; c++)
                {
                    for (int h = 0; h < triu.Height; h++)
                    {
                        for (int w = 0; w < triu.Width; w++)
                        {
                            if (h <= w - diagonal)
                                triu[b, c, h, w] = tensor[b, c, h, w];
                        }
                    }
                }
            }

            return triu;
        }
        /// <summary>
        /// Returns the tensor with the values rounded to integers.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor Int(Tensor tensor)
        {
            Tensor intTensor = new(tensor.shape);
            for (int i = 0; i < intTensor.data.Length; i++)
            {
                intTensor.data[i] = (int)tensor.data[i];
            }
            return intTensor;
        }
        /// <summary>
        /// Extracts a slice from the tensor along a specified axis.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="axis">The axis along which to slice.</param>
        /// <param name="startInclusive">The starting index (inclusive). Can be negative.</param>
        /// <param name="endExclusive">The ending index (exclusive). Can be negative.</param>
        /// <param name="step">The step size. Can be negative for reverse slicing.</param>
        /// <returns>A new tensor containing the sliced data.</returns>
        public static Tensor Slice(Tensor tensor, int axis, int startInclusive, int endExclusive, int step = 1)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (step == 0)
                throw new ArgumentException("Step cannot be zero.", nameof(step));

            HandleAxis(tensor, ref axis); 
            int dimSize = tensor.shape[axis];
            if (dimSize == 0)
                throw new ArgumentException("Cannot slice along an axis of size 0.");

            if (startInclusive < 0) startInclusive = Math.Max(0, dimSize + startInclusive);
            if (endExclusive < 0) endExclusive = Math.Max(0, dimSize + endExclusive);

            startInclusive = Math.Max(0, Math.Min(dimSize, startInclusive));
            endExclusive = Math.Max(0, Math.Min(dimSize, endExclusive));

            if ((step > 0 && startInclusive >= endExclusive) || (step < 0 && startInclusive <= endExclusive))
            {
                int[] newShape = (int[])tensor.shape.Clone();
                newShape[axis] = 0;
                return Zeros(newShape);
            }

            int slicedDimSize = 0;
            if (step > 0)
            {
                slicedDimSize = (endExclusive - startInclusive + step - 1) / step; 
            }
            else 
            {
                slicedDimSize = (startInclusive - endExclusive - step - 1) / (-step); 
            }
            slicedDimSize = Math.Max(0, slicedDimSize); 

            int[] resultShape = (int[])tensor.shape.Clone();
            resultShape[axis] = slicedDimSize;
            Tensor result = Zeros(resultShape);

            Dim dimEnum = AxisToDim(tensor, axis);

            int batch = tensor.Batch;
            int channels = tensor.Channels;
            int height = tensor.Height;
            int width = tensor.Width;

            for (int l = 0; l < batch; l++)
            {
                for (int k = 0; k < channels; k++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            int originalIndexAlongAxis = -1;
                            switch (dimEnum)
                            {
                                case Dim.width: originalIndexAlongAxis = i; break;
                                case Dim.height: originalIndexAlongAxis = j; break;
                                case Dim.channel: originalIndexAlongAxis = k; break;
                                case Dim.batch: originalIndexAlongAxis = l; break;
                            }

                            bool isInSlice = false;
                            int resultIndexAlongAxis = -1;

                            if (step > 0)
                            {
                                if (originalIndexAlongAxis >= startInclusive &&
                                    originalIndexAlongAxis < endExclusive &&
                                    (originalIndexAlongAxis - startInclusive) % step == 0)
                                {
                                    isInSlice = true;
                                    resultIndexAlongAxis = (originalIndexAlongAxis - startInclusive) / step;
                                }
                            }
                            else 
                            {
                                if (originalIndexAlongAxis <= startInclusive &&
                                    originalIndexAlongAxis > endExclusive &&
                                    (startInclusive - originalIndexAlongAxis) % (-step) == 0)
                                {
                                    isInSlice = true;
                                    resultIndexAlongAxis = (startInclusive - originalIndexAlongAxis) / (-step);
                                }
                            }

                            if (isInSlice)
                            {
                                int resultL = l, resultK = k, resultJ = j, resultI = i;
                                switch (dimEnum)
                                {
                                    case Dim.width: resultI = resultIndexAlongAxis; break;
                                    case Dim.height: resultJ = resultIndexAlongAxis; break;
                                    case Dim.channel: resultK = resultIndexAlongAxis; break;
                                    case Dim.batch: resultL = resultIndexAlongAxis; break;
                                }

                                result[resultL, resultK, resultJ, resultI] = tensor[l, k, j, i];
                            }
                        }
                    }
                }
            }

            return result;
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
            return Reshape(this, newShape);
        }
        public Tensor Permute(params int[] axes)
        {
            return Permute(this, axes);
        }
        public Tensor Squeeze(int? axis = null)
        {
            return Squeeze(this, axis);
        }
        public Tensor Unsqueeze(int axis)
        {
            return Unsqueeze(this, axis);
        }
        public Tensor Flatten(int startAxis = 0, int endAxis = -1)
        {
            return Flatten(this, startAxis, endAxis);
        }
        public Tensor Expand(int axis, int times)
        {
            return Expand(this, axis, times);
        }
        public Tensor Transpose(int axis0, int axis1)
        {
            return Transpose(this, axis0, axis1);
        }
        public Tensor T()
        {
            return T(this);
        }
        public Tensor[] Split(int axis, int split_size)
        {
            return Split(this, axis, split_size);
        }
        public Tensor[] Chunk(int axis, int num_chunks)
        {
            return Chunk(this, axis, num_chunks);
        }
        public Tensor Roll(int axis, int shifts)
        {
            return Roll(this, axis, shifts);
        }
        public Tensor Shuffle(int axis)
        {
            return Shuffle(this, axis);
        }
        public Tensor Sum(int axis, bool keepDim = false)
        {
            return Sum(this, axis, keepDim);
        }
        public Tensor Prod(int axis, bool keepDim = false)
        {
            return Prod(this, axis, keepDim);
        }
        public Tensor Mean(int axis, bool keepDim = false)
        {
            return Mean(this, axis, keepDim);
        }      
        public Tensor Var(int axis, int correction = 1, bool keepDim = false)
        {
            return Var(this, axis, correction, keepDim);
        }
        public Tensor Std(int axis, int correction = 1, bool keepDim = false)
        {
            return Std(this, axis, correction, keepDim);
        }
        public Tensor Min(int axis, bool keepDim = false)
        {
            return Min(this, axis, keepDim);
        }
        public Tensor Max(int axis, bool keepDim = false)
        {
            return Max(this, axis, keepDim);
        }
        public Tensor ArgMax(int axis, bool keepDim = false)
        {
            return Tensor.ArgMax(this, axis, keepDim);
        }
        public Tensor ArgMin(int axis, bool keepDim = false)
        {
            return ArgMin(this, axis, keepDim);
        }
        public Tensor Sort(int axis, bool ascending = true)
        {
            return Sort(this, axis, ascending);
        }
        public Tensor CumSum(int axis)
        {
            return CumSum(this, axis);
        }
        public Tensor CumProd(int axis)
        {
            return CumProd(this, axis);
        }
        public Tensor Pow(float power)
        {
            return Pow(this, power);
        }
        public Tensor Square()
        {
            return Square(this);
        }
        public Tensor Sqrt()
        {
            return Sqrt(this);
        }
        public Tensor RSqrt()
        {
            return RSqrt(this);
        }
        public Tensor Reciprocal()
        {
            return Reciprocal(this);
        }
        public Tensor Exp()
        {
            return Exp(this);
        }
        public Tensor Log(float @base = MathF.E)
        {
            return Log(this, @base);
        }
        public Tensor Abs()
        {
            return Abs(this);
        }
        public Tensor Sin()
        {
            return Sin(this);
        }
        public Tensor ArcSin()
        {
            return ArcSin(this);
        }
        public Tensor Cos()
        {
            return Cos(this);
        }
        public Tensor ArcCos()
        {
            return ArcCos(this);
        }
        public Tensor Tan()
        {
            return Tan(this);
        }
        public Tensor ArcTan()
        {
            return ArcTan(this);
        }
        public Tensor Sign()
        {
            return Sign(this);
        }
        public Tensor Clip(float min, float max)
        {
            return Clip(this, min, max);
        }
        public Tensor Ceil()
        {
            return Ceil(this);
        }
        public Tensor Floor()
        {
            return Floor(this);
        }
        public Tensor LogicalNot()
        {
            return LogicalNot(this);
        }
        public Tensor Tanh()
        {
            return Tanh(this);
        }
        public Tensor Sech()
        {
            return Sech(this);
        }
        public Tensor ReLU()
        {
            return ReLU(this);
        }
        public Tensor Sigmoid()
        {
            return Sigmoid(this);
        }
        public Tensor Softplus()
        {
            return Softplus(this);
        }
        public Tensor Norm(NormType norm = NormType.EuclideanL2, float eps = 1E-12f)
        {
            return Norm(this, norm, eps);
        }
        public Tensor Trace()
        {
            return Trace(this);
        }
        public Tensor Tril(int diagonal = 0)
        {
            return Tril(this, diagonal);
        }
        public Tensor Triu(int diagonal = 0)
        {
            return Triu(this, diagonal);
        }
        public Tensor Int()
        {
            return Int(this);
        } 
        public Tensor Slice(int axis, int startInclusive, int endExclusive, int step)
        {
            return Slice(this, axis, startInclusive, endExclusive, step);
        }
        #endregion Instance


        #region LINQ
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
            if (!second.shape.SequenceEqual(this.shape))
                throw new ArgumentException("The shape of the second tensor does not match the shape of this tensor.");
            Tensor result = new(shape);

            for (int i = 0; i < data.Length; i++)
            {
                result.data[i] = resultSelector(data[i], second.data[i]);
            }

            return result;
        }
        public bool Contains(float item)
        {
            return data.Contains(item);
        }
        public bool Any(Func<float, bool> predicate)
        {
            return data.Any(predicate);
        }
        public bool All(Func<float, bool> predicate)
        {
            return data.All(predicate);
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
        public float Average(Func<float, float> selector = null)
        {
            if (selector == null)
                return data.Average();
            else
                return data.Average(selector);
        }
        public float Sum(Func<float, float> selector = null)
        {
            if (selector == null)
                return data.Sum();
            else
                return data.Sum(selector);
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
        public object Clone()
        {
            return Identity(this);
        }
        public override string ToString()
        {
            int rank = Rank;

            bool shorten = data.Length > 1000;

            StringBuilder sb = new();

            sb.Append($"Tensor({Shape.ToCommaSeparatedString()})");
            sb.Append("\n[");

            if (!shorten)
            {
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

                                sb.Append(this[l, k, j, i].ToString(StringFormat, CultureInfo.InvariantCulture));
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

            void PrintRow(int l, int k, int j)
            {
                if (j > 0 && rank > 1)
                {
                    sb.Append("\n");
                    for (int indent = 0; indent < rank - 1; indent++)
                        sb.Append(" ");
                }
                if (rank > 1)
                    sb.Append("[");

                int cols = Width;
                if (cols <= 6)
                {
                    for (int i = 0; i < cols; i++)
                    {
                        if (i > 0) sb.Append(", ");
                        sb.Append(this[l, k, j, i].ToString(StringFormat, CultureInfo.InvariantCulture));
                    }
                }
                else
                {
                    for (int i = 0; i < 3; i++)
                    {
                        if (i > 0) sb.Append(", ");
                        sb.Append(this[l, k, j, i].ToString(StringFormat, CultureInfo.InvariantCulture));
                    }
                    sb.Append(", ..., ");
                    for (int i = cols - 3; i < cols; i++)
                    {
                        if (i > cols - 3) sb.Append(", ");
                        sb.Append(this[l, k, j, i].ToString(StringFormat, CultureInfo.InvariantCulture));
                    }
                }

                if (rank > 1)
                    sb.Append("]");
            }

            void PrintRows(int l, int k)
            {
                int rows = Height;
                if (rows <= 6)
                {
                    for (int j = 0; j < rows; j++) PrintRow(l, k, j);
                    return;
                }

                for (int j = 0; j < 3; j++) PrintRow(l, k, j);

                if (rank > 1)
                {
                    sb.Append("\n");
                    for (int indent = 0; indent < rank - 1; indent++)
                        sb.Append(" ");
                    sb.Append("...");
                }

                for (int j = rows - 3; j < rows; j++) PrintRow(l, k, j);
            }

            void PrintChannelBlock(int l, int k, bool addLeadingGapByK)
            {
                if (addLeadingGapByK)
                {
                    sb.Append("\n\n");
                    for (int indent = 0; indent < rank - 2; indent++)
                        sb.Append(" ");
                }
                if (rank > 2) sb.Append("[");

                PrintRows(l, k);

                if (rank > 2) sb.Append("]");
            }

            for (int l = 0; l < (Batch <= 6 ? Batch : 3); l++)
            {
                if (l > 0)
                {
                    sb.Append("\n\n\n");
                    for (int indent = 0; indent < rank - 3; indent++)
                        sb.Append(" ");
                }
                if (rank > 3) sb.Append("[");

                if (Channels <= 6)
                {
                    for (int k = 0; k < Channels; k++)
                        PrintChannelBlock(l, k, k > 0);
                }
                else
                {
                    for (int k = 0; k < 3; k++)
                        PrintChannelBlock(l, k, k > 0);
                    sb.Append("\n\n");
                    for (int indent = 0; indent < rank - 2; indent++)
                        sb.Append(" ");
                    sb.Append("...");

                    for (int k = Channels - 3; k < Channels; k++)
                        PrintChannelBlock(l, k, true);
                }

                if (rank > 3) sb.Append("]");
            }

            if (Batch > 6)
            {
                sb.Append("\n\n\n");
                for (int indent = 0; indent < rank - 3; indent++)
                    sb.Append(" ");
                sb.Append("...");

                for (int l = Batch - 3; l < Batch; l++)
                {
                    sb.Append("\n\n\n");
                    for (int indent = 0; indent < rank - 3; indent++)
                        sb.Append(" ");
                    if (rank > 3) sb.Append("[");

                    if (Channels <= 6)
                    {
                        for (int k = 0; k < Channels; k++)
                            PrintChannelBlock(l, k, k > 0);
                    }
                    else
                    {
                        for (int k = 0; k < 3; k++)
                            PrintChannelBlock(l, k, k > 0);

                        sb.Append("\n\n");
                        for (int indent = 0; indent < rank - 2; indent++)
                            sb.Append(" ");
                        sb.Append("...");

                        for (int k = Channels - 3; k < Channels; k++)
                            PrintChannelBlock(l, k, true);
                    }

                    if (rank > 3) sb.Append("]");
                }
            }

            sb.Append("]");
            return sb.ToString();
        }
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
        /// <summary>
        /// The format in which the tensor elements are displayed. <em>Default: "0.00000e0".</em>
        /// </summary>
        private static string StringFormat { get; set; } = "0.00000";

        #endregion LINQ



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
                    throw new ArgumentException($"Invalid axis value ({axis}) for a tensor with rank ({tensor.Rank})");

                axis = 0;
            }
            else
            {
                if (axis >= rank)
                    throw new ArgumentException($"Invalid axis value ({axis}) for a tensor with rank ({tensor.Rank})");

                if (axis < 0)
                   axis = rank + axis;
            }     
        }
        /// <summary>
        /// Squeezes the tensor in-place.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="axis"></param>
        private static void Squeeze_(Tensor tensor, int? axis = null)
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

                tensor.shape = squeezedShape.ToArray();
            }
            else
            {
                int ax = axis.Value;
                HandleAxis(tensor, ref ax);
                // if axis is not 1, tensor remains unchanged
                if (tensor.shape[ax] != 1)
                    return;

                // Else remove that axis
                if (tensor.shape.Length > 1)
                {
                    List<int> newShape = tensor.shape.ToList();
                    newShape.RemoveAt(ax);

                    tensor.shape = newShape.ToArray();
                }

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