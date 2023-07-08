using System;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using kbRadu;
using System.CodeDom;
using DeepUnity;

namespace DeepUnity
{
    
    /// <summary>
    /// Experimental tensor fully on GPU.
    /// Current problems:
    ///  - doesn t allow large tensors > 1024x1024
    ///  - fullfills the vram because the GC has low latency
    ///  
    /// 
    ///  Mean computing: 10k numbers. 0.008 on GPU, 0.0009 on CPU
    /// </summary>

    public class TensorGPU : IDisposable, IEquatable<TensorGPU>
    {
        public readonly static LinkedList<TensorGPU> tensors = new LinkedList<TensorGPU>();

        // i should go for a wrapper in the future to serialize this
        private  ComputeBuffer data;
        private int[] shape;
        private bool disposed = false;

        public int Size(int axis)
        {
            return shape[axis];
        }
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
        public int[] Shape => shape;



        // Create
        private TensorGPU(params int[] shape)
        {
            if (shape == null || shape.Length == 0)
                throw new ArgumentException("Tensor cannot be instantiated with null ");
            if (shape.Length > 4)
                throw new ArgumentException("Tensor cannot be instantiated with more than 4 dimensions.");

            int size = 1;
            foreach (var item in shape)
            {
                size *= item;
            }

            if (size > 4_194_304) // hardcoded like this because 4096x4096 max allowed matrix, on 8192 it crashes
                throw new NotSupportedException("Tensor dimensions is too large on initialization (cannot surpass 16,777,216 units).");


            this.shape = shape.ToArray();
            this.data = new ComputeBuffer(Count(), 4);

            tensors.AddLast(this);
            GC.Collect();
        }
        public static TensorGPU Arange(float start, float end, float step)
        {
            if (start == end)
                throw new System.ArgumentException("Start and end arguments should not be equal");

            int count = (int) MathF.Ceiling((end - start) / step);
            TensorGPU t = new(count);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Arange");
            cs.SetFloat("start", start);
            cs.SetFloat("end", end);
            cs.SetFloat("step", step);
            cs.SetBuffer(kernel, "result", t.data);

            cs.Dispatch(kernel, 1, 1, 1);
            return t;
        }
        public static TensorGPU Random01(params int[] shape)
        {
            TensorGPU t = new(shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;
           
            int kernel = cs.FindKernel("Random01");
            cs.SetInt("seed", DateTime.Now.Millisecond);
            cs.SetBuffer(kernel, "result", t.data);
            
            cs.Dispatch(kernel, 1, 1, 1);
            return t;
        }
        public static TensorGPU RandomNormal((float,float) mu_sigma, params int[] shape)
        {
            TensorGPU t = new(shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("RandomNormal");
            cs.SetInt("seed", DateTime.Now.Millisecond);
            cs.SetBuffer(kernel, "result", t.data);
            cs.SetFloat("mu", mu_sigma.Item1);
            cs.SetFloat("sigma", mu_sigma.Item2);

            cs.Dispatch(kernel, 1, 1, 1);
            return t;
        }
        public static TensorGPU Constant(float[,] matrix)
        {
            int height = matrix.GetLength(0);
            int width = matrix.GetLength(1);

            TensorGPU t = new(height, width);

            float[] dataarr = new float[width * height];
            int index = 0;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    dataarr[index++] = matrix[i, j];
                }

            }
            t.data.SetData(dataarr);

            return t;
        }


        // Operator overloading
        public static TensorGPU operator +(TensorGPU left, TensorGPU right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left and right tensors must have different shape for Element-Wise addition (+).");

            ComputeShader cs = DeepUnityMeta.TensorCS;
            TensorGPU result = new(left.shape);

            int kernel = cs.FindKernel("AdditionElementWise");
            cs.SetBuffer(kernel, "data1", left.data);
            cs.SetBuffer(kernel, "data2", right.data);
            cs.SetBuffer(kernel, "result", result.data);

            int threads = DeepUnityMeta.THREADS_NUM;
            cs.Dispatch(kernel, (result.Count() + threads - 1)/threads, 1, 1);
            return result;
        }
        public static TensorGPU operator -(TensorGPU left, TensorGPU right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left and right tensors must have different shape for Element-Wise subtraction (-).");

            ComputeShader cs = DeepUnityMeta.TensorCS;
            TensorGPU result = new(left.shape);

            int kernel = cs.FindKernel("SubtractionElementWise");
            cs.SetBuffer(kernel, "data1", left.data);
            cs.SetBuffer(kernel, "data2", right.data);
            cs.SetBuffer(kernel, "result", result.data);

            int threads = DeepUnityMeta.THREADS_NUM;
            cs.Dispatch(kernel, (result.Count() + threads - 1) / threads, 1, 1);
            return result;
        }
        public static TensorGPU operator *(TensorGPU left, TensorGPU right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left and right tensors must have different shape for Element-Wise multipication (*).");

            ComputeShader cs = DeepUnityMeta.TensorCS;
            TensorGPU result = new(left.shape);

            int kernel = cs.FindKernel("MultiplicationElementWise");
            cs.SetBuffer(kernel, "data1", left.data);
            cs.SetBuffer(kernel, "data2", right.data);
            cs.SetBuffer(kernel, "result", result.data);

            int threads = DeepUnityMeta.THREADS_NUM;
            cs.Dispatch(kernel, (result.Count() + threads - 1) / threads, 1, 1);
            return result;
        }
        public static TensorGPU operator /(TensorGPU left, TensorGPU right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left and right tensors must have different shape for Element-Wise division (/).");

            ComputeShader cs = DeepUnityMeta.TensorCS;
            TensorGPU result = new(left.shape);

            int kernel = cs.FindKernel("DivisionElementWise");
            cs.SetBuffer(kernel, "data1", left.data);
            cs.SetBuffer(kernel, "data2", right.data);
            cs.SetBuffer(kernel, "result", result.data);

            int threads = DeepUnityMeta.THREADS_NUM;
            cs.Dispatch(kernel, (result.Count() + threads - 1) / threads, 1, 1);
            return result;
        }
        public static TensorGPU operator +(TensorGPU left, float right)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            TensorGPU result = new(left.shape);

            int kernel = cs.FindKernel("AdditionWithScalar");
            cs.SetBuffer(kernel, "data1", left.data);
            cs.SetBuffer(kernel, "result", result.data);
            cs.SetFloat("value", right);

            int threads = DeepUnityMeta.THREADS_NUM;
            cs.Dispatch(kernel, (result.Count() + threads - 1) / threads, 1, 1);
            return result;
        }
        public static TensorGPU operator -(TensorGPU left, float right)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            TensorGPU result = new(left.shape);

            int kernel = cs.FindKernel("SubtractionWithScalar");
            cs.SetBuffer(kernel, "data1", left.data);
            cs.SetBuffer(kernel, "result", result.data);
            cs.SetFloat("value", right);

            int threads = DeepUnityMeta.THREADS_NUM;
            cs.Dispatch(kernel, (result.Count() + threads - 1) / threads, 1, 1);
            return result;
        }
        public static TensorGPU operator *(TensorGPU left, float right)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            TensorGPU result = new(left.shape);

            int kernel = cs.FindKernel("MultiplicationWithScalar");
            cs.SetBuffer(kernel, "data1", left.data);
            cs.SetBuffer(kernel, "result", result.data);
            cs.SetFloat("value", right);

            int threads = DeepUnityMeta.THREADS_NUM;
            cs.Dispatch(kernel, (result.Count() + threads - 1) / threads, 1, 1);
            return result;
        }
        public static TensorGPU operator /(TensorGPU left, float right)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            TensorGPU result = new(left.shape);

            int kernel = cs.FindKernel("DivisionWithScalar");
            cs.SetBuffer(kernel, "data1", left.data);
            cs.SetBuffer(kernel, "result", result.data);
            cs.SetFloat("value", right);

            int threads = DeepUnityMeta.THREADS_NUM;
            cs.Dispatch(kernel, (result.Count() + threads - 1) / threads, 1, 1);
            return result;
        }
        public static TensorGPU operator +(float left, TensorGPU right)
        {
            return right + left;
        }
        public static TensorGPU operator -(float left, TensorGPU right)
        {
            return right - left;
        }
        public static TensorGPU operator *(float left, TensorGPU right)
        {
            return right * left;
        }
       

        // Special Operations
        public static TensorGPU MatMul(TensorGPU left, TensorGPU right)
        {
            if (left.Width != right.Height)
                throw new ArgumentException("Tensor must have compatible shapes for matrix multiplication (height of left tensor is not matching the width of the right tensor).");

            if (left.Channels != right.Channels)
                throw new ArgumentException("Tensors must have similar number of channels for channeled matrix multiplication.");

            if (left.Rank != right.Rank)
                throw new ArgumentException("Tensors must have similar rank for matrix multiplication.");

            if(left.Rank < 2)
                throw new ArgumentException("Tensors must rave rank >= 2 (need to be matrices) for matrix multiplication.");

            TensorGPU result = null;
            if (left.Rank == 2)
                result = new(left.Height, right.Width);
            else if (left.Rank == 3)
                result = new(left.Channels, left.Height, right.Width);
            else if (left.Rank == 4)
                result = new(left.Batch, left.Channels, left.Height, right.Width);


            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("MatMul");

            cs.SetBuffer(kernel, "data1", left.data);
            cs.SetBuffer(kernel, "data2", right.data);
            cs.SetBuffer(kernel, "result", result.data);

            cs.SetInt("w1", left.Width);
            cs.SetInt("h1", left.Height);
            cs.SetInt("c1", left.Channels);
            cs.SetInt("b1", left.Batch);

            cs.SetInt("w2", right.Width);
            cs.SetInt("h2", right.Height);
            cs.SetInt("c2", right.Channels);
            cs.SetInt("b2", right.Batch);

            cs.SetInt("wr", result.Width);
            cs.SetInt("hr", result.Height);
            cs.SetInt("cr", result.Channels);
            cs.SetInt("br", result.Batch);

            cs.Dispatch(kernel,
                  (left.Height + 7) / 8,
                  (right.Width + 7) / 8,
                  (left.Channels + 7) / 8);


            return result;

        }


        // Operations
        public static TensorGPU Mean(TensorGPU tensor, int axis, bool keepDim = false)
        {
            if (axis < 0 || axis >= tensor.Rank)
                throw new ArgumentOutOfRangeException("Invalid axis value.");

            int[] newShape;
            if (keepDim)
            {
                newShape = tensor.shape.ToArray();
            }
            else
            {
                newShape = tensor.shape.ToArray();
                newShape[axis] = 1;
            }

            TensorGPU result = new TensorGPU(newShape);

            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Mean");
            
            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.SetBuffer(kernel, "result", result.data);

            cs.SetInt("w1", tensor.Width);
            cs.SetInt("h1", tensor.Height);
            cs.SetInt("c1", tensor.Channels);
            cs.SetInt("b1", tensor.Batch);
            cs.SetInt("r1", tensor.Rank);

            cs.SetInt("wr", result.Width);
            cs.SetInt("hr", result.Height);
            cs.SetInt("cr", result.Channels);
            cs.SetInt("br", result.Batch);
            cs.SetInt("rr", tensor.Rank);

            cs.SetInt("axis", axis);
            cs.SetBool("keepDim", keepDim);

            cs.Dispatch(kernel, 1, 1, 1);

            return result;
        }
        public static TensorGPU Sum(TensorGPU tensor, int axis, bool keepDim = false)
        {
            if (axis < 0 || axis >= tensor.Rank)
                throw new ArgumentOutOfRangeException("Invalid axis value.");

            int[] newShape;
            if (keepDim)
            {
                newShape = tensor.shape.ToArray();
            }
            else
            {
                newShape = tensor.shape.ToArray();
                newShape[axis] = 1;
            }

            TensorGPU result = new TensorGPU(newShape);

            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Sum");

            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.SetBuffer(kernel, "result", result.data);

            cs.SetInt("w1", tensor.Width);
            cs.SetInt("h1", tensor.Height);
            cs.SetInt("c1", tensor.Channels);
            cs.SetInt("b1", tensor.Batch);
            cs.SetInt("r1", tensor.Rank);

            cs.SetInt("wr", result.Width);
            cs.SetInt("hr", result.Height);
            cs.SetInt("cr", result.Channels);
            cs.SetInt("br", result.Batch);
            cs.SetInt("rr", tensor.Rank);

            cs.SetInt("axis", axis);
            cs.SetBool("keepDim", keepDim);

            cs.Dispatch(kernel, 1, 1, 1);

            return result;
        }
        public static TensorGPU Var(TensorGPU tensor, int axis, int correction = 1, bool keepDim = false)
        {
            if (axis < 0 || axis >= tensor.Rank)
                throw new ArgumentOutOfRangeException("Invalid axis value.");

            int[] newShape;
            if (keepDim)
            {
                newShape = tensor.shape.ToArray();
            }
            else
            {
                newShape = tensor.shape.ToArray();
                newShape[axis] = 1;
            }

            TensorGPU result = new TensorGPU(newShape);

            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Var");

            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.SetBuffer(kernel, "result", result.data);

            cs.SetInt("w1", tensor.Width);
            cs.SetInt("h1", tensor.Height);
            cs.SetInt("c1", tensor.Channels);
            cs.SetInt("b1", tensor.Batch);
            cs.SetInt("r1", tensor.Rank);

            cs.SetInt("wr", result.Width);
            cs.SetInt("hr", result.Height);
            cs.SetInt("cr", result.Channels);
            cs.SetInt("br", result.Batch);
            cs.SetInt("rr", tensor.Rank);
            
            
            cs.SetInt("axis", axis);
            cs.SetInt("correction", correction);
            cs.SetBool("keepDim", keepDim);

            cs.Dispatch(kernel, 1, 1, 1);

            return result;
        }
        public static TensorGPU Std(TensorGPU tensor, int axis, int correction = 1, bool keepDim = false)
        {
            if (axis < 0 || axis >= tensor.Rank)
                throw new ArgumentOutOfRangeException("Invalid axis value.");

            int[] newShape;
            if (keepDim)
            {
                newShape = tensor.shape.ToArray();
            }
            else
            {
                newShape = tensor.shape.ToArray();
                newShape[axis] = 1;
            }

            TensorGPU result = new TensorGPU(newShape);

            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Std");

            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.SetBuffer(kernel, "result", result.data);

            cs.SetInt("w1", tensor.Width);
            cs.SetInt("h1", tensor.Height);
            cs.SetInt("c1", tensor.Channels);
            cs.SetInt("b1", tensor.Batch);
            cs.SetInt("r1", tensor.Rank);

            cs.SetInt("wr", result.Width);
            cs.SetInt("hr", result.Height);
            cs.SetInt("cr", result.Channels);
            cs.SetInt("br", result.Batch);
            cs.SetInt("rr", tensor.Rank);


            cs.SetInt("axis", axis);
            cs.SetInt("correction", correction);
            cs.SetBool("keepDim", keepDim);

            cs.Dispatch(kernel, 1, 1, 1);

            return result;
        }



        // Math operations
        public static TensorGPU Pow(TensorGPU tensor, float power)
        {
            return null;
        }


        // other
        public int Count()
        {
            int count = 1;
            foreach (var item in shape)
            {
                count *= item;

            }
            return count;
        }
        public void Dispose()
        { 
            if(!disposed)
            {
                data.Release();
                disposed = true;
            }
            
            GC.SuppressFinalize(this);
        }
        public override string ToString()
        {
            int rank = Rank;
            float[] arrdata = new float[Count()];
            data.GetData(arrdata);

            StringBuilder sb = new();

            sb.Append("Tensor ");
            sb.Append($"[{shape.ToCommaSeparatedString()}]");


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

                            int index = l * Channels * Height * Width + k * Height * Width + j * Width + i;
                            sb.Append(arrdata[index].ToString("0.00000"));
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
        public bool Equals(TensorGPU other)
        {
            if (!shape.SequenceEqual(other.shape))
                return false;

            float[] data1 = new float[data.count];
            float[] data2 = new float[other.data.count];

            data.GetData(data1);
            other.data.GetData(data2);

            if (!data1.SequenceEqual(data2))
                return false;

            return true;
        }
    }


    [InitializeOnLoad]
    internal sealed class TensorGPUDisposer
    {
        static TensorGPUDisposer()
        {
            EditorApplication.playModeStateChanged += DisposeAllTensors;
        }

        private TensorGPUDisposer() { }
        private static void DisposeAllTensors(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.ExitingPlayMode)
            {
                foreach (var item in TensorGPU.tensors)
                {
                    item?.Dispose();
                }
            }
        }
    }
}

