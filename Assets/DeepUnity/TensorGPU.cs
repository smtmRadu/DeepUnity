using System;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

namespace DeepUnity
{
    [InitializeOnLoad]
    internal class TensorGPUDisposer
    {
        static TensorGPUDisposer()
        {
            EditorApplication.playModeStateChanged += DisposeAllTensors;
        }
        private static void DisposeAllTensors(PlayModeStateChange state)
        {
            if(state == PlayModeStateChange.ExitingPlayMode)
            {
                foreach (var item in TensorGPU.tensors)
                {
                    item?.Dispose();
                }
            }
        }
    }
    /// <summary>
    /// Experimental tensor fully on GPU.
    /// Current problems:
    ///  - doesn t allow large tensors > 1024x1024
    ///  - fullfills the vram because the GC has low latency
    ///  
    /// </summary>

    public class TensorGPU : IDisposable
    {
        public static LinkedList<TensorGPU> tensors = new LinkedList<TensorGPU>();

        private ComputeBuffer data;
        private readonly int[] shape;
        private bool disposed = false;

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
        public int Rank { get
            {
                if (shape.Length > 1)
                    return shape.Length;
                else if (shape[0] > 1)
                    return 1;
                else
                    return 0;
            }
        }
        public int[] Shape
        {
            get => shape;
        }

        // Create
        private TensorGPU(params int[] shape)
        {
            this.shape = shape;
            this.data = new ComputeBuffer(Count(), 4);
            tensors.AddLast(this);
        }       
        public static TensorGPU Random01(params int[] shape)
        {
            TensorGPU t = new(shape);
            ComputeShader cs = DeepUnityMeta.TensorGPUCS;
           
            int kernel = cs.FindKernel("Random01");
            cs.SetInt("value", DateTime.Now.Millisecond);
            cs.SetBuffer(kernel, "data1", t.data);
           

            int threads = DeepUnityMeta.TensorGPUThreads;
            cs.Dispatch(kernel, (t.Count() + threads - 1) / threads, 1, 1);
            return t;
        }


        public static TensorGPU operator +(TensorGPU left, TensorGPU right)
        {
            if (!left.shape.SequenceEqual(right.shape))
                throw new OperationCanceledException($"Left and right tensors must have different shape for Element-wise addition (+).");

            ComputeShader cs = DeepUnityMeta.TensorGPUCS;
            TensorGPU result = new(left.shape);

            int kernel = cs.FindKernel("Addition");
            cs.SetBuffer(kernel, "data1", left.data);
            cs.SetBuffer(kernel, "data2", right.data);
            cs.SetBuffer(kernel, "result", result.data);

            int threads = DeepUnityMeta.TensorGPUThreads;
            cs.Dispatch(kernel, (left.Count() + threads - 1) / threads, 1, 1);
            return result;
        }

        // Operations
        public static TensorGPU MatMul(TensorGPU left, TensorGPU right)
        {
            if(left.Width != right.Height)
                throw new ArgumentException("Tensor must have compatible shapes for matrix multiplication (height of left tensor is not matching the width of the right tensor).");

            if(left.Channels !=  right.Channels)
                throw new ArgumentException("Tensors must have similar number of channels for channeled matrix multiplication.");
           
            if(left.Rank != right.Rank)
                throw new ArgumentException("Tensors must have similar rank for matrix multiplication.");

            TensorGPU result = null;
            if (left.Rank == 2)
                result = new(left.Height, right.Width);
            else if(left.Rank == 3)
                result = new(left.Channels, left.Height, right.Width);

            ComputeShader CS = DeepUnityMeta.MatMulCS;

            CS.SetBuffer(0, "leftArr", left.data);
            CS.SetBuffer(0, "rightArr", right.data);
            CS.SetBuffer(0, "resultArr", result.data);

            CS.SetInt("w1", left.Height);
            CS.SetInt("h1w2", left.Width);
            CS.SetInt("h2", right.Width);

            CS.Dispatch(0,
                      (left.Height + DeepUnityMeta.numthreads[0] - 1) / DeepUnityMeta.numthreads[0],
                      (right.Width + DeepUnityMeta.numthreads[1] - 1) / DeepUnityMeta.numthreads[1],
                      (left.Channels + DeepUnityMeta.numthreads[2] - 1) / DeepUnityMeta.numthreads[2]);

            return result;
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

    }
}

