using System;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;


namespace DeepUnity
{
    [Serializable]
    public class TensorGPU : IDisposable, IEquatable<TensorGPU>, ISerializationCallbackReceiver
    {
        private ComputeBuffer data;
        [SerializeField] private float[] serialized_data;
        [SerializeField] private int[] shape;
        private bool disposed = false;
     
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
        public int[] Shape
        {
            get => shape.ToArray();
        }    
        public int Size(int axis)
        {
            if (axis >= 0)
                return shape[axis];
            else
                return shape[shape.Length + axis];
        }

        // Create
        private TensorGPU(params int[] shape)
        {
            if (shape == null)
                throw new ArgumentException("Tensor cannot be instantiated with null ");
            if (shape.Length == 0)
                throw new ArgumentException("Tensor cannot be instantiated with a shape of length 0");
            if (shape.Length > 4)
                throw new ArgumentException("Tensor cannot be instantiated with more than 4 dimensions.");
            if (shape.Any(x => x < 1))
                throw new ArgumentException("Tensor cannot be instantiated with a dimension < 1.");

            int size = 1;
            foreach (var item in shape)
            {
                size *= item;
            }

            if (size > 4_194_304) // hardcoded like this because 4096x4096 max allowed matrix, on 8192 it crashes
                throw new NotSupportedException("Tensor dimensions is too large on initialization (cannot surpass 16,777,216 units).");


            this.shape = shape.ToArray();
            this.data = new ComputeBuffer(size, 4);

            TensorGPUDisposer.tensors.AddLast(this);
            GC.Collect();
        }
        public static TensorGPU Reshape(TensorGPU tensor, params int[] newShape)
        {
            int count = 1;
            foreach (var item in newShape)
            {
                count *= item;
            }

            if (count != tensor.Count())
                throw new ArgumentException("The new shape must provide the same capacity of the tensor when reshaping it.");

            TensorGPU result = new TensorGPU(newShape);

            float[] tensor_data = new float[tensor.Count()];
            tensor.data.GetData(tensor_data);
            float[] reshaped_data = new float[tensor.Count()];

            int batch = result.Batch;
            int channels = result.Channels;
            int height = result.Height;
            int width = result.Width;

            int index = 0;
            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            reshaped_data[b * channels * height * width +  c * height * width + h * width + w] = tensor_data[index++];
                        }
                    }
                }
            }

            result.data.SetData(reshaped_data);
            return result;
        }
        public static TensorGPU Identity(TensorGPU other)
        {
            TensorGPU clone = new(other.shape);
            clone.data.SetData(other.ToArray());
            return clone;
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
        public static TensorGPU Constant(float scalar)
        {
            TensorGPU t = new(1);
            float[] dataarr = new float[] { 1 };
            t.data.SetData(dataarr);
            return t;
        }
        public static TensorGPU Constant(float[] vector)
        {

            int width = vector.GetLength(0);

            TensorGPU t = new(width);

            float[] dataarr = new float[width];
            int index = 0;
            for (int i = 0; i < width; i++)
            {

                    dataarr[index++] = vector[i];
                

            }
            t.data.SetData(dataarr);

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
        public static TensorGPU Constant(float[,,] cube)
        {
            int depth = cube.GetLength(0);
            int height = cube.GetLength(1);
            int width = cube.GetLength(2);

            TensorGPU t = new TensorGPU(depth, height, width);

            float[] dataarr = new float[depth * height * width];
            int index = 0;
            for (int d = 0; d < depth; d++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        dataarr[index++] = cube[d, h, w];
                    }
                }
            }
            t.data.SetData(dataarr);

            return t;
        }
        public static TensorGPU Constant(float[,,,] tesseract)
        {
            int depth = tesseract.GetLength(0);
            int height = tesseract.GetLength(1);
            int width = tesseract.GetLength(2);
            int extraDimension = tesseract.GetLength(3);

            TensorGPU t = new TensorGPU(depth, height, width, extraDimension);

            float[] dataarr = new float[depth * height * width * extraDimension];
            int index = 0;
            for (int d = 0; d < depth; d++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        for (int e = 0; e < extraDimension; e++)
                        {
                            dataarr[index++] = tesseract[d, h, w, e];
                        }
                    }
                }
            }
            t.data.SetData(dataarr);

            return t;
        }
        public static TensorGPU Zeros(params int[] shape)
        {
            return new(shape);
        }
        public static TensorGPU Ones(params int[] shape)
        {
            TensorGPU t = new(shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Ones");
            cs.SetBuffer(kernel, "result", t.data);
            cs.Dispatch(kernel, 1, 1, 1);
            return t;
        }
        public static TensorGPU Fill(float value, params int[] shape)
        {
            TensorGPU t = new(shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Fill");
            cs.SetFloat("value", value);
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
        public static TensorGPU RandomNormal((float, float) mu_sigma, params int[] shape)
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
        public static TensorGPU RandomRange((float, float) min_max, params int[] shape)
        {
            TensorGPU t = new(shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("RandomRange");
            cs.SetInt("seed", DateTime.Now.Millisecond);
            cs.SetBuffer(kernel, "result", t.data);
            cs.SetFloat("minvalue", min_max.Item1);
            cs.SetFloat("maxvalue", min_max.Item2);

            cs.Dispatch(kernel, 1, 1, 1);
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


            TensorGPU result = new(CreateShape(left.Rank, left.Batch, right.Channels, left.Height, right.Width));

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



            // Squeezing the result fast***
            LinkedList<int> squeezedShape = new LinkedList<int>();

            squeezedShape.AddFirst(result.Width);

            if (result.Batch > 1)
            {
                squeezedShape.AddFirst(result.Height);
                squeezedShape.AddFirst(result.Channels);
                squeezedShape.AddFirst(result.Batch);

            }
            else if (result.Channels > 1)
            {
                squeezedShape.AddFirst(result.Height);
                squeezedShape.AddFirst(result.Channels);

            }
            else if (result.Width > 1)
            {
                squeezedShape.AddFirst(result.Height);
            }

            result.shape = squeezedShape.ToArray();

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
            TensorGPU t = new(tensor.shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Pow");

            cs.SetFloat("power", power);
            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.SetBuffer(kernel, "result", t.data);

            cs.Dispatch(kernel, 1, 1, 1);

            return t;
        }


        // other
        public int Count(Func<float, bool> predicate = null)
        {         
            if (predicate == null)
            {
                return data.count;
            }
            else
            {
                int count = 0;
                float[] data_cpu = new float[data.count];
                data.GetData(data_cpu);

                return data_cpu.Count(predicate);
            }

        }
        public float Min(Func<float, float> selector = null)
        {
            float[] data_cpu = new float[data.count];
            data.GetData(data_cpu);

            if (selector == null)
                return data_cpu.Min();
            else
                return data_cpu.Min(selector);
            
        }
        public float Max(Func<float, float> selector = null)
        {
            float[] data_cpu = new float[data.count];
            data.GetData(data_cpu);

            if (selector == null)
                return data_cpu.Max();
            else
                return data_cpu.Max(selector);

        }
        public float[] ToArray()
        {
            float[] tosend = new float[data.count];
            data.GetData(tosend);
            return tosend;
        }       
        public bool Equals(TensorGPU other)
        {
            if (!shape.SequenceEqual(other.shape))
                return false;

            if (!ToArray().SequenceEqual(other.ToArray()))
                return false;

            return true;
        }
        public bool Equals(Tensor other)
        {
            if (!shape.SequenceEqual(other.Shape))
                return false;

            if (!ToArray().SequenceEqual(other.ToArray()))
                return false;

            return true;
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
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
        public void Dispose()
        {
            if (!disposed)
            {
                data.Release();
                disposed = true;
            }

            GC.SuppressFinalize(this);
        }
        public void OnBeforeSerialize()
        {
            serialized_data = new float[data.count];
            data.GetData(serialized_data);
        }
        public void OnAfterDeserialize()
        {
            data = new ComputeBuffer(serialized_data.Length, 4);
            disposed = false;
            data.SetData(serialized_data);
            TensorGPUDisposer.tensors.AddLast(this);
        }

        // inside use
        private static int[] CreateShape(int rank, int b, int c, int h, int w)
        {
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


    [InitializeOnLoad]
    internal sealed class TensorGPUDisposer
    {
        public readonly static LinkedList<TensorGPU> tensors = new LinkedList<TensorGPU>();
        static TensorGPUDisposer()
        {
            EditorApplication.playModeStateChanged += DisposeAllTensors;
        }
        private TensorGPUDisposer() { }
        private static void DisposeAllTensors(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.ExitingPlayMode)
            {
                foreach (var item in tensors)
                {
                    item?.Dispose();
                }
            }
        }
    }
}

