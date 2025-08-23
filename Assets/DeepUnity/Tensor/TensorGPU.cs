using System;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEngine;
using System.Collections.Generic;
using DeepUnity.Sensors;

namespace DeepUnity
{
    /// <summary>
    /// A tensor that lives in VRAM. Note that are hard to use, and each new GPU Tensor must be disposed when is no longer used. Also if they are serialized, they must be manually deserialized.
    /// </summary>
    [Serializable] // Initialize on Load but was removed
    public sealed class TensorGPU : ISerializationCallbackReceiver, IDisposable, ICloneable, IEquatable<Tensor>, IEquatable<TensorGPU>
    {
#if UNITY_EDITOR
        static TensorGPU()
        {
            UnityEditor.EditorApplication.playModeStateChanged += DeallocateTensors;
        }
        private readonly static Lazy<Dictionary<TensorGPU, TensorGPU>> AllocatedTensors = new Lazy<Dictionary<TensorGPU, TensorGPU>>();
        private readonly static int MAX_ALLOC_TENSORS = 1024;

        private static void DeallocateTensors(UnityEditor.PlayModeStateChange state)
        {
            if (state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
            {
                if(AllocatedTensors.Value.Count > 0)
                {
                    ConsoleMessage.Info($"<b>{AllocatedTensors.Value.Count}</b> TensorGPUs deallocated. Please dispose TensorGPUs after use");

                    foreach (var item in AllocatedTensors.Value)
                    {
                        item.Value.data.Release();
                       
                    }
                    AllocatedTensors.Value.Clear();
                }
              
            }
        }
#endif



        public ComputeBuffer data;
        [ViewOnly, SerializeField] private float[] serialized_data;
        [ViewOnly, SerializeField] private int[] shape;
        




        // These fields that are used for fast value extraction on indexing. ..
        private ComputeBuffer valueAtIndex;
        private float[] valueAtIndexRecv; 
        // .. From VRAM TO RAM
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
        public int[] Shape
        {
            get => shape.ToArray();
        }
        private int Width
        {
            get
            {
                return shape.Last();
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
            get
            {
                if (valueAtIndex == null)
                {
                    valueAtIndex = new ComputeBuffer(1, 4);
                    valueAtIndexRecv = new float[1];
                }
                ComputeShader cs = DeepUnityMeta.TensorCS;

                cs.SetBuffer(0, "data1", data);
                cs.SetBuffer(0, "result", valueAtIndex);

                cs.SetInt("w1", w);
                cs.SetInt("h1", 1);
                cs.SetInt("c1", 1);
                cs.SetInt("b1", 1);

                cs.Dispatch(0, 1, 1, 1);

                valueAtIndex.GetData(valueAtIndexRecv);
                return valueAtIndexRecv[0];
            }
            set
            {
                ComputeShader cs = DeepUnityMeta.TensorCS;

                cs.SetBuffer(1, "data1", data);
                cs.SetFloat("value", value);

                cs.SetInt("w1", w);
                cs.SetInt("h1", 1);
                cs.SetInt("c1", 1);
                cs.SetInt("b1", 1);

                cs.Dispatch(1, 1, 1, 1);
            }
        }
        public float this[int h, int w]
        {
            get
            {
                if (valueAtIndex == null)
                {
                    valueAtIndex = new ComputeBuffer(1, 4);
                    valueAtIndexRecv = new float[1];
                }
                ComputeShader cs = DeepUnityMeta.TensorCS;

                cs.SetBuffer(0, "data1", data);
                cs.SetBuffer(0, "result", valueAtIndex);

                cs.SetInt("w1", w);
                cs.SetInt("h1", h);
                cs.SetInt("c1", 1);
                cs.SetInt("b1", 1);

                cs.Dispatch(0, 1, 1, 1);

                valueAtIndex.GetData(valueAtIndexRecv);
                return valueAtIndexRecv[0];
            }
            set
            {
                ComputeShader cs = DeepUnityMeta.TensorCS;

                cs.SetBuffer(1, "data1", data);
                cs.SetFloat("value", value);

                cs.SetInt("w1", w);
                cs.SetInt("h1", h);
                cs.SetInt("c1", 1);
                cs.SetInt("b1", 1);

                cs.Dispatch(1, 1, 1, 1);
            }
        }
        public float this[int c, int h, int w]
        {
            get
            {
                if (valueAtIndex == null)
                {
                    valueAtIndex = new ComputeBuffer(1, 4);
                    valueAtIndexRecv = new float[1];
                }
                ComputeShader cs = DeepUnityMeta.TensorCS;

                cs.SetBuffer(0, "data1", data);
                cs.SetBuffer(0, "result", valueAtIndex);

                cs.SetInt("w1", w);
                cs.SetInt("h1", h);
                cs.SetInt("c1", c);
                cs.SetInt("b1", 1);

                cs.Dispatch(0, 1, 1, 1);

                valueAtIndex.GetData(valueAtIndexRecv);
                return valueAtIndexRecv[0];
            }
            set
            {
                ComputeShader cs = DeepUnityMeta.TensorCS;

                cs.SetBuffer(1, "data1", data);
                cs.SetFloat("value", value);

                cs.SetInt("w1", w);
                cs.SetInt("h1", h);
                cs.SetInt("c1", c);
                cs.SetInt("b1", 1);

                cs.Dispatch(1, 1, 1, 1);
            }
        }
        public float this[int n, int c, int h, int w]
        {
            get
            {
                if (valueAtIndex == null)
                {
                    valueAtIndex = new ComputeBuffer(1, 4);
                    valueAtIndexRecv = new float[1];
                }
                ComputeShader cs = DeepUnityMeta.TensorCS;

                cs.SetBuffer(0, "data1", data);
                cs.SetBuffer(0, "result", valueAtIndex);

                cs.SetInt("w1", w);
                cs.SetInt("h1", h);
                cs.SetInt("c1", c);
                cs.SetInt("b1", n);

                cs.Dispatch(0, 1, 1, 1);
                
                valueAtIndex.GetData(valueAtIndexRecv);
                return valueAtIndexRecv[0];
            }
            set
            {
                ComputeShader cs = DeepUnityMeta.TensorCS;

                cs.SetBuffer(1, "data1", data);
                cs.SetFloat("value", value);

                cs.SetInt("w1", w);
                cs.SetInt("h1", h);
                cs.SetInt("c1", c);
                cs.SetInt("b1", h);

                cs.Dispatch(1, 1, 1, 1);
            }
        }


      



        /// <summary>
        /// Deallocates the Tensor from VRAM and destroys the object.
        /// </summary>
        public void Dispose()
        {
#if UNITY_EDITOR
            AllocatedTensors.Value.Remove(this);
#endif
            data?.Release();
            GC.SuppressFinalize(this);

            // When application is built the deallocation from GPU is forced.
        }
        public int Size(int axis)
        {
            if (axis >= 0)
                return shape[axis];
            else
                return shape[shape.Length + axis];
        }
        private TensorGPU(params int[] shape)
        {
            if (shape == null)
                throw new ShapeException("Tensor cannot be instantiated with null shape");
            if (shape.Length == 0)
                throw new ShapeException("Tensor cannot be instantiated with a shape of length 0");
            if (shape.Length > 4)
                throw new ShapeException("Tensor cannot be instantiated with more than 4 dimensions.");
            if (shape.Any(x => x < 1))
                throw new ShapeException("Tensor cannot be instantiated with a dimension < 1.");

            int size = 1;
            foreach (var item in shape)
            {
                size *= item;
            }


            // if (size > 4_194_304) // hardcoded like this because 4096x4096 max allowed matrix, on 8192 it crashes
            //     throw new NotSupportedException("Tensor dimensions is too large on initialization (cannot surpass 4,194,304 units).");

          


            this.shape = shape.ToArray();
            this.data = new ComputeBuffer(size, 4, type: ComputeBufferType.Structured);

            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Zeros");
            cs.SetBuffer(kernel, "result", this.data);
            cs.Dispatch(kernel, 1, 1, 1);

#if UNITY_EDITOR

            AllocatedTensors.Value.Add(this, this);

            if (AllocatedTensors.Value.Count > MAX_ALLOC_TENSORS)
            {
                ConsoleMessage.Error($"Cannot allocate more than {MAX_ALLOC_TENSORS} TensorGPUs at the same time. Make sure there are no memory leaks and all unused TensorGPUs are Disposed!");
                UnityEditor.EditorApplication.isPlaying = false;
             }
#endif
        }
        public static TensorGPU Identity(TensorGPU other)
        {
            TensorGPU clone = new(other.shape);
            clone.data.SetData(other.ToArray());
            return clone;
        }
        public static TensorGPU Identity(Tensor other)
        {
            TensorGPU clone = new(other.Shape);
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
            float[] dataarr = new float[] { scalar };
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
        public static TensorGPU RandomNormal(params int[] shape)
        {
            return RandomNormal((0, 1), shape);
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

        public static void CopyTo(TensorGPU fromTensor, TensorGPU toTensor)
        {
            Array arr = new float[fromTensor.data.count];
            fromTensor.data.GetData(arr);
            toTensor.data.SetData(arr);
        }
        public static TensorGPU MatMul(TensorGPU left, TensorGPU right)
        {
            int left_rank = left.Rank;
            int right_rank = right.Rank;

            if (left_rank == 1 && right_rank == 1)
                return TensorGPU.Constant(left[0] * right[0]);

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

            TensorGPU result;
            if (left_rank == 1)
                result = new(CreateShape(left.Rank, J, K, 1, P));
            else if (right_rank == 1)
                result = new(CreateShape(left.Rank, J, K, 1, N));
            else
                result = new(CreateShape(left.Rank, J, K, N, P));


            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("MatMul");

            cs.SetBuffer(kernel, "data1", left.data);
            cs.SetBuffer(kernel, "data2", right.data);
            cs.SetBuffer(kernel, "result", result.data);

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
        public static TensorGPU BatchedMatMul(TensorGPU left, TensorGPU right)
        {

            int C = left.Channels;
            int N = left.Height;
            int M = left.Width;
            int P = right.Width;

            ComputeShader cs = DeepUnityMeta.TensorCS;

            TensorGPU result = new(C, N, P);

            int kernel = cs.FindKernel("BatchedMatMul");

            cs.SetBuffer(kernel, "data1", left.data);
            cs.SetBuffer(kernel, "data2", right.data);
            cs.SetBuffer(kernel, "result", result.data);

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

            return result;
        }

        #region Static operations  
        public static TensorGPU Concat(int? axis, params TensorGPU[] tensors)
        {
            Tensor result = Tensor.Concat(axis, tensors.Select(x => Tensor.Identity(x)).ToArray());
            return TensorGPU.Identity(result);
        }
        public static TensorGPU Expand(TensorGPU tensor, int axis, int times)
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
            TensorGPU result = new(shapex);

            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Expand");

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
            cs.SetInt("rr", result.Rank);

            cs.SetInt("axis", axis);
            cs.SetInt("times", times);

            cs.Dispatch(kernel, 1, 1, 1);

            return result;
        }
        public static TensorGPU Transpose(TensorGPU tensor, int axis0, int axis1)
        {
            HandleAxis(tensor, ref axis0);
            HandleAxis(tensor, ref axis1);

            if (axis0 == axis1)
                return Identity(tensor);

            int axis0_index = (int)AxisToDim(tensor, axis0);
            int axis1_index = (int)AxisToDim(tensor, axis1);
            int[] permutation = new int[] { tensor.Batch, tensor.Channels, tensor.Height, tensor.Width };

            var temp = permutation[axis0_index];
            permutation[axis0_index] = permutation[axis1_index];
            permutation[axis1_index] = temp;

            TensorGPU result = new(CreateShape(tensor.Rank, permutation[0], permutation[1], permutation[2], permutation[3]));


            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Transpose");

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
            cs.SetInt("rr", result.Rank);

            cs.SetInt("axis0", axis0);
            cs.SetInt("axis1", axis1);

            cs.Dispatch(kernel, 1, 1, 1);

            return result;
        }

        // These part was let like this. If you wanna create another tensor with this functions applied, just clone yours before that().
        private static TensorGPU Flatten(TensorGPU tensor, int startAxis = 0, int endAxis = -1)
        {
            if (startAxis > endAxis)
                throw new Exception($"Start axis ({startAxis}) must be greater or equal to the end axis ({endAxis}) when flattening.");

            HandleAxis(tensor, ref startAxis);
            HandleAxis(tensor, ref endAxis);

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

            TensorGPU result = new TensorGPU(newShape.ToArray());
            result.data.SetData(tensor.ToArray());
            return result;
        }
        private static TensorGPU Reshape(TensorGPU tensor, params int[] newShape)
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
            result.data.SetData(tensor_data);
            return result;
        }
        private static TensorGPU Squeeze(TensorGPU tensor, int? axis = null)
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

                TensorGPU result = new(squeezedShape.ToArray());
                float[] dataarr = new float[tensor.data.count];
                tensor.data.GetData(dataarr);
                result.data.SetData(dataarr);
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
                List<int> squeezedShape = tensor.shape.ToList();
                if (squeezedShape.Count > 1)
                    squeezedShape.RemoveAt(ax);

                TensorGPU result = new(squeezedShape.ToArray());
                float[] dataarr = new float[tensor.data.count];
                tensor.data.GetData(dataarr);
                result.data.SetData(dataarr);
                return result;
            }

        }
        private static TensorGPU Unsqueeze(TensorGPU tensor, int axis)
        {
            HandleAxis(tensor, ref axis);

            List<int> unsqueezedShape = tensor.shape.ToList();
            unsqueezedShape.Insert(axis, 1);
            TensorGPU result = new(unsqueezedShape.ToArray());
            float[] dataarr = new float[tensor.data.count];
            tensor.data.GetData(dataarr);
            result.data.SetData(dataarr);
            return result;
        }
        private static TensorGPU Mean(TensorGPU tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
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
            cs.SetInt("rr", result.Rank);

            cs.SetInt("axis", axis);

            cs.Dispatch(kernel, 1, 1, 1);

            if (!keepDim)
                return Squeeze_(result, axis);
            
                
            return result;
        }
        private static TensorGPU Sum(TensorGPU tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
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
            cs.SetInt("rr", result.Rank);

            cs.SetInt("axis", axis);

            cs.Dispatch(kernel, 1, 1, 1);

            if (!keepDim)
                return Squeeze_(result, axis);
            return result;
        }
        private static TensorGPU Var(TensorGPU tensor, int axis, int correction = 1, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
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
            cs.SetInt("rr", result.Rank);


            cs.SetInt("axis", axis);
            cs.SetInt("correction", correction);

            cs.Dispatch(kernel, 1, 1, 1);

            if (!keepDim)
                return Squeeze_(result, axis);
            return result;
        }
        private static TensorGPU Std(TensorGPU tensor, int axis, int correction = 1, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
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
            cs.SetInt("rr", result.Rank);


            cs.SetInt("axis", axis);
            cs.SetInt("correction", correction);

            cs.Dispatch(kernel, 1, 1, 1);

            if (!keepDim)
                return Squeeze_(result, axis);
            return result;
        }
        private static TensorGPU Min(TensorGPU tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
            TensorGPU result = new TensorGPU(newShape);


            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Min");

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
            cs.SetInt("rr", result.Rank);

            cs.SetInt("axis", axis);

            cs.Dispatch(kernel, 1, 1, 1);

            if (!keepDim)
                return Squeeze_(result, axis);
            return result;
        }
        private static TensorGPU Max(TensorGPU tensor, int axis, bool keepDim = false)
        {
            HandleAxis(tensor, ref axis);

            int[] newShape = tensor.shape.ToArray();
            newShape[axis] = 1;
            TensorGPU result = new TensorGPU(newShape);

            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Max");

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
            cs.SetInt("rr", result.Rank);

            cs.SetInt("axis", axis);

            cs.Dispatch(kernel, 1, 1, 1);

            if (!keepDim)
                return Squeeze_(result, axis);
            return result;
        }
        private static TensorGPU Pow(TensorGPU tensor, float power)
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
        private static TensorGPU Sqrt(TensorGPU tensor)
        {
            TensorGPU t = new(tensor.shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Sqrt");

            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.SetBuffer(kernel, "result", t.data);

            cs.Dispatch(kernel, 1, 1, 1);

            return t;
        }
        private static TensorGPU Exp(TensorGPU tensor)
        {
            TensorGPU t = new(tensor.shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Exp");

            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.SetBuffer(kernel, "result", t.data);

            cs.Dispatch(kernel, 1, 1, 1);

            return t;
        }
        private static TensorGPU Log(TensorGPU tensor, float @base = MathF.E)
        {
            if (@base != MathF.E && @base != 2f && @base != 10f)
                throw new ArgumentException("Supported base value for Log() is E, 2 or 10.");

            TensorGPU t = new(tensor.shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Log");

            cs.SetFloat("base", @base);
            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.SetBuffer(kernel, "result", t.data);

            cs.Dispatch(kernel, 1, 1, 1);

            return t;
        }
        private static TensorGPU Abs(TensorGPU tensor)
        {
            TensorGPU t = new(tensor.shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Abs");

            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.SetBuffer(kernel, "result", t.data);

            cs.Dispatch(kernel, 1, 1, 1);

            return t;
        }
        private static TensorGPU Sin(TensorGPU tensor)
        {
            TensorGPU t = new(tensor.shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Sin");

            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.SetBuffer(kernel, "result", t.data);

            cs.Dispatch(kernel, 1, 1, 1);

            return t;
        }
        private static TensorGPU Cos(TensorGPU tensor)
        {
            TensorGPU t = new(tensor.shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Cos");

            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.SetBuffer(kernel, "result", t.data);

            cs.Dispatch(kernel, 1, 1, 1);

            return t;
        }
        private static TensorGPU Clip(TensorGPU tensor, float min, float max)
        {
            TensorGPU t = new(tensor.shape);
            ComputeShader cs = DeepUnityMeta.TensorCS;

            int kernel = cs.FindKernel("Clip");

            cs.SetFloat("minvalue", min);
            cs.SetFloat("maxvalue", max);
            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.SetBuffer(kernel, "result", t.data);

            cs.Dispatch(kernel, 1, 1, 1);

            return t;
        }





        #endregion Static operations


        #region InPlace
        public static TensorGPU Flatten_(TensorGPU tensor, int startAxis = 0, int endAxis = -1)
        {
            if (startAxis > endAxis)
                throw new Exception($"Start axis ({startAxis}) must be greater or equal to the end axis ({endAxis}) when flattening.");

            HandleAxis(tensor, ref startAxis);
            HandleAxis(tensor, ref endAxis);

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

            tensor.shape = newShape.ToArray();
            return tensor;
        }
        public static TensorGPU Reshape_(TensorGPU tensor, params int[] newShape)
        {
            int count = 1;
            foreach (var item in newShape)
            {
                count *= item;
            }

            if (count != tensor.Count())
                throw new ArgumentException("The new shape must provide the same capacity of the tensor when reshaping it.");

            tensor.shape = newShape;

            return tensor;
        }
        public static TensorGPU Squeeze_(TensorGPU tensor, int? axis)
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
                return tensor;
            }
            else
            {
                int ax = axis.Value;
                HandleAxis(tensor, ref ax);


                // if axis is not 1, tensor remains unchanged
                if (tensor.shape[ax] != 1)
                    return Identity(tensor);

                // Esle remove that axis
                List<int> squeezedShape = tensor.shape.ToList();
                if (squeezedShape.Count > 1)
                    squeezedShape.RemoveAt(ax);

                tensor.shape = squeezedShape.ToArray();
                return tensor;
            }
        }
        public static TensorGPU Unsqueeze_(TensorGPU tensor, int axis)
        {
            HandleAxis(tensor, ref axis);

            List<int> unsqueezedShape = tensor.shape.ToList();
            unsqueezedShape.Insert(axis, 1);
            tensor.shape = unsqueezedShape.ToArray();
            return tensor;
        }
        public static TensorGPU Subtract_(TensorGPU tensor, float value)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("SubtractSingle_");

            cs.SetFloat("value", value);
            cs.SetBuffer(kernel, "data1", tensor.data);

            cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            return tensor;
        }
        public static TensorGPU Subtract_(TensorGPU tensor, TensorGPU other, float alpha = 1)
        {
            if (!tensor.shape.SequenceEqual(other.shape))
                throw new ArgumentException("Tensor shapes do not matches");

            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Subtract_");

            cs.SetFloat("alpha", alpha);
            cs.SetBuffer(kernel, "data1", tensor.data);

            if(tensor.data == other.data) // self subtraction
            {
                var x = TensorGPU.Identity(other);
                cs.SetBuffer(kernel, "data2", x.data);
                cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);
                x.Dispose();
            }
            else
            {
                cs.SetBuffer(kernel, "data2", other.data);
                cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            }
            

          
            return tensor;
        }
        public static TensorGPU Add_(TensorGPU tensor, float value)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("AddSingle_");

            cs.SetFloat("value", value);
            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            return tensor;
        }
        public static TensorGPU Add_(TensorGPU tensor, TensorGPU other, float alpha = 1)
        {
            if (!tensor.shape.SequenceEqual(other.shape))
                throw new ArgumentException("Tensor shapes do not matches");

            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Add_");

            cs.SetFloat("alpha", alpha);
            cs.SetBuffer(kernel, "data1", tensor.data);

            if (tensor.data == other.data) // self subtraction
            {
                var x = TensorGPU.Identity(other);
                cs.SetBuffer(kernel, "data2", x.data);
                cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);
                x.Dispose();
            }
            else
            {
                cs.SetBuffer(kernel, "data2", other.data);
                cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            }

            return tensor;
        }
        public static TensorGPU Multiply_(TensorGPU tensor, float value)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("MultiplySingle_");

            cs.SetFloat("value", value);
            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            return tensor;
        }
        public static TensorGPU HadamardMultiply_(TensorGPU tensor, TensorGPU other, float alpha = 1)
        {
            if (!tensor.shape.SequenceEqual(other.shape))
                throw new ArgumentException("Tensor shapes do not matches");

            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("HadamardMultiply_");

            cs.SetFloat("alpha", alpha);
            cs.SetBuffer(kernel, "data1", tensor.data);
            if (tensor.data == other.data) // self multiplcation
            {
                var x = TensorGPU.Identity(other);
                cs.SetBuffer(kernel, "data2", x.data);
                cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);
                x.Dispose();
            }
            else
            {
                cs.SetBuffer(kernel, "data2", other.data);
                cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            }
            return tensor;
        }
        public static TensorGPU Divide_(TensorGPU tensor, float value)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("DivideSingle_");

            cs.SetFloat("value", value);
            cs.SetBuffer(kernel, "data1", tensor.data);
            cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            return tensor;
        }
        public static TensorGPU Divide_(TensorGPU tensor, TensorGPU other, float alpha = 1)
        {
            if (!tensor.shape.SequenceEqual(other.shape))
                throw new ArgumentException("Tensor shapes do not matches");

            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Divide_");

            cs.SetFloat("alpha", alpha);
            cs.SetBuffer(kernel, "data1", tensor.data);
            if (tensor.data == other.data) // self division
            {
                var x = TensorGPU.Identity(other);
                cs.SetBuffer(kernel, "data2", x.data);
                cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);
                x.Dispose();
            }
            else
            {
                cs.SetBuffer(kernel, "data2", other.data);
                cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            }
            return tensor;
        }

        public static TensorGPU Pow_(TensorGPU tensor, float pow)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Pow_");

            cs.SetFloat("power", pow);
            cs.SetBuffer(kernel, "data1", tensor.data);

            cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            return tensor;
        }
        public static TensorGPU Sqrt_(TensorGPU tensor)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Sqrt_");

            cs.SetBuffer(kernel, "data1", tensor.data);

            cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            return tensor;
        }
        public static TensorGPU Log_(TensorGPU tensor)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Log_");

            cs.SetBuffer(kernel, "data1", tensor.data);

            cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            return tensor;
        }
        public static TensorGPU Exp_(TensorGPU tensor)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Exp_");

            cs.SetBuffer(kernel, "data1", tensor.data);

            cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            return tensor;
        }
        public static TensorGPU Sin_(TensorGPU tensor)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Sin_");

            cs.SetBuffer(kernel, "data1", tensor.data);

            cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            return tensor;
        }
        public static TensorGPU Cos_(TensorGPU tensor)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Cos_");

            cs.SetBuffer(kernel, "data1", tensor.data);

            cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            return tensor;
        }
        public static TensorGPU Clip_(TensorGPU tensor, float min, float max)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Clip_");

            cs.SetFloat("minvalue", min);
            cs.SetFloat("maxvalue", max);
            cs.SetBuffer(kernel, "data1", tensor.data);

            cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            return tensor;
        }
        public static TensorGPU Maximum_(TensorGPU tensor, TensorGPU other)
        {
            if (!tensor.shape.SequenceEqual(other.shape))
                throw new ArgumentException("Tensor shapes do not match");

            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Maximum_");

            cs.SetBuffer(kernel, "data1", tensor.data);
            if (tensor.data == other.data) // self subtraction
            {
                var x = TensorGPU.Identity(other);
                cs.SetBuffer(kernel, "data2", x.data);
                cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);
                x.Dispose();
            }
            else
            {
                cs.SetBuffer(kernel, "data2", other.data);
                cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            }
            return tensor;
        }
        public static TensorGPU Zero_(TensorGPU tensor)
        {
            ComputeShader cs = DeepUnityMeta.TensorCS;
            int kernel = cs.FindKernel("Zero_");

            cs.SetBuffer(kernel, "data1", tensor.data);

            cs.Dispatch(kernel, (tensor.Count() + DeepUnityMeta.THREADS_NUM - 1) / DeepUnityMeta.THREADS_NUM, 1, 1);

            return tensor;
        }
        #endregion InPlace




        // other
        public int Count(Func<float, bool> predicate = null)
        {         
            if (predicate == null)
            {
                return data.count;
            }
            else
            {
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
        public object Clone()
        {
            return Identity(this);
        }
        public override string ToString()
        {
            int rank = Rank;
            float[] arrdata = new float[Count()];
            data.GetData(arrdata);

            StringBuilder sb = new();

            sb.Append($"Tensor({shape.ToCommaSeparatedString()}) [GPU]");

            


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
                            sb.Append(arrdata[index].ToString(StringFormat));
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
     



        public void OnBeforeSerialize()
        {
            if (data == null)
                return;

            serialized_data = new float[data.count];
            data.GetData(serialized_data);
            data.Dispose();
        }
        public void OnAfterDeserialize()
        {
            if (serialized_data.Length == 0)
                return;

            data = new ComputeBuffer(serialized_data.Length, 4);
            data.SetData(serialized_data);
#if UNITY_EDITOR
            AllocatedTensors.Value.Add(this, this);
#endif
        }





        // inside use      
        private static void HandleAxis(TensorGPU tensor, ref int axis)
        {
            int rank = tensor.Rank;

            if (rank == 0)
            {
                // here are different things
                if (axis != 0 && axis != -1)
                    throw new ArgumentOutOfRangeException($"Invalid axis value ({axis}) for a tensor with rank ({tensor.Rank})");

                axis = 0;
            }
            else
            {
                if (axis >= rank)
                    throw new ArgumentOutOfRangeException($"Invalid axis value ({axis}) for a tensor with rank ({tensor.Rank})");

                if (axis < 0)
                    axis = rank + axis;
            }
        }
        private static Dim AxisToDim(TensorGPU t, int axis)
        {
            // Returns the index in the full shape array of the axis. ([0,1,2,3])
            // Used only for methods along the axis, that uses full shape call.
            int rank = t.Rank;

            if (axis > rank)
                throw new ArgumentException($"Cannot use axis {axis} for a tensor of rank {rank}.");

            // check for rank 0
            if (rank == 0 && (axis == 0 || axis == -1))
                return (Dim)3;

            // check for negative axis as well
            if (axis >= 0)
                return (Dim)4 - rank + axis;
            else
                return (Dim)4 + axis;

        }
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
        public static string StringFormat { get; set; } = "0.00000";
    }
}

