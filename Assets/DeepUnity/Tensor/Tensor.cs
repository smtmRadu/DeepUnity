using System;
using System.Linq;
using System.Text;

namespace DeepUnity
{
    public class Tensor : NDArray
    {
        private static int[] ReversedShape(int[] shape)
        {
            int[] rev_shape = shape.ToArray();
            Array.Reverse(rev_shape);
            return rev_shape;
        }
        private static int ReversedAxis(int rank, int axis)
        {
            if (axis >= 0)
                return rank - 1 - axis;
            else
                return -1 - axis;
        }

        private Tensor(NDArray ndarray)
        {
            this.shape = ndarray.Shape;
            this.data = ndarray.Data;
        }
        private Tensor(params int[] short_shape) : base(ReversedShape(short_shape)) { }
        public static Tensor Identity(Tensor other) => new Tensor(NDArray.Identity(other));
        public new static Tensor Constant(float scalar) => new Tensor(NDArray.Constant(scalar));
        public new static Tensor Constant(float[,] matrix) => new Tensor(NDArray.Constant(matrix));
        public new static Tensor Constant(float[,,] cuboid) => new Tensor(NDArray.Constant(cuboid));
        public new static Tensor Constant(float[,,,] tesseract) => new Tensor(NDArray.Constant(tesseract));
        public new static Tensor Zeros(params int[] shape) => new Tensor(NDArray.Zeros(ReversedShape(shape)));
        public new static Tensor Ones(params int[] shape) => new Tensor(NDArray.Ones(ReversedShape(shape)));
        public new static Tensor Random(params int[] shape) => new Tensor(NDArray.Random(ReversedShape(shape)));
        public new static Tensor RandomNormal(params int[] shape) => new Tensor(NDArray.RandomNormal(ReversedShape(shape)));
        public new static Tensor Fill(float value, params int[] shape) => new Tensor(NDArray.Fill(value, ReversedShape(shape)));

        /// <summary>
        /// Dot product between a Tensor n x m with another Tensor m x p.
        /// </summary>
        public static Tensor MatMul(Tensor left, Tensor right) => new Tensor(NDArray.MatMul(Transpose(left, 1, 0), Transpose(right, 1, 0)));
        public static Tensor Transpose(Tensor tensor, int dim0, int dim1) => new Tensor(NDArray.Transpose(tensor, ReversedAxis(tensor.Rank, dim0), ReversedAxis(tensor.Rank, dim1)));
        public static Tensor Join(int dim, params Tensor[] tensors) => new Tensor(NDArray.Join(ReversedAxis(tensors[0].Rank, dim), tensors));
        public static Tensor[] Split(Tensor tensor, int dim, int split_size) => NDArray.Split(tensor, ReversedAxis(tensor.Rank, dim), split_size).Select(x => new Tensor(x)).ToArray();
        public static Tensor Shuffle(Tensor tensor, int dim) => new Tensor(NDArray.Shuffle(tensor, ReversedAxis(tensor.Rank, dim)));
        public static Tensor Sum(Tensor tensor, int dim) => new Tensor(NDArray.Sum(tensor, ReversedAxis(tensor.Rank, dim)));
        public static Tensor Mean(Tensor tensor, int dim) => new Tensor(NDArray.Mean(tensor, ReversedAxis(tensor.Rank, dim)));
        public static Tensor Std(Tensor tensor, int dim, int correction) => new Tensor(NDArray.Std(tensor, ReversedAxis(tensor.Rank, dim), correction));
        public static Tensor Var(Tensor tensor, int dim, int correction) => new Tensor(NDArray.Var(tensor, ReversedAxis(tensor.Rank, dim), correction));


        public static Tensor Exp(Tensor tensor) => new Tensor(NDArray.Exp(tensor));
        public static Tensor Pow(Tensor @base, float power) => new Tensor(NDArray.Pow(@base, power));
        public static Tensor Sqrt(Tensor @base) => new Tensor(NDArray.Sqrt(@base));
        public static Tensor Abs(Tensor tensor) => new Tensor(NDArray.Abs(tensor));
        public static Tensor Cos(Tensor tensor) => new Tensor(NDArray.Cos(tensor));
        public static Tensor Sin(Tensor tensor) => new Tensor(NDArray.Sin(tensor)); 
        public static Tensor Clip(Tensor tensor, float min, float max) => new Tensor(NDArray.Clip(tensor, min, max));
        public static Tensor Minimum(Tensor tensor1, Tensor tensor2) => new Tensor(NDArray.Minimum(tensor1, tensor2));
        public static Tensor Maximum(Tensor tensor1, Tensor tensor2) => new Tensor(NDArray.Maximum(tensor1, tensor2));
        public static Tensor Norm(Tensor tensor, NormType normType = NormType.ManhattanL1) => new Tensor(NDArray.Norm(tensor, normType));
        public static Tensor Min(Tensor tensor) => new Tensor(NDArray.Min(tensor));
        public static Tensor Max(Tensor tensor) => new Tensor(NDArray.Min(tensor));
        
        public new int Rank => base.Rank;
        public new int[] Shape
        {
            get
            {
                return ReversedShape(base.Shape);
            }
        }
        public new string ShapeToString
        {
            get
            {
                StringBuilder sb = new StringBuilder();
                sb.Append('[');
                int rank = Rank;
                if (rank == 0 || rank == 1)
                {
                    sb.Append(Shape[3]);
                }
                else if (rank == 2)
                {
                    sb.Append(Shape[2]);
                    sb.Append(", ");
                    sb.Append(Shape[3]);
                }
                else if (rank == 3)
                {
                    sb.Append(Shape[1]);
                    sb.Append(", ");
                    sb.Append(Shape[2]);
                    sb.Append(", ");
                    sb.Append(Shape[3]);
                }
                else if (rank == 4)
                {
                    sb.Append(Shape[0]);
                    sb.Append(", ");
                    sb.Append(Shape[1]);
                    sb.Append(", ");
                    sb.Append(Shape[2]);
                    sb.Append(", ");
                    sb.Append(Shape[3]);
                }

                sb.Append(']');

                return sb.ToString();
            }
        }
        public new float this[int w]
        {
            get => base[w];
            set => base[w] = value;
        }
        public new float this[int h, int w]
        {
            get => base[h, w];
            set => base[h, w] = value;
        }
        public new float this[int b, int h, int w]
        {
            get => base[b, h, w];
            set => base[b, h, w] = value;
        }
        public new float this[int n, int b, int h, int w]
        {
            get => base[n, b, h, w];
            set => base[n, b, h, w] = value;
        }

        public bool Equals(Tensor other) => base.Equals(other);
        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
                return false;

            return Equals(obj as Tensor);
        }
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

      
        
    }
}

