using DeepUnity;
using UnityEngine;

namespace kbRadu
{
    public class TensorUnitTests : MonoBehaviour
    {
        private void Start()
        {
            var t = Tensor.Normal(2048, 2048);
            print(Tensor.Var(t, 0));
            print(Tensor.Var(t, 1));
            //Transpose();
            //Slicing();
            //SliceJoinTensor();
        }
        public void Transpose()
        {
            var t = Tensor.Random(5, 5);
            print(t);
            print(Tensor.Transpose(t, 0, 1));
            print(Tensor.Transpose(t, 0, 1).ShapeToString);
        }
        public void Slicing()
        {
            var t = Tensor.Random(10, 10);
            var batches = Tensor.Split(t, 0, 2);

            print(t);
            foreach (var b in batches)
            {
                print("firstSlice" + b);
            }

            batches = Tensor.Split(t, 1, 3);
            foreach (var b in batches)
            {
                print("SecondSlice" + b);
            }

        }
        public void SliceJoinTensor()
        {
            Tensor array = Tensor.Random(10);
            Tensor array2 = Tensor.Random(10);

            print(array);
            print(array2);

            Tensor mat = Tensor.Join(1, array, array2);
            print(mat);

            Tensor cube = Tensor.Join(2, mat, mat);
            print(cube);

            Tensor[] mats = Tensor.Split(cube, 2, 1);
            print(mats[0] + "\n----\n" + mats[1]);

            Tensor[] arrays = Tensor.Split(mats[0], 1, 1);
            print(arrays[0] + "\n----\n" + arrays[1]);


            print("expansion");
            
            print(Tensor.Expand(Tensor.Constant(1f), 0, 10));

            print(array);
            print(Tensor.Expand(array, 1, 10));

            mat = Tensor.Random(2, 3);
            print(mat);
            print(Tensor.Expand(mat, 2, 10));

            print("Some mat" + mat);
            print("Transposed mat" + Tensor.Transpose(mat, 0, 1));
            
        }
    }
}

