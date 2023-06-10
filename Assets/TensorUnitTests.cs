using DeepUnity;
using UnityEngine;

namespace kbRadu
{
    public class TensorUnitTests : MonoBehaviour
    {
        private void Start()
        {
            SliceJoinTensor();
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

            Tensor[] mats = Tensor.Slice(cube, 2);
            print(mats[0] + "\n----\n" + mats[1]);

            Tensor[] arrays = Tensor.Slice(mats[0], 1);
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

