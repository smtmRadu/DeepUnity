using DeepUnity;
using System;
using System.Linq;
using UnityEngine;

namespace kbRadu
{
    public class NDArrayUnitTests : MonoBehaviour
    {
        private void Start()
        {
            TensorVSNDArray();
            //JoinBenchmark();
            //Shuffle();
            //Transpose();
            //Slicing();
            //SliceJoinTensor();
        }
        public void TensorVSNDArray()
        {
            
            var tx = Tensor.Random(3, 4);
            var ty = Tensor.Random(4, 3);
            print(tx);
            print(ty);
            print(Tensor.MatMul(tx, ty));
            return;
            Tensor t = Tensor.Random(3, 4);
            Tensor t1 = Tensor.Random(4, 3);
            print(t);
            print(t1);
            print(string.Join(",", t.Shape));
            print(string.Join(",", t1.Shape));
            // matmul must be 3x3
            Tensor mm = Tensor.MatMul(t, t1);
            print(mm);
        }
        public void JoinBenchmark()
        {
            NDArray[] slices = Enumerable.Range(0, 1000).Select(x => NDArray.Random(20,29)).ToArray();

            TimeSpan end;
            var start = DateTime.Now;
            NDArray.Join(-1, slices);
            end = DateTime.Now - start;
            Debug.Log(end);
        }
        public void Shuffle()
        {
            var tarr = NDArray.Random(10);
            print(tarr);
            print(NDArray.Shuffle(tarr, 0));

            var cube = NDArray.Random(2, 2, 3);
            print(cube);
            print(NDArray.Shuffle(cube, 2));
        }
        public void Transpose()
        {
            var t = NDArray.Random(5, 5);
            print(t);
            print(NDArray.Transpose(t, 0, 1));
            print(NDArray.Transpose(t, 0, 1).ShapeToString);
        }
        public void Slicing()
        {
            var t = NDArray.Random(10, 10);
            var batches = NDArray.Split(t, 0, 2);

            print(t);
            foreach (var b in batches)
            {
                print("firstSlice" + b);
            }

            batches = NDArray.Split(t, 1, 3);
            foreach (var b in batches)
            {
                print("SecondSlice" + b);
            }

        }
        public void SliceJoinTensor()
        {
            NDArray array = NDArray.Random(10);
            NDArray array2 = NDArray.Random(10);

            print(array);
            print(array2);

            NDArray mat = NDArray.Join(1, array, array2);
            print(mat);

            NDArray cube = NDArray.Join(2, mat, mat);
            print(cube);

            NDArray[] mats = NDArray.Split(cube, 2, 1);
            print(mats[0] + "\n----\n" + mats[1]);

            NDArray[] arrays = NDArray.Split(mats[0], 1, 1);
            print(arrays[0] + "\n----\n" + arrays[1]);


            print("expansion");
            
            print(NDArray.Expand(NDArray.Constant(1f), 0, 10));

            print(array);
            print(NDArray.Expand(array, 1, 10));

            mat = NDArray.Random(2, 3);
            print(mat);
            print(NDArray.Expand(mat, 2, 10));

            print("Some mat" + mat);
            print("Transposed mat" + NDArray.Transpose(mat, 0, 1));
            
        }
    }
}

