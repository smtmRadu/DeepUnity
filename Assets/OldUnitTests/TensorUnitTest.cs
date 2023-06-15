using DeepUnity;
using System;
using System.Linq;
using UnityEngine;

namespace kbRadu
{
    public class TensorUnitTest : MonoBehaviour
    {
      /*  private void Start()
        {
            FastCast();
            //MatPad();
            //TensorVSNDArray();
            //JoinBenchmark();
            //Shuffle();
            //Transpose();
            //Slicing();
            //SliceJoinTensor();
        }
        public void MatPad()
        {
            var d = InversedTensor.Random(3, 3);
            print(d);
            print(InversedTensor.MatPad(d, 1, PaddingType.Mirror));
            print(InversedTensor.MatPad(d, 2, PaddingType.Mirror));

        }

        public void FastCast()
        {
            float[,] twoDimensionalArray = new float[3, 4]
            {
                { 1.0f, 2.0f, 3.0f, 4.0f },
                { 5.0f, 6.0f, 7.0f, 8.0f },
                { 9.0f, 10.0f, 11.0f, 12.0f }
            };

            float[] oneDimensionalArray = twoDimensionalArray.Cast<float>().ToArray();

            InversedTensor first = InversedTensor.Constant(twoDimensionalArray);
            InversedTensor second = InversedTensor.Constant(oneDimensionalArray);

            print(Utils.StringOf(first.Data));
            print(Utils.StringOf(second.Data));

        }
        public void TensorVSNDArray()
        {

            Tensor tensor = Tensor.RandomNormal(20, 5, 5);
            print(Tensor.Mean(tensor, -1));
            return;
        }
        public void JoinBenchmark()
        {
            InversedTensor[] slices = Enumerable.Range(0, 1000).Select(x => InversedTensor.Random(20,29)).ToArray();

            TimeSpan end;
            var start = DateTime.Now;
            InversedTensor.Join(-1, slices);
            end = DateTime.Now - start;
            Debug.Log(end);
        }
        public void Shuffle()
        {
            var tarr = InversedTensor.Random(10);
            print(tarr);
            print(InversedTensor.Shuffle(tarr, 0));

            var cube = InversedTensor.Random(2, 2, 3);
            print(cube);
            print(InversedTensor.Shuffle(cube, 2));
        }
        public void Transpose()
        {
            var t = InversedTensor.Random(5, 5);
            print(t);
            print(InversedTensor.Transpose(t, 0, 1));
            print(InversedTensor.Transpose(t, 0, 1).ShapeToString);
        }
        public void Slicing()
        {
            var t = InversedTensor.Random(10, 10);
            var batches = InversedTensor.Split(t, 0, 2);

            print(t);
            foreach (var b in batches)
            {
                print("firstSlice" + b);
            }

            batches = InversedTensor.Split(t, 1, 3);
            foreach (var b in batches)
            {
                print("SecondSlice" + b);
            }

        }
        public void SliceJoinTensor()
        {
            InversedTensor array = InversedTensor.Random(10);
            InversedTensor array2 = InversedTensor.Random(10);

            print(array);
            print(array2);

            InversedTensor mat = InversedTensor.Join(1, array, array2);
            print(mat);

            InversedTensor cube = InversedTensor.Join(2, mat, mat);
            print(cube);

            InversedTensor[] mats = InversedTensor.Split(cube, 2, 1);
            print(mats[0] + "\n----\n" + mats[1]);

            InversedTensor[] arrays = InversedTensor.Split(mats[0], 1, 1);
            print(arrays[0] + "\n----\n" + arrays[1]);


            print("expansion");
            
            print(InversedTensor.Expand(InversedTensor.Constant(1f), 0, 10));

            print(array);
            print(InversedTensor.Expand(array, 1, 10));

            mat = InversedTensor.Random(2, 3);
            print(mat);
            print(InversedTensor.Expand(mat, 2, 10));

            print("Some mat" + mat);
            print("Transposed mat" + InversedTensor.Transpose(mat, 0, 1));
            
        }*/
    }
}

