using System;
using UnityEngine;
using DeepUnity;
using System.Collections.Generic;

namespace kbRadu
{
    public class TensorUnitTests : MonoBehaviour
    {
        event Action tests;
        Tensor t1;
        Tensor t2;
       
        private void Start()
        {
            tests += Create;
            tests += OperatorOverload;
            tests += SpecialOperations;
            tests += DimensionOperations;
            tests += MathOperations;
            tests += OtherOperations;

            tests.Invoke();
            Debug.Log("All tests were run. If no 'TestFailed' message appeared, all tests succeded.");
        }
        public void Create()
        {
            
            try
            {
                t1 = Tensor.Zeros(2, 4, 5);
                t2 = Tensor.Ones(2);
                t1 = Tensor.Identity(t2);
                t1 = Tensor.Constant(2);
                t1 = Tensor.Constant(new float[] { 2, 3, 4, 5, 5 });
                t1 = Tensor.Constant(new float[2, 3] { { 2, 3, 4, }, { 2, 3, 4 } });
                t1 = Tensor.Constant(new float[1, 2, 3] { { { 2, 3, 3, }, { 2, 3, 3 } } });
                t1 = Tensor.Constant(new float[1, 1, 2, 3] { { { { 2, 3, 3, }, { 2, 3, 3 } } } });
                t2 = Tensor.Random01(2, 3, 4);
                t1 = Tensor.RandomNormal((0, 1), 2, 3);
                t1 = Tensor.Fill(10, 2, 4);
                t1 = Tensor.RandomRange((10, 100), 3, 1);
            }
            catch
            {
                Debug.Log("Test failed.");
            }
        }
        public void OperatorOverload()
        {
            try
            {
                t1 = Tensor.Random01(5, 5);
                t2 = Tensor.Random01(5, 5);

                var x = t1 * t2;
                x = t1 + t2;
                x = t1 - t2;
                x = t1 / t2;
                x = t1 / 2;
                x = t1 - 2;
                x = t1 + 2;
                x = t1 * 2;
                x = 2 * t1;
                x = 2 + t2;
            }
            catch
            {
                Debug.Log("Test failed.");
            }
        }

        public void SpecialOperations()
        {
            try
            {
                t1 = Tensor.Random01(5, 5);
                t2 = Tensor.Random01(5, 5);

                Tensor.MatMul(t1, t2);
                Tensor.MatPad(t1, 2, PaddingType.Mirror);
            }
            catch
            {
                Debug.Log("Test failed.");
            }
        }

        public void DimensionOperations()
        {
            try
            {
                t1 = Tensor.Random01(5, 5);
                t2 = Tensor.Random01(1, 5, 1, 5);

                Tensor.Expand(t1, 1, 3);
                Tensor.Var(t1, 1);
                Tensor.Std(t1, 0, 2, true);
                Tensor.Sum(t1, 1, false);
                Tensor.Shuffle(t1, 0);
                Tensor.Transpose(t1, 0, 1);
                Tensor.Split(t1, 0, 2);
                Tensor.Min(t1, 0, false);
                Tensor.Max(t1, 1, true);
                Tensor.Squeeze(t2);
                Tensor.Unsqueeze(t2, 0);
            }
            catch
            {
                Debug.Log("Test failed.");
            }
        }

        public void MathOperations()
        {
           
            try
            {
                t1 = Tensor.Random01(5, 5);
                t2 = Tensor.Random01(1, 5, 1, 5);
                Tensor t3 = Tensor.RandomNormal((0, 1), 5, 5);

                Tensor.Pow(t1, 3);
                Tensor.Sqrt(t1);
                Tensor.Exp(t1);
                Tensor.Log(t2);
                Tensor.Abs(t2);
                Tensor.Sin(t1);
                Tensor.Cos(t2);
                Tensor.Minimum(t1, t3);
                Tensor.Maximum(t3, t1);

                Tensor.Clip(t1, -1, 0.5f);
                Tensor.Norm(t1, NormType.EuclideanL2);
                Tensor.LogPDF(t1, t1, t1);
                Tensor.PDF(t1, t1, t1);
            }
            catch
            {
                Debug.Log("Test failed.");
            }
        }

        public void OtherOperations()
        {
            try
            {
                t1 = Tensor.Random01(5, 5);
                t2 = Tensor.Random01(1, 5, 1, 5);
                Tensor t3 = Tensor.RandomNormal((0, 1), 5, 5);

                Tensor.Reshape(t1, 25);
                var x = t1.Select(x => 2);
                var j = t1.Select(x => x * x + 2);
                t1.Zip(t3, (a, b) => a * b);
                var s = t1.ToArray();
                t1.Count(x => x < 1);
                t1.ToString();
                t1.GetHashCode();
                t1.Equals(t2);
            }
            catch
            {
                Debug.Log("Test failed.");
            }
        }
 
    }
}

