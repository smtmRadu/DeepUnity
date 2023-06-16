using UnityEngine;
using DeepUnity;
using System;

namespace kbRadu
{
    public class ModulesTest : MonoBehaviour
    {
        private void Start()
        {
            var mish = new Mish();

            Tensor x = Tensor.Random01(10);

            var start = DateTime.Now;

            for (int i = 0; i < 1000; i++)
            {
                mish.Forward(x);
                mish.Backward(x);
            }
            
            var end = DateTime.Now - start;

            print(end);
        }
    }
}

