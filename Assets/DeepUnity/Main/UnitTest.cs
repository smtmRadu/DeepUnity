using UnityEngine;
using DeepUnity.Models;
using System.Collections.Generic;
using UnityEngine.UI;
using DeepUnity.Optimizers;
using DeepUnity.Modules;
using DeepUnity.Activations;
using DeepUnity.ReinforcementLearning;

namespace DeepUnity.Tutorials
{
    public class UnitTest : MonoBehaviour
    {
        public Device device = Device.CPU;
        public Sequential net;
        public GameObject canvas;
        public Optimizer optim;

        private List<RawImage> displays;
        public PerformanceGraph performanceGraph = new PerformanceGraph();
        public PerformanceGraph performanceGraph2 = new PerformanceGraph();

        private MultiheadAttention mha;
        private Tensor x, y;
        private OUNoise ounoise;
        public float mu = 0, sigma = 1f, theta = 0.2f, dt = 0.02f, x0 = 2f; 

        private void Start()
        {
            Tensor x = Tensor.Random01(10);
            Tensor y = x.Clone() as Tensor;
            print(x);
            y[0] = 10;
            print(y);
        }
    }

}


