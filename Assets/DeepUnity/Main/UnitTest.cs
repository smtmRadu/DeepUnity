using DeepUnity;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class UnitTest : MonoBehaviour
    {
        public Sequential net;
        Optimizer optim;
        [SerializeField] PerformanceGraph graph = new PerformanceGraph();

        Tensor input = Tensor.RandomNormal(64, 6, 10);
        Tensor target = Tensor.Random01(64, 1);

        public VariationalAutoencoder vae;
        private void Start()
        {
            MultiheadAttention mhatt = new MultiheadAttention(120, 6, device: Device.CPU);

            Tensor input = Tensor.Random01(64, 30, 120);
            BenchmarkClock.Start();
            var output = mhatt.Forward(input);
            mhatt.Backward(output);
            BenchmarkClock.Stop();
        }
    }

}


