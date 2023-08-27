using UnityEngine;
using DeepUnity;
public class MatMulBenchmark : MonoBehaviour
{
	[SerializeField] int runs = 100;
	[SerializeField] int batch_size = 64;
	[SerializeField] int dense_in_features = 64;
	[SerializeField] int dense_out_features = 64;
	[SerializeField] Device device = Device.CPU;


    private void Start()
    {
		Dense dense = new Dense(dense_in_features, dense_out_features, device: device);
		Tensor input = Tensor.RandomNormal(batch_size, dense_in_features);

		ClockTimer.Start();
		for (int i = 0; i < runs; i++)
		{
			dense.Forward(input);
		}
		ClockTimer.Stop();
    }

}


