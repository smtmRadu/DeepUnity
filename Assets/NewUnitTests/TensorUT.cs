using UnityEngine;
using DeepUnity;


namespace kbRadu
{
    public class TensorUT : MonoBehaviour
    {
        private void Start()
        {
            Tensor x = Tensor.Ones(1);
            print(x);
            print(Tensor.Expand(x, -1, 32));
            NeuralNetwork net = new NeuralNetwork(
                new Dense(10, 100),
                new ReLU(),
                new Dense(100, 10));

            print(net.Forward(Tensor.Random01(10)));
            print(net.Backward(Tensor.Random01(10)));
        }
    }
}

