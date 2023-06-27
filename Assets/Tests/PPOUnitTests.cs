using UnityEngine;
using DeepUnity;

namespace kbRadu
{
    public class PPOUnitTests : MonoBehaviour
    {

        private void Test1()
        {
            Tensor mu = Tensor.RandomRange((-1, 1), 10);
            Tensor sigma = Tensor.Random01(10);
            Tensor gaussian = mu.Zip(sigma, (m, s) => Utils.Random.Gaussian(m, s));

            print("mu " + mu);
            print("sigma" + sigma);
            print("gaussian" + gaussian);

            Tensor logProbs = Tensor.Zeros(10);
            for (int i = 0; i < 10; i++)
            {
                logProbs[i] = Utils.Numerics.LogDensity(gaussian[i], mu[i], sigma[i]);
            }
            print(logProbs);

            print(Tensor.LogDensity(gaussian, mu, sigma));
        }
    }
}

