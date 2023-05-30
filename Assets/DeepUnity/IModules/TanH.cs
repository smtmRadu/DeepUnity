using UnityEngine;

namespace DeepUnity
{
    public class TanH : IModule
    {
        public Tensor<float> InputCache { get; set; }

        public Tensor<float> Forward(Tensor<float> input)
        {
            InputCache = input.Clone() as Tensor<float>;
            return InputCache.Select(x =>
            {
                float e2x = Mathf.Exp(2 * x);
                float tanh = (e2x - 1) / (e2x + 1);
                return tanh;
            });

        }

        public Tensor<float> Backward(Tensor<float> loss)
        {
            var dTanHdInput = InputCache.Select(x =>
            {
                float e2x = Mathf.Exp(2 * x);
                float tanh = (e2x - 1) / (e2x + 1);
                return 1 - tanh * tanh;
            });

            var dCostdInput = dTanHdInput * loss;
            return dCostdInput;
        }
    }

}
