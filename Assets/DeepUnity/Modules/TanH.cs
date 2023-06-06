using UnityEngine;

namespace DeepUnity
{
    public class TanH : IModule
    {
        public Tensor InputCache { get; set; }

        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);
            return InputCache.Select(x =>
            {
                float e2x = Mathf.Exp(2 * x);
                float tanh = (e2x - 1) / (e2x + 1);
                return tanh;
            });

        }

        public Tensor Backward(Tensor loss)
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