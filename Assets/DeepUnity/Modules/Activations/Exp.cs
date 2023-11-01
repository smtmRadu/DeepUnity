using System;
namespace DeepUnity
{
    [Serializable]
    public class Exp : Activation
    {
        protected override Tensor Activate(Tensor x) => x.Exp();
        protected override Tensor Derivative(Tensor x) => x.Exp();

        public override object Clone() => new Exp();
    }
}
