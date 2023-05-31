using System.Linq;

namespace DeepUnity
{
    public class NeuralNetwork : IModule
    {
        private string Name;
        private IModule[] Modules;
        private IOptimizer Optimizer;
        public Tensor InputCache { get; set; }

        public NeuralNetwork(params IModule[] modules) => this.Modules = modules;
        public void Compile(IOptimizer optimizer, string name)
        {
            this.Optimizer = optimizer;
            this.Name = name;
        }

        public Tensor Forward(Tensor input)
        {
            foreach (var module in Modules)
            {
                input = module.Forward(input);
            }
            return input;
        }
        public Tensor Backward(Tensor loss)
        {
            for (int i = Modules.Length - 1; i >= 0; i--)
            {
                loss = Modules[i].Backward(loss);
            }
            return loss;
        }

        public void ZeroGrad()
        {
            foreach (var module in Modules)
            {
                if (module.GetType() != typeof(Dense))
                    continue;

                Dense dense = (Dense) module;

                dense.gWeights.ForEach(x => 0f);
                dense.gBiases.ForEach(x => 0f);
            }
        }
        public void Step() => Optimizer.Step(Modules.Where(x => x.GetType() == typeof(Dense)).Select(x => (Dense)x).ToArray());

    }
}