/*using System;
using System.Linq;
using UnityEngine;

namespace DeepUnity.Modules
{
    [Serializable]
    public class ResidualBlock : IModule
    {
        [SerializeField] private IModule[] modules;


        public ResidualBlock(IModule[] residual_mapping, IModule[] skip_connection = null)
        {
            if( residual_mapping == null ||  residual_mapping.Length == 0)
            {
                throw new ArgumentException("Residual Block modules are null");
            }

            this.modules = residual_mapping;
        }
        private ResidualBlock() { }

        public Tensor Predict(Tensor input)
        {
            Tensor modules_output = input.Clone() as Tensor;

            for (int i = 0; i < modules.Length; i++)
            {
                modules_output = modules[i].Predict(modules_output);
            }

            if (!Enumerable.SequenceEqual(input.Shape, modules_output.Shape))
                throw new ArgumentException("Residual block input is not matching the residual block output for addition");
            return modules_output + input;
        }
        public Tensor Forward(Tensor input)
        {

        }
        public Tensor Backward(Tensor dLdY)
        {

        }
        public object Clone()
        {
            var resb = new ResidualBlock();
            resb.modules = modules.Select(x => (IModule)x.Clone()).ToArray();
            return resb;
        }
    }

}



*/