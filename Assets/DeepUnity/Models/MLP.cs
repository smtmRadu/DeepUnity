/*using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace DeepUnity
{
    public class MLP : Model<MLP>
    {
        [SerializeField] private IModule[] modules;

        /// <summary>
        /// Not implemented yet.
        /// Experimental. A fast multilayer-perceptron that lives inside GPU. ReLU hidden activation.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        /// <param name="num_layers"></param>
        /// <param name="hidden_size"></param>
        /// <param name="outputActivation">If null, Identity activation is considered</param>
        public MLP(int inputs, int outputs, int num_layers = 2, int hidden_size = 128, Activation outputActivation = null)
        {

            if(num_layers < 2)
            {
                throw new ArgumentException("Num_layers argument must be at least 2");
            }

            List<IModule> mods = new List<IModule>();
            mods.Add(new Dense(inputs, hidden_size, InitType.HE_Uniform, InitType.Zeros));
            mods.Add(new ReLU());
            for (int i = 1; i < num_layers; i++)
            {
                mods.Add(new Dense(hidden_size, hidden_size, InitType.HE_Uniform, InitType.Zeros));
                mods.Add(new ReLU());
            }
            mods.Add(new Dense(hidden_size, outputs, InitType.HE_Uniform, InitType.Zeros));
            if(outputActivation != null)
                mods.Add(outputActivation);

            this.modules = mods.ToArray();
        }

        public override void Backward(Tensor lossDerivative)
        {
            throw new NotImplementedException();
        }

        public override string Summary()
        {
            throw new NotImplementedException();
        }

        public override Learnable[] Parameters()
        {
            return modules.OfType<Learnable>().ToArray();
        }
    }

}



*/