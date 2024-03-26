using System.Collections.Generic;
using System.Linq;
using System.Text;
using System;
using UnityEngine;
using DeepUnity.Activations;
using DeepUnity.Modules;
namespace DeepUnity.Models
{
    public class MultilayerPerceptron : Model<MultilayerPerceptron, Tensor>, ISerializationCallbackReceiver
    {
        [NonSerialized] private IModule[] modules;
        [SerializeField] private IModuleWrapper[] serializedModules;

        /// <summary>
        /// Simple MLP. Hidden activation default is Tanh. If Output Activation is null, the model ends with a linear activation.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        /// <param name="numLayers"></param>
        /// <param name="hidUnits"></param>
        /// <param name="hiddenActivation"></param>
        /// <param name="outputActivation"></param>
        /// <exception cref="ArgumentException"></exception>
        public MultilayerPerceptron(int inputs, int outputs, int numLayers, int hidUnits, IActivation hiddenActivation = null, IActivation outputActivation = null)
        {
            if (numLayers < 1)
                throw new ArgumentException("Num layers must be > 0");

            if (hidUnits < 1)
                throw new ArgumentException("Hid units must be > 0");

            if (hiddenActivation == null)
                hiddenActivation = new Tanh();

            List<IModule> mds = new();
            mds.Add(new Dense(inputs, hidUnits));
            mds.Add(hiddenActivation.Clone() as IModule);

            for (int i = 0; i < numLayers - 1; i++)
            {
                mds.Add(new Dense(hidUnits, hidUnits));
                mds.Add(hiddenActivation.Clone() as IModule);
            }

            mds.Add(new Dense(hidUnits, outputs));
            if (outputActivation != null)
                mds.Add(outputActivation.Clone() as IModule);

            modules = mds.ToArray();
        }

        /// <summary>
        /// Same as Forward but used only for network inference.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public override Tensor Predict(Tensor input)
        {
            Tensor output = modules[0].Predict(input);
            for (int i = 1; i < modules.Length; i++)
            {
                output = modules[i].Predict(output);
            }
            return output;
        }
        /// <summary>
        /// Forwards the input and caches each module's input. Used in pair with Backward().
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public override Tensor Forward(Tensor input)
        {
            Tensor output = modules[0].Forward(input);
            for (int i = 1; i < modules.Length; i++)
            {
                output = modules[i].Forward(output);
            }
            return output;
        }
        /// <summary>
        /// Backpropagates the loss derivative wrt. output and computes the gradients of learnable parameters.
        /// </summary>
        /// <param name="lossGradient">Derivative of the loss function w.r.t output (dLdY).</param>
        /// <returns><returns>The backpropagated loss derivative (dLdX).</returns>
        public override Tensor Backward(Tensor lossGradient)
        {
            Tensor loss = modules[modules.Length - 1].Backward(lossGradient);
            for (int i = modules.Length - 2; i >= 0; i--)
            {
                loss = modules[i].Backward(loss);
            }
            return loss;
        }

        /// <summary>
        /// Get all <see cref="Parameter"/>s of this model, consisting of Theta and ThetaGrad.
        /// </summary>
        /// <returns></returns>
        public override Parameter[] Parameters()
        {
            List<Parameter> param = new();
            foreach (var item in modules.OfType<ILearnable>())
            {
                param.AddRange(item.Parameters());
            }
            return param.ToArray();
        }
        /// <summary>
        /// Changes the device of all <see cref="ILearnable"/> modules.
        /// </summary>
        public override Device Device
        {
            set
            {
                foreach (var item in modules.OfType<ILearnable>())
                {
                    item.Device = value;
                }
            }
        }
        /// <summary>
        /// Gives a brief summary of the model.
        /// </summary>
        /// <returns></returns>
        public string Summary()
        {
            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.AppendLine($"Name: {name}");
            stringBuilder.AppendLine($"Type: {GetType().Name}");
            stringBuilder.AppendLine($"Layers : {modules.Length}");
            foreach (var module in modules)
            {
                stringBuilder.AppendLine($"         {module.GetType().Name}");
            }
            stringBuilder.AppendLine($"Parameters: {modules.Where(x => x is ILearnable).Select(x => (ILearnable)x).Sum(x => x.ParametersCount())}");
            return stringBuilder.ToString();
        }



        public void OnBeforeSerialize()
        {
            serializedModules = modules.Select(x => IModuleWrapper.Wrap(x)).ToArray();
        }
        public void OnAfterDeserialize()
        {
            modules = serializedModules.Select(x => IModuleWrapper.Unwrap(x)).ToArray();
        }
        public override object Clone()
        {
            var cloned_modules = modules.Select(x => (IModule)x.Clone()).ToArray();
            var net = new Sequential(cloned_modules);
            return net;
        }
    }


}

