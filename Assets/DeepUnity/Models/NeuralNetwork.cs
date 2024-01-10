using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Sequencial model.
    /// </summary>
    [Serializable]
    public class NeuralNetwork : Model<NeuralNetwork, Tensor>, ISerializationCallbackReceiver
    {
        [NonSerialized] private IModule[] modules;
        [SerializeField] private IModuleWrapper[] serializedModules;

        public NeuralNetwork(params IModule[] modules) => this.modules = modules;

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
        /// <param name="lossDerivativeWrtPrediction">Derivative of the loss function w.r.t output (dLdY).</param>
        /// <returns><returns>The backpropagated loss derivative (dLdX).</returns>
        public override Tensor Backward(Tensor lossDerivativeWrtPrediction)
        {
            Tensor loss = modules[modules.Length - 1].Backward(lossDerivativeWrtPrediction);
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
        /// <param name="device"></param>
        public void SetDevice(Device device)
        {
            foreach (var item in modules.OfType<ILearnable>())
            {
                item.SetDevice(device);
            }
        }
        /// <summary>
        /// Gives a brief summary of the model.
        /// </summary>
        /// <returns></returns>
        public override string Summary()
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
            var cloned_modules = this.modules.Select(x => (IModule)x.Clone()).ToArray();
            var net = new NeuralNetwork(cloned_modules);
            return net;
        }
    }

    // If you turn this ON you will not see the param updates in the inspector
    // [CustomEditor(typeof(Model<Sequential>), true)]
    // [CanEditMultipleObjects]
    // class ScriptlessSequential : Editor
    // {
    //     public override void OnInspectorGUI()
    //     {
    //         List<string> dontDrawMe = new List<string>() { "m_Script" };
    // 
    //         DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
    //         serializedObject.ApplyModifiedProperties();
    //     }
    // }
}