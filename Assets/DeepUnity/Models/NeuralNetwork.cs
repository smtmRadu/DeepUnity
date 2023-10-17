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
        /// Forward used only for network utility.
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
        /// Forwards the input and caches each module input. Used in pair with Backward().
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
        /// <param name="lossDerivative"></param>
        /// <returns></returns>
        public override Tensor Backward(Tensor lossDerivative)
        {
            Tensor loss = modules[modules.Length - 1].Backward(lossDerivative);
            for (int i = modules.Length - 2; i >= 0; i--)
            {
                loss = modules[i].Backward(loss);
            }
            return loss;
        }

        /// <summary>
        /// Get all <see cref="Learnable"/> modules of this model.
        /// </summary>
        /// <returns></returns>
        public override Learnable[] Parameters() => modules.OfType<Learnable>().ToArray();     
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
            stringBuilder.AppendLine($"Parameters: {modules.Where(x => x is Learnable).Select(x => (Learnable)x).Sum(x => x.ParametersCount())}");
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