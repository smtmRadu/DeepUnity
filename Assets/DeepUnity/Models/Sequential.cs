using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using DeepUnity.Modules;
namespace DeepUnity.Models
{
    /// <summary>
    /// Sequential model. Models can be serialized using <b>JsonUtility.ToJson</b>...
    /// </summary>

    [Serializable]
    public class Sequential : Model<Sequential, Tensor>, ISerializationCallbackReceiver
    {
        [NonSerialized] private IModule[] Modules;
        [SerializeField] private IModuleWrapper[] serializedModules;

        public Sequential(params IModule[] modules) => this.Modules = modules;

        /// <summary>
        /// Same as Forward but used only for network inference.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public override Tensor Predict(Tensor input)
        {
            Tensor output = Modules[0].Predict(input);
            for (int i = 1; i < Modules.Length; i++)
            {
                output = Modules[i].Predict(output);
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
            Tensor output = Modules[0].Forward(input);
            for (int i = 1; i < Modules.Length; i++)
            {
                output = Modules[i].Forward(output);
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
            Tensor loss = Modules[Modules.Length - 1].Backward(lossGradient);
            for (int i = Modules.Length - 2; i >= 0; i--)
            {
                loss = Modules[i].Backward(loss);
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
            foreach (var item in Modules.OfType<ILearnable>())
            {
                param.AddRange(item.Parameters());
            }
            return param.ToArray();
        }
        /// <summary>
        /// Changes the device of all <see cref="ILearnable"/> modules.
        /// </summary>
        /// <param name="device"></param>
        public override Device Device
        {
            set
            {
                foreach (var item in Modules.OfType<ILearnable>())
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
            stringBuilder.AppendLine($"Layers : {Modules.Length}");
            foreach (var module in Modules)
            {
                stringBuilder.AppendLine($"         {module.GetType().Name}");
            }
            stringBuilder.AppendLine($"Parameters: {Modules.Where(x => x is ILearnable).Select(x => (ILearnable)x).Sum(x => x.ParametersCount())}");
            return stringBuilder.ToString();
        }



        public void OnBeforeSerialize()
        {
            serializedModules = Modules.Select(x => IModuleWrapper.Wrap(x)).ToArray();
        }
        public void OnAfterDeserialize()
        {
            Modules = serializedModules.Select(x => IModuleWrapper.Unwrap(x)).ToArray();
        }
        public override object Clone()
        {
            var cloned_modules = Modules.Select(x => (IModule)x.Clone()).ToArray();
            var net = new Sequential(cloned_modules);
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