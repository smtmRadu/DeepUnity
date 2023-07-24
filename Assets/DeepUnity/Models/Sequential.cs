using System;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Sequential : ScriptableObject, IModel
    {
        [NonSerialized] private IModule[] modules;
        [SerializeField] private IModuleWrapper[] serializedModules;

        public Sequential(params IModule[] modules) => this.modules = modules;

        /// <summary>
        /// Forwards the input without caching.
        /// </summary>
        /// <param name="input"></param>
        /// <returns>output</returns>
        public Tensor Predict(Tensor input)
        {
            Tensor inputclone = Tensor.Identity(input);
            foreach (var module in modules)
            {
                inputclone = module.Predict(inputclone);
            }
            return inputclone;
        }
        /// <summary>
        /// Forwards the inputs and every module caches it.
        /// </summary>
        /// <param name="input"></param>
        /// <returns>output</returns>
        public Tensor Forward(Tensor input)
        {
            Tensor inputclone = Tensor.Identity(input);
            foreach (var module in modules)
            {
                inputclone = module.Forward(inputclone);
            }
            return inputclone;
        }
        /// <summary>
        /// Backpropagates the loss derivative w.r.t outputs and computes the gradients.
        /// </summary>
        /// <param name="loss">Derivative of the loss function w.r.t output. (dLdY)</param>
        /// <returns></returns>
        public void Backward(Tensor loss)
        {
            Tensor lossclone = Tensor.Identity(loss);
            for (int i = modules.Length - 1; i >= 0; i--)
            {
                lossclone = modules[i].Backward(lossclone);
            }
        }


        /// <summary>
        /// Gets all <typeparamref name="Learnable"/> modules.
        /// </summary>
        /// <returns></returns>
        public Learnable[] Parameters()
        {
            return modules.Where(x => x is Learnable P).Select(x => (Learnable)x).ToArray(); 
        }
        /// <summary>
        /// Save path: "Assets/". Creates/Overwrites model on the same path.
        /// For specific existing folder saving, <b><paramref name="name"/> = "folder_name/model_name"</b>
        /// </summary>
        public void Save(string name)
        {   
            var instance = AssetDatabase.LoadAssetAtPath<Sequential>("Assets/" + name + ".asset");
            if (instance == null)
                AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");

            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
        }
        public string Summary()
        {
            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.AppendLine("Model: Sequencial");
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
}