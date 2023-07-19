using System;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Sequential : ScriptableObject, IModule, ISerializationCallbackReceiver
    {
        [SerializeField] private ModuleWrapper[] serializedModules;
        [NonSerialized]  private IModule[] Modules;


        public Sequential(params IModule[] modules) => Modules = modules;

        /// <summary>
        /// Forwards the input without caching.
        /// </summary>
        /// <param name="input"></param>
        /// <returns>output</returns>
        public Tensor Predict(Tensor input)
        {
            foreach (var module in Modules)
            {
                input = module.Predict(input);
            }
            return input;
        }
        /// <summary>
        /// Forwards the inputs and every module caches it.
        /// </summary>
        /// <param name="input"></param>
        /// <returns>output</returns>
        public Tensor Forward(Tensor input)
        {
            foreach (var module in Modules)
            {
                input = module.Forward(input);
            }
            return input;
        }
        /// <summary>
        /// Backpropagates the loss derivative w.r.t outputs and computes the gradients.
        /// </summary>
        /// <param name="loss">Derivative of the loss function w.r.t output. (dLdY)</param>
        /// <returns></returns>
        public Tensor Backward(Tensor loss)
        {
            for (int i = Modules.Length - 1; i >= 0; i--)
            {
                loss = Modules[i].Backward(loss);
            }
            return loss;
        }


        /// <summary>
        /// Gets all <typeparamref name="Learnable"/> modules.
        /// </summary>
        /// <returns></returns>
        public Learnable[] Parameters { get => Modules.Where(x => x is Learnable P).Select(x => (Learnable)x).ToArray(); }

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
        public void OnBeforeSerialize()
        {
            serializedModules = Modules.Select(x => ModuleWrapper.Wrap(x)).ToArray();
        }
        public void OnAfterDeserialize()
        {
            Modules = serializedModules.Select(x => ModuleWrapper.Unwrap(x)).ToArray();
        }
    }
}