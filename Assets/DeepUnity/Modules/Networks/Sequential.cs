using System;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Sequential : ScriptableObject, ISerializationCallbackReceiver
    {       
        [NonSerialized]  private IModule[] modules;
        [SerializeField] private IModuleWrapper[] serializedModules;

        public Sequential(params IModule[] modules) => this.modules = modules;

        /// <summary>
        /// Forwards the input without caching.
        /// </summary>
        /// <param name="input"></param>
        /// <returns>output</returns>
        public Tensor Predict(Tensor input)
        {
            foreach (var module in modules)
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
            foreach (var module in modules)
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
        public void Backward(Tensor loss)
        {
            for (int i = modules.Length - 1; i >= 0; i--)
            {
                loss = modules[i].Backward(loss);
            }
            // return loss;
        }


        /// <summary>
        /// Gets all <typeparamref name="Learnable"/> modules.
        /// </summary>
        /// <returns></returns>
        public Learnable[] Parameters { get => modules.Where(x => x is Learnable P).Select(x => (Learnable)x).ToArray(); }

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
            serializedModules = modules.Select(x => IModuleWrapper.Wrap(x)).ToArray();
        }
        public void OnAfterDeserialize()
        {
            modules = serializedModules.Select(x => IModuleWrapper.Unwrap(x)).ToArray();
        }
    }
}