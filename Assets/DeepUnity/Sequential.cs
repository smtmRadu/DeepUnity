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
        private IModule[] Modules;


        public Sequential(params IModule[] modules) => Modules = modules;

        public Tensor Predict(Tensor input)
        {
            foreach (var module in Modules)
            {
                input = module.Predict(input);
            }
            return input;
        }
        public Tensor Forward(Tensor input)
        {
            foreach (var module in Modules)
            {
                input = module.Forward(input);
            }
            return input;
        }
        /// <summary>
        /// Backpropagates the loss and calculates the gradients.
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
        /// All gradients are set to 0.
        /// </summary>
        public void ZeroGrad()
        {
            foreach (var module in Modules)
            {
                if (module is Learnable param)
                    param.ZeroGrad();
            }
        }
        /// <summary>
        /// The gradients are clipped in the range [-clip_value, clip_value]
        /// </summary>
        public void ClipGradValue(float clip_value)
        {
            foreach (var module in Modules)
            {
                if (module is Learnable param)
                    param.ClipGradValue(clip_value);
            }
        }
        /// <summary>
        /// The norm is computed over each module separately.
        /// </summary>
        public void ClipGradNorm(float max_norm)
        {
            foreach (var module in Modules)
            {
                if (module is Learnable param)
                    param.ClipGradNorm(max_norm);
            }
        }

        public Learnable[] Parameters() => Modules.Where(x => x is Learnable P).Select(x => (Learnable)x).ToArray();


        // Saving
        /// <summary>
        /// Saves the network in Unity Assets folder. Overwrites the network file with the same name.
        /// </summary>
        /// <exception cref="Uncompiled Network error."></exception>
        public void Save(string name)
        {
            if (name == null)
                throw new Exception("Cannot save a non-compiled Neural Network.");

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