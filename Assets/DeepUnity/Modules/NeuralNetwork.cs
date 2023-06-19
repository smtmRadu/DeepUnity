using System;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class NeuralNetwork : ScriptableObject, IModule, IParameters, ISerializationCallbackReceiver
    {
        [SerializeField] private string Name;
        [SerializeField] private OptimizerWrapper serializedOptimizer;
        [SerializeField] private ModuleWrapper[] serializedModules;

        private Optimizer Optimizer;
        private IModule[] Modules;


        public NeuralNetwork(params IModule[] modules) => Modules = modules;
        public void Compile(Optimizer optimizer, string name)
        {
            this.Optimizer = optimizer;
            this.Name = name;
            Optimizer.Initialize(Modules);
        }

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
                if (module is IParameters param)
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
                if (module is IParameters param)
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
                if (module is IParameters param)
                    param.ClipGradNorm(max_norm);
            }
        }
        /// <summary>
        /// Optimize parameters of the neural network.
        /// </summary>
        /// <exception cref="Uncompiled network error."></exception>
        public void Step()
        {
            if (Optimizer == null)
                throw new Exception("Cannot train an uncompiled network.");

            Optimizer.Step(Modules);
        }

        /// <summary>
        /// Called internally.
        /// </summary>
        public void InitializeGradients() { }

        // Saving
        /// <summary>
        /// Saves the network in Unity Assets folder. Overwrites the network file with the same name.
        /// </summary>
        /// <exception cref="Uncompiled Network error."></exception>
        public void Save()
        {
            if (Name == null)
                throw new Exception("Cannot save a non-compiled Neural Network.");

            var instance = AssetDatabase.LoadAssetAtPath<NeuralNetwork>("Assets/" + Name + ".asset");
            if (instance == null)
                AssetDatabase.CreateAsset(this, "Assets/" + Name + ".asset");

            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
        }
        public void OnBeforeSerialize()
        {
            serializedModules = Modules.Select(x => ModuleWrapper.Wrap(x)).ToArray();
            serializedOptimizer = OptimizerWrapper.Wrap(Optimizer);
        }
        public void OnAfterDeserialize()
        {
            Modules = serializedModules.Select(x => ModuleWrapper.Unwrap(x)).ToArray();
            Optimizer = OptimizerWrapper.Unwrap(serializedOptimizer);
            Optimizer.Initialize(Modules);

        }
    }
}