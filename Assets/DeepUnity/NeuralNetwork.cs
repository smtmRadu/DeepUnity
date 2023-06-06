using System;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class NeuralNetwork : ScriptableObject, IModule, ISerializationCallbackReceiver
    {
        [SerializeField] private string Name;
        [SerializeField] private OptimizerWrapper serializedOptimizer;
        [SerializeField] private ModuleWrapper[] serializedModules;
        
        private IOptimizer Optimizer;
        private IModule[] Modules;
        
        
        public NeuralNetwork(params IModule[] modules) => this.Modules = modules;
        public void Compile(IOptimizer optimizer, string name)
        {
            this.Optimizer = optimizer;
            this.Name = name;
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

        public void ZeroGrad()
        {
            foreach (var module in Modules)
            {
                if (module.GetType() != typeof(Dense))
                    continue;

                Dense dense = (Dense) module;

                dense.g_W.ForEach(x => 0f);
                dense.g_B.ForEach(x => 0f);
            }
        }
        public void Step()
        {
            if (Optimizer == null)
                throw new Exception("Cannot train an uncompiled network.");
            Optimizer.Step(Modules.Where(x => x.GetType() == typeof(Dense)).Select(x => (Dense)x).ToArray());
        }
  
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
            serializedModules = Modules.Select(x => new ModuleWrapper(x)).ToArray();
            serializedOptimizer = new OptimizerWrapper(Optimizer);
        }
        public void OnAfterDeserialize()
        {
            Modules = serializedModules.Select(x => ModuleWrapper.Get(x)).ToArray();
            Optimizer = OptimizerWrapper.Get(serializedOptimizer);

        }
    }

   
    

}