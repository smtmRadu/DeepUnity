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
            Optimizer.Initialize(Modules);
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
                if(module is Dense D)
                {
                    D.grad_W.ForEach(x => 0f);
                    D.grad_B.ForEach(x => 0f);
                }      
                else if(module is BatchNorm B)
                {
                    B.grad_Gamma.ForEach(x => 0f);
                    B.grad_Beta.ForEach(x => 0f);
                }
            }
        }
        public void Step()
        {
            if (Optimizer == null)
                throw new Exception("Cannot train an uncompiled network.");

            Optimizer.Step(Modules);
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
            Optimizer = OptimizerWrapper.Unwrap(serializedOptimizer);
            Optimizer.Initialize(Modules);

        }
    }

   
    

}