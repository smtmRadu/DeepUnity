using System;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Sequential : Model<Sequential>, ISerializationCallbackReceiver
    {
        [NonSerialized] private IModule[] modules;
        [SerializeField] private IModuleWrapper[] serializedModules;

        public Sequential(params IModule[] modules) => this.modules = modules;



        /// <summary>
        /// Same as forward but faster. Method used only for network utility.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Tensor Predict(Tensor input)
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
        public Tensor Forward(Tensor input)
        {
            Tensor output = modules[0].Forward(input);
            for (int i = 1; i < modules.Length; i++)
            {
                output = modules[i].Forward(output);
            }
            return output;
        }     
        public override void Backward(Tensor lossDerivative)
        {
            Tensor loss = modules[modules.Length - 1].Backward(lossDerivative);
            for (int i = modules.Length - 2; i >= 0; i--)
            {
                loss = modules[i].Backward(loss);
            }
        }


        public override Learnable[] Parameters()
        {
            return modules.OfType<Learnable>().ToArray();
        }       
        public override Sequential Compile(string name)
        {
            var instance = AssetDatabase.LoadAssetAtPath<Sequential>("Assets/" + name + ".asset");
            if(instance != null)
            {
                throw new InvalidOperationException($"A Sequencial asset called '{name}' already exists!.");
            }
           
            this.name = name;
            AssetDatabase.CreateAsset(this, $"Assets/{name}.asset");
            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
            return this;

        }
        public override void Save(string version = "1.0")
        {
            this.version += "v" + version;
            // Check if asset exists:
            var instance = AssetDatabase.LoadAssetAtPath<Sequential>("Assets/" + name + ".asset");
            if (instance == null)
                throw new InvalidOperationException($"Cannot save Sequencial because it was not compiled.");
            
            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
        }
        public override string Summary()
        {
            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.AppendLine($"Name: {name} ({version})");
            stringBuilder.AppendLine("Type: Sequencial");
            stringBuilder.AppendLine($"Layers : {modules.Length}");
            foreach (var module in modules)
            {
                stringBuilder.AppendLine($"         {module.GetType().Name}");
            }
            stringBuilder.AppendLine($"Parameters: {modules.Where(x => x is Learnable).Select(x => (Learnable)x).Sum(x => x.LearnableParametersCount)}");
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