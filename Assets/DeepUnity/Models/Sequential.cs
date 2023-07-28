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
        [NonSerialized] public IModule[] modules;
        [SerializeField] private IModuleWrapper[] serializedModules;

        public Sequential(params IModule[] modules) => this.modules = modules;




        public Tensor Predict(Tensor input)
        {
            Tensor output = modules[0].Predict(input);
            for (int i = 1; i < modules.Length; i++)
            {
                output = modules[i].Predict(output);
            }
            return output;
        }
        public Tensor Forward(Tensor input)
        {
            Tensor output = modules[0].Forward(input);
            for (int i = 1; i < modules.Length; i++)
            {
                output = modules[i].Forward(output);
            }
            return output;
        }     
        public void Backward(Loss loss) => Backward(loss.Derivative);
        /// <summary>
        /// Backpropagates the loss derivative w.r.t outputs and computes the gradients.
        /// </summary>
        /// <param name="lossDerivative">Derivative of the loss function w.r.t output (dLdY).</param>
        /// <returns></returns>
        public void Backward(Tensor lossDerivative)
        {
            Tensor loss = modules[modules.Length - 1].Backward(lossDerivative);
            for (int i = modules.Length - 2; i >= 0; i--)
            {
                loss = modules[i].Backward(loss);
            }
        }




        /// <summary>
        /// Gets all <typeparamref name="Learnable"/> modules.
        /// </summary>
        /// <returns></returns>
        public Learnable[] Parameters { get => modules.Where(x => x is Learnable P).Select(x => (Learnable)x).ToArray(); }

        /// <summary>
        /// <b>The name of the asset is unmodifiable.</b> <br></br>
        /// Save path: "Assets/". Creates/Overwrites model on the same path. <br></br>
        /// For specific existing folder saving, <b><paramref name="name"/> = "folder_name/model_name"</b>
        /// </summary>
        public void Save(string name)
        {
            var instance = AssetDatabase.LoadAssetAtPath<Sequential>("Assets/" + name + ".asset");
            if (instance == null)
            {
                try
                {
                    AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
                }
                catch { }
            }
               

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