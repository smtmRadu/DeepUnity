using UnityEditor;
using UnityEngine;
using System;
using System.Linq;
using System.Collections.Generic;

namespace DeepUnity
{
    [SerializeField]
    public class RNN : ScriptableObject, ISerializationCallbackReceiver
    {
        [NonSerialized] private IModuleRNN[] modules;
        [SerializeField] private IModuleRNNWrapper[] serializedModules;


        /// <summary>
        /// Inputs: (input, h_0). <br></br>
        /// input: <b>(B, S, H_in)</b> or <b>(S, H_in)</b> for unbatched input. <br></br>
        /// h_0:<b>(num_layers, B, H_in)</b> or <b>(num_layers, H_in)</b> for unbatched input. <br></br>
        /// <br></br>
        /// Outputs: (output, h_n). <br></br>
        /// output: <b>(B, S, H_out)</b> or <b>(S, H_in)</b> for unbatched input. <br></br>
        /// h_n: <b>(num_layers, B, H_out)</b> or <b>(num_layers, H_out)</b> for unbatched input. <br></br>
        /// <br></br>
        /// where B = batch_size, S = sequence_length, H_in = input_size, H_out = hidden_size.
        /// </summary>
        /// <param name="modules">RNNCell, Dropout or LayerNorm.</param>
        public RNN(IModuleRNN[] modules) => this.modules = modules;
        private RNN(int input_size, int hidden_size, int num_layers = 2, NonLinearity nonlinearity = NonLinearity.Tanh, float dropout = 0f)
        {
            if (num_layers < 1)
            {
                throw new ArgumentException($"An RNN must have at least one layer, not {num_layers}.");
            }

            List<IModuleRNN> listedModules = new() { new RNNCell(input_size, hidden_size, nonlinearity) };
            for (int i = 0; i < num_layers; i++)
            {
                listedModules.Add(new RNNCell(hidden_size, hidden_size, nonlinearity));
                if (dropout > 0f && i < num_layers - 1)
                {
                    // Add dropout modules after each layer but last.
                    listedModules.Add(new Dropout(dropout));
                }
            }

            modules = listedModules.ToArray();
        }

        public (Tensor, Tensor) Forward(Tensor input, Tensor hidden)
        {
            /// when forwarding, remember to cache the state last state of each RNN also when forwarding. Actually them are cached inside each rnncell.
            /// Make sure everything is computed ok before commiting the implementation.

            Tensor[] h_0 = hidden.Split(axis: 0, split_size: 1); // split 
            Tensor[] h_n = new Tensor[h_0.Length];
            int h_index = 0;
            foreach (var module in modules)
            {
                if(module is RNNCell rnncell)
                {
                    h_n[h_index] = rnncell.Forward(input, h_0[h_index]);
                    input = Tensor.Identity(h_n[h_index]);

                    h_index++;
                }
                else if(module is Dropout drop)
                {
                    input = drop.Forward(input);
                }
                else if(module is LayerNorm ln)
                {
                    input = ln.Forward(input);
                }
            }
            return (input, Tensor.Cat(null, h_0));
        }
        public void Backward(Tensor loss) // it does nt actually need hidden
        {
            for (int i = modules.Length - 1; i >= 0; i--)
            {
                loss = modules[i].Backward(loss);
            }
        }




        /// <summary>
        /// Gets all <typeparamref name="RNNCell"/> modules.
        /// </summary>
        /// <returns></returns>
        public Learnable[] Parameters { get => modules.Where(x => x is Learnable P).Select(x => (Learnable)x).ToArray(); }
        /// <summary>
        /// Save path: "Assets/". Creates/Overwrites model on the same path.
        /// For specific existing folder saving, <b><paramref name="name"/> = "folder_name/model_name"</b>
        /// </summary>
        public void Save(string name)
        {
            var instance = AssetDatabase.LoadAssetAtPath<RNN>("Assets/" + name + ".asset");
            if (instance == null)
                AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");

            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
        }
        public void OnBeforeSerialize()
        {
            serializedModules = modules.Select(x => IModuleRNNWrapper.Wrap(x)).ToArray();
        }
        public void OnAfterDeserialize()
        {
            modules = serializedModules.Select(x => IModuleRNNWrapper.Unwrap(x)).ToArray();
        }

    }
}

