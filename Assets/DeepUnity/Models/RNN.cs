using UnityEditor;
using UnityEngine;
using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine.Windows;
using Unity.VisualScripting;
using System.Drawing.Printing;
using System.Text;

namespace DeepUnity
{
    [SerializeField]
    public class RNN : ScriptableObject, IModel
    {
        [NonSerialized] private IModuleRNN[] modules;
        [SerializeField] private bool batchFirst;
        [SerializeField] private IModuleRNNWrapper[] serializedModules;
        

        /// <summary>
        /// 
        /// Inputs: (input, h_0). <br></br>
        /// input:  <b>(L, H_in)</b> for unbatched input, <b>(L, B, H_in)</b> when batch_first = false or <b>(B, L, H_in)</b> when batch_first = true. <br></br>
        /// h_0:    <b>(num_layers, H_out)</b> for unbatched input, or <b>(num_layers, B, H_out)</b>. <br></br>
        /// 
        /// <br></br>
        /// Outputs: (output, h_n). <br></br>
        /// output: <b>(L, H_in)</b> for unbatched input, or <b>(L, B, H_out)</b> when batch_first = false or <b>(B, L, H_out)</b> when batch_first = true. <br></br>
        /// h_n: <b>(num_layers, H_out)</b> for unbatched input or <b>(num_layers, B, H_out)</b>. <br></br>
        /// 
        /// <br></br>
        /// where B = batch_size, L = sequence_length, H_in = input_size, H_out = hidden_size.
        /// </summary>
        /// <param name="modules">RNNCell, Dropout or LayerNorm.</param>
        public RNN(bool batch_first = false, params IModuleRNN[] modules)
        {
            this.modules = modules;
        }
        /// <summary>
        /// 
        /// Inputs: (input, h_0). <br></br>
        /// input:  <b>(L, H_in)</b> for unbatched input, <b>(L, B, H_in)</b> when batch_first = false or <b>(B, L, H_in)</b> when batch_first = true. <br></br>
        /// h_0:    <b>(num_layers, H_in)</b> for unbatched input, or <b>(num_layers, B, H_in)</b>. <br></br>
        /// 
        /// <br></br>
        /// Outputs: (output, h_n). <br></br>
        /// output: <b>(L, H_in)</b> for unbatched input, or <b>(L, B, H_out)</b> when batch_first = false or <b>(B, L, H_out)</b> when batch_first = true. <br></br>
        /// h_n: <b>(num_layers, H_out)</b> for unbatched input or <b>(num_layers, B, H_out)</b>. <br></br>
        /// 
        /// <br></br>
        /// where B = batch_size, L = sequence_length, H_in = input_size, H_out = hidden_size.
        /// </summary>
        public RNN(int input_size, int hidden_size, int num_layers = 2, NonLinearity nonlinearity = NonLinearity.Tanh, bool batch_first = false, float dropout = 0f)
        {
            this.batchFirst = batch_first;
            if (num_layers < 1)
            {
                throw new ArgumentException($"An RNN must have at least one layer, not {num_layers}.");
            }

            List<IModuleRNN> moduleList = new() { new RNNCell(input_size, hidden_size, nonlinearity) };

            for (int i = 1; i < num_layers; i++)
            {
                if(dropout > 0 && i < num_layers - 1)
                    moduleList.Add(new Dropout(dropout));

                moduleList.Add(new RNNCell(hidden_size, hidden_size, nonlinearity));
            }

            modules = moduleList.ToArray();         
        }



        /// <summary>
        /// input:  <b>(L, H_in)</b> for unbatched input, <b>(L, B, H_in)</b> when batch_first = false or <b>(B, L, H_in)</b> when batch_first = true. <br></br>
        /// h_0:    <b>(num_layers, H_out)</b> for unbatched input, or <b>(num_layers, B, H_out)</b>. <br></br>
        /// </summary>
        /// <param name="input"></param>
        /// <param name="h_0"></param>
        /// <returns>
        /// output: <b>(L, H_out)</b> for unbatched input, or <b>(L, B, H_out)</b> when batch_first = false or <b>(B, L, H_out)</b> when batch_first = true. <br></br>
        /// h_n: <b>(num_layers, H_out)</b> for unbatched input or <b>(num_layers, B, H_out)</b>. <br></br>
        /// </returns>
        public (Tensor, Tensor) Forward(Tensor input, Tensor h_0)
        {
            Tensor input_clone = Tensor.Identity(input);
            Tensor h_0_clone = Tensor.Identity(h_0);

            if(input_clone.Rank != h_0_clone.Rank)
                throw new Exception($"Input ({input_clone.Shape.ToCommaSeparatedString()}) or H_0({h_0_clone.Shape.ToCommaSeparatedString()}) must have the same rank.");

            if (h_0_clone.Size(0) != modules.Count(x => x is RNNCell))
                throw new Exception($"H_0 must have the first dimension equal to num_layers ({modules.Count(x => x is RNNCell)})");


            bool isBatched = input_clone.Rank == 3;

            // Split input into sequence of length L and h_0 per layers           
            Tensor[] input_sequence = null;
            Tensor[] h_0_per_layers = null;
            if (isBatched)
            {
                if (batchFirst) // (B, L, H_in)
                {
                    input_sequence = Tensor.Split(input_clone, -2, 1);
                    for (int i = 0; i < input_sequence.Length; i++)
                    {
                        input_sequence[i].Squeeze(-2);
                    }
                }
                else // (L, B, H_in)
                {
                    input_sequence = Tensor.Split(input_clone, -3, 1);
                    for (int i = 0; i < input_sequence.Length; i++)
                    {
                        input_sequence[i].Squeeze(-3);
                    }

                }

                h_0_per_layers = Tensor.Split(h_0_clone, -3, 1);
                for (int i = 0; i < h_0_per_layers.Length; i++)
                {
                    h_0_per_layers[i].Squeeze(-3);
                }
            }
            else
            {
                input_sequence = Tensor.Split(input_clone, -2, 1);
                for (int i = 0; i < input_sequence.Length; i++)
                {
                    input_sequence[i].Squeeze(-2);
                }
                h_0_per_layers = Tensor.Split(h_0_clone, -2, 1);
                for (int i = 0; i < h_0_per_layers.Length; i++)
                {
                    h_0_per_layers[i].Squeeze(-2);
                }

            }


            // Parse sequencially through each module ---------------- done
            // input_sequence[i] (B, H_in)  or (H_in)

            int rnncell_index = 0;
            foreach (var module in modules)
            {
                if (module is RNNCell r)
                {
                    for (int t = 0; t < input_sequence.Length; t++)
                    {
                        h_0_per_layers[rnncell_index] = r.Forward(input_sequence[t], h_0_per_layers[rnncell_index]);
                        input_sequence[t] = Tensor.Identity(h_0_per_layers[rnncell_index]);
                        
                    }
                    rnncell_index++;
                }                  
                else if (module is Dropout d)
                {
                    for (int t = 0; t < input_sequence.Length; t++)
                    {
                        input_sequence[t] = d.Forward(input_sequence[t]);
                    }
                }
                else if (module is LayerNorm l)
                {
                    for (int t = 0; t < input_sequence.Length; t++)
                    {
                        input_sequence[t] = l.Forward(input_sequence[t]);
                    }
                }

                   
                // they are expelled ok
                // Debug.Log(h_0_per_layers[rnncell_index - 1]);

            }
           



            // input_sequence (B, H_in)
            // Join into output and h_n ----------------done
            Tensor h_n = Tensor.Cat(null, h_0_per_layers);
            Tensor output = null;
            if(batchFirst)
            {
                for (int i = 0; i < input_sequence.Length; i++)
                {
                    input_sequence[i].Unsqueeze(1);
                }
                output = Tensor.Cat(1, input_sequence);
                
            }
            else
            {
                output = Tensor.Cat(null, input_sequence);
            }
            
            return (output, h_n);
        }

        /// <summary>
        /// loss: <b>(L, H_out)</b> for unbatched input, or <b>(L, B, H_out)</b> when batch_first = false or <b>(B, L, H_out)</b> when batch_first = true. <br></br>
        /// </summary>
        /// <param name="loss"></param>
        public void Backward(Tensor loss)
        {
            Tensor loss_clone = Tensor.Identity(loss);
            // loss (L, B, H_out)/(B, L, H_Out) or (L, H_out)
            bool isBatched = loss.Rank == 3;

            // Split the loss into an array of sequences
            Tensor[] loss_sequence = null;
            if(isBatched) // (L, H_out)           
            {
                if (batchFirst) // (B, L, H_out)
                {
                    loss_sequence = loss_clone.Split(-2, 1);
                    for (int i = 0; i < loss_sequence.Length; i++)
                    {
                        loss_sequence[i].Squeeze(-2);
                    }
                }
                else // (L, B, H_out)
                {
                    loss_sequence = loss_clone.Split(-3, 1);
                    for (int i = 0; i < loss_sequence.Length; i++)
                    {
                        loss_sequence[i].Squeeze(-3);
                    }
                }
            }
            else
            {
                loss_sequence = loss_clone.Split(-2, 1);
                for (int i = 0; i < loss_sequence.Length; i++)
                {
                    loss_sequence[i].Squeeze(-2);
                }
            }
           


            // Backpropagate each sequence
            for (int t = loss_sequence.Length - 1; t >= 0; t--)
            {
                for (int m = modules.Length - 1; m >= 0; m--)
                {
                    //Debug.Log(loss_sequence[t]);
                    loss_sequence[t] = modules[m].Backward(loss_sequence[t]);
                }
            }
        }

        /// <summary>
        /// Gets all <typeparamref name="RNNCell"/> modules.
        /// </summary>
        /// <returns></returns>
        public Learnable[] Parameters() 
        { 
            return modules.Where(x => x is Learnable).Select(x => (Learnable) x).ToArray(); 
        }

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
            serializedModules = modules.Select(x => IModuleRNNWrapper.Wrap(x)).ToArray();
        }
        public void OnAfterDeserialize()
        {
            modules = serializedModules.Select(x => IModuleRNNWrapper.Unwrap(x)).ToArray();
        }
    }
}
