using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    public class MultiheadNN : Model<MultiheadNN, Tensor[]>, ISerializationCallbackReceiver
    {
        [NonSerialized] private IModule[] backboneModules;
        [NonSerialized] private List<IModule[]> inputHeadsModules;
        [NonSerialized] private List<IModule[]> outputHeadsModules;

        [SerializeField] private IModuleWrapper[] serializedBackbone;
        [SerializeField] private List<IModuleWrapper[]> serializedInputHeads;
        [SerializeField] private List<IModuleWrapper[]> serializedOutputHeads;

        public MultiheadNN(params IModule[] backbone)
        {
            this.backboneModules = backbone;
        }
        public void AddInputHead(params IModule[] inputHead)
        {
            if (inputHeadsModules == null)
                inputHeadsModules = new List<IModule[]>();

            inputHeadsModules.Add(inputHead);
        }
        public void AddOutputHead(params IModule[] outputHead)
        {
            if(outputHeadsModules == null)
                outputHeadsModules = new List<IModule[]>();

            outputHeadsModules.Add(outputHead);
        }

        public override Tensor[] Predict(Tensor[] inputs)
        {
            if (inputs.Length != inputHeadsModules.Count)
                throw new ArgumentException("Input[] length does not matches the input heads of the multihead network.");


            // Pass the inputs through each input head.
            Tensor inputHeadsOutput = null;
            for (int i = 0; i < inputHeadsModules.Count; i++)
            {
                IModule[] input_head = inputHeadsModules[i];
                Tensor output = input_head[0].Predict(inputs[i]);
                for (int j = 1; j < input_head.Length; j++)
                {
                    output = input_head[j].Predict(output);
                }
                if (inputHeadsOutput == null)
                    inputHeadsOutput = output;
                else
                    inputHeadsOutput += output;
            }


            // Pass the summed input heads outputs through the backbone.
            Tensor backboneOutput = backboneModules[0].Predict(inputHeadsOutput);
            for (int i = 1; i < backboneModules.Length; i++)
            {
                backboneOutput = backboneModules[i].Predict(backboneOutput);
            }


            // Pass the backbone output through each output head.
            List<Tensor> outputs = new List<Tensor>();
            foreach (var output_head in outputHeadsModules)
            {
                Tensor output = output_head[0].Predict(backboneOutput);
                for (int i = 1; i < output_head.Length; i++)
                {
                    output = output_head[i].Predict(output);
                }
                outputs.Add(output);
            }

            return outputs.ToArray();
        }
        public override Tensor[] Forward(Tensor[] inputs)
        {
            if (inputs.Length != inputHeadsModules.Count)
                throw new ArgumentException("Input[] length does not matches the input heads of the multihead network.");


            // Pass the inputs through each input head.
            Tensor inputHeadsOutput = null;
            for (int i = 0; i < inputHeadsModules.Count; i++)
            {
                IModule[] input_head = inputHeadsModules[i];
                Tensor output = input_head[0].Forward(inputs[i]);
                for (int j = 1; j < input_head.Length; j++)
                {
                    output = input_head[i].Forward(output);
                }
                if (inputHeadsOutput == null)
                    inputHeadsOutput = output;
                else
                    inputHeadsOutput += output;
            }


            // Pass the summed input heads outputs through the backbone.
            Tensor backboneOutput = backboneModules[0].Forward(inputHeadsOutput);
            for (int i = 1; i < backboneModules.Length; i++)
            {
                backboneOutput = backboneModules[i].Forward(backboneOutput);
            }

            // Pass the backbone output through each output head.
            List<Tensor> outputs = new List<Tensor>();
            foreach (var output_head in outputHeadsModules)
            {
                Tensor output = output_head[0].Forward(backboneOutput);
                for (int i = 1; i < output_head.Length; i++)
                {
                    output = output_head[i].Forward(output);
                }
                outputs.Add(output);
            }

            return outputs.ToArray();
        }
        public override Tensor[] Backward(Tensor[] lossDerivative)
        {
            throw new NotImplementedException();
        }


        public override Tensor[] Parameters()
        {
            List<Tensor> parameters = new List<Tensor>();
            foreach (var item in inputHeadsModules.OfType<ILearnable>())
            {
                parameters.AddRange(item.Parameters());
            }
            foreach (var item in backboneModules.OfType<ILearnable>())
            {
                parameters.AddRange(item.Parameters());
            }
            foreach (var item in outputHeadsModules.OfType<ILearnable>())
            {
                parameters.AddRange(item.Parameters());
            }
            return parameters.ToArray();
        }
        public override Tensor[] Gradients()
        {
            List<Tensor> parameters = new List<Tensor>();
            foreach (var item in inputHeadsModules.OfType<ILearnable>())
            {
                parameters.AddRange(item.Gradients());
            }
            foreach (var item in backboneModules.OfType<ILearnable>())
            {
                parameters.AddRange(item.Gradients());
            }
            foreach (var item in outputHeadsModules.OfType<ILearnable>())
            {
                parameters.AddRange(item.Gradients());
            }
            return parameters.ToArray();
        }
        public void OnBeforeSerialize()
        {
            for (int i = 0; i < inputHeadsModules.Count; i++)
            {
                serializedInputHeads[i] = inputHeadsModules[i].Select(x => IModuleWrapper.Wrap(x)).ToArray();
            }
            serializedBackbone = backboneModules.Select(x => IModuleWrapper.Wrap(x)).ToArray();
            for (int i = 0; i < outputHeadsModules.Count; i++)
            {
                serializedOutputHeads[i] = outputHeadsModules[i].Select(x => IModuleWrapper.Wrap(x)).ToArray();
            }
        }
        public void OnAfterDeserialize()
        {
            for (int i = 0; i < inputHeadsModules.Count; i++)
            {
                inputHeadsModules[i] = serializedInputHeads[i].Select(x => IModuleWrapper.Unwrap(x)).ToArray();
            }
            backboneModules = serializedBackbone.Select(x => IModuleWrapper.Unwrap(x)).ToArray();
            for (int i = 0; i < outputHeadsModules.Count; i++)
            {
                outputHeadsModules[i] = serializedOutputHeads[i].Select(x => IModuleWrapper.Unwrap(x)).ToArray();
            }
        }


        public override string Summary()
        {
            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.AppendLine($"Name: {name}");
            stringBuilder.AppendLine($"Type: {GetType().Name}");
            stringBuilder.AppendLine($"Input heads: {inputHeadsModules.Count}");
            int head_index = 1;
            foreach (var head in inputHeadsModules)
            {
                stringBuilder.AppendLine($"         Head {head_index++}:");
                foreach (var layer in head)
                {
                    stringBuilder.AppendLine($"               {layer.GetType().Name}");
                }
                
            }
            stringBuilder.AppendLine($"Backbone : {backboneModules.Length}");
            foreach (var module in backboneModules)
            {
                stringBuilder.AppendLine($"         {module.GetType().Name}");
            }
            head_index = 1;
            foreach (var head in outputHeadsModules)
            {
                stringBuilder.AppendLine($"         Head {head_index++}:");
                foreach (var layer in head)
                {
                    stringBuilder.AppendLine($"               {layer.GetType().Name}");
                }

            }
            int total_params = inputHeadsModules.Sum(x => x.OfType<ILearnable>().Sum(x => x.ParametersCount()));
            total_params += backboneModules.OfType<ILearnable>().Sum(x => x.ParametersCount());
            total_params += outputHeadsModules.Sum(x => x.OfType<ILearnable>().Sum(x => x.ParametersCount()));
            stringBuilder.AppendLine($"Parameters: {total_params}");
            return stringBuilder.ToString();
        }

        public override object Clone()
        {
            IModule[] cloned_back = backboneModules.Select(x => (IModule)x.Clone()).ToArray();
            var multihead = new MultiheadNN(cloned_back);
            foreach (var item in inputHeadsModules)
            {
                multihead.AddInputHead((IModule)item.Clone());
            }
            foreach (var item in outputHeadsModules)
            {
                multihead.AddOutputHead((IModule)item.Clone());
            }
            return multihead;
        }


    }
}


