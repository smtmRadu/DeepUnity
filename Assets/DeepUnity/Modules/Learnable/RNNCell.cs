using System;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    // https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html#torch.nn.RNNCell
    /// <summary>
    /// An Elman RNNCell with Tanh or ReLU non-linearity
    /// </summary>
    [Serializable]
    public class RNNCell : ILearnable, IModule2
    {
        [SerializeField] private Device device;
        [SerializeField] private NonLinearity nonlinearity;

        // These ones are lists for each timestep in the sequence
        [NonSerialized] private Stack<Tensor> InputCache;
        [NonSerialized] private Stack<Tensor> HiddenCache;
        [NonSerialized] private Stack<Activation> Activations;


        [SerializeField] private Tensor weights;
        [SerializeField] private Tensor biases;
        [NonSerialized]  private Tensor weightsGrad;
        [NonSerialized]  private Tensor biasesGrad;
        [SerializeField, Tooltip("weight hidden->hidden")]          private Tensor recurrentWeights;
        [SerializeField, Tooltip("bias hidden->hidden")]            private Tensor recurrentBiases;
        [NonSerialized, Tooltip("weight hidden->hidden gradient")]  private Tensor recurrentWeightsGrad;
        [NonSerialized, Tooltip("bias hidden->hidden gradient")]    private Tensor recurrentBiasesGrad;


        /// <summary>
        /// Inputs: <b>input (B, H_in)</b> or <b>(H_in)</b> and <b>hidden (B, H_out)</b> or <b>(H_out)</b>. <br></br>
        /// Output: <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input. <br></br>
        /// where B = batch_size, H_in = input_size and H_out = hidden_size.
        /// </summary>
        /// <param name="input_size"></param>
        /// <param name="hidden_size"></param>
        /// <param name="nonlinearity"></param>
        public RNNCell(int input_size, int hidden_size, NonLinearity nonlinearity = NonLinearity.Tanh, Device device = Device.CPU)
        {
            this.device = device;
            this.nonlinearity = nonlinearity;
            InputCache = new Stack<Tensor>();
            HiddenCache = new Stack<Tensor>();
            Activations = new Stack<Activation>();


            weights = Initializer.CreateParameter(new int[] { hidden_size, input_size }, input_size, hidden_size, InitType.Glorot_Uniform);
            biases = Initializer.CreateParameter(new int[] { hidden_size }, input_size, hidden_size, InitType.Zeros);
            weightsGrad = Tensor.Zeros(weights.Shape);
            biasesGrad = Tensor.Zeros(biases.Shape);
            recurrentWeights = Initializer.CreateParameter(new int[] { hidden_size, hidden_size }, hidden_size, hidden_size, InitType.Glorot_Uniform);
            recurrentBiases = Initializer.CreateParameter(new int[] { hidden_size }, hidden_size, hidden_size, InitType.Zeros);
            recurrentWeightsGrad = Tensor.Zeros(hidden_size, hidden_size);  // weight_hh_g
            recurrentBiasesGrad = Tensor.Zeros(hidden_size);                // bias_hh_g
        }

        /// <summary>
        /// <b>input: (B, H_in)</b> or <b>(H_in)</b> for unbatched input. <br></br> 
        /// <b>hidden: (B, H_out)</b> or <b>(H_out)</b> for unbatched input. <br></br>
        /// </summary>
        /// <param name="input"></param>
        /// <param name="hidden"></param>
        /// <returns><b>h_n</b>: <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input.</returns>
        public Tensor Predict(Tensor input, Tensor hidden)
        {
            if (input.Size(-1) != weights.Size(-1))
                throw new ShapeException($"Input last dimension ({input.Size(-1)}) must be equal to input_size ({weights.Size(-1)})");

            if (hidden.Size(-1) != weights.Size(-2))
                throw new ShapeException($"Hidden last dimension ({hidden.Size(-2)}) must be equal to hidden_size ({weights.Size(-2)})");



            switch (nonlinearity)
            {
                case NonLinearity.Tanh:
                    Activations.Push(new Tanh());
                    break;
                case NonLinearity.ReLU:
                    Activations.Push(new ReLU());
                    break;
                default:
                    throw new Exception("Unhandled nonlinearity type for RNNCell.");
            }

            // gamma (H_in, H_out)
            // rgamma(H_out, H_out)
            // input (B, H_in)
            // hidden[t-1](H_out)
            // h[t] = tanh(input * gammaT + beta + hidden[t-1]*recurrentGammaT + recurrentBeta)
            int batch_size = input.Rank == 2 ? input.Size(-2) : 1;


            Tensor H_Prime;
            Tensor expandedBiases = batch_size == 1 ? biases : Tensor.Expand(Tensor.Unsqueeze(biases, 0), 0, batch_size);
            Tensor expandedRecurrentBiases = batch_size == 1 ? recurrentBiases : Tensor.Expand(Tensor.Unsqueeze(recurrentBiases, 0), 0, batch_size);
            if (device == Device.CPU)
            {
                H_Prime = Tensor.MatMul(input, Tensor.Transpose(weights, 0, 1)) + expandedBiases +
                                 Tensor.MatMul(hidden, Tensor.Transpose(recurrentWeights, 0, 1)) + expandedRecurrentBiases;

            }
            else
            {
                H_Prime = Tensor.MatMulGPU(input, Tensor.Transpose(weights, 0, 1)) + expandedBiases +
                                 Tensor.MatMulGPU(hidden, Tensor.Transpose(recurrentWeights, 0, 1)) + expandedRecurrentBiases;
            }

            return Activations.Peek().Forward(H_Prime);
        }
        /// <summary>
        /// <b>input: (B, H_in)</b> or <b>(H_in)</b> for unbatched input. <br></br> 
        /// <b>hidden: (B, H_out)</b> or <b>(H_out)</b> for unbatched input. <br></br>
        /// </summary>
        /// <param name="input"></param>
        /// <param name="hidden"></param>
        /// <returns><b>h_n</b>: <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input.</returns>
        public Tensor Forward(Tensor input, Tensor hidden)
        {
            
            InputCache.Push(Tensor.Identity(input));
            HiddenCache.Push(Tensor.Identity(hidden));

            return Predict(input, hidden);

        }
        /// <summary>
        /// h_n_grad:  <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input.
        /// </summary>
        /// <returns></returns>
        public Tensor Backward(Tensor dLdhPrime)
        {
            // backward through the activation
            dLdhPrime = Activations.Pop().Backward(dLdhPrime);

            bool isBatched = dLdhPrime.Rank == 2;
            int batch_size = isBatched ? dLdhPrime.Size(-2) : 1;
            Tensor InputCache_t = InputCache.Pop();
            Tensor H_0Cache_t = HiddenCache.Pop();
            
            if (isBatched && batch_size == 1)
            {
                dLdhPrime = dLdhPrime.Unsqueeze(0);
                InputCache_t = InputCache_t.Unsqueeze(0);
            }
            // loss (B, H_out)
            // lossT (H_out, B)
            // input (B, H_in)
            // w_ih (H_out, H_in) gamma
            // b_ih (H_out) beta
            // w_hh (H_out, H_out) rgamma
            // b_hh (H_out) rbeta
            Tensor transposedLoss = Tensor.Transpose(dLdhPrime, 0, 1);

            // compute gradients wrt parameters.
            // h[t] = tanh(input * gammaT + beta + hidden[t-1]*recurrentGammaT + recurrentBeta) = tanh(z)
            //grad W = dL/dz * dzdW = dL/dz * x
            //grad B = dL/dz
            //grad Wr = dL/dz * hidden
            //grad Br = dL/dz
            //grad x = W * dLdz
            if(device == Device.CPU)
            {
                Tensor.CopyTo(weightsGrad + Tensor.MatMul(transposedLoss, InputCache_t) / batch_size, weightsGrad);
                Tensor.CopyTo(recurrentWeightsGrad + Tensor.MatMul(transposedLoss, H_0Cache_t) / batch_size, recurrentWeightsGrad);
            }
            else
            {
                Tensor.CopyTo(weightsGrad + Tensor.MatMulGPU(transposedLoss, InputCache_t) / batch_size, weightsGrad);
                Tensor.CopyTo(recurrentWeightsGrad + Tensor.MatMulGPU(transposedLoss, H_0Cache_t) / batch_size, recurrentWeightsGrad);
            }

            Tensor.CopyTo(biasesGrad + Tensor.Mean(dLdhPrime, axis: 0), biasesGrad);      
            Tensor.CopyTo(recurrentBiasesGrad + Tensor.Mean(dLdhPrime, axis: 0), recurrentBiasesGrad);

            // compute gradients wrt InputCache.
            Tensor inputGrad = Tensor.MatMul(dLdhPrime, weights);
            return inputGrad.Squeeze(-2);
        }

        public void SetDevice(Device device) { this.device = device; }
        public int ParametersCount()
        {
            return weights.Count() + biases.Count() + recurrentWeights.Count() + recurrentBiases.Count();
        }
        public Parameter[] Parameters()
        {
            if (weightsGrad == null)
                OnAfterDeserialize();

            var w = new Parameter(weights, weightsGrad);
            var b = new Parameter(biases, biasesGrad);
            var rw = new Parameter(recurrentWeights, recurrentWeightsGrad);
            var rb = new Parameter(recurrentBiases, recurrentBiasesGrad);

            return new Parameter[] { w, b, rw, rb };
        }

        public virtual void OnBeforeSerialize()
        {

        }
        public virtual void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (weights.Shape == null)
                return;

            if (weights.Shape.Length == 0)
                return;

            InputCache = new Stack<Tensor>();
            HiddenCache = new Stack<Tensor>();
            Activations = new Stack<Activation>();

            // do not check if gamma is != null...
            this.weightsGrad = Tensor.Zeros(weights.Shape);
            this.biasesGrad = Tensor.Zeros(biases.Shape);

            recurrentWeightsGrad = Tensor.Zeros(recurrentWeights.Shape);  // weight_hh_g
            recurrentBiasesGrad = Tensor.Zeros(recurrentBiases.Shape);    // bias_hh_g  

        }     
        public object Clone()
        {
            var rnncell = new RNNCell(1, 1, this.nonlinearity);
            rnncell.InputCache = new Stack<Tensor>();
            rnncell.HiddenCache = new Stack<Tensor>();
            rnncell.Activations = new Stack<Activation>();
            rnncell.weights = (Tensor)this.weights.Clone();
            rnncell.biases = (Tensor)this.biases.Clone();
            rnncell.recurrentWeights = (Tensor)this.recurrentWeights.Clone();
            rnncell.recurrentBiases = (Tensor)this.recurrentBiases.Clone();
            rnncell.weightsGrad = (Tensor)this.weightsGrad.Clone();
            rnncell.biasesGrad = (Tensor)this.biasesGrad.Clone();
            rnncell.recurrentWeightsGrad = (Tensor)this.recurrentWeightsGrad.Clone();
            rnncell.recurrentBiasesGrad = (Tensor)this.recurrentBiasesGrad.Clone();
            return rnncell;
        }
    }
}

