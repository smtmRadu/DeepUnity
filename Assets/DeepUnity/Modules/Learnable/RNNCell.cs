using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class RNNCell : Learnable, IModuleS, ISelfOptimizable
    {
        [SerializeField] private NonLinearity nonlinearity;

        // These ones are lists for each timestep in the sequence
        [NonSerialized] private Stack<Tensor> InputCache;
        [NonSerialized] private Stack<Tensor> HiddenCache;
        [NonSerialized] private Stack<Activation> Activations;
       

        
        [SerializeField, Tooltip("weight hidden->hidden")]          public Tensor recurrentGamma;
        [SerializeField, Tooltip("bias hidden->hidden")]            public Tensor recurrentBeta;
        [NonSerialized, Tooltip("weight hidden->hidden gradient")]  public Tensor recurrentGammaGrad;
        [NonSerialized, Tooltip("bias hidden->hidden gradient")]    public Tensor recurrentBetaGrad;


        /// <summary>
        /// Inputs: <b>input (B, H_in)</b> or <b>(H_in)</b> and <b>hidden (B, H_out)</b> or <b>(H_out)</b>. <br></br>
        /// Output: <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input. <br></br>
        /// where B = batch_size, H_in = input_size and H_out = hidden_size.
        /// </summary>
        /// <param name="input_size"></param>
        /// <param name="hidden_size"></param>
        /// <param name="nonlinearity"></param>
        public RNNCell(int input_size, int hidden_size, NonLinearity nonlinearity = NonLinearity.Tanh) : 
            base(Device.CPU,
                InitType.Glorot_Uniform,
                InitType.Zeros,
                new int[] {hidden_size, input_size},
                new int[] {hidden_size},
                input_size,
                hidden_size) 
        {
            this.nonlinearity = nonlinearity;
            InputCache = new Stack<Tensor>();
            HiddenCache = new Stack<Tensor>();
            Activations = new Stack<Activation>();


            float sqrtK = MathF.Sqrt(1f / hidden_size);
            var range = (-sqrtK, sqrtK);

            recurrentGamma = Tensor.RandomRange(range, hidden_size, hidden_size);  // weight_hh
            recurrentBeta = Tensor.RandomRange(range, hidden_size);                // bias_hh
            recurrentGammaGrad = Tensor.Zeros(hidden_size, hidden_size);  // weight_hh_g
            recurrentBetaGrad = Tensor.Zeros(hidden_size);                // bias_hh_g
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
            if(input.Size(-1) != gamma.Size(-1))
                throw new ShapeException($"Input last dimension ({input.Size(-1)}) must be equal to input_size ({gamma.Size(-1)})");

            if(hidden.Size(-1) != gamma.Size(-2))
                throw new ShapeException($"Hidden last dimension ({hidden.Size(-2)}) must be equal to hidden_size ({gamma.Size(-2)})");

            InputCache.Push(Tensor.Identity(input));
            HiddenCache.Push(Tensor.Identity(hidden));
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
            // h' = tanh(input * gammaT + beta + hidden[t-1]*recurrentGammaT + recurrentBeta)
            int batch_size = input.Rank == 2 ? input.Size(-2) : 1;


            Tensor H_Prime = null;
            if(batch_size == 1)
            {
                H_Prime = Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + beta +
                                 Tensor.MatMul(hidden, Tensor.Transpose(recurrentGamma, 0, 1)) + recurrentBeta;
               
            }
            else
            {
                H_Prime = Tensor.MatMul(input, Tensor.Transpose(gamma, 0, 1)) + 
                                 Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size) +
                                 Tensor.MatMul(hidden, Tensor.Transpose(recurrentGamma, 0, 1)) + 
                                 Tensor.Expand(Tensor.Unsqueeze(recurrentBeta, 0), 0, batch_size);
            }

            return Activations.Peek().Forward(H_Prime);
        }

        /// <summary>
        /// h_n_grad:  <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input.
        /// </summary>
        /// <returns></returns>
        public Tensor Backward(Tensor h_n_grad)
        {
            h_n_grad = Activations.Pop().Backward(h_n_grad);

            int batch_size = h_n_grad.Rank == 2 ? h_n_grad.Size(-2) : 1;
            Tensor InputCache_t = InputCache.Last();
            Tensor H_0Cache_t = HiddenCache.Last();
            InputCache.Pop();
            HiddenCache.Pop();
            
            if (batch_size == 1)
            {
                h_n_grad = h_n_grad.Unsqueeze(0);
                InputCache_t = InputCache_t.Unsqueeze(0);
            }
            // loss (B, H_out)
            // lossT (H_out, B)
            // input (B, H_in)
            // w_ih (H_out, H_in) gamma
            // b_ih (H_out) beta
            // w_hh (H_out, H_out) rgamma
            // b_hh (H_out) rbeta
            Tensor transposedLoss = Tensor.Transpose(h_n_grad, 0, 1);

            // compute gradients wrt parameters.
            gammaGrad += Tensor.MatMul(transposedLoss, InputCache_t) / batch_size;
            betaGrad += Tensor.Mean(transposedLoss, axis: 1);
            recurrentGammaGrad += Tensor.MatMul(transposedLoss, H_0Cache_t) / batch_size;
            recurrentBetaGrad += Tensor.Mean(transposedLoss, axis: 1);

            // compute gradients wrt InputCache.
            Tensor inputGrad = Tensor.MatMul(h_n_grad, gamma);
            return inputGrad.Squeeze(-2);
        }

        public void SelfOptimise(float lr)
        {
            recurrentGamma = -lr * recurrentGammaGrad;
            recurrentBeta = -lr * recurrentBetaGrad;
        }
        public override void ZeroGrad()
        {
            base.ZeroGrad();
            recurrentGammaGrad = Tensor.Zeros(recurrentGammaGrad.Shape);
            recurrentBetaGrad = Tensor.Zeros(recurrentBetaGrad.Shape);
        }
        public override void ClipGradValue(float clip_value)
        {
            base.ClipGradValue(clip_value);
            recurrentGammaGrad = Tensor.Clip(recurrentGammaGrad, -clip_value, clip_value);
            recurrentBetaGrad = Tensor.Clip(recurrentBetaGrad, -clip_value, clip_value);
        }
        public override void ClipGradNorm(float max_norm)
        {
            // Maybe it can be modified in the future ...

            Tensor normG = Tensor.Norm(gammaGrad, NormType.ManhattanL1);

            if (normG[0] > max_norm)
            {
                float scale = max_norm / normG[0];
                gammaGrad *= scale;
            }

            Tensor normB = Tensor.Norm(betaGrad, NormType.ManhattanL1);

            if (normB[0] > max_norm)
            {
                float scale = max_norm / normB[0];
                betaGrad *= scale;
            }

            Tensor rnormG = Tensor.Norm(recurrentGammaGrad, NormType.ManhattanL1);

            if (rnormG[0] > max_norm)
            {
                float scale = max_norm / rnormG[0];
                recurrentGammaGrad *= scale;
            }

            Tensor rnormB = Tensor.Norm(recurrentBetaGrad, NormType.ManhattanL1);

            if (rnormB[0] > max_norm)
            {
                float scale = max_norm / rnormB[0];
                recurrentBetaGrad *= scale;
            }
            
        }       
        public override int ParametersCount()
        {
            return base.ParametersCount() + recurrentGamma.Count() + recurrentBeta.Count();
        }
        public override void OnBeforeSerialize() { }
        public override void OnAfterDeserialize()
        {
            base.OnAfterDeserialize();

            InputCache = new Stack<Tensor>();
            HiddenCache = new Stack<Tensor>();
            Activations = new Stack<Activation>();

            if (recurrentGamma.Shape == null)
                return;

            if (recurrentGamma.Shape.Length == 0)
                return;

            recurrentGammaGrad = Tensor.Zeros(recurrentGamma.Shape);  // weight_hh_g
            recurrentBetaGrad = Tensor.Zeros(recurrentBeta.Shape);    // bias_hh_g  

            InputCache = new();
            HiddenCache = new();
        }
    }
}

