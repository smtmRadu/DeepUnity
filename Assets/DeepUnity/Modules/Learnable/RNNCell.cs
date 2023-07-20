using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class RNNCell : Learnable, IModuleRNN
    {
        public Tensor InputCache { get; set; }

        [SerializeField] private NonLinearity nonlinearity;
        [SerializeField] public Tensor weightIH;
        [SerializeField] public Tensor weightHH;
        [SerializeField] public Tensor biasIH;
        [SerializeField] public Tensor biasHH;

        [NonSerialized] private ActivationBase activation;
        [NonSerialized] public Tensor weightIHGrad;
        [NonSerialized] public Tensor weightHHGrad;
        [NonSerialized] public Tensor biasIHGrad;
        [NonSerialized] public Tensor biasHHGrad;


        /// <summary>
        /// Inputs: <b>input (B, H_in)</b> or <b>(H_in)</b> [and <b>hidden (B, H_in)</b> or <b>(H_in)]</b>. <br></br>
        /// Output: <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input. <br></br>
        /// where B = batch_size, H_in = input_size and H_out = hidden_size.
        /// </summary>
        /// <param name="input_size"></param>
        /// <param name="hidden_size"></param>
        /// <param name="nonlinearity"></param>
        public RNNCell(int input_size, int hidden_size, NonLinearity nonlinearity = NonLinearity.Tanh) : base(Device.CPU) 
        {
            this.nonlinearity = nonlinearity;
            switch (nonlinearity)
            {
                case NonLinearity.Tanh:
                    this.activation = new Tanh();
                    break;
                case NonLinearity.ReLU:
                    this.activation = new ReLU();
                    break;
                default:
                    throw new NotImplementedException("Unhandled nonlinearity type.");
            }

            float sqrtK = MathF.Sqrt(1f / hidden_size);
            weightHH = Tensor.RandomRange((-sqrtK, sqrtK), hidden_size, input_size);
            weightHH = Tensor.RandomRange((-sqrtK, sqrtK), hidden_size, hidden_size);
            biasIH = Tensor.RandomRange((-sqrtK, sqrtK), hidden_size);
            biasHH = Tensor.RandomRange((-sqrtK, sqrtK), hidden_size);
        }
        public Tensor Forward(Tensor input, Tensor hidden)
        {
            InputCache = Tensor.Identity(input);



            return activation.Forward(input);
        }
        public Tensor Backward(Tensor hPrimeGrad)
        {
            hPrimeGrad = activation.Backward(hPrimeGrad);
            ZeroGrad();

            // compute gradients wrt parameters.
            // compute gradients wrt InputCache.

            Optimise();
            return null;
        }

        public void ZeroGrad()
        {
            weightIHGrad = Tensor.Zeros(weightIHGrad.Shape);
            weightHHGrad = Tensor.Zeros(weightHHGrad.Shape);
            biasIHGrad = Tensor.Zeros(biasIHGrad.Shape);
            biasHHGrad = Tensor.Zeros(biasHHGrad.Shape);
        }
        public void Optimise(float lr = 0.01f)
        {
            weightIH = -lr * weightIHGrad;
            weightHH = -lr * weightHHGrad;
            biasIH = -lr * biasIHGrad;
            biasHH = -lr * biasHHGrad;
        }
        public override void OnBeforeSerialize() { }
        public override void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            try
            {
                var dev = device;

                // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.
                var x = weightIH.Shape;
                if (x == null || x.Length == 0)
                    throw new Exception("Is not even important...");

            }
            catch
            {
                return;
            }

            switch (nonlinearity)
            {
                case NonLinearity.Tanh:
                    this.activation = new Tanh();
                    break;
                case NonLinearity.ReLU:
                    this.activation = new ReLU();
                    break;
                default:
                    throw new NotImplementedException("Unhandled nonlinearity type.");
            }

            this.weightIHGrad = Tensor.Zeros(weightIH.Shape);
            this.weightHHGrad = Tensor.Zeros(weightHH.Shape);
            this.biasIHGrad = Tensor.Zeros(biasIH.Shape);
            this.biasHHGrad = Tensor.Zeros(biasIH.Shape);          
        }
    }
}

