using System;
using Unity.VisualScripting;
using UnityEngine;


namespace DeepUnity.Modules
{
    [Serializable]
    public class DenseGPU : ILearnable, IModule
    {
        [SerializeField] public Device Device { get => Device.GPU; set { } }
        private TensorGPU gpu_InputCache;
        private bool UseBias { get => bias != null; }

        [SerializeField] public TensorGPU weight;
        [SerializeField] public TensorGPU bias;
        [NonSerialized] private TensorGPU weightGrad;
        [NonSerialized] private TensorGPU biasGrad;

        /// <summary>
        /// <b>A Dense layer with the parameters allocated on GPU.</b> <br><br></br></br>
        /// Input: <b>(B, H_in)</b>, <b>(H_in)</b> or <b>(B, L, H_in)</b>, <b>(L, H_in)</b> for sequential input.<br></br>
        /// Output: <b>(B, H_out)</b>, <b>(H_out)</b> or <b>(B, L, H_out)</b>, <b>(L, H_out)</b> for sequential input.<br></br>
        /// where B = batch_size, L = sequence_length, H_in = in_features and H_out = out_features.
        /// </summary>
        /// <param name="in_features"></param>
        /// <param name="out_features"></param>
        /// <param name="use_bias"></param>
        /// <param name="weight_init"></param>
        /// <param name="bias_init"></param>
        /// <exception cref="ArgumentException"></exception>
        public DenseGPU(int in_features, int out_features, bool use_bias = true, InitType weight_init = InitType.LeCun_Uniform, InitType bias_init = InitType.LeCun_Uniform)
        {
            throw new NotImplementedException("DenseGPU is in testing and was not released yet, it seems that is not even more efficient that the standard Dense");
            if (in_features < 1)
                throw new ArgumentException("In_features cannot be less than 1.");
            if (out_features < 1)
                throw new ArgumentException("Out_features cannot be less than 1.");

            weight = Parameter.CreateOnGPU(new int[] { out_features, in_features }, in_features, out_features, weight_init);
            weightGrad = TensorGPU.Zeros(weight.Shape);

            if (use_bias)
            {
                bias = Parameter.CreateOnGPU(new int[] { out_features }, in_features, out_features, bias_init);
                biasGrad = TensorGPU.Zeros(bias.Shape);
            }
        }
        private DenseGPU() { }

        public Tensor Predict(Tensor input)
        {

            if (input.Rank > 3)
                throw new ArgumentException($"Input must have the shape as (H_in), (B, H_in), (L, H_in) or (B, L, H_in), and the received input is ({input.Shape.ToCommaSeparatedString()})");

            if (input.Size(-1) != weight.Size(-1))
                throw new ShapeException($"Input features ({input.Size(-1)}) does not match with the Dense Layer features_num ({weight.Size(-1)}).");



            if (gpu_InputCache != null)
            {
                gpu_InputCache.Dispose();
            }

            gpu_InputCache = TensorGPU.Identity(input);




            if (input.Rank == 3)
            {
                throw new ArgumentException("DenseGPU was not optimized for rank 3 tensors");

                TensorGPU transposedWeights = TensorGPU.Transpose(weight, 0, 1);
                TensorGPU.Unsqueeze_(transposedWeights, 0);


                TensorGPU output = TensorGPU.BatchedMatMul(gpu_InputCache, transposedWeights);

                if (UseBias)
                {
                    TensorGPU expandedBiases = TensorGPU.Identity(bias);

                    TensorGPU.Unsqueeze_(expandedBiases, 0);
                    TensorGPU.Unsqueeze_(expandedBiases, 0);
                    TensorGPU expandedBiases1 = TensorGPU.Expand(expandedBiases, 1, output.Shape[1]);
                    TensorGPU expandedBiases2 = TensorGPU.Expand(expandedBiases1, 0, output.Shape[0]);
                    TensorGPU.Add_(output, expandedBiases2);

                    expandedBiases.Dispose();
                    expandedBiases1.Dispose();
                    expandedBiases2.Dispose();
                }

                transposedWeights.Dispose();


                Tensor oncpu = Tensor.Identity(output);
                output.Dispose();
                return oncpu;
            }



            bool isBatched = input.Rank == 2;
            int batch_size = isBatched ? input.Size(-2) : 1;
            int H_in = weight.Size(-1);
            int H_out = bias.Size(-1);
            ComputeShader cs = DeepUnityMeta.DenseCS;

            cs.SetBuffer(0, "input", gpu_InputCache.data);
            cs.SetBuffer(0, "gamma", weight.data);
            cs.SetBuffer(0, "beta", bias.data);
            ComputeBuffer gpu_outputBuff = new ComputeBuffer(batch_size * H_out, 4);
            cs.SetBuffer(0, "output", gpu_outputBuff);

            cs.SetInt("batch_size", batch_size);
            cs.SetInt("in_features", H_in);
            cs.SetInt("out_features", H_out);
            
            cs.Dispatch(0,
                   (H_out + 31) / 32,
                   (batch_size + 31) / 32,
                   1);

            Tensor result = isBatched ?
                   Tensor.Constant(gpu_outputBuff, batch_size, H_out) :
                   Tensor.Constant(gpu_outputBuff, H_out);

            gpu_outputBuff.Dispose();

            return result;
        }
        public Tensor Forward(Tensor input) => Predict(input);
        public Tensor Backward(Tensor loss)
        {
            if (loss.Size(-1) != weight.Size(0))
                throw new ArgumentException($"Hidden features of the loss ({loss.Size(-1)}) doesn't correspond to the hidden features returned by the dense layer ({weight.Size(0)}).");

            if (loss.Rank == 3)
            {
                throw new ArgumentException("DenseGPU was not optimized for rank 3 tensors");
                // TensorGPU wsGrad = TensorGPU.BatchedMatMul(loss.Transpose(1, 2), InputCache);
                // wsGrad /= loss.Size(0); // divide by batch size
                // wsGrad = wsGrad.Sum(0);
                // TensorGPU.Add_()
                // TensorGPU.CopyTo(wsGrad, weightsGrad);
                // 
                // if (UseBias)
                // {
                //     Tensor bGrad = loss.Mean(0).Sum(0);
                //     Tensor.CopyTo(bGrad, biasesGrad);
                // }
                // 
                // Tensor inputGrad = Tensor.BatchedMatMul(loss, weights.Unsqueeze(0).Expand(0, loss.Size(0)));
                // return inputGrad;
            }

            bool isBatched = loss.Rank == 2;
            int batch_size = isBatched ? loss.Size(-2) : 1;
            int H_in = weight.Size(-1);
            int H_out = bias.Size(-1);

            // dLoss w.r.t theta
            ComputeShader cs = DeepUnityMeta.DenseCS;

            TensorGPU gpu_loss = TensorGPU.Identity(loss);
            cs.SetBuffer(1, "loss", gpu_loss.data);

            cs.SetBuffer(1, "input", gpu_InputCache.data);

            cs.SetBuffer(1, "gamma_grad", weightGrad.data);

            if(UseBias)
            {
                cs.SetBuffer(1, "beta_grad", biasGrad.data);
            }
            else
            {
                ComputeBuffer biasesGradBuffer = new ComputeBuffer(H_out, 4);
                biasesGradBuffer.SetData(new float[H_out]);
            }
                    

            cs.SetInt("batch_size", batch_size);
            cs.SetInt("in_features", H_in);
            cs.SetInt("out_features", H_out);

            cs.Dispatch(1,
                (H_in + 31) / 32,
                (H_out + 31) / 32,
                1);

            TensorGPU gpu_inputGrad = TensorGPU.MatMul(gpu_loss, weight);
            Tensor inputGradOnCPU = Tensor.Identity(gpu_inputGrad);

            gpu_inputGrad.Dispose();
            gpu_loss.Dispose();
            gpu_InputCache.Dispose();

            return inputGradOnCPU;
        }

        public Parameter[] Parameters()
        {
            if (weightGrad == null)
                OnAfterDeserialize();

            return UseBias ?
                new Parameter[] { new Parameter(weight, weightGrad), new Parameter(bias, biasGrad) } :
                new Parameter[] { new Parameter(weight, weightGrad) };
        }
        public object Clone()
        {
            var dense = new DenseGPU();
            dense.weight = (TensorGPU)weight.Clone();
            dense.weightGrad = (TensorGPU)weightGrad.Clone();

            if (UseBias)
            {
                dense.bias = (TensorGPU)bias.Clone();
                dense.biasGrad = (TensorGPU)biasGrad.Clone();
            }

            return dense;
        }


        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (weight.Shape == null)
                return;

            if (weight.Shape.Length == 0)
                return;

            // do not check if gamma is != null...
            weightGrad = TensorGPU.Zeros(weight.Shape);

            if (UseBias)
                biasGrad = TensorGPU.Zeros(bias.Shape);

        }

    }


}



