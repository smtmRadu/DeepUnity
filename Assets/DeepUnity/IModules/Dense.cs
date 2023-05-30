using UnityEngine;
using System;
using UnityEditor;

namespace DeepUnity
{
    public enum WeightInit
    {
        HE,
        Xavier,
        Normal,
        Random01,
        Ones
    }
    public enum Device
    {
        CPU,
        GPU
    }
    public class Dense : IModule
    {
        public ComputeShader MatMulCS;
        public Tensor<float> InputCache { get; set; }
        
        // Parameters
        public Tensor<float> Weights;
        public Tensor<float> Biases;

        // Gradients
        public Tensor<float> gWeights;
        public Tensor<float> gBiases;

        // Momentums
        public Tensor<float> mWeights;
        public Tensor<float> mBiases;

        // Velocities
        public Tensor<float> vWeights;
        public Tensor<float> vBiases;

        public Dense(int inputs, int outputs, WeightInit init = WeightInit.HE, Device device = Device.GPU)
        {
            if (device == Device.GPU)
            {
                string csguid = AssetDatabase.FindAssets("MatMulCS")[0];
                string cspath = AssetDatabase.GUIDToAssetPath(csguid);
                this.MatMulCS = AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
            }
            else
                this.MatMulCS = null;
                        
            this.Weights = Tensor<float>.Zeros(outputs, inputs);
            this.Biases = Tensor<float>.Zeros(outputs);

            this.gWeights = Tensor<float>.Zeros(outputs, inputs);
            this.gBiases = Tensor<float>.Zeros(outputs);

            this.mWeights = Tensor<float>.Zeros(outputs, inputs);
            this.mBiases = Tensor<float>.Zeros(outputs);

            this.vWeights = Tensor<float>.Zeros(outputs, inputs);
            this.vBiases = Tensor<float>.Zeros(outputs);

            switch (init)
            {
                case WeightInit.HE:
                    float sigmaHE = MathF.Sqrt(2f / Weights.FullShape[1]);
                    Weights.ForEach(x => Utils.Random.Gaussian(0f, sigmaHE, out _));
                    break;
                case WeightInit.Xavier:
                    float sigmaXA = MathF.Sqrt(2f / (Weights.FullShape[0] + Weights.FullShape[1]));
                    Weights.ForEach(x => Utils.Random.Gaussian(0f, sigmaXA, out _));
                    break;
                case WeightInit.Normal:
                    Weights.ForEach(x => Utils.Random.Gaussian(0f, 1f, out _));
                    break;
                case WeightInit.Random01:
                    Weights.ForEach(x => Utils.Random.Value * 2f - 1f);
                    break;
                case WeightInit.Ones:
                    Weights.ForEach(x => 1f);
                    break;
                default:
                    throw new Exception("Unhandled initialization type!");
            }
        }

        public Tensor<float> Forward(Tensor<float> input)
        {
            InputCache = input.Clone() as Tensor<float>;
            return Tensor<float>.MatMul(Weights, input, MatMulCS) + Biases;
        }
        public Tensor<float> Backward(Tensor<float> loss)
        {
            var transposedInput = Tensor<float>.MatTranspose(InputCache);
            Tensor<float> gradW = Tensor<float>.MatMul(loss, transposedInput, MatMulCS);
            Tensor<float> gradB = Tensor<float>.MatMul(loss, transposedInput.Select(x => 1f), MatMulCS);

            // Average the gradients
            float batch = loss.FullShape[1];

            // Update the gradients
            gWeights += gradW / batch;
            gBiases += gradB / batch;

            // Backpropagate the loss
            Tensor<float> dLossdInput = Tensor<float>.MatMul(Tensor<float>.MatTranspose(Weights), loss, MatMulCS);

            return dLossdInput;
        }
    }

}

