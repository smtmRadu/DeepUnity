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
        Random,
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
        public Tensor InputCache { get; set; }

        // Parameters
        public Tensor Weights;
        public Tensor Biases;

        // Gradients
        public Tensor gWeights;
        public Tensor gBiases;

        // Momentums
        public Tensor mWeights;
        public Tensor mBiases;

        // Velocities
        public Tensor vWeights;
        public Tensor vBiases;

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

            this.Weights = Tensor.Zeros(outputs, inputs);
            this.Biases = Tensor.Zeros(outputs);

            this.gWeights = Tensor.Zeros(outputs, inputs);
            this.gBiases = Tensor.Zeros(outputs);

            this.mWeights = Tensor.Zeros(outputs, inputs);
            this.mBiases = Tensor.Zeros(outputs);

            this.vWeights = Tensor.Zeros(outputs, inputs);
            this.vBiases = Tensor.Zeros(outputs);

            switch (init)
            {
                case WeightInit.HE:
                    float sigmaHE = MathF.Sqrt(2f / Weights.Shape[1]);
                    Weights.ForEach(x => Utils.Random.Gaussian(0f, sigmaHE, out _));
                    break;
                case WeightInit.Xavier:
                    float sigmaXA = MathF.Sqrt(2f / (Weights.Shape[0] + Weights.Shape[1]));
                    Weights.ForEach(x => Utils.Random.Gaussian(0f, sigmaXA, out _));
                    break;
                case WeightInit.Normal:
                    Weights.ForEach(x => Utils.Random.Gaussian(0f, 1f, out _));
                    break;
                case WeightInit.Random:
                    Weights.ForEach(x => Utils.Random.Value * 2f - 1f);
                    break;
                case WeightInit.Ones:
                    Weights.ForEach(x => 1f);
                    break;
                default:
                    throw new Exception("Unhandled initialization type!");
            }
        }

        public Tensor Forward(Tensor input)
        {
            InputCache = input.Clone() as Tensor;
            return Tensor.MatMul(Weights, input, MatMulCS) + Biases;
        }
        public Tensor Backward(Tensor loss)
        {
            var transposedInput = Tensor.MatTranspose(InputCache);
            Tensor gradW = Tensor.MatMul(loss, transposedInput, MatMulCS);
            Tensor gradB = Tensor.MatMul(loss, Tensor.Ones(transposedInput.Shape), MatMulCS);

            // Average the gradients
            float batch = loss.Shape[1];

            // Update the gradients
            gWeights += gradW / batch;
            gBiases += gradB / batch;

            // Backpropagate the loss
            Tensor dLossdActivation = Tensor.MatMul(Tensor.MatTranspose(Weights), loss, MatMulCS);

            return dLossdActivation;
        }
    }

}