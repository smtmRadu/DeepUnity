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
    
    [Serializable]
    public class Dense : IModule, ISerializationCallbackReceiver
    {
        public Tensor InputCache { get; set; }
        [SerializeField] public ComputeShader MatMulCS;
        
        // Parameters (theta)
        [SerializeField] public Tensor t_W;
        [SerializeField] public Tensor t_B;

        // Gradients (g)
        [NonSerialized] public Tensor g_W;
        [NonSerialized] public Tensor g_B;

        // 1st momentum buffer
        [NonSerialized] public Tensor m_W;
        [NonSerialized] public Tensor m_B;

        // 2nd momentum buffer 
        [NonSerialized] public Tensor v_W;
        [NonSerialized] public Tensor v_B;

        public Dense(int inputs, int outputs, WeightInit init = WeightInit.HE, Device device = Device.GPU)
        {
            if(device == Device.GPU)
            {
                string csguid = AssetDatabase.FindAssets("MatMulCS")[0];
                string cspath = AssetDatabase.GUIDToAssetPath(csguid);
                this.MatMulCS = AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
            }
            else 
                MatMulCS = null;


            this.t_W = Tensor.Zeros(outputs, inputs);
            this.t_B = Tensor.Zeros(outputs);

            this.g_W = Tensor.Zeros(outputs, inputs);
            this.g_B = Tensor.Zeros(outputs);

            this.m_W = Tensor.Zeros(outputs, inputs);
            this.m_B = Tensor.Zeros(outputs);

            this.v_W = Tensor.Zeros(outputs, inputs);
            this.v_B = Tensor.Zeros(outputs);

            switch (init)
            {
                case WeightInit.HE:
                    float sigmaHE = MathF.Sqrt(2f / t_W.Shape[1]); //fanIn
                    t_W.ForEach(x => Utils.Random.Gaussian(0f, sigmaHE, out _));
                    break;
                case WeightInit.Xavier:
                    float sigmaXA = MathF.Sqrt(2f / (t_W.Shape[0] + t_W.Shape[1])); // fanIn + fanOut
                    t_W.ForEach(x => Utils.Random.Gaussian(0f, sigmaXA, out _));
                    break;
                case WeightInit.Normal:
                    t_W.ForEach(x => Utils.Random.Gaussian(0f, 1f, out _));
                    break;
                case WeightInit.Random:
                    t_W.ForEach(x => Utils.Random.Value * 2f - 1f);
                    break;
                case WeightInit.Ones:
                    t_W.ForEach(x => 1f);
                    break;
                default:
                    throw new Exception("Unhandled initialization type!");
            }
        }

        public Tensor Forward(Tensor input) 
        {
            // it seems like forward is always faster with CPU rather than GPU for matrices < 1024 size. Maybe on large scales it must be changed again on GPU.
            InputCache = Tensor.Identity(input);
            return Tensor.MatMul(t_W, input, null) + Tensor.ExpandVec(t_B, input.Shape[1]);
        }
        public Tensor Backward(Tensor loss)
        {
            var transposedInput = Tensor.TransposeMat(InputCache);
            Tensor gradW = Tensor.MatMul(loss, transposedInput, MatMulCS);
            Tensor gradB = Tensor.MatMul(loss, Tensor.Ones(transposedInput.Shape), MatMulCS);

            // Average the gradients
            float batch = loss.Shape[1];

            // Update the gradients
            g_W += gradW / batch;
            g_B += gradB / batch;

            // Backpropagate the loss
            // Tensor dLossdActivation = Tensor.MatMul(Tensor.MatTranspose(t_W), loss, MatMulCS);
            // return dLossdActivation;

            // A bit faster back with double tranposition on loss (may work better on large dense)
            Tensor dLossActivation = Tensor.MatMul(Tensor.TransposeMat(loss), t_W, MatMulCS);
            return Tensor.TransposeMat(dLossActivation);
        }

        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization, and one on them is called when weights.shape.length == 0.
            if (t_W.Shape.Length == 0)
                return;

            int outputs = t_W.Shape[0];
            int inputs = t_W.Shape[1];

            this.g_W = Tensor.Zeros(outputs, inputs);
            this.g_B = Tensor.Zeros(outputs);

            this.m_W = Tensor.Zeros(outputs, inputs);
            this.m_B = Tensor.Zeros(outputs);

            this.v_W = Tensor.Zeros(outputs, inputs);
            this.v_B = Tensor.Zeros(outputs);
        }
    }

}