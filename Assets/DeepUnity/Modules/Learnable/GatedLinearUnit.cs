using DeepUnity.Activations;
using DeepUnity.Modules;
using System;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{


    [Serializable]
    public class GatedLinearUnit : ILearnable, IModule
    {
        public Device Device { get; set; } = Device.CPU;
        public bool RequiresGrad { get; set; } = true;

        [SerializeField] public string activation = "swish";
        [SerializeField] public Dense up_proj;
        [SerializeField] public Dense gate_proj; // note in the future to merge these two for faster inference or create an inference kernel;
        [SerializeField] public Dense down_proj;
        [NonSerialized] private IActivation _activation = null;

        Tensor UpProjCache { get; set; }
        Tensor GateProjCache { get; set; }

        private void InitActivation()
        {
            switch(this.activation.ToLower())
            {
                case "swish" or "silu":
                    _activation = new SiLU();
                    break;
                case "gelu":
                    _activation = new GELU();
                    break;
                case "relu":
                    _activation = new ReLU();
                    break;
                default:
                    throw new ArgumentException($"Unhandled {activation} activation.DEVNOTE: It cannot handle parametrized activations!!");
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="input_size"></param>
        /// <param name="hidden_size"></param>
        /// <param name="output_size"></param>
        /// <param name="init"></param>
        /// <param name="activation">["swish", "gelu", "relu"]</param>
        /// <param name="device"></param>
        public GatedLinearUnit(int input_size, int hidden_size, int output_size, InitType init = InitType.LeCun_Uniform, string activation = "swish", Device device = Device.CPU)
        {
            up_proj= new Dense(input_size, hidden_size, bias: false, weight_init: init, device:device);
            gate_proj = new Dense(input_size, hidden_size, bias: false, weight_init: init, device: device);
            down_proj = new Dense(hidden_size, output_size, bias: false, weight_init: init, device: device);
            this.activation = activation;
            InitActivation();
        }
        private GatedLinearUnit() { }
        public Tensor Predict(Tensor x)
        {
            var up = this.up_proj.Predict(x);
            var g = this.gate_proj.Predict(x);
            g = this._activation.Predict(g);
            Debug.Log("Intermediate GLU: " + (g * up).ToArray().ToCommaSeparatedString());
            var dp = this.down_proj.Predict(g * up);
            return dp;
        }
        public Tensor Forward(Tensor x)
        {
            UpProjCache = this.up_proj.Forward(x);
            GateProjCache = this._activation.Forward(this.gate_proj.Forward(x));
            return this.down_proj.Forward(UpProjCache * GateProjCache);
        }
        
        public Tensor Backward(Tensor dLdy)
        {
            Tensor dLd_downprojinput = this.down_proj.Backward(dLdy);
            Tensor dLd_upprojoutput = dLd_downprojinput * GateProjCache;
            Tensor dLd_gateprojoutput = dLd_upprojoutput * UpProjCache;

            Tensor dLdx = this.up_proj.Backward(dLd_upprojoutput) + this.gate_proj.Backward(this._activation.Backward(dLd_gateprojoutput));
            return dLdx;
        }

        public Parameter[] Parameters()
        {
            var params_ = new List<Parameter>();

            params_.AddRange(up_proj.Parameters());
            params_.AddRange(gate_proj.Parameters());
            params_.AddRange(down_proj.Parameters());

            return params_.ToArray();

        }
        public object Clone()
        {
            var glu = new GatedLinearUnit();
            glu.Device = Device;
            glu.RequiresGrad = RequiresGrad;
            glu.activation = activation;
            glu._activation = (IActivation)_activation.Clone();
            glu.up_proj = (Dense)up_proj.Clone();
            glu.gate_proj = (Dense)gate_proj.Clone();
            glu.down_proj = (Dense)down_proj.Clone();
            return glu;
        }


        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            up_proj.OnAfterDeserialize();
            gate_proj.OnAfterDeserialize();
            down_proj.OnAfterDeserialize();
        }
    }

}
