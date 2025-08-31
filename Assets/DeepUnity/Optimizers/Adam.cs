using DeepUnity.Modules;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
namespace DeepUnity.Optimizers
{
    // https://arxiv.org/pdf/1412.6980.pdf also adamax algorithm is there, though i extended it with pytorch doc.
    public sealed class Adam : Optimizer
    {
        [SerializeField]  private float beta1;
        [SerializeField] private float beta2;
        [SerializeField] private bool amsgrad;

        [SerializeField] private float beta1_t = 1f; // beta1^t caching
        [SerializeField] private float beta2_t = 1f; // beta2^t caching

        // CPU
        [SerializeField]private Tensor[] m;
        [SerializeField]private Tensor[] v;
        [SerializeField] private Tensor[] vHatMax;

        // GPU
        private readonly TensorGPU[] gpu_m;
        private readonly TensorGPU[] gpu_v;
        private readonly TensorGPU[] gpu_vHatMax;

        [SerializeField] private bool AreThereGPUParams = false;

        /// <summary>
        /// Adam optimizer. This version has no long-term support, so use AdamW for the newest features.
        /// </summary>
        /// <param name="parameters"></param>
        /// <param name="lr"></param>
        /// <param name="beta1"></param>
        /// <param name="beta2"></param>
        /// <param name="eps">Value for numerical stability</param>
        /// <param name="weight_decay">If > 0, it is better to use <see cref="AdamW"/>.</param>
        /// <param name="amsgrad">Use AMSGrad version.</param>
        /// <param name="maximize">If true, gradients are added to the parameters on <see cref="Step()"/>.</param>
        public Adam(Parameter[] parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8F, float weight_decay = 0f, bool amsgrad = false, bool maximize = false) : base(parameters, lr, eps, weight_decay, maximize)
        {
            this.amsgrad = amsgrad;
            this.beta1 = beta1;
            this.beta2 = beta2;

            m = new Tensor[parameters.Length];
            v = new Tensor[parameters.Length];

            if (amsgrad)
                vHatMax = new Tensor[parameters.Length];

            if (parameters.Any(x => x.Device == Device.GPU))
            {
                AreThereGPUParams = true;
                gpu_m = new TensorGPU[parameters.Length];
                gpu_v = new TensorGPU[parameters.Length];

                if (amsgrad)
                    gpu_vHatMax = new TensorGPU[parameters.Length];
            }


            for (int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i].Device == Device.CPU)
                {
                    m[i] = Tensor.Zeros(parameters[i].param.Shape);
                    v[i] = Tensor.Zeros(parameters[i].param.Shape);

                    if (amsgrad)
                        vHatMax[i] = Tensor.Zeros(parameters[i].param.Shape);

                }
                else // GPU
                {
                    gpu_m[i] = TensorGPU.Zeros(parameters[i].paramGPU.Shape);
                    gpu_v[i] = TensorGPU.Zeros(parameters[i].paramGPU.Shape);

                    if (amsgrad)
                        gpu_vHatMax[i] = TensorGPU.Zeros(parameters[i].paramGPU.Shape);
                }
            }   

           
        }

        public override void Step()
        {
            t++;

            beta1_t *= beta1;
            beta2_t *= beta2;

            Parallel.For(0, parameters.Length, i =>
            {
                if (parameters[i].Device == Device.CPU)
                {
                    if (maximize)
                        Tensor.CopyTo(-parameters[i].g, parameters[i].g);

                    if (lambda != 0)
                        Tensor.CopyTo(parameters[i].g + lambda * parameters[i].param, parameters[i].g);

                    Tensor.CopyTo(beta1 * m[i] + (1f - beta1) * parameters[i].g, m[i]);
                    Tensor.CopyTo(beta2 * v[i] + (1f - beta2) * parameters[i].g.Square(), v[i]);

                    Tensor mHat = m[i] / (1f - beta1_t);
                    Tensor vHat = v[i] / (1f - beta2_t);

                    if (amsgrad)
                    {
                        Tensor.CopyTo(Tensor.Maximum(vHatMax[i], vHat), vHatMax[i]);
                        Tensor.CopyTo(parameters[i].param - gamma * mHat / (vHatMax[i].Sqrt() + epsilon), parameters[i].param);
                    }
                    else
                        Tensor.CopyTo(parameters[i].param - gamma * mHat / (vHat.Sqrt() + epsilon), parameters[i].param);
                }
            });
            if (AreThereGPUParams)         
                for (int i = 0; i < parameters.Length; i++)
                    if (parameters[i].Device == Device.GPU)
                    { 
                        if (maximize)
                            TensorGPU.Subtract_(parameters[i].paramGPU, parameters[i].paramGPU, 2);  // double subtraction

                        if (lambda != 0)                     
                            TensorGPU.Add_(parameters[i].gGPU, parameters[i].paramGPU, lambda);

                        TensorGPU.Multiply_(gpu_m[i], beta1);
                        TensorGPU.Add_(gpu_m[i], parameters[i].gGPU, 1f - beta1);


                        TensorGPU gGPUSquared = TensorGPU.Identity(parameters[i].gGPU);
                        TensorGPU.Pow_(gGPUSquared, 2f);

                        TensorGPU.Multiply_(gpu_v[i], beta2);
                        TensorGPU.Add_(gpu_v[i], gGPUSquared, 1f - beta2);
                        gGPUSquared.Dispose();

                        TensorGPU mHat = TensorGPU.Identity(gpu_m[i]);
                        TensorGPU vHat = TensorGPU.Identity(gpu_v[i]);

                        TensorGPU.Divide_(mHat, 1f - beta1_t);
                        TensorGPU.Divide_(vHat, 1f - beta2_t);

                        if (amsgrad)
                        {
                            TensorGPU.Maximum_(gpu_vHatMax[i], vHat);
                            TensorGPU vHatMax = TensorGPU.Identity(gpu_vHatMax[i]);
                            TensorGPU.Sqrt_(vHatMax);
                            TensorGPU.Add_(vHatMax, epsilon);
                            TensorGPU.Divide_(mHat, vHatMax);
                         
                            TensorGPU.Subtract_(parameters[i].paramGPU, mHat, gamma);

                            vHatMax.Dispose();
                        }
                        else
                        {
                            TensorGPU.Sqrt_(vHat);
                            TensorGPU.Add_(vHat, epsilon);
                            TensorGPU.Divide_(mHat, vHat);
                            TensorGPU.Subtract_(parameters[i].paramGPU, mHat, gamma);                            
                        }

                        mHat.Dispose();
                        vHat.Dispose();
                    }
                

               
                
            
        }
    }

}