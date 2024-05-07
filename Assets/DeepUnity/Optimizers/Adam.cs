using System.Linq;
using System.Threading.Tasks;
using DeepUnity.Modules;
namespace DeepUnity.Optimizers
{
    // https://arxiv.org/pdf/1412.6980.pdf also adamax algorithm is there, though i extended it with pytorch doc.
    public sealed class Adam : Optimizer
    {
        private readonly float beta1;
        private readonly float beta2;
        private readonly bool amsgrad;
        private readonly bool maximize;

        private float beta1_t = 1f; // beta1^t caching
        private float beta2_t = 1f; // beta2^t caching

        // 1st momentum buffer
        private readonly Tensor[] m;

        // 2nd momentum buffer 
        private readonly Tensor[] v;
        private readonly Tensor[] vHatMax;

        private readonly TensorGPU[] gpu_m;

        // 2nd momentum buffer 
        private readonly TensorGPU[] gpu_v;
        private readonly TensorGPU[] gpu_vHatMax;

        private readonly bool AreThereGPUParams = false;

        public Adam(Parameter[] parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-7f, float weightDecay = 0f, bool amsgrad = false, bool maximize = false) : base(parameters, lr, eps, weightDecay)
        {
            this.amsgrad = amsgrad;
            this.maximize = maximize;
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

                    m[i] = beta1 * m[i] + (1f - beta1) * parameters[i].g;
                    v[i] = beta2 * v[i] + (1f - beta2) * parameters[i].g.Pow(2f);
                    Tensor mHat = m[i] / (1f - beta1_t);
                    Tensor vHat = v[i] / (1f - beta2_t);

                    if (amsgrad)
                    {
                        vHatMax[i] = Tensor.Maximum(vHatMax[i], vHat);
                        Tensor.CopyTo(parameters[i].param - gamma * mHat / (vHatMax[i].Sqrt() + epsilon), parameters[i].param);
                    }
                    else
                        Tensor.CopyTo(parameters[i].param - gamma * mHat / (vHat.Sqrt() + epsilon), parameters[i].param);
                }
            });
            if(AreThereGPUParams)
                for (int i = 0; i < parameters.Length; i++)
                    if (parameters[i].Device == Device.GPU)
                    {
                        if (maximize)
                            TensorGPU.Subtract_(parameters[i].paramGPU, parameters[i].paramGPU, 2);  // double subtraction

                        if (lambda != 0)                     
                            TensorGPU.Add_(parameters[i].gGPU, parameters[i].paramGPU, lambda);
                        

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

                            mHat.Dispose();
                            vHat.Dispose();
                        }

                     
                    }
                

               
                
            
        }
    }

}