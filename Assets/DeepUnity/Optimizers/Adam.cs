namespace DeepUnity
{
    // https://arxiv.org/pdf/1412.6980.pdf also adamax algorithm is there
    // 
    public sealed class Adam : Optimizer
    {
        private readonly float beta1;
        private readonly float beta2;
        private readonly bool amsgrad;
        private readonly bool maximize;

        private float beta1_t = 1f; // beta1^t caching
        private float beta2_t = 1f;

        // 1st momentum buffer
        private readonly Tensor[] m;

        // 2nd momentum buffer 
        private readonly Tensor[] v;

        //
        private readonly Tensor[] vMax;



        public Adam(Parameter[] parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 0f, bool amsgrad = false, bool maximize = false) : base(parameters, lr, weightDecay)
        {
            this.amsgrad = amsgrad;
            this.maximize = maximize;
            this.beta1 = beta1;
            this.beta2 = beta2;
            
            m = new Tensor[parameters.Length];
            v = new Tensor[parameters.Length];

            if(amsgrad)
                vMax = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                m[i] = Tensor.Zeros(parameters[i].theta.Shape);
                v[i] = Tensor.Zeros(parameters[i].theta.Shape);

                if (amsgrad)
                    vMax[i] = Tensor.Zeros(parameters[i].theta.Shape);
            }           
        }

        public override void Step()
        {
            t++;

            beta1_t *= beta1;
            beta2_t *= beta2;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                //   // Paper algorithm 
                //    // Update biased first momentum estimate
                //    m[i] = beta1 * m[i] + (1f - beta1) * parameters[i].g;
                //   
                //    // Update biased second raw momentum estimate
                //    v[i] = beta2 * v[i] + (1f - beta2) * parameters[i].g.Pow(2f);
                //   
                //    // Compute bias-corrected first momentum estimate
                //    Tensor mHat = m[i] / (1f - beta1_t);
                //   
                //    // Compute bias-corrected second raw momentum estimate
                //    Tensor vHat = v[i] / (1f - beta2_t);
                //   
                //    // Update parameters 
                //    Tensor.CopyTo(parameters[i].theta - lr * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON), parameters[i].theta);

                // Pytorch version

                if (maximize)
                    Tensor.CopyTo(-parameters[i].g, parameters[i].g);
                
                if (lambda != 0)
                    Tensor.CopyTo(parameters[i].g + lambda * parameters[i].theta, parameters[i].g);
                
                m[i] = beta1 * m[i] + (1f - beta1) * parameters[i].g;
                v[i] = beta2 * v[i] + (1f - beta2) * parameters[i].g.Pow(2f);
                Tensor mHat = m[i] / (1f - beta1_t);
                Tensor vHat = v[i] / (1f - beta2_t);
                
                if(amsgrad)
                {
                    vMax[i] = Tensor.Maximum(vMax[i], vHat);
                    Tensor.CopyTo(parameters[i].theta - gamma * mHat / (vMax[i].Sqrt() + Utils.EPSILON), parameters[i].theta);
                }
                else
                    Tensor.CopyTo(parameters[i].theta - gamma * mHat / (vHat.Sqrt() + Utils.EPSILON), parameters[i].theta);
                

            });

        }
    }

}