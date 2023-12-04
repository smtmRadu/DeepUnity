namespace DeepUnity
{
    public class Parameter
    {
        public Tensor theta;     
        public Tensor g;

        public TensorGPU thetaGPU;
        public TensorGPU gGPU;
        public Parameter(Tensor param, Tensor grad)
        {
            this.theta = param;
            this.g = grad;
        }

        public Parameter(TensorGPU paramGPU, TensorGPU gradGPU)
        {
            this.thetaGPU = paramGPU;
            this.gGPU = gradGPU;

        }
    }
}



