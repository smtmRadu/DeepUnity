namespace DeepUnity
{
    public class Parameter
    {
        public Tensor theta;
        public Tensor g;
        public Parameter(Tensor param, Tensor grad)
        {
            this.theta = param;
            this.g = grad;
        }
    }
}



