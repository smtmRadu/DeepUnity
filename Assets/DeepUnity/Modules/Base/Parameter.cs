namespace DeepUnity.Modules
{
    /// <summary>
    /// The parameter of a learnable layer. 
    /// It's defined by <b>theta</b> and it's gradient <b>g</b>.
    /// </summary>
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



