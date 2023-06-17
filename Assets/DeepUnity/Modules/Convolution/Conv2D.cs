namespace DeepUnity
{
    public class Conv2D : IModule
    {
        public Tensor InputCache { get; set; }

        /// <summary>
        /// Input [batch, channels, height, width] => Output [batch, height, width]
        /// </summary>
        public Conv2D(int in_channels, int out_channels, PaddingType paddingType = PaddingType.Mirror)
        {

        }
        public Tensor Predict(Tensor input)
        {
            return null;
        }
        public Tensor Forward(Tensor input)
        {
            return null;
        }
        public Tensor Backward(Tensor loss)
        {
            return null;
        }
    }


}
