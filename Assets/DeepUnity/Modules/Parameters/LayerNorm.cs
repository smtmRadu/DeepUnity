using DeepUnity;

namespace kbRadu
{
    public class LayerNorm : IModule, IParameters
    {

        public Tensor Forward(Tensor input)
        {
            return null;
        }
        public Tensor Backward(Tensor loss)
        {
            return null;
        }


        public void ZeroGrad()
        {

        }
        public void ClipGradValue(float clip_value)
        {

        }
        public void ClipGradNorm(float norm_value)
        {

        }
    }
}

