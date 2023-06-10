namespace DeepUnity
{
    public interface IParameters
    {
        public void ZeroGrad();
        public void ClipGradValue(float clip_value);
        public void ClipGradNorm(float max_norm);

    }
}

