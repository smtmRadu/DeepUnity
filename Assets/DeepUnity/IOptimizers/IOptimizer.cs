namespace DeepUnity
{
    public interface IOptimizer
    {
        public void Step(Dense[] layers);
    }
}
