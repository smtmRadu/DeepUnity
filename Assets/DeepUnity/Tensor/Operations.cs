namespace DeepUnity
{
    public interface Operation
    {
        public Tensor Backward();
    }

    public sealed class AdditionOperation : Operation
    {
        public Tensor Backward()
        {
            return null;
        }
    }
}
