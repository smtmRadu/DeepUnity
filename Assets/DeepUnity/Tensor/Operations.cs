namespace DeepUnity
{
    public interface Operation
    {
        public Tensor Backward();
    }

    public sealed class AdditionOperation : Operation
    {
        private Tensor left;
        private Tensor right;
        private Tensor result;

        public AdditionOperation(Tensor left, Tensor right, Tensor result)
        {
            this.left = left;
            this.right = right;
            this.result = result;
        }

        public Tensor Backward()
        {
            return null;
        }
    }
    public sealed class MatMulOperation : Operation
    {
        private Tensor left;
        private Tensor right;
        private Tensor result;

        public MatMulOperation(Tensor left, Tensor right, Tensor result)
        {
            this.left = left;
            this.right = right;
            this.result = result;
        }

        public Tensor Backward()
        {
            return null;
        }
    }
}
