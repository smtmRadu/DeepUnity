namespace DeepUnity
{
    public interface Operation
    {
        public NDArray Backward();
    }

    public sealed class AdditionOperation : Operation
    {
        private NDArray left;
        private NDArray right;
        private NDArray result;

        public AdditionOperation(NDArray left, NDArray right, NDArray result)
        {
            this.left = left;
            this.right = right;
            this.result = result;
        }

        public NDArray Backward()
        {
            return null;
        }
    }
    public sealed class MatMulOperation : Operation
    {
        private NDArray left;
        private NDArray right;
        private NDArray result;

        public MatMulOperation(NDArray left, NDArray right, NDArray result)
        {
            this.left = left;
            this.right = right;
            this.result = result;
        }

        public NDArray Backward()
        {
            return null;
        }
    }
}
