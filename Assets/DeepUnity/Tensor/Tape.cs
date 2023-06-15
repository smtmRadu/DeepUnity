using System.Collections.Generic;

namespace DeepUnity
{
    public sealed class Tape
    {
        List<Operation> operations;
        public Tape() => operations = new List<Operation>();
        public void Add(Operation operation) => operations.Add(operation);
        public Tensor Backward()
        {
            return null;
        }
    }
}
