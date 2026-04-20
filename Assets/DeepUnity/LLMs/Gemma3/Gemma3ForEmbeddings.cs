using System;

namespace DeepUnity
{
    [Obsolete("Gemma3ForEmbeddings is deprecated. The CPU-based embedding model has been removed.")]
    public class Gemma3ForEmbeddings
    {
        public Gemma3ForEmbeddings()
        {
            throw new NotSupportedException("Gemma3ForEmbeddings is deprecated. The CPU-based embedding model has been removed.");
        }

        public System.Collections.IEnumerator EncodeQuery(string prompt, Action<Tensor> onEmbeddingReceived)
        {
            throw new NotSupportedException("Gemma3ForEmbeddings is deprecated.");
        }
    }
}
