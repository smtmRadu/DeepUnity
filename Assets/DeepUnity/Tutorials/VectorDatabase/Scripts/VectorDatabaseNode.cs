using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class VectorDatabaseNode
    {
        string document;
        Tensor embedding;
        public enum EmbeddingComparison
        {
            cosine,
            dot
        }

        public string Document { get => document; }
        public Tensor Embedding { get => embedding; }


        public Tensor Similarity(Tensor input, EmbeddingComparison mode = EmbeddingComparison.cosine)
        {
            if (mode == EmbeddingComparison.cosine)
                return Tensor.CosineSimilarity(input, embedding);
            else if (mode == EmbeddingComparison.dot)
                return Tensor.Dot(input, embedding);
            else
                throw new System.NotImplementedException();
        }


        public VectorDatabaseNode(string document, Tensor embedding)
        {
            this.document = document;
            this.embedding = embedding;
        }
    }
}
