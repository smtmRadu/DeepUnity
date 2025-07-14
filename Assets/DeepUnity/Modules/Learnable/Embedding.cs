using System;
using System.Threading.Tasks;
using Unity.VisualScripting;
using Unity.VisualScripting.Antlr3.Runtime;
using UnityEngine;


namespace DeepUnity.Modules
{
    /// <summary>
    /// Input: <b>(L)</b> or <b>(B, L)</b> for batched input.<br></br>
    /// Output: <b>(L, E)</b> or <b>(B, L, E)</b> for batched input.<br></br>
    /// where B = batch_size, L = sequence_length, E = embedding_dim.
    /// </summary>
    public class Embedding : ILearnable, IModule
    {
        public Device Device { get; set; } = Device.CPU;
        public bool RequiresGrad { get; set; } = true;
        private Tensor InputCache { get; set; }

        [SerializeField] public Tensor embeddings;
        [SerializeField] public Tensor embeddingsGrad;
        /// <summary>
        /// Input: <b>(L)</b> or <b>(B, L)</b> for batched input.<br></br>
        /// Output: <b>(L, E)</b> or <b>(B, L, E)</b> for batched input.<br></br>
        /// where B = batch_size, L = sequence_length, E = embedding_dim.
        /// </summary>
        /// <param name="num_embeddings">Size of the dictionary of embeddings.</param>
        /// <param name="embedding_dim">The size of each embedding vector.</param>
        /// <param name="max_norm">If given, each embedding vector with norm larger than <paramref name="max_norm"/> is renormalized to have norm <paramref name="max_norm"/>.</param>
        /// <param name="init">Initializer used for embeddings</param>
        /// <param name="normType">The p of the p-norm to compute for the max_norm option. Default L2.</param>
        /// <param name="normEps">Value for numerical stability when computing the norm (only if <paramref name="max_norm"/>!=null.</param>
        public Embedding(int num_embeddings, int embedding_dim, InitType init = InitType.Normal, float? max_norm = null, NormType normType = NormType.EuclideanL2, float normEps = 1e-12F)
        {
            if (num_embeddings < 2)
                throw new ArgumentException($"Num_embeddings cannot be less than two (recived {num_embeddings})");

            if (embedding_dim < 1)
                throw new ArgumentException($"Embedding_dim cannot be less than one (received {embedding_dim})");


            embeddings = Parameter.Create(new int[] { num_embeddings, embedding_dim }, embedding_dim, num_embeddings, initializer: init);
            embeddingsGrad = Tensor.Zeros(embeddings.Shape);

            // normalize each row to have norm = max_norm
            if(max_norm != null)
            {
                Parallel.For(0, num_embeddings, n =>
                {
                    var embedding = Tensor.Zeros(embedding_dim);
                    for(int e = 0; e < embedding_dim; e++)
                    {
                        embedding[e] = embeddings[n, e];
                    }
                    float norm = embedding.Norm(norm:normType, eps:normEps)[0];
                    for (int e = 0; e < embedding_dim; e++)
                    {
                        embeddings[n, e] = embeddings[n, e] / norm;
                    }
                });
            }
        }
        private Embedding() { }

        public Tensor Predict(Tensor input)
        {
            if(input.Rank > 3 || input.Rank < 1)
                throw new ArgumentException($"Input must have the shape as (L) or (B, L), and the received input is ({input.Shape.ToCommaSeparatedString()})");

            // removed because we need efficiency boys
            //if (!input.Equals(input.Int()))
            //    throw new ArgumentException($"Input must contain only integer values");

            if (input.Rank == 1)
            {
                // L, E
                Tensor y = Tensor.Zeros(input.Size(-1), this.embeddings.Size(-1));


                Parallel.For(0, this.embeddings.Size(-1), e =>
                {
                    for (var l = 0; l < input.Size(-1); l++)
                    {
                        y[l, e] = this.embeddings[(int)input[l], e];
                    }
                });
                return y;
            }
            else if (input.Rank == 2)
            {
                // B, L, E
                Tensor y = Tensor.Zeros(input.Size(-2), input.Size(-1), this.embeddings.Size(-1));


                Parallel.For(0, this.embeddings.Size(-1), e =>
                {
                    for (var b = 0; b < input.Size(-2); b++)
                    {
                        for (var l = 0; l < input.Size(-1); l++)
                        {
                            y[b, l, e] = this.embeddings[(int)input[l], e];
                        }
                    }
                });
                return y;
            }
            else
                throw new Exception("Unhandled for input.Rank != [1, 2] (not possible)");     
        }

        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);

            return Predict(input);
        }

        public Tensor Backward(Tensor loss)
        {
            bool isBatched = loss.Rank == 3;
            int m = isBatched ? loss.Size(-3) : 1;

            if (isBatched)
            {
                Parallel.For(0, embeddings.Size(-1), e =>
                {
                    for (int b = 0; b < m; b++)
                    {
                        for (int l = 0; l <= loss.Size(-1); l++)
                        {
                            embeddingsGrad[(int)InputCache[l], e] += loss[b, l, e] / m; // mean across the batch
                        }
                    }
                });
            }
            else
            {
                Parallel.For(0, embeddings.Size(-1), e =>
                {

                    for (int l = 0; l <= loss.Size(-1); l++)
                    {
                        embeddingsGrad[(int)l, e] += loss[l, e];
                    }
                    
                });
            }

            // gradients through indices are always None
            return null; // isBatched ? Tensor.Zeros(m, loss.Size(-2)) : Tensor.Zeros(loss.Size(-2));
        }

        public Parameter[] Parameters() => new Parameter[] { new Parameter(embeddings, embeddingsGrad) };
        
        public object Clone()
        {
            var emb = new Embedding();
            emb.Device = Device;
            emb.RequiresGrad = RequiresGrad;
            emb.embeddings = (Tensor)embeddings.Clone();
            emb.embeddingsGrad = (Tensor)embeddingsGrad.Clone();
            return emb;
        }
        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (embeddings.Shape == null)
                return;

            if (embeddings.Shape.Length == 0)
                return;

            // do not check if gamma is != null...

            embeddingsGrad = Tensor.Zeros(embeddings.Shape);
        }

    }

}
