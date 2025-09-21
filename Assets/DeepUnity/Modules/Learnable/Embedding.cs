using System;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Assertions;

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

        [SerializeField] private int pad_index = int.MinValue; // hardcoded value so we know was not inited.
        [SerializeField] private int vocab_size;
        [SerializeField] private int hidden_dim;
        [SerializeField] public Tensor embeddings;
        [SerializeField] public Tensor embeddingsGrad;
        /// <summary>
        /// Input: <b>(L)</b> or <b>(B, L)</b> for batched input.<br></br>
        /// Output: <b>(L, E)</b> or <b>(B, L, E)</b> for batched input.<br></br>
        /// where B = batch_size, L = sequence_length, E = embedding_dim.
        /// <br></br><br></br>
        /// <i>On initialization, it is recommended to scale down the weights by 0.02 as if initalization was made from N(0, 0.02) <br></br>
        /// E.g. <b>emb_layer.embeddings *= 0.02F;</b>
        /// </i>
        /// </summary>
        /// <param name="num_embeddings">Size of the dictionary of embeddings.</param>
        /// <param name="embedding_dim">The size of each embedding vector.</param>
        /// <param name="pad_idx">If specified, the entries at padding_idx do not contribute to the gradient; therefore, the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”. For a newly constructed Embedding, the embedding vector at padding_idx will default to all zeros, but can be updated to another value to be used as the padding vector.</param>
        /// <param name="max_norm">If given, each embedding vector with norm larger than <paramref name="max_norm"/> is renormalized to have norm <paramref name="max_norm"/>.</param>
        /// <param name="init">Initializer used for embeddings</param>
        /// <param name="normType">The p of the p-norm to compute for the max_norm option. Default L2.</param>
        /// <param name="norm_eps">Value for numerical stability when computing the norm (only if <paramref name="max_norm"/>!=null.</param>
        public Embedding(int num_embeddings, int embedding_dim, int? pad_idx = null, InitType init = InitType.Normal, float? max_norm = null, NormType normType = NormType.EuclideanL2, float norm_eps = 1e-12F)
        {
            if (num_embeddings < 2)
                throw new ArgumentException($"Num_embeddings cannot be less than two (recived {num_embeddings})");

            if (embedding_dim < 1)
                throw new ArgumentException($"Embedding_dim cannot be less than one (received {embedding_dim})");

            this.vocab_size = num_embeddings;
            this.hidden_dim = embedding_dim;

            embeddings = Parameter.Create(new int[] { num_embeddings, embedding_dim }, embedding_dim, num_embeddings, initializer: init);
            embeddingsGrad = null;// Tensor.Zeros(embeddings.Shape);

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
                    float norm = embedding.Norm(norm:normType, eps:norm_eps)[0];
                    for (int e = 0; e < embedding_dim; e++)
                    {
                        embeddings[n, e] = embeddings[n, e] / norm;
                    }
                });
            }

            if(pad_idx is not null)
            {
                if (pad_idx.Value < 0 || pad_idx >= num_embeddings)
                {
                    throw new ArgumentException($"The received pad index value ({pad_idx}) is not a valid index for a vocab of {num_embeddings}");
                }
                this.pad_index = pad_idx.Value;

                for (int e = 0; e < embedding_dim; e++)
                {
                    embeddings[this.pad_index, e] = 0f;
                }
            }
        }
        private Embedding() { }
        public Tensor Predict(Tensor input)
        {
            if(input.Rank > 3)
                throw new ArgumentException($"Input must have the shape as (L) or (B, L), and the received input is ({input.Shape.ToCommaSeparatedString()})");

            // removed because we need efficiency boys
            //if (!input.Equals(input.Int()))
            //    throw new ArgumentException($"Input must contain only integer values");

            if(input.Rank == 0)
            {
                Tensor y = Tensor.Zeros(1, hidden_dim);
                for (int e = 0; e < hidden_dim; e++)
                {
                    y[e] = this.embeddings[(int)input[0], e];
                }
                return y;
            }
            else if (input.Rank == 1)
            {
                // L, E
                int seq_len = input.Size(-1);
                
                Tensor y = Tensor.Zeros(seq_len, hidden_dim);


                Parallel.For(0, hidden_dim, e =>
                {
                    for (var l = 0; l < seq_len; l++)
                    {
                        y[l, e] = this.embeddings[(int)input[l], e];
                    }
                });
                return y;
            }
            else if (input.Rank == 2)
            {
                // B, L, E
                int batch_size = input.Size(-2);
                int seq_len = input.Size(-1);
                Tensor y = Tensor.Zeros(batch_size, seq_len, hidden_dim);

                
                Parallel.For(0, hidden_dim, e =>
                {
                    for (var b = 0; b < batch_size; b++)
                    {
                        for (var l = 0; l < seq_len; l++)
                        {
                            y[b, l, e] = this.embeddings[(int)input[l], e];
                        }
                    }
                });
                return y;
            }
            else
                throw new Exception("Unhandled for input.Rank not in [0, 1, 2] (not possible)");     
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

            Assert.AreEqual(loss.Size(-1), this.hidden_dim);

            if(embeddingsGrad == null)
            {
                embeddingsGrad = Tensor.Zeros(embeddings.Shape);
            }
            if (isBatched)
            {
                Parallel.For(0, this.vocab_size, e =>
                {
                    for (int b = 0; b < m; b++)
                    {
                        for (int l = 0; l <= this.hidden_dim; l++)
                        {
                            embeddingsGrad[(int)InputCache[l], e] += loss[b, l, e] / m; // mean across the batch
                        }
                    }
                });
            }
            else
            {
                Parallel.For(0, this.vocab_size, e =>
                {

                    for (int l = 0; l <= this.hidden_dim; l++)
                    {
                        embeddingsGrad[l, e] += loss[l, e];
                    }
                    
                });
            }

            if (this.pad_index != int.MinValue)
            {
                for (int e = 0; e < this.hidden_dim; e++)
                {
                    embeddingsGrad[this.pad_index, e] = 0f;
                }
            }
            // gradients through indices are always None
            return null; // isBatched ? Tensor.Zeros(m, loss.Size(-2)) : Tensor.Zeros(loss.Size(-2));
        }

        public Parameter[] Parameters()
        {
            if (embeddingsGrad == null)
                embeddingsGrad = Tensor.Zeros(embeddings.Shape);


            return new Parameter[] { new Parameter(embeddings, embeddingsGrad) };
        }
            
        
        public object Clone()
        {
            var emb = new Embedding();
            emb.vocab_size = this.vocab_size;
            emb.pad_index = this.pad_index;
            emb.hidden_dim = this.hidden_dim;
            emb.Device = Device;
            emb.RequiresGrad = RequiresGrad;
            emb.embeddings = (Tensor)embeddings.Clone();
            if(embeddingsGrad != null) 
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
