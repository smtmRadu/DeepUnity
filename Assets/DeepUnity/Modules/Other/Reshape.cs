using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace DeepUnity
{
    public class Reshape : IModule
    {
        [SerializeField] private int[] inputShape;
        [SerializeField] private int[] outputShape;

        /// <summary>
        /// Batch is not included in the shape args.
        /// </summary>
        /// <param name="input_shape"></param>
        /// <param name="output_shape"></param>
        public Reshape(IEnumerable<int> input_shape, IEnumerable<int> output_shape)
        {
            this.inputShape = input_shape.ToArray();
            this.outputShape = output_shape.ToArray();
        }

        public Tensor Predict(Tensor input)
        {
            bool isBatched = !input.Shape.SequenceEqual(this.inputShape);

            if(isBatched)
            {
                int[] batch_size = new int[] { input.Size(0) };
                return input.Reshape(batch_size.Concat(outputShape).ToArray());
            }
            else
            {
                return input.Reshape(outputShape);
            }
        }
        public Tensor Forward(Tensor input)
        {
            bool isBatched = !input.Shape.SequenceEqual(this.inputShape);

            if (isBatched)
            {
                int[] batch_size = new int[] { input.Size(0) };
                return input.Reshape(batch_size.Concat(outputShape).ToArray());
            }
            else
            {
                return input.Reshape(outputShape);
            }
        }
        public Tensor Backward(Tensor loss)
        {
            bool isBatched = !loss.Shape.SequenceEqual(this.outputShape);

            if (isBatched)
            {
                int[] batch_size = new int[] { loss.Size(0) };
                return loss.Reshape(batch_size.Concat(inputShape).ToArray());
            }
            else
            {
                return loss.Reshape(inputShape);
            }
        }

    }
}

