using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    public class Reshape : IModule
    {
        [SerializeField] private int[] inputShape;
        [SerializeField] private int[] outputShape;

        /// <summary>
        /// Input: (B, *) or (*) for unbatched input. <br></br>
        /// Output: (B, *') or (*') for unbatched input. <br></br>
        /// where * = input_shape and *' = output_shape.
        /// </summary>
        /// <param name="input_shape"></param>
        /// <param name="output_shape"></param>
        public Reshape(int[] input_shape, int[] output_shape)
        {
            int count = 1;
            foreach (var item in input_shape)
            {
                count *= item;
            }
            int count2 = 1;
            foreach (var item2 in output_shape)
            {
                count2*= item2;
            }
            if (count != count2)
                throw new ShapeException($"Input_shape({input_shape.ToCommaSeparatedString()}) and output_shape({output_shape.ToCommaSeparatedString()}) paramters are not valid for tensor reshaping.");

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

