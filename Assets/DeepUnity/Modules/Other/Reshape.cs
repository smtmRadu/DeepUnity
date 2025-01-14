using System;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;
namespace DeepUnity.Modules
{
    /// <summary>
    /// Input: <b>(B, *)</b> or <b>(*)</b> for unbatched input. <br></br>
    /// Output: <b>(B, *')</b> or <b>(*')</b> for unbatched input. <br></br>
    /// where * = input_shape and *' = output_shape.
    /// </summary>
    [Serializable]
    public class Reshape : IModule
    {
        [SerializeField] private int[] inputShape;
        [SerializeField] private int[] outputShape;

        /// <summary>
        /// Input: <b>(B, *)</b> or <b>(*)</b> for unbatched input. <br></br>
        /// Output: <b>(B, *')</b> or <b>(*')</b> for unbatched input. <br></br>
        /// where * = input_shape and *' = output_shape.
        /// </summary>
        /// <param name="input_shape">Value of <b>*</b>, where B dimension is not included.</param>
        /// <param name="output_shape">Value of <b>*'</b>, where B dimension is not included.</param>
        public Reshape(int[] input_shape, int[] output_shape)
        {
            if (input_shape == null || input_shape.Length == 0)
                throw new ArgumentException("Input_shape cannot be null or have a length of 0.");

            if (output_shape == null || output_shape.Length == 0)
                throw new ArgumentException("Output_shape cannot be null or have a length of 0.");

            int count = 1;
            foreach (var item in input_shape)
            {
                count *= item;
            }
            int count2 = 1;
            foreach (var item2 in output_shape)
            {
                count2 *= item2;
            }
            if (count != count2)
                throw new ShapeException($"Input_shape({input_shape.ToCommaSeparatedString()}) and output_shape({output_shape.ToCommaSeparatedString()}) paramters are not valid for tensor reshaping.");

            inputShape = input_shape.ToArray();
            outputShape = output_shape.ToArray();
        }

        public Tensor Predict(Tensor input)
        {
            bool isBatched = !input.Shape.SequenceEqual(inputShape);

            if (isBatched)
            {
                int[] batch_size = new int[] { input.Size(0) };
                return Tensor.Reshape(input, batch_size.Concat(outputShape).ToArray());
            }
            else
            {
                return Tensor.Reshape(input, outputShape);
            }
        }
        public Tensor Forward(Tensor input)
        {
            bool isBatched = !input.Shape.SequenceEqual(inputShape);

            if (isBatched)
            {
                int[] batch_size = new int[] { input.Size(0) };
                return Tensor.Reshape(input, batch_size.Concat(outputShape).ToArray());
            }
            else
            {
                return Tensor.Reshape(input, outputShape);
            }
        }
        public Tensor Backward(Tensor loss)
        {
            bool isBatched = !loss.Shape.SequenceEqual(outputShape);

            if (isBatched)
            {
                int[] batch_size = new int[] { loss.Size(0) };
                return Tensor.Reshape(loss, batch_size.Concat(inputShape).ToArray());
            }
            else
            {
                return Tensor.Reshape(loss, inputShape);
            }
        }

        public object Clone() => new Reshape(inputShape, outputShape);

    }
}

