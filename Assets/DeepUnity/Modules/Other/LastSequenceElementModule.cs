using System;
using Unity.VisualScripting;

namespace DeepUnity.Modules
{
    /// <summary>
    /// <b>Returns the last element in the sequence for each batch element.</b> <br></br>
    /// Input: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
    /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.
    /// </summary>
    [Serializable]
    public class LastSequenceElementModule : IModule
    {
        private int[] InputShape { get; set; }
        /// <summary>
        /// <b>Returns the last element in the sequence for each batch element.</b> <br></br>
        /// Input: <b>(B, L, H)</b> or <b>(L, H)</b> for unbatched input. <br></br>
        /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.
        /// </summary>
        public LastSequenceElementModule() { }

        public Tensor Predict(Tensor input)
        {
            if (input.Rank == 3)
            {
                int batch_size = input.Size(0);
                int sequence_size = input.Size(1);
                int feature_size = input.Size(2);
                Tensor output = Tensor.Zeros(batch_size, feature_size);


                for (int b = 0; b < batch_size; b++)
                {
                    for (int h = 0; h < feature_size; h++)
                    {
                        output[b, h] = input[b, sequence_size - 1, h];
                    }
                }
                return output;

            }
            else if (input.Rank == 2)
            {
                int sequence_size = input.Size(0);
                int feature_size = input.Size(1);
                Tensor output = Tensor.Zeros(feature_size);
                for (int h = 0; h < feature_size; h++)
                {
                    output[h] = input[sequence_size - 1, h];
                }
                return output;
            }
            else
                throw new ShapeException($"Input must be of shape (B, L, H) or (L, H) (received ({input.Shape.ToCommaSeparatedString()})).");
           
        }


        public Tensor Forward(Tensor input)
        {
            InputShape = input.Shape;
            return Predict(input);  
        }

        public Tensor Backward(Tensor dLdY)
        {
            Tensor grad = Tensor.Zeros(InputShape);

            if (InputShape.Length == 3)
            {
                int batch_size = InputShape[0];
                int seq_length = InputShape[1];
                int feature_size = InputShape[2];

                for (int b = 0; b < batch_size; b++)
                {
                    for (int h = 0; h < feature_size; h++)
                    {
                        grad[b, seq_length-1, h] = dLdY[b, h];
                    }
                }
            }
            else if (InputShape.Length == 2)
            {
                int seq_length = InputShape[0];
                int feature_size = InputShape[1];

                for (int h = 0; h < feature_size; h++)
                {
                    grad[seq_length - 1, h] = dLdY[h];
                }

            }
            else
                throw new Exception("Something went wrong with the implementation i suppose");


            return grad;
        }
        public object Clone() => new LastSequenceElementModule();



    }

}



