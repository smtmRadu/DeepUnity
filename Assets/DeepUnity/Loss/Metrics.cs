using System;

namespace DeepUnity
{
    public static class Metrics
    {
        /// <summary>
        /// <b>Returns the accuracy value in range [0, 1].</b> <br></br>
        /// Predictions: <b>(B, H)</b> or <b>(H)</b> for unbatched input. <br></br>
        /// Targets: <b>(B, H)</b> or <b>(H)</b> for unbatched input. <br></br>
        /// whre B = batch_size and H = output_size
        /// </summary>
        public static float Accuracy(Tensor predictions, Tensor targets)
        {
            if (predictions.Rank > 2 || targets.Rank > 2)
                throw new ArgumentException("Prediction and targets must be of shape (B, H) or (H) for unbatched input.");


            Tensor errors = predictions.Zip(targets, (p, t) => MathF.Abs(p - t)); // [batch x outputs]

            if(predictions.Rank == 2) // batched
                errors = Tensor.Mean(errors, -2);

            errors = Tensor.Sum(errors, -1);

            float acc = 1.0f - errors[0];
            return acc;
        }
    }
}