using System;
namespace DeepUnity
{
    /// <summary>
    /// Tools used for classification problems.
    /// </summary>
    public static class Metrics
    {

        /// <summary>
        /// <b>Accuracy = Correct Predictions / (Correct Predictions + Wrong Predictions)</b><br></br>
        /// <br></br>
        /// Predictions: <b>(B, H)</b> or <b>(H)</b> for unbatched input. <br></br>
        /// Targets: <b>(B, H)</b> or <b>(H)</b> for unbatched input. <br></br>
        /// where B = batch_size and H = output_size
        /// </summary>
        /// <returns>Returns a float value in range [0, 1].</returns>
        public static float Accuracy(Tensor predictions, Tensor targets)
        {
            if (predictions.Rank > 2 || targets.Rank > 2)
                throw new ArgumentException("Prediction and targets must be of shape (B, H) or (H) for unbatched input.");


            Tensor[] pred = predictions.ArgMax(-1).Split(0, 1);
            Tensor[] targ = targets.ArgMax(-1).Split(0, 1);

            float guess = 0f;
            float wrong = 0f;

            for (int i = 0; i < pred.Length; i++)
            {
                if (pred[i].Equals(targ[i]))
                    guess += 1f;
                else
                    wrong += 1f;
            }


            return guess / (guess + wrong);
        }
        /// <summary>
        /// <b>F1 Score = 2 * (Precision * Recall) / (Precision + Recall)</b><br></br>
        /// <br></br>
        /// Predictions: <b>(B, H)</b> or <b>(H)</b> for unbatched input. <br></br>
        /// Targets: <b>(B, H)</b> or <b>(H)</b> for unbatched input. <br></br>
        /// where B = batch_size and H = output_size
        /// </summary>
        /// <returns>Returns a float value in range [0, 1].</returns>
        private static float F1Score(Tensor predictions, Tensor targets)
        {
            if (predictions.Rank > 2 || targets.Rank > 2)
                throw new ArgumentException("Prediction and targets must be of shape (B, H) or (H) for unbatched input.");

            Tensor[] pred = predictions.ArgMax(-1).Split(0, 1);
            Tensor[] targ = targets.ArgMax(-1).Split(0, 1);

            float truePositives = 0f;
            float falsePositives = 0f;
            float falseNegatives = 0f;

            for (int i = 0; i < pred.Length; i++)
            {
                if (pred[i].Equals(targ[i]))
                {
                    truePositives += 1f;
                }
                else
                {
                    falsePositives += 1f;
                    falseNegatives += 1f;
                }
            }

            float precision = truePositives / (truePositives + falsePositives);
            float recall = truePositives / (truePositives + falseNegatives);
            return 2 * (precision * recall) / (precision + recall);
        }

        public static float MeanAbsoluteError(Tensor predictions, Tensor targets)
        {
            if (predictions.Rank > 2 || targets.Rank > 2)
                throw new ArgumentException("Prediction and targets must be of shape (B, H) or (H) for unbatched input.");

            Tensor errors = Tensor.Abs(predictions - targets);

            if (predictions.Rank == 2) // batched
                errors = Tensor.Mean(errors, 0);
            errors = Tensor.Mean(errors, 0);

            return errors[0];
        }
        public static float MeanSquaredError(Tensor predictions, Tensor targets)
        {
            if (predictions.Rank > 2 || targets.Rank > 2)
                throw new ArgumentException("Prediction and targets must be of shape (B, H) or (H) for unbatched input.");

            Tensor errors = Tensor.Pow(predictions - targets, 2);

            if (predictions.Rank == 2) // batched
                errors = Tensor.Mean(errors, 0);
            errors = Tensor.Mean(errors, 0);

            return errors[0];
        }
    }
}