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
        /// where B = batch_size and H = output_size (H > 2)
        /// </summary>
        /// <returns>Returns a float value in range [0, 1].</returns>
        public static float Accuracy(Tensor predictions, Tensor targets)
        {
            if (predictions.Rank > 2 || targets.Rank > 2)
                throw new ArgumentException("Prediction and targets must be of shape (B, H) or (H) for unbatched input.");

            if (predictions.Rank == 1 && targets.Rank == 1)
            {
                predictions = predictions.Unsqueeze(0);
                targets = targets.Unsqueeze(0);
            }

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
        /// https://www.v7labs.com/blog/f1-score-guide
        /// Basically is used when we have an uneven balanced of training data from both classes. So for example,
        /// if we have a data set with 90% of data from class-1 and 10% of data from class-2 and manages to detect all class-1 
        /// samples, it doesn't mean has 90% accuracy.
        /// <summary>
        /// <b><em>F1 Score</em> is only used for Binary Classification.</b> <br></br>
        /// <b>F1 Score = (1 + β^2) * Precision * Recall / (β^2 * Precision + Recall)</b><br></br>
        /// <br></br>
        /// Predictions: <b>(B, H)</b> or <b>(H)</b> for unbatched input. <br></br>
        /// Targets: <b>(B, H)</b> or <b>(H)</b> for unbatched input. <br></br>
        /// where B = batch_size and H = output_size (H = 2)
        /// </summary>
        /// <param name="beta">weight</param>
        /// <returns>Returns a float value in range [0, 1].</returns>
        public static float F1Score(Tensor predictions, Tensor targets, AverageType average = AverageType.Weighted, float beta = 1f)
        {  
            if (predictions.Rank > 2 || targets.Rank > 2)
                throw new ArgumentException("Prediction and targets must be of shape (B, H) or (H) for unbatched input.");

            if (predictions.Rank == 1 && targets.Rank == 1)
            {
                predictions = predictions.Unsqueeze(0);
                targets = targets.Unsqueeze(0);
            }

            if (beta <= 0f)
                throw new ArgumentException("Beta must be always greater than 0");

            if (predictions.Size(-1) != 2)
                throw new ArgumentException("F1Score is used only for Binary Classification problems.");
            
            Tensor[] pred = predictions.ArgMax(-1).Split(0, 1);
            Tensor[] targ = targets.ArgMax(-1).Split(0, 1);

            float truePositives = 0f;
            float falsePositives = 0f;
            float trueNegatives = 0f;
            float falseNegatives = 0f;

            for (int i = 0; i < pred.Length; i++)
            {
                if (pred[i].Equals(targ[i]) && targ[i][1] == 0f)
                {
                    truePositives += 1f;
                }
                else if (pred[i].Equals(targ[i]) && targ[i][0] == 0f)
                {
                    trueNegatives += 1f;
                }
                else if (!pred[i].Equals(targ[i]) && targ[i][1] == 0f)
                {
                    falsePositives += 1f;
                }
                else if (!pred[i].Equals(targ[i]) && targ[i][0] == 0f)
                {
                    falseNegatives += 1f;
                }
            }

            switch (average)
            {
                // Weighted F1 Score = (1 + β^2) * Precision * Recall / (β^2 * Precision + Recall)
                case AverageType.Weighted:
                    float precision = truePositives / (truePositives + falsePositives);
                    float recall = truePositives / (truePositives + falseNegatives);
                    if (precision + recall == 0f) return 0f;
                    return (1 + beta * beta) * (precision * recall) / (beta * beta * precision + recall);

                // Micro F1 Score = (TP + TN) / (TP + TN + FP + FN)
                case AverageType.Micro:
                    float microPrecision = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives);
                    float microRecall = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives);
                    if (microPrecision + microRecall == 0f) return 0f;
                    return (1 + beta * beta) * (microPrecision * microRecall) / (beta * beta * microPrecision + microRecall);

                default:
                    throw new ArgumentException("Unhandled AverageType for F1Score");
            }
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