using System;

namespace DeepUnity
{
    /// <summary>
    /// A tool for computing a loss function for predictions and targets. There are 3 properties: <br></br>
    /// <b>Item</b>: Returns the mean of all loss values in the tensor. <br></br>
    /// <b>Value</b>: Returns the loss <see cref="Tensor"/> applied element-wisely over predictions and targets.<br></br>
    /// <b>Derivative</b>: Returns the derivative of the loss function <see cref="Tensor"/> applied element-wisely over predictions and targets. 
    /// <b>Used for backpropagation.</b>
    /// </summary>
    public class Loss
    {
        private LossType lossType;
        private Tensor predicts;
        private Tensor targets;

        private Loss(LossType type, Tensor predicts, Tensor targets)
        {
            lossType = type;
            this.predicts = predicts;
            this.targets = targets;
        }
        /// <summary>
        /// Mean Square Error loss. <br></br>
        /// Predicts: (B, *) or (*) for unbatched input <br></br>
        /// Targets: (B, *) or (*) for unbatched input <br></br>
        /// where * = input Shape
        /// </summary>
        public static Loss MSE(Tensor predicts, Tensor targets) => new Loss(LossType.MSE, predicts, targets);
        /// <summary>
        /// Mean Absolute Error loss. <br></br>
        /// Predicts: (B, *) or (*) for unbatched input <br></br>
        /// Targets: (B, *) or (*) for unbatched input <br></br>
        /// where * = input Shape
        /// </summary>
        public static Loss MAE(Tensor predicts, Tensor targets) => new Loss(LossType.MAE, predicts, targets);
        /// <summary>
        /// Cross Entropy loss. <br></br>
        /// Predicts: (B, *) or (*) for unbatched input <br></br>
        /// Targets: (B, *) or (*) for unbatched input <br></br>
        /// where * = input Shape
        /// </summary>
        public static Loss CE(Tensor predicts, Tensor targets) => new Loss(LossType.CE, predicts, targets);
        /// <summary>
        /// Hinge Hmbedded loss. <br></br>
        /// Predicts: (B, *) or (*) for unbatched input <br></br>
        /// Targets: (B, *) or (*) for unbatched input <br></br>
        /// where * = input Shape
        /// </summary>
        public static Loss HE(Tensor predicts, Tensor targets) => new Loss(LossType.HE, predicts, targets);
        /// <summary>
        /// Binary Cross Entropy loss. <br></br>
        /// Predicts: (B, *) or (*) for unbatched input <br></br>
        /// Targets: (B, *) or (*) for unbatched input <br></br>
        /// where * = input Shape
        /// </summary>
        public static Loss BCE(Tensor predicts, Tensor targets) => new Loss(LossType.BCE, predicts, targets);
        /// <summary>
        /// Hullback-Liebler Divergence loss. <br></br>
        /// Predicts: (B, *) or (*) for unbatched input <br></br>
        /// Targets: (B, *) or (*) for unbatched input <br></br>
        /// where * = input Shape
        /// </summary>
        public static Loss KLD(Tensor predicts, Tensor targets) => new Loss(LossType.KLD, predicts, targets);

        /// <summary>
        /// Returns the mean loss value (positive number).
        /// </summary>
        public float Item { get
            {
                Tensor lossItem = Value;
                if (lossItem.Rank == 2)// case is batched
                    return lossItem.Mean(0).Mean(0).Abs()[0];
                else
                    return lossItem.Mean(0).Abs()[0];

            }
        }
        /// <summary>
        /// Returns the computed loss function.
        /// </summary>
        public Tensor Value { get
            {
                switch (lossType)
                {
                    case LossType.MSE:
                        return Tensor.Pow(predicts - targets, 2);
                    case LossType.MAE:
                        return Tensor.Abs(predicts - targets);

                    case LossType.CE:
                        return -targets * Tensor.Log(predicts);
                    case LossType.BCE:
                        return - (targets * Tensor.Log(predicts + Utils.EPSILON) + (-targets + 1f) * Tensor.Log(-predicts + 1f + Utils.EPSILON));

                    case LossType.HE:
                        return predicts.Zip(targets, (p, t) => MathF.Max(0f, 1f - p * t));
                    case LossType.KLD:
                        return targets * Tensor.Log(targets / (predicts + Utils.EPSILON));
                    default:
                        throw new NotImplementedException("Unhandled loss type.");
                }
            }
        }
        /// <summary>
        /// Returns the loss partial derivative with respect to the predicts (y).
        /// </summary>
        public Tensor Derivative { get
            {
                switch (lossType)
                {
                    case LossType.MSE:
                        return 2f * (predicts - targets);
                    case LossType.MAE:
                        return predicts.Zip(targets, (p, t) => p - t > 0 ? 1f : -1f);

                    case LossType.CE:
                        return -targets / predicts;
                    case LossType.BCE:
                        return (predicts - targets) / (predicts * (-predicts + 1f) + Utils.EPSILON);

                    case LossType.HE:
                        return predicts.Zip(targets, (p, t) => 1f - p * t > 0f ? -t : 0f);
                    case LossType.KLD:
                        return -targets / (predicts + Utils.EPSILON);
                    default:
                        throw new NotImplementedException("Unhandled loss type.");
                }
            }
        }
        private enum LossType
        {
            MSE,
            MAE,
            CE,
            BCE,
            HE,            
            KLD
        }
    }
}