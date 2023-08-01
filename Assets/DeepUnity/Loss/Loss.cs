using System;

namespace DeepUnity
{
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
        public static Loss MSE(Tensor predicts, Tensor targets) => new Loss(LossType.MSE, predicts, targets);
        public static Loss MAE(Tensor predicts, Tensor targets) => new Loss(LossType.MAE, predicts, targets);
        public static Loss CategoricalCrossEntropy(Tensor predicts, Tensor targets) => new Loss(LossType.CategoricalCrossEntropy, predicts, targets);   
        public static Loss HingeEmbedded(Tensor predicts, Tensor targets) => new Loss(LossType.HingeEmbedded, predicts, targets);
        public static Loss BinaryCrossEntropy(Tensor predicts, Tensor targets) => new Loss(LossType.BinaryCrossEntropy, predicts, targets);
        public static Loss KLDivergence(Tensor predicts, Tensor targets) => new Loss(LossType.KLDivergence, predicts, targets);

        /// <summary>
        /// Returns the loss summed along the last axis. The value is meaned along the batch axis.
        /// </summary>
        public float Value { get
            {
                Tensor lossItem = Item;
                if (lossItem.Rank == 2)// case is batched
                    return lossItem.Mean(-2).Sum(-1)[0];
                else
                    return lossItem.Sum(-1)[0];

            }
        }
        public Tensor Item { get
            {
                switch (lossType)
                {
                    case LossType.MSE:
                        return Tensor.Pow(predicts - targets, 2);
                    case LossType.MAE:
                        return Tensor.Abs(predicts - targets);

                    case LossType.CategoricalCrossEntropy:
                        return -targets * Tensor.Log(predicts);
                    case LossType.BinaryCrossEntropy:
                        return targets * Tensor.Log(predicts + Utils.EPSILON) - (-targets + 1f) * Tensor.Log(-predicts + 1f + Utils.EPSILON);

                    case LossType.HingeEmbedded:
                        return predicts.Zip(targets, (p, t) => MathF.Max(0f, 1f - p * t));
                    case LossType.KLDivergence:
                        return targets * Tensor.Log(targets / (predicts + Utils.EPSILON));
                    default:
                        throw new NotImplementedException("Unhandled loss type.");
                }
            }
        }
        public Tensor Derivative { get
            {
                switch (lossType)
                {
                    case LossType.MSE:
                        return 2f * (predicts - targets);
                    case LossType.MAE:
                        return predicts.Zip(targets, (p, t) => p - t > 0 ? 1f : -1f);

                    case LossType.CategoricalCrossEntropy:
                        return -targets / predicts;
                    case LossType.BinaryCrossEntropy:
                        return (targets - predicts) / (predicts * (predicts - 1f) + Utils.EPSILON);

                    case LossType.HingeEmbedded:
                        return predicts.Zip(targets, (p, t) => 1f - p * t > 0f ? -t : 0f);
                    case LossType.KLDivergence:
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
            CategoricalCrossEntropy,
            BinaryCrossEntropy,
            HingeEmbedded,            
            KLDivergence
        }
    }
}