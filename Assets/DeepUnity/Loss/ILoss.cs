using System;
using System.Runtime.Remoting.Messaging;
using UnityEngine;

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
        public static Loss CrossEntropy(Tensor predicts, Tensor targets) => new Loss(LossType.CrossEntropy, predicts, targets);   
        public static Loss HingeEmbedded(Tensor predicts, Tensor targets) => new Loss(LossType.HingeEmbedded, predicts, targets);
        public static Loss BinaryCrossEntropy(Tensor predicts, Tensor targets) => new Loss(LossType.BinaryCrossEntropy, predicts, targets);


        public Tensor Item { get
            {
                switch (lossType)
                {
                    case LossType.MSE:
                        return predicts.Zip(targets, (p, t) => (p - t) * (p - t));
                    case LossType.MAE:
                        return predicts.Zip(targets, (p, t) => Math.Abs(p - t));
                    case LossType.CrossEntropy:
                        return predicts.Zip(targets, (p, t) =>
                        {
                            float error = -t * MathF.Log(p);
                            return float.IsNaN(error) ? 0f : error;
                        });
                    case LossType.HingeEmbedded:
                        return predicts.Zip(targets, (p, t) => MathF.Max(0f, 1f - p * t));
                    case LossType.BinaryCrossEntropy:
                        return predicts.Zip(targets, (p, t) =>
                        {
                            float error = -t * MathF.Log(p + Utils.EPSILON) - (1f - t) * MathF.Log(1f - p + Utils.EPSILON);
                            return float.IsNaN(error) ? 0f : error;
                        });
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
                        return predicts.Zip(targets, (p, t) => 2f * (p - t));
                    case LossType.MAE:
                        return predicts.Zip(targets, (p, t) => p - t > 0 ? 1f : -1f);
                    case LossType.CrossEntropy:
                        return predicts.Zip(targets, (p, t) => (t - p) / (p * (p - 1f) + Utils.EPSILON));
                    case LossType.HingeEmbedded:
                        return predicts.Zip(targets, (p, t) => 1f - p * t > 0f ? -t : 0f);
                    case LossType.BinaryCrossEntropy:
                        return predicts.Zip(targets, (p, t) => (p - t) / (p * (1f - p) + Utils.EPSILON));
                    default:
                        throw new NotImplementedException("Unhandled loss type.");
                }
            }
        }
        private enum LossType
        {
            MSE,
            MAE,
            CrossEntropy,
            HingeEmbedded,
            BinaryCrossEntropy
        }
    }
}