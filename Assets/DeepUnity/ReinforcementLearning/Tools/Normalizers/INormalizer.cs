using System;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// https://developers.google.com/machine-learning/data-prep/transform/normalization
    /// </summary>
    public interface INormalizer
    {
        // public static Normalizer Create(int size, NormalizationType type)
        // {
        //     switch (type)
        //     {
        //         case NormalizationType.None:
        //             return null;
        //         case NormalizationType.Scale:
        //             return new ScaleNormalizer(size, 0f, 1f);
        //         case NormalizationType.ZScore:
        //             return new ZScoreNormalizer(size, true);
        //         case NormalizationType.LogScale:
        //             return new LogScaleNormalizer();
        //         default:
        //             throw new System.NotImplementedException("Unhandled normalization type");
        // 
        //     }
        // 
        // }

        public Tensor Normalize(Tensor input, bool update = true);
        public void Update(Tensor input);
    }
}



