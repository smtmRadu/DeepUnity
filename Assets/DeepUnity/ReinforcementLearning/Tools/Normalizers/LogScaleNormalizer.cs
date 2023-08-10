using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class LogScaleNormalizer : INormalizer
    {
        [SerializeField] private readonly float CLIP_MIN;
        [SerializeField] private readonly float CLIP_MAX;
        public LogScaleNormalizer() { }
        public Tensor Normalize(Tensor input, bool update = true) => input.Log();
        public void Update(Tensor input) { }

    }
}

