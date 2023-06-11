using UnityEngine;

namespace DeepUnity
{
    public interface IParameters : ISerializationCallbackReceiver
    {
        public void ZeroGrad();
        public void ClipGradValue(float clip_value);
        public void ClipGradNorm(float max_norm);
    }
}

