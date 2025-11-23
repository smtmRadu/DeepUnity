using System;
using DeepUnity.Modules;

namespace DeepUnity.Activations
{
    // https://www.youtube.com/watch?v=09c7bkxpv9I

    /// <summary>
    /// <b>Applies the log(Softmax) function over the last input's dimension H (axis: -1).</b> <br></br>
    /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input <br></br>
    /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input <br></br>
    /// where * = any shape and H = features_num
    /// </summary>
    [Serializable]
    public class LogSoftmax : IModule, IActivation
    {
        /// <summary>
        /// <b>Applies the log(Softmax) function over the last input's dimension H (axis: -1).</b> <br></br>
        /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input <br></br>
        /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input <br></br>
        /// where * = any shape and H = features_num
        /// </summary>
        public LogSoftmax() { }
        private Softmax sm;
        private Tensor softmaxOutputCache;
        public Tensor Predict(Tensor input)
        {
            return sm.Predict(input).Log();
        }
        public Tensor Forward(Tensor input)
        {
            softmaxOutputCache = sm.Forward(input);
            return softmaxOutputCache.Log();
        }
        public Tensor Backward(Tensor dLdY)
        {
            return sm.Backward(1f / softmaxOutputCache);
        }

        public object Clone() => new Softmax();
    }

}
