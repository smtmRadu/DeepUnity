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
        private Softmax EnsureSoftmax()
        {
            if (sm == null)
                sm = new Softmax();

            return sm;
        }
        public Tensor Predict(Tensor input)
        {
            return EnsureSoftmax().Predict(input).Log();
        }
        public Tensor Forward(Tensor input)
        {
            softmaxOutputCache = EnsureSoftmax().Forward(input);
            return softmaxOutputCache.Log();
        }
        public Tensor Backward(Tensor dLdY)
        {
            return EnsureSoftmax().Backward(dLdY / softmaxOutputCache);
        }

        public object Clone() => new LogSoftmax();
    }

}
