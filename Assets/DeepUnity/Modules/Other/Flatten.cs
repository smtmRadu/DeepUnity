using System;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// <b>Use negative values when explicit the start and end axis (-1, -2 or -3).</b> <br></br>
    /// Input: <b>(B, *)</b> or <b>(*)</b> for unbatched input.<br></br>
    /// Output: <b>(B, *')</b> or <b>(*')</b> for unbatched input.<br></br>
    /// where B = batch size, * = input shape and *' = output shape.
    /// </summary>
    /// <param name="startAxis"></param>
    /// <param name="endAxis"></param>
    [Serializable]
    public class Flatten : IModule
    {
        private int[] InputShapeCache { get; set; }

        [SerializeField] private int startAxis;
        [SerializeField] private int endAxis;

        /// <summary>
        /// <b>Use negative values when explicit the start and end axis (-1, -2 or -3).</b> <br></br>
        /// Input: <b>(B, *)</b> or <b>(*)</b> for unbatched input.<br></br>
        /// Output: <b>(B, *')</b> or <b>(*')</b> for unbatched input.<br></br>
        /// where B = batch size, * = input shape and *' = output shape.
        /// </summary>
        /// <param name="startAxis"></param>
        /// <param name="endAxis"></param>
        public Flatten(int startAxis, int endAxis)
        {
            if (startAxis >= 0)
                throw new ArgumentException("Use negative axis value for startAxis.");

            if (endAxis >= 0)
                throw new ArgumentException("Use negative axis value for endAxis.");

            if (startAxis >= endAxis)
                throw new ArgumentException("Start axis must be smaller than end axis.");

            this.startAxis = startAxis;
            this.endAxis = endAxis;
        }
        /// <summary>
        /// Input: <b>(B, C, H, W)</b> or <b>(C, H, W)</b> for unbatched input.<br></br>
        /// Output: <b>(B, F)</b> or <b>(F)</b> for unbatched input.<br></br>
        /// where <br>
        /// </br>F = C * H * W, <br></br>
        /// B = batch size, <br></br>
        /// C = input channels, <br></br>
        /// H = input height,
        /// <br></br>W_in = input width.
        /// </summary>
        public Flatten()
        {
            startAxis = -3;
            endAxis = -1;
        }

        public Tensor Predict(Tensor input)
        {
            return Tensor.Flatten(input, startAxis, endAxis);

        }
        public Tensor Forward(Tensor input)
        {
            InputShapeCache = input.Shape;
            return Tensor.Flatten(input, startAxis, endAxis);
        }
        public Tensor Backward(Tensor loss)
        {
            return Tensor.Reshape(loss, InputShapeCache);
        }


        public object Clone() => new Flatten(startAxis, endAxis);
    }

}