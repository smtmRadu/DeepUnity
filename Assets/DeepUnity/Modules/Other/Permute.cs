using UnityEngine;
using System;
using System.Linq;

namespace DeepUnity.Modules
{
    /// <summary>
    /// Permutes the dimensions of the input according to the specified order. <br></br> <br></br>
    /// Input: <b>(B, *)</b> or <b>(*)</b> <br></br>
    /// Output: <b>(B, *')</b> or <b>(*')</b> <br></br>
    /// 
    /// where <br></br>
    /// * = input shape, *' = permuted input shape, <br></br>
    /// </summary>
    [Serializable]
    public class Permute : IModule
    {
        [SerializeField] private int[] axes;

        /// <summary>
        /// Permutes the dimensions of the input according to the specified order. <br></br> <br></br>
        /// Input: <b>(B, *)</b> or <b>(*)</b> <br></br>
        /// Output: <b>(B, *')</b> or <b>(*')</b> <br></br>
        /// 
        /// where <br></br>
        /// * = input shape, *' = permuted input shape, <br></br>
        /// </summary>
        /// <param name="axes">Use negative values only</param>
        public Permute(params int[] axes)
        {
            for (int i = 0; i < axes.Length; i++)
            {
                if (axes[i] >= 0)
                    throw new Exception("Permute axes must be declared with negative indexes to avoid batched/non-batched input coincidence");
            }
            this.axes = axes.Clone() as int[];
        }

        public Tensor Predict(Tensor input)
        {
            if (input.Shape.Length == axes.Length)
                return input.Permute(axes);
            else // batched
                return input.Permute(new int[] { 0 }.Concat(axes).ToArray());
        }
        public Tensor Forward(Tensor input)
        {
            if (input.Shape.Length == axes.Length)
                return input.Permute(axes);
            else // batched
                return input.Permute(new int[] { 0 }.Concat(axes).ToArray());
        }

        public Tensor Backward(Tensor dLdY)
        {
            if (dLdY.Shape.Length == axes.Length)
                return dLdY.Permute(InverseMapping(axes));
            else // batched
                return dLdY.Permute(new int[] { 0 }.Concat(InverseMapping(axes)).ToArray());
        }
        public object Clone()
        {
            return new Permute(axes);
        }

        private static int[] InverseMapping(int[] mapping)
        {
            int[] inverseMapping = new int[mapping.Length];
            for (int i = 0; i < mapping.Length; i++)
            {
                inverseMapping[mapping[i]] = i;
            }
            return inverseMapping;
        }
    }

}


