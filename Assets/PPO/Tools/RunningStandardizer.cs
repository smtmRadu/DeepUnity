using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class RunningStandardizer
    {
        [SerializeField] private string name;

        [SerializeField] private int count = 0;
        [SerializeField] private Tensor mean; 
        [SerializeField] private Tensor std;


        public RunningStandardizer(string name) => this.name = name;

        public void Step(Tensor tuples)
        {
            if(mean == null)
            {
                mean = Tensor.Zeros(tuples.Shape.width);
                std = Tensor.Ones(tuples.Shape.width);
            }

            Tensor[] batches = Tensor.Split(tuples, TDim.height, 1);
            foreach (var tuple in batches)
            {
                count++;
                Tensor deltaMean = tuple - mean;
                mean += deltaMean / count;
                Tensor deltaStd = tuple - mean;
                std = Tensor.Sqrt(deltaMean * deltaStd / count);
            }   
        }

        public Tensor Standardise(Tensor tuples)
        {
            int batch = tuples.Shape.height;
            return (tuples - Tensor.Expand(mean, TDim.height, batch)) / 
                Tensor.Expand(std, TDim.height, batch);
        }
    }
}

