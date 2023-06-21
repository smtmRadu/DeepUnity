using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class RunningStandardizer
    {
        [SerializeField] private int step;
        [SerializeField] private Tensor mean; 
        [SerializeField] private Tensor variance;


        public RunningStandardizer(int size)
        {
            step = 0;

            mean = Tensor.Zeros(size);
            variance = Tensor.Ones(size);
        }

        private void Update(Tensor tuples)
        {

            int batch_size = tuples.Size(TDim.height);
            float total = step + batch_size;


            // convert tuples to tuple

            Tensor deltaMu = Tensor.Mean(tuples, TDim.height) - mean;
            mean = mean * (step / total) + deltaMu * (batch_size / total);

            Tensor deltaVar = Tensor.Var(tuples, TDim.height) - variance;
            variance = variance * (step / total) + deltaVar * (batch_size / total);

            step = (int)total;
        }

        public Tensor Standardise(Tensor tuples)
        {
            Update(tuples);

            int batch = tuples.Size(TDim.height);
            return (tuples - Tensor.Expand(mean, TDim.height, batch)) / 
                Tensor.Expand(Tensor.Sqrt(variance), TDim.height, batch);
        }
    }
}

