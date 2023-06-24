using DeepUnity;

namespace kbRadu
{
    public class StepLR
    {
        private readonly Optimizer optimizer;
        private readonly float initialLR;
        private readonly int stepSize;
        private readonly float decay;
        private readonly int lastEpoch;
        private int currentEpoch;

        /// <summary>
        /// Decays the learning rate of each parameter group by gamma every step_size epochs. 
        /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this schedule (see Adagrad)
        /// When last_epoch=-1, sets initial lr as lr.
        /// </summary>
        /// <param name="optimizer"></param>
        /// <param name="step_size">Period of learning rate decay.</param>
        /// <param name="gamma">Multiplicative factor of learning rate decay.</param>
        /// <param name="last_epoch">The index of last epoch. </param>
 	    public StepLR(Optimizer optimizer, int step_size, float gamma = 0.1f, int last_epoch = -1)
        {
            this.optimizer = optimizer;
            initialLR = optimizer.learningRate;
            stepSize = step_size;
            decay = gamma;
            lastEpoch = last_epoch;

            currentEpoch = 0;
        }

        public void Step()
        {
            currentEpoch++;

            if(stepSize % currentEpoch == 0)
                optimizer.learningRate *= decay;

            if(currentEpoch == lastEpoch)
                optimizer.learningRate = initialLR;  
        }
        public float CurrentLR() => optimizer.learningRate;
    }
}

