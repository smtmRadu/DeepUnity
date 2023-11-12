namespace DeepUnity
{
    public class LRScheduler
    {
        private readonly Optimizer optimizer;
        private readonly float initialLR;
        private readonly int stepSize;
        private readonly float decay;
        private readonly int lastEpoch;
        private int currentEpoch;

        /// <summary>
        /// Decays the learning rate of each parameter group by <paramref name="gamma"/> every <paramref name="step_size"/> epochs. 
        /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this schedule (see Adagrad)
        /// When current_epoch = <paramref name="last_epoch"/>, learning rate is reinitialized.
        /// </summary>
        /// <param name="optimizer"></param>
        /// <param name="step_size">Period of learning rate decay.</param>
        /// <param name="gamma">Multiplicative factor of learning rate decay.</param>
        /// <param name="last_epoch">The index of last epoch. </param>
 	    public LRScheduler(Optimizer optimizer, int step_size, float gamma = 0.1f, int last_epoch = -1)
        {
            if (step_size <= 0)
                throw new System.ArgumentException("Step size cannot be equal or less 0");
            if (gamma <= 0f || gamma >= 1f)
                throw new System.ArgumentException("Gamma must be in (0, 1) range.");

            this.optimizer = optimizer;
            initialLR = optimizer.lr;
            stepSize = step_size;
            decay = gamma;
            lastEpoch = last_epoch;

            currentEpoch = 0;
        }

        public void Step()
        {
            currentEpoch++;

            if(currentEpoch % stepSize == 0)
                optimizer.lr *= decay;

            if(currentEpoch == lastEpoch)
                optimizer.lr = initialLR;  
        }
        public float CurrentLR { get => optimizer.lr; }
    }
}

