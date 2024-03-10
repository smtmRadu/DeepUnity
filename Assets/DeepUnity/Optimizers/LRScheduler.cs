namespace DeepUnity.Optimizers
{
    /// <summary>
    /// Base class for all learning rate schedulers.
    /// </summary>
    public abstract class LRScheduler
    {
        protected readonly Optimizer optimizer;
        protected readonly float initialLR;
        protected readonly int lastEpoch;
        protected int currentStep;
        public LRScheduler(Optimizer optimizer, int last_epoch = -1)
        {
            this.optimizer = optimizer;
            this.initialLR = optimizer.gamma;
            this.lastEpoch = last_epoch;
            this.currentStep = 0;
        }
        public abstract void Step();
        public float CurrentLR { get => optimizer.gamma; }
    }

}


