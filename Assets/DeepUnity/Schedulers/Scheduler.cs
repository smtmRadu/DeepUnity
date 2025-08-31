using System;
using UnityEngine;

namespace DeepUnity.Optimizers
{
    public partial class Optimizer
    {
        [SerializeField] private Lazy<Scheduler> _scheduler;

        /// <summary>
        /// The learning rate scheduler of <see cref="this"/> optimizer. You can also keep the scheduler object separate, but 
        /// using the optimizer as a wrapper for the scheduler brings cleaner code.
        /// </summary>
        public Scheduler Scheduler { get => _scheduler.Value; set { _scheduler = new Lazy<Scheduler>(() => value); } }
    }

    /// <summary>
    /// Base class for all learning rate schedulers.
    /// </summary>
    [System.Serializable]
    public abstract class Scheduler
    {
        protected readonly Optimizer optimizer;
        protected readonly float initialLR;
        protected readonly int lastEpoch;
        protected int currentStep;
        public Scheduler(Optimizer optimizer, int last_epoch = -1)
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


