namespace DeepUnity.Optimizers
{
    /// <summary>
    /// https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR
    /// </summary>
    [System.Serializable]
    public class LinearAnnealing : Scheduler
    {
        private readonly float endFactor;
        private readonly float startFactor;
        private readonly int totalIters;

        /// <summary>
        /// Decays the learning rate by linearly changing small multiplicative factor until the number of <see cref="Step"/> calls reaches <paramref name="total_iters"/> argument. After that, any <see cref="Step"/> call does no longer affect the lr. 
        /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler.
        /// </summary>
        public LinearAnnealing(Optimizer optimizer, float start_factor = 1f, float end_factor = 1e-8f, int total_iters = 10)
            :base(optimizer, -1)
        {
            this.endFactor = end_factor;
            this.startFactor = start_factor;
            this.totalIters = total_iters;


            optimizer.gamma = initialLR * start_factor;
        }

        public override void Step()
        {
            currentStep++;

            if(currentStep <= totalIters)
            {
                optimizer.gamma = initialLR * (startFactor + (endFactor - startFactor) / totalIters * currentStep);
            }
        }

    }

}


