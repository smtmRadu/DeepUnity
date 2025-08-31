using System;

namespace DeepUnity.Optimizers
{
    /// <summary>
    /// https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR
    /// </summary>
    [System.Serializable]
    public class CosineAnnealingWithWarmup : Scheduler
    {
        private readonly int totalIters;
        private readonly int warmupIters;
        private readonly float minLR;
        /// <summary>
        /// Decays the learning rate in cosine fashion until the number of <see cref="Step"/> calls reaches <paramref name="max_steps"/> argument. After that, any <see cref="Step"/> call does no longer affect the lr. 
        /// Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler.
        /// </summary>
        public CosineAnnealingWithWarmup(Optimizer optimizer, int max_steps, int warmup_steps = 1000, float min_lr = 0F)
            : base(optimizer, -1)
        {
            this.warmupIters = warmup_steps;
            this.totalIters = max_steps;
            this.minLR = min_lr;
            optimizer.gamma = initialLR / warmup_steps;
        }

        public override void Step()
        {
            currentStep++;

            if (currentStep < warmupIters)
            {
                optimizer.gamma = currentStep * initialLR / warmupIters;
            }
            else if (currentStep <= totalIters)
            {
                optimizer.gamma = 
                    minLR + 
                    (initialLR - minLR) / 2F * 
                    (1f + MathF.Cos((float)(currentStep - warmupIters) / (totalIters - warmupIters) * MathF.PI));
            }
        }

    }

}


