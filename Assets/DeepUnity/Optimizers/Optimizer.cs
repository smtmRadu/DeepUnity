namespace DeepUnity
{
    // Note that parameters and gradients value should not be reassigned elsewhere, only the values of the tensors inside
    public abstract class Optimizer
    {
        protected Tensor[] parameters;
        protected Tensor[] gradients;
        public float lr;
        protected float lambda;
        protected int t; // step counter

        protected Optimizer(Tensor[] parameters, Tensor[] gradients, float lr, float weightDecay)
        {
            this.parameters = parameters;
            this.gradients = gradients;
            this.lr = lr;
            lambda = weightDecay;
            t = 0;       
        }
        public abstract void Step();

        /// <summary>
        /// Resets all gradients of a <see cref="Learnable"/> layer to 0.
        /// </summary>
        public void ZeroGrad()
        {
            foreach (var grad in gradients)
            {
                grad.AssignAs(grad.Select(x => 0));
            }
        }
        /// <summary>
        /// Clips the gradients of a <see cref="Learnable"/> layer in the range [-clip_value, clip_value]
        /// </summary>
        public void ClipGradValue(float clip_value)
        {
            foreach (var param in gradients)
            {
                param.Clip(-clip_value, clip_value);
            }
        }
        /// <summary>
        /// Computes the clip grad norm globally over all <see cref="Learnable"/> layers. If <paramref name="max_norm"/> = 0, no changes are made.
        /// </summary>
        /// <param name="max_norm">If is 0, nothing is changed</param>
        public void ClipGradNorm(float max_norm, NormType normType = NormType.EuclideanL2)
        {
            if (max_norm == 0)
                return;

            int totalCount = 0;
            foreach (var param in parameters)
            {
                totalCount += param.Count();
            }

            // Concatenate all gradients in a single tensor vector
            Tensor vector = Tensor.Zeros(totalCount);
            int index = 0;
            foreach (var param in gradients)
            {
                float[] gradTheta = param.ToArray();

                for (int i = 0; i < gradTheta.Length; i++)
                {
                    vector[index++] = gradTheta[i];
                }
            }

            // Compute norm
            Tensor norm = Tensor.Norm(vector, normType);

            if (norm[0] <= max_norm)
                return;

            float scale = max_norm / norm[0];

            foreach (var item in gradients)
            {
                item.AssignAs(item * scale);

            }     
        }
        
    }
}
