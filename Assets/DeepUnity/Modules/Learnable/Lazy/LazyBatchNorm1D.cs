using System;
using UnityEngine;

namespace DeepUnity.Modules
{
    [Serializable]
    public class LazyBatchNorm1D : ILearnable, IModule
    {
        [SerializeField] public Device Device { get => BatchNorm1D.Device; set { if (BatchNorm1D != null) BatchNorm1D.Device = value; } }
        [SerializeField] public bool RequiresGrad { get => BatchNorm1D.RequiresGrad; set { if (BatchNorm1D != null) BatchNorm1D.RequiresGrad = value; } }
        [SerializeField] BatchNorm1D BatchNorm1D;
        [SerializeField] private bool initialized = false;
        private float epsilon { get; set; }
        private float momentum { get; set; }
        private bool affine { get; set; }

        public LazyBatchNorm1D(float eps = 1e-5f, float momentum = 0.9f, bool affine = true)
        {
            this.momentum = momentum;
            this.epsilon = eps;
            this.affine = affine;
            this.initialized = false;

        }

        public Tensor Predict(Tensor input)
        {
            if (!initialized)
            {
                BatchNorm1D = new BatchNorm1D(input.Size(-1), epsilon, momentum, affine);
                initialized = true;
            }
               

            return BatchNorm1D.Predict(input);
        }

        public Tensor Forward(Tensor input)
        {
            if (!initialized)
            {
                BatchNorm1D = new BatchNorm1D(input.Size(-1), epsilon, momentum, affine);
                initialized = true;
            }

            return BatchNorm1D.Forward(input);
        }
        public Tensor Backward(Tensor input)
        {
            return BatchNorm1D.Backward(input);
        }

        public Parameter[] Parameters()
        {
            if (BatchNorm1D == null)
                throw new Exception("LazyModule was not initialized through inference");

            return BatchNorm1D.Parameters();
        }

        /// <summary>
        /// The cloned version is the base one (not lazy).
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            return BatchNorm1D.Clone();
        }

        public void OnBeforeSerialize()
        {
            BatchNorm1D.OnBeforeSerialize();
        }
        public void OnAfterDeserialize()
        {
            BatchNorm1D.OnAfterDeserialize();

        }

    }

}