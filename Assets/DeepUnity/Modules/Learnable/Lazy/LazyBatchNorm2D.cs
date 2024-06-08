using System;
using UnityEngine;

namespace DeepUnity.Modules
{
    [Serializable]
    public class LazyBatchNorm2D : ILearnable, IModule
    {
        [SerializeField] public Device Device { get => BatchNorm2D.Device; set { if (BatchNorm2D != null) BatchNorm2D.Device = value; } }
        [SerializeField] public bool RequiresGrad { get => BatchNorm2D.RequiresGrad; set { if (BatchNorm2D != null) BatchNorm2D.RequiresGrad = value; } }
        [SerializeField] BatchNorm2D BatchNorm2D;
        [SerializeField] private bool initialized = false;
        private float epsilon { get; set; }
        private float momentum { get; set; }
        private bool affine { get; set; }

        public LazyBatchNorm2D(float eps = 1e-5f, float momentum = 0.9f, bool affine = true)
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
                BatchNorm2D = new BatchNorm2D(input.Size(-3), epsilon, momentum, affine);
                initialized = true;
            }
               

            return BatchNorm2D.Predict(input);
        }

        public Tensor Forward(Tensor input)
        {
            if (!initialized)
            {
                BatchNorm2D = new BatchNorm2D(input.Size(-3), epsilon, momentum, affine);
                initialized = true;
            }
            return BatchNorm2D.Forward(input);
        }
        public Tensor Backward(Tensor input)
        {
            return BatchNorm2D.Backward(input);
        }

        public Parameter[] Parameters()
        {
            if (BatchNorm2D == null)
                throw new Exception("LazyModule was not initialized through inference");

            return BatchNorm2D.Parameters();
        }

        /// <summary>
        /// The cloned version is the base one (not lazy).
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            return BatchNorm2D.Clone();
        }

        public void OnBeforeSerialize()
        {
            BatchNorm2D.OnBeforeSerialize();
        }
        public void OnAfterDeserialize()
        {
            BatchNorm2D.OnAfterDeserialize();

        }

    }

}