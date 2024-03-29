using System;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// A fast way to create a residual connection in a sequetial network.
    /// </summary>
    [Serializable]
    public class SkipConnectionFork : IModule
    {
        public Tensor ConnectionGrad { private get; set; }
        public Tensor Identity { get; private set; }

        /// <summary>
        /// A fast way to create a residual connection in a sequetial network.
        /// </summary>
        public SkipConnectionFork()
        {

        }

        public Tensor Backward(Tensor loss)
        {
            return loss + ConnectionGrad;
        }

        public object Clone()
        {
            throw new ArgumentException("Skip connections cannot be cloned");
        }

        public Tensor Forward(Tensor input)
        {
            Identity = input.Clone() as Tensor;
            return input;
        }

        public Tensor Predict(Tensor input)
        {
            Identity = input.Clone() as Tensor;
            return input;
        }
    }

    /// <summary>
    /// A fast way to add a residual connection back to the main path in a sequetial network.
    /// </summary>
    [Serializable]
    public class SkipConnectionJoin : IModule
    {
        [SerializeReference] SkipConnectionFork forkSource;

        /// <summary>
        /// A fast way to add a residual connection back to the main path in a sequetial network.
        /// </summary>
        public SkipConnectionJoin(SkipConnectionFork connectionReference)
        {
            if (connectionReference == null)
                throw new System.ArgumentException("A skip connection join requires a valid non-null fork.");
            forkSource = connectionReference;
        }

        public Tensor Backward(Tensor loss)
        {
            this.forkSource.ConnectionGrad = loss.Clone() as Tensor;
            return loss;
        }

        public object Clone()
        {
            throw new ArgumentException("Skip connections cannot be cloned");
        }

        public Tensor Forward(Tensor input)
        {
            return input + forkSource.Identity;
        }

        public Tensor Predict(Tensor input)
        {
            return input + forkSource.Identity;
        }
    }
}



