using System;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.Modules
{
    public static class ResidualConnection
    {
        [Serializable]
        public class Fork : IModule
        {
            public static Lazy<Stack<Fork>> UnjoinedForks = new Lazy<Stack<Fork>>();
            public Tensor ConnectionGrad { private get; set; }
            public Tensor Identity { get; private set; }

            [SerializeField] private Tensor linearProjection;

            public Fork()
            {
                UnjoinedForks.Value.Push(this);
            }

            // /// <summary>
            // /// Applies a linear projection to the identity to match the result
            // /// </summary>
            // /// <param name="linearProjection"></param>
            // public Fork(Tensor linearProjection)
            // {
            //     UnjoinedForks.Value.Push(this);
            //     this.linearProjection = linearProjection;
            // }

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

        [Serializable]
        public class Join : IModule
        {
            [SerializeReference] Fork forkSource;

            public Join()
            {
                if (Fork.UnjoinedForks.Value.Count == 0)
                    throw new Exception("Before joining a residual connection to a main path, a fork must be created firstly.");

                forkSource = Fork.UnjoinedForks.Value.Pop();
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
    
}



