using System;
using UnityEngine;

namespace DeepUnity
{
    public interface IOptimizer
    {
        public void Step(Dense[] layers);
    }

    [Serializable]
    public class OptimizerWrapper
    {
        public string name;
        [Space]
        public Adam adam;
        public SGD sgd;
        public RMSProp rmsprop;
        public Adadelta adadelta;
        public OptimizerWrapper(IOptimizer optimizer)
        {
            // Initialize the fields based on the optimizer type
            name = optimizer.GetType().Name;
            if (optimizer is Adam adamOptimizer)
            {
                adam = adamOptimizer;
            }
            else if (optimizer is SGD sgdOptimizer)
            {
                sgd = sgdOptimizer;
            }
            else if(optimizer is RMSProp rmspropOptimizer)
            {
                rmsprop = rmspropOptimizer;
            }
            else if(optimizer is Adadelta adadeltaOptimizer)
            {
                adadelta = adadeltaOptimizer;
            }
            else
                throw new Exception("Unhandled optimizer type on wrapping.");
        }
        public static IOptimizer Get(OptimizerWrapper optimizerWrapper)
        {
            IOptimizer optimizer = null;

            if (typeof(Adam).Name.Equals(optimizerWrapper.name))
            {
                optimizer = optimizerWrapper.adam;
            }
            else if (typeof(SGD).Name.Equals(optimizerWrapper.name))
            {
                optimizer = optimizerWrapper.sgd;
            }
            else if(typeof(RMSProp).Name.Equals(optimizerWrapper.name))
            {
                optimizer = optimizerWrapper.rmsprop;
            }
            else if(typeof(Adadelta).Name.Equals(optimizerWrapper.name))
            {
                optimizer = optimizerWrapper.adadelta;
            }
            else
                throw new Exception("Unhandled optimizer type on unwrapping.");

            return optimizer;
        }
    }

}
