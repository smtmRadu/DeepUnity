using System;
using UnityEngine;

namespace DeepUnity
{
    public interface IOptimizer
    {
        /// For now this generic method is deprecated.    
        private static IOptimizer Create(OptimizerType optimizerType)
        {
            switch(optimizerType)
            {
                case OptimizerType.Adam:
                    return new Adam();
                case OptimizerType.RMSProp:
                    return new RMSProp();
                case OptimizerType.SGD:
                    return new SGD();
                case OptimizerType.AdaMax:
                    return new AdaMax();
                case OptimizerType.Adagrad:
                    return new Adagrad();
                case OptimizerType.Adadelta:
                    return new Adadelta();
                default:
                    throw new Exception("Optimizer type not handled");
            }
        }
        private enum OptimizerType
        {
            Adam,
            AdaMax,
            SGD,
            Adagrad,
            Adadelta,
            RMSProp,
        }


        public void Initialize(IModule[] modules);
        public void Step(IModule[] modules);
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
        public Adagrad adagrad;
        public AdaMax adamax;
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
            else if(optimizer is Adagrad adagradOptimizer)
            {
                adagrad = adagradOptimizer;
            }
            else if(optimizer is AdaMax adamaxOptimizer)
            {
                adamax = adamaxOptimizer;
            }
            else
                throw new Exception("Unhandled optimizer type on wrapping.");
        }
        public static IOptimizer Unwrap(OptimizerWrapper optimizerWrapper)
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
            else if(typeof(Adagrad).Name.Equals(optimizerWrapper.name))
            {
                optimizer = optimizerWrapper.adagrad;
            }
            else if(typeof(AdaMax).Name.Equals(optimizerWrapper.name))
            {
                optimizer = optimizerWrapper.adamax;
            }
            else
                throw new Exception("Unhandled optimizer type on unwrapping.");

            return optimizer;
        }
    }

}
