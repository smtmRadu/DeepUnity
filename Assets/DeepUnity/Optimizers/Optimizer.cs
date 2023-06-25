using System;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Optimizer serialization is off.
    /// </summary>
    public abstract class Optimizer
    {
        protected Learnable[] parameters;

        public float learningRate;
        protected float weightDecay;
        protected int t; // step counter

        protected Optimizer(Learnable[] param, float lr, float L2Penalty)
        {
            parameters = param;
            learningRate = lr;
            weightDecay = L2Penalty;
            t = 0;       
        }
        public abstract void Step();

        /// <summary>
        /// Resets all gradients to 0.
        /// </summary>
        public void ZeroGrad()
        {
            foreach (var param in parameters)
            {
                param.ZeroGrad();
            }
        }
        /// <summary>
        /// The gradients are clipped in the range [-clip_value, clip_value]
        /// </summary>
        public void ClipGradValue(float clip_value)
        {
            foreach (var param in parameters)
            {
                param.ClipGradValue(clip_value);
            }
        }
        /// <summary>
        /// The norm is computed globaly.
        /// </summary>
        public void ClipGradNorm(float max_norm)
        {
            // Compute the grad norm on each learnable module
            throw new NotImplementedException();
        }


        
    }

    /// <summary>
    /// [Deprecated]
    /// </summary>
    [Serializable]
    public class OptimizerWrapper
    {
        public string name;

        [Space]
        public Adam adam;
        public SGD sgd;       
        public Adadelta adadelta;
        public Adagrad adagrad;
        public RMSProp rmsprop;
        public AdaMax adamax;

        private OptimizerWrapper(Optimizer optimizer)
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
            else if (optimizer is RMSProp rmspropOptimizer)
            {
                rmsprop = rmspropOptimizer;
            }
            else if (optimizer is Adadelta adadeltaOptimizer)
            {
                adadelta = adadeltaOptimizer;
            }
            else if (optimizer is Adagrad adagradOptimizer)
            {
                adagrad = adagradOptimizer;
            }
            else if (optimizer is AdaMax adamaxOptimizer)
            {
                adamax = adamaxOptimizer;
            }
            else
                throw new Exception("Unhandled optimizer type on wrapping.");
        }

        public static OptimizerWrapper Wrap(Optimizer optimizer)
        {
            return new OptimizerWrapper(optimizer);
        }     
        public static Optimizer Unwrap(OptimizerWrapper optimizerWrapper)
        {
            Optimizer optimizer;

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
