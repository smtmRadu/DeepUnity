using System;
using System.Collections.Generic;
using System.Linq;
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
            int totalCount = 0;
            foreach (var param in parameters)
            {
                totalCount += param.gamma.Count();
                totalCount += param.beta.Count();
            }

            // Concatenate all gradients in a single tensor vector
            float[] vector = new float[totalCount];
            int index = 0;
            foreach (var param in parameters)
            {
                float[] gradG = param.gammaGrad.ToArray();
                float[] gradB = param.betaGrad.ToArray();

                for (int i = 0; i < gradG.Length; i++)
                {
                    vector[index++] = gradG[i];
                }
                for (int i = 0; i < gradB.Length; i++)
                {
                    vector[index++] = gradB[i];
                }
            }

            // Compute norm
            Tensor norm = Tensor.Norm(Tensor.Constant(vector), NormType.ManhattanL1);

            if (norm[0] <= max_norm)
                return;

            float scale = max_norm / norm[0];

            foreach (var item in parameters)
            {
                item.gammaGrad *= scale;
                item.betaGrad *= scale;
            }
            
        }


        
    }

    /// <summary>
    /// [Deprecated]
    /// </summary>
    [Serializable]
    internal class OptimizerWrapper
    {
        public string name;

        [Space]
        public Adam adam;
        public SGD sgd;       
        public Adadelta adadelta;
        public Adagrad adagrad;
        public RMSProp rmsprop;
        public Adamax adamax;

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
            else if (optimizer is Adamax adamaxOptimizer)
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
            else if(typeof(Adamax).Name.Equals(optimizerWrapper.name))
            {
                optimizer = optimizerWrapper.adamax;
            }
            else
                throw new Exception("Unhandled optimizer type on unwrapping.");

            return optimizer;
        }
    }

}
