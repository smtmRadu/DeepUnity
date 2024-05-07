/*using DeepUnity.Modules;
using System;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

///// ******************************************* Unity doesn t support recursive serialization, so this is not possible :( ***************************************************

namespace DeepUnity
{
    [Serializable]
    public class ResidualBlock : IModule, ISerializationCallbackReceiver
    {
        [SerializeField] private IModule[] modules;
        [SerializeField] private IModuleWrapper[] serializedModules;

        [SerializeField] private IModule[] modules2nd;
        [SerializeField] private IModuleWrapper[] serializedModules2nd;

        /// <summary>
        /// A residual block with two paths. The main and the second path. If the shapes of the main and the 2nd path are not equal, initialize the secondary path with a linear projection.
        /// </summary>
        /// <param name="mainPath"></param>
        /// <param name="secondaryPath"></param>
        /// <exception cref="ArgumentException"></exception>
        public ResidualBlock(IModule[] mainPath, IModule[] secondaryPath = null)
        {
            if (mainPath == null || mainPath.Length == 0)
                throw new ArgumentException("Main path modules is null or has no elements");

            this.modules = mainPath;
            this.modules2nd = secondaryPath;
        }

        public ResidualBlock(params IModule[] mainPath)
        {
            if (mainPath == null || mainPath.Length == 0)
                throw new ArgumentException("Main path modules is null or has no elements");

            this.modules = mainPath;
        }
        public Tensor Forward(Tensor input)
        {
            Tensor mainPath = input;
            for (int i = 0; i < modules.Length; i++)
            {
                mainPath = modules[i].Forward(mainPath);
            }

            Tensor secondaryPath = input;
            for (int i = 0; modules2nd != null && i < modules2nd.Length; i++)
            {
                secondaryPath = modules2nd[i].Forward(secondaryPath);
            }

            if (!mainPath.Shape.SequenceEqual(secondaryPath.Shape))
                throw new ShapeException($"The main path's shape ({mainPath.Shape.ToCommaSeparatedString()}) is not equal to the 2nd path's shape ({secondaryPath.Shape.ToCommaSeparatedString()}).");


            return mainPath + secondaryPath;
        }

        public Tensor Predict(Tensor input)
        {
            Tensor mainPath = input;
            for (int i = 0; i < modules.Length; i++)
            {
                mainPath = modules[i].Predict(mainPath);
            }

            Tensor secondaryPath = input;
            for (int i = 0; modules2nd != null && i < modules2nd.Length; i++)
            {
                secondaryPath = modules2nd[i].Predict(secondaryPath);
            }

            if (!mainPath.Shape.SequenceEqual(secondaryPath.Shape))
                throw new ShapeException($"The main path's shape ({mainPath.Shape.ToCommaSeparatedString()}) is not equal to the 2nd path's shape ({secondaryPath.Shape.ToCommaSeparatedString()}).");


            return mainPath + secondaryPath;       
        }

        public Tensor Backward(Tensor loss)
        {
            Tensor mainPath = loss;
            for (int i = modules.Length - 1; i > 0; i--)
            {
                mainPath = modules[i].Backward(mainPath);
            }

            Tensor secondaryPath = loss;
            if(modules2nd != null)
            for (int i = modules2nd.Length - 1; i > 0; i--)
            {
                secondaryPath = modules2nd[i].Backward(secondaryPath);
            }

            if (!mainPath.Shape.SequenceEqual(secondaryPath.Shape))
                throw new ShapeException($"The main path's shape ({mainPath.Shape.ToCommaSeparatedString()}) is not equal to the 2nd path's shape ({secondaryPath.Shape.ToCommaSeparatedString()}).");


            return mainPath + secondaryPath;
        }

        public void OnBeforeSerialize()
        {  // 
           //  if (serializedModules != null && serializedModules.Length > 0)
           //      serializedModules = modules.Select(x => IModuleWrapper.Wrap(x)).ToArray();
           // 
           //  if (serializedModules2nd != null && serializedModules2nd.Length > 0)
           //      serializedModules2nd = modules2nd.Select(x => IModuleWrapper.Wrap(x)).ToArray();
        }
        public void OnAfterDeserialize()
        {  // 
           //  if (modules != null && modules.Length > 0)
           //      modules = serializedModules.Select(x => IModuleWrapper.Unwrap(x)).ToArray();
           // 
           //  if (modules2nd != null && modules2nd.Length > 0)
           //      modules2nd = serializedModules2nd.Select(x => IModuleWrapper.Unwrap(x)).ToArray();
        }


        public object Clone()
        {
            IModule[] clonedModules = modules.Select(x => x.Clone() as IModule).ToArray();
            IModule[] clonedModules2nd = modules2nd.Select(x => x.Clone() as IModule).ToArray();
            return new ResidualBlock(clonedModules, clonedModules2nd);
        }

    }

}


*/