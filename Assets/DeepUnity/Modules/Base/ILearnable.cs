using System.Linq;
using UnityEngine;

namespace DeepUnity.Modules
{
    public interface ILearnable
    {
        /// <summary>
        /// On which device the computations are made on this <see cref="IModule"/>?
        /// </summary>
        [SerializeField] public Device Device { get; set; }
        /// <summary>
        /// Does this <see cref="IModule"/> computes the gradients for its <see cref="Parameter"/>s?
        /// </summary>
        [SerializeField] public bool RequiresGrad { get; set; }

        public Parameter[] Parameters(); 
        
        // Has default implementation
        public int ParametersCount()
        {
            return Parameters().Sum(x => x.param.Count());
        }
    }

}


