using System.Linq;
using UnityEngine;

namespace DeepUnity.Modules
{
    public interface ILearnable
    {
        [SerializeField] public Device Device { get; set; }

        public Parameter[] Parameters(); 
        
        // Has default implementation
        public int ParametersCount()
        {
            return Parameters().Sum(x => x.param.Count());
        }
    }

}


