using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, RequireComponent(typeof(HyperParameters))]
    public class Agent : MonoBehaviour
    {
        public MemoryBuffer memory { get; private set; }
        public BehaviourType behaviour = BehaviourType.Inference;
        public ActorCritic network;

        private HyperParameters hp;
        
    }

    public enum BehaviourType
    {
        Inactive,
        Active,
        Inference,
        Manual,
        Test
    }
}

