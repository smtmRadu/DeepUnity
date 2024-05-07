/*using DeepUnity.ReinforcementLearning;
using UnityEngine;

// i do not know exactly why it doesn t work consistently.
namespace DeepUnity
{ 
    [RequireComponent(typeof(Agent))]
    public class EpisodeEndRagdoll : MonoBehaviour
    {
        [Header("Must be placed after Agent script.")]
        [Header("This script deactivates the behaviour of the agent \n for few seconds when his episode ends to get an\n effect of a dead ragdoll.")]
        [Min(0f), SerializeField] private float waitSeconds = 2f;
        Agent ag;

        [SerializeField, ViewOnly] bool isRagdoll = false;
        BehaviourType initialBehaviour;


        [SerializeField, ViewOnly] float elapsedTime = 0f;

        private void Awake()
        {
            ag = GetComponent<Agent>();
            initialBehaviour = ag.behaviourType;
        }

        private void FixedUpdate()
        {
            if (ag.Timestep.done[0] == 1 && !isRagdoll)
            {
                isRagdoll = true;
                ag.behaviourType = BehaviourType.Off;
            }

            if (isRagdoll)
            {
                elapsedTime += Time.fixedDeltaTime;

                if (elapsedTime >= waitSeconds)
                {
                    isRagdoll = false;
                    ag.behaviourType = initialBehaviour;
                    elapsedTime = 0f;
                }
            }
        }
    }

}


*/