using DeepUnity.Models;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    public class BehaviourModifier : MonoBehaviour
    {
        [SerializeField] private AgentBehaviour behavior;

        [Space]
        [Range(0f, 0.1f)] public float uniformDistRange = 0.001f;


        [Header("Modify your policy's inputs or outputs")]
        [Button("ModifyInputs")]
        public int newSpaceSize = 0;


        public void ModifyInputs()
        {
            if (newSpaceSize == 0)
            {
                ConsoleMessage.Error("Invalid new space size");
                return;
            }

            ModifyNetInputs(behavior.vNetwork);
            ModifyNetInputs(behavior.q1Network);
            ModifyNetInputs(behavior.q2Network);
            ModifyNetInputs(behavior.muNetwork);
            ModifyNetInputs(behavior.sigmaNetwork);

            behavior.observationSize = newSpaceSize;
            behavior.observationsNormalizer = new RunningNormalizer(newSpaceSize);
            behavior.Save();
        }

        private void ModifyNetInputs(Sequential network)
        {
            if (network == null)
                return;

            var oldWeight = network.Parameters()[0].theta;
            Tensor newWeight = Tensor.Zeros(oldWeight.Size(-2), newSpaceSize);

            int oldSpaceSize = oldWeight.Size(-1);

            for (int i = 0; i < oldWeight.Size(-2); i++)
            {
                for (int J = 0; J < newSpaceSize; J++)
                {
                    if(J >= oldSpaceSize)
                        newWeight[i, J] = Utils.Random.Range(-uniformDistRange, uniformDistRange);
                    else
                        newWeight[i, J] = oldWeight[i, J];

                }
            }

            Tensor.CopyTo(newWeight, oldWeight);
            ConsoleMessage.Info($"The network {network.name} was modified");
        }
    }


}

