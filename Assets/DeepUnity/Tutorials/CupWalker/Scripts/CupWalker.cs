using UnityEngine;
using DeepUnity;
using DeepUnity.ReinforcementLearning;

namespace DeepUnityTutorials
{
    public class CupWalker : Agent
    {
       [SerializeField] GameObject head;
       [SerializeField] GameObject thigh_r;
       [SerializeField] GameObject shin_r;
       [SerializeField] GameObject foot_r;
       [SerializeField] GameObject thigh_l;
       [SerializeField] GameObject shin_l;
       [SerializeField] GameObject foot_l;
        private BodyController bodyController;

        public override void Awake()
        {
            base.Awake();
            bodyController = GetComponent<BodyController>();
            bodyController.AddBodyPart(head);
            bodyController.AddBodyPart(thigh_r);
            bodyController.AddBodyPart(shin_r);
            bodyController.AddBodyPart(foot_r);
            bodyController.AddBodyPart(thigh_l);
            bodyController.AddBodyPart(shin_l);
            bodyController.AddBodyPart(foot_l);
        }

        public override void CollectObservations(StateVector stateVector)
        {
            var jdDict = bodyController.bodyPartsDict;

            // 7 x 10
            foreach (var kvp in bodyController.bodyPartsList) 
            {
                stateVector.AddObservation(kvp.rigidbody.velocity);
                stateVector.AddObservation(kvp.rigidbody.angularVelocity);
                stateVector.AddObservation(kvp.CurrentNormalizedEulerRotation);
                stateVector.AddObservation(kvp.CurrentNormalizedStrength);
            }
        }
    }


}


