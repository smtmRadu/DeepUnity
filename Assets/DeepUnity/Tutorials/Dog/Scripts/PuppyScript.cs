using UnityEngine;
using DeepUnity;
using System.Runtime.InteropServices;
using Unity.Burst.CompilerServices;

namespace DeepUnityTutorials
{

    public class PuppyScript : Agent
    {
        public GameObject target;

        public GameObject body;
        public GameObject head;
        public GameObject tail;
        public GameObject tail2;

        public GameObject tlf;
        public GameObject slf;
        public GameObject flf;                            
        public GameObject trf;
        public GameObject srf;
        public GameObject frf;
        public GameObject tlb;
        public GameObject slb;
        public GameObject flb;
        public GameObject trb;
        public GameObject srb;
        public GameObject frb;

        private BodyController bodyController;

        public float initial_distance;

        public override void Awake()
        {
            base.Awake();
            bodyController = GetComponent<BodyController>();

            initial_distance = Vector3.Distance(head.transform.position, target.transform.position);

            bodyController.AddBodyPart(body);
            bodyController.AddBodyPart(head);
            bodyController.AddBodyPart(tail);
            bodyController.AddBodyPart(tail2);
            bodyController.AddBodyPart(tlf);
            bodyController.AddBodyPart(slf);
            bodyController.AddBodyPart(flf);
            bodyController.AddBodyPart(trf);
            bodyController.AddBodyPart(srf);
            bodyController.AddBodyPart(frf);
            bodyController.AddBodyPart(tlb);
            bodyController.AddBodyPart(slb);
            bodyController.AddBodyPart(flb);
            bodyController.AddBodyPart(trb);
            bodyController.AddBodyPart(srb);
            bodyController.AddBodyPart(frb);

            bodyController.bodyPartsList.ForEach(x =>
            {
                if (x.gameObject == head || x.gameObject == tail || x.gameObject == tail2 || x.gameObject == body ||
                    x.gameObject == tlf || x.gameObject == trf || x.gameObject == tlb || x.gameObject == trb)
                    x.ColliderContact.OnEnter = (col) =>
                    {
                        if(col.collider.CompareTag("Ground"))
                        {
                            float reward =  (initial_distance - Vector3.Distance(head.transform.position, target.transform.position)) / initial_distance;
                            SetReward(reward);
                            EndEpisode();
                        }                  
                    };
            });

            bodyController[head].TriggerContact.OnEnter = (col) =>
            {
                if(col.CompareTag("Target"))
                {
                    SetReward(+1);
                    EndEpisode();
                }
            };



        }

        public override void CollectObservations(StateVector stateVector)
        {

            const float velocity_norm = 30f;

            // 15 x 10
            foreach (var bp in bodyController.bodyPartsList)
            {
                if (bp.gameObject == body)
                    continue;

                stateVector.AddObservation(bp.rigidbody.velocity / velocity_norm);
                stateVector.AddObservation(bp.rigidbody.angularVelocity / velocity_norm);
                stateVector.AddObservation(bp.CurrentNormalizedEulerRotation);
                stateVector.AddObservation(bp.CurrentNormalizedStrength);
            }

            // + 6
            stateVector.AddObservation(head.transform.forward);
            stateVector.AddObservation((target.transform.position - head.transform.position).normalized);


        }


        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            float[] actions_vector = actionBuffer.ContinuousActions;
            int index = 0;

            // 15 x 4 = 60
            foreach (var item in bodyController.bodyPartsList)
            {
                if (item.gameObject == body)
                    continue;

                item.SetJointTargetRotation(actions_vector[index++], actions_vector[index++], actions_vector[index++]);
                item.SetJointStrength(actions_vector[index++]);
            }

            if (body.transform.position.y < -10f)
                EndEpisode();
        }


    }

}



