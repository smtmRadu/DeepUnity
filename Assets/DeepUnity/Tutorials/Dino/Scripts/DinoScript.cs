using DeepUnity;
using DeepUnity.ReinforcementLearning;
using JetBrains.Annotations;
using System.Runtime.InteropServices;
using Unity.Burst.CompilerServices;
using UnityEngine;


namespace DeepUnityTutorials
{
    public class DinoScript : Agent
    {
        [Header("Body Parts 28")]
        public Transform head;
        public Transform neck1;
        public Transform neck2;
        public Transform neck3;
        public Transform neck4;
        public Transform neck5;
        public Transform neck6;
        public Transform neck7;

        public Transform torso;
        public Transform tail1;
        public Transform tail2;
        public Transform tail3;
        public Transform tail4;
        public Transform tail5;
        public Transform tail6;
        public Transform tail7;
        public Transform tail8;
        public Transform tail9;
        public Transform tail10;
        public Transform tail11;

        public Transform l_leg;
        public Transform l_thigh;
        public Transform l_shin;
        public Transform l_foot;
        bool lfoot_gr;

        public Transform r_leg;
        public Transform r_thigh;
        public Transform r_shin;
        public Transform r_foot;
        bool rfoot_gr;

        BodyController bodyController;
        public override void Awake()
        {
            base.Awake();

            bodyController = GetComponent<BodyController>();

            bodyController.AddBodyPart(head.gameObject);
            bodyController.AddBodyPart(neck1.gameObject);
            bodyController.AddBodyPart(neck2.gameObject);
            bodyController.AddBodyPart(neck3.gameObject);
            bodyController.AddBodyPart(neck4.gameObject);
            bodyController.AddBodyPart(neck5.gameObject);
            bodyController.AddBodyPart(neck6.gameObject);
            bodyController.AddBodyPart(neck7.gameObject);
            bodyController.AddBodyPart(torso.gameObject);
            bodyController.AddBodyPart(tail1.gameObject);
            bodyController.AddBodyPart(tail2.gameObject);
            bodyController.AddBodyPart(tail3.gameObject);
            bodyController.AddBodyPart(tail4.gameObject);
            bodyController.AddBodyPart(tail5.gameObject);
            bodyController.AddBodyPart(tail6.gameObject);
            bodyController.AddBodyPart(tail7.gameObject);
            bodyController.AddBodyPart(tail8.gameObject);
            bodyController.AddBodyPart(tail9.gameObject);
            bodyController.AddBodyPart(tail10.gameObject);
            bodyController.AddBodyPart(tail11.gameObject);

            bodyController.AddBodyPart(l_leg.gameObject);
            bodyController.AddBodyPart(l_thigh.gameObject);
            bodyController.AddBodyPart(l_shin.gameObject);
            bodyController.AddBodyPart(l_foot.gameObject);


            bodyController.AddBodyPart(r_leg.gameObject);
            bodyController.AddBodyPart(r_thigh.gameObject);
            bodyController.AddBodyPart(r_shin.gameObject);
            bodyController.AddBodyPart(r_foot.gameObject);


            bodyController.bodyPartsList.ForEach(x =>
            {
                if (x.rigidbody.transform != l_foot && x.rigidbody.transform != r_foot &&
                    x.rigidbody.transform != l_shin && x.rigidbody.transform != r_shin)
                    x.ColliderContact.OnEnter = (col) => { if (col.collider.CompareTag("Ground"))  EndEpisode(); };
            });

            bodyController.bodyPartsDict[l_foot.gameObject].ColliderContact.OnEnter = (col) => { if (col.collider.CompareTag("Ground")) lfoot_gr = true; };
            bodyController.bodyPartsDict[r_foot.gameObject].ColliderContact.OnEnter = (col) => { if (col.collider.CompareTag("Ground")) rfoot_gr = true; };

            bodyController.bodyPartsDict[l_foot.gameObject].ColliderContact.OnExit = (col) => { if (col.collider.CompareTag("Ground")) lfoot_gr = false; };
            bodyController.bodyPartsDict[r_foot.gameObject].ColliderContact.OnExit = (col) => { if (col.collider.CompareTag("Ground")) rfoot_gr = false; };

        }

        public override void CollectObservations(StateVector stateVector)
        {
            // 282 inputs
            var jdDict = bodyController.bodyPartsDict;

            // 28 x 10 inputs
            foreach (var bp in bodyController.bodyPartsList)
            {
                stateVector.AddObservation(bp.rigidbody.velocity / 30f);
                stateVector.AddObservation(bp.rigidbody.angularVelocity / 30f);
                stateVector.AddObservation(bp.CurrentNormalizedEulerRotation);
                stateVector.AddObservation(bp.CurrentNormalizedStrength);
            }
            // + 2 more

            stateVector.AddObservation(lfoot_gr);
            stateVector.AddObservation(rfoot_gr);        
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            // 27 x 4 actions
            float[] actions_vector = actionBuffer.ContinuousActions;
            int index = 0;
            foreach (var bp in bodyController.bodyPartsList)
            {
                if (bp.gameObject == torso.gameObject)
                    continue;

                bp.SetJointTargetRotation(actions_vector[index++], actions_vector[index++], actions_vector[index++]);
                bp.SetJointStrength(actions_vector[index++]);
            }

            AddReward(head.position.x * 0.001f);
        }
    }


}


