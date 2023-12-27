using DeepUnity;
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

        public Transform r_leg;
        public Transform r_thigh;
        public Transform r_shin;
        public Transform r_foot;

        BodyController bodyController;
        public override void Awake()
        {
            base.Awake();

            bodyController = GetComponent<BodyController>();

            bodyController.AddBodyPart(head);
            bodyController.AddBodyPart(neck1);
            bodyController.AddBodyPart(neck2);
            bodyController.AddBodyPart(neck3);
            bodyController.AddBodyPart(neck4);
            bodyController.AddBodyPart(neck5);
            bodyController.AddBodyPart(neck6);
            bodyController.AddBodyPart(neck7);
            bodyController.AddBodyPart(torso);
            bodyController.AddBodyPart(tail1);
            bodyController.AddBodyPart(tail2);
            bodyController.AddBodyPart(tail3);
            bodyController.AddBodyPart(tail4);
            bodyController.AddBodyPart(tail5);
            bodyController.AddBodyPart(tail6);
            bodyController.AddBodyPart(tail7);
            bodyController.AddBodyPart(tail8);
            bodyController.AddBodyPart(tail9);
            bodyController.AddBodyPart(tail10);
            bodyController.AddBodyPart(tail11);

            bodyController.AddBodyPart(l_leg);
            bodyController.AddBodyPart(l_thigh);
            bodyController.AddBodyPart(l_shin);
            bodyController.AddBodyPart(l_foot);


            bodyController.AddBodyPart(r_leg);
            bodyController.AddBodyPart(r_thigh);
            bodyController.AddBodyPart(r_shin);
            bodyController.AddBodyPart(r_foot);


            bodyController.bodyPartsList.ForEach(x =>
            {
                if (x.rb.transform != l_foot && x.rb.transform != r_foot &&
                    x.rb.transform != l_shin && x.rb.transform != r_shin)
                    x.GroundContact.endEpisodeOnContact = true;
            });

        }

        public override void CollectObservations(StateVector stateVector)
        {
            // 282 inputs
            var jdDict = bodyController.bodyPartsDict;

            // 28 x 10 inputs
            foreach (var bp in bodyController.bodyPartsList)
            {
                stateVector.AddObservation(bp.rb.velocity / 20f);
                stateVector.AddObservation(bp.rb.angularVelocity / 20f);
                stateVector.AddObservation(bp.CurrentNormalizedRotation);
                stateVector.AddObservation(bp.CurrentNormalizedStrength);
            }
            // + 2 more

            stateVector.AddObservation(jdDict[l_foot].GroundContact.IsGrounded);
            stateVector.AddObservation(jdDict[r_foot].GroundContact.IsGrounded);        
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

            AddReward(head.position.y * 0.001f);
        }
    }


}


