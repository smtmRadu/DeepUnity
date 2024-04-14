using DeepUnity.ReinforcementLearning;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class ReacherWalker : Agent
    {
        // Spring: 3000 | Damper: 100 | MaxForce: 6000

        [Header("Needs normalization")]

        public float target_space = 5f;
        public float targets_num = 5f;
        public GameObject targetPrefab;

        public GameObject head;
        public GameObject hips;
        public GameObject chest;

        public GameObject armL;
        public GameObject forearmL;
        public GameObject handL;

        public GameObject armR;
        public GameObject forearmR;
        public GameObject handR;

        public GameObject legL;
        public GameObject shinL;
        public GameObject footL;

        public GameObject legR;
        public GameObject shinR;
        public GameObject footR;

        BodyController bodyController;
        private List<GameObject> spawnedTargets = new();
        public override void Awake()
        {
            base.Awake();

            bodyController = GetComponent<BodyController>();

            // 16 body parts
            bodyController.AddBodyPart(head);
            bodyController.AddBodyPart(chest);
            bodyController.AddBodyPart(hips);

            bodyController.AddBodyPart(armL);
            bodyController.AddBodyPart(forearmL);
            bodyController.AddBodyPart(handL);

            bodyController.AddBodyPart(armR);
            bodyController.AddBodyPart(forearmR);
            bodyController.AddBodyPart(handR);

            bodyController.AddBodyPart(legL);
            bodyController.AddBodyPart(shinL);
            bodyController.AddBodyPart(footL);

            bodyController.AddBodyPart(legR);
            bodyController.AddBodyPart(shinR);
            bodyController.AddBodyPart(footR);

            bodyController.bodyPartsList.ForEach(bodypart =>
            {
                if (bodypart.gameObject != footL.gameObject && bodypart.gameObject != footR.gameObject && bodypart.gameObject != shinL.gameObject && bodypart.gameObject != shinR.gameObject)
                {
                    bodypart.ColliderContact.OnEnter = (col) =>
                    {
                        if (col.collider.CompareTag("Ground"))
                        {
                            EndEpisode();
                        }
                    };
                }
                if (bodypart.gameObject == head.gameObject)
                {
                    bodypart.TriggerContact.OnEnter = (col) =>
                    {
                        if (col.CompareTag("Target"))
                        {
                            AddReward(+0.1f);
                        }

                        col.gameObject.SetActive(false);
                    };
                }
            });
        }

        public override void OnEpisodeBegin()
        {
            float degree = Utils.Random.Range(0f, 360f);
            float xMove = Mathf.Cos(degree);
            float zMove = Mathf.Sin(degree);

            foreach (var item in spawnedTargets)
            {
                Destroy(item);
            }
            spawnedTargets.Clear();

            for (int i = 1; i <= targets_num; i++)
            {
                Vector3 spawnPosition = transform.position + new Vector3(xMove, 0f, zMove) * i * target_space;
                spawnedTargets.Add(Instantiate(targetPrefab, spawnPosition, Quaternion.identity));
            }

            degree = Utils.Random.Range(0, 360f);
            transform.rotation *= Quaternion.Euler(0, degree, 0);

        }

        // 120 input variables
        public override void CollectObservations(StateVector stateBuffer)
        {
            var jdDict = bodyController.bodyPartsDict;

            // + 4
            stateBuffer.AddObservation(chest.transform.rotation * (chest.transform.position - targetPrefab.transform.position));

            // + 10
            BodyPart _head = jdDict[head];
            {
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_head.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_head.rigidbody.angularVelocity));
                stateBuffer.AddObservation(_head.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_head.CurrentNormalizedStrength);
            }

            // + 6
            BodyPart _chest = jdDict[chest];
            {
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_chest.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_chest.rigidbody.angularVelocity));
            }

            // + 10
            BodyPart _torso = jdDict[hips];
            {
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_torso.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_torso.rigidbody.angularVelocity));
                stateBuffer.AddObservation(_torso.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_torso.CurrentNormalizedStrength);
            }

            // + 36
            BodyPart _armR = jdDict[armR];
            BodyPart _armL = jdDict[armL];
            BodyPart _faR = jdDict[forearmR];
            BodyPart _faL = jdDict[forearmL];
            {
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_armR.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_armR.rigidbody.angularVelocity));
                stateBuffer.AddObservation(_armR.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_armR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_armL.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_armL.rigidbody.angularVelocity));
                stateBuffer.AddObservation(_armL.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_armL.CurrentNormalizedStrength);



                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_faR.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_faR.rigidbody.angularVelocity));
                stateBuffer.AddObservation(_faR.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_faR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_faL.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_faL.rigidbody.angularVelocity));
                stateBuffer.AddObservation(_faL.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_faL.CurrentNormalizedStrength);
            }

            // + 54
            BodyPart _legR = jdDict[legR];
            BodyPart _legL = jdDict[legL];
            BodyPart _shinR = jdDict[shinR];
            BodyPart _shinL = jdDict[shinL];
            BodyPart _footR = jdDict[footR];
            BodyPart _footL = jdDict[footL];
            {
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_legR.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_legR.rigidbody.angularVelocity));
                stateBuffer.AddObservation(_legR.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_legR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_legL.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_legL.rigidbody.angularVelocity));
                stateBuffer.AddObservation(_legL.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_legL.CurrentNormalizedStrength);

                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_shinR.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_shinR.rigidbody.angularVelocity));
                stateBuffer.AddObservation(_shinR.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_shinR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_shinL.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_shinL.rigidbody.angularVelocity));
                stateBuffer.AddObservation(_shinL.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_shinL.CurrentNormalizedStrength);

                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_footR.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_footR.rigidbody.angularVelocity));
                stateBuffer.AddObservation(_footR.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_footR.CurrentNormalizedStrength);
                stateBuffer.AddObservation(_footR.ColliderContact.IsGrounded);

                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_footL.rigidbody.velocity));
                stateBuffer.AddObservation(spawnedTargets.Last().transform.InverseTransformDirection(_footL.rigidbody.angularVelocity));
                stateBuffer.AddObservation(_footL.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_footL.CurrentNormalizedStrength);
                stateBuffer.AddObservation(_footL.ColliderContact.IsGrounded);
            }
        }

        // 36 continuous actions
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            var jdDict = bodyController.bodyPartsDict;

            float[] actions_vector = actionBuffer.ContinuousActions;

            int i = 0;
            jdDict[head].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[hips].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);

            jdDict[armL].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[forearmL].SetJointTargetRotation(actions_vector[i++], 0f, 0f);

            jdDict[armR].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[forearmR].SetJointTargetRotation(actions_vector[i++], 0f, 0f);

            jdDict[legL].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[shinL].SetJointTargetRotation(actions_vector[i++], 0f, 0f);
            jdDict[footL].SetJointTargetRotation(actions_vector[i++], 0f, 0f);

            jdDict[legR].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[shinR].SetJointTargetRotation(actions_vector[i++], 0f, 0f);
            jdDict[footR].SetJointTargetRotation(actions_vector[i++], 0f, 0f);


            jdDict[head].SetJointStrength(actions_vector[i++]);
            jdDict[hips].SetJointStrength(actions_vector[i++]);

            jdDict[armL].SetJointStrength(actions_vector[i++]);
            jdDict[forearmL].SetJointStrength(actions_vector[i++]);

            jdDict[armR].SetJointStrength(actions_vector[i++]);
            jdDict[forearmR].SetJointStrength(actions_vector[i++]);

            jdDict[legL].SetJointStrength(actions_vector[i++]);
            jdDict[shinL].SetJointStrength(actions_vector[i++]);
            jdDict[footL].SetJointStrength(actions_vector[i++]);

            jdDict[legR].SetJointStrength(actions_vector[i++]);
            jdDict[shinR].SetJointStrength(actions_vector[i++]);
            jdDict[footR].SetJointStrength(actions_vector[i++]);

            if (transform.position.y < -10f) // Falls of the platform
                EndEpisode();


            // look to target reward
            // Vector3 dir = spawnedTargets.Last().transform.position - head.transform.position;
            // float alignment = Vector3.Dot(head.transform.forward.normalized, dir.normalized);
            // 
            // AddReward(alignment / 100f);
            // alive reward
            // AddReward(0.0003f);
        }
    }


}

