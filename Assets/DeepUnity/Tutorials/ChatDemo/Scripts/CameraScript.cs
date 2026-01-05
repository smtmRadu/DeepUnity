using System.Collections;
using UnityEngine;


namespace DeepUnity.Tutorials.ChatDemo
{
    public class CameraScript : MonoBehaviour
    {
        [SerializeField] private Transform defaultPosition;
        [SerializeField] private Transform npcInteractionPosition;
        [SerializeField] private float transitionDuration = 1.0f;


        public float TransitionDuration => transitionDuration;
        private Coroutine moveCoroutine;

        public void MoveToDefault()
        {
            StartMove(defaultPosition);
        }

        public void MoveToInteraction()
        {
            StartMove(npcInteractionPosition);
        }

        private void StartMove(Transform target)
        {
            if (moveCoroutine != null)
                StopCoroutine(moveCoroutine);

            moveCoroutine = StartCoroutine(MoveCamera(target));
        }

        private IEnumerator MoveCamera(Transform target)
        {
            Vector3 startPos = transform.position;
            Quaternion startRot = transform.rotation;

            Vector3 endPos = target.position;
            Quaternion endRot = target.rotation;

            float elapsed = 0f;

            while (elapsed < transitionDuration)
            {
                elapsed += Time.deltaTime;
                float t = Mathf.Clamp01(elapsed / transitionDuration);

                // Sigmoid-like easing (SmoothStep)
                t = t * t * (3f - 2f * t);

                transform.position = Vector3.Lerp(startPos, endPos, t);
                transform.rotation = Quaternion.Slerp(startRot, endRot, t);

                yield return null;
            }

            transform.position = endPos;
            transform.rotation = endRot;
        }
    }
}