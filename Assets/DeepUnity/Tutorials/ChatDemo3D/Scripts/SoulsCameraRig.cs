using System.Collections;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Third person orbit camera (mouse look around the player) that can also blend to a fixed
    /// dialogue viewpoint and back — same contract as the 2D demo's CameraScript
    /// (MoveToInteraction / MoveToDefault / TransitionDuration).
    /// </summary>
    public class SoulsCameraRig : MonoBehaviour
    {
        [SerializeField] private Transform target;            // the player
        [SerializeField] private float distance = 4.4f;
        [SerializeField] private float pivotHeight = 1.7f;
        [SerializeField] private float mouseSensitivity = 2.4f;
        [SerializeField] private float minPitch = -28f;
        [SerializeField] private float maxPitch = 58f;
        [SerializeField] private float transitionDuration = 1.1f;
        [Tooltip("Layers the camera collides with when pulling in to avoid clipping through walls. Exclude the player's layer.")]
        [SerializeField] private LayerMask collisionMask = 1;  // Default layer only

        public float TransitionDuration => transitionDuration;

        /// <summary>Current orbit yaw in degrees — the player character aligns its back to this.</summary>
        public float Yaw => yaw;

        /// <summary>Camera forward projected on the ground plane — used for camera-relative movement.</summary>
        public Vector3 PlanarForward
        {
            get
            {
                Vector3 f = transform.forward; f.y = 0f;
                return f.sqrMagnitude < 1e-4f ? Vector3.forward : f.normalized;
            }
        }

        private enum Mode { Orbit, Dialogue }
        private Mode mode = Mode.Orbit;
        private float yaw;
        private float pitch = 16f;
        private Coroutine moveCoroutine;

        private void Start()
        {
            if (target != null)
                yaw = target.eulerAngles.y;
        }

        private void LateUpdate()
        {
            if (mode != Mode.Orbit || target == null || moveCoroutine != null)
                return;

            yaw += Input.GetAxis("Mouse X") * mouseSensitivity;
            pitch = Mathf.Clamp(pitch - Input.GetAxis("Mouse Y") * mouseSensitivity, minPitch, maxPitch);

            Vector3 pivot = target.position + Vector3.up * pivotHeight;
            Quaternion rot = Quaternion.Euler(pitch, yaw, 0f);

            float d = distance;
            if (Physics.SphereCast(pivot, 0.25f, rot * Vector3.back, out RaycastHit hit, distance, collisionMask, QueryTriggerInteraction.Ignore))
                d = Mathf.Max(0.6f, hit.distance - 0.1f);

            transform.position = pivot + rot * Vector3.back * d;
            transform.rotation = rot;
        }

        /// <summary>Blends the camera to a fixed dialogue viewpoint (a transform placed by the NPC).</summary>
        public void MoveToInteraction(Transform viewpoint)
        {
            mode = Mode.Dialogue;
            StartMove(viewpoint.position, viewpoint.rotation, null);
        }

        /// <summary>Blends back behind the player and resumes orbiting.</summary>
        public void MoveToDefault()
        {
            Vector3 pivot = target.position + Vector3.up * pivotHeight;
            Quaternion rot = Quaternion.Euler(pitch, yaw, 0f);
            StartMove(pivot + rot * Vector3.back * distance, rot, () => mode = Mode.Orbit);
        }

        private void StartMove(Vector3 endPos, Quaternion endRot, System.Action onDone)
        {
            if (moveCoroutine != null)
                StopCoroutine(moveCoroutine);
            moveCoroutine = StartCoroutine(MoveCamera(endPos, endRot, onDone));
        }

        private IEnumerator MoveCamera(Vector3 endPos, Quaternion endRot, System.Action onDone)
        {
            Vector3 startPos = transform.position;
            Quaternion startRot = transform.rotation;
            float elapsed = 0f;

            while (elapsed < transitionDuration)
            {
                elapsed += Time.deltaTime;
                float t = Mathf.Clamp01(elapsed / transitionDuration);
                t = t * t * (3f - 2f * t);   // smoothstep easing
                transform.position = Vector3.Lerp(startPos, endPos, t);
                transform.rotation = Quaternion.Slerp(startRot, endRot, t);
                yield return null;
            }

            transform.position = endPos;
            transform.rotation = endRot;
            moveCoroutine = null;
            onDone?.Invoke();
        }
    }
}
