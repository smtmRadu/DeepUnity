using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo2D
{
    /// <summary>
    /// Orthographic follow camera for the farm. Smooth-follows the player clamped to the map
    /// bounds; when a dialogue opens it glides to a fixed focus point (the player/NPC midpoint,
    /// nudged down so the pair sits in the upper half of the screen, clear of the bottom-docked
    /// chat panel) and zooms in. The 2D twin of SoulsCameraRig's MoveToInteraction/MoveToDefault.
    /// </summary>
    public class CameraFollow2D : MonoBehaviour
    {
        [SerializeField] private Transform target;
        [Tooltip("SmoothDamp time while following the player.")]
        [SerializeField] private float followSmoothTime = 0.16f;
        [Tooltip("SmoothDamp time while blending into/out of a dialogue framing.")]
        [SerializeField] private float dialogueSmoothTime = 0.28f;
        [SerializeField] private float defaultOrthoSize = 5.5f;
        [SerializeField] private float dialogueOrthoSize = 3.4f;
        [Tooltip("Seconds the NPC waits before opening the chat window (~3x the dialogue smooth time, when the blend has visually settled).")]
        [SerializeField] private float transitionDuration = 0.85f;
        [Tooltip("Half-extents of the farm map in world units (40x30 tiles at 1 unit/tile -> 20x15).")]
        [SerializeField] private Vector2 mapHalfExtents = new Vector2(20f, 15f);

        public float TransitionDuration => transitionDuration;

        private Camera cam;
        private bool inDialogue;
        private Vector3 dialogueFocus;
        private Vector3 posVelocity;
        private float sizeVelocity;

        private void Awake()
        {
            cam = GetComponent<Camera>();
        }

        public void EnterDialogue(Vector3 focus)
        {
            inDialogue = true;
            dialogueFocus = focus;
        }

        public void ExitDialogue()
        {
            inDialogue = false;
        }

        private void LateUpdate()
        {
            if (cam == null || target == null) return;

            float wantedSize = inDialogue ? dialogueOrthoSize : defaultOrthoSize;
            cam.orthographicSize = Mathf.SmoothDamp(cam.orthographicSize, wantedSize, ref sizeVelocity, dialogueSmoothTime);

            Vector3 wanted = inDialogue ? dialogueFocus : target.position;
            wanted = Clamp(wanted);
            wanted.z = transform.position.z;
            transform.position = Vector3.SmoothDamp(transform.position, wanted,
                ref posVelocity, inDialogue ? dialogueSmoothTime : followSmoothTime);
        }

        // keep the view inside the painted map — no dark void past the edges
        private Vector3 Clamp(Vector3 p)
        {
            float halfH = cam.orthographicSize;
            float halfW = halfH * cam.aspect;
            float limX = Mathf.Max(0f, mapHalfExtents.x - halfW);
            float limY = Mathf.Max(0f, mapHalfExtents.y - halfH);
            p.x = Mathf.Clamp(p.x, -limX, limX);
            p.y = Mathf.Clamp(p.y, -limY, limY);
            return p;
        }
    }
}
