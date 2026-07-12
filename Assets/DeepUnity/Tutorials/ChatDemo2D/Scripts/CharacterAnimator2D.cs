using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo2D
{
    /// <summary>
    /// Procedural animation for the single-frame Kenney "tiny" characters: a little hop while
    /// moving, a subtle breathing bob while idle, and X-flip for left/right facing. The sprite
    /// lives on a CHILD transform ("Visual") so the bob never disturbs physics on the root.
    /// NPCs set `talking` while their reply streams in — the bob quickens, a cheap "he's
    /// speaking" tell that matches the streaming TTS.
    /// </summary>
    public class CharacterAnimator2D : MonoBehaviour
    {
        [SerializeField] private SpriteRenderer target;
        [SerializeField] private float hopHeight = 0.09f;
        [SerializeField] private float hopSpeed = 11f;
        [SerializeField] private float idleBobHeight = 0.02f;
        [SerializeField] private float idleBobSpeed = 1.7f;

        [Tooltip("Set by NPCInteractor2D while a reply streams — quickens the idle bob.")]
        public bool talking;

        private Vector3 baseLocal;
        private float clock;
        private bool moving;
        private float faceSign = 1f;

        private void Awake()
        {
            if (target != null) baseLocal = target.transform.localPosition;
        }

        /// <summary>Feed the current movement every frame (zero = idle).</summary>
        public void SetMotion(Vector2 velocity)
        {
            moving = velocity.sqrMagnitude > 0.0001f;
            if (Mathf.Abs(velocity.x) > 0.0001f) faceSign = Mathf.Sign(velocity.x);
        }

        /// <summary>Turn toward a horizontal delta (e.g. face the player when a chat opens).</summary>
        public void Face(float dx)
        {
            if (Mathf.Abs(dx) > 0.0001f) faceSign = Mathf.Sign(dx);
        }

        private void Update()
        {
            if (target == null) return;
            clock += Time.deltaTime;
            target.flipX = faceSign < 0f;

            float offset;
            if (moving)
                offset = Mathf.Abs(Mathf.Sin(clock * hopSpeed)) * hopHeight;
            else if (talking)
                offset = Mathf.Abs(Mathf.Sin(clock * idleBobSpeed * 5f)) * idleBobHeight * 2.5f;
            else
                offset = Mathf.Sin(clock * idleBobSpeed) * idleBobHeight;

            target.transform.localPosition = baseLocal + Vector3.up * offset;
        }
    }
}
