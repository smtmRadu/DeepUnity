using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo2D
{
    /// <summary>
    /// Top-down farm movement: WASD / arrow keys drive a dynamic Rigidbody2D (gravity 0), so
    /// fences, buildings and the map edge block through ordinary physics. Facing (the last
    /// non-zero move direction) is what FarmingSystem probes ahead of to pick the target plot,
    /// and CharacterAnimator2D turns the motion into the tiny-sprite hop. During dialogue the
    /// controller sits in Interaction mode: input is ignored (WASD then types into the chat
    /// window instead) and the farmer turns toward the NPC.
    /// </summary>
    [RequireComponent(typeof(Rigidbody2D))]
    public class PlayerController2D : MonoBehaviour
    {
        public enum PlayerMode { Walking = 0, Interaction = 1 }

        [SerializeField, ViewOnly] private PlayerMode mode = PlayerMode.Walking;
        [SerializeField] public CameraFollow2D cam;
        [SerializeField] private CharacterAnimator2D anim;
        [SerializeField] private float runSpeed = 4.2f;

        /// <summary>True while a dialogue owns the input (mirrors SoulsPlayerController.IsBusy).</summary>
        public bool IsBusy => mode == PlayerMode.Interaction;

        /// <summary>Unit vector of the last non-zero movement — the farming probe direction.</summary>
        public Vector2 Facing { get; private set; } = Vector2.down;

        private Rigidbody2D rb;
        private Vector2 input;

        private void Awake()
        {
            rb = GetComponent<Rigidbody2D>();
        }

        public void EnterInteractiveMode()
        {
            mode = PlayerMode.Interaction;
            input = Vector2.zero;
            rb.velocity = Vector2.zero;
            anim?.SetMotion(Vector2.zero);
        }

        public void ExitInteractiveMode()
        {
            mode = PlayerMode.Walking;
        }

        /// <summary>Turn the sprite toward a world point (the NPC when a dialogue opens).</summary>
        public void FaceTowards(Vector2 worldPos)
        {
            anim?.Face(worldPos.x - transform.position.x);
        }

        private void Update()
        {
            if (mode != PlayerMode.Walking)
            {
                input = Vector2.zero;
                return;
            }

            input = new Vector2(
                (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow) ? 1f : 0f) -
                (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow) ? 1f : 0f),
                (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow) ? 1f : 0f) -
                (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow) ? 1f : 0f));

            if (input.sqrMagnitude > 0f)
                Facing = input.normalized;
            anim?.SetMotion(input);
        }

        private void FixedUpdate()
        {
            rb.velocity = input.normalized * runSpeed;
        }
    }
}
