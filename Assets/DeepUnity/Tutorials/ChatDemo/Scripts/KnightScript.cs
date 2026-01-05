using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DeepUnity;


namespace DeepUnity.Tutorials.ChatDemo
{
    public enum KnightMode
    {
        Walking = 0,
        Interaction = 1,
    }
    public class KnightScript : MonoBehaviour
    {
        [SerializeField, ViewOnly] private KnightMode mode = KnightMode.Walking;

        [SerializeField] public CameraScript cam;

        [SerializeField] private SpriteRenderer road;
        [SerializeField] private float allowedRoadRadius = 0.5f;
        [SerializeField] private Vector2 positionOffsetFromTransform = new Vector2(0, -0.5f);

        [SerializeField] private float runSpeed = 5f;
        [ViewOnly, SerializeField] private Animator animator;
        [ViewOnly, SerializeField] private int direction = 0;
        const float DIAGONAL_REDUCTION = 0.7092198f;

        bool[] roadChecks;
        Vector2[] directions8 =
            {
            new Vector2( 0,  1),
            new Vector2( 1,  1),
            new Vector2( 1,  0),
            new Vector2( 1, -1),
            new Vector2( 0, -1),
            new Vector2(-1, -1),
            new Vector2(-1,  0),
            new Vector2(-1,  1),
        };



        // Start is called before the first frame update
        void Start()
        {
            animator = GetComponent<Animator>();
        }


        public void EnterInteractiveMode()
        {
            this.mode = KnightMode.Interaction;
            cam.MoveToInteraction();

        }
        public void ExitInteractiveMode()
        {
            this.mode = KnightMode.Walking;
            cam.MoveToDefault();
        }
        // Update is called once per frame
        void Update()
        {

            roadChecks = ValidateMovement();
            Move(roadChecks);
        }
        bool[] ValidateMovement()
        {
            bool[] checks = new bool[9];




            for (int i = 0; i < 8; i++)
            {
                Vector2 worldPoint = (Vector2)transform.position + positionOffsetFromTransform +
                                     directions8[i].normalized * allowedRoadRadius;

                checks[i + 1] = IsRedAtWorldPoint(worldPoint);
            }


            return checks;
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying) return;

            Vector2 origin = (Vector2)transform.position + positionOffsetFromTransform;

            for (int i = 0; i < directions8.Length; i++)
            {
                Vector2 dir = directions8[i].normalized;
                Vector2 target = origin + dir * allowedRoadRadius;

                Gizmos.color = roadChecks[i + 1] ? Color.green : Color.red;
                Gizmos.DrawLine(origin, target);
                Gizmos.DrawSphere(target, 0.01f);
            }
        }

        bool IsRedAtWorldPoint(Vector2 worldPos)
        {
            // Convert world → local sprite space
            Vector2 localPos = road.transform.InverseTransformPoint(worldPos);

            // Convert local → pixel coordinates
            Vector2 pivot = road.sprite.pivot;
            float pixelsPerUnit = road.sprite.pixelsPerUnit;

            int x = Mathf.RoundToInt(pivot.x + localPos.x * pixelsPerUnit);
            int y = Mathf.RoundToInt(pivot.y + localPos.y * pixelsPerUnit);

            // Bounds check
            if (x < 0 || y < 0 || x >= road.sprite.texture.width || y >= road.sprite.texture.height)
                return false;

            Color c = road.sprite.texture.GetPixel(x, y);

            // Check if pixel is red & visible
            return c.a > 0f;
        }


        void Move(bool[] validRoad)
        {
            if (mode != KnightMode.Walking)
                return;

            if (Input.GetKey(KeyCode.W) && Input.GetKey(KeyCode.D) && validRoad[2])
                direction = 2;
            else if (Input.GetKey(KeyCode.D) && Input.GetKey(KeyCode.S) && validRoad[4])
                direction = 4;
            else if (Input.GetKey(KeyCode.S) && Input.GetKey(KeyCode.A) && validRoad[6])
                direction = 6;
            else if (Input.GetKey(KeyCode.A) && Input.GetKey(KeyCode.W) && validRoad[8])
                direction = 8;
            else if (Input.GetKey(KeyCode.W) && validRoad[1])
                direction = 1;
            else if (Input.GetKey(KeyCode.D) && validRoad[3])
                direction = 3;
            else if (Input.GetKey(KeyCode.S) && validRoad[5])
                direction = 5;
            else if (Input.GetKey(KeyCode.A) && validRoad[7])
                direction = 7;
            else
                direction = 0;

            animator.SetInteger("Direction", direction);

            if (direction == 0)
                return;

            if (direction == 1)
            {
                transform.Translate(0, 1 * Time.deltaTime * runSpeed, 0);
            }
            else if (direction == 2)
            {
                transform.Translate(1 * Time.deltaTime * runSpeed * DIAGONAL_REDUCTION, 1 * Time.deltaTime * runSpeed * DIAGONAL_REDUCTION, 0);
            }
            else if (direction == 3)
            {
                transform.Translate(1 * Time.deltaTime * runSpeed, 0, 0);
            }
            else if (direction == 4)
            {
                transform.Translate(1 * Time.deltaTime * runSpeed * DIAGONAL_REDUCTION, -1 * Time.deltaTime * runSpeed * DIAGONAL_REDUCTION, 0);
            }
            else if (direction == 5)
            {
                transform.Translate(0, -1 * Time.deltaTime * runSpeed, 0);
            }
            else if (direction == 6)
            {
                transform.Translate(-1 * Time.deltaTime * runSpeed * DIAGONAL_REDUCTION, -1 * Time.deltaTime * runSpeed * DIAGONAL_REDUCTION, 0);
            }
            else if (direction == 7)
            {
                transform.Translate(-1 * Time.deltaTime * runSpeed, 0, 0);
            }
            else if (direction == 8)
            {
                transform.Translate(-1 * Time.deltaTime * runSpeed * DIAGONAL_REDUCTION, 1 * Time.deltaTime * runSpeed * DIAGONAL_REDUCTION, 0);
            }
        }
    }
}