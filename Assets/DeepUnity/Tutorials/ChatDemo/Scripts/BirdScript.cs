using DeepUnity;
using UnityEngine;


namespace DeepUnity.Tutorials.ChatDemo
{
    public class BirdScript : MonoBehaviour
    {
        [SerializeField] private Transform map;
        [SerializeField] private float maximumDistance = 50f;
        [SerializeField] private float speed = 4f;
        [SerializeField] private float maxAngleFromXAxis = 10f;
        [SerializeField, ViewOnly] private Vector2 direction;

        void Start()
        {
            RandomizeHorizontalDirection();
        }

        void Update()
        {
            Move();

            if (IsTooFarFromMap())
            {
                RandomizeHorizontalDirectionTowardsMap();
            }

            UpdateVisualFacing();
        }

        private void UpdateVisualFacing()
        {
            if (direction.x < 0f)
                transform.rotation = Quaternion.Euler(0f, 180f, 0f);
            else
                transform.rotation = Quaternion.Euler(0f, 0f, 0f);
        }


        private void Move()
        {
            transform.position += (Vector3)(direction * speed * Time.deltaTime);
        }

        private bool IsTooFarFromMap()
        {
            return Vector2.Distance(transform.position, map.position) > maximumDistance;
        }

        private void RandomizeHorizontalDirection()
        {
            // Choose left (-1) or right (+1)
            float horizontalSign = Random.value < 0.5f ? -1f : 1f;

            // Random angle within ±maxAngleFromXAxis
            float angle = Random.Range(-maxAngleFromXAxis, maxAngleFromXAxis);

            float radians = angle * Mathf.Deg2Rad;

            direction = new Vector2(
                Mathf.Cos(radians) * horizontalSign,
                Mathf.Sin(radians)
            ).normalized;
        }

        private void RandomizeHorizontalDirectionTowardsMap()
        {
            Vector2 toMap = (map.position - transform.position);
            float horizontalSign = Mathf.Sign(toMap.x);

            // If exactly centered, pick a random side
            if (horizontalSign == 0)
                horizontalSign = Random.value < 0.5f ? -1f : 1f;

            float angle = Random.Range(-maxAngleFromXAxis, maxAngleFromXAxis);
            float radians = angle * Mathf.Deg2Rad;

            direction = new Vector2(
                Mathf.Cos(radians) * horizontalSign,
                Mathf.Sin(radians)
            ).normalized;
        }
    }

}

