using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo2D
{
    /// <summary>
    /// Barnyard ambience: a chicken/cow/sheep ambling between random points inside its pen
    /// rectangle, pausing to "graze" between hops. Pure transform movement (no physics — the
    /// area rect is the fence), hop animation via CharacterAnimator2D.
    /// </summary>
    public class CritterWander2D : MonoBehaviour
    {
        [Tooltip("World-space rectangle the critter roams in (xMin, yMin, width, height).")]
        [SerializeField] private Rect area = new Rect(-2f, -2f, 4f, 4f);
        [SerializeField] private float speed = 1.0f;
        [SerializeField] private Vector2 grazeSecondsRange = new Vector2(1.5f, 5f);
        [SerializeField] private CharacterAnimator2D anim;

        private Vector2 target;
        private float grazeTimer;

        private void Start()
        {
            target = transform.position;
            grazeTimer = Random.Range(0f, grazeSecondsRange.y);   // desync the herd
        }

        private void Update()
        {
            Vector2 pos = transform.position;

            if (grazeTimer > 0f)
            {
                grazeTimer -= Time.deltaTime;
                anim?.SetMotion(Vector2.zero);
                if (grazeTimer <= 0f)
                    target = new Vector2(Random.Range(area.xMin, area.xMax),
                                         Random.Range(area.yMin, area.yMax));
                return;
            }

            Vector2 delta = target - pos;
            if (delta.magnitude < 0.05f)
            {
                grazeTimer = Random.Range(grazeSecondsRange.x, grazeSecondsRange.y);
                anim?.SetMotion(Vector2.zero);
                return;
            }

            Vector2 step = delta.normalized * speed * Time.deltaTime;
            transform.position = pos + step;
            anim?.SetMotion(step);
        }
    }
}
