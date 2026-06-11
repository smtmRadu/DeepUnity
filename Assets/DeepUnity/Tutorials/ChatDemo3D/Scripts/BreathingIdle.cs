using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Subtle procedural breathing layered on top of whatever the Animator plays: a slow
    /// sine pitch on the chest (and faint echo on the head). Runs in LateUpdate so it
    /// survives the Animator's pose write.
    /// </summary>
    public class BreathingIdle : MonoBehaviour
    {
        [SerializeField] private float cycleSpeed = 1.7f;
        [SerializeField] private float chestPitchDegrees = 2.4f;

        private Transform chest;
        private Transform head;
        private float seed;

        private void Start()
        {
            var anim = GetComponentInChildren<Animator>();
            if (anim != null && anim.isHuman)
            {
                chest = anim.GetBoneTransform(HumanBodyBones.Chest);
                head = anim.GetBoneTransform(HumanBodyBones.Head);
            }
            seed = transform.position.x * 1.73f;   // desync multiple characters
        }

        private void LateUpdate()
        {
            if (chest == null) return;
            float s = Mathf.Sin(Time.time * cycleSpeed + seed);
            chest.localRotation *= Quaternion.Euler(s * chestPitchDegrees, 0f, 0f);
            if (head != null)
                head.localRotation *= Quaternion.Euler(s * chestPitchDegrees * 0.4f, 0f, 0f);
        }
    }
}
