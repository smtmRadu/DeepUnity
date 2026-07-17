using System.Collections;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Sheathes the player's gear for conversation: when the chat opens the sword tweens to a
    /// scabbard socket on the right hip and the shield to a carry socket on the back; when the chat
    /// closes both return to the hands. The sockets are parented to the hip/chest bones so stowed
    /// gear rides the idle sway. The move itself is a short eased arc — the animation packs ship no
    /// sheath clips, so the motion is procedural. Socket placement is expressed in CHARACTER-ROOT
    /// axes (x=right, y=up, z=forward) and converted onto the bones at Awake; tuned via the
    /// StowTuneProbe screenshots.
    /// </summary>
    public class WeaponStower : MonoBehaviour
    {
        [SerializeField] private Transform sword;
        [SerializeField] private Transform shield;
        [SerializeField, Tooltip("Seconds for each item's hand->socket move.")]
        private float duration = 0.55f;

        [Header("Stow placement (character-root axes, tuned via StowTuneProbe screenshots)")]
        [SerializeField] private Vector3 waistPosition = new Vector3(0.24f, 0.88f, -0.10f);
        [SerializeField] private Vector3 waistEuler = new Vector3(105f, -10f, 90f);   // blade local +Z; z=90 lays the flat against the leg
        [SerializeField] private Vector3 backPosition = new Vector3(0.02f, 0.80f, -0.26f);   // chibi rig: torso sits LOW
        [SerializeField] private Vector3 backEuler = new Vector3(270f, 180f, 0f);   // ornate face OUT, point DOWN (lineup pick #4)

        Transform waistSocket, backSocket, swordHome, shieldHome;
        Vector3 swordHomePos, shieldHomePos;
        Quaternion swordHomeRot, shieldHomeRot;
        Coroutine job;

        void Awake()
        {
            var anim = GetComponentInChildren<Animator>();
            if (anim == null || sword == null || shield == null) { enabled = false; return; }
            var hips = anim.GetBoneTransform(HumanBodyBones.Hips);
            var chest = anim.GetBoneTransform(HumanBodyBones.Chest);
            if (chest == null) chest = anim.GetBoneTransform(HumanBodyBones.Spine);
            if (hips == null || chest == null) { enabled = false; return; }

            waistSocket = MakeSocket("StowSocket_Waist", hips, waistPosition, waistEuler);
            backSocket = MakeSocket("StowSocket_Back", chest, backPosition, backEuler);

            swordHome = sword.parent; swordHomePos = sword.localPosition; swordHomeRot = sword.localRotation;
            shieldHome = shield.parent; shieldHomePos = shield.localPosition; shieldHomeRot = shield.localRotation;
        }

        // socket authored in root axes (pose-independent numbers), then attached to the bone
        Transform MakeSocket(string name, Transform bone, Vector3 rootPos, Vector3 rootEuler)
        {
            var s = new GameObject(name).transform;
            s.position = transform.TransformPoint(rootPos);
            s.rotation = transform.rotation * Quaternion.Euler(rootEuler);
            s.SetParent(bone, true);
            return s;
        }

        public void Stow() => Restart(true);
        public void Draw() => Restart(false);

        void Restart(bool stow)
        {
            if (!enabled) return;
            if (job != null) StopCoroutine(job);
            job = StartCoroutine(MoveBoth(stow));
        }

        IEnumerator MoveBoth(bool stow)
        {
            // reparent up-front (world pose kept) so both tween endpoints ride the animating body
            sword.SetParent(stow ? waistSocket : swordHome, true);
            shield.SetParent(stow ? backSocket : shieldHome, true);
            Vector3 sp0 = sword.localPosition, hp0 = shield.localPosition;
            Quaternion sr0 = sword.localRotation, hr0 = shield.localRotation;
            Vector3 spT = stow ? Vector3.zero : swordHomePos, hpT = stow ? Vector3.zero : shieldHomePos;
            Quaternion srT = stow ? Quaternion.identity : swordHomeRot, hrT = stow ? Quaternion.identity : shieldHomeRot;

            const float shieldLag = 0.12f;   // shield trails the sword slightly — reads deliberate
            float t = 0f;
            while (t < duration + shieldLag)
            {
                t += Time.deltaTime;
                float ts = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(t / duration));
                float th = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01((t - shieldLag) / duration));
                sword.localPosition = Vector3.Lerp(sp0, spT, ts);
                sword.localRotation = Quaternion.Slerp(sr0, srT, ts);
                shield.localPosition = Vector3.Lerp(hp0, hpT, th);
                shield.localRotation = Quaternion.Slerp(hr0, hrT, th);
                // small outward bulge so the item arcs around the body instead of clipping through it
                sword.position += transform.right * (Mathf.Sin(ts * Mathf.PI) * 0.10f);
                shield.position += (transform.up * 0.5f - transform.forward * 0.85f).normalized
                                   * (Mathf.Sin(th * Mathf.PI) * 0.14f);
                yield return null;
            }
            sword.localPosition = spT; sword.localRotation = srT;
            shield.localPosition = hpT; shield.localRotation = hrT;
            job = null;
        }
    }
}
