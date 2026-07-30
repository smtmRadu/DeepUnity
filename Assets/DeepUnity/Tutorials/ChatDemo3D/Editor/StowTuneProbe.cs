#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D.EditorTools
{
    /// <summary>
    /// Edit-mode preview of the WeaponStower placement: parents the sword/shield onto the hip/chest
    /// bones at the candidate stow poses (same root-axis math as WeaponStower.MakeSocket) so a
    /// poseAnimators screenshot shows exactly what the stowed gear will look like in game. Iterate
    /// the numbers here, then copy the winners into WeaponStower's serialized defaults. Restore (or
    /// a scene rebuild) puts the weapons back in the hands.
    /// </summary>
    public static class StowTuneProbe
    {
        // ---- candidates under tuning (root axes: x=right, y=up, z=forward) ----
        static readonly Vector3 WaistPos = new Vector3(0.24f, 0.88f, -0.10f);
        static readonly Vector3 WaistEuler = new Vector3(105f, -10f, 90f);   // blade local +Z; z=90 flat against the leg
        static readonly Vector3 BackPos = new Vector3(0.02f, 0.80f, -0.26f);   // chibi rig: torso is LOW
        static readonly Vector3 BackEuler = new Vector3(270f, 180f, 0f);   // face out, point down

        static Transform swordHome, shieldHome;
        static Vector3 swordPos, shieldPos;
        static Quaternion swordRot, shieldRot;

        [MenuItem("DeepUnity/ChatDemo3D/StowTune - Snap Stowed")]
        public static void Snap()
        {
            if (!Find(out var root, out var hips, out var chest, out var sword, out var shield)) return;
            if (swordHome == null) { swordHome = sword.parent; swordPos = sword.localPosition; swordRot = sword.localRotation; }
            if (shieldHome == null) { shieldHome = shield.parent; shieldPos = shield.localPosition; shieldRot = shield.localRotation; }

            Place(sword, root, hips, WaistPos, WaistEuler);
            Place(shield, root, chest, BackPos, BackEuler);
            Debug.Log($"[StowTune] snapped: waist {WaistPos}/{WaistEuler}  back {BackPos}/{BackEuler}");
        }

        // ---- rotation lineup: N shield clones + N sword clones floating in a row behind the player,
        // each with a different candidate rotation, so ONE screenshot picks the right axes ----
        static readonly Vector3[] ShieldCands =
        { new Vector3(90,0,180), new Vector3(90,180,0), new Vector3(270,0,0), new Vector3(270,180,0) };
        static readonly Vector3[] SwordCands =
        { new Vector3(100,0,0), new Vector3(112,-18,0), new Vector3(125,10,0), new Vector3(100,0,90) };

        [MenuItem("DeepUnity/ChatDemo3D/StowTune - Rotation Lineup")]
        public static void Lineup()
        {
            LineupClear();
            if (!Find(out var root, out _, out _, out var sword, out var shield)) return;
            var parent = new GameObject("__stowLineup").transform;
            for (int i = 0; i < ShieldCands.Length; i++)
            {
                var c = Object.Instantiate(shield.gameObject, parent);
                c.transform.localScale = shield.lossyScale;   // keep WORLD size (bone chain shrinks the original)
                c.transform.position = root.position + root.right * (-1.8f + i * 1.2f) + Vector3.up * 1.1f + root.forward * 1.2f;
                c.transform.rotation = root.rotation * Quaternion.Euler(ShieldCands[i]);
            }
            for (int i = 0; i < SwordCands.Length; i++)
            {
                var c = Object.Instantiate(sword.gameObject, parent);
                c.transform.localScale = sword.lossyScale;
                c.transform.position = root.position + root.right * (-1.8f + i * 1.2f) + Vector3.up * 2.4f + root.forward * 1.2f;
                c.transform.rotation = root.rotation * Quaternion.Euler(SwordCands[i]);
            }
            Debug.Log($"[StowTune] lineup spawned: shields row y=1.6, swords row y=3.0, left->right: " +
                      $"shields {string.Join(" | ", ShieldCands)} swords {string.Join(" | ", SwordCands)}");
        }

        [MenuItem("DeepUnity/ChatDemo3D/StowTune - Lineup Clear")]
        public static void LineupClear()
        {
            var g = GameObject.Find("__stowLineup");
            if (g != null) Object.DestroyImmediate(g);
        }

        // ---- in-hand sword orientation: pre-rotate the sword inside the Weapon.R mount ----
        static readonly Vector3 HandSwordEuler = new Vector3(0f, -45f, 0f);
        static Quaternion handOrig; static bool handStored;

        [MenuItem("DeepUnity/ChatDemo3D/StowTune - Hand Sword Apply")]
        public static void HandApply()
        {
            if (!Find(out _, out _, out _, out var sword, out _)) return;
            if (!handStored) { handOrig = sword.localRotation; handStored = true; }
            sword.localRotation = Quaternion.Euler(HandSwordEuler) * handOrig;
            Debug.Log($"[StowTune] hand sword local pre-rot {HandSwordEuler} applied");
        }

        [MenuItem("DeepUnity/ChatDemo3D/StowTune - Hand Sword Reset")]
        public static void HandReset()
        {
            if (!handStored) return;
            if (!Find(out _, out _, out _, out var sword, out _)) return;
            sword.localRotation = handOrig; handStored = false;
        }

        [MenuItem("DeepUnity/ChatDemo3D/StowTune - Verify Wiring")]
        public static void Verify()
        {
            var pc = Object.FindObjectOfType<SoulsPlayerController>();
            var st = pc != null ? pc.GetComponent<WeaponStower>() : null;
            if (st == null) { Debug.LogError("[StowTune] VERIFY FAIL: no WeaponStower on player"); return; }
            var so = new SerializedObject(st);
            Debug.Log($"[StowTune] VERIFY OK: sword={so.FindProperty("sword").objectReferenceValue} " +
                      $"shield={so.FindProperty("shield").objectReferenceValue} " +
                      $"waist={so.FindProperty("waistPosition").vector3Value}/{so.FindProperty("waistEuler").vector3Value} " +
                      $"back={so.FindProperty("backPosition").vector3Value}/{so.FindProperty("backEuler").vector3Value}");
        }

        [MenuItem("DeepUnity/ChatDemo3D/StowTune - Restore Hands")]
        public static void Restore()
        {
            if (!Find(out _, out _, out _, out var sword, out var shield)) return;
            if (swordHome != null) { sword.SetParent(swordHome, true); sword.localPosition = swordPos; sword.localRotation = swordRot; }
            if (shieldHome != null) { shield.SetParent(shieldHome, true); shield.localPosition = shieldPos; shield.localRotation = shieldRot; }
            swordHome = shieldHome = null;
            Debug.Log("[StowTune] weapons restored to hands");
        }

        static void Place(Transform item, Transform root, Transform bone, Vector3 rootPos, Vector3 rootEuler)
        {
            item.SetParent(bone, true);   // rides the bone so poseAnimators keeps it attached
            item.position = root.TransformPoint(rootPos);
            item.rotation = root.rotation * Quaternion.Euler(rootEuler);
        }

        static bool Find(out Transform root, out Transform hips, out Transform chest, out Transform sword, out Transform shield)
        {
            root = hips = chest = sword = shield = null;
            var pc = Object.FindObjectOfType<SoulsPlayerController>();
            if (pc == null) { Debug.LogError("[StowTune] no SoulsPlayerController in open scene"); return false; }
            root = pc.transform;
            var anim = pc.GetComponentInChildren<Animator>();
            hips = anim.GetBoneTransform(HumanBodyBones.Hips);
            chest = anim.GetBoneTransform(HumanBodyBones.Chest);
            if (chest == null) chest = anim.GetBoneTransform(HumanBodyBones.Spine);
            sword = FindDeep(root, "Sword");
            shield = FindDeep(root, "Shield");
            if (sword == null || shield == null) { Debug.LogError("[StowTune] Sword/Shield not found under player"); return false; }
            // the scene ships empty-handed since the gear beat (PlayerGear deactivates both until
            // Velmire hands them over), and inactive objects render nothing — every menu item here
            // poses that gear for a screenshot, so force it visible. A rebuild puts it back off.
            sword.gameObject.SetActive(true);
            shield.gameObject.SetActive(true);
            return true;
        }

        static Transform FindDeep(Transform t, string name)
        {
            if (t.name == name) return t;
            foreach (Transform c in t) { var r = FindDeep(c, name); if (r != null) return r; }
            return null;
        }
    }
}
#endif
