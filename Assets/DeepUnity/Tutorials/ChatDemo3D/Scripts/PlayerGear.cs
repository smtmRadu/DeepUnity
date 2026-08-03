using System;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// The player's gear ownership for the ChatDemo3D sale beat: the warrior starts EMPTY-HANDED and
    /// only ever gets a sword and shield because Velmire sells his own and the player presses Accept
    /// on his GiveItem offer (<see cref="NPCGearOffer"/>). Sword and shield are still built, posed and
    /// referenced by the scene builder exactly as before — they simply start deactivated, so the
    /// hand-tuned held poses and the <see cref="WeaponStower"/> wiring survive untouched.
    /// <para>This component is also the single source of truth the NPC's internal CheckMyGear tool
    /// reads, which is what lets the model stay honest about the gear across a context compaction:
    /// the fact lives in the world, not in the transcript.</para>
    /// </summary>
    public class PlayerGear : MonoBehaviour
    {
        [SerializeField, Tooltip("The held sword — deactivated until the player owns it.")]
        private Transform sword;
        [SerializeField, Tooltip("The held shield — deactivated until the player owns it.")]
        private Transform shield;

        [Header("HUD quick slots (icons hide, the empty frames stay)")]
        [SerializeField] private GameObject swordSlotIcon;
        [SerializeField] private GameObject shieldSlotIcon;

        [Header("Start state")]
        [SerializeField, Tooltip("Uncheck to start the demo already armed (the pre-gear-beat behaviour).")]
        private bool startEmptyHanded = true;

        WeaponStower stower;

        public bool HasSword { get; private set; }
        public bool HasShield { get; private set; }

        /// <summary>Fired after ownership changes, so the animation set can swap the moment the gear
        /// lands instead of polling for it every frame (<see cref="SoulsPlayerController"/> listens).
        /// The hand-over happens MID-CONVERSATION, so there is no scene reload to hide it behind.</summary>
        public event Action GearChanged;

        void Awake()
        {
            stower = GetComponent<WeaponStower>();
            if (!startEmptyHanded)
            {
                // the scene builder ships the objects and the quick-slot icons DEACTIVATED, so
                // starting armed has to switch them back on — otherwise this flag hands out the armed
                // animation set (which now keys off HasSword/HasShield) around gear nobody can see.
                HasSword = sword != null;
                HasShield = shield != null;
                if (sword != null) sword.gameObject.SetActive(true);
                if (shield != null) shield.gameObject.SetActive(true);
                if (swordSlotIcon != null) swordSlotIcon.SetActive(HasSword);
                if (shieldSlotIcon != null) shieldSlotIcon.SetActive(HasShield);
                return;
            }
            HasSword = HasShield = false;
            if (sword != null) sword.gameObject.SetActive(false);
            if (shield != null) shield.gameObject.SetActive(false);
            if (swordSlotIcon != null) swordSlotIcon.SetActive(false);
            if (shieldSlotIcon != null) shieldSlotIcon.SetActive(false);
        }

        /// <summary>Hands the player the sword and shield: the held objects come alive, the two quick
        /// slots fill, and (mid-conversation, which is the only way this fires in game) the gear snaps
        /// to the stow sockets so it is drawn when the dialogue closes. Idempotent.</summary>
        /// <param name="stowed">False draws straight into the hands — for testing outside a dialogue.</param>
        public void GrantSwordAndShield(bool stowed = true)
        {
            if (HasSword && HasShield) return;
            if (sword != null) { sword.gameObject.SetActive(true); HasSword = true; }
            if (shield != null) { shield.gameObject.SetActive(true); HasShield = true; }
            if (swordSlotIcon != null) swordSlotIcon.SetActive(true);
            if (shieldSlotIcon != null) shieldSlotIcon.SetActive(true);
            if (stowed) stower?.StowInstant(); else stower?.Draw();
            GearChanged?.Invoke();
        }

        [ContextMenu("Debug/Grant sword and shield")]
        void DebugGrant() => GrantSwordAndShield(stowed: false);
    }
}
