using System;
using TMPro;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// The player's purse for the ChatDemo3D sale beat: souls are what Velmire's sword costs, so the
    /// demo needs somewhere for the number to live and somewhere for it to be seen. This is both —
    /// it owns the count AND writes itself into its HUD label, the same way <see cref="PlayerGear"/>
    /// owns the two quick-slot icons rather than having the HUD reach in for them.
    /// <para>It is the single source of truth behind the NPC's GiveTool accept-gate (can the player
    /// afford this offer?) and behind the deduction when they accept — see
    /// <see cref="NPCGearOffer"/>, which reads it through <see cref="NPCChatBase.ToolGiveAcceptGate"/>
    /// and spends it in <see cref="NPCChatBase.ToolGiveAccepted"/>. The model never touches it: it can
    /// name a price, and that is all.</para>
    /// </summary>
    public class PlayerSouls : MonoBehaviour
    {
        [SerializeField, Tooltip("Souls in hand. 100 in the demo — enough for Velmire's 80-soul asking price, and not enough twice over, so haggling him down actually matters.")]
        private int souls = 100;

        [Header("HUD (the label this purse writes itself into)")]
        [SerializeField, Tooltip("Top-left counter. Written on Awake and on every change; null = no counter in this scene.")]
        private TMP_Text hudLabel;

        public int Souls => souls;

        /// <summary>Fired after the count changes, for anything that is not the label above.</summary>
        public event Action SoulsChanged;

        void Awake() => Render();

        /// <summary>Can this price be paid right now? A price of zero or less is free, which is what a
        /// GiveTool call with no price at all comes through as.</summary>
        public bool CanAfford(int price) => price <= souls;

        /// <summary>Pay, and report whether it went through. Refuses rather than going negative — the
        /// accept-gate should already have hidden this case, but the transaction is the authority.</summary>
        public bool TrySpend(int price)
        {
            if (price > souls) return false;
            if (price <= 0) return true;
            souls -= price;
            Render();
            SoulsChanged?.Invoke();
            return true;
        }

        void Render()
        {
            if (hudLabel != null) hudLabel.text = souls + " souls";
        }
    }
}
