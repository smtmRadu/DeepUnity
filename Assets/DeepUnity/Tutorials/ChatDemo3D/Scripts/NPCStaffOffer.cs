using System;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Corvus' staff SALE — Velmire's sword loop (see <see cref="NPCGearOffer"/>, the full
    /// model-decides / engine-executes story) running on the second NPC: the same internal
    /// <c>CheckMyGear</c> read with staff-flavoured returns, and the same GiveItem decision
    /// binding under the id <c>sell_staff</c>. The model may offer; only the player's Accept
    /// press pays the 20 souls and lands the staff via <see cref="PlayerGear.GrantStaff"/> —
    /// which is what arms <see cref="GlintstoneStaff"/>.
    /// </summary>
    [DisallowMultipleComponent]
    public class NPCStaffOffer : MonoBehaviour, INPCToolProvider, INPCDecisionGate
    {
        /// <summary>The binding id ChatDemo3DBuilder authors on Corvus' decision table.</summary>
        public const string SellStaffDecisionId = "sell_staff";

        [SerializeField, Tooltip("The player's gear ownership — read by the tool, written on accept.")]
        private PlayerGear playerGear;
        [SerializeField, Tooltip("The player's purse — the accept-gate reads it, the hand-over spends it. Null = the offer is free.")]
        private PlayerSouls playerSouls;
        [SerializeField, Tooltip("The staff in the NPC's own model — hidden the moment the player takes it.")]
        private Transform npcStaff;

        bool gaveMine;

        // Same trained name as Velmire's read (the corpus knows CheckMyGear); only the returns
        // are staff-flavoured. Terse on purpose — it is spliced into a 1024-token context.
        const string CheckMyGearSchema =
            "{\"type\": \"function\", \"function\": {\"name\": \"CheckMyGear\", \"description\": " +
            "\"Look at whether you still carry your staff, and whether the player already owns one. " +
            "Call this before offering it. Returns you_have_staff, already_given_away, player_has_staff. " +
            "The player never sees this call.\", " +
            "\"parameters\": {\"type\": \"object\", \"properties\": {}}}}";

        public IEnumerable<string> ToolSchemas
        {
            get { yield return CheckMyGearSchema; }
        }

        public string TryHandleTool(string toolName, string argumentsJson)
        {
            if (!"CheckMyGear".Equals(toolName, StringComparison.OrdinalIgnoreCase)) return null;
            bool playerHas = playerGear != null && playerGear.HasStaff;
            return "{\"you_have_staff\": " + Bool(!gaveMine)
                 + ", \"already_given_away\": " + Bool(gaveMine)
                 + ", \"player_has_staff\": " + Bool(playerHas) + "}";
        }

        static string Bool(bool b) => b ? "true" : "false";

        /// <summary><see cref="INPCDecisionGate"/>: souls in hand must cover the price he named.</summary>
        public bool CanAccept(NPCDecisionResult pending)
            => !pending.price.HasValue || playerSouls == null || playerSouls.CanAfford(pending.price.Value);

        /// <summary>The hand-over, as the <c>sell_staff</c> binding's onResolved target — payment
        /// re-checked here because THIS is the transaction; the gate was only the button state.</summary>
        public void OnDecisionResolved(NPCDecisionResult decision)
        {
            if (!decision.accepted) return;
            if (decision.price.HasValue && playerSouls != null
                && !playerSouls.TrySpend(decision.price.Value))
            {
                ConsoleMessage.Warning($"[Staff] {name}: accepted at {decision.price.Value} but the purse only " +
                                       $"holds {playerSouls.Souls} — nothing handed over.");
                return;
            }
            if (playerGear == null)
            {
                ConsoleMessage.Warning($"[Staff] {name}: no PlayerGear wired — nothing to hand over.");
                return;
            }
            if (gaveMine) return;
            playerGear.GrantStaff();
            if (npcStaff != null) npcStaff.gameObject.SetActive(false);
            gaveMine = true;
            ConsoleMessage.Info($"[Staff] {name}: staff handed to the player — CheckMyGear now reports already_given_away.");
        }
    }
}
