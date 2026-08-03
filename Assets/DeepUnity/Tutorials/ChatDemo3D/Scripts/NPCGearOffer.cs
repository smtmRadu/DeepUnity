using System;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Velmire's sword SALE — the demo's full model-decides / engine-executes loop, in one component:
    /// <list type="number">
    /// <item>an INTERNAL tool, <c>CheckMyGear</c>, that reports what the NPC carries and whether he has
    /// already handed it over. It is a READ, so it is ungated: the model may call it freely, and its
    /// prompt tells it to look before it offers. Because the answer comes from the world and not from
    /// the transcript, he still knows the sword is gone after a context compaction has wiped the
    /// conversation that mentioned it.</item>
    /// <item>the GATED ACTION, through <b>GiveItem</b> and a DECISION BINDING: the NPC declares a
    /// decision with the id <c>sell_sword</c> (see <see cref="NPCDecision"/>, wired by
    /// ChatDemo3DBuilder and visible on the NPC in the inspector). This component is that binding's
    /// <see cref="INPCDecisionGate"/> — it says whether the player can afford the price the model
    /// named — and its <see cref="OnDecisionResolved"/> is the binding's event: it takes the souls and
    /// gives the gear, only once the PLAYER has pressed Accept. The model can offer; it cannot give,
    /// and it cannot set the player's purse.</item>
    /// </list>
    /// <para>Nothing here reads the NPC's words any more. The old beat ran on AskUserQuestion, so it had
    /// to tell a click of "Take them" from a click of "Keep your steel" with keyword lists — GiveItem
    /// returns a BOOL, which is the whole reason it exists: <c>{"accepted": true}</c> or
    /// <c>{"accepted": false}</c>, no wording to interpret and nothing to get wrong.</para>
    /// <para>It hangs off the BINDING rather than the NPC-wide
    /// <see cref="NPCChatBase.GiveItemAcceptGate"/> / <see cref="NPCChatBase.GiveItemAccepted"/> hooks
    /// (which still exist and still fire) for the reason the table exists at all: those are one gate
    /// and one event per NPC with no key, so the day Velmire also sells a shield they cannot tell the
    /// two offers apart. The binding matches the ITEM the model named against the aliases the designer
    /// authored, and hands this component a decision that already carries the id.</para>
    /// </summary>
    [DisallowMultipleComponent]
    public class NPCGearOffer : MonoBehaviour, INPCToolProvider, INPCDecisionGate
    {
        /// <summary>The binding id this component serves — the string ChatDemo3DBuilder authors on
        /// Velmire's decision table. Public so the builder and the headless probe agree on it by
        /// reference instead of by two copies of a literal.</summary>
        public const string SellSwordDecisionId = "sell_sword";

        [SerializeField, Tooltip("The player's gear ownership — read by the tool, written on accept.")]
        private PlayerGear playerGear;
        [SerializeField, Tooltip("The player's purse — the accept-gate reads it, the hand-over spends it. Null = the offer is free (no gate, no deduction).")]
        private PlayerSouls playerSouls;

        [Header("This NPC's own carried copies (hidden the moment the player takes them)")]
        [SerializeField] private Transform npcSword;
        [SerializeField] private Transform npcShield;

        bool gaveMine;

        // A designer aid, not machinery: this component does nothing unless the NPC's decision table
        // actually points at it, and "nothing happened and nothing was logged" is the failure this
        // whole refactor exists to end. Checked once, on enable.
        void OnEnable()
        {
            var npc = GetComponent<NPCChatBase>();
            if (npc == null || BoundOnAnyDecision(npc)) return;
            ConsoleMessage.Warning($"[Gear] {name}: no decision on this NPC names this component as its " +
                                   "gate or its onResolved target — the sale cannot land. Wire a binding " +
                                   $"(id \"{SellSwordDecisionId}\") in the inspector, or rebuild the scene.");
        }

        bool BoundOnAnyDecision(NPCChatBase npc)
        {
            foreach (var d in npc.Decisions)
            {
                if (d == null) continue;
                if (ReferenceEquals(d.gate, this)) return true;
                if (d.onResolved == null) continue;
                for (int i = 0; i < d.onResolved.GetPersistentEventCount(); i++)
                    if (ReferenceEquals(d.onResolved.GetPersistentTarget(i), this)) return true;
            }
            return false;
        }

        // ------------------------------------------------------------------ INPCToolProvider

        // Named and worded from the NPC's OWN point of view — "check what I have to give" — because that
        // is the sentence the model narrates before calling it, and the previous player-centric name
        // (GetPlayerGear) never read as a reason to look at his own belt (user 2026-07-25: "it forgot to
        // check if itself has the sword to give"). Deliberately terse otherwise: it is spliced into the
        // prompt of an NPC on a small context, and the persona is where the WHEN lives.
        const string CheckMyGearSchema =
            "{\"type\": \"function\", \"function\": {\"name\": \"CheckMyGear\", \"description\": " +
            "\"Look at what weapons and armour you are carrying and could hand over, and whether the player " +
            "already has a weapon. Call this before offering anyone a weapon. Returns you_have_sword, " +
            "you_have_shield, already_given_away, player_has_weapon. The player never sees this call.\", " +
            "\"parameters\": {\"type\": \"object\", \"properties\": {}}}}";

        public IEnumerable<string> ToolSchemas
        {
            get { yield return CheckMyGearSchema; }
        }

        public string TryHandleTool(string toolName, string argumentsJson)
        {
            if (!"CheckMyGear".Equals(toolName, StringComparison.OrdinalIgnoreCase)) return null;
            bool playerArmed = playerGear != null && (playerGear.HasSword || playerGear.HasShield);
            // his own gear is gone the moment he hands it over — the objects are deactivated then
            bool mine = !gaveMine;
            return "{\"you_have_sword\": " + Bool(mine)
                 + ", \"you_have_shield\": " + Bool(mine)
                 + ", \"already_given_away\": " + Bool(gaveMine)
                 + ", \"player_has_weapon\": " + Bool(playerArmed) + "}";
        }

        static string Bool(bool b) => b ? "true" : "false";

        // ------------------------------------------------------------------ the gated action

        /// <summary><see cref="INPCDecisionGate"/>: souls in hand must cover the price he named. A
        /// priceless offer (a gift) and a scene without a purse are both free. This is what the
        /// <c>sell_sword</c> binding's gate slot points at, so the headless probe exercises the SAME
        /// gate the dialogue asks.</summary>
        public bool CanAccept(NPCDecisionResult pending) => CanAfford(pending.price);

        /// <summary>The hand-over, as the binding's <c>onResolved</c> target: take the souls, then give
        /// the gear. Only ever reached from the player's own Accept press, and only after the gate
        /// above said yes — but the payment is re-checked here anyway, because THIS is the transaction
        /// and the gate was only the button state.
        /// <para>Public and single-argument on purpose: that is the signature
        /// <c>UnityEventTools.AddPersistentListener</c> can bind, which is what makes the wiring show
        /// up (and stay editable) in the inspector instead of hiding in a runtime <c>+=</c>.</para></summary>
        public void OnDecisionResolved(NPCDecisionResult decision)
        {
            if (!decision.accepted) return;   // declines never reach here, but the contract is the flag
            if (decision.price.HasValue && playerSouls != null
                && !playerSouls.TrySpend(decision.price.Value))
            {
                ConsoleMessage.Warning($"[Gear] {name}: accepted at {decision.price.Value} but the purse only " +
                                       $"holds {playerSouls.Souls} — nothing handed over.");
                return;
            }
            Grant();
        }

        bool CanAfford(int? price)
        {
            if (!price.HasValue || playerSouls == null) return true;
            return playerSouls.CanAfford(price.Value);
        }

        void Grant()
        {
            if (playerGear == null)
            {
                ConsoleMessage.Warning($"[Gear] {name}: no PlayerGear wired — nothing to hand over.");
                return;
            }
            if (gaveMine) return;
            playerGear.GrantSwordAndShield();
            if (npcSword != null) npcSword.gameObject.SetActive(false);
            if (npcShield != null) npcShield.gameObject.SetActive(false);
            gaveMine = true;
            ConsoleMessage.Info($"[Gear] {name}: sword and shield handed to the player — " +
                                "CheckMyGear now reports already_given_away.");
        }

        [ContextMenu("Debug/Hand the gear over now")]
        void DebugGrant() => Grant();
    }
}
