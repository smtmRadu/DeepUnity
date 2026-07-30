using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Velmire's gear beat — the demo's full model-decides / engine-executes loop, in one component:
    /// <list type="number">
    /// <item>an INTERNAL tool, <c>GetPlayerGear</c>, that reports what the player carries and whether
    /// this NPC has already given his own away. It is a READ, so it is ungated: the model may call it
    /// freely, and its prompt tells it to call it before offering anything. Because the answer comes
    /// from the world and not from the transcript, the NPC still knows he gave his sword away after a
    /// context compaction has wiped the conversation that mentioned it.</item>
    /// <item>the GATED ACTION: the gear only changes hands when
    /// <see cref="NPCChatBase.ToolQuestionAnswered"/> reports that the PLAYER accepted the offer the
    /// NPC put up through AskUserQuestion. The model can offer; it cannot give.</item>
    /// </list>
    /// Accept/refuse is read out of the picked option's own words, because the NPC writes the options
    /// himself in character. Refusals are tested FIRST — "No, keep your sword" contains "keep" AND
    /// "sword", and a false accept would hand over gear the player just declined.
    /// </summary>
    [DisallowMultipleComponent]
    public class NPCGearOffer : MonoBehaviour, INPCToolProvider
    {
        [SerializeField, Tooltip("The player's gear ownership — read by the tool, written on accept.")]
        private PlayerGear playerGear;

        [Header("This NPC's own carried copies (hidden the moment the player takes them)")]
        [SerializeField] private Transform npcSword;
        [SerializeField] private Transform npcShield;

        // NOTHING here makes the model CALL a tool — the model decides that entirely on its own, from
        // the schemas and rules in its prompt. These words are only read AFTERWARDS, to interpret the
        // option the player clicked, because the model words its own options and a click of
        // "Keep your steel" has to be told apart from a click of "Take them" somehow.
        [Header("Reading the player's pick (the NPC words the options himself)")]
        [SerializeField, Tooltip("Fallback recognition only: if the NPC never called CheckMyGear in this exchange, the question must mention one of these for the pick to count as a gear offer.")]
        private string[] offerKeywords = { "sword", "shield", "blade", "weapon", "weapons", "arms", "steel", "gear" };
        [SerializeField, Tooltip("Checked FIRST — any of these in the picked option means refused.")]
        private string[] refuseKeywords = { "no", "nay", "not", "never", "keep", "keep them", "refuse", "decline", "leave", "nothing" };
        [SerializeField, Tooltip("Only then: any of these in the picked option means accepted.")]
        private string[] acceptKeywords = { "yes", "aye", "take", "take them", "accept", "i will", "gladly", "please", "thank you", "give" };

        NPCChatBase npc;
        bool gaveMine;
        // Set when the model actually looked at his belt through CheckMyGear. That is a REAL signal
        // from the engine — this provider served the call — so when the pipeline runs as intended
        // (check, then offer) the hand-over no longer depends on how he happened to word the question.
        bool checkedGearRecently;

        void Awake() => npc = GetComponent<NPCChatBase>();

        void OnEnable()
        {
            if (npc != null) npc.ToolQuestionAnswered += OnToolQuestionAnswered;
        }

        void OnDisable()
        {
            if (npc != null) npc.ToolQuestionAnswered -= OnToolQuestionAnswered;
        }

        // ------------------------------------------------------------------ INPCToolProvider

        // Named and worded from the NPC's OWN point of view — "check what I have to give" — because that
        // is the sentence the model narrates before calling it, and the previous player-centric name
        // (GetPlayerGear) never read as a reason to look at his own belt (user 2026-07-25: "it forgot to
        // check if itself has the sword to give"). Deliberately terse otherwise: it is spliced into the
        // prompt of an NPC on a 1200-token context, and the persona is where the WHEN lives.
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
            checkedGearRecently = true;
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

        void OnToolQuestionAnswered(string question, string picked)
        {
            if (playerGear == null || gaveMine) return;
            if (playerGear.HasSword && playerGear.HasShield) return;
            // Is this pick OURS? First choice: he looked at his own belt through CheckMyGear during this
            // exchange, which is a fact the engine handed us, not a guess about wording. Only if he
            // skipped the check do we fall back to reading the question — and since the question is
            // impersonal by convention ("Take Velmire's sword and shield?"), it still names the gear.
            if (!checkedGearRecently && !Mentions(question, offerKeywords)) return;
            if (Mentions(picked, refuseKeywords))
            {
                ConsoleMessage.Info($"[Gear] {name}: offer declined (\"{picked}\") — gear stays with him.");
                checkedGearRecently = false;
                return;
            }
            if (!Mentions(picked, acceptKeywords))
            {
                ConsoleMessage.Warning($"[Gear] {name}: could not read \"{picked}\" as accept or refuse — " +
                                       "nothing handed over. Add the wording to acceptKeywords/refuseKeywords.");
                return;
            }
            Grant();
        }

        void Grant()
        {
            checkedGearRecently = false;
            playerGear.GrantSwordAndShield();
            if (npcSword != null) npcSword.gameObject.SetActive(false);
            if (npcShield != null) npcShield.gameObject.SetActive(false);
            gaveMine = true;
            ConsoleMessage.Info($"[Gear] {name}: sword and shield handed to the player — " +
                                "GetPlayerGear now reports you_already_gave_yours.");
        }

        /// <summary>Whole-word / whole-phrase match on a lowercased, punctuation-stripped copy, so
        /// "Yes, deal." matches "yes" while "nothing" does not match "no".</summary>
        static bool Mentions(string text, string[] keywords)
        {
            if (string.IsNullOrWhiteSpace(text) || keywords == null) return false;
            var sb = new StringBuilder(text.Length + 2).Append(' ');
            foreach (char c in text)
            {
                if (char.IsLetterOrDigit(c)) sb.Append(char.ToLowerInvariant(c));
                else if (sb[sb.Length - 1] != ' ') sb.Append(' ');   // collapse runs so multi-word keys still match
            }
            if (sb[sb.Length - 1] != ' ') sb.Append(' ');
            string haystack = sb.ToString();
            foreach (string k in keywords)
            {
                if (string.IsNullOrWhiteSpace(k)) continue;
                if (haystack.Contains(" " + k.Trim().ToLowerInvariant() + " ")) return true;
            }
            return false;
        }

        [ContextMenu("Debug/Hand the gear over now")]
        void DebugGrant()
        {
            if (playerGear == null) { ConsoleMessage.Warning($"[Gear] {name}: no PlayerGear wired."); return; }
            Grant();
        }
    }
}
