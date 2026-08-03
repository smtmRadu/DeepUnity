using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Events;

namespace DeepUnity
{
    /// <summary>
    /// Which interactive tool produced a decision. The two are filed under ONE concept on purpose:
    /// from the host game's side "the player picked an option" and "the player pressed Accept" are the
    /// same event — a decision, with an id — and a game that had to subscribe to two unrelated APIs
    /// with two unrelated keying stories is exactly what the binding table replaces.
    /// </summary>
    public enum NPCDecisionKind
    {
        /// <summary>An AskUserQuestion pick. <c>subject</c> is the question, <c>picked</c> the option.</summary>
        Question,
        /// <summary>A GiveItem offer. <c>subject</c> is the item; <c>picked</c> is null.</summary>
        GiveItem,
    }

    /// <summary>
    /// One settled (or pending) player decision, as the host game reads it. The payload of
    /// <see cref="NPCDecision.onResolved"/> and the argument an <see cref="INPCDecisionGate"/> judges.
    /// </summary>
    public struct NPCDecisionResult
    {
        /// <summary>The binding this decision resolved to, or NULL when nothing matched. Null is not
        /// an error: the global hooks still fire and the engine logs which subject went unbound.</summary>
        public string decisionId;
        /// <summary>Which tool this came from — the two fields below read differently per kind.</summary>
        public NPCDecisionKind kind;
        /// <summary>What the decision was ABOUT, in the model's own words, and the string the binding
        /// table is matched against: the QUESTION text for a question, the ITEM for an offer.</summary>
        public string subject;
        /// <summary>Question: the option the player clicked, exactly as the model worded it — branch on
        /// it inside the handler. GiveItem: null (the offer's answer is <see cref="accepted"/>).</summary>
        public string picked;
        /// <summary>GiveItem: which button was pressed. Question: always true — an option was picked,
        /// and WHICH one is <see cref="picked"/>.
        /// <para>In an <c>onResolved</c> payload for a GiveItem this is therefore always true, because
        /// a decline invokes nothing (same contract as the global
        /// <see cref="NPCChatBase.GiveItemAccepted"/>). It is the live button state in the
        /// <c>pending</c> result an <see cref="INPCDecisionGate"/> is handed, where it is still false.</para></summary>
        public bool accepted;
        /// <summary>GiveItem only: the price the NPC named, or null when the offer carried none (a
        /// gift). Null on a question.</summary>
        public int? price;
        /// <summary>GiveItem only: how many, or null when the NPC did not say. Null on a question.</summary>
        public int? quantity;

        public override string ToString()
            => $"{kind} \"{subject}\" → {(decisionId ?? "<unbound>")}"
             + (picked != null ? $" picked \"{picked}\"" : "")
             + (kind == NPCDecisionKind.GiveItem ? $" accepted={accepted}" : "")
             + (price.HasValue ? $" @{price.Value}" : "")
             + (quantity.HasValue ? $" x{quantity.Value}" : "");
    }

    /// <summary>A UnityEvent carrying an <see cref="NPCDecisionResult"/>. It has to be a named
    /// SUBCLASS rather than <c>UnityEvent&lt;NPCDecisionResult&gt;</c> used inline: Unity only
    /// serializes (and only draws in the inspector) a concrete, [Serializable] event type.</summary>
    [Serializable] public class NPCDecisionEvent : UnityEvent<NPCDecisionResult> { }

    /// <summary>
    /// One row of an NPC's decision-binding table: a stable id the designer authors, the words the
    /// NPC might actually use for it, what to run when it resolves, and (offers only) who decides
    /// whether the player may accept it at all.
    /// </summary>
    [Serializable]
    public class NPCDecision
    {
        [Tooltip("Stable, designer-authored key for this decision — \"sell_sword\", \"open_gate\". It is what the game branches on, so it must NOT read like something the model would generate: the whole point is that it never changes when the NPC rewords itself.")]
        public string id;

        [Tooltip("How the NPC might word this decision's SUBJECT — the item for a GiveItem offer (\"sword\", \"arming sword\", \"blade\"), the question text for an AskUserQuestion. Matched case-, whitespace-, punctuation- and article-insensitively, exactly first, then by either string containing the other. The id itself is always matched too, so an id that reads naturally needs no alias.")]
        public string[] aliases;

        [Tooltip("Fires when this binding resolves, BEFORE the decision goes back to the model — so the world is already updated when the reply streams. Must not block. For an offer it fires only on Accept, exactly like the global GiveItemAccepted hook.")]
        public NPCDecisionEvent onResolved = new NPCDecisionEvent();

        [Tooltip("Optional per-binding accept-gate for GiveItem offers: a Component implementing INPCDecisionGate that answers \"can the player take THIS one?\" (enough souls, room in the pack). When set it decides instead of the NPC's global GiveItemAcceptGate. Unused for questions.")]
        public Component gate;
    }

    /// <summary>
    /// A per-binding accept-gate. Implement it on any Component and drop that component into a
    /// binding's <see cref="NPCDecision.gate"/> slot; it is asked ONCE, when the offer panel opens,
    /// and false renders Accept disabled (Decline is never gated, so an exchange cannot dead-end).
    /// <para>Same contract as <see cref="NPCChatBase.GiveItemAcceptGate"/>: it is a PRESENTATION hint,
    /// not the transaction — the authoritative check belongs in whatever performs the hand-over — and a
    /// gate that throws is reported and read as "yes".</para>
    /// </summary>
    public interface INPCDecisionGate
    {
        /// <summary>Can the player accept <paramref name="pending"/>? Its <c>accepted</c> field is
        /// still false: nothing has been pressed yet.</summary>
        bool CanAccept(NPCDecisionResult pending);
    }

    /// <summary>
    /// The ONE resolver both interactive tool paths run a decision's subject through, so a question
    /// and an offer are bound by identical rules. Pure and static: no scene, no NPC, no Unity state —
    /// which is what lets <c>NpcGiveItemProbe</c> assert the matching rules headlessly.
    /// </summary>
    public static class NPCDecisionTable
    {
        /// <summary>
        /// Comparison form of an id / alias / subject: trimmed, lowercased, internal whitespace
        /// collapsed to single spaces, surrounding punctuation stripped, and a leading article
        /// (a / an / the) dropped. So "The Arming Sword." , "  arming   sword " and "an arming sword"
        /// are one string, while nothing INSIDE the words is touched.
        /// </summary>
        public static string Normalize(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return "";
            var sb = new StringBuilder(s.Length);
            bool pendingSpace = false;
            foreach (char raw in s)
            {
                char c = char.ToLowerInvariant(raw);
                if (char.IsWhiteSpace(c)) { pendingSpace = sb.Length > 0; continue; }
                if (pendingSpace) { sb.Append(' '); pendingSpace = false; }
                sb.Append(c);
            }
            string t = sb.ToString();
            // surrounding punctuation only — an apostrophe or a hyphen INSIDE a word is part of it
            // ("Velmire's sword", "take-it-or-leave-it")
            t = t.Trim(Surround);
            // leading article, then re-trim: "the  sword." has already collapsed to "the sword"
            foreach (string art in Articles)
            {
                if (!t.StartsWith(art, StringComparison.Ordinal)) continue;
                t = t.Substring(art.Length).Trim(Surround);
                break;
            }
            return t;
        }

        static readonly char[] Surround = { ' ', '.', ',', ';', ':', '!', '?', '"', '\'', '`',
                                            '(', ')', '[', ']', '{', '}', '-', '_', '*', '…' };
        // with the trailing space: "theatre" must not lose its "the"
        static readonly string[] Articles = { "the ", "an ", "a " };

        /// <summary>
        /// The binding <paramref name="subject"/> resolves to, or null when nothing matches.
        /// <para>Tried in order, and the FIRST tier that hits anything decides — so a weaker rule can
        /// never outrank a stronger one:</para>
        /// <list type="number">
        /// <item>exact match on the binding's own <c>id</c>;</item>
        /// <item>exact match on one of its <c>aliases</c>;</item>
        /// <item>an alias contained in the subject, or the subject contained in an alias — this is what
        /// catches "this old blade of mine" for the alias "blade".</item>
        /// </list>
        /// Within a tier the FIRST DECLARED binding wins, always, so the outcome depends on the table's
        /// order and on nothing else; when more than one could have hit,
        /// <paramref name="ambiguity"/> comes back describing the clash so the caller can log it.
        /// Rows with a blank id and blank aliases are skipped (an empty row must not match everything).
        /// </summary>
        public static NPCDecision Resolve(IList<NPCDecision> decisions, string subject, out string ambiguity)
        {
            ambiguity = null;
            if (decisions == null || decisions.Count == 0) return null;
            string subj = Normalize(subject);
            if (subj.Length == 0) return null;

            NPCDecision hit = null;
            List<string> also = null;

            for (int tier = 0; tier < 3 && hit == null; tier++)
            {
                for (int i = 0; i < decisions.Count; i++)
                {
                    NPCDecision d = decisions[i];
                    if (d == null || !Matches(d, subj, tier)) continue;
                    if (hit == null) { hit = d; continue; }
                    (also ?? (also = new List<string>())).Add(Label(d, i));
                }
            }
            if (hit != null && also != null)
                ambiguity = $"\"{subject}\" also matches {string.Join(", ", also)} — " +
                            $"took \"{hit.id}\" because it is declared first. Tighten the aliases if " +
                            "that is not the one you meant.";
            return hit;
        }

        static string Label(NPCDecision d, int index)
            => string.IsNullOrWhiteSpace(d.id) ? $"decision #{index}" : $"\"{d.id}\"";

        static bool Matches(NPCDecision d, string subj, int tier)
        {
            if (tier == 0) return Normalize(d.id) == subj;
            if (d.aliases == null) return false;
            foreach (string a in d.aliases)
            {
                string n = Normalize(a);
                if (n.Length == 0) continue;
                if (tier == 1) { if (n == subj) return true; }
                else if (subj.Contains(n) || n.Contains(subj)) return true;
            }
            return false;
        }

        /// <summary>Every declared id, comma-joined — what the unmatched warning lists so a designer can
        /// see at a glance what the subject was compared against. "(none declared)" on an empty table.</summary>
        public static string DeclaredIds(IList<NPCDecision> decisions)
        {
            if (decisions == null || decisions.Count == 0) return "(none declared)";
            var ids = new List<string>(decisions.Count);
            for (int i = 0; i < decisions.Count; i++)
                if (decisions[i] != null) ids.Add(Label(decisions[i], i));
            return ids.Count == 0 ? "(none declared)" : string.Join(", ", ids);
        }
    }
}
