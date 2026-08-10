using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Notification-style speech bubble above a strolling villager's head, revealed IN SYNC with
    /// the voice: each pocket-tts clause arrives with its spoken duration (OnClauseSpoken — the
    /// same event the dialogue window's audio-synced reveal runs on) and types itself out over
    /// ~that window. The text sits on a rounded dark plate with a gold rim (the demo UI's own
    /// palette); the plate POPS in when a line starts and re-fits its height every frame as the
    /// clause wraps onto new lines. Ambient banter only: clauses fired while the owner is in a
    /// real dialogue are ignored (the chat window owns the text then), and the bubble hides a
    /// beat after the voice goes quiet.
    /// </summary>
    public class VillageSpeechBubble : MonoBehaviour
    {
        [SerializeField] private VillageStroller owner;
        [SerializeField] private TMP_Text text;
        [SerializeField] private SpriteRenderer plate;      // translucent rounded rect, sliced
        [SerializeField] private SpriteRenderer plateRim;   // rim: same sprite, slightly larger
        [SerializeField] private SpriteRenderer tail;       // rotated square under the plate — the little arrow to the speaker
        [Tooltip("Seconds of silence before the bubble fades out.")]
        [SerializeField] private float holdAfterQuiet = 1.4f;
        [Tooltip("Beyond this distance from the camera the bubble hides (it would be unreadable anyway).")]
        [SerializeField] private float maxReadDistance = 24f;
        [Tooltip("Text wrap width in meters; the plate hugs shorter lines.")]
        [SerializeField] private float maxWidth = 1.7f;

        readonly Queue<(string clause, float dur)> pending = new Queue<(string, float)>();
        string shown = "";
        string revealing;        // clause currently typing itself out
        float revealT0, revealDur;
        float quietSince = -1f;
        float alpha;
        float pop = 1f;          // 0..1 scale-in progress, restarted on every appearance
        bool visibleNow;
        bool subscribed;
        Color plateBase, rimBase, tailBase, textBase;

        void Start()
        {
            if (plate != null) plateBase = plate.color;
            if (plateRim != null) rimBase = plateRim.color;
            if (tail != null) tailBase = tail.color;
            // the builder authors the text with alpha 0 (hidden until the first clause) — the
            // BASE opacity is full, only the runtime fade owns the alpha
            if (text != null) { textBase = text.color; textBase.a = 1f; }
        }

        void LateUpdate()
        {
            if (!subscribed && owner != null && owner.Voice != null)
            {
                owner.Voice.OnClauseSpoken += OnClause;
                subscribed = true;
            }

            var cam = Camera.main;
            bool tooFar = cam == null ||
                (cam.transform.position - transform.position).sqrMagnitude > maxReadDistance * maxReadDistance;

            // typewriter pacing: reveal a growing prefix of the active clause across its spoken
            // duration (elapsed-clock, so a hitched frame catches up instead of lagging behind)
            if (revealing == null && pending.Count > 0)
            {
                (revealing, revealDur) = pending.Dequeue();
                revealT0 = Time.realtimeSinceStartup;
                if (shown.Length > 0) shown += " ";
            }
            if (revealing != null)
            {
                float t = Mathf.Clamp01((Time.realtimeSinceStartup - revealT0) / Mathf.Max(0.05f, revealDur * 0.92f));
                int chars = Mathf.CeilToInt(revealing.Length * t);
                // snap DOWN to the last completed word — a notification board reading "Thi" is
                // worse than one running a beat behind the voice (user 2026-08-10)
                if (chars < revealing.Length)
                {
                    int cut = revealing.LastIndexOf(' ', Mathf.Clamp(chars - 1, 0, revealing.Length - 1));
                    chars = cut >= 0 ? cut : 0;
                }
                if (text != null) text.text = shown + revealing.Substring(0, chars);
                if (chars >= revealing.Length)
                {
                    shown += revealing;
                    revealing = null;
                }
                quietSince = -1f;
            }
            else if (shown.Length > 0)
            {
                bool audible = owner != null && owner.VoiceBusy;
                if (audible) quietSince = -1f;
                else if (quietSince < 0f) quietSince = Time.time;
                else if (Time.time - quietSince > holdAfterQuiet) Clear();
            }

            bool visible = (revealing != null || pending.Count > 0 || shown.Length > 0) && !tooFar
                           && (owner == null || !owner.InConversation);
            if (visible && !visibleNow) pop = 0f;   // fresh appearance: play the pop
            visibleNow = visible;

            alpha = Mathf.MoveTowards(alpha, visible ? 1f : 0f, Time.deltaTime * 6f);
            pop = Mathf.MoveTowards(pop, 1f, Time.deltaTime / 0.16f);
            // ease-out-back: overshoots slightly past full size then settles — the notification pop
            float p = pop - 1f;
            float scale = 0.65f + 0.35f * (1f + p * p * (2.70158f * p + 1.70158f));
            transform.localScale = Vector3.one * Mathf.Max(0.01f, scale);

            // fit the plate to the CURRENT text: width hugs short lines, height follows the wrap
            string cur = text != null ? text.text : "";
            if (text != null && plate != null && cur.Length > 0)
            {
                Vector2 pref = text.GetPreferredValues(cur, maxWidth, 0f);
                float w = Mathf.Clamp(pref.x, 0.5f, maxWidth);
                float h = Mathf.Max(0.24f, pref.y);
                text.rectTransform.sizeDelta = new Vector2(maxWidth, h);
                plate.size = new Vector2(w + 0.24f, h + 0.16f);
                if (plateRim != null) plateRim.size = plate.size + new Vector2(0.04f, 0.04f);
                Vector3 c0 = new Vector3(0f, h * 0.5f, 0.012f);
                plate.transform.localPosition = c0;
                if (plateRim != null) plateRim.transform.localPosition = c0 + new Vector3(0, 0, 0.008f);
            }

            if (text != null)
            {
                var c = textBase; c.a = textBase.a * alpha; text.color = c;
                if (plate != null) { var pc = plateBase; pc.a = plateBase.a * alpha; plate.color = pc; }
                if (plateRim != null) { var rc = rimBase; rc.a = rimBase.a * alpha; plateRim.color = rc; }
                if (tail != null) { var tc = tailBase; tc.a = tailBase.a * alpha; tail.color = tc; }
                if (cam != null)
                    transform.rotation = Quaternion.LookRotation(transform.position - cam.transform.position);
            }
        }

        void OnClause(string clause, float duration)
        {
            // dialogue clauses belong to the chat window; the bubble is the AMBIENT channel
            if (owner != null && owner.InConversation) return;
            pending.Enqueue((clause, duration));
        }

        /// <summary>A fresh line starts: wipe whatever the last one left standing.</summary>
        public void BeginUtterance() => Clear();

        public void HideNow()
        {
            Clear();
            alpha = 0f;
            visibleNow = false;
            if (text != null) { var c = text.color; c.a = 0f; text.color = c; }
            if (plate != null) { var c = plate.color; c.a = 0f; plate.color = c; }
            if (plateRim != null) { var c = plateRim.color; c.a = 0f; plateRim.color = c; }
            if (tail != null) { var c = tail.color; c.a = 0f; tail.color = c; }
        }

        void Clear()
        {
            pending.Clear();
            shown = "";
            revealing = null;
            quietSince = -1f;
            if (text != null) text.text = "";
        }

        void OnDestroy()
        {
            if (subscribed && owner != null && owner.Voice != null)
                owner.Voice.OnClauseSpoken -= OnClause;
        }
    }
}
