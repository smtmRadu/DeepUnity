using TMPro;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// The walk-up interaction prompt ("Speak — [ I ]") as its OWN component on its OWN GameObject —
    /// the NPC no longer toggles a bare GameObject; it calls Show()/Hide()/HideInstant() and this
    /// component owns the presentation: fade in/out through a CanvasGroup (added at runtime) and a
    /// gentle idle bob. It carries its own knobs (text, fade times, bob) so each demo can restyle
    /// its prompt without touching NPC code.
    ///
    /// Works on a screen-space rect (bobs anchoredPosition) or any world object (bobs localPosition).
    /// The builders save the prompt INACTIVE in the scene: Show() activates it first, and after a
    /// fade-out it deactivates itself again (zero cost while hidden). The CanvasGroup is added in
    /// Awake, not at build time, so edit-mode tools that just SetActive(true) it (screenshot probes)
    /// still see it fully opaque.
    /// </summary>
    public class NPCInteractPrompt : MonoBehaviour
    {
        [Tooltip("Prompt label. Applied to the child TMP text on Awake when non-empty; leave empty to keep whatever text the label was built with.")]
        [SerializeField] string promptText = "";
        [Tooltip("Seconds to fade in on Show().")]
        [Min(0f)] [SerializeField] float fadeInSeconds = 0.15f;
        [Tooltip("Seconds to fade out on Hide(); the GameObject deactivates once fully transparent.")]
        [Min(0f)] [SerializeField] float fadeOutSeconds = 0.12f;
        [Tooltip("Idle bob amplitude while visible (canvas px for UI rects, meters for world objects). 0 = static.")]
        [Min(0f)] [SerializeField] float bobAmplitude = 5f;
        [Tooltip("Bob speed (oscillations scale with this; ~2.4 reads as a calm invite).")]
        [Min(0f)] [SerializeField] float bobSpeed = 2.4f;

        CanvasGroup group;
        RectTransform rect;
        Vector3 basePos;
        bool baseCached;
        float target;   // 0 = hidden, 1 = shown

        public bool Visible => target > 0.5f && gameObject.activeSelf;

        void Awake()
        {
            group = GetComponent<CanvasGroup>();
            if (group == null) group = gameObject.AddComponent<CanvasGroup>();
            group.blocksRaycasts = false;
            group.interactable = false;
            rect = transform as RectTransform;
            CacheBase();
            if (!string.IsNullOrEmpty(promptText))
            {
                var label = GetComponentInChildren<TMP_Text>(true);
                if (label != null) label.text = promptText;
            }
        }

        void CacheBase()
        {
            if (baseCached) return;
            basePos = rect != null ? (Vector3)rect.anchoredPosition : transform.localPosition;
            baseCached = true;
        }

        /// <summary>Fade the prompt in (activates the GameObject if the scene stored it inactive).</summary>
        public void Show()
        {
            if (!gameObject.activeSelf)
            {
                gameObject.SetActive(true);          // Awake runs here on first activation
                if (group != null) group.alpha = 0f; // start the fade from transparent
            }
            target = 1f;
        }

        /// <summary>Fade the prompt out; it deactivates itself once fully transparent.</summary>
        public void Hide() => target = 0f;

        /// <summary>Hide with no fade (scene start / an already-open dialogue).</summary>
        public void HideInstant()
        {
            target = 0f;
            if (group != null) group.alpha = 0f;
            RestoreBase();
            gameObject.SetActive(false);
        }

        void Update()
        {
            float a = group.alpha;
            if (!Mathf.Approximately(a, target))
            {
                float dur = target > a ? fadeInSeconds : fadeOutSeconds;
                a = dur <= 0f ? target : Mathf.MoveTowards(a, target, Time.deltaTime / dur);
                group.alpha = a;
            }
            if (target <= 0f && a <= 0f) { HideInstant(); return; }

            if (bobAmplitude > 0f)
            {
                Vector3 p = basePos + Vector3.up * (Mathf.Sin(Time.time * bobSpeed * Mathf.PI) * bobAmplitude * a);
                if (rect != null) rect.anchoredPosition = p;
                else transform.localPosition = p;
            }
        }

        void RestoreBase()
        {
            if (!baseCached) return;
            if (rect != null) rect.anchoredPosition = basePos;
            else transform.localPosition = basePos;
        }
    }
}
