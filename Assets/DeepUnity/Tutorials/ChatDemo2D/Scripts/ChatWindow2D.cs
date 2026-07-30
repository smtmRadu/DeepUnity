using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials.ChatDemo2D
{
    /// <summary>
    /// Minimal farm-styled dialogue overlay — no chat box. Spoken lines float as
    /// drop-shadowed text near the bottom-center of the screen: each new line pushes the
    /// older ones up, lines drift slowly upward and fade out with height until they fully
    /// vanish near the top of the float region. The only chrome left is a thin 20%-alpha
    /// input strip at the bottom (underline-style input + text-only Say/Leave buttons).
    ///
    /// Derives from <see cref="NPCDialogueWindow"/> like every NPC window, so the send-loading
    /// pulse, caret, UI sounds, title/info, context bar and the AskUserQuestion choice popup are
    /// inherited; only the float presentation lives here.
    ///
    /// Streaming contract: the base pops + re-adds the newest NPC line many times a second
    /// (token stream and audio-synced reveal). PopLastMessage therefore PARKS the newest line
    /// instead of destroying it, and the AddMessage that follows in the same call stack
    /// reclaims it as a pure text mutation — same GameObject, same float state, no
    /// re-animation and no flicker. A parked line that is never reclaimed is destroyed in
    /// LateUpdate, so a standalone pop still removes the line by end of frame. The newest
    /// line stays pinned at the bottom slot until a subsequent AddMessage pushes it up.
    /// </summary>
    public class ChatWindow2D : NPCDialogueWindow
    {
        [Header("Farm overlay (wired by the scene builder)")]
        [SerializeField] private CanvasGroup canvasGroup;       // fades the whole overlay open/closed
        [SerializeField] private RectTransform linesContainer;  // float region: lines spawn at its bottom, die near its top
        [SerializeField] private GameObject lineTemplate;       // one TMP + CanvasGroup per line
        [SerializeField] private Button giveButton;      // hands the basket over; shown only when it has items

        [Header("Float animation")]
        [SerializeField] private float fadeDuration = 0.25f;
        [Tooltip("Slow upward drift of unpinned lines (canvas px/s).")]
        [SerializeField] private float driftSpeed = 24f;
        [Tooltip("Catch-up speed when a new/growing line below pushes the stack up (canvas px/s).")]
        [SerializeField] private float pushCatchupSpeed = 420f;
        [SerializeField] private float lineSpacing = 10f;

        /// <summary>2D-only extra: gives the harvested basket to the NPC (NPCInteractor2D toggles
        /// its visibility per dialogue state + basket contents).</summary>
        public Button GiveButton => giveButton;

        // farm-gold accent shared with the builder — the caret must be unmissable, and the
        // AskUserQuestion popup picks the same accent up through the base class
        private static readonly Color FarmGold = new Color(0.90f, 0.72f, 0.35f);
        protected override Color CaretColor => FarmGold;

        private class Line
        {
            public GameObject go;
            public RectTransform rect;
            public TMP_Text text;
            public CanvasGroup group;
            public float y;   // offset above the bottom slot; newest stays frozen, others drift
        }

        private readonly List<Line> lines = new List<Line>();  // index 0 = oldest, last = newest (pinned)
        private Line recycled;   // newest line parked by PopLastMessage, reclaimed by the next AddMessage
        private Coroutine fadeCoroutine;

        protected override void Awake()
        {
            if (canvasGroup == null) canvasGroup = GetComponent<CanvasGroup>();
            if (canvasGroup == null) canvasGroup = gameObject.AddComponent<CanvasGroup>();
            canvasGroup.alpha = 0f;
            if (lineTemplate != null) lineTemplate.SetActive(false);
            base.Awake();   // input listeners + caret
            gameObject.SetActive(false);
        }

        // ---------------------------------------------------------------- float animation

        protected override void Update()
        {
            base.Update();   // context-fill bar
            if (linesContainer == null || lines.Count == 0) return;

            // Stack maintenance, newest (bottom) to oldest (top). The newest line is frozen at
            // its slot — streaming mutates its text in place, and its growth alone raises the
            // "floor" that pushes every line above it. Unpinned lines drift up slowly and fade
            // with normalized height until they vanish near the top of the float region.
            float region = Mathf.Max(1f, linesContainer.rect.height) * 0.94f;
            float floor = 0f;
            for (int i = lines.Count - 1; i >= 0; i--)
            {
                Line L = lines[i];
                float h = Mathf.Max(L.text.preferredHeight, 24f);
                if (!Mathf.Approximately(L.rect.sizeDelta.y, h))
                    L.rect.sizeDelta = new Vector2(L.rect.sizeDelta.x, h);

                if (i < lines.Count - 1)   // newest stays frozen at its slot
                {
                    float target = Mathf.Max(L.y + driftSpeed * Time.deltaTime, floor);
                    L.y = Mathf.MoveTowards(L.y, target, pushCatchupSpeed * Time.deltaTime);
                }
                L.rect.anchoredPosition = new Vector2(0f, L.y);
                floor = L.y + h + lineSpacing;

                float fade = Mathf.Clamp01(1f - L.y / region);
                L.group.alpha = fade;
                if (fade <= 0.01f)
                {
                    lines.RemoveAt(i);
                    Destroy(L.go);
                }
            }
        }

        private void LateUpdate()
        {
            // a parked line the same-frame AddMessage never reclaimed = a genuine pop — drop it
            if (recycled != null)
            {
                Destroy(recycled.go);
                recycled = null;
            }
        }

        // ---------------------------------------------------------------- transcript

        public override void AddMessage(string username, string message)
        {
            if (lineTemplate == null || linesContainer == null) return;

            if (recycled != null)
            {
                // streaming mutation: same visual line, longer text — float state untouched,
                // so the reveal never flickers or re-animates
                Line l = recycled;
                recycled = null;
                l.text.text = Compose(username, message);
                lines.Add(l);
                return;
            }

            GameObject go = Instantiate(lineTemplate, linesContainer);
            go.SetActive(true);
            var line = new Line
            {
                go = go,
                rect = (RectTransform)go.transform,
                text = go.GetComponentInChildren<TMP_Text>(),
                group = go.GetComponent<CanvasGroup>(),
                y = 0f,
            };
            line.text.text = Compose(username, message);
            line.rect.anchoredPosition = Vector2.zero;
            line.rect.sizeDelta = new Vector2(line.rect.sizeDelta.x, Mathf.Max(line.text.preferredHeight, 24f));
            line.group.alpha = 1f;
            lines.Add(line);   // previous newest is unpinned by the index shift and gets pushed up
        }

        public override void PopLastMessage()
        {
            if (lines.Count == 0) return;
            Line last = lines[lines.Count - 1];
            lines.RemoveAt(lines.Count - 1);
            if (recycled != null) Destroy(recycled.go);   // two pops without an add — drop the older park
            recycled = last;   // parked for the AddMessage that follows (or LateUpdate cleanup)
        }

        // player lines slightly dimmer grey-white, NPC lines warm white, names tinted
        private static string Compose(string username, string message)
        {
            bool player = username == "You";
            string nameHex = player ? "#A9A69C" : "#E5B85C";
            string bodyHex = player ? "#C6C4BC" : "#F6EED9";
            return $"<color={nameHex}>{username}</color>  <color={bodyHex}>{message}</color>";
        }

        public override void Clear()
        {
            foreach (Line l in lines)
                if (l.go != null) Destroy(l.go);
            lines.Clear();
            if (recycled != null)
            {
                Destroy(recycled.go);
                recycled = null;
            }
            if (inputField != null) inputField.text = "";
            if (infoText != null) infoText.text = "";
        }

        // ---------------------------------------------------------------- open / close

        protected override void OnOpen() => FadeTo(1f, null);

        protected override void OnClose()
        {
            if (!gameObject.activeSelf) return;
            FadeTo(0f, () => gameObject.SetActive(false));
        }

        private void FadeTo(float alpha, System.Action onDone)
        {
            if (fadeCoroutine != null) StopCoroutine(fadeCoroutine);
            fadeCoroutine = StartCoroutine(Fade(alpha, onDone));
        }

        private IEnumerator Fade(float target, System.Action onDone)
        {
            float start = canvasGroup.alpha;
            float elapsed = 0f;
            while (elapsed < fadeDuration)
            {
                elapsed += Time.deltaTime;
                canvasGroup.alpha = Mathf.Lerp(start, target, Mathf.Clamp01(elapsed / fadeDuration));
                yield return null;
            }
            canvasGroup.alpha = target;
            fadeCoroutine = null;
            onDone?.Invoke();
        }
    }
}
