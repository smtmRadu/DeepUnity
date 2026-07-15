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
    /// Implements <see cref="INPCChatWindow"/> (same surface as the 3D SoulsChatWindow) so
    /// NPCChatBase drives it unchanged.
    ///
    /// Streaming contract: the base pops + re-adds the newest NPC line many times a second
    /// (token stream and audio-synced reveal). PopLastMessage therefore PARKS the newest line
    /// instead of destroying it, and the AddMessage that follows in the same call stack
    /// reclaims it as a pure text mutation — same GameObject, same float state, no
    /// re-animation and no flicker. A parked line that is never reclaimed is destroyed in
    /// LateUpdate, so a standalone pop still removes the line by end of frame. The newest
    /// line stays pinned at the bottom slot until a subsequent AddMessage pushes it up.
    /// </summary>
    public class ChatWindow2D : MonoBehaviour, INPCChatWindow
    {
        [Header("Reasoning models")]
        [Tooltip("Render <think> reasoning content in the window (dimmed italic). It is never spoken by the TTS either way.")]
        [SerializeField] private bool showThinkingTokens = false;
        public bool ShowThinkingTokens => showThinkingTokens;

        [Header("UI References (wired by the scene builder)")]
        [SerializeField] private CanvasGroup canvasGroup;       // fades the whole overlay open/closed
        [SerializeField] private RectTransform linesContainer;  // float region: lines spawn at its bottom, die near its top
        [SerializeField] private GameObject lineTemplate;       // one TMP + CanvasGroup per line
        [SerializeField] private TMP_InputField inputField;
        [SerializeField] private Button sendButton;
        [SerializeField] private Button giveButton;      // hands the basket over; shown only when it has items
        [SerializeField] private Button leaveButton;
        [SerializeField] private TMP_Text infoText;             // approach flavor, italic grey
        [SerializeField] private TMP_Text titleText;            // small gold NPC-name label above the strip

        [Header("Float animation")]
        [SerializeField] private float fadeDuration = 0.25f;
        [Tooltip("Slow upward drift of unpinned lines (canvas px/s).")]
        [SerializeField] private float driftSpeed = 24f;
        [Tooltip("Catch-up speed when a new/growing line below pushes the stack up (canvas px/s).")]
        [SerializeField] private float pushCatchupSpeed = 420f;
        [SerializeField] private float lineSpacing = 10f;

        [Header("UI sounds")]
        [SerializeField] private AudioSource uiAudio;   // lives on the canvas — must survive this overlay deactivating
        [SerializeField] private AudioClip buttonClip;
        [SerializeField] private AudioClip[] typeClips;

        public Button SendButton => sendButton;
        public Button LeaveButton => leaveButton;
        /// <summary>2D-only extra: gives the harvested basket to the NPC (NPCInteractor2D toggles
        /// its visibility per dialogue state + basket contents).</summary>
        public Button GiveButton => giveButton;
        public TMP_InputField InputField => inputField;
        public bool IsOpen { get; private set; }

        [Tooltip("Optional golden context-fill rect inside a silver track; null = no bar in this window.")]
        [SerializeField] private RectTransform contextFill;
        public void SetContextFill(float fill01)
        {
            if (contextFill != null) contextFill.anchorMax = new Vector2(Mathf.Clamp01(fill01), 1f);
        }

        // farm-gold accent shared with the builder — the caret must be unmissable
        private static readonly Color CaretGold = new Color(0.90f, 0.72f, 0.35f);

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

        private TMP_Text sendLabel;
        private string sendLabelIdle;
        private Coroutine sendLoadingCoroutine;
        private static readonly string[] loadingFrames = { ".", ". .", ". . ." };

        // input-state feedback caches (dim while the send button is loading, restore after)
        private Color inputTextIdle;
        private bool inputIdleCached;
        private string placeholderIdle;

        private void Awake()
        {
            if (canvasGroup == null) canvasGroup = GetComponent<CanvasGroup>();
            if (canvasGroup == null) canvasGroup = gameObject.AddComponent<CanvasGroup>();
            canvasGroup.alpha = 0f;
            if (lineTemplate != null) lineTemplate.SetActive(false);
            if (inputField != null)
            {
                inputField.onValueChanged.AddListener(_ => PlayTypeTick());
                // refocusing (e.g. after the model finishes loading) must not select the
                // half-typed question — the next keystroke would erase it
                inputField.onFocusSelectAll = false;
            }
            ConfigureCaret();
            gameObject.SetActive(false);
        }

        /// <summary>Thick, clearly blinking gold caret so it's obvious when typing is possible.
        /// The builder bakes the same settings; this re-applies them defensively so
        /// runtime-created/replaced input fields get the treatment too.</summary>
        private void ConfigureCaret()
        {
            if (inputField == null) return;
            inputField.customCaretColor = true;
            inputField.caretColor = CaretGold;
            inputField.caretWidth = 3;
            inputField.caretBlinkRate = 0.85f;
        }

        // ---------------------------------------------------------------- float animation

        private void Update()
        {
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

        // ---------------------------------------------------------------- INPCChatWindow

        public void AddMessage(string username, string message)
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

        public void PopLastMessage()
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

        public void Clear()
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

        public void SetTitle(string title)
        {
            if (titleText != null) titleText.text = title;
        }

        public void SetInfoText(string text)
        {
            if (infoText != null) infoText.text = text;
        }

        /// <summary>Send-button "loading" mode while the model streams in: the button is disabled
        /// and its label pulses dots, but the input field stays usable so the first question can
        /// be typed before the model is ready. Turning it off restores the original label.</summary>
        public void SetSendLoading(bool loading)
        {
            if (sendButton == null) return;
            if (sendLabel == null)
            {
                sendLabel = sendButton.GetComponentInChildren<TMP_Text>();
                sendLabelIdle = sendLabel != null ? sendLabel.text : "";
            }

            if (sendLoadingCoroutine != null)
            {
                StopCoroutine(sendLoadingCoroutine);
                sendLoadingCoroutine = null;
            }

            sendButton.interactable = !loading;
            if (loading && sendLabel != null && isActiveAndEnabled)
                sendLoadingCoroutine = StartCoroutine(PulseSendLabel());
            else if (sendLabel != null)
                sendLabel.text = sendLabelIdle;

            SetInputLoadingLook(loading);
        }

        private IEnumerator PulseSendLabel()
        {
            var step = new WaitForSeconds(0.4f);
            for (int i = 0; ; i = (i + 1) % loadingFrames.Length)
            {
                sendLabel.text = loadingFrames[i];
                yield return step;
            }
        }

        /// <summary>Subtle input-state feedback while the model streams in: the typed text dims
        /// slightly and the placeholder becomes "…"; both restore when Send is interactable
        /// again. The field itself stays usable (the first question can be typed early).</summary>
        private void SetInputLoadingLook(bool loading)
        {
            if (inputField == null) return;
            var txt = inputField.textComponent;
            if (txt != null)
            {
                if (!inputIdleCached) { inputTextIdle = txt.color; inputIdleCached = true; }
                txt.color = loading
                    ? new Color(inputTextIdle.r, inputTextIdle.g, inputTextIdle.b, inputTextIdle.a * 0.55f)
                    : inputTextIdle;
            }
            if (inputField.placeholder is TMP_Text ph)
            {
                if (placeholderIdle == null) placeholderIdle = ph.text;
                ph.text = loading ? "…" : placeholderIdle;
            }
        }

        // ---------------------------------------------------------------- open / close

        public void Open()
        {
            gameObject.SetActive(true);
            IsOpen = true;
            ConfigureCaret();   // defensive: fields wired/replaced at runtime get the caret too
            FadeTo(1f, null);
        }

        public void Close()
        {
            if (!gameObject.activeSelf) return;
            IsOpen = false;
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

        // ---------------------------------------------------------------- UI sounds

        /// <summary>Hooked to the Say/Leave buttons by the scene builder.</summary>
        public void PlayButtonClick()
        {
            if (uiAudio == null || buttonClip == null) return;
            uiAudio.pitch = Random.Range(0.96f, 1.04f);
            uiAudio.PlayOneShot(buttonClip, 0.5f);
        }

        private void PlayTypeTick()
        {
            // skip the programmatic clears (send/Clear set text to "") — only real keystrokes tick
            if (uiAudio == null || typeClips == null || typeClips.Length == 0 || inputField.text.Length == 0) return;
            uiAudio.pitch = Random.Range(0.92f, 1.12f);
            uiAudio.PlayOneShot(typeClips[Random.Range(0, typeClips.Length)], 0.35f);
        }
    }
}
