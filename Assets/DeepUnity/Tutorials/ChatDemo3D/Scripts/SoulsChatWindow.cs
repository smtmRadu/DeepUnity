using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Dark-souls styled chat panel docked to the right edge of the screen; slides in when the
    /// dialogue starts and out when it ends. Everything environment-agnostic — the send-loading
    /// pulse, caret, UI sounds, title/info, the context bar and the AskUserQuestion choice popup —
    /// comes from <see cref="NPCDialogueWindow"/>; this class only adds the souls presentation:
    /// the slide animation and the scrolling message list.
    /// </summary>
    public class SoulsChatWindow : NPCDialogueWindow
    {
        [Header("Souls panel (wired by the scene builder)")]
        [SerializeField] private RectTransform panel;
        [SerializeField] private Transform messageContainer;
        [SerializeField] private GameObject messageTemplate;
        [SerializeField] private ScrollRect scrollRect;
        [SerializeField] private float slideDuration = 0.4f;

        private readonly List<GameObject> messages = new List<GameObject>();
        private Coroutine slideCoroutine;
        private float shownX, hiddenX;

        // gold/parchment accent used by the scene builder — the caret must be unmissable, and the
        // choice popup inherits the same gold through the base class's theme hooks
        private static readonly Color SoulsGold = new Color(0.77f, 0.66f, 0.42f);
        protected override Color CaretColor => SoulsGold;

        // what a GiveItem price is quoted in here, so the offer panel reads "Longsword  -  80 souls"
        // instead of a bare number (the base class has no currency of its own, by design)
        protected override string GiveItemCurrency => "souls";

        protected override void Awake()
        {
            if (panel == null) panel = (RectTransform)transform;
            shownX = panel.anchoredPosition.x;
            hiddenX = shownX + panel.rect.width + 60f;
            panel.anchoredPosition = new Vector2(hiddenX, panel.anchoredPosition.y);
            if (messageTemplate != null) messageTemplate.SetActive(false);
            base.Awake();   // input listeners + caret
            gameObject.SetActive(false);
        }

        // ---------------------------------------------------------------- open / close

        protected override void OnOpen() => SlideTo(shownX, null);

        protected override void OnClose()
        {
            if (!gameObject.activeSelf) return;
            SlideTo(hiddenX, () => gameObject.SetActive(false));
        }

        private void SlideTo(float x, System.Action onDone)
        {
            if (slideCoroutine != null) StopCoroutine(slideCoroutine);
            slideCoroutine = StartCoroutine(Slide(x, onDone));
        }

        private IEnumerator Slide(float xTarget, System.Action onDone)
        {
            float xStart = panel.anchoredPosition.x;
            float elapsed = 0f;
            while (elapsed < slideDuration)
            {
                elapsed += Time.deltaTime;
                float t = Mathf.Clamp01(elapsed / slideDuration);
                t = t * t * (3f - 2f * t);
                panel.anchoredPosition = new Vector2(Mathf.Lerp(xStart, xTarget, t), panel.anchoredPosition.y);
                yield return null;
            }
            panel.anchoredPosition = new Vector2(xTarget, panel.anchoredPosition.y);
            slideCoroutine = null;
            onDone?.Invoke();
        }

        // ---------------------------------------------------------------- transcript
        // Streaming contract (see NPCDialogueWindow): the NPC pops + re-adds the newest line many
        // times a second. Destroy() is deferred to end of frame, so a naive pop+add briefly lays
        // out BOTH the old and the new message — the visible one-frame text bob. PopLastMessage
        // therefore PARKS the newest message and the AddMessage that follows in the same call
        // stack reclaims it as a pure text mutation on the SAME GameObject. A parked message never
        // reclaimed (a genuine pop) is destroyed in LateUpdate.
        private GameObject recycledMsg;
        // 2026-08-02 smoothness hunt: pin-to-bottom for STREAMING mutations, deferred one frame.
        // The reveal path calls AddMessage nearly every frame, and the Canvas.ForceUpdateCanvases
        // it used to pay per call is a synchronous rebuild of the WHOLE transcript (every TMP's
        // preferred height, the layout group, all canvases) — the 82-163 ms zero-tick frames in
        // every talk-perf worst-20, growing with conversation length because the transcript does.
        // The natural end-of-frame canvas pass lays the same text out ONCE anyway; the only thing
        // lost is same-frame scroll pinning, so pin on the NEXT frame's valid layout instead —
        // during continuous typing the view trails the bottom by at most one line for one frame.
        private bool scrollPinPending;

        private void LateUpdate()
        {
            if (recycledMsg != null)
            {
                Destroy(recycledMsg);
                recycledMsg = null;
                Canvas.ForceUpdateCanvases();
                if (scrollRect != null)
                    scrollRect.verticalNormalizedPosition = 0f;
            }
            else if (scrollPinPending)
            {
                scrollPinPending = false;
                if (scrollRect != null)
                    scrollRect.verticalNormalizedPosition = 0f;   // last frame's layout: exact pin
            }
        }

        public override void AddMessage(string username, string message)
        {
            if (messageTemplate == null || messageContainer == null) return;

            GameObject newMsg;
            bool streamingMutation = recycledMsg != null;
            if (streamingMutation)
            {
                newMsg = recycledMsg;          // streaming mutation: same object, longer text
                recycledMsg = null;
            }
            else
            {
                newMsg = Instantiate(messageTemplate, messageContainer);
                newMsg.SetActive(true);
            }

            TMP_Text[] texts = newMsg.GetComponentsInChildren<TMP_Text>();
            if (texts.Length >= 2)
            {
                texts[0].text = username;
                texts[1].text = message;
            }
            else if (texts.Length == 1)
            {
                texts[0].text = $"{username}: {message}";
            }

            messages.Add(newMsg);
            if (streamingMutation)
            {
                scrollPinPending = true;       // no forced rebuild — see the field's note
            }
            else
            {
                Canvas.ForceUpdateCanvases();  // genuinely new line (rare): settle + pin same-frame
                if (scrollRect != null)
                    scrollRect.verticalNormalizedPosition = 0f;
            }
        }

        public override void PopLastMessage()
        {
            if (messages.Count == 0) return;

            GameObject lastMsg = messages[messages.Count - 1];
            messages.RemoveAt(messages.Count - 1);
            if (lastMsg == null) return;

            if (recycledMsg != null) Destroy(recycledMsg);   // two pops back-to-back
            recycledMsg = lastMsg;
            // no canvas update here — the object is still in place; the paired AddMessage
            // (or LateUpdate for a genuine pop) settles the layout exactly once
        }

        public override void Clear()
        {
            recycledMsg = null;   // parked message is a container child — destroyed below
            if (messageContainer != null)
            {
                for (int i = messageContainer.childCount - 1; i >= 0; i--)
                {
                    GameObject child = messageContainer.GetChild(i).gameObject;
                    if (child == messageTemplate) continue;
                    Destroy(child);
                }
            }
            messages.Clear();
            if (inputField != null) inputField.text = "";
            if (infoText != null) infoText.text = "";
        }
    }
}
