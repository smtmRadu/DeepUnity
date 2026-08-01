using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace DeepUnity
{
    /// <summary>
    /// Base class for EVERY NPC dialogue window, in every environment (3D souls castle, 2D farm,
    /// Anya, and anything added later). It implements the whole <see cref="INPCChatWindow"/>
    /// surface that does NOT depend on how a given window looks — the send-button loading pulse,
    /// the input caret and typing sounds, title/info text, the context-fill bar — plus BOTH built-in
    /// interactive panels: the <b>AskUserQuestion</b> choice popup
    /// (<see cref="INPCToolQuestionWindow"/>) and the <b>GiveTool</b> offer popup
    /// (<see cref="INPCToolGiveWindow"/>), so every derived window supports the NPC's interactive
    /// tools for free. Those two are the whole set — an NPC's belt may carry either, both or neither,
    /// and anything else it does is an internal read that never reaches the screen.
    ///
    /// A concrete window only supplies its own PRESENTATION: how it opens and closes
    /// (<see cref="OnOpen"/> / <see cref="OnClose"/>) and how it renders the transcript
    /// (<see cref="AddMessage"/> / <see cref="PopLastMessage"/> / <see cref="Clear"/>). Restyle the
    /// popups by overriding the ToolQuestion* theme properties (shared by both) and
    /// <see cref="ToolGiveCurrency"/> / <see cref="FormatToolGive"/> — never by re-implementing them.
    ///
    /// Streaming contract every subclass must honor: the NPC pops + re-adds the newest line many
    /// times a second during the token stream, so <see cref="PopLastMessage"/> should PARK the
    /// newest line and let the <see cref="AddMessage"/> that follows in the same call stack reclaim
    /// it as a pure text mutation — destroying and re-instantiating it flickers.
    /// </summary>
    public abstract class NPCDialogueWindow : MonoBehaviour, INPCChatWindow, INPCToolQuestionWindow,
                                              INPCToolGiveWindow
    {
        [Header("Reasoning models")]
        [Tooltip("Render <think> reasoning content in the window (dimmed italic). It is never spoken by the TTS either way.")]
        [SerializeField] protected bool showThinkingTokens = false;
        public bool ShowThinkingTokens => showThinkingTokens;

        [Header("Dialogue UI (wired by the scene builder)")]
        [SerializeField] protected TMP_InputField inputField;
        [SerializeField] protected Button sendButton;
        [SerializeField] protected Button leaveButton;
        [SerializeField] protected TMP_Text infoText;
        [SerializeField] protected TMP_Text titleText;
        [Tooltip("Optional context-fill rect (its anchorMax.x is driven 0..1); null = no bar in this window.")]
        [SerializeField] protected RectTransform contextFill;

        [Header("UI sounds")]
        [Tooltip("Lives on the canvas, not on the panel — it must survive this window deactivating.")]
        [SerializeField] protected AudioSource uiAudio;
        [SerializeField] protected AudioClip buttonClip;
        [SerializeField] protected AudioClip[] typeClips;

        public Button SendButton => sendButton;
        public Button LeaveButton => leaveButton;
        public TMP_InputField InputField => inputField;
        public bool IsOpen { get; private set; }

        // ---------------------------------------------------------------- theme hooks

        /// <summary>Caret colour — each environment's accent (souls gold, farm gold, …).</summary>
        protected virtual Color CaretColor => new Color(0.85f, 0.72f, 0.40f);
        /// <summary>Font for the choice popup. Defaults to the window's own title font, so a
        /// derived window is themed correctly without overriding anything.</summary>
        protected virtual TMP_FontAsset ToolQuestionFont => titleText != null ? titleText.font : null;
        /// <summary>Frame / title colour of the choice popup.</summary>
        protected virtual Color ToolQuestionAccent => CaretColor;
        /// <summary>Panel background of the choice popup.</summary>
        protected virtual Color ToolQuestionPanel => new Color(0.07f, 0.062f, 0.05f, 0.97f);
        /// <summary>Question + option label colour of the choice popup.</summary>
        protected virtual Color ToolQuestionText => new Color(0.86f, 0.83f, 0.75f);
        /// <summary>Option button fill of the choice popup.</summary>
        protected virtual Color ToolQuestionOptionFill => new Color(0.15f, 0.13f, 0.105f, 0.97f);

        /// <summary>What a GiveTool price is quoted IN, in this environment — "souls" in the 3D castle,
        /// "coins" on the farm. Empty (the default) prints the bare number, because the base class has
        /// no business inventing a currency for a game it knows nothing about.</summary>
        protected virtual string ToolGiveCurrency => "";

        /// <summary>The one line the offer panel shows above Accept | Decline: the item, its quantity
        /// when the NPC named one, and its price when there is one. Override to reword or restyle it;
        /// the two buttons are fixed, because the tool's result is a yes/no and nothing else.</summary>
        protected virtual string FormatToolGive(ToolGiveOffer offer)
        {
            var s = new System.Text.StringBuilder(offer.item ?? "");
            if (offer.quantity.HasValue) s.Append(" x").Append(offer.quantity.Value);
            if (offer.price.HasValue)
            {
                s.Append("  -  ").Append(offer.price.Value);
                string cur = ToolGiveCurrency;
                if (!string.IsNullOrEmpty(cur)) s.Append(' ').Append(cur);
            }
            return s.ToString();
        }

        // ---------------------------------------------------------------- lifecycle

        protected virtual void Awake()
        {
            if (inputField != null)
            {
                inputField.onValueChanged.AddListener(_ => PlayTypeTick());
                // refocusing (e.g. after the model finishes loading) must not select the
                // half-typed question — the next keystroke would erase it
                inputField.onFocusSelectAll = false;
            }
            ConfigureCaret();
        }

        /// <summary>Thick, clearly blinking caret so it's obvious when typing is possible. The
        /// builders bake the same settings; this re-applies them defensively so runtime-created or
        /// replaced input fields get the treatment too.</summary>
        protected void ConfigureCaret()
        {
            if (inputField == null) return;
            inputField.customCaretColor = true;
            inputField.caretColor = CaretColor;
            inputField.caretWidth = 3;
            inputField.caretBlinkRate = 0.85f;
        }

        // The bar glides toward the live token count instead of snapping. Subclasses that need
        // their own Update MUST call base.Update().
        private float ctxTarget, ctxShown;

        public void SetContextFill(float fill01) => ctxTarget = Mathf.Clamp01(fill01);

        protected virtual void Update()
        {
            if (contextFill == null) return;
            ctxShown = Mathf.Lerp(ctxShown, ctxTarget, 1f - Mathf.Exp(-6f * Time.unscaledDeltaTime));
            contextFill.anchorMax = new Vector2(ctxShown, 1f);
        }

        /// <summary>Opens the window. Presentation (slide/fade/…) belongs in <see cref="OnOpen"/>.</summary>
        public void Open()
        {
            gameObject.SetActive(true);
            IsOpen = true;
            ConfigureCaret();   // defensive: fields wired/replaced at runtime get the caret too
            OnOpen();
        }

        /// <summary>Closes the window. A live interactive panel is ALWAYS torn down first — it must
        /// never outlive the dialogue underneath it (it replaces the input row).</summary>
        public void Close()
        {
            // clear IsOpen FIRST — must never stay stuck true (pause menus key their Esc-swallow
            // off it; a stale value soft-locks them). Safe even when already inactive.
            IsOpen = false;
            HideToolPanel();   // whichever of the two was up
            OnClose();
        }

        /// <summary>Show this window (slide in, fade in, …). Called with the GameObject already active.</summary>
        protected abstract void OnOpen();
        /// <summary>Hide this window (slide out, fade out, …). May early-out if already inactive.</summary>
        protected abstract void OnClose();

        // ---------------------------------------------------------------- transcript (per-window)

        /// <inheritdoc/>
        public abstract void AddMessage(string username, string message);
        /// <inheritdoc/>
        public abstract void PopLastMessage();
        /// <inheritdoc/>
        public abstract void Clear();

        public virtual void SetTitle(string title)
        {
            if (titleText != null) titleText.text = title;
        }

        public virtual void SetInfoText(string text)
        {
            if (infoText != null) infoText.text = text;
        }

        // ---------------------------------------------------------------- send-button loading

        private TMP_Text sendLabel;
        private string sendLabelIdle;
        private Coroutine sendLoadingCoroutine;
        private static readonly string[] loadingFrames = { ".", ". .", ". . ." };
        private Color inputTextIdle;
        private bool inputIdleCached;
        private string placeholderIdle;

        /// <summary>Send-button "loading" mode while the model streams in: the button is disabled
        /// and its label pulses dots, but the input field stays usable so the first question can be
        /// typed before the model is ready. Turning it off restores the original label.</summary>
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
        /// slightly and the placeholder becomes "…"; both restore when Send is interactable again.
        /// The field itself stays usable (the first question can be typed early).</summary>
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

        // ---------------------------------------------------------------- UI sounds

        /// <summary>Hooked to the Send/Leave buttons by the scene builders.</summary>
        public void PlayButtonClick()
        {
            if (uiAudio == null || buttonClip == null) return;
            uiAudio.pitch = Random.Range(0.96f, 1.04f);
            uiAudio.PlayOneShot(buttonClip, 0.5f);
        }

        private void PlayTypeTick()
        {
            // skip the programmatic clears (send/Clear set text to "") — only real keystrokes tick
            if (uiAudio == null || typeClips == null || typeClips.Length == 0
                || inputField == null || inputField.text.Length == 0) return;
            uiAudio.pitch = Random.Range(0.92f, 1.12f);
            uiAudio.PlayOneShot(typeClips[Random.Range(0, typeClips.Length)], 0.35f);
        }

        // ---------------------------------------------------------------- interactive tool panels
        // The NPC's TWO interactive tools — AskUserQuestion (a choice) and GiveTool (an item offer) —
        // implemented ONCE for every environment and sharing one panel builder, because they are the
        // same affordance with different labels: a line of text above a row of buttons whose click is
        // the tool's result. The panel TAKES THE PLACE OF THE INPUT ROW (user spec 2026-07-25): the
        // input field, the context bar and the Speak/Leave buttons go away and the panel occupies
        // exactly that strip until the player answers — you answer by clicking instead of typing, so
        // the window can never show both affordances at once. Built at runtime (no prefab, no scene
        // rebuild), themed through the ToolQuestion* properties, torn down on click / Close / demand;
        // teardown re-activates everything it hid, and ONLY that (a Speak button already disabled
        // for other reasons stays as it was). Only ONE panel is ever up: opening either tears down
        // whatever was there.

        private GameObject toolPanelRoot;
        // index-based, so the choice panel maps it back to the option TEXT and the offer panel to
        // accept/decline — the builder itself never needs to know which tool it is serving
        private System.Action<int> toolPanelPick;
        private readonly List<GameObject> toolPanelHidden = new List<GameObject>();

        /// <summary>The chrome an interactive panel replaces: the input row (with its Speak/Leave
        /// buttons) and the context bar. Shared by both panels. Override to hide more/less in a custom
        /// window.</summary>
        protected virtual void CollectToolQuestionChrome(List<GameObject> hide)
        {
            // one row usually parents input + Speak + Leave — hide the row itself so its layout
            // group collapses too, instead of leaving an empty 48 px gap under the panel
            Transform row = inputField != null ? inputField.transform.parent : null;
            if (row != null && sendButton != null && leaveButton != null
                && sendButton.transform.parent == row && leaveButton.transform.parent == row)
            {
                hide.Add(row.gameObject);
            }
            else
            {
                if (inputField != null) hide.Add(inputField.gameObject);
                if (sendButton != null) hide.Add(sendButton.gameObject);
                if (leaveButton != null) hide.Add(leaveButton.gameObject);
            }
            // the fill is a child of the bar's track — hide the track so the whole bar goes
            if (contextFill != null)
                hide.Add(contextFill.parent != null ? contextFill.parent.gameObject : contextFill.gameObject);
        }

        /// <inheritdoc/>
        public virtual void ShowToolQuestion(string npcName, string question, IReadOnlyList<string> options,
                                             System.Action<string> onPick)
        {
            // the pick comes back as the option's exact TEXT, which is what the model is answered with
            var labels = new List<string>(options);
            ShowToolPanel(question, labels, null, i => onPick?.Invoke(labels[i]));
        }

        /// <inheritdoc/>
        public virtual void ShowToolGive(string npcName, ToolGiveOffer offer, bool canAccept,
                                         System.Action<bool> onDecide)
        {
            // EXACTLY two buttons, always in this order. Accept can be gated off (no money); Decline
            // never is, so an offer the player cannot afford still ends the exchange with a real answer
            // instead of a dead panel.
            ShowToolPanel(FormatToolGive(offer),
                          new List<string> { AcceptLabel, DeclineLabel },
                          new List<bool> { canAccept, true },
                          i => onDecide?.Invoke(i == 0));
        }

        /// <summary>The offer panel's two fixed labels — the tool's result is a yes/no, so these are
        /// not authored per NPC.</summary>
        protected const string AcceptLabel = "Accept";
        protected const string DeclineLabel = "Decline";

        /// <summary>Build the interactive strip: one line of text above a row of buttons, one button per
        /// label, and report the clicked INDEX exactly once. Both tools' panels are this.</summary>
        /// <param name="enabled">Per-button interactability, or null for "all clickable". A disabled
        /// button is still drawn (dimmed) — the player must see the offer they cannot take.</param>
        private void ShowToolPanel(string prompt, IReadOnlyList<string> labels, IReadOnlyList<bool> enabled,
                                   System.Action<int> onPick)
        {
            HideToolPanel();
            EnsureEventSystem();   // a scene without one renders the buttons but never clicks them
            toolPanelPick = onPick;

            TMP_FontAsset font = ToolQuestionFont;
            Color accent = ToolQuestionAccent, text = ToolQuestionText;

            // stand the typing chrome down (remembering only what WE turned off)
            var chrome = new List<GameObject>();
            CollectToolQuestionChrome(chrome);
            foreach (var go in chrome)
            {
                if (go == null || !go.activeSelf) continue;
                go.SetActive(false);
                toolPanelHidden.Add(go);
            }

            // the panel sits in the strip the input row occupied, pinned to the window's bottom edge
            // and growing UPWARD (pivot y = 0 + a vertical fitter) as the button count demands
            toolPanelRoot = new GameObject("NPCToolPanel", typeof(RectTransform), typeof(Image),
                                           typeof(VerticalLayoutGroup), typeof(ContentSizeFitter), typeof(Outline));
            var boxRT = (RectTransform)toolPanelRoot.transform;
            boxRT.SetParent(transform, false);
            boxRT.anchorMin = new Vector2(0f, 0f);
            boxRT.anchorMax = new Vector2(1f, 0f);
            boxRT.pivot = new Vector2(0.5f, 0f);
            boxRT.offsetMin = new Vector2(18f, 6f);
            boxRT.offsetMax = new Vector2(-18f, 6f);
            boxRT.SetAsLastSibling();   // above the message list it may overlap
            toolPanelRoot.GetComponent<Image>().color = ToolQuestionPanel;
            var boxLine = toolPanelRoot.GetComponent<Outline>();
            boxLine.effectColor = new Color(accent.r, accent.g, accent.b, 0.9f);
            boxLine.effectDistance = new Vector2(2f, -2f);
            var lay = toolPanelRoot.GetComponent<VerticalLayoutGroup>();
            lay.padding = new RectOffset(20, 20, 14, 14);
            lay.spacing = 10f;
            lay.childControlWidth = true; lay.childControlHeight = true;
            lay.childForceExpandWidth = true; lay.childForceExpandHeight = false;
            toolPanelRoot.GetComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;

            MakeModalText(boxRT, prompt, font, text, 22f, FontStyles.Normal);

            // buttons laid out side by side like the ones they replace, splitting the width
            var rowGO = new GameObject("Options", typeof(RectTransform), typeof(HorizontalLayoutGroup),
                                       typeof(LayoutElement));
            rowGO.transform.SetParent(boxRT, false);
            var rowLay = rowGO.GetComponent<HorizontalLayoutGroup>();
            rowLay.spacing = 10f;
            rowLay.childControlWidth = true; rowLay.childControlHeight = true;
            rowLay.childForceExpandWidth = true; rowLay.childForceExpandHeight = true;
            rowGO.GetComponent<LayoutElement>().minHeight = 46f;

            for (int b = 0; b < labels.Count; b++)
            {
                int index = b;             // per-iteration capture for the click closure
                bool live = enabled == null || b >= enabled.Count || enabled[b];
                var btnGO = new GameObject("Option", typeof(RectTransform), typeof(Image),
                                           typeof(Outline), typeof(Button), typeof(LayoutElement));
                btnGO.transform.SetParent(rowGO.transform, false);
                var img = btnGO.GetComponent<Image>();
                img.color = ToolQuestionOptionFill;
                var line = btnGO.GetComponent<Outline>();
                line.effectColor = new Color(accent.r, accent.g, accent.b, live ? 0.5f : 0.2f);
                line.effectDistance = new Vector2(1f, -1f);
                var btnLay = btnGO.GetComponent<LayoutElement>();
                btnLay.minHeight = 46f;
                btnLay.flexibleWidth = 1f;   // 2-4 buttons share the row evenly
                var btn = btnGO.GetComponent<Button>();
                btn.targetGraphic = img;
                var colors = btn.colors;
                colors.normalColor = new Color(0.85f, 0.85f, 0.85f);
                colors.highlightedColor = Color.white;
                colors.selectedColor = Color.white;
                colors.pressedColor = new Color(0.6f, 0.6f, 0.6f);
                // spelled out rather than left at uGUI's default, so a gated Accept reads as clearly
                // unavailable at a glance instead of merely slightly paler
                colors.disabledColor = new Color(0.38f, 0.38f, 0.38f, 0.85f);
                colors.fadeDuration = 0.08f;
                btn.colors = colors;
                btn.interactable = live;
                Color labelColor = live
                    ? text
                    : new Color(text.r, text.g, text.b, text.a * 0.45f);
                var label = MakeModalText(btnGO.transform, labels[b], font, labelColor, 20f, FontStyles.Normal);
                var labelRT = (RectTransform)label.transform;   // fill the button, not layout-driven
                labelRT.anchorMin = Vector2.zero; labelRT.anchorMax = Vector2.one;
                labelRT.offsetMin = Vector2.zero; labelRT.offsetMax = Vector2.zero;
                label.raycastTarget = false;
                if (!live) continue;       // drawn, dimmed, and it answers nothing
                btn.onClick.AddListener(() =>
                {
                    PlayButtonClick();
                    var pick = toolPanelPick;   // the teardown clears the field — fire AFTER it
                    HideToolPanel();
                    pick?.Invoke(index);
                });
            }
        }

        /// <inheritdoc/>
        public virtual void HideToolQuestion() => HideToolPanel();

        /// <inheritdoc/>
        public virtual void HideToolGive() => HideToolPanel();

        /// <summary>Tear down whichever interactive panel is up and give the typing chrome back.
        /// Idempotent, and a no-op when no panel was ever built.</summary>
        private void HideToolPanel()
        {
            toolPanelPick = null;
            if (toolPanelRoot != null)
            {
                // DestroyImmediate off the play loop: an editor probe that renders this panel in edit
                // mode would otherwise leak it into the scene (plain Destroy is refused there)
                if (Application.isPlaying) Destroy(toolPanelRoot);
                else DestroyImmediate(toolPanelRoot);
                toolPanelRoot = null;
            }
            // give the typing chrome back — only the objects this panel actually hid
            foreach (var go in toolPanelHidden)
                if (go != null) go.SetActive(true);
            toolPanelHidden.Clear();
        }

        // A uGUI Button is dead without an EventSystem in the scene. Every demo builder makes one,
        // but a hand-built environment easily forgets — and the symptom (a popup that ignores
        // clicks, freezing the dialogue) is miserable to diagnose. Create one if it's missing.
        private static void EnsureEventSystem()
        {
            if (EventSystem.current != null) return;
#if UNITY_2023_1_OR_NEWER
            if (Object.FindFirstObjectByType<EventSystem>() != null) return;
#else
            if (Object.FindObjectOfType<EventSystem>() != null) return;
#endif
            var go = new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
            Debug.LogWarning("[NPCDialogueWindow] no EventSystem in the scene — created one so the " +
                             "interactive tool panel's buttons are clickable.");
            // Play mode ONLY: DontDestroyOnLoad THROWS from an editor script, and an edit-mode probe
            // that renders this panel in an empty scene hits exactly that (NpcGiveToolProbe did). The
            // object is created either way — it simply has nothing to survive outside play mode.
            if (Application.isPlaying) DontDestroyOnLoad(go);
        }

        private static TMP_Text MakeModalText(Transform parent, string content, TMP_FontAsset font,
                                              Color color, float size, FontStyles style)
        {
            var go = new GameObject("Text", typeof(RectTransform));
            go.transform.SetParent(parent, false);
            var tmp = go.AddComponent<TextMeshProUGUI>();
            if (font != null) tmp.font = font;
            tmp.text = content;
            tmp.color = color;
            tmp.fontSize = size;
            tmp.fontStyle = style;
            tmp.alignment = TextAlignmentOptions.Center;
            return tmp;
        }
    }
}
