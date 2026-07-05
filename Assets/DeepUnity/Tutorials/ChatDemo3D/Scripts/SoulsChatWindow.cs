using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Dark-souls styled chat panel docked to the right edge of the screen; slides in when the
    /// dialogue starts and out when it ends. Message API mirrors the 2D demo's ChatWindow
    /// (AddMessage / PopLastMessage / Clear / SetInfoText).
    /// </summary>
    public class SoulsChatWindow : MonoBehaviour
    {
        [Header("UI References (wired by the scene builder)")]
        [SerializeField] private RectTransform panel;
        [SerializeField] private Transform messageContainer;
        [SerializeField] private TMP_InputField inputField;
        [SerializeField] private Button sendButton;
        [SerializeField] private Button leaveButton;
        [SerializeField] private GameObject messageTemplate;
        [SerializeField] private ScrollRect scrollRect;
        [SerializeField] private TMP_Text infoText;
        [SerializeField] private TMP_Text titleText;
        [SerializeField] private float slideDuration = 0.4f;

        [Header("UI sounds")]
        [SerializeField] private AudioSource uiAudio;   // lives on the canvas — must survive this panel deactivating
        [SerializeField] private AudioClip buttonClip;
        [SerializeField] private AudioClip[] typeClips;

        public Button SendButton => sendButton;
        public Button LeaveButton => leaveButton;
        public TMP_InputField InputField => inputField;
        public bool IsOpen { get; private set; }

        private readonly List<GameObject> messages = new List<GameObject>();
        private Coroutine slideCoroutine;
        private float shownX, hiddenX;

        private TMP_Text sendLabel;
        private string sendLabelIdle;
        private Coroutine sendLoadingCoroutine;
        private static readonly string[] loadingFrames = { ".", ". .", ". . ." };

        private void Awake()
        {
            if (panel == null) panel = (RectTransform)transform;
            shownX = panel.anchoredPosition.x;
            hiddenX = shownX + panel.rect.width + 60f;
            panel.anchoredPosition = new Vector2(hiddenX, panel.anchoredPosition.y);
            if (messageTemplate != null) messageTemplate.SetActive(false);
            if (inputField != null)
            {
                inputField.onValueChanged.AddListener(_ => PlayTypeTick());
                // refocusing (e.g. after the model finishes loading) must not select the
                // half-typed question — the next keystroke would erase it
                inputField.onFocusSelectAll = false;
            }
            gameObject.SetActive(false);
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

        /// <summary>Hooked to the Speak/Leave buttons by the scene builder.</summary>
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

        public void SetTitle(string title)
        {
            if (titleText != null) titleText.text = title;
        }

        public void Open()
        {
            gameObject.SetActive(true);
            IsOpen = true;
            SlideTo(shownX, null);
        }

        public void Close()
        {
            if (!gameObject.activeSelf) return;
            IsOpen = false;
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

        public void AddMessage(string username, string message)
        {
            if (messageTemplate == null || messageContainer == null) return;

            GameObject newMsg = Instantiate(messageTemplate, messageContainer);
            newMsg.SetActive(true);

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
            Canvas.ForceUpdateCanvases();
            if (scrollRect != null)
                scrollRect.verticalNormalizedPosition = 0f;   // pin to bottom
        }

        public void PopLastMessage()
        {
            if (messages.Count == 0) return;

            GameObject lastMsg = messages[messages.Count - 1];
            messages.RemoveAt(messages.Count - 1);
            if (lastMsg != null) Destroy(lastMsg);

            Canvas.ForceUpdateCanvases();
            if (scrollRect != null)
                scrollRect.verticalNormalizedPosition = 0f;
        }

        public void Clear()
        {
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

        public void SetInfoText(string text)
        {
            if (infoText != null) infoText.text = text;
        }
    }
}
