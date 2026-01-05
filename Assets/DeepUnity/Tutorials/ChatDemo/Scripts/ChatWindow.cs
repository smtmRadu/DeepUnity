using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;

namespace DeepUnity.Tutorials.ChatDemo
{
    public class ChatWindow : MonoBehaviour
    {
        [Header("UI References")]
        [SerializeField] private Transform messageContainer;
        [SerializeField] private TMP_InputField inputField;
        [SerializeField] private Button sendButton;
        [SerializeField] private GameObject messageTemplate;
        [SerializeField] private ScrollRect scrollRect;
        [SerializeField] private TMP_Text infoText;


        public Button SendButton => sendButton;
        public TMP_InputField InputField => inputField;

        [Header("Settings")]
        private List<GameObject> messages = new List<GameObject>();
        private Canvas canvas;

        private void Awake()
        {
            gameObject.SetActive(false);
        }
        void Start()
        {
            canvas = GetComponentInParent<Canvas>();

            // Free the RectTransform from Canvas constraints
            RectTransform rect = GetComponent<RectTransform>();
            if (rect != null)
            {
                rect.anchorMin = new Vector2(0.5f, 0.5f);
                rect.anchorMax = new Vector2(0.5f, 0.5f);
                rect.pivot = new Vector2(0.5f, 0.5f);
            }

            // if (sendButton != null)
            //     sendButton.onClick.AddListener(WriteInputFieldMessageToMessagesContainer);
            // 
            // if (inputField != null)
            // {
            //     inputField.onSubmit.AddListener((text) => WriteInputFieldMessageToMessagesContainer());
            // }

            if (messageTemplate != null)
                messageTemplate.SetActive(false);

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

            // Auto-scroll to bottom
            if (scrollRect != null)
            {
                scrollRect.verticalNormalizedPosition = 0f;
            }
        }

        public void PopLastMessage()
        {
            if (messages == null || messages.Count == 0)
                return;

            GameObject lastMsg = messages[messages.Count - 1];
            messages.RemoveAt(messages.Count - 1);

            if (lastMsg != null)
            {
                Destroy(lastMsg);
            }

            // Force layout rebuild so UI updates immediately
            Canvas.ForceUpdateCanvases();

            if (scrollRect != null)
            {
                // Keep scroll pinned to bottom after removal
                scrollRect.verticalNormalizedPosition = 0f;
            }
        }


        public void Clear()
        {
            messages.Clear();
        }

        public void SetInfoText(string text)
        {
            infoText.text = text;
        }
    }
}