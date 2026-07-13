using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Esc menu with Continue / Exit. Soulslike: the world KEEPS RUNNING while the menu is up
    /// (no timescale change, NPCs/boss/audio all live) — only player input and camera orbit are
    /// gated, via the static IsOpen flag. Esc only toggles the menu while the chat window is
    /// closed: inside a dialogue Esc already means "leave the dialogue" (NPCChatBase handles it),
    /// so that press — whichever script sees it first — never also opens this menu.
    /// Exit stops play mode in the editor and quits the player in a build.
    /// </summary>
    public class PauseMenu : MonoBehaviour
    {
        [SerializeField] private GameObject panel;
        [SerializeField] private SoulsChatWindow chatWindow;
        [SerializeField] private Button continueButton;
        [SerializeField] private Button exitButton;

        /// <summary>Input gate for SoulsPlayerController / SoulsCameraRig while the menu is up.</summary>
        public static bool IsOpen { get; private set; }

        // Esc that closed the dialogue must not open the menu in the same/next frame — script
        // order decides whether we see the chat still open or just closed, so track both frames.
        private int lastChatOpenFrame = -10;

        private void Awake()
        {
            IsOpen = false;   // statics survive play sessions when domain reload is off
            if (panel != null) panel.SetActive(false);
            if (continueButton != null) continueButton.onClick.AddListener(Close);
            if (exitButton != null) exitButton.onClick.AddListener(Exit);
        }

        private void Update()
        {
            if (chatWindow != null && chatWindow.IsOpen) lastChatOpenFrame = Time.frameCount;
            if (!Input.GetKeyDown(KeyCode.Escape)) return;
            if (Time.frameCount - lastChatOpenFrame < 2) return;   // that Esc was "leave dialogue"
            if (IsOpen) Close(); else Open();
        }

        private void Open()
        {
            IsOpen = true;
            if (panel != null) panel.SetActive(true);
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        public void Close()
        {
            IsOpen = false;
            if (panel != null) panel.SetActive(false);
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        public void Exit()
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
    }
}
