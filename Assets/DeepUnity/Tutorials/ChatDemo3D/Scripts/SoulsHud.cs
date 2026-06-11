using TMPro;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Souls-style HUD: HP / FP / Stamina bars top-left and the four quick-slot windows
    /// bottom-left. HP and Stamina are live (driven by SoulsPlayerController); FP is full.
    /// The whole HUD fades out while the chat is open.
    /// </summary>
    public class SoulsHud : MonoBehaviour
    {
        [SerializeField] private SoulsPlayerController player;
        [SerializeField] private SoulsChatWindow chatWindow;
        [SerializeField] private RectTransform hpFill;
        [SerializeField] private RectTransform fpFill;
        [SerializeField] private RectTransform staminaFill;
        [SerializeField] private TMP_Text flaskCount;   // charges left, on the item quick-slot

        private CanvasGroup group;

        private void Awake()
        {
            group = GetComponent<CanvasGroup>();
        }

        private void Update()
        {
            if (player != null && staminaFill != null)
                staminaFill.anchorMax = new Vector2(Mathf.Clamp01(player.Stamina01), 1f);
            if (player != null && hpFill != null)
                hpFill.anchorMax = new Vector2(Mathf.Clamp01(player.Health01), 1f);
            if (player != null && flaskCount != null)
                flaskCount.text = player.FlaskCharges.ToString();

            if (group != null && chatWindow != null)
                group.alpha = Mathf.MoveTowards(group.alpha, chatWindow.IsOpen ? 0f : 1f, Time.deltaTime * 4f);
        }
    }
}
