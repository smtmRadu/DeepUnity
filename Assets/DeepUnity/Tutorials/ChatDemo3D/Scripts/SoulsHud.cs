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

        // Last values actually pushed into the UI. Writing a RectTransform anchor or a TMP string
        // marks that graphic dirty, and one dirty graphic makes Unity rebatch the WHOLE canvas — and
        // this scene has a single 113-graphic canvas. Before these guards the HUD dirtied it on every
        // single frame even while standing still at full health, and re-allocated the flask string
        // 60+ times a second. Same pixels, just written only when they change.
        private float lastStamina = -1f;
        private float lastHealth = -1f;
        private int lastFlasks = -1;

        private void Awake()
        {
            group = GetComponent<CanvasGroup>();
        }

        private void Update()
        {
            if (player != null)
            {
                float stamina = Mathf.Clamp01(player.Stamina01);
                if (staminaFill != null && stamina != lastStamina)
                {
                    staminaFill.anchorMax = new Vector2(stamina, 1f);
                    lastStamina = stamina;
                }

                float health = Mathf.Clamp01(player.Health01);
                if (hpFill != null && health != lastHealth)
                {
                    hpFill.anchorMax = new Vector2(health, 1f);
                    lastHealth = health;
                }

                int flasks = player.FlaskCharges;
                if (flaskCount != null && flasks != lastFlasks)
                {
                    flaskCount.text = flasks.ToString();
                    lastFlasks = flasks;
                }
            }

            if (group != null && chatWindow != null)
            {
                float target = chatWindow.IsOpen ? 0f : 1f;
                float alpha = Mathf.MoveTowards(group.alpha, target, Time.deltaTime * 4f);
                if (alpha != group.alpha) group.alpha = alpha;   // CanvasGroup.alpha dirties children
            }
        }
    }
}
