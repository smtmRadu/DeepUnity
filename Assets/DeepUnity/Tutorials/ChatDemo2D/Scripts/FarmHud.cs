using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials.ChatDemo2D
{
    /// <summary>
    /// Bottom-left toolbar (6 slots: hoe, water, three seed bags, harvest) with a gold frame on
    /// the selected slot, harvest counters beside it, and the day/time clock top-right. Pure
    /// view — FarmingSystem and DayCycle2D push state in.
    /// </summary>
    public class FarmHud : MonoBehaviour
    {
        [Header("Wired by the scene builder")]
        [SerializeField] private Image[] slotFrames;     // 6, tinted by selection
        [SerializeField] private TMP_Text[] counters;    // one per crop, "x N"
        [SerializeField] private TMP_Text clockText;
        [SerializeField] private TMP_Text coinText;      // coins earned from villagers
        [SerializeField] private Color idleFrame = new Color(0.35f, 0.27f, 0.18f, 0.9f);
        [SerializeField] private Color selectedFrame = new Color(0.95f, 0.78f, 0.35f, 1f);

        public void SetSelected(int index)
        {
            if (slotFrames == null) return;
            for (int i = 0; i < slotFrames.Length; i++)
                if (slotFrames[i] != null)
                    slotFrames[i].color = i == index ? selectedFrame : idleFrame;
        }

        public void SetCount(int crop, int n)
        {
            if (counters != null && crop >= 0 && crop < counters.Length && counters[crop] != null)
                counters[crop].text = "x " + n;
        }

        public void SetClock(string text)
        {
            if (clockText != null) clockText.text = text;
        }

        public void SetCoins(int n)
        {
            if (coinText != null) coinText.text = n + " g";
        }
    }
}
