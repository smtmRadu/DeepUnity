using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo2D
{
    /// <summary>
    /// Ambient day/night cycle: one in-game day every dayLengthSeconds, expressed as a color
    /// tint on a full-map overlay sprite (drawn above every world sprite, below the UI canvas).
    /// Dawn blushes warm, midday is clear, dusk goes amber, night settles into deep blue.
    /// Purely atmospheric — crop growth is real-time and unaffected (approved scope).
    /// </summary>
    public class DayCycle2D : MonoBehaviour
    {
        [Header("Wired by the scene builder")]
        [SerializeField] private SpriteRenderer overlay;
        [SerializeField] private FarmHud hud;

        [Header("Tuning")]
        [Tooltip("Real seconds per in-game day.")]
        [SerializeField] private float dayLengthSeconds = 240f;
        [SerializeField] private float startHour = 8f;

        // (hour, tint) anchors; lerped in order, wrapping midnight
        private static readonly (float hour, Color color)[] KEYS =
        {
            ( 4.5f, new Color(0.05f, 0.08f, 0.25f, 0.42f)),   // late night
            ( 6.5f, new Color(0.85f, 0.45f, 0.25f, 0.16f)),   // dawn blush
            ( 9.0f, new Color(0f, 0f, 0f, 0f)),               // clear morning
            (16.5f, new Color(0f, 0f, 0f, 0f)),               // clear afternoon
            (19.0f, new Color(0.90f, 0.45f, 0.15f, 0.24f)),   // dusk amber
            (21.5f, new Color(0.05f, 0.08f, 0.25f, 0.42f)),   // nightfall
        };

        private float hour;
        private int day = 1;

        private void Start()
        {
            hour = startHour;
            Apply();
        }

        private void Update()
        {
            hour += 24f * Time.deltaTime / dayLengthSeconds;
            if (hour >= 24f)
            {
                hour -= 24f;
                day++;
            }
            Apply();
        }

        private void Apply()
        {
            if (overlay != null) overlay.color = Evaluate(hour);
            hud?.SetClock($"Day {day}   {(int)hour:00}:{(int)((hour % 1f) * 60f):00}");
        }

        private static Color Evaluate(float h)
        {
            for (int i = 0; i < KEYS.Length; i++)
            {
                var a = KEYS[i];
                var b = KEYS[(i + 1) % KEYS.Length];
                float span = b.hour - a.hour;
                float t = h - a.hour;
                if (span < 0f) { span += 24f; if (t < 0f) t += 24f; }   // the wrap segment
                if (t >= 0f && t <= span)
                    return Color.Lerp(a.color, b.color, span < 1e-4f ? 0f : t / span);
            }
            return KEYS[0].color;
        }
    }
}
