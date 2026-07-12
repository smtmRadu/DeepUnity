using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Cheap candle-flame flicker for a point light: Perlin-noise modulation of intensity and a
    /// tiny positional wobble. Each instance self-seeds so neighboring candles never pulse in sync.
    /// </summary>
    [RequireComponent(typeof(Light))]
    public class CandleFlicker : MonoBehaviour
    {
        [Tooltip("Fraction of the base intensity the flicker may add/remove (0.25 = ±25%).")]
        [SerializeField] private float amplitude = 0.25f;
        [Tooltip("Flicker speed in noise-samples per second.")]
        [SerializeField] private float speed = 3.5f;

        private Light lite;
        private float baseIntensity;
        private Vector3 basePos;
        private float seed;

        private void Awake()
        {
            lite = GetComponent<Light>();
            baseIntensity = lite.intensity;
            basePos = transform.localPosition;
            seed = Random.value * 100f;
        }

        private void Update()
        {
            float t = Time.time * speed + seed;
            float n = Mathf.PerlinNoise(t, seed) * 2f - 1f;            // -1..1
            lite.intensity = baseIntensity * (1f + n * amplitude);
            float wob = (Mathf.PerlinNoise(seed, t * 0.7f) - 0.5f) * 0.02f;
            transform.localPosition = basePos + new Vector3(wob, Mathf.Abs(wob) * 0.5f, -wob);
        }
    }
}
