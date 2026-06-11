using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>Perlin-noise intensity flicker for torch point lights.</summary>
    [RequireComponent(typeof(Light))]
    public class TorchFlicker : MonoBehaviour
    {
        [SerializeField] private float amplitude = 0.45f;
        [SerializeField] private float speed = 7f;

        private Light torchLight;
        private float baseIntensity;
        private float seed;

        private void Awake()
        {
            torchLight = GetComponent<Light>();
            baseIntensity = torchLight.intensity;
            seed = Random.value * 100f;
        }

        private void Update()
        {
            torchLight.intensity = baseIntensity + (Mathf.PerlinNoise(Time.time * speed, seed) - 0.5f) * 2f * amplitude;
        }
    }
}
