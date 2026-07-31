using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Fits the directional shadow distance to THIS scene's playable area.
    /// <para>The built-in render pipeline has no per-scene shadow distance — it is a global
    /// QualitySettings value (150 m on the project's active "Ultra" level) — so the scene that wants
    /// a shorter range has to apply it on enable and put the old value back on disable. Restoring
    /// matters: the sibling ForestFork scene is bright daylight with almost no fog and genuinely
    /// shows shadows out to the horizon.</para>
    /// <para>Purely a rendering budget: nothing here touches gameplay.</para>
    /// </summary>
    public class ShadowDistanceFitter : MonoBehaviour
    {
        [Tooltip("Metres. The courtyard, the gate path and the boss chamber all sit within ~55 m of " +
                 "the camera, and the ExponentialSquared fog (density 0.024) is already 94% opaque at " +
                 "70 m — shadows cast from beyond that cannot be read through the mist. Trimming the " +
                 "range also makes each of the 4 cascades ~2.1x denser, so near shadows get crisper.")]
        [SerializeField] private float shadowDistance = 70f;

        private float previous;

        private void OnEnable()
        {
            previous = QualitySettings.shadowDistance;
            QualitySettings.shadowDistance = shadowDistance;
        }

        private void OnDisable()
        {
            QualitySettings.shadowDistance = previous;
        }
    }
}
