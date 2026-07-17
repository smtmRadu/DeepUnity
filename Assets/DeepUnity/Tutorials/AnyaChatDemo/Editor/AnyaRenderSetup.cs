#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// Project-level render settings the realistic-human demo needs. Linear color space is THE
    /// prerequisite for believable skin — under Gamma, lighting math is wrong and skin reads plastic.
    /// Kept as an explicit, revertible menu action (it affects the whole project, not just Anya).
    /// </summary>
    public static class AnyaRenderSetup
    {
        [MenuItem("DeepUnity/Anya/Render — Switch Project to LINEAR color space")]
        public static void GoLinear()
        {
            if (PlayerSettings.colorSpace == ColorSpace.Linear) { Debug.Log("[AnyaRender] already Linear"); return; }
            PlayerSettings.colorSpace = ColorSpace.Linear;
            Debug.Log("[AnyaRender] color space -> LINEAR (project-wide; revert with the Gamma menu item)");
        }

        [MenuItem("DeepUnity/Anya/Render — Revert Project to GAMMA color space")]
        public static void GoGamma()
        {
            if (PlayerSettings.colorSpace == ColorSpace.Gamma) { Debug.Log("[AnyaRender] already Gamma"); return; }
            PlayerSettings.colorSpace = ColorSpace.Gamma;
            Debug.Log("[AnyaRender] color space -> GAMMA");
        }
    }
}
#endif
