#if UNITY_EDITOR
using Unity.Collections;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    // Toggles the native-collections leak detector (the "Leak Detected : Persistent allocates N
    // individual allocations" console message). With stack traces on, every leaked allocation
    // logs WHERE it was allocated on the next report — the only way to attribute leaks, since
    // they come from native allocations inside engine machinery (readbacks, streaming clips,
    // packages), not from managed project code. Stack-trace mode slows allocations noticeably:
    // turn it back to plain Enabled after the hunt.
    public static class LeakDetectionMenu
    {
        [MenuItem("DeepUnity/Diagnostics/Leak Detection - Enabled With Stack Trace")]
        public static void EnableWithStackTrace()
        {
            NativeLeakDetection.Mode = NativeLeakDetectionMode.EnabledWithStackTrace;
            Debug.Log("[LeakDetection] EnabledWithStackTrace — reproduce the leak (play + exit play mode), " +
                      "then check the Console/Editor.log for per-allocation stacks.");
        }

        [MenuItem("DeepUnity/Diagnostics/Leak Detection - Enabled (default)")]
        public static void EnableDefault()
        {
            NativeLeakDetection.Mode = NativeLeakDetectionMode.Enabled;
            Debug.Log("[LeakDetection] Enabled (counts only, no stacks).");
        }
    }
}
#endif
